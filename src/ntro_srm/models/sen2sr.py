"""SEN2SR Model Adapter for NTRO-SRM.

Wraps the upstream ESAOpenSR SEN2SR implementation into a standardized,
reproducible interface suitable for the NTRO Super-Resolution Mapping pipeline.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure third_party/SEN2SR is available on sys.path without modifying upstream repo
_THIRD_PARTY_SEN2SR = Path(__file__).resolve().parents[3] / "third_party" / "SEN2SR"
if _THIRD_PARTY_SEN2SR.is_dir() and str(_THIRD_PARTY_SEN2SR) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_SEN2SR))

try:
    import sen2sr
    import mlstac
except ImportError as err:
    raise ImportError(
        f"Failed to import sen2sr or mlstac. Ensure dependencies are installed and "
        f"'{_THIRD_PARTY_SEN2SR}' exists. Error: {err}"
    ) from err

from ntro_srm.preprocessing.transforms import (
    S2_10BAND_NAMES,
    clamp_non_negative,
    handle_nans,
    normalize_reflectance,
)

# Hugging Face STAC metadata URLs for pretrained weights
DEFAULT_HF_SEN2SRLITE_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json"
)
DEFAULT_HF_SEN2SR_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/main/mlm.json"
)


class SEN2SRModel(nn.Module):
    """Thin adapter around upstream ESAOpenSR SEN2SR models.

    This adapter provides a uniform, documented interface for running
    inference on Sentinel-2 multi-spectral imagery.

    Supported Variants:
        - "lite": Fast CNN-based Swift Parameter-free Attention Network (SPAN),
          suitable for CPU and CUDA GPU execution (~0.47M parameters).
        - "swin2sr": Vision Transformer + MambaSR high-capacity architecture
          (~12.9M parameters), running with chunked selective scan on CUDA.

    Expected Input Specification:
        - Shape: (B, 10, H, W) or (10, H, W)
          Native model operates on 128x128 patches. Patches smaller than 128x128
          are automatically padded and cropped; larger images are automatically
          processed via tiled sliding windows with overlap.
        - Band Ordering: Standard Sentinel-2 10-band stack:
          [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
          (Blue, Green, Red, Red Edge 1, Red Edge 2, Red Edge 3, NIR, Narrow NIR, SWIR 1, SWIR 2)
        - Dynamic Range: Normalized surface reflectance in [0.0, 1.0].
          If raw integer reflectance (range [0, 10000]) is provided, pass
          `auto_normalize=True` to scale by 1/10000.

    Expected Output Specification:
        - Shape: (B, 10, 4*H, 4*W) or (10, 4*H, 4*W)
        - Resolution: ~2.5m Ground Sampling Distance (4x spatial upscaling).
        - Band Ordering: Preserves exact input Sentinel-2 10-band ordering.
    """

    def __init__(
        self,
        model_variant: str = "lite",
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        auto_download: bool = True,
    ) -> None:
        """Initialize the SEN2SR adapter.

        Parameters
        ----------
        model_variant : str, default="lite"
            Model architecture variant ("lite" or "swin2sr" / "swin").
        device : str or torch.device, optional
            Computation device ("cuda" or "cpu"). If None, uses CUDA if available.
        checkpoint_dir : str or Path, optional
            Path to directory containing downloaded checkpoint safetensors and mlm.json.
            If None, defaults to `checkpoints/SEN2SRLite` or `checkpoints/SEN2SR`.
        auto_download : bool, default=True
            Whether to download pretrained weights from Hugging Face if checkpoint_dir
            does not exist.
        """
        super().__init__()

        variant = model_variant.lower()
        if variant in ("lite", "sen2srlite"):
            self.model_variant = "lite"
        elif variant in ("swin", "swin2sr", "sen2sr", "full"):
            self.model_variant = "swin2sr"
            from ntro_srm.models.mamba_scan import register_mamba_shim
            register_mamba_shim()
        else:
            raise ValueError(
                f"Unsupported model variant '{model_variant}'. "
                f"Supported variants: 'lite' (SEN2SR-Lite), 'swin2sr' (SEN2SR-Swin2SR)."
            )

        # Determine compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve checkpoint path
        if checkpoint_dir is None:
            workspace_root = Path(__file__).resolve().parents[3]
            ckpt_folder = "SEN2SR" if self.model_variant == "swin2sr" else "SEN2SRLite"
            self.checkpoint_dir = workspace_root / "checkpoints" / ckpt_folder
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        # Load or download pretrained model
        self.model = self._load_model(auto_download=auto_download)
        self.model.to(self.device)
        self.model.eval()

        # Freeze all weights (pure inference)
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_model(self, auto_download: bool) -> nn.Module:
        """Load compiled model from MLSTAC checkpoint."""
        mlm_file = self.checkpoint_dir / "mlm.json"

        if not mlm_file.is_file():
            if auto_download:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                download_url = (
                    DEFAULT_HF_SEN2SR_URL
                    if self.model_variant == "swin2sr"
                    else DEFAULT_HF_SEN2SRLITE_URL
                )
                print(f"[NTRO-SRM] Downloading pretrained {self.model_variant} to {self.checkpoint_dir}...")
                mlstac.download(
                    file=download_url,
                    output_dir=str(self.checkpoint_dir),
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at {self.checkpoint_dir} and auto_download=False."
                )

        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        stac_item = mlstac.load(str(self.checkpoint_dir))
        return stac_item.compiled_model(device=device_str)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the underlying SEN2SR model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 10, H, W) with normalized reflectance in [0, 1].
            Spatial dimensions must be 128x128 for direct forward evaluation.

        Returns
        -------
        torch.Tensor
            Super-resolved tensor of shape (B, 10, 4*H, 4*W).
        """
        return self.model(x)

    @torch.no_grad()
    def predict(
        self,
        lr: torch.Tensor,
        auto_normalize: bool = False,
        clamp_output: bool = True,
        overlap: int = 32,
    ) -> torch.Tensor:
        """High-level prediction interface with tensor sanitization and validation.

        Automatically handles:
        - 3D (10, H, W) and 4D (B, 10, H, W) tensors.
        - Arbitrary spatial sizes (padding small patches < 128, and tiling large tiles > 128).
        - NaN / Inf sanitization.
        - Band count validation (strictly 10 bands).
        - Optional reflectance auto-normalization.

        Parameters
        ----------
        lr : torch.Tensor
            Input Sentinel-2 tensor of shape (B, 10, H, W) or (10, H, W).
            Band ordering must follow:
            [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
        auto_normalize : bool, default=False
            If True, divides input by 10,000 to convert from raw integer reflectance.
            If False, verifies input is within a reasonable reflectance range.
        clamp_output : bool, default=True
            If True, clamps negative values in the super-resolved output to 0.0.
        overlap : int, default=32
            Overlap in pixels when processing large tiles (> 128x128).

        Returns
        -------
        torch.Tensor
            Super-resolved 2.5m tensor of shape (B, 10, 4*H, 4*W) or (10, 4*H, 4*W).
        """
        # Validate dimensions
        is_3d = lr.ndim == 3
        if is_3d:
            lr = lr.unsqueeze(0)
        elif lr.ndim != 4:
            raise ValueError(
                f"Expected 3D (10, H, W) or 4D (B, 10, H, W) tensor, got shape {lr.shape}"
            )

        if lr.shape[1] != 10:
            raise ValueError(
                f"Expected 10 Sentinel-2 bands at channel dimension (dim 1), but received {lr.shape[1]}. "
                f"Expected bands: {S2_10BAND_NAMES}"
            )

        # Ensure floating point
        if not lr.is_floating_point():
            lr = lr.float()

        # Handle NaNs and Infs
        lr = handle_nans(lr, fill_value=0.0)

        # Normalize if requested
        if auto_normalize:
            lr = normalize_reflectance(lr)
        else:
            if lr.max() > 10.0:
                print(
                    "[NTRO-SRM Warning] Input maximum exceeds 10.0. "
                    "Did you mean to set auto_normalize=True for raw [0, 10000] S2 data?"
                )

        # Non-negative clamping on input
        lr = clamp_non_negative(lr, min_value=0.0)

        orig_device = lr.device
        lr = lr.to(self.device)

        batch_size, channels, in_h, in_w = lr.shape

        # Handle spatial dimensions (square & non-square, arbitrary sizes)
        # Upstream sen2sr.predict_large requires square inputs matching stride grid: 128 + n * (128 - overlap)
        step = max(1, 128 - overlap)
        max_side = max(in_h, in_w)
        if max_side <= 128:
            target_dim = 128
        else:
            n_steps = math.ceil((max_side - 128) / step)
            target_dim = 128 + n_steps * step

        pad_h = target_dim - in_h
        pad_w = target_dim - in_w
        needs_padding = (pad_h > 0) or (pad_w > 0)

        if needs_padding:
            lr_padded = F.pad(lr, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            lr_padded = lr

        if target_dim == 128:
            sr_padded = self.forward(lr_padded)
        else:
            sr_batches = []
            for b in range(batch_size):
                sample_lr = lr_padded[b]  # (10, target_dim, target_dim)
                sr_sample = sen2sr.predict_large(
                    X=sample_lr,
                    model=self.model,
                    overlap=overlap,
                )
                sr_batches.append(sr_sample)
            sr_padded = torch.stack(sr_batches, dim=0).to(self.device)

        if needs_padding:
            sr = sr_padded[:, :, : in_h * 4, : in_w * 4]
        else:
            sr = sr_padded

        # Post-processing
        if clamp_output:
            sr = clamp_non_negative(sr, min_value=0.0)

        sr = sr.to(orig_device)
        if is_3d:
            sr = sr.squeeze(0)

        return sr
