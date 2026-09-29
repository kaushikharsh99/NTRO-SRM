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
from ntro_srm.utils.device import empty_device_cache, select_device

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
          suitable for CPU, CUDA, and Apple MPS execution (~0.47M parameters).
        - "swin2sr": Vision Transformer + MambaSR high-capacity architecture
          (~12.9M parameters), running with chunked selective scan on CUDA,
          Apple MPS, or CPU.

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
        trainable: bool = False,
    ) -> None:
        """Initialize the SEN2SR adapter.

        Parameters
        ----------
        model_variant : str, default="lite"
            Model architecture variant ("lite" or "swin2sr" / "swin").
        device : str or torch.device, optional
            Computation device ("cuda", "mps", or "cpu"). If omitted, selects
            CUDA, then Apple MPS, then CPU.
        checkpoint_dir : str or Path, optional
            Path to directory containing downloaded checkpoint safetensors and mlm.json.
            If None, defaults to `checkpoints/SEN2SRLite` or `checkpoints/SEN2SR`.
        auto_download : bool, default=True
            Whether to download pretrained weights from Hugging Face if checkpoint_dir
            does not exist.
        trainable : bool, default=False
            Enable gradients on the upstream backbone for fine-tuning. Inference
            remains the safe default.
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

        self.device = select_device(device)

        # Resolve checkpoint path
        if checkpoint_dir is None:
            workspace_root = Path(__file__).resolve().parents[3]
            ckpt_folder = "SEN2SR" if self.model_variant == "swin2sr" else "SEN2SRLite"
            self.checkpoint_dir = workspace_root / "checkpoints" / ckpt_folder
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        # Load or download pretrained model
        self.model = self._load_model(auto_download=auto_download)
        self._materialize_inference_tensors(preserve_parameters=False)
        self.model.to(self.device)
        self._move_upstream_runtime_tensors()
        self.set_trainable(trainable)

    def _move_upstream_runtime_tensors(self) -> None:
        """Move SEN2SR tensor attributes that upstream did not register as buffers.

        SEN2SR's Fourier hard constraint stores each low-pass mask as a plain
        attribute. ``Module.to`` therefore leaves those masks on CPU, which causes
        a device mismatch during MPS inference. This also installs the
        vectorized FFT forward and precomputes complementary high-pass masks
        (see ``ntro_srm.models.fourier_filters``); upstream files are untouched.
        """
        for module in self.model.modules():
            low_pass_mask = getattr(module, "low_pass_mask", None)
            if isinstance(low_pass_mask, torch.Tensor):
                module.low_pass_mask = low_pass_mask.to(self.device)
        try:
            from ntro_srm.models.fourier_filters import optimize_upstream_masks

            optimize_upstream_masks(self.model, self.device)
        except Exception:
            pass

    def _materialize_inference_tensors(self, *, preserve_parameters: bool) -> None:
        """Copy MLSTAC inference tensors into ordinary autograd tensors."""
        for module in self.model.modules():
            for name, parameter in list(module.named_parameters(recurse=False)):
                if parameter.is_inference():
                    if preserve_parameters:
                        # Keep optimizer references valid after an upstream
                        # evaluation layer refreshes fused parameter data.
                        parameter.data = parameter.detach().clone()
                    else:
                        setattr(
                            module,
                            name,
                            nn.Parameter(
                                parameter.detach().clone(),
                                requires_grad=parameter.requires_grad,
                            ),
                        )
            for name, buffer in list(module.named_buffers(recurse=False)):
                if buffer is not None and buffer.is_inference():
                    module._buffers[name] = buffer.detach().clone()

    def set_trainable(self, trainable: bool) -> None:
        """Switch backbone parameter gradients and training mode explicitly."""
        # Some upstream evaluation layers refresh fused weights during an
        # inference-mode forward. Convert those tensors in place as well.
        self._materialize_inference_tensors(preserve_parameters=True)
        for parameter in self.model.parameters():
            parameter.requires_grad = trainable
        self.model.train(mode=trainable)

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

        # MLSTAC currently accepts CPU/CUDA at construction time. MPS models are
        # materialized on CPU first, then moved to Metal by this adapter.
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        stac_item = mlstac.load(str(self.checkpoint_dir))
        return stac_item.compiled_model(device=device_str)

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
    @torch.inference_mode()
    def predict(
        self,
        lr: torch.Tensor,
        auto_normalize: bool = False,
        clamp_output: bool = True,
        overlap: int = 32,
        tile_size: int = 128,
        batch_size: int = 4,
        blend_mode: str = "linear",
        use_amp: bool = False,
        amp_dtype: Optional[str] = None,
        use_legacy_tiling: bool = False,
        empty_cache: bool = False,
    ) -> torch.Tensor:
        """High-level prediction interface with tensor sanitization and validation.

        Automatically handles:
        - 3D (10, H, W) and 4D (B, 10, H, W) tensors.
        - Arbitrary spatial sizes (padding small patches < tile_size, and tiling large tiles > tile_size).
        - NaN / Inf sanitization.
        - Band count validation (strictly 10 bands).
        - Optional reflectance auto-normalization.

        Inference optimizations (default path):
        - Batched tile forward passes (``batch_size`` tiles per forward instead
          of one), with accumulation on CPU to bound VRAM.
        - Weighted overlap blending (``blend_mode="linear"`` partition-of-unity
          ramps, ``"hann"`` cosine, or ``"average"``) instead of hard overlap
          cropping, removing seam discontinuities.
        - Independent per-axis tiling so rectangular scenes are covered exactly
          with no square-padding waste and no duplicate tiles.
        - Optional autocast mixed precision (``use_amp=True``, CUDA float16 /
          CPU bfloat16; MPS stays in float32).
        - Cached vectorized Fourier masks with an efficient ``fft2`` hard
          constraint (installed at init; see ``fourier_filters``).

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
            Overlap in pixels when processing large tiles (> tile_size).
        tile_size : int, default=128
            LR tile edge for tiled inference (model-native 128).
        batch_size : int, default=4
            Tiles per forward pass in the optimized tiler.
        blend_mode : {"linear", "hann", "average"}, default="linear"
            Overlap blending strategy ("average" divides by coverage).
        use_amp : bool, default=False
            Enable autocast mixed precision (CUDA/CPU; MPS is a no-op).
        amp_dtype : str, optional
            Explicit autocast dtype ("float16"/"bfloat16"); auto-selected if None.
        use_legacy_tiling : bool, default=False
            Reproduce the upstream single-tile hard-crop path for ablations.
        empty_cache : bool, default=False
            Release cached accelerator memory after inference.

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

        n_batch, channels, in_h, in_w = lr.shape
        tile = int(tile_size)
        if tile <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= int(overlap) < tile:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if blend_mode not in ("linear", "hann", "average"):
            raise ValueError(f"Unknown blend_mode '{blend_mode}'")

        if use_legacy_tiling:
            sr = self._predict_legacy(lr, overlap=overlap)
        elif in_h <= tile and in_w <= tile:
            # Small patch: replicate-pad to the native tile, forward once, crop.
            pad_h = tile - in_h
            pad_w = tile - in_w
            if pad_h > 0 or pad_w > 0:
                lr_padded = F.pad(lr, (0, pad_w, 0, pad_h), mode="replicate")
            else:
                lr_padded = lr
            from ntro_srm.inference.tiled import autocast_context as _amp_ctx

            with _amp_ctx(self.device, use_amp, amp_dtype):
                sr_padded = self.forward(lr_padded)
            sr = sr_padded[:, :, : in_h * 4, : in_w * 4]
            del sr_padded
        else:
            # Large / rectangular scene: batched weighted blending, no square padding.
            from ntro_srm.inference.tiled import TiledInferenceConfig, tiled_predict

            cfg = TiledInferenceConfig(
                tile_size=tile,
                overlap=int(overlap),
                batch_size=int(batch_size),
                blend_mode=blend_mode,
                scale_factor=4,
                use_amp=bool(use_amp),
                amp_dtype=amp_dtype,
                empty_cache_at_end=False,
            )
            sr = tiled_predict(self.model, lr, cfg, device=self.device)

        # Post-processing
        if clamp_output:
            sr = clamp_non_negative(sr, min_value=0.0)

        sr = sr.to(orig_device)
        if empty_cache:
            try:
                empty_device_cache(self.device)
            except Exception:
                pass
        if is_3d:
            sr = sr.squeeze(0)

        return sr

    def _predict_legacy(self, lr: torch.Tensor, overlap: int = 32) -> torch.Tensor:
        """Upstream-compatible single-tile hard-crop path (ablation only).

        Preserves the original square-padding + per-sample
        ``sen2sr.predict_large`` behaviour, including its rectangular-scene
        limitations, for benchmark comparisons.
        """
        n_batch, _, in_h, in_w = lr.shape
        step = max(1, 128 - int(overlap))
        max_side = max(in_h, in_w)
        if max_side <= 128:
            target_dim = 128
        else:
            n_steps = math.ceil((max_side - 128) / step)
            target_dim = 128 + n_steps * step

        pad_h = target_dim - in_h
        pad_w = target_dim - in_w
        if pad_h > 0 or pad_w > 0:
            lr_padded = F.pad(lr, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            lr_padded = lr

        if target_dim == 128:
            sr_padded = self.forward(lr_padded)
        else:
            sr_batches = []
            for b in range(n_batch):
                sample_lr = lr_padded[b]
                sr_sample = sen2sr.predict_large(
                    X=sample_lr,
                    model=self.model,
                    overlap=int(overlap),
                )
                sr_batches.append(sr_sample)
            sr_padded = torch.stack(sr_batches, dim=0).to(self.device)

        if pad_h > 0 or pad_w > 0:
            return sr_padded[:, :, : in_h * 4, : in_w * 4]
        return sr_padded
