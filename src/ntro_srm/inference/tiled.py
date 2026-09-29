"""Batched tiled inference with weighted overlap blending.

Replaces the upstream ``sen2sr.predict_large`` loop
(``third_party/SEN2SR/sen2sr/utils.py``) which:

* runs one 128x128 tile per forward (batch=1) with a ``tqdm`` Python loop,
* hard-crops overlap borders (visible seams),
* assumes square scenes (output allocated as ``W x W``, x/y axes swapped in
  the chunk slicer, LR/HR units mixed in border checks),
* pads rectangular scenes up to a square ``max(H, W)`` stride grid,
* emits duplicate tiles on non-divisible dimensions via ``fix_lastchunk``.

This module fixes all of the above while staying a drop-in tensor-level
replacement: input ``(B, C, H, W)`` -> output ``(B, C, H*scale, W*scale)``.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import json

import torch

ModelFn = Callable[[torch.Tensor], torch.Tensor]

_WINDOW_CACHE: Dict[tuple, torch.Tensor] = {}


@dataclass
class TiledInferenceConfig:
    """Reproducible tiled-inference configuration.

    Attributes
    ----------
    tile_size: LR tile edge (model-native 128 for SEN2SR).
    overlap: LR overlap between adjacent tiles.
    batch_size: tiles per forward (spatial*batch dims are packed).
    blend_mode: "linear" (default, partition-of-unity ramps), "hann"
        (cosine ramps), or "average" (uniform, divide-by-coverage).
    scale_factor: spatial upscale (4 for 10m -> 2.5m).
    use_amp: enable autocast mixed precision (CUDA only by default).
    amp_dtype: explicit autocast dtype; resolved automatically if None.
    empty_cache_at_end: release cached accelerator memory after inference.
    enable_cudnn_benchmark: set cudnn.benchmark=True on CUDA for fixed tiles.
    """

    tile_size: int = 128
    overlap: int = 32
    batch_size: int = 4
    blend_mode: str = "linear"
    scale_factor: int = 4
    use_amp: bool = False
    amp_dtype: Optional[str] = None
    empty_cache_at_end: bool = False
    enable_cudnn_benchmark: bool = True

    def validate(self) -> "TiledInferenceConfig":
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= self.overlap < self.tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.blend_mode not in ("linear", "hann", "average"):
            raise ValueError(f"Unknown blend_mode '{self.blend_mode}'")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def compute_tile_grid(
    h: int, w: int, tile_size: int = 128, overlap: int = 32
) -> List[Tuple[int, int]]:
    """Compute top-left ``(y, x)`` tile origins covering ``(h, w)``.

    Guarantees: full coverage, no duplicates, minimal tile count, correct
    rectangular handling, single ``[(0, 0)]`` entry when the image fits in
    one tile.  The last row/column is clamped to ``(h - tile, w - tile)``
    instead of emitting out-of-bounds origins.
    """
    h, w = int(h), int(w)
    tile_size, overlap = int(tile_size), int(overlap)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if h <= tile_size and w <= tile_size:
        return [(0, 0)]
    stride = tile_size - overlap

    def _axis(n: int) -> List[int]:
        if n <= tile_size:
            return [0]
        pos = list(range(0, n - tile_size + 1, stride))
        if pos[-1] != n - tile_size:
            pos.append(n - tile_size)
        return pos

    ys, xs = _axis(h), _axis(w)
    grid = [(y, x) for y in ys for x in xs]
    # Deduplicate defensively while preserving order.
    seen, out = set(), []
    for p in grid:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _ramp_1d(
    n: int, overlap: int, has_before: bool, has_after: bool, mode: str
) -> torch.Tensor:
    if mode == "average" or overlap <= 0:
        return torch.ones(n, dtype=torch.float32)
    ov = min(int(overlap), n)
    t = torch.linspace(0.0, 1.0, ov, dtype=torch.float32)
    if mode == "linear":
        ramp_in, ramp_out = t, 1.0 - t
    elif mode == "hann":
        ramp_in = 0.5 - 0.5 * torch.cos(torch.pi * t)
        ramp_out = 1.0 - ramp_in
    else:
        raise ValueError(f"Unknown blend_mode '{mode}'")
    w = torch.ones(n, dtype=torch.float32)
    if has_before:
        w[:ov] = ramp_in
    if has_after:
        tail = ramp_out
        seg = w[n - ov :]
        w[n - ov :] = seg * tail[: len(seg)]
    return w


def blend_window(
    tile_hr: int,
    overlap_hr: int,
    has_top: bool,
    has_left: bool,
    has_bottom: bool,
    has_right: bool,
    mode: str = "linear",
) -> torch.Tensor:
    """Return a ``(tile_hr, tile_hr)`` blending weight window.

    Weights taper only toward *interior* neighbours, so image-border pixels
    keep weight 1 and the ``output / weight_sum`` normalization is exact.
    Windows are cached by spec.
    """
    key = (int(tile_hr), int(overlap_hr), bool(has_top), bool(has_left),
           bool(has_bottom), bool(has_right), str(mode))
    hit = _WINDOW_CACHE.get(key)
    if hit is not None:
        return hit
    wy = _ramp_1d(tile_hr, overlap_hr, has_top, has_bottom, mode)
    wx = _ramp_1d(tile_hr, overlap_hr, has_left, has_right, mode)
    win = (wy[:, None] * wx[None, :]).contiguous()
    _WINDOW_CACHE[key] = win
    return win


def clear_window_cache() -> None:
    _WINDOW_CACHE.clear()


def resolve_amp_dtype(device: Union[str, torch.device], requested: Optional[str] = None) -> Optional[torch.dtype]:
    """Map a requested AMP dtype string to torch.dtype (None disables)."""
    if requested is None:
        return None
    r = str(requested).lower()
    if r in ("float16", "fp16", "half"):
        return torch.float16
    if r in ("bfloat16", "bf16"):
        return torch.bfloat16
    if r in ("float32", "fp32", "none", "off"):
        return None
    raise ValueError(f"Unknown amp_dtype '{requested}'")


def autocast_context(device: Union[str, torch.device], enabled: bool, amp_dtype: Optional[str] = None):
    """Return an autocast context (or nullcontext) appropriate for ``device``.

    CUDA uses float16 by default; CPU uses bfloat16; MPS disables AMP because
    Metal autocast is unreliable for FFT-heavy models.
    """
    if not enabled:
        return nullcontext()
    dev = torch.device(device)
    if dev.type == "cuda":
        dtype = resolve_amp_dtype(dev, amp_dtype) or torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if dev.type == "cpu":
        dtype = resolve_amp_dtype(dev, amp_dtype) or torch.bfloat16
        return torch.autocast(device_type="cpu", dtype=dtype)
    return nullcontext()


@torch.inference_mode()
def tiled_predict(
    model_fn: ModelFn,
    lr: torch.Tensor,
    config: Optional[TiledInferenceConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Run batched tiled super-resolution with weighted blending.

    Parameters
    ----------
    model_fn: callable mapping ``(N, C, tile, tile)`` -> ``(N, C, tile*scale,
        tile*scale)``.  Pass ``model`` or ``model.forward``.
    lr: input tensor ``(B, C, H, W)`` (float).
    config: tiling/blending/AMP configuration.
    device: accelerator for model execution (defaults to ``lr.device``).
    verbose: print tiling plan to stdout.

    Returns
    -------
    torch.Tensor
        ``(B, C, H*scale, W*scale)`` float32 tensor on ``lr.device``.
    """
    cfg = (config or TiledInferenceConfig()).validate()
    if lr.ndim != 4:
        raise ValueError(f"tiled_predict expects 4D (B, C, H, W), got {tuple(lr.shape)}")
    tile, overlap, scale = int(cfg.tile_size), int(cfg.overlap), int(cfg.scale_factor)
    exec_device = torch.device(device) if device is not None else lr.device
    B, C, H, W = (int(v) for v in lr.shape)
    out_H, out_W = H * scale, W * scale
    tile_hr, overlap_hr = tile * scale, overlap * scale

    if exec_device.type == "cuda" and cfg.enable_cudnn_benchmark:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # Fast path: single tile per sample (with replicate pad when smaller).
    if H <= tile and W <= tile:
        pad_h, pad_w = tile - H, tile - W
        import torch.nn.functional as F_pad

        batch = lr.to(exec_device, non_blocking=True)
        if pad_h > 0 or pad_w > 0:
            batch = F_pad.pad(batch, (0, pad_w, 0, pad_h), mode="replicate")
        with autocast_context(exec_device, cfg.use_amp, cfg.amp_dtype):
            sr = model_fn(batch)
        sr = sr.float().cpu()
        if pad_h > 0 or pad_w > 0:
            sr = sr[:, :, :out_H, :out_W]
        out = sr.to(lr.device)
        if cfg.empty_cache_at_end:
            _empty_cache(exec_device)
        return out

    grid = compute_tile_grid(H, W, tile, overlap)
    if verbose:
        print(f"[tiled] {H}x{W} -> {out_H}x{out_W} | tiles={len(grid)} "
              f"tile={tile} overlap={overlap} batch={cfg.batch_size} blend={cfg.blend_mode}")

    # Accumulate on CPU float32 (a 512x512 LR -> 2048x2048x10 SR is ~160 MB;
    # keeping it on GPU would needlessly pressure VRAM).
    output = torch.zeros((B, C, out_H, out_W), dtype=torch.float32, device="cpu")
    weight = torch.zeros((B, 1, out_H, out_W), dtype=torch.float32, device="cpu")

    jobs: List[Tuple[int, int, int]] = []  # (b, y0, x0)
    for b in range(B):
        for (y0, x0) in grid:
            jobs.append((b, y0, x0))

    lr_on_exec = lr.to(exec_device, non_blocking=True) if lr.device != exec_device else lr
    amp_ctx = autocast_context(exec_device, cfg.use_amp, cfg.amp_dtype)
    batch_size = max(1, int(cfg.batch_size))
    with amp_ctx:
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start:start + batch_size]
            tiles = torch.stack(
                [lr_on_exec[b, :, y0:y0 + tile, x0:x0 + tile] for (b, y0, x0) in chunk],
                dim=0,
            )
            sr_tiles = model_fn(tiles)
            if sr_tiles.shape[-2:] != (tile_hr, tile_hr):
                raise RuntimeError(
                    f"Model returned {tuple(sr_tiles.shape[-2:])}, expected {(tile_hr, tile_hr)} "
                    f"for tile={tile} scale={scale}"
                )
            sr_cpu = sr_tiles.detach().float().cpu()
            del sr_tiles
            for i, (b, y0, x0) in enumerate(chunk):
                oy, ox = y0 * scale, x0 * scale
                has_top = y0 > 0
                has_left = x0 > 0
                has_bottom = (y0 + tile) < H
                has_right = (x0 + tile) < W
                win = blend_window(tile_hr, overlap_hr, has_top, has_left, has_bottom, has_right, cfg.blend_mode)
                patch = sr_cpu[i] * win
                output[b, :, oy:oy + tile_hr, ox:ox + tile_hr] += patch
                weight[b, :, oy:oy + tile_hr, ox:ox + tile_hr] += win
            del tiles, sr_cpu
    output = output / weight.clamp_min(1e-8)
    if cfg.empty_cache_at_end:
        _empty_cache(exec_device)
    return output.to(lr.device)


def _empty_cache(device: torch.device) -> None:
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
    except Exception:
        pass


def _git_hash(root: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def make_inference_receipt(
    config: TiledInferenceConfig,
    *,
    model_variant: str = "lite",
    device: Union[str, torch.device] = "cpu",
    input_shape: Optional[tuple] = None,
    output_shape: Optional[tuple] = None,
    inference_time_ms: Optional[float] = None,
    peak_gpu_memory_mb: Optional[float] = None,
    extra: Optional[dict] = None,
    workspace_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Build a JSON-serializable reproducibility receipt."""
    import torch as _torch

    root = Path(workspace_root) if workspace_root else None
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_variant": str(model_variant),
        "device": str(torch.device(device)),
        "torch_version": getattr(_torch, "__version__", "unknown"),
        "cuda_available": bool(_torch.cuda.is_available()),
        "git_commit": _git_hash(root),
        "config": config.validate().to_dict(),
        "input_shape": list(input_shape) if input_shape is not None else None,
        "output_shape": list(output_shape) if output_shape is not None else None,
        "inference_time_ms": inference_time_ms,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
    if extra:
        receipt["extra"] = extra
    return receipt


def write_inference_receipt(receipt: dict, path: Union[str, Path]) -> Path:
    """Write ``receipt`` as pretty-printed JSON; return the resolved path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def timed_tiled_predict(
    model_fn: ModelFn,
    lr: torch.Tensor,
    config: Optional[TiledInferenceConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, float]:
    """Run :func:`tiled_predict` and return ``(output, elapsed_ms)``."""
    t0 = time.perf_counter()
    out = tiled_predict(model_fn, lr, config, device=device)
    return out, (time.perf_counter() - t0) * 1000.0
