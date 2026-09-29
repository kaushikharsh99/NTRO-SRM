"""Compute-device selection and synchronization helpers."""

from __future__ import annotations

from typing import Optional, Union

import torch


DeviceLike = Optional[Union[str, torch.device]]


def mps_available() -> bool:
    """Return whether PyTorch can use Apple's Metal Performance Shaders backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built() and backend.is_available())


def select_device(requested: DeviceLike = None) -> torch.device:
    """Resolve an explicit device or choose CUDA, MPS, then CPU in that order."""
    if requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA device")
    if device.type == "mps" and not mps_available():
        raise ValueError("MPS was requested, but PyTorch cannot access the Apple GPU")
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported compute device '{device.type}'; use cuda, mps, or cpu")
    return device


def synchronize_device(device: Union[str, torch.device]) -> None:
    """Wait for queued accelerator work so timing and error reporting are accurate."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: Union[str, torch.device]) -> None:
    """Release unused cached accelerator allocations when the backend supports it."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


def device_name(device: Union[str, torch.device]) -> str:
    """Return a concise display name for the active compute device."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.cuda.get_device_name(torch.device(device))
    if device_type == "mps":
        return "Apple GPU (MPS)"
    return "CPU"
