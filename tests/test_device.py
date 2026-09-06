"""Tests for portable compute-device selection."""

import pytest
import torch

from ntro_srm.utils import device as device_utils


def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_utils, "mps_available", lambda: True)
    assert device_utils.select_device().type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_utils.select_device().type == "mps"

    monkeypatch.setattr(device_utils, "mps_available", lambda: False)
    assert device_utils.select_device().type == "cpu"


def test_explicit_unavailable_accelerator_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils, "mps_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA was requested"):
        device_utils.select_device("cuda")
    with pytest.raises(ValueError, match="MPS was requested"):
        device_utils.select_device("mps")


def test_cpu_is_always_available() -> None:
    assert device_utils.select_device("cpu") == torch.device("cpu")
    assert device_utils.device_name("cpu") == "CPU"
