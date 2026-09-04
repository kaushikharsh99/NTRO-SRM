"""Pure PyTorch memory-efficient selective scan fallback for MambaSR.

Allows running the upstream ESAOpenSR full SEN2SR architecture on PyTorch 2.x
and CUDA without requiring custom C++ binary extensions of mamba-ssm, while
preventing GPU Out-Of-Memory errors by chunking recurrence steps.
"""

from __future__ import annotations

import sys
import types
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def ultra_chunked_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    chunk_size: int = 256,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Memory-efficient selective scan implementation in pure PyTorch.

    Processes recurrence in temporal chunks (default: 256 steps) to cap
    intermediate tensor allocations below 600 MB on 6GB GPUs, completely
    avoiding the multi-gigabyte 4D allocations of naive reference scans.

    Parameters
    ----------
    u : torch.Tensor
        Input tensor of shape (batch, dim, L).
    delta : torch.Tensor
        Step size tensor of shape (batch, dim, L).
    A : torch.Tensor
        State transition matrix of shape (dim, dstate).
    B : torch.Tensor
        Input projection tensor of shape (batch, G, dstate, L).
    C : torch.Tensor
        Output projection tensor of shape (batch, G, dstate, L).
    D : torch.Tensor, optional
        Skip connection parameter of shape (dim,).
    z : torch.Tensor, optional
        Gate tensor of shape (batch, dim, L).
    delta_bias : torch.Tensor, optional
        Step bias of shape (dim,).
    delta_softplus : bool, default=False
        Whether to apply softplus to delta.
    return_last_state : bool, default=False
        Whether to return the final hidden state.
    chunk_size : int, default=256
        Temporal chunk size for bounded GPU memory execution.
    """
    dtype_in = u.dtype
    batch, dim, L = u.shape
    dstate = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    G = B.shape[1]
    H = dim // G

    y_out = torch.empty((batch, dim, L), device=u.device, dtype=torch.float32)
    x = torch.zeros((batch, dim, dstate), device=u.device, dtype=torch.float32)

    num_chunks = (L + chunk_size - 1) // chunk_size

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, L)
        clen = end - start

        u_c = u[:, :, start:end]
        delta_c = delta[:, :, start:end]

        # Slices for this chunk only - avoid global multi-GB tensor materialization
        B_c = B[:, :, :, start:end].repeat_interleave(H, dim=1)
        C_c = C[:, :, :, start:end].repeat_interleave(H, dim=1)

        dA_c = torch.exp(delta_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
        dB_u_c = (delta_c * u_c).unsqueeze(-1) * B_c.permute(0, 1, 3, 2)

        for t in range(clen):
            x = dA_c[:, :, t] * x + dB_u_c[:, :, t]
            y_out[:, :, start + t] = (x * C_c[:, :, :, t]).sum(dim=-1)

    if D is not None:
        y_out = y_out + u * D.view(1, -1, 1)
    if z is not None:
        y_out = y_out * F.silu(z)

    y_out = y_out.to(dtype=dtype_in)
    return y_out if not return_last_state else (y_out, x)


def register_mamba_shim() -> None:
    """Register pure PyTorch mamba_ssm shim in sys.modules if not installed.

    Allows upstream sen2sr to load checkpoints using MambaSR without modifying
    the third_party/SEN2SR codebase.
    """
    if "mamba_ssm" not in sys.modules:
        try:
            import mamba_ssm  # noqa: F401
        except ImportError:
            mamba_mod = types.ModuleType("mamba_ssm")
            ops_mod = types.ModuleType("mamba_ssm.ops")
            scan_mod = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
            scan_mod.selective_scan_fn = ultra_chunked_selective_scan
            scan_mod.selective_scan_ref = ultra_chunked_selective_scan
            ops_mod.selective_scan_interface = scan_mod
            mamba_mod.ops = ops_mod
            sys.modules["mamba_ssm"] = mamba_mod
            sys.modules["mamba_ssm.ops"] = ops_mod
            sys.modules["mamba_ssm.ops.selective_scan_interface"] = scan_mod
