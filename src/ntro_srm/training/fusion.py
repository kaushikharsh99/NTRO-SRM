"""Learned K-frame fusion on top of a frozen single-image SR backbone.

Per-frame Lite SR outputs are cached once; a tiny attention head learns
per-pixel frame weights (with change cues) and outputs the fused 10-band SR.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusionHead(nn.Module):
    """Per-pixel softmax weighting over K SR frames + light refine.

    Input: stacked SR frames (B,K,10,H,W) and LR frames (B,K,10,h,w).
    Change cue: mean |LR_k - LR_0| upsampled to H,W (K channels).
    """

    def __init__(self, frames: int = 3, width: int = 32) -> None:
        super().__init__()
        self.frames = frames
        in_ch = 10 * frames + frames
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, frames, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 10, 3, padding=1),
        )

    def forward(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        b, k, _, h, w = sr.shape
        with torch.no_grad():
            cue = lr.sub(lr[:, :1]).abs().mean(dim=2, keepdim=True)
            cue = F.interpolate(cue.reshape(b * k, 1, *lr.shape[-2:]),
                                size=(h, w), mode="bilinear",
                                align_corners=False).reshape(b, k, h, w)
        feats = torch.cat([sr.reshape(b, k * 10, h, w), cue], dim=1)
        weights = self.attn(feats).softmax(dim=1)  # (B,K,H,W)
        fused = (sr * weights.unsqueeze(2)).sum(dim=1)
        return fused + self.refine(fused)

    def weight_map(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        b, k, _, h, w = sr.shape
        with torch.no_grad():
            cue = lr.sub(lr[:, :1]).abs().mean(dim=2, keepdim=True)
            cue = F.interpolate(cue.reshape(b * k, 1, *lr.shape[-2:]),
                                size=(h, w), mode="bilinear",
                                align_corners=False).reshape(b, k, h, w)
        feats = torch.cat([sr.reshape(b, k * 10, h, w), cue], dim=1)
        return self.attn(feats).softmax(dim=1)
