"""Self-contained ShallowConvNet (Schirrmeister et al. 2017) — no dependency on the old packages.

    input [B, C, T] -> [B,1,C,T]
    temporal Conv2d(1, F, (1, k_t))
    spatial  Conv2d(F, F, (C, 1))           # across ALL eeg channels
    BatchNorm2d(F)
    square
    AvgPool2d((1, k_p), stride=(1, s_p))
    safe log (clamp >= eps)
    flatten -> z                            # the log-power representation (PRE-dropout)
    dropout(z) -> Linear -> logits

``z`` is the dropout-FREE log-power representation; the dummy forward used to infer the flatten
dimension runs in ``eval()`` + ``no_grad`` and restores the original mode, leaving every buffer
(incl. BatchNorm running stats) byte-identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .output import ModelOutput, RepresentationClassifier


def _square(x):
    return x * x


def validate_shallow_geometry(in_chans, in_times, backbone_block) -> dict:
    """Input-dependent geometry check for ShallowConvNet before any GPU/model construction."""
    bb = backbone_block
    tf, tk = int(bb.temporal_filters), int(bb.temporal_kernel_samples)
    pk, ps = int(bb.pool_kernel_samples), int(bb.pool_stride_samples)
    dr, eps, ic, it = float(bb.dropout), float(bb.safe_log_eps), int(in_chans), int(in_times)
    if tf < 1:
        raise ValueError("temporal_filters >= 1")
    if ic < 1:
        raise ValueError("in_chans >= 1")
    if tk > it:
        raise ValueError(f"temporal_kernel_samples ({tk}) <= in_times ({it})")
    post = it - tk + 1                                  # length after the temporal conv
    if pk > post:
        raise ValueError(f"pool_kernel_samples ({pk}) <= post-temporal times ({post})")
    if ps < 1:
        raise ValueError("pool_stride_samples >= 1")
    pooled = (post - pk) // ps + 1                      # AvgPool2d output length
    if pooled < 1:
        raise ValueError("final pooled time dimension must be >= 1")
    if not (0 <= dr < 1):
        raise ValueError("0 <= dropout < 1")
    if eps <= 0:
        raise ValueError("safe_log_eps > 0")
    return {"post_temporal_times": post, "pooled_times": pooled, "feat_dim": tf * pooled}


class ShallowConvNet(RepresentationClassifier):
    def __init__(self, in_chans: int, in_times: int, n_classes: int, temporal_filters: int = 40,
                 temporal_kernel_samples: int = 25, pool_kernel_samples: int = 75,
                 pool_stride_samples: int = 15, dropout: float = 0.5, safe_log_eps: float = 1e-6):
        super().__init__()
        self.safe_log_eps = float(safe_log_eps)
        self.temporal = nn.Conv2d(1, temporal_filters, (1, temporal_kernel_samples))
        self.spatial = nn.Conv2d(temporal_filters, temporal_filters, (in_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(temporal_filters)
        self.pool = nn.AvgPool2d((1, pool_kernel_samples), stride=(1, pool_stride_samples))
        self.drop = nn.Dropout(dropout)
        # infer flatten dim with a SAFE dummy forward (no state mutation)
        p = self._infer_feat_dim(in_chans, in_times)
        self.classifier = nn.Linear(p, n_classes)
        self.feat_dim = p

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                              # [B,1,C,T]
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = _square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=self.safe_log_eps))
        return torch.flatten(x, 1)                      # z (pre-dropout)

    def _infer_feat_dim(self, in_chans: int, in_times: int) -> int:
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device                 # match params (robust if built on GPU)
        try:
            with torch.no_grad():
                z = self._features(torch.zeros(1, in_chans, in_times, device=dev))
        finally:
            self.train(was_training)
        return int(z.shape[1])

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self._features(x)
        logits = self.classifier(self.drop(z))          # dropout applied to the head, NOT to z
        return ModelOutput(logits=logits, z=z)
