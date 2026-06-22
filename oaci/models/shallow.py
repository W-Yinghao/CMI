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
        try:
            with torch.no_grad():
                z = self._features(torch.zeros(1, in_chans, in_times))
        finally:
            self.train(was_training)
        return int(z.shape[1])

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self._features(x)
        logits = self.classifier(self.drop(z))          # dropout applied to the head, NOT to z
        return ModelOutput(logits=logits, z=z)
