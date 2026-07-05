"""Minimal, faithful internal EEG MI decoders for the CIGL Phase 3A-S backbone-sanity check.

Pure PyTorch (NO braindecode / PyG) so the whole sanity check — tests, CPU dry-run, and the real GPU
run — uses a single env. These are compact reimplementations of well-established raw-EEG decoders
(EEGNet, ShallowConvNet, DeepConvNet); they are the "known-good" task references against which
GraphCMINet's near-chance baseline (Phase 3A-R) is judged. Each exposes the harness contract
`forward(x[B,C,T]) -> (logits, z)` with a probed `z_dim` (penultimate features), so `train_model` can
train them with plain ERM exactly like the graph backbones. They deliberately expose NO graph/node/edge
objects (no `forward_graph`) — non-graph CNNs must not fabricate graph leakage.

Refs: EEGNet (arXiv:1611.08024), ShallowConvNet/DeepConvNet (Schirrmeister 2017, arXiv:1703.05051).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _odd(k):
    return int(k) if int(k) % 2 == 1 else int(k) + 1


class EEGNetMini(nn.Module):
    """EEGNetv4-style: temporal conv -> depthwise spatial conv -> separable conv -> linear head.
    Window-robust: the temporal kernel and pools shrink for short inputs so it builds at any n_times."""
    def __init__(self, n_chans, n_times, n_classes, F1=8, D=2, F2=16, kern=64, drop=0.25):
        super().__init__()
        kern = min(int(kern), max(2, n_times // 2))
        p1 = min(4, max(1, n_times // 8))
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),          # depthwise spatial
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, p1)), nn.Dropout(drop))
        sk = min(16, max(2, (n_times // p1) // 2))
        p2 = min(8, max(1, (n_times // p1) // 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, sk), padding=(0, sk // 2), groups=F1 * D, bias=False),  # depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),                           # pointwise (separable)
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, p2)), nn.Dropout(drop))
        with torch.no_grad():
            z = self._features(torch.zeros(1, 1, n_chans, n_times))
        self.z_dim = int(z.shape[1])
        self.head = nn.Linear(self.z_dim, n_classes)

    def _features(self, x):
        return self.block2(self.block1(x)).flatten(1)

    def forward(self, x):
        z = self._features(x.unsqueeze(1))
        return self.head(z), z


class ShallowConvNetMini(nn.Module):
    """ShallowConvNet (FBCSP-inspired): temporal conv -> spatial conv -> square -> mean-pool -> log."""
    def __init__(self, n_chans, n_times, n_classes, n_filt=40, t_kern=25, pool=75, stride=15, drop=0.5):
        super().__init__()
        t_kern = min(int(t_kern), max(2, n_times // 2))
        self.temporal = nn.Conv2d(1, n_filt, (1, t_kern))
        self.spatial = nn.Conv2d(n_filt, n_filt, (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filt)
        t_after = n_times - t_kern + 1
        pool = min(int(pool), max(2, t_after))
        stride = min(int(stride), max(1, pool // 5))
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            z = self._features(torch.zeros(1, 1, n_chans, n_times))
        self.z_dim = int(z.shape[1])
        self.head = nn.Linear(self.z_dim, n_classes)

    def _features(self, x):
        x = self.bn(self.spatial(self.temporal(x)))
        x = self.pool(x ** 2)                       # square then mean-pool (band-power-like)
        x = torch.log(x.clamp(min=1e-6))
        return self.drop(x).flatten(1)

    def forward(self, x):
        z = self._features(x.unsqueeze(1))
        return self.head(z), z


class DeepConvNetMini(nn.Module):
    """Compact DeepConvNet: a temporal+spatial first block then 2 conv-pool blocks (ELU, max-pool)."""
    def __init__(self, n_chans, n_times, n_classes, drop=0.5):
        super().__init__()
        k = min(10, max(2, n_times // 8))
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, k)), nn.Conv2d(25, 25, (n_chans, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, min(3, max(1, (n_times - k) // 3)))), nn.Dropout(drop))
        self.b2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, k), padding=(0, k // 2)), nn.BatchNorm2d(50), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(drop))
        self.b3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, k), padding=(0, k // 2)), nn.BatchNorm2d(100), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(drop))
        with torch.no_grad():
            z = self._features(torch.zeros(1, 1, n_chans, n_times))
        self.z_dim = int(z.shape[1])
        self.head = nn.Linear(self.z_dim, n_classes)

    def _features(self, x):
        return self.b3(self.b2(self.b1(x))).flatten(1)

    def forward(self, x):
        z = self._features(x.unsqueeze(1))
        return self.head(z), z


class EEGNetMiniCSPInit(EEGNetMini):
    """CIGL_56 P10: EEGNetMini whose DEPTHWISE SPATIAL conv (block1[2]: Conv2d(F1, F1*D, (C,1), groups=F1))
    is initialized from SOURCE-only CSP filters, then trained normally (not frozen). `spatial_init='source_csp'`
    triggers train_model's generic CSP hook, which fits CSP on source-train ONLY (target excluded by LOSO,
    source-val excluded before the fit) and calls init_spatial_from_csp. Tests whether the CSP-init mechanism
    (proven on the FBCSP-LGG graph model in P8) also helps the strongest compact decoder. EEGNetMini itself is
    unchanged (frozen baseline)."""

    def __init__(self, n_chans, n_times, n_classes, cov_shrinkage=0.1, **kw):
        super().__init__(n_chans, n_times, n_classes, **kw)
        self.n_classes = int(n_classes)
        self.spatial_init = "source_csp"
        self.cov_shrinkage = float(cov_shrinkage)
        self.csp_meta = {"spatial_init": "source_csp"}

    @torch.no_grad()
    def init_spatial_from_csp(self, X, y, n_cls, m=None, source_domains=None, excluded_val=None):
        import numpy as np
        from cmi.models.csp_init import source_csp_filters
        conv = self.block1[2]                              # depthwise spatial conv, weight [F1*D, 1, C, 1]
        n_out, _, C, _ = conv.weight.shape
        n_contrasts = n_cls if n_cls > 2 else 2            # one-vs-rest (n_cls) or binary (2)
        m = int(m) if m else max(1, -(-n_out // n_contrasts))   # ceil(n_out/n_contrasts): pool >= n_out slots
        W, disc, present = source_csp_filters(X, y, n_cls, m, shrinkage=self.cov_shrinkage)
        order = np.argsort(disc)[::-1]                     # most discriminative first
        Wt = torch.as_tensor(W[order], dtype=conv.weight.dtype)
        k = min(n_out, Wt.shape[0])
        for o in range(k):
            conv.weight.data[o, 0, :, 0] = Wt[o]           # top-k CSP filters -> spatial slots; rest random
        self.csp_meta = {
            "spatial_init": "source_csp",
            "csp_fit_subjects": (sorted(int(d) for d in np.unique(source_domains))
                                 if source_domains is not None else None),
            "csp_excluded_target": True,
            "csp_excluded_source_val": sorted(int(v) for v in excluded_val) if excluded_val else [],
            "csp_n_filters_used": int(k), "csp_n_filters_pool": int(W.shape[0]),
            "csp_rank": int(np.linalg.matrix_rank(W)), "csp_cov_shrinkage": float(self.cov_shrinkage),
            "csp_m_per_contrast": int(m), "csp_classes_present": [int(c) for c in present],
        }
        return self.csp_meta


def build_sanity_backbone(name, n_chans, n_times, n_classes):
    """name -> pure-torch (logits, z) decoder. graphcmi/dgcnn come from cmi.models.gnn (graph backbones);
    eegnet/shallow_convnet/deep_convnet are the minimal internal CNNs here."""
    if name in ("graphcmi", "graphcmi_current_ref"):
        from cmi.models.gnn import GraphCMINet
        return GraphCMINet(n_chans, n_times, n_classes)
    if name == "dgcnn":
        from cmi.models.gnn import DGCNNBackbone
        return DGCNNBackbone(n_chans, n_times, n_classes)
    return {"eegnet": EEGNetMini, "shallow_convnet": ShallowConvNetMini,
            "deep_convnet": DeepConvNetMini}[name](n_chans, n_times, n_classes)
