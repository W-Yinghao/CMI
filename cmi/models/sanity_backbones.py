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


class EEGConformerMini(nn.Module):
    """Faithful-minimal EEG Conformer (Song et al. 2022, arXiv:2106.11170): a shallow conv tokenizer (temporal
    conv -> spatial conv collapsing channels -> BN/ELU -> windowed avgpool) turns the raw EEG into a token
    sequence, a small Transformer encoder mixes the tokens, then flatten -> a SINGLE nn.Linear head. Same harness
    contract `forward(x[B,C,T]) -> (logits, z)` with feature_z = flattened transformer output, so head-replay is
    exact (logits = z @ head.Wᵀ + b in eval). Minimal INTERNAL reimplementation — NOT the official/braindecode
    EEGConformer (pure torch, single env). High-capacity arm vs the EEGNetMini anchor."""
    def __init__(self, n_chans, n_times, n_classes, emb=32, depth=2, heads=4, k_t=25, pool=15, stride=7, drop=0.3):
        super().__init__()
        k_t = min(int(k_t), max(2, n_times // 2))
        pool = min(int(pool), max(2, n_times // 4)); stride = min(int(stride), max(1, pool // 2))
        self.tok = nn.Sequential(
            nn.Conv2d(1, emb, (1, k_t), padding=(0, k_t // 2)),                  # temporal
            nn.Conv2d(emb, emb, (n_chans, 1)),                                  # spatial (collapse channels)
            nn.BatchNorm2d(emb), nn.ELU(),
            nn.AvgPool2d((1, pool), stride=(1, stride)), nn.Dropout(drop))
        layer = nn.TransformerEncoderLayer(emb, heads, dim_feedforward=emb * 2, dropout=drop,
                                           activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        with torch.no_grad():
            z = self._features(torch.zeros(1, 1, n_chans, n_times))
        self.z_dim = int(z.shape[1])
        self.head = nn.Linear(self.z_dim, n_classes)

    def _features(self, x):
        h = self.tok(x)                                                          # [B, emb, 1, T']
        h = h.squeeze(2).transpose(1, 2)                                         # [B, T', emb] tokens
        h = self.transformer(h)                                                  # [B, T', emb]
        return h.flatten(1)                                                      # [B, T'*emb] pre-head feature

    def forward(self, x):
        z = self._features(x.unsqueeze(1))
        return self.head(z), z


class EEGConformerFull(nn.Module):
    """INTERNAL full-capacity / official-geometry-INSPIRED EEG Conformer arm (after Song et al. 2022,
    arXiv:2106.11170) — the CIGL_69A2 high-capacity VALIDATION arm for the ConformerMini audit. This is NOT the
    official braindecode EEGConformer (that class is unimportable in eeg2025 -- moabb lacks BNCI2014001); it is a
    project-internal pure-torch reimplementation that mirrors the published GEOMETRY: shallow conv tokenizer with
    the paper's kernels (temporal (1,25) -> spatial (chans,1) -> BN/ELU -> AvgPool(1,75) stride 15 -> 1x1
    projection), a depth-6 / emb-40 / 10-head Transformer, then the paper's 3-layer MLP classification head
    (Linear->ELU->Drop->Linear->ELU->Drop->Linear). ~8-10x the params of EEGConformerMini.

    Contract `forward(x[B,C,T]) -> (logits, feature_z)` with feature_z = the flattened transformer output = the
    INPUT to the MLP head. Because the head is an MLP (not a single nn.Linear), head-replay is NOT exact; R3
    therefore falls back to the source-fit probe (removal_mode='probe_replay'). Pure torch (no braindecode),
    single env. The paper geometry needs n_times >= ~100 (temporal conv is valid); the pool is clamped only for
    tiny synthetic smokes so it never crashes, and equals the paper's (75,15) on real EEG."""
    def __init__(self, n_chans, n_times, n_classes, emb=40, depth=6, heads=10, k_t=25,
                 pool=75, stride=15, drop=0.5, mlp_hidden=(256, 32)):
        super().__init__()
        feat_t = n_times - (int(k_t) - 1)                                        # valid temporal conv output width
        pool = max(2, min(int(pool), feat_t)); stride = max(1, min(int(stride), max(1, pool // 5)))
        self.tok = nn.Sequential(
            nn.Conv2d(1, emb, (1, k_t)),                                          # temporal (valid, official)
            nn.Conv2d(emb, emb, (n_chans, 1)),                                   # spatial (collapse channels)
            nn.BatchNorm2d(emb), nn.ELU(),
            nn.AvgPool2d((1, pool), stride=(1, stride)), nn.Dropout(drop),
            nn.Conv2d(emb, emb, (1, 1)))                                          # 1x1 projection (official)
        layer = nn.TransformerEncoderLayer(emb, heads, dim_feedforward=emb * 4, dropout=drop,
                                           activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        with torch.no_grad():
            z = self._features(torch.zeros(1, 1, n_chans, n_times))
        self.z_dim = int(z.shape[1])
        h1, h2 = mlp_hidden                                                       # official 3-layer MLP head (NOT linear)
        self.head = nn.Sequential(nn.Linear(self.z_dim, h1), nn.ELU(), nn.Dropout(drop),
                                  nn.Linear(h1, h2), nn.ELU(), nn.Dropout(0.3),
                                  nn.Linear(h2, n_classes))

    def _features(self, x):
        h = self.tok(x)                                                          # [B, emb, 1, T']
        h = h.squeeze(2).transpose(1, 2)                                         # [B, T', emb] tokens
        h = self.transformer(h)                                                  # [B, T', emb]
        return h.flatten(1)                                                      # [B, T'*emb] pre-head feature_z

    def forward(self, x):
        z = self._features(x.unsqueeze(1))
        return self.head(z), z


def build_sanity_backbone(name, n_chans, n_times, n_classes):
    """name -> pure-torch (logits, z) decoder. graphcmi/dgcnn come from cmi.models.gnn (graph backbones);
    eegnet/shallow_convnet/deep_convnet/conformer are the minimal internal decoders here."""
    if name in ("graphcmi", "graphcmi_current_ref"):
        from cmi.models.gnn import GraphCMINet
        return GraphCMINet(n_chans, n_times, n_classes)
    if name == "dgcnn":
        from cmi.models.gnn import DGCNNBackbone
        return DGCNNBackbone(n_chans, n_times, n_classes)
    return {"eegnet": EEGNetMini, "shallow_convnet": ShallowConvNetMini,
            "deep_convnet": DeepConvNetMini, "conformer": EEGConformerMini,
            "conformer_full": EEGConformerFull}[name](n_chans, n_times, n_classes)
