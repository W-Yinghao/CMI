"""EEG-specific encoder (review section 5.2): three branches -> fused (z_c, z_n).

  TemporalBranch   EEGNet-style learnable filterbank: temporal conv + depthwise spatial
                   conv + pooling. Captures ERD/ERS, rhythms, transients.
  SPDBranch        per-band regularised channel covariance -> BiMap -> matrix-log
                   tangent vector. EEG cross-subject/session shift is largely spatial
                   mixing/scaling/covariance shift, so this branch is first-class, not a
                   plain latent CORAL afterthought (review 5.2; SPDIM motivation).
  GraphBranch      permutation-aware electrode set/graph encoder over per-channel band
                   features, so different channel counts/montages are handled natively
                   rather than by interpolation.

The fused representation is split into a task latent ``z_c`` and a nuisance latent
``z_n`` (review 5.6). A constrained near-identity ``Canonicalizer`` acts on ``z_c`` so
acquisition nuisance is removed by a *structured* map rather than a free MLP; its
deviation from identity is exposed via ``canon_penalty`` for the trust-region term.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from h2cmi.config import EncoderConfig


def _spd_logm_vech(C: torch.Tensor, eig_floor: float = 1e-3) -> torch.Tensor:
    """Matrix-log of a batch of SPD matrices [B,r,r] -> tangent vech vector [B,r(r+1)/2].

    Off-diagonals scaled by sqrt(2) so the Euclidean norm of the vech equals the
    Frobenius norm of the (symmetric) log -- the standard tangent-space metric.
    """
    C = 0.5 * (C + C.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(C)
    evals = evals.clamp_min(eig_floor)
    logC = evecs @ torch.diag_embed(torch.log(evals)) @ evecs.transpose(-1, -2)
    r = logC.shape[-1]
    iu = torch.triu_indices(r, r, offset=0, device=C.device)
    vech = logC[..., iu[0], iu[1]]
    off = iu[0] != iu[1]
    scale = torch.where(off, math.sqrt(2.0), 1.0).to(vech.dtype)
    return vech * scale


class TemporalBranch(nn.Module):
    """EEGNet-ish learnable filterbank temporal+spatial encoder."""

    def __init__(self, n_chans, n_times, n_bands, temporal_filters, dropout):
        super().__init__()
        F1 = temporal_filters
        k = max(9, n_times // 16) | 1                       # odd temporal kernel
        self.temporal = nn.Conv2d(1, F1 * n_bands, (1, k), padding=(0, k // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1 * n_bands)
        self.spatial = nn.Conv2d(F1 * n_bands, F1 * n_bands, (n_chans, 1),
                                 groups=F1 * n_bands, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * n_bands)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.drop = nn.Dropout(dropout)
        self.out_dim = F1 * n_bands * 8

    def forward(self, x):                                   # x [B, C, T]
        h = x.unsqueeze(1)                                  # [B,1,C,T]
        h = self.bn1(self.temporal(h))
        h = F.elu(self.bn2(self.spatial(h)))                # [B,F,1,T]
        h = self.drop(self.pool(h))                         # [B,F,1,8]
        return h.flatten(1)


class SPDBranch(nn.Module):
    """Per-band regularised covariance -> BiMap -> log-tangent vector."""

    def __init__(self, n_chans, n_times, n_bands, rank, shrinkage, eig_floor=1e-3):
        super().__init__()
        self.n_bands, self.n_chans = n_bands, n_chans
        self.shrinkage, self.eig_floor = shrinkage, eig_floor
        ksz = min(25, n_times) | 1
        # learnable depthwise band filters (one bank per band), groups=n_chans
        self.filters = nn.ModuleList([
            nn.Conv1d(n_chans, n_chans, ksz, padding=ksz // 2, groups=n_chans, bias=False)
            for _ in range(n_bands)])
        rank = min(rank, n_chans)
        self.rank = rank
        # BiMap weights with (near-)orthonormal rows
        self.W = nn.ParameterList([
            nn.Parameter(self._orth(rank, n_chans)) for _ in range(n_bands)])
        self.out_dim = n_bands * (rank * (rank + 1) // 2)

    @staticmethod
    def _orth(r, c):
        w = torch.randn(r, c)
        q, _ = torch.linalg.qr(w.T)
        return q.T[:r].contiguous()

    def forward(self, x):                                   # x [B, C, T]
        B, C, T = x.shape
        outs = []
        for b in range(self.n_bands):
            xb = self.filters[b](x)                         # band-limited [B,C,T]
            xb = xb - xb.mean(2, keepdim=True)
            cov = torch.einsum("bct,bdt->bcd", xb, xb) / max(T - 1, 1)
            tr = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)            # [B]
            ridge = self.shrinkage * tr.view(B, 1, 1) * torch.eye(C, device=x.device)
            cov = (1 - self.shrinkage) * cov + ridge                       # shrinkage
            W = self.W[b]                                                  # [r,C]
            Cb = W @ cov @ W.transpose(-1, -2)                             # [B,r,r] BiMap
            outs.append(_spd_logm_vech(Cb, self.eig_floor))
        return torch.cat(outs, dim=1)


class GraphBranch(nn.Module):
    """Permutation-aware electrode set/graph encoder over per-channel band features.

    Node feature = per-channel log band-power across the filterbank bands. One
    cosine-adjacency message-passing layer + DeepSets (mean/max) pooling -> vector. The
    pooling is permutation-invariant, so different channel orders/counts are native.
    """

    def __init__(self, n_chans, n_times, bands, fs, hidden, dropout):
        super().__init__()
        self.bands, self.fs = bands, fs
        self.node_in = len(bands)
        self.embed = nn.Sequential(nn.Linear(self.node_in, hidden), nn.ELU())
        self.msg = nn.Linear(hidden, hidden)
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ELU(), nn.Dropout(dropout))
        self.out_dim = 2 * hidden

    def _band_power(self, x):                               # x [B,C,T] -> [B,C,n_bands]
        B, C, T = x.shape
        Xf = torch.fft.rfft(x, dim=2)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(x.device)
        psd = (Xf.real ** 2 + Xf.imag ** 2)
        feats = []
        for lo, hi in self.bands:
            m = (freqs >= lo) & (freqs < hi)
            p = psd[:, :, m].mean(2) if m.any() else psd.mean(2)
            feats.append(torch.log1p(p))
        return torch.stack(feats, dim=2)                    # [B,C,n_bands]

    def forward(self, x):                                   # x [B,C,T]
        feat = self._band_power(x)                          # [B,C,n_bands]
        h = self.embed(feat)                                # [B,C,H]
        # cosine adjacency, row-normalised message passing
        hn = F.normalize(h, dim=2)
        adj = torch.bmm(hn, hn.transpose(1, 2)).clamp_min(0)   # [B,C,C]
        adj = adj / (adj.sum(2, keepdim=True) + 1e-6)
        agg = torch.bmm(adj, self.msg(h))                   # [B,C,H]
        h = self.upd(torch.cat([h, agg], dim=2))            # [B,C,H]
        return torch.cat([h.mean(1), h.max(1).values], dim=1)


class Canonicalizer(nn.Module):
    """Constrained near-identity affine map on z_c (review 5.2).

    Initialised to identity; ``penalty()`` returns ||W - I||_F^2 + ||b||^2 for the
    trust-region term so the canonicaliser removes structured nuisance without freely
    rewriting the latent geometry.
    """

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return z @ self.W.T + self.b

    def penalty(self):
        I = torch.eye(self.W.shape[0], device=self.W.device)
        return ((self.W - I) ** 2).sum() + (self.b ** 2).sum()


class H2Encoder(nn.Module):
    """Three-branch EEG encoder -> (z_c, z_n)."""

    def __init__(self, cfg: EncoderConfig, n_classes: int = 2):
        super().__init__()
        self.cfg = cfg
        n_bands = len(cfg.bands)
        branches, dim = [], 0
        self.temporal = self.spd = self.graph = None
        if cfg.use_temporal:
            self.temporal = TemporalBranch(cfg.n_chans, cfg.n_times, n_bands,
                                           cfg.temporal_filters, cfg.dropout)
            dim += self.temporal.out_dim
        if cfg.use_spd:
            self.spd = SPDBranch(cfg.n_chans, cfg.n_times, n_bands, cfg.spd_rank,
                                 cfg.cov_shrinkage)
            dim += self.spd.out_dim
        if cfg.use_graph:
            self.graph = GraphBranch(cfg.n_chans, cfg.n_times, cfg.bands, cfg.fs,
                                     cfg.graph_hidden, cfg.dropout)
            dim += self.graph.out_dim
        if dim == 0:
            raise ValueError("at least one encoder branch must be enabled")
        self.fuse = nn.Sequential(
            nn.Linear(dim, cfg.fuse_hidden), nn.ELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fuse_hidden, cfg.z_c_dim + cfg.z_n_dim))
        self.canon = Canonicalizer(cfg.z_c_dim) if cfg.canonicalizer else None
        self.z_c_dim, self.z_n_dim = cfg.z_c_dim, cfg.z_n_dim

    def forward(self, x):
        """x [B, C, T] -> (z_c [B,z_c_dim], z_n [B,z_n_dim])."""
        feats = []
        if self.temporal is not None:
            feats.append(self.temporal(x))
        if self.spd is not None:
            feats.append(self.spd(x))
        if self.graph is not None:
            feats.append(self.graph(x))
        h = self.fuse(torch.cat(feats, dim=1))
        z_c, z_n = h[:, :self.z_c_dim], h[:, self.z_c_dim:]
        if self.canon is not None:
            z_c = self.canon(z_c)
        return z_c, z_n

    def canon_penalty(self):
        return self.canon.penalty() if self.canon is not None else torch.zeros((), device=next(self.parameters()).device)


if __name__ == "__main__":
    cfg = EncoderConfig(n_chans=16, n_times=256, z_c_dim=32, z_n_dim=16)
    enc = H2Encoder(cfg)
    x = torch.randn(8, 16, 256, requires_grad=True)
    zc, zn = enc(x)
    print("z_c", tuple(zc.shape), "z_n", tuple(zn.shape),
          "params", sum(p.numel() for p in enc.parameters()))
    (zc.sum() + zn.sum() + enc.canon_penalty()).backward()
    print("grad ok:", x.grad is not None and bool(torch.isfinite(x.grad).all()))
