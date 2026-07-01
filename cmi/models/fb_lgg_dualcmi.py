"""CIGL_47 — FB-LGG-DualCMI backbone (FilterBank temporal + Local–Global electrode Graph + gated fusion).

Successor to the static ``DGCNNGraph`` SOTA track (closed by the CIGL_46 G2 pilot: near-chance on
BNCI2014_001 with source bAcc = 1.000, and no stable dual-CMI gain on BNCI2015_001). The design goal is
a *stronger* ERM graph backbone whose cross-subject target bAcc is credible BEFORE any dual-CMI
regularization is added — regularizing a weak/overfitting backbone (the DGCNN result) only flattens or
hurts.

Exposes the CIGL graph contract with a **distinct** ``fused_z`` (5-tuple), unlike DGCNNGraph's 4-tuple:

    forward_graph(x) -> (logits, graph_z, node_z, edge_logits_or_none, fused_z)

  * ``node_z``  [B, C, node_z_dim] — per-channel features after the local graph stage → node-CMI head.
  * ``graph_z`` [B, z_dim]         — global (local→global) graph readout → encoder graph-CMI head.
  * ``edge_logits`` = None         — static shared global adjacency A0 in v1 (NO free per-sample A(x);
                                     that was the v0.6 subject-fingerprint failure mode).
  * ``fused_z`` [B, fused_z_dim]   — gated fusion of the graph readout and a temporal (non-graph) readout;
                                     this is the representation the classifier reads → decoder residual.

``ablate(x, mode)`` supports ``zero_graph`` / ``zero_temporal`` / ``permute_nodes`` so the runner can
verify BOTH branches contribute (with fusion, ``zero_graph`` no longer collapses to chance — it should
drop materially; a hidden temporal-only bypass would show near-zero drop).

Pure torch (no PyG); CPU-friendly. ERM-capable via ``forward(x) -> (logits, fused_z)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------------
# channel grouping (name-aware with a robust index-partition fallback)
# --------------------------------------------------------------------------------------------------
# 10–20 name prefix -> functional region. Best-effort; unknown names fall back to a contiguous
# index partition so the builder never crashes on an unfamiliar montage.
_REGION_PREFIXES = (
    ("frontal", ("FP", "AF", "F")),
    ("central", ("FC", "C", "CP")),
    ("temporal", ("T", "FT", "TP")),
    ("parietal", ("P",)),
    ("occipital", ("O", "PO")),
)


def _region_of(name):
    up = str(name).upper().strip()
    # longest-prefix match so 'CP3' -> central (CP) before 'C', 'PO7' -> occipital (PO) before 'P'
    best = None
    for region, prefixes in _REGION_PREFIXES:
        for p in prefixes:
            if up.startswith(p) and (best is None or len(p) > best[1]):
                best = (region, len(p))
    return best[0] if best else None


def _index_partition(n_chans, max_groups):
    g = max(1, min(int(max_groups), int(n_chans)))
    # nearly-equal contiguous chunks of channel indices
    base, rem = divmod(n_chans, g)
    groups, start = [], 0
    for i in range(g):
        size = base + (1 if i < rem else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return [grp for grp in groups if grp]


def build_channel_groups(n_chans, ch_names=None, max_groups=6):
    """Return a list of channel-index lists partitioning ``range(n_chans)``.

    Name-aware when ``ch_names`` (len == n_chans) maps to ≥2 known 10–20 regions; otherwise a
    contiguous index partition into ``min(max_groups, n_chans)`` groups. Never raises on unknown names.
    """
    if ch_names is not None and len(ch_names) == n_chans:
        buckets = {}
        for i, nm in enumerate(ch_names):
            r = _region_of(nm)
            buckets.setdefault(r, []).append(i)
        known = {r: idx for r, idx in buckets.items() if r is not None}
        if len(known) >= 2:
            groups = [known[r] for r, _ in _REGION_PREFIXES if r in known]
            leftover = buckets.get(None, [])
            if leftover:  # attach unrecognized channels to the nearest (first) group
                groups[0] = sorted(groups[0] + leftover)
            covered = sorted(c for grp in groups for c in grp)
            if covered == list(range(n_chans)):
                return groups
    return _index_partition(n_chans, max_groups)


def _norm_adj(A):
    """Symmetric GCN normalization with self-loops: D^-1/2 (A+I) D^-1/2. A: [n, n] non-negative."""
    n = A.shape[-1]
    A = A + torch.eye(n, device=A.device, dtype=A.dtype)
    d = A.sum(-1).clamp(min=1e-6).pow(-0.5)
    return d.unsqueeze(-1) * A * d.unsqueeze(-2)


# --------------------------------------------------------------------------------------------------
# filterbank temporal stem (channel-preserving -> node features [B, C, F_node])
# --------------------------------------------------------------------------------------------------
class _TemporalBand(nn.Module):
    """One filterbank band: temporal conv -> BN -> square -> windowed mean-pool -> log, KEEPING channels
    as nodes. Output per channel: [B, C, n_filt * n_windows]."""

    def __init__(self, n_filt, kern, n_times, pool=50, stride=12):
        super().__init__()
        kern = min(int(kern), max(2, n_times // 2))
        self.temporal = nn.Conv2d(1, n_filt, (1, kern))
        self.bn = nn.BatchNorm2d(n_filt)
        t_after = n_times - kern + 1
        pool = min(int(pool), max(2, t_after))
        stride = min(int(stride), max(1, pool // 2))
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        with torch.no_grad():
            self.feat_dim = int(self._feat(torch.zeros(1, 1, 2, n_times)).shape[-1])

    def _feat(self, x):                                    # x [B,1,C,T]
        h = self.bn(self.temporal(x))                      # [B,n_filt,C,T']
        h = torch.log(self.pool(h ** 2).clamp(min=1e-6))   # [B,n_filt,C,n_win]
        B, Fc, C, W = h.shape
        return h.permute(0, 2, 1, 3).reshape(B, C, Fc * W)  # [B,C,n_filt*n_win]


class _FilterBankStem(nn.Module):
    """Multi-kernel-size temporal filterbank. node_raw = concat over bands -> [B, C, F_node]."""

    def __init__(self, n_times, n_filt=6, kernels=(11, 21, 45), pool=50, stride=12):
        super().__init__()
        self.bands = nn.ModuleList(_TemporalBand(n_filt, k, n_times, pool, stride) for k in kernels)
        self.feat_dim = int(sum(b.feat_dim for b in self.bands))

    def forward(self, x):                                  # x [B,C,T]
        xin = x.unsqueeze(1)                               # [B,1,C,T]
        return torch.cat([b._feat(xin) for b in self.bands], dim=-1)   # [B,C,F_node]


# --------------------------------------------------------------------------------------------------
# FB-LGG-DualCMI backbone
# --------------------------------------------------------------------------------------------------
class FBLGGDualCMIBackbone(nn.Module):
    """FilterBank temporal stem -> Local (within-group) graph -> Global (group) graph -> gated fusion
    with a temporal branch. Static shared adjacencies (edge_logits=None). Returns a 5-tuple with a
    distinct ``fused_z`` (the classifier input)."""

    def __init__(self, n_chans, n_times, n_classes, ch_names=None,
                 n_filt=6, kernels=(11, 21, 45), loc_dim=16, node_z_dim=16,
                 glob_dim=16, z_dim=32, temp_dim=32, fused_z_dim=32, max_groups=6, dropout=0.25):
        super().__init__()
        self.n_chans = int(n_chans)
        self.drop = nn.Dropout(float(dropout))    # regularization vs source memorization (G2 had source bAcc=1.0)
        self.stem = _FilterBankStem(n_times, n_filt=n_filt, kernels=kernels)
        fin = self.stem.feat_dim

        # --- channel groups + block-diagonal (within-group) mask and row-normalized group-pool ---
        groups = build_channel_groups(self.n_chans, ch_names=ch_names, max_groups=max_groups)
        self.groups = groups
        self.n_groups = len(groups)
        mask = torch.zeros(self.n_chans, self.n_chans)
        pool_mat = torch.zeros(self.n_groups, self.n_chans)
        for gi, grp in enumerate(groups):
            for a in grp:
                pool_mat[gi, a] = 1.0 / len(grp)
                for b in grp:
                    mask[a, b] = 1.0
        self.register_buffer("group_mask", mask)           # [C,C] within-group connectivity
        self.register_buffer("group_pool", pool_mat)       # [n_groups,C] mean-pool within group

        # --- local graph stage: per-channel projection + within-group GCN -> node_z [B,C,node_z_dim] ---
        self.local_proj = nn.Linear(fin, loc_dim)
        self.A_local = nn.Parameter(torch.randn(self.n_chans, self.n_chans) * 0.01)
        self.local_gcn = nn.Linear(loc_dim, node_z_dim)
        self.node_z_dim = int(node_z_dim)

        # --- global graph stage: shared A0 over group tokens -> graph_z [B,z_dim] ---
        self.A_global = nn.Parameter(torch.randn(self.n_groups, self.n_groups) * 0.01)
        self.glob_gcn = nn.Linear(node_z_dim, glob_dim)
        self.readout = nn.Linear(self.n_groups * glob_dim, z_dim)
        self.z_dim = int(z_dim)

        # --- temporal (non-graph) branch: channel-mean of node_raw -> temporal_z [B,temp_dim] ---
        self.temp_proj = nn.Linear(fin, temp_dim)
        self.temp_dim = int(temp_dim)

        # --- gated fusion -> fused_z [B,fused_z_dim] (distinct object) -> classifier ---
        self.fuse_g = nn.Linear(z_dim, fused_z_dim)
        self.fuse_t = nn.Linear(temp_dim, fused_z_dim)
        self.gate = nn.Linear(2 * fused_z_dim, fused_z_dim)
        self.head = nn.Linear(fused_z_dim, n_classes)
        self.fused_z_dim = int(fused_z_dim)

        self.meta = dict(graph_compatible=True, edge_logits_dynamic=False,
                         node_identity_preserved=True, distinct_fused_z=True,
                         ablation_modes=("zero_graph", "zero_temporal", "permute_nodes"))

    # ---- internal branches (shared by forward_graph and ablate) ----
    def _graph_branch(self, node_raw):
        """node_raw [B,C,F_node] -> (graph_z [B,z_dim], node_z [B,C,node_z_dim])."""
        node_h = self.drop(F.elu(self.local_proj(node_raw)))            # [B,C,loc_dim]
        S_local = _norm_adj(F.softplus(self.A_local) * self.group_mask)  # [C,C] within-group
        node_z = F.elu(self.local_gcn(torch.einsum("ij,bjf->bif", S_local, node_h)))  # [B,C,node_z_dim]
        grp = torch.einsum("gc,bcf->bgf", self.group_pool, node_z)      # [B,n_groups,node_z_dim]
        S_glob = _norm_adj(F.softplus(self.A_global))                    # [n_groups,n_groups] shared A0
        gh = F.elu(self.glob_gcn(torch.einsum("ij,bjf->bif", S_glob, grp)))  # [B,n_groups,glob_dim]
        graph_z = self.drop(F.elu(self.readout(gh.reshape(gh.shape[0], -1))))   # [B,z_dim]
        return graph_z, node_z

    def _temporal_branch(self, node_raw):
        return self.drop(F.elu(self.temp_proj(node_raw.mean(dim=1))))    # [B,temp_dim]

    def _fuse(self, graph_z, temporal_z):
        gp = self.fuse_g(graph_z)                                        # [B,fused_z_dim]
        tp = self.fuse_t(temporal_z)                                     # [B,fused_z_dim]
        gate = torch.sigmoid(self.gate(torch.cat([gp, tp], dim=-1)))     # [B,fused_z_dim]
        return self.drop(gate * gp + (1.0 - gate) * tp)                  # fused_z (distinct object)

    # ---- CIGL graph contract ----
    def forward_graph(self, x):
        node_raw = self.stem(x)
        graph_z, node_z = self._graph_branch(node_raw)
        temporal_z = self._temporal_branch(node_raw)
        fused_z = self._fuse(graph_z, temporal_z)
        return self.head(fused_z), graph_z, node_z, None, fused_z        # 5-tuple, edge_logits=None

    def forward(self, x):
        logits, _, _, _, fused_z = self.forward_graph(x)
        return logits, fused_z                                           # classifier representation

    @torch.no_grad()
    def ablate(self, x, mode):
        """Branch-contribution hook. zero_graph: fuse with graph_z:=0 (drop = graph contribution).
        zero_temporal: fuse with temporal_z:=0 (drop = temporal contribution). permute_nodes: permute
        node features across the batch in the graph branch only (breaks trial<->node correspondence)."""
        node_raw = self.stem(x)
        temporal_z = self._temporal_branch(node_raw)
        B = node_raw.shape[0]
        if mode == "zero_graph":
            gz0 = torch.zeros(B, self.z_dim, device=x.device, dtype=node_raw.dtype)
            return self.head(self._fuse(gz0, temporal_z))
        if mode == "zero_temporal":
            graph_z, _ = self._graph_branch(node_raw)
            tz0 = torch.zeros(B, self.temp_dim, device=x.device, dtype=node_raw.dtype)
            return self.head(self._fuse(graph_z, tz0))
        if mode == "permute_nodes":
            perm = torch.randperm(B, device=node_raw.device)
            graph_z_p, _ = self._graph_branch(node_raw[perm])
            return self.head(self._fuse(graph_z_p, temporal_z))
        graph_z, _ = self._graph_branch(node_raw)
        return self.head(self._fuse(graph_z, temporal_z))
