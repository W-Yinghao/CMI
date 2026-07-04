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


# --------------------------------------------------------------------------------------------------
# central_strip_v1 — explicit sensorimotor-strip electrode groups for the MI datasets (P3-H).
# The coarse region taxonomy degenerated on these caps (BNCI2014 -> one 17-node central blob;
# BNCI2015 -> index fallback), which is not a credible "local-global montage graph". These presets
# split the FC / C / CP strip by anterior-central-posterior x left-mid-right so no group dominates.
# --------------------------------------------------------------------------------------------------
_CENTRAL_STRIP_V1 = {
    "BNCI2014_001": {
        "FC_left": ["FC3", "FC1"], "FC_mid": ["Fz", "FCz"], "FC_right": ["FC2", "FC4"],
        "C_left": ["C5", "C3", "C1"], "C_mid": ["Cz"], "C_right": ["C2", "C4", "C6"],
        "CP_left": ["CP3", "CP1", "P1"], "CP_mid": ["CPz", "Pz", "POz"], "CP_right": ["CP2", "CP4", "P2"],
    },
    "BNCI2015_001": {   # 13 sparser channels -> keep FC/CP as strips, split only the dense C row L/mid/R
        "FC_strip": ["FC3", "FCz", "FC4"],
        "C_left": ["C5", "C3", "C1"], "C_mid": ["Cz"], "C_right": ["C2", "C4", "C6"],
        "CP_strip": ["CP3", "CPz", "CP4"],
    },
}


def central_strip_groups(dataset, ch_names):
    """Resolve the central_strip_v1 preset for `dataset` against `ch_names` (matched BY NAME, so channel
    order does not matter). Returns (index_groups, named_groups, warning).

    Fail-closed: if the dataset has a preset but an electrode is missing or coverage is not exactly-once,
    returns (None, None, <warning>) so the caller can refuse rather than silently mis-group. Datasets
    with no preset return (None, None, "no preset ...").
    """
    spec = _CENTRAL_STRIP_V1.get(dataset)
    if spec is None or ch_names is None:
        return None, None, f"no central_strip_v1 preset for dataset={dataset}"
    name_to_idx = {nm: i for i, nm in enumerate(ch_names)}
    index_groups, named_groups, used = [], {}, []
    for gname, elecs in spec.items():
        idx = []
        for e in elecs:
            if e not in name_to_idx:
                return None, None, f"central_strip_v1[{dataset}]: electrode {e!r} not in ch_names"
            idx.append(name_to_idx[e])
        index_groups.append(idx)
        named_groups[gname] = list(elecs)
        used.extend(idx)
    if sorted(used) != list(range(len(ch_names))):
        return None, None, (f"central_strip_v1[{dataset}] covers {sorted(set(used))} but data has "
                            f"{len(ch_names)} channels (need each exactly once)")
    return index_groups, named_groups, None


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
                 glob_dim=16, z_dim=32, temp_dim=32, fused_z_dim=32, max_groups=6, dropout=0.25,
                 groups=None, group_names=None, grouping_scheme=None):
        super().__init__()
        self.n_chans = int(n_chans)
        self.drop = nn.Dropout(float(dropout))    # regularization vs source memorization (G2 had source bAcc=1.0)
        self.stem = _FilterBankStem(n_times, n_filt=n_filt, kernels=kernels)
        fin = self.stem.feat_dim

        # --- channel groups + block-diagonal (within-group) mask and row-normalized group-pool ---
        # Explicit `groups` (e.g. central_strip_v1, P3-H) override the region/index auto-builder.
        if groups is not None:
            self.groups = [list(g) for g in groups]
            self.grouping_scheme = grouping_scheme or "explicit"
        else:
            self.groups = build_channel_groups(self.n_chans, ch_names=ch_names, max_groups=max_groups)
            self.grouping_scheme = grouping_scheme or "region_or_index"
        self.group_names = group_names            # {group_name: [electrode names]} for provenance, or None
        # fail closed on a malformed grouping: every channel must appear in exactly one group
        _flat = sorted(c for g in self.groups for c in g)
        if _flat != list(range(self.n_chans)):
            raise ValueError(f"channel groups must cover 0..{self.n_chans - 1} exactly once; got {self.groups}")
        self.n_groups = len(self.groups)
        groups = self.groups
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


# --------------------------------------------------------------------------------------------------
# CIGL_49 P5 — FBCSP-style spatial-spectral branch (fixes the CSP gap found in P4/CIGL_48).
# P4 showed FBLGG underperforms classical CSP+LDA on 4-class cross-subject because it omits spatial
# collapse (channels stay as graph nodes). This branch reintroduces the CSP recipe:
#   per temporal band -> learned per-band spatial projection (K components) -> log-variance.
# --------------------------------------------------------------------------------------------------
class _FBCSPBand(nn.Module):
    """One filterbank band: temporal conv -> BN -> per-band spatial projection (C -> K, grouped per
    temporal filter) -> log-variance over time. Output [B, n_filt*K] CSP-style features."""

    def __init__(self, n_filt, kern, n_times, n_chans, K=4):
        super().__init__()
        kern = min(int(kern), max(2, n_times // 2))
        self.temporal = nn.Conv2d(1, n_filt, (1, kern))
        self.bn = nn.BatchNorm2d(n_filt)
        # grouped (C,1) conv = independent spatial filters per temporal band -> K components each
        self.spatial = nn.Conv2d(n_filt, n_filt * K, (n_chans, 1), groups=n_filt, bias=False)
        self.out_dim = n_filt * K

    def forward(self, x):                                  # x [B,1,C,T]
        h = self.bn(self.temporal(x))                      # [B,n_filt,C,T']
        h = self.spatial(h).squeeze(2)                     # [B,n_filt*K,T'] (spatially filtered)
        return torch.log(h.var(dim=-1).clamp(min=1e-6))    # [B,n_filt*K] log-variance (CSP feature)


class _FBCSPSpatialBranch(nn.Module):
    """Multi-band FBCSP spatial-spectral branch -> spatial_z [B, spatial_z_dim]."""

    def __init__(self, n_chans, n_times, n_filt=8, kernels=(11, 21, 45), K=4, spatial_z_dim=32, dropout=0.25):
        super().__init__()
        self.bands = nn.ModuleList(_FBCSPBand(n_filt, k, n_times, n_chans, K) for k in kernels)
        feat = int(sum(b.out_dim for b in self.bands))
        self.proj = nn.Linear(feat, spatial_z_dim)
        self.drop = nn.Dropout(float(dropout))
        self.out_dim = int(spatial_z_dim)

    def forward(self, x):                                  # x [B,C,T]
        xin = x.unsqueeze(1)                               # [B,1,C,T]
        v = torch.cat([b(xin) for b in self.bands], dim=-1)  # [B, sum(n_filt*K)]
        return self.drop(F.elu(self.proj(v)))              # [B, spatial_z_dim]

    @torch.no_grad()
    def init_from_csp(self, W, disc):
        """P8-A: write CSP filters into each band's per-temporal-filter K spatial slots (top-K by
        discriminability, broadband/shared across bands). Extra slots keep their random init. Returns the
        number of CSP filters written per group (k = min(K, n_csp))."""
        import numpy as np
        order = np.argsort(disc)[::-1]                     # most discriminative first
        Wt = torch.as_tensor(W[order], dtype=torch.float32)  # [F, C]
        used = 0
        for band in self.bands:
            conv = band.spatial                            # Conv2d(n_filt, n_filt*K, (C,1), groups=n_filt)
            n_filt = conv.groups; K = conv.out_channels // n_filt
            k = min(K, Wt.shape[0])
            topk = Wt[:k].to(conv.weight.dtype)            # [k, C]
            w = conv.weight.data                           # [n_filt*K, 1, C, 1]
            for g in range(n_filt):
                for j in range(k):
                    w[g * K + j, 0, :, 0] = topk[j]
            used = k
        return used


class FBCSPLGGGraph(FBLGGDualCMIBackbone):
    """FBLGG + an FBCSP-style spatial-spectral branch, combined by a 3-way softmax-gated fusion.

    Branches: graph_z (local-global electrode graph), temporal_z (channel-mean temporal), spatial_z
    (FBCSP per-band spatial projection + log-var). The softmax gate over the 3 branches is bounded and
    sums to 1, so no branch can unconstrainedly dominate the scale; per-batch gate mean/std are exposed
    for instrumentation. Keeps the CIGL 5-tuple contract with a distinct fused_z and central_strip_v1
    grouping. ERM-capable; graphdualpc head-split works (distinct fused_z)."""

    def __init__(self, n_chans, n_times, n_classes, ch_names=None, spatial_n_filt=8, spatial_K=4,
                 spatial_z_dim=32, dropout=0.25, fusion_floor=0.0, spatial_init="random", **kw):
        super().__init__(n_chans, n_times, n_classes, ch_names=ch_names, dropout=dropout, **kw)
        self.spatial = _FBCSPSpatialBranch(self.n_chans, n_times, n_filt=spatial_n_filt,
                                           K=spatial_K, spatial_z_dim=spatial_z_dim, dropout=dropout)
        self.spatial_z_dim = int(spatial_z_dim)
        self.n_classes = int(n_classes)
        self.spatial_init = str(spatial_init)              # P8-A: 'random' (default) | 'source_csp'
        self.fusion_floor = float(fusion_floor)            # P6-B: gate floor eps; 0 -> plain softmax (off)
        fdim = self.fused_z_dim
        self.fuse_s = nn.Linear(spatial_z_dim, fdim)       # reuse super's fuse_g / fuse_t for graph/temporal
        self.gate3 = nn.Linear(3 * fdim, 3)                # softmax over [graph, temporal, spatial]
        self.head3 = nn.Linear(fdim, n_classes)            # new head on the 3-way fused_z
        self.spatial_aux_head = nn.Linear(spatial_z_dim, n_classes)  # P8-B: source-only aux classifier on spatial_z
        self.last_gate = None                              # [B,3] gate weights (set each forward_graph)
        self.last_spatial_z = None                         # P6-A: grad-carrying spatial_z for spatial encoder CMI
        self.last_aux = {}                                 # {graph_z, temporal_z, spatial_z, fused_z} (detached)
        self.csp_meta = {"spatial_init": self.spatial_init}  # P8-A: populated by init_spatial_from_csp
        self.meta = dict(graph_compatible=True, edge_logits_dynamic=False, node_identity_preserved=True,
                         distinct_fused_z=True, has_spatial_branch=True, spatial_init=self.spatial_init,
                         ablation_modes=("zero_graph", "zero_temporal", "zero_spatial", "permute_nodes"))

    @torch.no_grad()
    def init_spatial_from_csp(self, X, y, n_cls, m=None, shrinkage=0.1, source_domains=None, excluded_val=None):
        """P8-A: initialize the spatial branch from SOURCE-only CSP filters (target already excluded by the
        LOSO caller; source-val excluded before this is called), then leave them trainable. Records
        provenance in self.csp_meta. n_cls==2 -> binary CSP, n_cls>2 -> one-vs-rest."""
        import numpy as np
        from cmi.models.csp_init import source_csp_filters
        conv0 = self.spatial.bands[0].spatial
        K = conv0.out_channels // conv0.groups
        m = int(m) if m else K
        W, disc, present = source_csp_filters(X, y, n_cls, m, shrinkage=shrinkage)
        used = self.spatial.init_from_csp(W, disc)
        self.csp_meta = {
            "spatial_init": "source_csp",
            "csp_fit_subjects": (sorted(int(d) for d in np.unique(source_domains))
                                 if source_domains is not None else None),
            "csp_excluded_target": True,
            "csp_excluded_source_val": sorted(int(v) for v in excluded_val) if excluded_val else [],
            "csp_n_filters_used": int(used), "csp_n_filters_pool": int(W.shape[0]),
            "csp_rank": int(np.linalg.matrix_rank(W)), "csp_cov_shrinkage": float(shrinkage),
            "csp_m_per_contrast": int(m), "csp_classes_present": [int(c) for c in present],
        }
        return self.csp_meta

    def spatial_aux_logits(self, x):
        """P8-B: aux classifier logits on the (grad-carrying) spatial_z. Runs a forward first."""
        self.forward_graph(x)
        return self.spatial_aux_head(self.last_spatial_z)

    def _spatial_branch(self, x):
        return self.spatial(x)                             # [B, spatial_z_dim]

    def _fuse3(self, graph_z, temporal_z, spatial_z):
        gp = self.fuse_g(graph_z); tp = self.fuse_t(temporal_z); sp = self.fuse_s(spatial_z)  # [B,fdim] each
        gate = torch.softmax(self.gate3(torch.cat([gp, tp, sp], dim=-1)), dim=-1)             # [B,3], sums to 1
        if self.fusion_floor > 0.0:                          # P6-B: floor so no branch is fully starved
            eps = self.fusion_floor
            gate = (1.0 - 3.0 * eps) * gate + eps            # still sums to 1; each weight >= eps
        self.last_gate = gate.detach()
        fused = gate[:, 0:1] * gp + gate[:, 1:2] * tp + gate[:, 2:3] * sp
        return self.drop(fused)

    def forward_graph(self, x):
        node_raw = self.stem(x)
        graph_z, node_z = self._graph_branch(node_raw)
        temporal_z = self._temporal_branch(node_raw)
        spatial_z = self._spatial_branch(x)
        fused_z = self._fuse3(graph_z, temporal_z, spatial_z)
        self.last_spatial_z = spatial_z                      # grad-carrying (for spatial encoder CMI, P6-A)
        self.last_aux = dict(graph_z=graph_z.detach(), temporal_z=temporal_z.detach(),
                             spatial_z=spatial_z.detach(), fused_z=fused_z.detach())
        return self.head3(fused_z), graph_z, node_z, None, fused_z

    def forward(self, x):
        logits, _, _, _, fused_z = self.forward_graph(x)
        return logits, fused_z

    @torch.no_grad()
    def ablate(self, x, mode):
        """zero_graph / zero_temporal / zero_spatial: fuse with that branch's z:=0. permute_nodes:
        permute the graph-branch node features across the batch. Returns [B, n_cls] logits."""
        node_raw = self.stem(x); B = node_raw.shape[0]
        graph_z, _ = self._graph_branch(node_raw)
        temporal_z = self._temporal_branch(node_raw)
        spatial_z = self._spatial_branch(x)
        if mode == "zero_graph":
            graph_z = torch.zeros(B, self.z_dim, device=x.device, dtype=node_raw.dtype)
        elif mode == "zero_temporal":
            temporal_z = torch.zeros(B, self.temp_dim, device=x.device, dtype=node_raw.dtype)
        elif mode == "zero_spatial":
            spatial_z = torch.zeros(B, self.spatial_z_dim, device=x.device, dtype=node_raw.dtype)
        elif mode == "permute_nodes":
            perm = torch.randperm(B, device=node_raw.device)
            graph_z, _ = self._graph_branch(node_raw[perm])
        return self.head3(self._fuse3(graph_z, temporal_z, spatial_z))

    @torch.no_grad()
    def gate_summary(self, x):
        """Per-batch fusion-gate stats [graph, temporal, spatial] — instrumentation only (aggregate)."""
        was = self.training; self.eval()
        self.forward_graph(x); g = self.last_gate
        if was:
            self.train()
        names = ["graph", "temporal", "spatial"]
        out = {}
        for i, nm in enumerate(names):
            out[f"gate_{nm}_mean"] = float(g[:, i].mean()); out[f"gate_{nm}_std"] = float(g[:, i].std())
        ent = -(g.clamp(min=1e-8).log() * g).sum(dim=1)      # per-sample gate entropy (nats); max ln(3)=1.0986
        out["gate_entropy_mean"] = float(ent.mean())         # low -> collapsed to one branch; high -> balanced
        return out
