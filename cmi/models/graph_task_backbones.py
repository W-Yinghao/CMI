"""CIGL Phase 3A-G — task-capable graph-compatible backbones (ERM-only redesign; pure torch, no PyG).

Phase 3A-S showed the strict source-only BNCI2014_001 fold-0 is learnable by known-good MI decoders
(EEGNet/ShallowConvNet/DeepConvNet ~0.52-0.56, even DGCNN 0.46) while GraphCMINet stays ~0.33 — so the
bottleneck is GraphCMINet's specific design, not graph learning. The likely culprit: its PowerLayer
collapses the ENTIRE time axis to one power value per (channel, filter), discarding the temporal
dynamics ShallowConvNet keeps via windowed pooling.

These backbones fix that while keeping electrodes as graph nodes and exposing the CIGL objects via
`forward_graph(x) -> (logits, graph_z, node_z, edge_logits_or_none)`:

  - ShallowGraphStemNet : ShallowConvNet-style windowed log-power per channel -> dynamic A(x) -> SGC
                          propagation -> flatten readout. Dynamic per-sample edge_logits.
  - EEGNetGraphStemNet  : EEGNet-style temporal blocks per channel (NO spatial collapse -> node identity
                          preserved) -> dynamic A(x) -> SGC -> flatten readout.
  - DGCNNForwardGraphAdapter : wraps the existing (task-passing) DGCNNBackbone and exposes graph_z/node_z;
                          its adjacency is SHARED/static so edge_logits is None and edge_logits_dynamic
                          is False (no faked dynamic edge object).

There is NO CNN bypass: logits depend ONLY on the graph readout `graph_z` (see `.ablate(x, mode)` and the
runner's graph-usage check). ERM-only; no CMI regularization here.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from cmi.models.gnn import Adjacency, DGCNNBackbone, _sgc_norm


class _ShallowChannelStem(nn.Module):
    """ShallowConvNet-style per-channel temporal features (temporal conv -> square -> windowed mean-pool
    -> log), KEEPING channels as nodes -> node features [B, C, n_filt*n_windows]."""
    def __init__(self, n_times, n_filt=20, t_kern=25, pool=50, stride=12):
        super().__init__()
        t_kern = min(int(t_kern), max(2, n_times // 2))
        self.temporal = nn.Conv2d(1, n_filt, (1, t_kern))
        self.bn = nn.BatchNorm2d(n_filt)
        t_after = n_times - t_kern + 1
        pool = min(int(pool), max(2, t_after))
        stride = min(int(stride), max(1, pool // 4))
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        with torch.no_grad():
            self.feat_dim = int(self._feat(torch.zeros(1, 1, 2, n_times)).shape[-1])

    def _feat(self, x):                                   # x [B,1,C,T]
        h = self.bn(self.temporal(x))                     # [B,n_filt,C,T']
        h = torch.log(self.pool(h ** 2).clamp(min=1e-6))  # [B,n_filt,C,n_win]
        B, Fc, C, W = h.shape
        return h.permute(0, 2, 1, 3).reshape(B, C, Fc * W)  # [B,C,n_filt*n_win]

    def forward(self, x):                                 # x [B,C,T]
        return self._feat(x.unsqueeze(1))


class _EEGNetChannelStem(nn.Module):
    """EEGNet-style temporal blocks per channel (temporal conv -> BN/ELU -> avgpool -> separable
    temporal). The depthwise SPATIAL conv is intentionally OMITTED so channel/node identity survives to
    node_z -> node features [B, C, F2*n_windows]."""
    def __init__(self, n_times, F1=8, kern=64, F2=16, drop=0.25):
        super().__init__()
        kern = min(int(kern), max(2, n_times // 2))
        p1 = min(4, max(1, n_times // 8))
        self.t1 = nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.pool1 = nn.AvgPool2d((1, p1))
        sk = min(16, max(2, (n_times // p1) // 2))
        p2 = min(8, max(1, (n_times // p1) // 2))
        self.t2 = nn.Conv2d(F1, F2, (1, sk), padding=(0, sk // 2), bias=False)   # temporal (no spatial collapse)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, p2))
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            self.feat_dim = int(self._feat(torch.zeros(1, 1, 2, n_times)).shape[-1])

    def _feat(self, x):
        h = self.pool1(F.elu(self.bn1(self.t1(x))))
        h = self.pool2(self.drop(F.elu(self.bn2(self.t2(h)))))   # [B,F2,C,T'']
        B, Fc, C, W = h.shape
        return h.permute(0, 2, 1, 3).reshape(B, C, Fc * W)

    def forward(self, x):
        return self._feat(x.unsqueeze(1))


class _DynamicGraphHead(nn.Module):
    """node_raw [B,C,Fin] -> per-sample adjacency A(x) (dynamic edge_logits) -> SGC propagation ->
    node_z [B,C,hidden] -> flatten readout graph_z [B,zdim] -> logits. Logits depend ONLY on graph_z."""
    def __init__(self, n_chans, fin, n_classes, hidden=32, hops=2, zdim=64):
        super().__init__()
        self.adj = Adjacency(n_chans, fin)
        self.hops = hops
        self.W = nn.Linear(fin, hidden)
        self.proj = nn.Linear(n_chans * hidden, zdim)
        self.head = nn.Linear(zdim, n_classes)
        self.zdim = zdim

    def propagate(self, node_raw):
        A, edge_logits = self.adj(node_raw)
        S = _sgc_norm(A)
        H = node_raw
        for _ in range(self.hops):
            H = S @ H
        return F.elu(self.W(H)), edge_logits          # node_z [B,C,hidden], edge_logits [B,C,C]

    def readout(self, node_z):
        return F.elu(self.proj(node_z.reshape(node_z.shape[0], -1)))    # graph_z [B,zdim]

    def classify(self, graph_z):
        return self.head(graph_z)

    def forward_graph(self, node_raw):
        node_z, edge_logits = self.propagate(node_raw)
        graph_z = self.readout(node_z)
        return self.classify(graph_z), graph_z, node_z, edge_logits


class _GraphStemNet(nn.Module):
    """Temporal stem (channel-preserving) -> dynamic graph head. Common to the two redesigned candidates."""
    def __init__(self, stem, n_chans, n_classes, hidden=32, hops=2):
        super().__init__()
        self.stem = stem
        self.gh = _DynamicGraphHead(n_chans, stem.feat_dim, n_classes, hidden, hops)
        self.z_dim = self.gh.zdim
        self.meta = dict(graph_compatible=True, edge_logits_dynamic=True, node_identity_preserved=True)

    def forward_graph(self, x):
        return self.gh.forward_graph(self.stem(x))

    def forward(self, x):
        logits, graph_z, _, _ = self.forward_graph(x)
        return logits, graph_z

    @torch.no_grad()
    def ablate(self, x, mode):
        """Graph-usage hook. zero_graph: classify a zeroed readout (logits lose all graph info ->
        ~chance, proving NO bypass). permute_nodes: read out node_z permuted across the batch (breaks the
        trial<->node correspondence -> task collapses if node content is used)."""
        node_z, _ = self.gh.propagate(self.stem(x))
        if mode == "zero_graph":
            return self.gh.classify(torch.zeros(node_z.shape[0], self.gh.zdim, device=x.device))
        if mode == "permute_nodes":
            perm = torch.randperm(node_z.shape[0], device=node_z.device)
            return self.gh.classify(self.gh.readout(node_z[perm]))
        return self.gh.classify(self.gh.readout(node_z))


class ShallowGraphStemNet(_GraphStemNet):
    def __init__(self, n_chans, n_times, n_classes, hidden=32, hops=2):
        super().__init__(_ShallowChannelStem(n_times), n_chans, n_classes, hidden, hops)


class EEGNetGraphStemNet(_GraphStemNet):
    def __init__(self, n_chans, n_times, n_classes, hidden=32, hops=2):
        super().__init__(_EEGNetChannelStem(n_times), n_chans, n_classes, hidden, hops)


class DGCNNForwardGraphAdapter(nn.Module):
    """Wrap the existing (task-passing) DGCNNBackbone and expose forward_graph. DGCNN's adjacency is a
    single SHARED parameter (not per-sample), so there is NO genuine per-sample edge object:
    edge_logits=None, edge_logits_dynamic=False (we do not fabricate dynamic edge leakage). forward() is
    byte-identical to DGCNNBackbone.forward, so training matches the Phase 3A-S DGCNN that cleared 0.45."""
    def __init__(self, n_chans, n_times, n_classes, feat=16, hidden=16):
        super().__init__()
        self.net = DGCNNBackbone(n_chans, n_times, n_classes, feat=feat, hidden=hidden)
        self.z_dim = self.net.z_dim
        self.meta = dict(graph_compatible=True, edge_logits_dynamic=False, node_identity_preserved=True)

    def _node_z(self, x):
        return F.elu(self.net.conv(self.net.enc(x.unsqueeze(1)), self.net.adj()))   # [B,C,hidden]

    def _readout(self, node_z):
        return F.elu(self.net.proj(node_z.reshape(node_z.shape[0], -1)))            # [B,64]

    def forward_graph(self, x):
        node_z = self._node_z(x)
        graph_z = self._readout(node_z)
        return self.net.head(graph_z), graph_z, node_z, None     # static adjacency -> no dynamic edge object

    def forward(self, x):
        logits, graph_z, _, _ = self.forward_graph(x)
        return logits, graph_z

    @torch.no_grad()
    def ablate(self, x, mode):
        node_z = self._node_z(x)
        if mode == "zero_graph":
            return self.net.head(torch.zeros(node_z.shape[0], self.net.z_dim, device=x.device))
        if mode == "permute_nodes":
            perm = torch.randperm(node_z.shape[0], device=node_z.device)
            return self.net.head(self._readout(node_z[perm]))
        return self.net.head(self._readout(node_z))


GRAPH_TASK_BACKBONES = ["shallow_graph_stem", "eegnet_graph_stem", "dgcnn_forward_graph_adapter"]


def build_graph_task_backbone(name, n_chans, n_times, n_classes, **kwargs):
    if name == "shallow_graph_stem":
        return ShallowGraphStemNet(n_chans, n_times, n_classes, **kwargs)
    if name == "eegnet_graph_stem":
        return EEGNetGraphStemNet(n_chans, n_times, n_classes, **kwargs)
    if name == "dgcnn_forward_graph_adapter":
        return DGCNNForwardGraphAdapter(n_chans, n_times, n_classes, **kwargs)
    raise KeyError(f"unknown graph task backbone: {name}")
