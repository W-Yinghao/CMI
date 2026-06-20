"""GraphCMINet — a RAW-SIGNAL EEG graph backbone for the Tri-CMI harness (plain PyTorch, no PyG/braindecode).
Design from notes/gnn_design.md. Pipeline:
  x[B,C,T] -> LGGNet-style per-channel temporal encoder (PowerLayer = log-avgpool-of-square, a learned
  band-power surrogate FROM RAW, no DE) -> node features [B,C,F] -> per-sample learnable adjacency A(x)
  (free symmetric base + data-dependent node-pair similarity) -> SGC propagation S^L X -> mean readout.
Exposes forward(x)->(logits, graph_Z) for the existing harness AND forward_graph(x)->(logits, graph_Z,
node_Z[B,C,d], edge_logits[B,C,C]) for the novel node-level (Σ_v I(Z_v;D|Y)) and edge-level (I(A;D|Y)) CMI.

Refs: LGGNet (arXiv:2105.02786, PowerLayer), RGNN/SGC (arXiv:1907.07835 / 1902.07153), SOGNN per-sample adj.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNodeEncoder(nn.Module):
    """Per-channel raw temporal encoder. Multi-scale temporal convs + PowerLayer -> node embedding [B,C,F]."""
    def __init__(self, n_times, num_T=16, out=16):
        super().__init__()
        ks = [max(2, n_times // 8), max(2, n_times // 16), max(2, n_times // 32)]   # ~0.5/0.25/0.125 * fs
        self.branches = nn.ModuleList([nn.Conv2d(1, num_T, (1, k), padding=(0, k // 2)) for k in ks])
        self.bn = nn.BatchNorm2d(num_T * 3)
        self.fuse = nn.Conv2d(num_T * 3, out, (1, 1))

    def forward(self, x):                       # x [B,1,C,T]
        outs = []
        for conv in self.branches:
            h = conv(x)                          # [B,num_T,C,T]
            outs.append(torch.log(F.avg_pool2d(h ** 2, (1, h.shape[-1])) + 1e-6))   # PowerLayer [B,num_T,C,1]
        h = F.elu(self.fuse(self.bn(torch.cat(outs, 1))))                            # [B,out,C,1]
        return h.squeeze(-1).transpose(1, 2)     # [B,C,out]


class Adjacency(nn.Module):
    """Per-sample learnable adjacency A(x): symmetric free base + data-dependent node-pair similarity.
    Returns A[B,C,C] (ReLU, symmetric) and pre-ReLU edge_logits[B,C,C] (the object for edge-level CMI)."""
    def __init__(self, n_chans, feat, emb=16):
        super().__init__()
        self.base = nn.Parameter(torch.empty(n_chans, n_chans)); nn.init.xavier_uniform_(self.base)
        self.We = nn.Linear(feat, emb)

    def forward(self, node_feat):                # [B,C,F]
        g = self.We(node_feat)                   # [B,C,emb]
        S = g @ g.transpose(1, 2)                # [B,C,C] per-sample similarity (the fingerprint)
        edge_logits = (self.base + self.base.t()).unsqueeze(0) + S
        A = F.relu(edge_logits)
        return 0.5 * (A + A.transpose(1, 2)), edge_logits


def _sgc_norm(A):                                # D^{-1/2}(A+I)D^{-1/2}, batched
    C = A.shape[-1]
    A = A + torch.eye(C, device=A.device).unsqueeze(0)
    dinv = A.sum(-1).clamp(min=1e-6).pow(-0.5)
    return dinv.unsqueeze(-1) * A * dinv.unsqueeze(1)


class _GlobalAdj(nn.Module):
    """Single SHARED learnable adjacency (DGCNN/RGNN style — one A for the whole dataset, not per-sample)."""
    def __init__(self, n_chans):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_chans, n_chans)); nn.init.xavier_uniform_(self.A)

    def forward(self):
        return _sgc_norm(F.relu(self.A + self.A.t()).unsqueeze(0)).squeeze(0)   # [C,C] sym, normalized


class _ChebConv(nn.Module):
    """Chebyshev graph conv, order K (DGCNN's conv), using the normalized adjacency as the operator."""
    def __init__(self, fin, fout, K=2):
        super().__init__()
        self.K = K; self.lin = nn.Linear(fin * K, fout)

    def forward(self, X, S):                       # X [B,C,fin], S [C,C]
        Ts = [X] + ([S @ X] if self.K > 1 else [])
        for _ in range(2, self.K):
            Ts.append(2 * (S @ Ts[-1]) - Ts[-2])
        return self.lin(torch.cat(Ts, -1))


class DGCNNBackbone(nn.Module):
    """DGCNN (Song 2018) baseline, raw-signal: temporal node encoder -> shared learnable adjacency ->
    ChebNet (K=2) -> flatten readout. Trained with CE (or hosted by our CMI). (logits, Z)."""
    def __init__(self, n_chans, n_times, n_classes, feat=16, hidden=16, K=2):
        super().__init__()
        self.enc = TemporalNodeEncoder(n_times, out=feat)
        self.adj = _GlobalAdj(n_chans)
        self.conv = _ChebConv(feat, hidden, K)
        self.proj = nn.Linear(n_chans * hidden, 64)
        self.head = nn.Linear(64, n_classes)
        self.z_dim = 64

    def forward(self, x):
        H = F.elu(self.conv(self.enc(x.unsqueeze(1)), self.adj()))   # [B,C,hidden]
        z = F.elu(self.proj(H.reshape(H.shape[0], -1)))              # flatten readout
        return self.head(z), z


class RGNNBackbone(nn.Module):
    """RGNN (Zhong 2020) baseline, raw-signal: temporal node encoder -> shared learnable adjacency ->
    SGC (L=2) -> sum readout. The key contrast with GraphCMINet is the SHARED (not per-sample) adjacency."""
    def __init__(self, n_chans, n_times, n_classes, feat=16, hidden=32, hops=2):
        super().__init__()
        self.enc = TemporalNodeEncoder(n_times, out=feat)
        self.adj = _GlobalAdj(n_chans)
        self.hops = hops
        self.W = nn.Linear(feat, hidden)
        self.head = nn.Linear(hidden, n_classes)
        self.z_dim = hidden

    def forward(self, x):
        H = self.enc(x.unsqueeze(1))
        S = self.adj()
        for _ in range(self.hops):
            H = S @ H
        z = self.W(H).sum(1)                                          # sum readout
        return self.head(z), z


class GraphCMINet(nn.Module):
    def __init__(self, n_chans, n_times, n_classes, feat=16, hidden=32, hops=2):
        super().__init__()
        self.enc = TemporalNodeEncoder(n_times, out=feat)
        self.adj = Adjacency(n_chans, feat)
        self.hops = hops
        self.W = nn.Linear(feat, hidden)         # SGC weight
        self.head = nn.Linear(hidden, n_classes)
        self.z_dim = hidden

    def forward_graph(self, x):
        X0 = self.enc(x.unsqueeze(1))            # [B,C,F]
        A, edge_logits = self.adj(X0)
        S = _sgc_norm(A)
        H = X0
        for _ in range(self.hops):
            H = S @ H                            # SGC propagation S^L X
        node_Z = self.W(H)                       # [B,C,hidden]
        graph_Z = node_Z.mean(1)                 # mean readout [B,hidden]
        return self.head(graph_Z), graph_Z, node_Z, edge_logits

    def forward(self, x):
        logits, graph_Z, _, _ = self.forward_graph(x)
        return logits, graph_Z
