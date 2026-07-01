"""Node- and edge-level conditional-leakage heads for GraphCMINet (Stage 2 of the GNN design).
Same variational posterior-KL machinery as the global I(Z;D|Y), now per-channel and on the learned adjacency:
  L = λ·I(graph_Z;D|Y) [existing] + λ_node·Σ_v I(Z_v;D|Y) + λ_edge·I(A;D|Y).
NodePosterior: ONE weight-shared trunk applied to every node (PGExplainer-style amortization) -> a per-channel
"which electrodes leak subject identity" map (the diagnostic NodeDAT cannot produce, since it is conditional on Y
and non-adversarial). EdgePosterior: q(D | summary(edge_logits), Y) -> penalizes subject-info in the learned graph.
Both reuse the label-prior π_y(D)=p(D|Y) (so they inherit the imbalance correction)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_in, d_out, hidden=64):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


def _kl_to_prior(logits, log_pi):                 # KL(q(.|.) || π) per row; logits [...,n_dom], log_pi [...,n_dom]
    logq = F.log_softmax(logits, -1)
    return (logq.exp() * (logq - log_pi)).sum(-1)


class NodePosterior(nn.Module):
    """Shared per-node q(D | Z_v, e_v, Y). reg() = mean_v Ê KL(q ‖ π_y) = the Σ_v I(Z_v;D|Y) estimate.

    When ``n_chans`` is given, the head is conditioned on a learned per-node embedding ``e_v``
    (``nn.Embedding(n_chans, node_emb_dim)``), matching the manuscript's ``q(D | Z_v, v, Y)`` — the shared
    trunk can then distinguish electrodes even when their node features coincide. With ``n_chans=None`` the
    head falls back to the earlier node-id-agnostic ``q(D | Z_v, Y)`` (byte-identical to the prior version)."""
    def __init__(self, d, n_dom, n_cls, priors, n_chans=None, node_emb_dim=8):
        super().__init__()
        self.n_cls = n_cls
        self.use_node_id = n_chans is not None
        emb_dim = node_emb_dim if self.use_node_id else 0
        if self.use_node_id:
            self.node_emb = nn.Embedding(int(n_chans), int(node_emb_dim))   # e_v
            self.register_buffer("node_ids", torch.arange(int(n_chans)))
        self.body = _mlp(d + emb_dim + n_cls, n_dom)
        self.register_buffer("log_pi", torch.log(torch.as_tensor(priors[0], dtype=torch.float32) + 1e-8))  # [n_cls,n_dom]
        # GLS reference domain marginal p~(D)=p(D) (Route-B / dualpc semantics); used by reference="marginal".
        p_d = torch.as_tensor(priors[1], dtype=torch.float32)
        p_d = p_d / p_d.sum().clamp(min=1e-12)
        self.register_buffer("log_pd_ref", torch.log(p_d + 1e-8))                                          # [n_dom]

    def _logits(self, node_Z, y):                 # node_Z [B,C,d] -> [B,C,n_dom]
        B, C, _ = node_Z.shape
        y_oh = F.one_hot(y, self.n_cls).float().unsqueeze(1).expand(B, C, self.n_cls)
        if self.use_node_id:
            e_v = self.node_emb(self.node_ids[:C]).unsqueeze(0).expand(B, C, -1)   # [B,C,node_emb_dim]
            return self.body(torch.cat([node_Z, e_v, y_oh], -1))
        return self.body(torch.cat([node_Z, y_oh], -1))

    def step_a_loss(self, node_Z, y, d, weight=None):   # fit q (detached Z): per-node domain CE
        lg = self._logits(node_Z, y)                    # [B,C,n_dom]
        B, C, n_dom = lg.shape
        tgt = d.unsqueeze(1).expand(-1, C).reshape(-1)
        if weight is None:
            return F.cross_entropy(lg.reshape(-1, n_dom), tgt)
        per = F.cross_entropy(lg.reshape(-1, n_dom), tgt, reduction="none").reshape(B, C)   # Route-B GLS
        return (weight[:, None] * per).sum() / (weight.sum() * C).clamp(min=1e-8)

    def reg(self, node_Z, y, weight=None, reference="prior"):   # penalty (grad to encoder)
        lg = self._logits(node_Z, y)                            # [B,C,n_dom]
        log_ref = self.log_pd_ref.view(1, 1, -1) if reference == "marginal" else self.log_pi[y].unsqueeze(1)
        kl = _kl_to_prior(lg, log_ref)                          # [B,C]
        if weight is None:
            return kl.mean()
        return (weight[:, None] * kl).sum() / (weight.sum() * kl.shape[1]).clamp(min=1e-8)

    @torch.no_grad()
    def leakage_map(self, node_Z, y):              # length-C per-channel residual KL (diagnostic figure)
        return _kl_to_prior(self._logits(node_Z, y), self.log_pi[y].unsqueeze(1)).mean(0)


class EdgePosterior(nn.Module):
    """q(D | summary(adjacency), Y). reg() = I(A;D|Y): the learned adjacency is a subject fingerprint."""
    def __init__(self, n_chans, n_dom, n_cls, priors, e_a=64):
        super().__init__()
        iu = torch.triu_indices(n_chans, n_chans, 1)
        self.register_buffer("iu0", iu[0]); self.register_buffer("iu1", iu[1])
        self.compress = nn.Linear(iu.shape[1], e_a)
        self.body = _mlp(e_a + n_cls, n_dom)
        self.n_cls = n_cls
        self.register_buffer("log_pi", torch.log(torch.as_tensor(priors[0], dtype=torch.float32) + 1e-8))
        p_d = torch.as_tensor(priors[1], dtype=torch.float32)
        p_d = p_d / p_d.sum().clamp(min=1e-12)
        self.register_buffer("log_pd_ref", torch.log(p_d + 1e-8))   # GLS reference domain marginal p~(D)

    def _logits(self, edge_logits, y):             # edge_logits [B,C,C] -> [B,n_dom]
        a = self.compress(edge_logits[:, self.iu0, self.iu1])     # upper-triangle -> e_a
        return self.body(torch.cat([a, F.one_hot(y, self.n_cls).float()], -1))

    def step_a_loss(self, edge_logits, y, d, weight=None):
        lg = self._logits(edge_logits, y)          # [B,n_dom]
        if weight is None:
            return F.cross_entropy(lg, d)
        per = F.cross_entropy(lg, d, reduction="none")
        return (weight * per).sum() / weight.sum().clamp(min=1e-8)

    def reg(self, edge_logits, y, weight=None, reference="prior"):
        lg = self._logits(edge_logits, y)          # [B,n_dom]
        log_ref = self.log_pd_ref.view(1, -1) if reference == "marginal" else self.log_pi[y]
        kl = _kl_to_prior(lg, log_ref)             # [B]
        if weight is None:
            return kl.mean()
        return (weight * kl).sum() / weight.sum().clamp(min=1e-8)
