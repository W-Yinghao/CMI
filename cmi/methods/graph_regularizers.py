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
    """Shared per-node q(D | Z_v, Y). reg() = mean_v Ê KL(q ‖ π_y) = the Σ_v I(Z_v;D|Y) estimate."""
    def __init__(self, d, n_dom, n_cls, priors):
        super().__init__()
        self.body = _mlp(d + n_cls, n_dom)
        self.n_cls = n_cls
        self.register_buffer("log_pi", torch.log(torch.as_tensor(priors[0], dtype=torch.float32) + 1e-8))  # [n_cls,n_dom]

    def _logits(self, node_Z, y):                 # node_Z [B,C,d] -> [B,C,n_dom]
        B, C, _ = node_Z.shape
        y_oh = F.one_hot(y, self.n_cls).float().unsqueeze(1).expand(B, C, self.n_cls)
        return self.body(torch.cat([node_Z, y_oh], -1))

    def step_a_loss(self, node_Z, y, d):          # fit q (detached Z): per-node domain CE
        lg = self._logits(node_Z, y)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                               d.unsqueeze(1).expand(-1, lg.shape[1]).reshape(-1))

    def reg(self, node_Z, y):                      # penalty (grad to encoder)
        lg = self._logits(node_Z, y)
        return _kl_to_prior(lg, self.log_pi[y].unsqueeze(1)).mean()

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

    def _logits(self, edge_logits, y):             # edge_logits [B,C,C] -> [B,n_dom]
        a = self.compress(edge_logits[:, self.iu0, self.iu1])     # upper-triangle -> e_a
        return self.body(torch.cat([a, F.one_hot(y, self.n_cls).float()], -1))

    def step_a_loss(self, edge_logits, y, d):
        return F.cross_entropy(self._logits(edge_logits, y), d)

    def reg(self, edge_logits, y):
        return _kl_to_prior(self._logits(edge_logits, y), self.log_pi[y]).mean()
