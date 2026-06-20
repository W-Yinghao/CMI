"""EEG augmentations + self-supervised contrastive losses (SimCLR, BYOL) — a SELF-SUPERVISED contrastive
FRAMEWORK, distinct from supervised SupCon. Two augmented views per trial; SimCLR pulls the matched views
together (NT-Xent, with negatives); BYOL predicts one view's projection from the other (no negatives, EMA
target). Used as an auxiliary loss alongside CE: loss = CE + gamma * ssl. The DG hypothesis is that
augmentation-invariant representations transfer better across subjects."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def augment(x, noise=0.1, scale=0.2, tmask=0.2, chdrop=0.1, shift=0.1):
    """Stochastic EEG augmentation on [B,C,T]: amplitude scale, Gaussian noise, time-mask, channel-dropout,
    circular time-shift. Each call samples a fresh composition -> two calls give two views."""
    B, C, T = x.shape
    x = x * (1 + scale * (2 * torch.rand(B, 1, 1, device=x.device) - 1))      # per-trial amplitude scale
    x = x + noise * x.std() * torch.randn_like(x)                              # additive noise
    if tmask > 0:                                                             # time mask
        L = int(tmask * T); s = int(torch.randint(0, max(1, T - L), (1,)).item()); x[:, :, s:s + L] = 0
    if chdrop > 0:                                                            # channel dropout
        x = x * (torch.rand(B, C, 1, device=x.device) > chdrop).float()
    if shift > 0:                                                            # circular time shift
        x = torch.roll(x, int(shift * T * (2 * torch.rand(1).item() - 1)), dims=-1)
    return x


def two_views(x):
    return augment(x), augment(x)


def simclr_loss(z1, z2, temperature=0.5):
    """NT-Xent over a batch of 2B normalized embeddings; positives are the matched views (i, i+B)."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = z @ z.t() / temperature
    sim.masked_fill_(torch.eye(2 * B, device=z.device, dtype=torch.bool), -1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def byol_loss(p1, z2, p2, z1):
    """Symmetric BYOL: predictor outputs p1/p2, target projections z1/z2 (detached). 2 - 2*cos."""
    def d(p, z):
        return (2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1)).mean()
    return 0.5 * (d(p1, z2) + d(p2, z1))


class MLPHead(torch.nn.Module):
    """Projector / predictor MLP for SimCLR/BYOL (on the backbone's z)."""
    def __init__(self, dim, hidden=256, out=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.BatchNorm1d(hidden),
                                       torch.nn.ReLU(), torch.nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


SSL_METHODS = {"simclr", "byol"}                 # pure self-supervised contrastive baselines
LPC_SSL_METHODS = {"lpc_simclr", "lpc_byol"}     # SSL framework HOSTING our CMI term (CE + γ·SSL + λ·I(Z;D|Y))
ALL_SSL = SSL_METHODS | LPC_SSL_METHODS
