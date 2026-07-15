"""Standard DG framework baselines as batch penalties on (Z, logits, Y, D).

Frameworks (DomainBed-style), all operating on the same backbone (logits, Z) interface:
  coral    : align mean+covariance of Z across source domains (Deep CORAL)
  label_coral : standalone class-CONDITIONAL CORAL — align mean+cov WITHIN each class across source
             domains (a.k.a. conditional-CORAL / C-CORAL). Distinct from `scldgn`'s composite per-class
             CORAL: no supervised-contrastive term, explicit (Y,D) support handling, declared weighting.
  mmd      : Gaussian-kernel MMD of Z across source domains
  irm      : invariant risk minimization gradient penalty (dummy-scale grad)
  vrex     : variance of per-domain risks (V-REx)
  groupdro : worst-domain reweighting (handled statefully in the trainer)
  dann     : adversarial domain confusion via gradient-reversal on a discriminator q(D|Z)
  cdann    : conditional DANN — discriminator q(D|Z,Y), gradient-reversed
  scldgn   : Supervised-Contrastive DG (TBME2024) essence — class-conditional cross-domain supervised
             contrastive + per-class CORAL alignment (non-adversarial class-conditional invariance)

Reimplemented MIT-clean; math grounded on DomainBed (facebookresearch/DomainBed).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

PENALTY_METHODS = {"coral", "label_coral", "mmd", "irm", "vrex", "chsic", "scldgn"}
ADV_METHODS = {"dann", "cdann", "cdan"}   # cdan = CDAN multilinear conditioning (R2, additive)
# moment-matching penalties applied to the SAME representation objects encoder-CMI controls (graph
# readout + channel-mean node feature) when the backbone exposes forward_graph; else flat z.
GRAPH_PEN_METHODS = {"coral", "label_coral"}
DG_METHODS = PENALTY_METHODS | ADV_METHODS | {"groupdro"}


# ---- gradient reversal (DANN/CDANN) ----
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return _GradReverse.apply(x, alpha)


def _groups(z, d, n_dom, min_n=2):
    gs = [z[d == i] for i in range(n_dom)]
    return [g for g in gs if g.size(0) >= min_n]


def _cov(x):
    xm = x - x.mean(0, keepdim=True)
    return (xm.t() @ xm) / (x.size(0) - 1)


def coral(z, d, n_dom):
    gs = _groups(z, d, n_dom)
    pens = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            md = (gs[i].mean(0) - gs[j].mean(0)).pow(2).mean()
            cd = (_cov(gs[i]) - _cov(gs[j])).pow(2).mean()
            pens.append(md + cd)
    return torch.stack(pens).mean() if pens else z.new_zeros(())


def _coral_pair(a, b):
    """CORAL discrepancy between two feature groups: squared mean gap + squared covariance gap."""
    md = (a.mean(0) - b.mean(0)).pow(2).mean()
    cd = (_cov(a) - _cov(b)).pow(2).mean()
    return md + cd


def label_coral(z, y, d, n_cls, n_dom, min_n=4, return_support=False):
    """Standalone class-CONDITIONAL CORAL (conditional-CORAL / C-CORAL).

    For each class c that has adequate support (>= min_n samples) in AT LEAST TWO source domains, align the
    per-domain mean+covariance of z WITHIN that class across those domains, then aggregate. Declared
    weighting rule: equal weight per QUALIFYING class (mean over classes), and within a class the mean over
    unordered qualifying domain-pairs. Class/domain cells with < min_n samples are SKIPPED EXPLICITLY (they
    are recorded in the returned support diagnostics), never silently treated as zero evidence.

    Key contrast vs marginal CORAL: because alignment is done WITHIN each class, a pure label-prior
    (class-proportion) shift across domains — with matched within-class moments — yields ~0 penalty. A
    within-class moment gap yields a positive penalty. (n_cls kept for API symmetry; classes present in y
    are what actually drive the sum.)"""
    per_class_pens, support, skipped = [], {}, []
    classes = range(n_cls) if n_cls else [int(c) for c in torch.unique(y)]
    for c in classes:
        mc = (y == c)
        doms = []
        for i in range(n_dom):
            n = int((mc & (d == i)).sum())
            if n >= min_n:
                doms.append(i)
            elif n > 0:
                skipped.append((int(c), int(i), n))     # under-supported (Y,D) cell: recorded, not dropped-silently
        support[int(c)] = [int(x) for x in doms]
        if len(doms) < 2:
            continue
        gs = [z[mc & (d == i)] for i in doms]
        pens = [_coral_pair(gs[a], gs[b]) for a in range(len(gs)) for b in range(a + 1, len(gs))]
        per_class_pens.append(torch.stack(pens).mean())
    pen = torch.stack(per_class_pens).mean() if per_class_pens else z.new_zeros(())
    if return_support:
        diag = dict(min_n=int(min_n),
                    qualifying_classes=[c for c in support if len(support[c]) >= 2],
                    per_class_domains=support,
                    skipped_cells=skipped,
                    n_qualifying_classes=len(per_class_pens),
                    n_skipped_cells=len(skipped))
        return pen, diag
    return pen


def graph_moment_penalty(kind, gz, nz, y, d, n_cls, n_dom, lam_graph, lam_node):
    """Apply a moment-matching DG penalty (kind in GRAPH_PEN_METHODS) to the SAME representation objects
    encoder-CMI controls: the graph readout `gz` (weight lam_graph) and the channel-MEAN node feature
    nz.mean(1) (weight lam_node), matching the node-CMI term's (1/C) Σ_v structure. Returns (penalty tensor,
    support diag). Graph-only ablation = lam_node 0."""
    if kind == "coral":
        pg = coral(gz, d, n_dom)
        pn = coral(nz.mean(1), d, n_dom) if lam_node else gz.new_zeros(())
        sd = {}
    elif kind == "label_coral":
        pg, sd_g = label_coral(gz, y, d, n_cls, n_dom, return_support=True)
        if lam_node:
            pn, sd_n = label_coral(nz.mean(1), y, d, n_cls, n_dom, return_support=True)
        else:
            pn, sd_n = gz.new_zeros(()), {}
        sd = {"graph": sd_g, "node": sd_n}
    else:
        raise ValueError(f"graph_moment_penalty: unsupported kind '{kind}'")
    pen = lam_graph * pg + lam_node * pn
    sd["graph_pen"] = float(pg.detach()); sd["node_pen"] = float(pn.detach())
    return pen, sd


def scldgn(z, y, d, n_cls, n_dom, temperature=0.1):
    """SCLDGN (Supervised Contrastive Learning DG Network, TBME2024) — re-implementation of its essence:
    CLASS-CONDITIONAL domain invariance WITHOUT an adversary, via (1) supervised contrastive with
    same-class / different-domain positives (pull a class's features together across domains, push other
    classes away → domain-invariant class clusters) + (2) per-class CORAL (align each class's mean+cov
    across domains). The non-adversarial class-conditional rival to lpc_prior's posterior-KL term."""
    from cmi.methods.contrastive import sup_con_loss          # lazy: avoid import cycle
    scl = sup_con_loss(z, y, d, temperature=temperature, cross_domain=True)
    cc, k = z.new_zeros(()), 0
    for c in range(n_cls):
        m = (y == c)
        if int(m.sum()) < 4:
            continue
        cc = cc + coral(z[m], d[m], n_dom); k += 1
    return scl + (cc / k if k else z.new_zeros(()))


def _mmd_kernel(a, b):
    d2 = torch.cdist(a, b).pow(2)
    # multi-bandwidth RBF (DomainBed-style fixed gamma set)
    return sum(torch.exp(-g * d2) for g in (0.1, 0.5, 1.0, 2.0, 5.0)).mean() / 5.0


def mmd(z, d, n_dom):
    gs = _groups(z, d, n_dom)
    pens = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            a, b = gs[i], gs[j]
            pens.append(_mmd_kernel(a, a) + _mmd_kernel(b, b) - 2 * _mmd_kernel(a, b))
    return torch.stack(pens).mean() if pens else z.new_zeros(())


def irm(logits, y, d, n_dom):
    """Per-domain IRMv1 penalty: squared grad of CE w.r.t. a dummy scale (=1)."""
    pens = []
    for i in range(n_dom):
        m = d == i
        if m.sum() < 2:
            continue
        scale = torch.ones((), device=logits.device, requires_grad=True)
        ce = F.cross_entropy(logits[m] * scale, y[m])
        g = torch.autograd.grad(ce, [scale], create_graph=True)[0]
        pens.append(g.pow(2))
    return torch.stack(pens).mean() if pens else logits.new_zeros(())


def vrex(logits, y, d, n_dom):
    risks = [F.cross_entropy(logits[d == i], y[d == i]) for i in range(n_dom) if (d == i).sum() > 0]
    if len(risks) < 2:
        return logits.new_zeros(())
    return torch.stack(risks).var()


def _rbf(Z):
    d2 = torch.cdist(Z, Z).pow(2)
    sig2 = d2.detach()[d2.detach() > 0].median() + 1e-8     # median heuristic bandwidth
    return torch.exp(-d2 / (2 * sig2))


def _hsic(Z, Doh):
    """Biased HSIC between Z (RBF kernel) and one-hot domain Doh (linear kernel)."""
    n = Z.size(0)
    K = _rbf(Z)
    L = Doh @ Doh.t()
    H = torch.eye(n, device=Z.device) - 1.0 / n
    return (K @ H * (L @ H).t()).sum() / (n - 1) ** 2


def chsic(z, y, d, n_cls, n_dom):
    """Class-stratified conditional HSIC: sum_y p(y) HSIC(Z, D | Y=y). A kernel-based
    competitor to LPC-CMI for Z _||_ D | Y (the discrete-Y reduction of CIRCE/HSCIC).
    Parameter-free, differentiable, no auxiliary network."""
    tot = z.new_zeros(())
    for c in range(n_cls):
        m = y == c
        if m.sum() < 4 or len(torch.unique(d[m])) < 2:
            continue
        tot = tot + m.float().mean() * _hsic(z[m], F.one_hot(d[m], n_dom).float())
    return tot


def make_discriminator(z_dim, n_dom, conditional, n_cls, hidden=128):
    din = z_dim + (n_cls if conditional else 0)
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, n_dom))


def adv_penalty(disc, z, y, d, n_cls, conditional, alpha=1.0):
    """DANN/CDANN: discriminator predicts D from gradient-reversed Z (and Y if conditional).
    The discriminator minimizes domain CE; the GRL flips the gradient into the encoder,
    so the encoder is pushed to make Z (conditionally) uninformative of D."""
    feat = z if not conditional else torch.cat([z, F.one_hot(y, n_cls).float()], 1)
    return F.cross_entropy(disc(grad_reverse(feat, alpha)), d)


def make_cdan_discriminator(z_dim, n_dom, n_cls, hidden=128):
    """CDAN (Long et al. 2018): discriminator over the multilinear map z (outer) yhat -> dim z_dim*n_cls."""
    return nn.Sequential(nn.Linear(z_dim * n_cls, hidden), nn.ReLU(), nn.Linear(hidden, n_dom))


def cdan_penalty(disc, z, logits, d, alpha=1.0):
    """CDAN: multilinear conditional adversarial. The discriminator predicts D from the outer product
    z (outer) softmax(yhat), gradient-reversed into the encoder. Conditions the domain alignment on the
    classifier's PREDICTION (detached) rather than the true label (marginal DANN) or a concatenated one-hot
    (CDANN) — captures cross-covariance between features and class predictions."""
    yhat = F.softmax(logits.detach(), dim=1)                 # condition on prediction (standard CDAN, detached)
    feat = (z.unsqueeze(2) * yhat.unsqueeze(1)).flatten(1)   # [B, z_dim * n_cls] multilinear map
    return F.cross_entropy(disc(grad_reverse(feat, alpha)), d)
