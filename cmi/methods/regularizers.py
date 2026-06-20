"""LPC-CMI regularizer + ablation variants (the framework's core), productionized
from synthetic/sanity_check.py.

The trainable criterion is the conditional domain leakage I(Z;D|Y), estimated as a
posterior-KL PLUG-IN surrogate:  L_CMI = E_i KL( q_psi(D|z_i,y_i) || pi_{y_i}(D) ).
NOTE (not an upper bound): with the variational posterior q_psi this EQUALS I(Z;D|Y)
only when q_psi = p(D|z,y) (Step-A convergence); for a sub-optimal critic it generally
UNDER-estimates the true CMI (it is NOT >= I for arbitrary q_psi -- e.g. q_psi = pi_y(D)
gives L_CMI = 0 while I(Z;D|Y) can be > 0). So: a consistent plug-in estimator, tight at
convergence, not a Barber-Agakov-style bound. Report critic capacity / convergence gap.
Variants share the same posterior machinery and only change the conditioning / prior:

  lpc_prior   E KL(q(D|Z,Y) || pi_y(D))   <- ours (I(Z;D|Y), label-prior corrected)
  lpc_uniform E KL(q(D|Z,Y) || Uniform)   CDANN target; mis-specified under imbalance
  marginal    E KL(q(D|Z)   || p(D))      ~ I(Z;D)        (label erasure)
  chain       E KL(q(S|Z)   || p(S))      ~ I(Z;(D,Y))    (Y erasure), S=(D,Y)
  erm         0
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ["erm", "marginal", "chain", "lpc_uniform", "lpc_prior"]


def _mlp(din, dout, hidden=128):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, dout))


def kl_to_prior(logits, log_prior, weight=None):
    """E_x KL(softmax(logits) || prior); log_prior broadcast or per-sample [B,K].
    If `weight` ([B] float) is given, return the per-sample-weighted mean
    sum_i w_i KL_i / sum_i w_i (Route B reweighted encoder CMI); else the plain mean."""
    logq = F.log_softmax(logits, dim=1)
    kl = (logq.exp() * (logq - log_prior)).sum(1)            # [B]
    if weight is None:
        return kl.mean()
    return (weight * kl).sum() / weight.sum().clamp(min=1e-8)


def empirical_priors(y, d, n_dom, n_cls, alpha=1.0):
    """pi_y(d)=p(d|y) (Laplace-smoothed), marginal p(d), joint p(d,y) over S=(d,y)."""
    counts = np.zeros((n_cls, n_dom))
    for yi, di in zip(y, d):
        counts[yi, di] += 1
    pi_y = (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * n_dom)
    p_d = (counts.sum(0) + alpha) / (counts.sum() + alpha * n_dom)
    p_dy = (counts + alpha) / (counts.sum() + alpha * n_cls * n_dom)
    return pi_y, p_d, p_dy


def subject_priors(y, d, n_dom, n_cls, alpha=1.0):
    """SUBJECT-balanced pi_y(d|y): each subject contributes EQUALLY (presence), not trial-weighted —
    the LOSO domain unit is the subject, so trial-count imbalance should not skew the prior.
    (Addresses the trial-vs-subject prior critique; reduces to empirical_priors when trials are balanced.)"""
    present = np.zeros((n_cls, n_dom))
    for yi, di in zip(y, d):
        present[yi, di] = 1.0
    pi_y = (present + alpha) / (present.sum(1, keepdims=True) + alpha * n_dom)
    p_d = (present.sum(0) + alpha) / (present.sum() + alpha * n_dom)
    p_dy = (present + alpha) / (present.sum() + alpha * n_cls * n_dom)
    return pi_y, p_d, p_dy


def effective_priors(y, d, n_dom, n_cls, alpha=1.0):
    """Priors CONSISTENT with a (class,domain)-balanced sampler: within each class the sampler
    makes domains ~uniform, so the matching KL target pi_y(d|y) is uniform over the domains
    PRESENT in class y (not the raw empirical p(d|y)). Used by the 'eff' lpc_prior variant."""
    counts = np.zeros((n_cls, n_dom))
    for yi, di in zip(y, d):
        counts[yi, di] += 1
    present = (counts > 0).astype(float)
    pi_y = (present + alpha) / (present.sum(1, keepdims=True) + alpha * n_dom)   # uniform over present
    p_d = (present.sum(0) + alpha) / (present.sum() + alpha * n_dom)
    p_dy = (present + alpha) / (present.sum() + alpha * n_cls * n_dom)
    return pi_y, p_d, p_dy


class DomainPosteriors(nn.Module):
    """Holds q(D|Z,Y), q(D|Z), q(S|Z) and the precomputed (log) priors."""
    def __init__(self, z_dim, n_dom, n_cls, priors, device="cpu"):
        super().__init__()
        self.n_dom, self.n_cls = n_dom, n_cls
        self.q_dzy = _mlp(z_dim + n_cls, n_dom)
        self.q_dz = _mlp(z_dim, n_dom)
        self.q_sz = _mlp(z_dim, n_dom * n_cls)
        self.h_ydz = _mlp(z_dim + n_dom, n_cls)   # IIB auxiliary predictor h(Y|Z,D) [FULL domain decoder]
        self.q_yz = _mlp(z_dim, n_cls)            # separate Y|Z probe for the decoder CMI (NOT the task head)
        # Route C: INTERCEPT-ONLY domain decoder h0(Y|Z,D)=u(Z)+b_D -> D shifts only a per-domain logit
        # bias (prior/calibration). Residual CE(h0)-CE(h) isolates domain-dependent DECISION-BOUNDARY
        # change (genuine concept shift), robust to D=subject degeneracy + label-prior/calibration shift.
        self.u_yz = _mlp(z_dim, n_cls)
        self.b_d = nn.Parameter(torch.zeros(n_dom, n_cls))
        pi_y, p_d, p_dy = priors
        # GLS reference domain marginal (P0-5 fix). The GLS weight w(d,y)=pi*(y)/p(y|d) reweights to
        # p~(d,y) ∝ p(d) pi*(y), hence p~(d) = p(d) EXACTLY — the raw domain marginal, NOT sum_y pi*(y) p(d|y)
        # (the old `pi_y.mean(0)`, which equals p(d) only when the source is class-balanced). Use p_d so the
        # training reference, the GLS-induced measure, and the evaluation reference share one probability measure.
        p_d_ref = np.asarray(p_d, dtype=float)
        p_d_ref = p_d_ref / np.maximum(p_d_ref.sum(), 1e-12)
        self.register_buffer("log_pi_y", torch.log(torch.tensor(pi_y, dtype=torch.float32)))
        self.register_buffer("log_pd", torch.log(torch.tensor(p_d, dtype=torch.float32)))
        self.register_buffer("log_pd_ref", torch.log(torch.tensor(p_d_ref, dtype=torch.float32)))
        self.register_buffer("log_unif", torch.log(torch.full((n_dom,), 1.0 / n_dom)))
        self.register_buffer("log_pS", torch.log(torch.tensor(p_dy.reshape(-1), dtype=torch.float32)))
        self.to(device)

    def posterior_loss(self, z_detached, y, d, weight=None):
        """Step-A cross-entropy that fits the posteriors to the current Z.
        If `weight` ([B] float) is given, fit them to the REWEIGHTED measure (Route B): each
        sample's CE is scaled by w_i=pi*(y_i)/pi_{d_i}(y_i) so q(D|Z,Y) is the variational
        posterior under I~(Y;D)=0, making the Step-B KL a valid bound on the reweighted CMI.
        Reduces EXACTLY to the plain sum of cross-entropies when weight is all-ones."""
        y_oh = F.one_hot(y, self.n_cls).float()
        s = d * self.n_cls + y
        h0 = self.u_yz(z_detached) + self.b_d[d]            # intercept-only decoder h0(Y|Z,D) (Route C)
        if weight is None:
            return (F.cross_entropy(self.q_dzy(torch.cat([z_detached, y_oh], 1)), d)
                    + F.cross_entropy(self.q_dz(z_detached), d)
                    + F.cross_entropy(self.q_sz(z_detached), s)
                    + F.cross_entropy(self.q_yz(z_detached), y)      # separate domain-blind Y|Z probe
                    + F.cross_entropy(h0, y))                        # intercept-only domain decoder
        wmean = lambda per: (weight * per).sum() / weight.sum().clamp(min=1e-8)
        return (wmean(F.cross_entropy(self.q_dzy(torch.cat([z_detached, y_oh], 1)), d, reduction="none"))
                + wmean(F.cross_entropy(self.q_dz(z_detached), d, reduction="none"))
                + wmean(F.cross_entropy(self.q_sz(z_detached), s, reduction="none"))
                + wmean(F.cross_entropy(self.q_yz(z_detached), y, reduction="none"))
                + wmean(F.cross_entropy(h0, y, reduction="none")))

    def iib_ce_h(self, z, y, d, weight=None):
        """CE of the domain-conditioned predictor h(Y|Z,D). Used both to fit h (Step A,
        detached z) and to form IIB's I(Y;D|Z)=H(Y|Z)-H(Y|D,Z)=CE_q-CE_h (Step B, grad z).
        If `weight` ([B] float) is given, return the per-sample-weighted mean (Route B:
        the reweighted H(Y|Z,D) half of the decoder CMI); reduces EXACTLY to the plain CE
        when weight is all-ones."""
        d_oh = F.one_hot(d, self.n_dom).float()
        logits = self.h_ydz(torch.cat([z, d_oh], 1))
        if weight is None:
            return F.cross_entropy(logits, y)
        per = F.cross_entropy(logits, y, reduction="none")
        return (weight * per).sum() / weight.sum().clamp(min=1e-8)

    def dec_cmi(self, z, y, d, weight=None):
        """Decoder CMI I(Y;D|Z)=H(Y|Z)-H(Y|Z,D) via the SEPARATE frozen probe q_yz (NOT the task head),
        so the penalty does not silently rescale the task CE (report §8.1). Grad flows to z; both probes
        (q_yz, h_ydz) are fit on detached z in Step A via posterior_loss + iib_ce_h."""
        d_oh = F.one_hot(d, self.n_dom).float()
        lq, lh = self.q_yz(z), self.h_ydz(torch.cat([z, d_oh], 1))
        if weight is None:
            return F.cross_entropy(lq, y) - F.cross_entropy(lh, y)
        wmean = lambda per: (weight * per).sum() / weight.sum().clamp(min=1e-8)
        return wmean(F.cross_entropy(lq, y, reduction="none")) - wmean(F.cross_entropy(lh, y, reduction="none"))

    def dec_cmi_residual(self, z, y, d, weight=None):
        """Route C RESIDUAL decoder CMI = CE(h0) - CE(h_full), where h0(Y|Z,D)=u(Z)+b_D is INTERCEPT-ONLY
        (D may shift only a per-domain bias = prior/calibration) and h is the FULL domain decoder. Measures
        ONLY the domain-dependent decision-BOUNDARY change = genuine concept shift. Unlike raw CE(a)-CE(h),
        this is ~0 under the D=subject degeneracy (b_D absorbs subject->label) and under label-prior shift."""
        d_oh = F.one_hot(d, self.n_dom).float()
        l0 = self.u_yz(z) + self.b_d[d]
        lh = self.h_ydz(torch.cat([z, d_oh], 1))
        if weight is None:
            return F.cross_entropy(l0, y) - F.cross_entropy(lh, y)
        wmean = lambda per: (weight * per).sum() / weight.sum().clamp(min=1e-8)
        return wmean(F.cross_entropy(l0, y, reduction="none")) - wmean(F.cross_entropy(lh, y, reduction="none"))

    def dec_js_residual(self, z, d, weight=None):
        """Prediction-distribution residual for DualPC's P(Y|Z) side.

        The CE residual above is the diagnostic CMI estimate. As a training loss it can be noisy because it
        optimizes a difference of two label CEs. For the AAAI DualPC objective, directly match the full
        domain decoder h(Y|Z,D) to the intercept-only decoder h0(Y|Z,D)=u(Z)+b_D using Jensen-Shannon
        divergence. This is nonnegative, label-prior/calibration tolerant, and targets the predictor
        distribution P(Y|Z) more directly while keeping the same full-vs-intercept concept-shift contrast.
        """
        d_oh = F.one_hot(d, self.n_dom).float()
        l0 = self.u_yz(z) + self.b_d[d]
        lh = self.h_ydz(torch.cat([z, d_oh], 1))
        p0 = F.softmax(l0, 1).clamp_min(1e-8)
        ph = F.softmax(lh, 1).clamp_min(1e-8)
        m = (0.5 * (p0 + ph)).clamp_min(1e-8)
        js = 0.5 * (p0 * (p0.log() - m.log())).sum(1) + 0.5 * (ph * (ph.log() - m.log())).sum(1)
        if weight is None:
            return js.mean()
        return (weight * js).sum() / weight.sum().clamp(min=1e-8)

    def reg(self, method, z, y, weight=None, reference="prior"):
        """Step-B regularizer as a function of the (grad-carrying) Z.

        Route B (reweighted-dual) extras for the lpc_prior encoder CMI E[KL(q(D|Z,Y)||.)]:
          weight    : per-sample importance weight w_i=pi*(y_i)/pi_{d_i}(y_i) -> WEIGHTED batch
                      mean (sum w_i KL_i / sum w_i). None => plain mean (naive 'dual', unchanged).
          reference : 'prior' uses the empirical pi_y(D) target (default, naive dual);
                      'marginal'/'ref_marginal' use the post-GLS domain marginal
                      sum_y pi*(y) p(D|Y=y); 'uniform' is kept as an ablation target."""
        if method == "erm":
            return z.new_zeros(())
        if method == "marginal":
            log_ref = {"marginal": self.log_pd, "raw_marginal": self.log_pd,
                       "ref_marginal": self.log_pd_ref}.get(reference, self.log_pd)
            return kl_to_prior(self.q_dz(z), log_ref, weight=weight)
        if method == "chain":
            return kl_to_prior(self.q_sz(z), self.log_pS)
        y_oh = F.one_hot(y, self.n_cls).float()
        logits = self.q_dzy(torch.cat([z, y_oh], 1))
        if method == "lpc_uniform":
            return kl_to_prior(logits, self.log_unif, weight=weight)
        if method == "lpc_prior":
            # Post-GLS the correct reference is the reweighted domain marginal
            # p~(D)=sum_y pi*(y)p(D|Y=y) (=pi~(D|Y), since D⊥Y after reweighting),
            # not uniform unless domains are equal-sized.
            log_ref = {"uniform": self.log_unif, "marginal": self.log_pd_ref,
                       "ref_marginal": self.log_pd_ref, "raw_marginal": self.log_pd}.get(
                           reference, self.log_pi_y[y])
            return kl_to_prior(logits, log_ref, weight=weight)
        raise ValueError(method)
