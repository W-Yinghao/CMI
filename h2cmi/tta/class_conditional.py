"""Class-conditional probabilistic TTA: fit a constrained near-identity transform and the
target prior to the source class-conditional density (review section 6).

Objective (maximise over A, b, pi_T):

  sum_i log sum_y pi_T(y) p_phi(A u_i + b | y)
      + n log|det A|                    anti-collapse (Jacobian volume)
      - tau ||A - I||_F^2 - tau_b ||b||^2   trust region (stay near identity)
      + Dirichlet pseudo-count anchor on pi_T toward pi_S (penalises H(pi_S,pi_T) =
        KL(pi_S||pi_T)+const, the REVERSE direction -- NOT a forward KL(pi_T||pi_S))

solved by EM: E-step responsibilities r_iy ∝ pi_T(y) p_phi(A u_i+b|y); M-step updates
pi_T in closed form (Dirichlet-anchored at pi_S) and (A,b) by a few gradient ascent steps.

This faces target adaptation at the SOURCE class-conditional geometry, rather than matching
a single label-prior-mixed covariance (pooled CORAL).  Encoder/classifier stay FROZEN
(review: gradient-based encoder TTA is unreliable on EEG); only A, b, pi_T move.

Identity fallback when the target is too small or effectively single-class -- exactly the
identifiability boundary the review insists the method must respect.

Returns rich diagnostics consumed by the safety gate (``h2cmi.gate``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from h2cmi.config import TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity


class Transform:
    """A constrained near-identity affine map z = A u + b with three parameterisations."""

    def __init__(self, dim: int, kind: str = "diag_affine", lowrank: int = 4, device="cpu"):
        self.dim, self.kind = dim, kind
        self.b = torch.zeros(dim, device=device, requires_grad=True)
        if kind == "diag_affine":
            self.a = torch.zeros(dim, device=device, requires_grad=True)   # A = diag(exp(a))
            self.params = [self.a, self.b]
        elif kind == "lowrank_affine":
            # A = I + U V^T. With U=V=0 the gradient w.r.t. both is 0 (d(UV^T)/dU ∝ V),
            # so the transform could never leave identity. Seed BOTH with small random
            # values so both carry gradient while A starts ~identity (||UV^T|| ~ 1e-4). (P0-3)
            self.U = (1e-2 * torch.randn(dim, lowrank, device=device)).requires_grad_(True)
            self.V = (1e-2 * torch.randn(dim, lowrank, device=device)).requires_grad_(True)
            self.params = [self.U, self.V, self.b]
        elif kind == "full_affine":
            self.W = torch.eye(dim, device=device, requires_grad=True)     # A = W (init I)
            self.params = [self.W, self.b]
        else:
            raise ValueError(f"unknown transform kind {kind}")

    def matrix(self) -> torch.Tensor:
        if self.kind == "diag_affine":
            return torch.diag(torch.exp(self.a))
        if self.kind == "lowrank_affine":
            return torch.eye(self.dim, device=self.b.device) + self.U @ self.V.T
        return self.W

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        if self.kind == "diag_affine":
            return u * torch.exp(self.a) + self.b
        return u @ self.matrix().T + self.b

    def logdet(self) -> torch.Tensor:
        if self.kind == "diag_affine":
            return self.a.sum()
        return torch.linalg.slogdet(self.matrix())[1]

    def trust(self) -> torch.Tensor:
        I = torch.eye(self.dim, device=self.b.device)
        return ((self.matrix() - I) ** 2).sum()


@dataclass
class TTAResult:
    transform: Transform
    pi_T: np.ndarray
    adapted: bool
    diagnostics: dict = field(default_factory=dict)

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        return self.transform.apply(u)


def _effective_classes(y: np.ndarray, n_classes: int) -> int:
    return int((np.bincount(y, minlength=n_classes) > 0).sum())


class ClassConditionalTTA:
    def __init__(self, density: ClassConditionalDensity, source_prior: np.ndarray,
                 cfg: TTAConfig, n_classes: int, device: str = "cpu"):
        self.density = density
        self.cfg = cfg
        self.n_classes = n_classes
        self.device = device
        self.pi_S = np.asarray(source_prior, dtype=np.float64)
        self.pi_S /= self.pi_S.sum()

    # -- evidence / posteriors ---------------------------------------------------
    def _log_evidence(self, z, log_pi):
        """log sum_y pi(y) p(z|y) per sample -> [N]."""
        return torch.logsumexp(self.density.log_prob_all(z) + log_pi.view(1, -1), dim=1)

    @torch.no_grad()
    def _identity_diagnostics(self, U, log_piS):
        ev = self._log_evidence(U, log_piS)
        return float((-ev).mean())

    @torch.no_grad()
    def _change_of_var_nll(self, U, T, pi_T) -> float:
        """Mean NEGATIVE change-of-variable log-evidence -[log p_Z(T(u)) + log|det A|].

        This is the quantity the EM objective actually optimises, so it is what the
        identity-vs-adapt comparison must use (review P0-4: nll_after previously omitted
        the Jacobian term).
        """
        z = T.apply(U)
        log_pi = torch.log(pi_T.clamp_min(1e-8))
        ev = self._log_evidence(z, log_pi) + T.logdet()
        return float((-ev).mean())

    def _fit_transform(self, U: torch.Tensor, fixed_prior: torch.Tensor | None = None,
                       fixed_resp: torch.Tensor | None = None):
        """Run the EM (transform + target prior) on U. Density must already be frozen.

        ``fixed_prior`` (tensor [K]) freezes pi_T (skip the prior M-step) -- the oracle-prior
        diagnostic. ``fixed_resp`` (tensor [N,K]) freezes the responsibilities (skip the
        E-step) -- the oracle-labels / supervised-transform diagnostic. Both default None =
        the unsupervised behaviour. Returns (Transform, pi_T tensor)."""
        d = U.shape[1]
        T = Transform(d, self.cfg.transform, self.cfg.lowrank, self.device)
        pi_T = (fixed_prior.clone() if fixed_prior is not None
                else torch.tensor(self.pi_S, dtype=torch.float32, device=self.device))
        opt = torch.optim.Adam(T.params, lr=self.cfg.em_lr)
        # Dirichlet pseudo-count anchor toward pi_S (penalises H(pi_S,pi_T)=KL(pi_S||pi_T)
        # +const; NOT a forward KL(pi_T||pi_S)):
        anchor = torch.tensor((self.cfg.dirichlet + self.cfg.prior_anchor_strength) * self.pi_S,
                              dtype=torch.float32, device=self.device)
        for _ in range(self.cfg.em_iters):
            with torch.no_grad():
                if fixed_resp is not None:
                    r = fixed_resp                                  # supervised responsibilities
                else:
                    z = T.apply(U)
                    logits = self.density.log_prob_all(z) + torch.log(pi_T.clamp_min(1e-8)).view(1, -1)
                    r = F.softmax(logits, dim=1)                    # E-step
                if fixed_prior is None:
                    counts = r.sum(0)
                    pi_T = (counts + anchor) / (counts.sum() + anchor.sum())   # M-step prior
            log_piT = torch.log(pi_T.clamp_min(1e-8))
            for _ in range(3):                                      # M-step transform
                z = T.apply(U)
                ll = (r * (self.density.log_prob_all(z) + log_piT.view(1, -1))).sum(1).mean()
                obj = (ll + self.cfg.logdet_weight * T.logdet()
                       - self.cfg.trust_region * T.trust() / d
                       - self.cfg.trust_region_b * (T.b ** 2).sum() / d)
                opt.zero_grad(); (-obj).backward(); opt.step()
        return T, pi_T.detach()

    def _crossfit_evidence_gain(self, U: torch.Tensor) -> float:
        """2-fold cross-fitted held-out change-of-variable NLL improvement (review P0-4).

        Fit the transform on one half, score the *other* half's NLL improvement over
        identity; average both directions. The same data never both fits and judges.
        """
        U = U.detach()
        n = U.shape[0]
        perm = torch.randperm(n, device=U.device)
        half = n // 2
        Tid = Transform(U.shape[1], "diag_affine", device=self.device)   # identity
        pi_S = torch.tensor(self.pi_S, dtype=torch.float32, device=self.device)
        gains = []
        for fit_idx, ev_idx in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
            T, pi = self._fit_transform(U[fit_idx])
            nll_id = self._change_of_var_nll(U[ev_idx], Tid, pi_S)
            nll_ad = self._change_of_var_nll(U[ev_idx], T, pi)
            gains.append(nll_id - nll_ad)
        return float(np.mean(gains))

    # -- offline batch TTA -------------------------------------------------------
    def fit(self, U: torch.Tensor, pseudo_labels: np.ndarray | None = None) -> TTAResult:
        U = U.detach().to(self.device)          # TTA operates on FIXED target embeddings
        N, d = U.shape
        log_piS = torch.log(torch.tensor(self.pi_S, dtype=torch.float32, device=self.device).clamp_min(1e-8))
        nll_before = self._identity_diagnostics(U, log_piS)
        # identity-fallback guards (review identifiability boundary) -- BEFORE fitting
        if N < self.cfg.min_target:
            return self._identity_result(U, log_piS, nll_before, reason="too_few_target")
        if pseudo_labels is not None and _effective_classes(pseudo_labels, self.n_classes) < self.cfg.min_effective_classes:
            return self._identity_result(U, log_piS, nll_before, reason="single_class_target")

        # freeze density params during TTA (only A,b,pi_T move)
        frozen = [(p, p.requires_grad) for p in self.density.parameters()]
        for p, _ in frozen:
            p.requires_grad_(False)
        try:
            # decide on held-out evidence, then refit on full target if it passes
            gain = self._crossfit_evidence_gain(U)
            if gain <= self.cfg.min_heldout_evidence:
                res = self._identity_result(U, log_piS, nll_before, reason="no_heldout_evidence")
                res.diagnostics["crossfit_evidence_gain"] = gain
                return res
            T, pi_T = self._fit_transform(U)
            diag = self._diagnostics(U, T, pi_T, log_piS, nll_before)
            diag["crossfit_evidence_gain"] = gain
            return TTAResult(T, pi_T.cpu().numpy(), adapted=True, diagnostics=diag)
        finally:
            for p, req in frozen:
                p.requires_grad_(req)

    # -- online streaming TTA ----------------------------------------------------
    @torch.no_grad()
    def fit_online(self, batches) -> TTAResult:
        """Streaming PRIOR-ONLY adaptation (review P0-2): causally EMA-updates the target
        prior pi_T as batches arrive; the transform stays IDENTITY.

        Online *transform* adaptation is deliberately deferred (review: gradient-based TTA
        is unreliable on EEG; do it as a second stage with a causal, prequential protocol).
        Each item of ``batches`` is a [b,d] tensor in time order; no access to future
        samples. Reported under the honest label 'online_prior_only'.
        """
        ema = self.cfg.online_ema
        pi_T = torch.tensor(self.pi_S, dtype=torch.float32, device=self.device)
        d = None
        T = None
        log_piS = torch.log(torch.tensor(self.pi_S, dtype=torch.float32, device=self.device).clamp_min(1e-8))
        seen = 0
        Ubuf = []
        for xb in batches:
            xb = xb.to(self.device)
            seen += xb.shape[0]
            Ubuf.append(xb)
            if T is None:
                d = xb.shape[1]
                T = Transform(d, "diag_affine", device=self.device)
            z = T.apply(xb)
            logits = self.density.log_prob_all(z) + torch.log(pi_T.clamp_min(1e-8)).view(1, -1)
            r = F.softmax(logits, dim=1)
            counts = r.sum(0)
            pi_batch = (counts + 1e-3) / (counts.sum() + 1e-3 * self.n_classes)
            pi_T = ema * pi_T + (1 - ema) * pi_batch                  # EMA target prior
        U = torch.cat(Ubuf, 0) if Ubuf else torch.zeros(0, 1, device=self.device)
        nll_before = self._identity_diagnostics(U, log_piS) if seen else 0.0
        adapted = seen >= self.cfg.min_target
        diag = self._diagnostics(U, T, pi_T, log_piS, nll_before) if (T is not None and seen) else {}
        diag["online_seen"] = seen
        return TTAResult(T if T is not None else Transform(1, "diag_affine"),
                         pi_T.detach().cpu().numpy(), adapted=adapted, diagnostics=diag)

    # -- helpers -----------------------------------------------------------------
    def _identity_result(self, U, log_piS, nll_before, reason):
        d = U.shape[1] if U.ndim == 2 and U.shape[0] else 1
        T = Transform(d, "diag_affine", device=self.device)
        diag = dict(adapted=False, reason=reason, delta_density_nll=0.0,
                    transform_norm=0.0, condition_number=1.0, prior_shift=0.0,
                    pred_disagreement=0.0, ood_score=float(nll_before), ess=float(U.shape[0]),
                    cmi_residual=0.0)
        return TTAResult(T, self.pi_S.copy(), adapted=False, diagnostics=diag)

    @torch.no_grad()
    def _diagnostics(self, U, T, pi_T, log_piS, nll_before):
        z = T.apply(U)
        log_piT = torch.log(pi_T.clamp_min(1e-8))
        # change-of-variable NLL: include the +log|det A| Jacobian (review P0-4)
        nll_after = float((-(self._log_evidence(z, log_piT) + T.logdet())).mean())
        A = T.matrix()
        svals = torch.linalg.svdvals(A)
        cond = float((svals.max() / svals.clamp_min(1e-8).min()).cpu())
        pi_T_np = pi_T.detach().cpu().numpy()
        prior_shift = float(np.abs(pi_T_np - self.pi_S).sum())          # total variation x2
        # prediction disagreement: identity vs adapted argmax
        p_id = self.density.class_posterior(U, log_piS).argmax(1)
        p_ad = self.density.class_posterior(z, log_piT).argmax(1)
        disagree = float((p_id != p_ad).float().mean().cpu())
        # responsibilities ESS (min over classes)
        r = self.density.class_posterior(z, log_piT)
        cls_w = r.sum(0)
        ess = float((cls_w ** 2 / (r ** 2).sum(0).clamp_min(1e-8)).min().cpu())
        return dict(adapted=True, delta_density_nll=float(nll_before - nll_after),
                    transform_norm=float((T.trust().sqrt() + (T.b ** 2).sum().sqrt()).cpu()),
                    condition_number=cond, prior_shift=prior_shift,
                    pred_disagreement=disagree, ood_score=float(nll_after),
                    ess=ess, cmi_residual=0.0,
                    nll_before=float(nll_before), nll_after=float(nll_after))


if __name__ == "__main__":
    from h2cmi.config import DensityConfig
    torch.manual_seed(0)
    d, K = 8, 3
    dens = ClassConditionalDensity(d, K, DensityConfig(n_components=1, cov_rank=2))
    # plant separated class means so the density is informative
    with torch.no_grad():
        dens.mu[:, 0] = torch.eye(K, d)[:, :d] * 3.0
    pi_S = np.full(K, 1.0 / K)
    # target = source classes shifted by an affine map + prior skew
    rng = np.random.default_rng(0)
    yt = rng.choice(K, size=300, p=[0.6, 0.3, 0.1])
    U = dens.mu[yt, 0] + 0.3 * torch.randn(300, d)
    A_true = torch.eye(d) + 0.1 * torch.randn(d, d)
    U = U @ A_true.T + 0.2
    tta = ClassConditionalTTA(dens, pi_S, TTAConfig(em_iters=15), K)
    res = tta.fit(U, pseudo_labels=yt)
    print("adapted:", res.adapted, "pi_T:", np.round(res.pi_T, 3))
    print("diag:", {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in res.diagnostics.items()})
    # online
    batches = [U[i:i + 32] for i in range(0, len(U), 32)]
    ores = tta.fit_online(batches)
    print("online pi_T:", np.round(ores.pi_T, 3), "seen:", ores.diagnostics.get("online_seen"))
