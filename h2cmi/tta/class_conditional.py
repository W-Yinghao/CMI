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
from sklearn.metrics import balanced_accuracy_score

from h2cmi.config import TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.grid_io import stable_hash_int


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


@dataclass(frozen=True)
class VariantSpec:
    """One Stage-B1a adaptation variant, decomposed along three axes the study isolates:

      responsibility : 'none' | 'gen_oneshot' | 'gen_iterative' | 'oracle'
        how soft class assignments are obtained -- not at all (pooled), generated ONCE on the
        identity geometry and frozen (Q0), re-estimated every EM round (U2 feedback), or the
        true labels (the responsibility ceiling).
      update         : 'identity' | 'pooled' | 'prior_fixed' | 'joint'
        what moves -- nothing, a classless EMPIRICAL diagonal moment match, the transform with
        the prior pinned at pi_S (geometry-only), or transform + prior M-step (the joint).
      kind           : 'diag_affine' | 'lowrank_affine'   (transform family)
      restarts       : deterministic restarts; the low-rank fit is non-convex, so >1 restart
        (selected by the TRAINING-fold objective, never held-out labels) guards a `lowrank<=diag`
        result against being mere optimisation failure rather than family adequacy.
    """
    name: str
    responsibility: str
    update: str
    kind: str = "diag_affine"
    restarts: int = 1


# The frozen Stage-B1a matrix (CMI-off main experiment; a CMI-on retention arm is added only
# after a single candidate is chosen). oracle_oneshot_{diag,lowrank} give the C_family contrast
# (diagonal vs low-rank under the responsibility ceiling) for conditional rotation.
B1A_VARIANTS = (
    VariantSpec("identity",               "none",          "identity",    "diag_affine"),
    VariantSpec("pooled_empirical_diag",  "none",          "pooled",      "diag_affine"),    # empirical CORAL-diag
    VariantSpec("gen_oneshot_diag",       "gen_oneshot",   "prior_fixed", "diag_affine"),    # Q0_U0
    VariantSpec("gen_iterative_diag",     "gen_iterative", "prior_fixed", "diag_affine"),    # Q0_U2
    VariantSpec("oracle_oneshot_diag",    "oracle",        "prior_fixed", "diag_affine"),    # ceiling / diag-OOF
    VariantSpec("oracle_oneshot_lowrank", "oracle",        "prior_fixed", "lowrank_affine", 3),  # lowrank-OOF
    VariantSpec("joint_iterative_diag",   "gen_iterative", "joint",       "diag_affine"),    # current_joint
)
B1A_VARIANTS_BY_NAME = {v.name: v for v in B1A_VARIANTS}


@dataclass
class VariantFit:
    """Result of one variant fit. r_* expose the responsibilities ACTUALLY in play (review §3):
    r_initial   the identity-geometry posterior generated at the start (None for pooled/identity);
    r_last_used the responsibility the FINAL transform M-step consumed (== r_initial for one-shot
                and oracle; the last E-step for iterative; None for pooled/identity);
    r_final     the model posterior AFTER the fitted transform (always present)."""
    transform: Transform
    pi_T: torch.Tensor
    r_initial: torch.Tensor | None
    r_last_used: torch.Tensor | None
    r_final: torch.Tensor
    objective: float


@torch.no_grad()
def reference_weighted_source_moments(Us, ys, pi_star) -> tuple[torch.Tensor, torch.Tensor]:
    """EMPIRICAL pooled SOURCE-latent mean/std under the reference prior pi*: mu* = sum_y pi*(y)
    E[z|y], sigma*^2 = sum_y pi*(y) E[z^2|y] - mu*^2, estimated directly from source embeddings
    (NOT the density's parametric moments). This makes `pooled_empirical_diag` a baseline that is
    genuinely independent of p_phi(z|y), so C_class-cond measures the value of the class-
    conditional density, not a re-parameterisation of it."""
    Us = (Us.detach().to(device="cpu", dtype=torch.float32) if isinstance(Us, torch.Tensor)
          else torch.as_tensor(np.asarray(Us), dtype=torch.float32))   # accept a CUDA embedding
    ys = np.asarray(ys)
    K = len(np.asarray(pi_star))
    d = Us.shape[1]
    mu_y = torch.zeros(K, d)
    m2_y = torch.zeros(K, d)
    present = torch.zeros(K)
    for c in range(K):
        m = ys == c
        if m.sum() == 0:
            continue
        z = Us[m]
        mu_y[c] = z.mean(0)
        m2_y[c] = (z ** 2).mean(0)
        present[c] = 1.0
    pi = torch.as_tensor(np.asarray(pi_star), dtype=torch.float32) * present
    pi = (pi / pi.sum().clamp_min(1e-8)).view(-1, 1)              # renormalise over present classes
    mu = (pi * mu_y).sum(0)
    var = (pi * m2_y).sum(0) - mu ** 2
    return mu, var.clamp_min(1e-6).sqrt()


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
                       fixed_resp: torch.Tensor | None = None, kind: str | None = None,
                       return_resp: bool = False):
        """Run the EM (transform + target prior) on U. Density must already be frozen.

        ``fixed_prior`` (tensor [K]) freezes pi_T (skip the prior M-step) -- the oracle-prior
        diagnostic. ``fixed_resp`` (tensor [N,K]) freezes the responsibilities (skip the
        E-step) -- the oracle-labels / supervised-transform diagnostic. Both default None =
        the unsupervised behaviour. ``kind`` overrides the transform family (default
        ``self.cfg.transform``). With ``return_resp`` returns (T, pi_T, r_initial, r_last) where
        r_initial is the first E-step's responsibilities and r_last is the responsibility the
        final M-step consumed; else (T, pi_T)."""
        d = U.shape[1]
        T = Transform(d, kind or self.cfg.transform, self.cfg.lowrank, self.device)
        pi_T = (fixed_prior.clone() if fixed_prior is not None
                else torch.tensor(self.pi_S, dtype=torch.float32, device=self.device))
        opt = torch.optim.Adam(T.params, lr=self.cfg.em_lr)
        # Dirichlet pseudo-count anchor toward pi_S (penalises H(pi_S,pi_T)=KL(pi_S||pi_T)
        # +const; NOT a forward KL(pi_T||pi_S)):
        anchor = torch.tensor((self.cfg.dirichlet + self.cfg.prior_anchor_strength) * self.pi_S,
                              dtype=torch.float32, device=self.device)
        r_initial = r_last = None
        for it in range(self.cfg.em_iters):
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
            if it == 0:
                r_initial = r.detach().clone()
            r_last = r.detach()
            log_piT = torch.log(pi_T.clamp_min(1e-8))
            for _ in range(3):                                      # M-step transform
                z = T.apply(U)
                ll = (r * (self.density.log_prob_all(z) + log_piT.view(1, -1))).sum(1).mean()
                obj = (ll + self.cfg.logdet_weight * T.logdet()
                       - self.cfg.trust_region * T.trust() / d
                       - self.cfg.trust_region_b * (T.b ** 2).sum() / d)
                opt.zero_grad(); (-obj).backward(); opt.step()
        if return_resp:
            return T, pi_T.detach(), r_initial, r_last
        return T, pi_T.detach()

    @torch.no_grad()
    def _fit_objective(self, U, T, pi_T, r) -> float:
        """The (weighted) EM objective on U with responsibilities r -- used to SELECT among
        deterministic restarts by the TRAINING-fold fit, never held-out labels."""
        d = U.shape[1]
        log_piT = torch.log(pi_T.clamp_min(1e-8))
        ll = (r * (self.density.log_prob_all(T.apply(U)) + log_piT.view(1, -1))).sum(1).mean()
        return float(ll + self.cfg.logdet_weight * T.logdet()
                     - self.cfg.trust_region * T.trust() / d
                     - self.cfg.trust_region_b * (T.b ** 2).sum() / d)

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

    # -- action decomposition (Stage B0) -----------------------------------------
    def _fit_prior_only(self, U: torch.Tensor) -> torch.Tensor:
        """Estimate pi_T with the transform held at IDENTITY (prior-only action)."""
        pi_T = torch.tensor(self.pi_S, dtype=torch.float32, device=self.device)
        anchor = torch.tensor((self.cfg.dirichlet + self.cfg.prior_anchor_strength) * self.pi_S,
                              dtype=torch.float32, device=self.device)
        logp = self.density.log_prob_all(U)                     # T = I, constant across iters
        for _ in range(self.cfg.em_iters):
            r = F.softmax(logp + torch.log(pi_T.clamp_min(1e-8)).view(1, -1), dim=1)
            counts = r.sum(0)
            pi_T = (counts + anchor) / (counts.sum() + anchor.sum())
        return pi_T.detach()

    def fit_action(self, U: torch.Tensor, action: str) -> TTAResult:
        """Run ONE adaptation action UNCONDITIONALLY (no rollback) for decomposition:
        'identity' | 'prior_only' (T=I, estimate pi_T) | 'geometry_only' (fit T, hold pi=pi_S)
        | 'joint' (fit T and pi_T). Returns a TTAResult whose pi_T is the FIT prior (the
        caller chooses the DECISION prior separately -- review §3 confound fix)."""
        U = U.detach().to(self.device)
        d = U.shape[1]
        log_piS = torch.log(torch.tensor(self.pi_S, dtype=torch.float32, device=self.device).clamp_min(1e-8))
        nll_before = self._identity_diagnostics(U, log_piS)
        frozen = [(p, p.requires_grad) for p in self.density.parameters()]
        for p, _ in frozen:
            p.requires_grad_(False)
        try:
            if action == "identity":
                T = Transform(d, "diag_affine", device=self.device)
                pi = torch.tensor(self.pi_S, dtype=torch.float32, device=self.device)
            elif action == "prior_only":
                T = Transform(d, "diag_affine", device=self.device)
                pi = self._fit_prior_only(U)
            elif action == "geometry_only":
                T, pi = self._fit_transform(
                    U, fixed_prior=torch.tensor(self.pi_S, dtype=torch.float32, device=self.device))
            elif action == "joint":
                T, pi = self._fit_transform(U)
            else:
                raise ValueError(f"unknown action {action}")
            diag = self._diagnostics(U, T, pi, log_piS, nll_before)
            diag["action"] = action
            return TTAResult(T, pi.detach().cpu().numpy(), adapted=(action != "identity"),
                             diagnostics=diag)
        finally:
            for p, req in frozen:
                p.requires_grad_(req)

    # -- Stage-B1a variant decomposition (responsibility x update x family) -------
    def _fit_pooled_diag(self, U: torch.Tensor, pooled_ref):
        """Classless diagonal moment match: A,b so transformed U matches the EMPIRICAL source
        pooled per-dim mean/std (CORAL-diagonal). pooled_ref=(mu*, sigma*) comes from source
        embeddings (reference_weighted_source_moments), NOT the density -> the p(z|y) ablation."""
        if pooled_ref is None:
            raise ValueError("pooled_empirical_diag requires pooled_ref=(source_mu, source_std)")
        mu_S, sd_S = (torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in pooled_ref)
        mu_T = U.mean(0)
        sd_T = U.std(0, unbiased=False).clamp_min(1e-6)
        a = torch.log((sd_S / sd_T).clamp_min(1e-6))
        T = Transform(U.shape[1], "diag_affine", device=self.device)
        with torch.no_grad():
            T.a.copy_(a)
            T.b.copy_(mu_S - torch.exp(a) * mu_T)
        return T, torch.tensor(self.pi_S, dtype=torch.float32, device=self.device), None, None

    def _variant_core(self, U, spec: VariantSpec, oracle_labels, pooled_ref):
        """Returns (T, pi_T, r_initial, r_last_used); r_* are None for identity/pooled."""
        d = U.shape[1]
        pi_S_t = torch.tensor(self.pi_S, dtype=torch.float32, device=self.device)
        log_piS = torch.log(pi_S_t.clamp_min(1e-8))
        if spec.update == "identity":
            return Transform(d, "diag_affine", device=self.device), pi_S_t, None, None
        if spec.update == "pooled":
            return self._fit_pooled_diag(U, pooled_ref)
        if spec.responsibility == "gen_oneshot":
            with torch.no_grad():                                   # generate r once on identity
                fixed_resp = F.softmax(self.density.log_prob_all(U) + log_piS.view(1, -1), dim=1)
        elif spec.responsibility == "oracle":
            if oracle_labels is None:
                raise ValueError(f"variant {spec.name} requires oracle_labels")
            yl = torch.as_tensor(np.asarray(oracle_labels), dtype=torch.long, device=self.device)
            fixed_resp = F.one_hot(yl, self.n_classes).to(torch.float32)
        else:                                                       # gen_iterative: re-estimate
            fixed_resp = None
        fixed_prior = pi_S_t if spec.update == "prior_fixed" else None
        return self._fit_transform(U, fixed_prior=fixed_prior, fixed_resp=fixed_resp,
                                   kind=spec.kind, return_resp=True)

    def fit_variant(self, U: torch.Tensor, spec: VariantSpec, *, oracle_labels=None,
                    pooled_ref=None, tta_seed: int | None = None) -> VariantFit:
        """Fit ONE B1a variant (density frozen) -> VariantFit exposing the responsibilities
        ACTUALLY used (r_initial/r_last_used) plus the post-fit posterior (r_final). DETERMINISTIC:
        with a tta_seed the fit randomness (low-rank init) is forked + seeded, so the result
        depends only on the seed -- never on call order or ambient RNG (variant-order invariance).
        spec.restarts>1 (low-rank) runs that many seeded restarts and keeps the one with the best
        TRAINING-fold objective (never held-out labels)."""
        U = U.detach().to(self.device)
        frozen = [(p, p.requires_grad) for p in self.density.parameters()]
        for p, _ in frozen:
            p.requires_grad_(False)
        try:
            best = None
            for ri in range(max(1, spec.restarts)):
                if tta_seed is not None:
                    seed = stable_hash_int(int(tta_seed), "restart", ri)
                    fork = [U.device.index] if (U.is_cuda and U.device.index is not None) else []
                    with torch.random.fork_rng(devices=fork):
                        torch.manual_seed(seed)
                        T, pi_T, r_init, r_last = self._variant_core(U, spec, oracle_labels, pooled_ref)
                else:
                    T, pi_T, r_init, r_last = self._variant_core(U, spec, oracle_labels, pooled_ref)
                obj = (self._fit_objective(U, T, pi_T, r_last) if r_last is not None
                       else -self._change_of_var_nll(U, T, pi_T))   # pooled/identity: more evidence better
                if best is None or obj > best[0]:
                    best = (obj, T, pi_T, r_init, r_last)
            obj, T, pi_T, r_init, r_last = best
            with torch.no_grad():
                r_final = self.density.class_posterior(T.apply(U), torch.log(pi_T.clamp_min(1e-8)))
            return VariantFit(T, pi_T.detach(),
                              None if r_init is None else r_init.detach(),
                              None if r_last is None else r_last.detach(),
                              r_final.detach(), float(obj))
        finally:
            for p, req in frozen:
                p.requires_grad_(req)

    def grouped_heldout(self, U: torch.Tensor, subject_ids, spec: VariantSpec, *, true_labels,
                        oracle_labels=None, pooled_ref=None, decision_prior=None, seed_parts=()) -> dict:
        """Per-target-SUBJECT LOSO held-out evidence. identity is scored as an ACTION (predict
        every trial with identity, evidence-gain 0). For non-identity, fit on the OTHER subjects
        and score the held-out one; a fold is skipped only when its fit set is below min_target,
        and ONLY oracle variants may read fit-set labels to require >=2 classes -- unsupervised
        variants never condition on yt[fit]. Reports coverage so a partial LOSO is not read as
        complete. Returns NaN metrics when LOSO is undefined (a single target subject)."""
        U = U.detach().to(self.device)
        subj = np.asarray(subject_ids)
        yt = np.asarray(true_labels)
        groups = np.unique(subj)
        n_total = int(len(groups))
        K = self.n_classes
        dp = np.full(K, 1.0 / K) if decision_prior is None else np.asarray(decision_prior, float)
        log_dp = torch.log(torch.tensor(dp, dtype=torch.float32, device=self.device).clamp_min(1e-8))
        pi_S_t = torch.tensor(self.pi_S, dtype=torch.float32, device=self.device)
        Tid = Transform(U.shape[1], "diag_affine", device=self.device)
        nan = dict(grouped_crossfit_evidence_gain=float("nan"), grouped_oof_bacc=float("nan"),
                   grouped_oof_nll=float("nan"), grouped_n_groups=n_total,
                   grouped_n_groups_total=n_total, grouped_n_groups_scored=0, grouped_oof_coverage=0.0)
        if n_total < 1 or (spec.name != "identity" and n_total < 2):
            return nan
        frozen = [(p, p.requires_grad) for p in self.density.parameters()]
        for p, _ in frozen:
            p.requires_grad_(False)
        oof_pred = np.full(len(yt), -1, dtype=np.int64)
        oof_nll = np.full(len(yt), np.nan)
        gains: list[float] = []
        n_scored = 0
        try:
            if spec.name == "identity":                            # identity is an action, not a fit
                with torch.no_grad():
                    p = self.density.class_posterior(U, log_dp).cpu().numpy()
                oof_pred[:] = p.argmax(1)
                oof_nll[:] = -np.log(np.clip(p[np.arange(len(yt)), yt], 1e-8, None))
                gains.append(0.0); n_scored = n_total
            else:
                for g in groups:
                    ev = subj == g
                    fit = ~ev
                    if fit.sum() < self.cfg.min_target:
                        continue
                    if spec.responsibility == "oracle" and _effective_classes(yt[fit], K) < 2:
                        continue                                    # ONLY oracle may read fit labels
                    ol = yt[fit] if spec.responsibility == "oracle" else None
                    seed = stable_hash_int(*seed_parts, spec.name, "fold", int(g))
                    fitres = self.fit_variant(U[fit], spec, oracle_labels=ol, pooled_ref=pooled_ref,
                                              tta_seed=seed)
                    Tg, pig = fitres.transform, fitres.pi_T
                    gains.append(self._change_of_var_nll(U[ev], Tid, pi_S_t)   # identity
                                 - self._change_of_var_nll(U[ev], Tg, pig))    # adapted
                    with torch.no_grad():
                        p = self.density.class_posterior(Tg.apply(U[ev]), log_dp).cpu().numpy()
                    idx = np.where(ev)[0]
                    oof_pred[idx] = p.argmax(1)
                    oof_nll[idx] = -np.log(np.clip(p[np.arange(len(idx)), yt[ev]], 1e-8, None))
                    n_scored += 1
        finally:
            for p, req in frozen:
                p.requires_grad_(req)
        scored = oof_pred >= 0
        if scored.sum() == 0 or not gains:
            return nan
        return dict(grouped_crossfit_evidence_gain=float(np.mean(gains)),
                    grouped_oof_bacc=float(balanced_accuracy_score(yt[scored], oof_pred[scored])),
                    grouped_oof_nll=float(np.nanmean(oof_nll[scored])),
                    grouped_n_groups=n_total, grouped_n_groups_total=n_total,
                    grouped_n_groups_scored=int(n_scored),
                    grouped_oof_coverage=float(scored.sum()) / len(yt))

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
