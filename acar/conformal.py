"""Leave-one-source-cohort-out, cohort-CLUSTERED split-conformal upper bound + conservative router.

For deployment cohort c, the regressor and the conformal quantile are fit on the OTHER same-disease cohorts only.
Residuals e = ΔR_a − ĝ_a are pooled with EQUAL WEIGHT PER COHORT (clustered): we take the per-cohort one-sided
(1−α) residual quantile and use the max across cohorts — a clinical-cohort-level, not i.i.d.-per-batch, bound.

    U_a(B) = ĝ_a(φ_a(B)) + q_{1−α}        # P(ΔR_a ≤ U_a) ≥ 1−α (cohort-clustered)

Router: among non-identity actions with U_a(B) < −δ, execute argmin U_a; else identity (abstain).
"""
from __future__ import annotations
import numpy as np

from .regressor import ActionRegressor


def _onesided_quantile(residuals, alpha):
    """Conformal (1−α) upper quantile with finite-sample correction."""
    r = np.sort(np.asarray(residuals, float))
    n = len(r)
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(r[k])


def fit_action_router(action, train_by_cohort, alpha, seed=0):
    """train_by_cohort: {cohort: (X[n,f], dr[n])}. Returns (regressor, q_clustered)."""
    Xs = np.vstack([v[0] for v in train_by_cohort.values()])
    drs = np.concatenate([v[1] for v in train_by_cohort.values()])
    reg = ActionRegressor(seed=seed).fit(Xs, drs)
    per_cohort_q = []
    for X, dr in train_by_cohort.values():
        if len(X) == 0:
            continue
        res = dr - reg.predict(X)                              # one-sided residual ΔR − ĝ
        per_cohort_q.append(_onesided_quantile(res, alpha))
    q = float(max(per_cohort_q)) if per_cohort_q else 0.0      # clustered: conservative across cohorts
    return reg, q


def route(routers, phi_by_action, delta):
    """routers: {action: (reg, q)}; phi_by_action: {action: feature_vector}. Returns (chosen_action, U_by_action)."""
    U = {}
    for a, (reg, q) in routers.items():
        U[a] = float(reg.predict(phi_by_action[a][None])[0] + q)
    eligible = {a: u for a, u in U.items() if u < -delta}
    chosen = min(eligible, key=eligible.get) if eligible else "identity"
    return chosen, U


# ---------- closed-loop replay (G2) ----------
def replay(batches_eval, routers, delta, rng, fixed_adapter="matched_coral"):
    """Replay the router on held-out cohort batches and compare deployed-loss reduction vs baselines.

    Each item in batches_eval: dict(phi={action: fvec}, dr={action: ΔR_a}). All ΔR are realized (Phase-2).
    Returns deployed-NLL-reduction (= −mean ΔR; higher is better) for never/always/random/router, plus retained
    beneficial fraction of the router vs always_adapt.
    """
    n = len(batches_eval)
    if n == 0:
        return None
    always = np.array([b["dr"][fixed_adapter] for b in batches_eval])     # ΔR of always applying the fixed adapter
    routed = np.zeros(n); abstained = np.zeros(n, bool)
    for i, b in enumerate(batches_eval):
        chosen, _ = route(routers, b["phi"], delta)
        routed[i] = 0.0 if chosen == "identity" else b["dr"][chosen]
        abstained[i] = chosen == "identity"
    cov = float(1.0 - abstained.mean())                                   # adaptation coverage of the router
    # matched-coverage random abstention over the SAME fixed adapter
    k = int(round(abstained.sum()))
    rand_abst = np.zeros(n, bool)
    if 0 < k <= n:
        rand_abst[rng.choice(n, size=k, replace=False)] = True
    random_dr = np.where(rand_abst, 0.0, always)
    # beneficial alignment retained: sum of negative ΔR captured by router vs by always_adapt
    benefit_always = -np.minimum(always, 0.0).sum()
    benefit_router = -np.minimum(routed, 0.0).sum()
    retained = float(benefit_router / benefit_always) if benefit_always > 1e-12 else np.nan
    red = lambda x: float(-np.mean(x))                                    # NLL reduction = −mean ΔR
    return dict(
        n=n, coverage=cov, abstain_rate=float(abstained.mean()),
        nll_red_never=0.0, nll_red_always=red(always),
        nll_red_random=red(random_dr), nll_red_router=red(routed),
        retained_benefit_frac=retained,
        harmful_batches_always=int((always > 0).sum()),
        harmful_batches_router=int((routed > 0).sum()),
    )
