"""Phase 1.3.1b -- finite-sample COMPETENCE CERTIFICATE for the task-risk gate (the power floor).

The diagnosis showed the nested critic UNDER-detects conditional task info at small n / low
capacity, so probe_task_gain_ucb <= delta_Y is NOT evidence of safety unless the critic has the
POWER to have detected a dangerous effect at this sample size. We certify that power with
matched positive controls (NOT a permutation null, which only bounds false positives, and NOT a
hardcoded n>=N threshold).

Construction. For a prefix with kept dim d_base and deleted dim d_extra at gate sample size n_eff
(and cluster structure), build a Gaussian explaining-away control in intrinsic coordinates:

    u = muY[y] + c * a_d e0  + noise      (kept: class signal on e0, contaminated by domain a_d)
    n = a_d e0' + noise                    (deleted: reveals the domain a_d)

so I(Y; n | u) > 0 purely through explaining-away (knowing a_d de-confounds u's class signal).
The effect size is tuned via the EXACT Bayes oracle (Z=[u,n], P deletes the n-block) by scaling c.
For each target Delta in a grid {delta_Y, delta_Y+eps, 2 delta_Y}, run the SAME nested critic
(same task capacity, restarts, folds, bootstrap) R times and estimate

    pi_k(Delta) = Pr[ probe_task_gain_ucb > delta_Y | Delta_Y* = Delta ] ,

with a one-sided lower bound LCB(pi). The minimum detectable effect is

    MDE_k(1-beta) = min { Delta in grid : LCB(pi_k(Delta)) >= 1 - beta } ,

and the prefix is power-qualified iff MDE_k <= delta_Y. The table is built OFFLINE on seeds and
cells DISJOINT from the evaluation grid; the gate does a CONSERVATIVE lookup (uncovered ->
power NOT ok -> abstain). This is a competence certificate for the pre-registered explaining-away
family, NOT a distribution-free guarantee for all nonlinear dependencies.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np

from ..score_fisher import (_metric, _SplitPlan, _GatePlan, _nested_residual_score,
                            _one_sided_bound, _intrinsic_coords)
from .bayes_oracle import bayes_conditional_task_delta, classify_safety


def make_control(d_base, d_extra, n_eff, n_cls, n_dom, base_sep, conf_c, sigma, seed, sigma_n=0.2):
    """Gaussian explaining-away control in intrinsic coords. The deleted carrier `n` reveals the
    domain at LOW noise (sigma_n) so the de-confounding (and hence I(Y;n|u)) is controllable via
    the confound scale conf_c. Returns (u, n, y, d, bayes_delta). NB the oracle uses a per-block
    diagonal covariance (sigma on u, sigma_n on n)."""
    rng = np.random.default_rng(seed)
    muY = np.zeros((n_cls, d_base))
    muY[:, 0] = base_sep * np.linspace(-1, 1, n_cls)          # class signal on e0
    a = rng.standard_normal(n_dom)                            # per-domain confound scalar
    y = rng.integers(0, n_cls, n_eff); d = rng.integers(0, n_dom, n_eff)
    u = muY[y] + sigma * rng.standard_normal((n_eff, d_base))
    u[:, 0] += conf_c * a[d]                                  # contaminate the class signal
    n = sigma_n * rng.standard_normal((n_eff, d_extra))
    n[:, 0] += a[d]                                           # reveals the domain (low noise)
    Dd = d_base + d_extra
    mu_yd = np.zeros((n_cls, n_dom, Dd))
    for c in range(n_cls):
        for e in range(n_dom):
            mu_yd[c, e, :d_base] = muY[c]; mu_yd[c, e, 0] += conf_c * a[e]
            mu_yd[c, e, d_base] = a[e]
    P = np.zeros((Dd, Dd))
    for j in range(d_base, Dd):
        P[j, j] = 1.0                                         # delete n-block
    std = np.concatenate([np.full(d_base, sigma), np.full(d_extra, sigma_n)])
    py = np.full(n_cls, 1 / n_cls); pdy = np.full((n_cls, n_dom), 1 / n_dom)
    b = bayes_conditional_task_delta(mu_yd, std, py, pdy, P, n_mc=12000, seed=seed + 1)["delta"]
    return u, n, y, d, b


def tune_confound(target_delta, d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, seed,
                  sigma_n=0.2, lo=0.0, hi=10.0, iters=20):
    """Bisection on the confound scale c so the control's Bayes Delta* ~ target_delta."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, _, _, _, b = make_control(d_base, d_extra, max(n_eff, 4000), n_cls, n_dom,
                                     base_sep, mid, sigma, seed, sigma_n=sigma_n)
        if b < target_delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _critic_ucb(u, n, y, n_cls, cfg, seed, cluster_id=None):
    """Run the deployed nested task critic on (u,n) and return probe_task_gain_ucb."""
    plan = _SplitPlan(len(y), cfg.n_folds, seed + 3, group_id=cluster_id)
    if not plan.coverage_ok(y, n_cls):
        return None
    gplan = _GatePlan(plan, seed + 5)
    task_cfg = replace(cfg, hidden=cfg.task_gate_hidden, epochs=cfg.task_gate_epochs)
    dY, _ = _nested_residual_score(u, n, y, n_cls, task_cfg, gplan, seed + 100, "nll",
                                   restarts=cfg.task_gate_restarts)
    return float(_one_sided_bound(dY[:, None], cluster_id, cfg.gate_alpha, "upper",
                                  cfg.gate_boot, seed + 60, cfg.boot_estimand)[0])


def estimate_power(target_delta, d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, cfg,
                   R=6, seed=0):
    """pi(Delta)=Pr[UCB>delta_Y] over R matched controls + Wilson LCB."""
    c = tune_confound(target_delta, d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, seed)
    det = 0; used = 0; deltas = []
    for r in range(R):
        u, n, y, d, b = make_control(d_base, d_extra, n_eff, n_cls, n_dom, base_sep, c, sigma,
                                     seed + 17 * (r + 1))
        ucb = _critic_ucb(u, n, y, n_cls, cfg, seed + 211 * (r + 1))
        if ucb is None:
            continue
        used += 1; deltas.append(b); det += int(ucb > cfg.delta_Y)
    if used == 0:
        return {"pi": 0.0, "lcb": 0.0, "delta_real": target_delta, "used": 0}
    p = det / used
    z = 1.64                                                  # one-sided ~95% Wilson lower bound
    denom = 1 + z * z / used
    centre = p + z * z / (2 * used)
    half = z * np.sqrt(p * (1 - p) / used + z * z / (4 * used * used))
    lcb = max(0.0, (centre - half) / denom)
    return {"pi": p, "lcb": float(lcb), "delta_real": float(np.mean(deltas)), "used": used}


def load_table(path):
    import json
    with open(path) as f:
        return json.load(f)


def lookup_power(table, n_eff, d_base, d_extra, n_cls):
    """CONSERVATIVE lookup: power_ok only if some calibrated cell with the SAME (d_base,d_extra,
    n_cls) and n_eff' <= n_eff is power_ok (power is monotone in n, so a smaller-n pass implies the
    larger actual n passes). Uncovered (d_base,d_extra,n_cls) -> power NOT ok (abstain). Returns
    (power_ok, info)."""
    cells = [t for t in table["table"] if t["d_base"] == d_base and t["d_extra"] == d_extra
             and t["n_cls"] == n_cls and t["n_eff"] <= n_eff]
    if not cells:
        return False, {"covered": False, "reason": "uncovered_shape"}
    best = min(cells, key=lambda t: n_eff - t["n_eff"])       # closest grid n_eff <= actual
    ok = any(t["power_ok"] for t in cells)
    return bool(ok), {"covered": True, "matched_n_eff": best["n_eff"], "mde": best["mde"],
                      "power_ok_cell": best["power_ok"]}


def prefix_mde(d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, cfg, R=6, beta=0.2, seed=0,
               grid=None):
    """MDE_k(1-beta) over a target grid; power_ok iff MDE <= delta_Y."""
    if grid is None:
        eps = 0.3 * cfg.delta_Y
        grid = [cfg.delta_Y, cfg.delta_Y + eps, 2 * cfg.delta_Y]
    rows = []
    mde = float("inf")
    for g in grid:
        r = estimate_power(g, d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, cfg,
                           R=R, seed=seed)
        rows.append({"target": g, **r})
        if r["lcb"] >= 1 - beta and g < mde:
            mde = g
    return {"mde": (None if mde == float("inf") else mde),
            "power_ok": (mde <= cfg.delta_Y + 1e-12), "rows": rows,
            "n_eff": n_eff, "d_base": d_base, "d_extra": d_extra, "n_cls": n_cls}
