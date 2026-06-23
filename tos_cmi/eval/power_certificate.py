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
                            _plugin_logratio_score, _one_sided_bound, _intrinsic_coords)
from .bayes_oracle import bayes_conditional_task_delta, classify_safety, logpost_true_label


def _control_geometry(d_base, d_extra, n_cls, n_dom, base_sep, conf_c, sigma, sigma_n, geom_seed):
    """FIXED geometry (class means muY + per-domain confound a) -> the Bayes Delta* depends ONLY
    on the geometry, NOT on the resampled observations, so every replicate at a given (geom_seed,
    conf_c) has EXACTLY the same effect size (fixes the per-replicate effect drift)."""
    grng = np.random.default_rng(geom_seed)
    muY = np.zeros((n_cls, d_base)); muY[:, 0] = base_sep * np.linspace(-1, 1, n_cls)
    a = grng.standard_normal(n_dom)
    Dd = d_base + d_extra
    mu_yd = np.zeros((n_cls, n_dom, Dd))
    for c in range(n_cls):
        for e in range(n_dom):
            mu_yd[c, e, :d_base] = muY[c]; mu_yd[c, e, 0] += conf_c * a[e]
            mu_yd[c, e, d_base] = a[e]
    return muY, a, mu_yd


def bayes_delta_of_geometry(d_base, d_extra, n_cls, n_dom, base_sep, conf_c, sigma, sigma_n,
                            geom_seed):
    """Exact Bayes Delta* = I(Y;n|u) for the FIXED control geometry (no observations needed)."""
    _, _, mu_yd = _control_geometry(d_base, d_extra, n_cls, n_dom, base_sep, conf_c, sigma,
                                    sigma_n, geom_seed)
    Dd = d_base + d_extra
    P = np.zeros((Dd, Dd))
    for j in range(d_base, Dd):
        P[j, j] = 1.0
    std = np.concatenate([np.full(d_base, sigma), np.full(d_extra, sigma_n)])
    py = np.full(n_cls, 1 / n_cls); pdy = np.full((n_cls, n_dom), 1 / n_dom)
    return bayes_conditional_task_delta(mu_yd, std, py, pdy, P, n_mc=14000, seed=geom_seed + 1)["delta"]


def make_control(d_base, d_extra, n_eff, n_cls, n_dom, base_sep, conf_c, sigma, sample_seed,
                 sigma_n=0.2, geom_seed=0):
    """Gaussian explaining-away control: kept `u` carries a class signal on e0 contaminated by a
    per-domain confound; deleted `n` reveals the domain at low noise. Geometry is FIXED by
    geom_seed (so Bayes Delta* is exact); only (y, d, noise) vary with sample_seed. Returns
    (u, n, y, d, bayes_delta)."""
    muY, a, mu_yd = _control_geometry(d_base, d_extra, n_cls, n_dom, base_sep, conf_c, sigma,
                                      sigma_n, geom_seed)
    srng = np.random.default_rng(sample_seed)
    y = srng.integers(0, n_cls, n_eff); d = srng.integers(0, n_dom, n_eff)
    u = muY[y] + sigma * srng.standard_normal((n_eff, d_base)); u[:, 0] += conf_c * a[d]
    n = sigma_n * srng.standard_normal((n_eff, d_extra)); n[:, 0] += a[d]
    b = bayes_delta_of_geometry(d_base, d_extra, n_cls, n_dom, base_sep, conf_c, sigma, sigma_n,
                                geom_seed)
    return u, n, y, d, b


def tune_confound(target_delta, d_base, d_extra, n_cls, n_dom, base_sep, sigma,
                  sigma_n=0.2, geom_seed=0, lo=0.0, hi=10.0, iters=22):
    """Bisection on the confound scale c so the FIXED-geometry Bayes Delta* ~ target_delta."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        b = bayes_delta_of_geometry(d_base, d_extra, n_cls, n_dom, base_sep, mid, sigma, sigma_n,
                                    geom_seed)
        if b < target_delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wilson_lcb(det, R, z=1.64):
    """One-sided Wilson lower bound on a binomial proportion det/R."""
    if R == 0:
        return 0.0
    p = det / R
    denom = 1 + z * z / R
    centre = p + z * z / (2 * R)
    half = z * np.sqrt(p * (1 - p) / R + z * z / (4 * R * R))
    return max(0.0, (centre - half) / denom)


def assert_power_feasible(R, beta, z=1.64):
    """Fail-fast: the Wilson LCB ceiling (at det=R) is R/(R+z^2); if it is below 1-beta the
    certificate can NEVER pass regardless of n -> a configuration bug, not a finding."""
    max_lcb = R / (R + z * z)
    if max_lcb < 1.0 - beta - 1e-9:
        raise ValueError("Power certificate infeasible: max Wilson LCB at det=R is %.3f < 1-beta="
                         "%.3f (R=%d, z=%.2f). Increase R (>=%d needed)."
                         % (max_lcb, 1 - beta, R, z, int(np.ceil(z * z * (1 - beta) / beta)) + 1))


def _critic_ucb(u, n, y, n_cls, cfg, seed, cluster_id=None):
    """Run the DEPLOYED task critic on (u,n) and return its probe_task_gain_ucb. Honours
    cfg.task_estimator ('plugin' = the Phase 1.3.3 cross-fitted log-ratio; 'nested' = diagnostic)
    and cfg.task_gate_folds, so the competence table certifies exactly the deployed estimator."""
    plan = _SplitPlan(len(y), cfg.task_gate_folds, seed + 3, group_id=cluster_id)
    if not plan.coverage_ok(y, n_cls):
        return None
    gplan = _GatePlan(plan, seed + 5)
    task_cfg = replace(cfg, hidden=cfg.task_gate_hidden, epochs=cfg.task_gate_epochs)
    score = (_plugin_logratio_score if cfg.task_estimator == "plugin" else _nested_residual_score)
    dY, _ = score(u, n, y, n_cls, task_cfg, gplan, seed + 100, restarts=cfg.task_gate_restarts)
    return float(_one_sided_bound(dY[:, None], cluster_id, cfg.gate_alpha, "upper",
                                  cfg.gate_boot, seed + 60, cfg.boot_estimand)[0])


def oracle_info_ucb(u, n, y, mu_yd, std, n_cls, cfg, seed, cluster_id=None):
    """BEST-POSSIBLE detector: per-sample info density s_i = log p(y|u,n) - log p(y|u) under the
    TRUE mixture (in the control's canonical coords the kept comp IS the u-block), one-sided UCB
    of mean(s) at the SAME level/bootstrap as the critic. Compares estimator vs information limit:
    oracle high & critic low => estimator bottleneck; both low => intrinsic sample-complexity."""
    d_base = u.shape[1]
    Z = np.concatenate([u, n], 1)
    py = np.full(n_cls, 1.0 / n_cls)
    lp_z = logpost_true_label(Z, y, mu_yd, np.diag(1.0 / std ** 2), py)
    lp_u = logpost_true_label(u, y, mu_yd[:, :, :d_base], np.diag(1.0 / std[:d_base] ** 2), py)
    s = (lp_z - lp_u)[:, None]
    return float(_one_sided_bound(s, cluster_id, cfg.gate_alpha, "upper", cfg.gate_boot,
                                  seed + 80, cfg.boot_estimand)[0])


def estimate_power(target_delta, d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, cfg,
                   R=30, seed=0, geom_seed=0, cluster_id=None, sigma_n=0.2, with_oracle=True):
    """pi(Delta)=Pr[UCB>delta_Y] over R matched controls (FIXED geometry -> exact effect) with
    one-sided Wilson LCB, for the CRITIC and (paired, same samples) the ORACLE info-density
    detector. Reports raw detection counts + the exact Bayes effect (delta_real)."""
    c = tune_confound(target_delta, d_base, d_extra, n_cls, n_dom, base_sep, sigma,
                      sigma_n=sigma_n, geom_seed=geom_seed)
    delta_real = bayes_delta_of_geometry(d_base, d_extra, n_cls, n_dom, base_sep, c, sigma,
                                         sigma_n, geom_seed)   # exact, same for every replicate
    _, _, mu_yd = _control_geometry(d_base, d_extra, n_cls, n_dom, base_sep, c, sigma, sigma_n,
                                    geom_seed)
    std = np.concatenate([np.full(d_base, sigma), np.full(d_extra, sigma_n)])
    det = 0; det_o = 0; used = 0
    for r in range(R):
        u, n, y, d, _ = make_control(d_base, d_extra, n_eff, n_cls, n_dom, base_sep, c, sigma,
                                     sample_seed=seed + 17 * (r + 1), geom_seed=geom_seed,
                                     sigma_n=sigma_n)
        ucb = _critic_ucb(u, n, y, n_cls, cfg, seed + 211 * (r + 1), cluster_id=cluster_id)
        if ucb is None:
            continue
        used += 1; det += int(ucb > cfg.delta_Y)
        if with_oracle:
            o = oracle_info_ucb(u, n, y, mu_yd, std, n_cls, cfg, seed + 311 * (r + 1), cluster_id)
            det_o += int(o > cfg.delta_Y)
    return {"pi": (det / used if used else 0.0), "lcb": float(wilson_lcb(det, used)),
            "det": det, "used": used, "delta_real": float(delta_real),
            "pi_oracle": (det_o / used if used else 0.0), "lcb_oracle": float(wilson_lcb(det_o, used)),
            "det_oracle": det_o}


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


def prefix_mde(d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, cfg, R=30, beta=0.2, seed=0,
               grid=None, geom_seed=0, cluster_id=None):
    """MDE_k(1-beta) over a target grid; power_ok iff MDE <= delta_Y. Fail-fast if (R,beta) make
    the Wilson LCB ceiling unreachable (would force power_ok=False at every n -- a config bug)."""
    assert_power_feasible(R, beta)
    if grid is None:
        eps = 0.3 * cfg.delta_Y
        grid = [cfg.delta_Y, cfg.delta_Y + eps, 2 * cfg.delta_Y]
    rows = []; mde = float("inf")
    for g in grid:
        r = estimate_power(g, d_base, d_extra, n_eff, n_cls, n_dom, base_sep, sigma, cfg,
                           R=R, seed=seed, geom_seed=geom_seed, cluster_id=cluster_id)
        rows.append({"target": g, **r})
        if r["lcb"] >= 1 - beta and g < mde:
            mde = g
    return {"mde": (None if mde == float("inf") else mde),
            "power_ok": (mde <= cfg.delta_Y + 1e-12), "rows": rows,
            "n_eff": n_eff, "d_base": d_base, "d_extra": d_extra, "n_cls": n_cls}
