"""Phase 1.3 -- saturation phase diagram as an UNSAFE-ACCEPTANCE SEARCH (not average heatmaps).

For each cell (task-margin x domain-margin x sample-size x seed x generator), we run the gate
under several candidate sources -- crucially INCLUDING oracle-injected candidates (the
Bayes-unsafe 'safe' span fed directly to the gate), so we test the gate's POWER rather than
relying on the selector/projector to avoid danger -- and under both oracle-T and learned-T. Every
candidate prefix k is scored by BOTH the gate (probe_task_gain_ucb, domain_lcb, feasible) and the
EXACT Bayes oracle (delta + CI -> SAFE/UNSAFE/BAYES_AMBIGUOUS), then cross-classified:

    SAFE_ACCEPT | SAFE_REJECT | UNSAFE_REJECT | UNSAFE_ACCEPT | BAYES_AMBIGUOUS

UNSAFE_ACCEPT is the failure we hunt: the gate accepts a deletion the Bayes oracle says loses
conditional task info. The runner forces task_protect=True (the deployment projector); the global
ScoreFisherConfig default is flipped only after this search shows zero UNSAFE_ACCEPT.

NB the 'learned' candidate set is discovered in-sample (candidate_order on the same cell) -- this
is a gate-POWER diagnostic, not the deployed split pipeline (that is select_score_fisher, covered
by test_bayes_calibration). The point here is whether the gate ACCEPT/REJECT decision is
Bayes-calibrated when handed candidates, including adversarially-injected unsafe ones.
"""
from __future__ import annotations
import numpy as np

from ..score_fisher import (_metric, _SplitPlan, _cross_fit_fisher, estimate_task_basis,
                            task_protected_projector, candidate_order, ucb_rank_gate,
                            _m_orthonormal)
from .bayes_oracle import bayes_conditional_task_delta, classify_safety


def decision_class(accepted, verdict):
    """5-way class from (gate accepted this deletion?) x (Bayes verdict)."""
    if verdict == "BAYES_AMBIGUOUS":
        return "BAYES_AMBIGUOUS"
    if accepted:
        return "SAFE_ACCEPT" if verdict == "SAFE" else "UNSAFE_ACCEPT"
    return "SAFE_REJECT" if verdict == "SAFE" else "UNSAFE_REJECT"


def _fishers(Z, y, d, n_cls, n_dom, cfg, seed):
    z_dim = Z.shape[1]; y_oh = np.eye(n_cls)[y]
    plan = _SplitPlan(len(Z), cfg.n_folds, seed + 1)
    G_Y = _cross_fit_fisher(Z, y, None, n_cls, z_dim, 0, cfg, plan, seed)
    G_DgY = _cross_fit_fisher(Z, d, y_oh, n_dom, z_dim, n_cls, cfg, plan, seed + 100)
    return G_Y, G_DgY, _metric(Z, y, n_cls, cfg)


def run_cell(data, cfg, seed, candidate_modes=("learned", "oracle_nuisance"),
             t_sources=("oracle", "learned"), n_mc=12000):
    """Run the gate on one data cell under each (candidate_mode, t_source) and Bayes-score every
    prefix. Returns a list of per-prefix record dicts."""
    s = data["spec"]; n_cls, n_dom = s.n_cls, s.n_dom
    Z = data["Z"].astype(np.float64); y = data["y"]; d = data["d"]; truth = data["truth"]
    G_Y, G_DgY, M = _fishers(Z, y, d, n_cls, n_dom, cfg, seed)

    V_learned = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]
    V_nuis = _m_orthonormal(data["nuisance_basis"].astype(np.float64), M)   # injected candidate
    cand = {"learned": V_learned, "oracle_nuisance": V_nuis}
    tmap = {"oracle": _m_orthonormal(data["task_overlap_basis"].astype(np.float64), M),
            "learned": estimate_task_basis(G_Y, M, cfg)}

    recs = []
    for cm in candidate_modes:
        V = cand[cm]
        if V is None or V.shape[1] == 0:
            continue
        for ts in t_sources:
            T = tmap[ts]
            kstar, grecs, reason = ucb_rank_gate(Z, y, d, V, M, n_cls, n_dom, cfg, seed,
                                                 cluster_id=None, T_task=T)
            for g in grecs:
                base = {"candidate_mode": cm, "t_source": ts, "k": g["k"], "kstar": kstar,
                        "decision_reason": g.get("decision_reason")}
                if "task_info_ucb" not in g:                  # intersection sentinel record
                    recs.append({**base, "accepted": False, "bayes_delta": None,
                                 "class": "INTERSECTION"}); continue
                P, _ = task_protected_projector(V[:, :g["k"]], T, M)
                if P is None:
                    recs.append({**base, "accepted": False, "bayes_delta": None,
                                 "decision_reason": "TASK_SUBSPACE_INTERSECTION",
                                 "class": "INTERSECTION"}); continue
                b = bayes_conditional_task_delta(truth["mu_yd"], truth["sigma"], truth["py"],
                                                 truth["pdy"], P, n_mc=n_mc, seed=seed)
                verdict = classify_safety(b["ci_lo"], b["ci_hi"], cfg.delta_Y)
                accepted = bool(g["risk_feasible"])
                recs.append({**base, "accepted": accepted, "verdict": verdict,
                             "bayes_delta": b["delta"], "bayes_ci": [b["ci_lo"], b["ci_hi"]],
                             "probe_task_delta": g["task_info_delta_mean"],
                             "probe_task_ucb": g["probe_task_gain_ucb"],
                             "bayes_minus_probe_ucb": b["delta"] - g["probe_task_gain_ucb"],
                             "domain_brier_lcb": g["domain_lcb"], "domain_gain": g["domain_gain_mean"],
                             "class": decision_class(accepted, verdict)})
    return recs


def summarize(records):
    """Aggregate per-prefix records into 5-class counts + the UNSAFE_ACCEPT list + the worst
    under-detection (max bayes_delta among prefixes the gate ACCEPTED)."""
    from collections import Counter
    cnt = Counter(r["class"] for r in records)
    unsafe = [r for r in records if r["class"] == "UNSAFE_ACCEPT"]
    accepted = [r for r in records if r.get("accepted") and r.get("bayes_delta") is not None]
    worst = max(accepted, key=lambda r: r["bayes_delta"], default=None)
    return {"counts": dict(cnt), "n_unsafe_accept": len(unsafe), "unsafe_accept": unsafe,
            "worst_accepted_bayes_delta": (worst["bayes_delta"] if worst else 0.0)}
