"""Target-X observability audit (Fork 2, PM-approved AUDIT ONLY; pre-reg + amendment 01 frozen).

Question: can an UNLABELED-target statistic on a calibration split rank beneficial deletion actions by their
true (query-set) utility? NOT: can we build an adapter. Firewall: target-X observables use T_cal X only;
T_query X/Y enter ONLY the hidden-outcome utility. Eligible action set EXCLUDES the target-greedy path
(Y_target contamination). Primary observable = G1 (source-target mean-discrepancy reduction). Identity action
score = 0; if all deletion scores <= 0 the selector returns identity. Pure numpy + sklearn.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from tos_cmi.eval.dg_identifiability import get_candidate_basis, source_greedy_select, _bacc


# ============================================================ session-aware calibration / query split (A6/F2.1)
def session_split(session_target, y_target, seed=0):
    """PRIMARY session-aware split: first session -> T_cal, remaining -> T_query. Fallback (single session):
    stratified temporal block (first-half cal). Returns (cal_mask, query_mask, info)."""
    st = np.asarray(session_target); yt = np.asarray(y_target)
    sessions = sorted(np.unique(st).tolist())
    if len(sessions) >= 2:
        cal = st == sessions[0]; qry = np.isin(st, sessions[1:]); fallback = False
    else:                                                       # single session -> temporal block by index/class
        cal = np.zeros(len(yt), bool)
        for c in np.unique(yt):
            idx = np.where(yt == c)[0]
            cal[idx[: max(1, len(idx) // 2)]] = True
        qry = ~cal; fallback = True
    info = dict(cal_sessions=[sessions[0]] if len(sessions) >= 2 else ["<temporal-first-half>"],
                query_sessions=sessions[1:] if len(sessions) >= 2 else ["<temporal-second-half>"],
                n_cal=int(cal.sum()), n_query=int(qry.sum()), fallback_used=bool(fallback),
                query_session_labels=(sessions[1:] if len(sessions) >= 2 else []))
    return cal, qry, info


# ============================================================ eligible action set (A1: NO target-greedy)
def eligible_actions(B, Zs, ys, ds, seed=0, smoke=False, max_subset_rank=3):
    """Deployable candidate deletions (index lists into B): identity; singletons; all rank-<=k subsets;
    source-greedy prefixes; fixed-seed matched-rank random. NEVER the target-greedy path. smoke -> rank<=2,
    singletons only + identity + one random. Returns list of (name, S)."""
    from itertools import combinations
    r = B.shape[0]
    acts = [("identity", [])]
    acts += [(f"singleton_{j}", [j]) for j in range(r)]
    kmax = 2 if smoke else max_subset_rank
    for k in range(2, min(kmax, r) + 1):
        acts += [(f"rank{k}_{'-'.join(map(str, c))}", list(c)) for c in combinations(range(r), k)]
    if not smoke:
        S_src = source_greedy_select(Zs, ys, ds, B, seed=seed)   # source-only path (allowed)
        acts += [(f"srcgreedy_prefix{m}", S_src[:m]) for m in range(1, len(S_src) + 1)]
    rng = np.random.default_rng(2024 + seed)
    n_rand = 1 if smoke else 10
    for t in range(n_rand):
        k = rng.integers(1, min(max_subset_rank, r) + 1)
        acts.append((f"random_{t}", sorted(rng.choice(r, min(int(k), r), replace=False).tolist())))
    # dedup by frozenset
    seen, out = set(), []
    for name, S in acts:
        key = frozenset(S)
        if key not in seen:
            seen.add(key); out.append((name, S))
    return out


# ============================================================ target-X observables (A4 frozen); T_cal X only
def _proj_energy(B, S, v):
    if not S:
        return 0.0
    Bs = B[list(S)]
    return float(np.sum((v @ Bs.T) ** 2))


def _delete(Z, B, S):
    if not S:
        return Z
    Bs = B[list(S)]
    return Z - (Z @ Bs.T) @ Bs


def observable_G1(S, ctx):
    """PRIMARY: reduction in squared source-target-cal mean discrepancy after deleting span(B[S])."""
    diff = ctx["mu_s"] - ctx["mu_tcal"]
    full = float(diff @ diff)
    return full - float((_delete(diff[None, :], ctx["B"], S)[0]) @ (_delete(diff[None, :], ctx["B"], S)[0]))


def observable_G2(S, ctx):
    return _proj_energy(ctx["B"], S, (ctx["mu_tcal"] - ctx["mu_s"])[None, :])


def observable_G5(S, ctx):
    if not S:
        return 0.0
    Zc = _delete(ctx["Zs"], ctx["B"], S)
    ev = np.linalg.eigvalsh(np.cov(Zc.T) + 1e-9 * np.eye(Zc.shape[1]))
    ev = np.clip(ev, 1e-12, None)
    kappa = float(ev.max() / ev.min())
    return -(np.log(kappa) - ctx["log_kappa_identity"])


def observable_P4(S, ctx):
    Zc = _delete(ctx["Xcal"], ctx["B"], S)
    pred = ctx["head"].predict(Zc)
    pt = np.bincount(pred, minlength=ctx["n_cls"]).astype(float) + 1e-6; pt /= pt.sum()
    ps = ctx["p_source_prior"]
    m = 0.5 * (pt + ps)
    jsd = 0.5 * float((pt * np.log(pt / m)).sum() + (ps * np.log(ps / m)).sum())
    return -jsd


def observable_C3(S, ctx):
    """Macro-average cosine between source task-contrast directions and target-cal pseudo-task contrasts
    after deleting S (binary -> single contrast; 4-class -> all class pairs)."""
    Zc_cal = _delete(ctx["Xcal"], ctx["B"], S)
    pl = ctx["head"].predict(Zc_cal)
    classes = ctx["classes"]; cos = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            ma = Zc_cal[pl == classes[a]]; mb = Zc_cal[pl == classes[b]]
            if len(ma) < 2 or len(mb) < 2:
                continue
            v_t = ma.mean(0) - mb.mean(0)
            v_s = ctx["src_contrasts"].get((a, b))
            if v_s is None or np.linalg.norm(v_t) < 1e-8 or np.linalg.norm(v_s) < 1e-8:
                continue
            cos.append(float(v_t @ v_s / (np.linalg.norm(v_t) * np.linalg.norm(v_s))))
    return float(np.mean(cos)) if cos else 0.0


PRIMARY = "G1"
OBSERVABLES = {"G1": observable_G1, "G2": observable_G2, "G5": observable_G5, "P4": observable_P4, "C3": observable_C3}


# ============================================================ selector (A3 identity fallback) + utility
def targetx_select(scores):
    """S_TX = argmax score over NON-identity actions; identity fallback if every deletion score <= 0.
    `scores` = list of (name, S, score). Returns (name, S)."""
    dele = [(n, S, sc) for (n, S, sc) in scores if S]
    if not dele or max(sc for _, _, sc in dele) <= 0:
        return ("identity", [])
    n, S, _ = max(dele, key=lambda t: t[2])
    return (n, S)


def utility(S, Zs, ys, Zt_q, yt_q, B, seed=0):
    """Hidden ground-truth utility on T_query (uses Y_query ONLY here): bAcc_query(delete S) - identity."""
    ident = _bacc(Zs, ys, Zt_q, yt_q, seed)
    got = _bacc(_delete(Zs, B, S), ys, _delete(Zt_q, B, S), yt_q, seed)
    return float(got - ident)


# ============================================================ per-fold audit (firewall-traced)
def audit_fold(feat, seed=0, family="cond", max_rank=10, smoke=False, observables=None):
    """One LOSO fold. Firewall-traced: records exactly which arrays each stage touched. Returns a dict with
    per-action target-X scores (T_cal only), hidden utilities (T_query), the G1-selected action + its utility,
    and comparators (identity/random/source-greedy/target-hindsight-oracle)."""
    from tos_cmi.eeg.relaxation_ladder import _dense
    Zs = np.asarray(feat["Z_source"], float); ys = np.asarray(feat["y_source"]).astype(int)
    ds = _dense(feat["subj_source"]); Zt = np.asarray(feat["Z_target"], float); yt = np.asarray(feat["y_target"]).astype(int)
    st = feat["session_target"]; n_cls = int(feat["n_cls"]); classes = sorted(np.unique(ys).tolist())
    cal, qry, sinfo = session_split(st, yt, seed)
    Xcal, ycal = Zt[cal], yt[cal]; Xq, yq = Zt[qry], yt[qry]
    trace = {"basis_fit_on": "source_only", "targetx_scores_use": "T_cal_X_only",
             "query_x_used_for_selection": False, "query_y_used_for_selection": False,
             "query_x_used_for_outcome": True, "query_y_used_for_outcome": True,
             "target_greedy_in_action_set": False, "fallback_used": sinfo["fallback_used"]}
    B = get_candidate_basis(family, False, Zs, ys, ds, max_rank=max_rank, seed=seed)  # SOURCE-only basis
    if B.shape[0] == 0:
        return None
    head = LogisticRegression(max_iter=300).fit(Zs, ys)                              # source-only head
    ev = np.linalg.eigvalsh(np.cov(Zs.T) + 1e-9 * np.eye(Zs.shape[1])); ev = np.clip(ev, 1e-12, None)
    src_contrasts = {}
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            src_contrasts[(a, b)] = Zs[ys == classes[a]].mean(0) - Zs[ys == classes[b]].mean(0)
    p_src = np.bincount(ys, minlength=n_cls).astype(float); p_src /= p_src.sum()
    ctx = dict(B=B, Zs=Zs, mu_s=Zs.mean(0), mu_tcal=Xcal.mean(0), Xcal=Xcal, head=head, n_cls=n_cls,
               classes=classes, src_contrasts=src_contrasts, p_source_prior=p_src,
               log_kappa_identity=float(np.log(ev.max() / ev.min())))
    obs = observables or (["G1"] if smoke else list(OBSERVABLES))
    actions = eligible_actions(B, Zs, ys, ds, seed=seed, smoke=smoke)
    rows = []
    for name, S in actions:
        sc = {ob: (0.0 if not S else float(OBSERVABLES[ob](S, ctx))) for ob in obs}   # identity score = 0
        u = utility(S, Zs, ys, Xq, yq, B, seed)                                        # T_query outcome
        rows.append({"action": name, "S": S, "scores": sc, "utility": u})
    # G1-selected action + comparators
    g1_scores = [(rw["action"], rw["S"], rw["scores"]["G1"]) for rw in rows]
    sel_name, sel_S = targetx_select(g1_scores)
    delta_tx = next(rw["utility"] for rw in rows if rw["action"] == sel_name and rw["S"] == sel_S) \
        if any(rw["action"] == sel_name for rw in rows) else utility(sel_S, Zs, ys, Xq, yq, B, seed)
    rand_us = [rw["utility"] for rw in rows if rw["action"].startswith("random")]
    return {"heldout_subject": str(feat["heldout_subject"]), "seed": int(feat.get("seed", seed)),
            "dataset": feat.get("dataset", ""), "session_info": sinfo, "firewall": trace,
            "n_actions": len(rows), "selected_action": sel_name, "selected_S": sel_S,
            "delta_tx": float(delta_tx), "delta_random_mean": float(np.mean(rand_us)) if rand_us else 0.0,
            "rows": rows}
