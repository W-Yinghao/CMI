"""Target-X observability audit (Fork 2, AUDIT ONLY; pre-reg + amendments 01 & 02 frozen).

Question: can an UNLABELED-target statistic on a calibration split rank beneficial deletion ACTIONS by their
true (query-set) utility? NOT: build an adapter. Firewall: target-X observables use T_cal X only; T_query X/Y
enter ONLY the hidden-outcome utility. Actions are typed (informed / random / baseline); only informed actions
are eligible for selection. Random controls are AMBIENT same-rank orthonormal projectors (not basis coordinate
subsets), >=50 per informed rank, comparators only. Primary observable = G1 with a source-task-safety gate and
a random-specificity gate; identity fallback when no safe+specific action exists. BNCI2015 outcome is session-
macro. All per-action rows are preserved. Pure numpy + sklearn.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from tos_cmi.eval.dg_identifiability import get_candidate_basis, source_greedy_select, _bacc, _source_loso_gain

PRIMARY = "G1"
TASK_SAFETY_MAX_DROP = 0.02       # source-LOSO bAcc drop allowed for an eligible deletion (amendment 02 B2)
SPECIFICITY_Q = 0.95              # G1 must exceed this quantile of the same-rank ambient random null


# ============================================================ session-aware calibration / query split
def session_split(session_target, y_target, seed=0):
    st = np.asarray(session_target); yt = np.asarray(y_target)
    sessions = sorted(np.unique(st).tolist())
    if len(sessions) >= 2:
        cal = st == sessions[0]; qry = np.isin(st, sessions[1:]); fallback = False
        query_labels = [s for s in sessions[1:]]
    else:
        cal = np.zeros(len(yt), bool)
        for c in np.unique(yt):
            idx = np.where(yt == c)[0]; cal[idx[: max(1, len(idx) // 2)]] = True
        qry = ~cal; fallback = True; query_labels = ["<temporal-second-half>"]
    info = dict(cal_sessions=[sessions[0]] if len(sessions) >= 2 else ["<temporal-first-half>"],
                query_sessions=query_labels, n_cal=int(cal.sum()), n_query=int(qry.sum()),
                fallback_used=bool(fallback))
    return cal, qry, info


# ============================================================ typed actions (informed / random / baseline)
def _orthonormal(rows):
    """Return an orthonormal row basis of the row space of `rows` [k,D] (drops rank deficiency)."""
    U, s, Vt = np.linalg.svd(np.atleast_2d(rows), full_matrices=False)
    keep = s > 1e-8 * (s.max() if s.size else 1.0)
    return Vt[keep]


def ambient_random_projectors(D, k, n, seed):
    """n AMBIENT random orthonormal rank-k projector direction-sets [k,D] (NOT coordinate subsets)."""
    out = []
    for t in range(n):
        rng = np.random.default_rng(90000 + seed * 131 + k * 17 + t)
        Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
        out.append(Q[:, :k].T)
    return out


def build_actions(B, Zs, ys, ds, seed=0, smoke=False, max_subset_rank=3, n_random_per_rank=50):
    """Typed action list. informed (eligible): identity/singletons/rank<=k subsets/source-greedy prefixes.
    random (NOT eligible): ambient same-rank orthonormal projectors, >=n_random_per_rank per informed rank.
    baseline (NOT eligible): whitening, target-mean-centering (added by the caller with T_cal stats)."""
    from itertools import combinations
    r = B.shape[0]; D = B.shape[1]
    kmax = 2 if smoke else max_subset_rank
    nrand = (8 if smoke else n_random_per_rank)
    acts = [dict(name="identity", kind="informed", rank=0, dirs=np.zeros((0, D)), eligible=True)]
    for j in range(r):
        acts.append(dict(name=f"singleton_{j}", kind="informed", rank=1, dirs=B[[j]], eligible=True))
    for k in range(2, min(kmax, r) + 1):
        for c in combinations(range(r), k):
            acts.append(dict(name=f"rank{k}_{'-'.join(map(str, c))}", kind="informed", rank=k,
                             dirs=_orthonormal(B[list(c)]), eligible=True))
    if not smoke:
        S_src = source_greedy_select(Zs, ys, ds, B, seed=seed)
        for m in range(1, len(S_src) + 1):
            acts.append(dict(name=f"srcgreedy_prefix{m}", kind="informed", rank=m,
                             dirs=_orthonormal(B[S_src[:m]]), eligible=True))
    informed_ranks = sorted({a["rank"] for a in acts if a["kind"] == "informed" and a["rank"] >= 1})
    for k in informed_ranks:                                    # ambient same-rank random controls (not eligible)
        for i, Q in enumerate(ambient_random_projectors(D, k, nrand, seed)):
            acts.append(dict(name=f"random_r{k}_{i}", kind="random", rank=k, dirs=Q, eligible=False))
    return acts


def apply_action(Z, action):
    dirs = action.get("dirs")
    if action.get("apply_fn") is not None:
        return action["apply_fn"](Z)
    if dirs is None or dirs.shape[0] == 0:
        return Z
    return Z - (Z @ dirs.T) @ dirs


# ============================================================ target-X observables (T_cal X only)
def observable_G1(dirs, ctx):
    """PRIMARY: reduction in squared source-target-cal mean discrepancy = ||P_dirs d||^2 (orthonormal dirs)."""
    if dirs.shape[0] == 0:
        return 0.0
    d = ctx["mu_s"] - ctx["mu_tcal"]
    return float(np.sum((dirs @ d) ** 2))


def observable_G2_sanity(dirs, ctx):
    """SANITY ONLY (algebraically identical to G1 on an orthonormal basis; NOT independent evidence)."""
    return observable_G1(dirs, ctx)


def observable_G5(dirs, ctx):
    """Condition number in the RETAINED (complement) subspace: drop the k deleted (~0) eigenvalues."""
    k = dirs.shape[0]
    if k == 0:
        return 0.0
    Zc = ctx["Zs"] - (ctx["Zs"] @ dirs.T) @ dirs
    ev = np.sort(np.linalg.eigvalsh(np.cov(Zc.T) + 1e-12 * np.eye(Zc.shape[1])))[::-1]
    ev = ev[: ev.shape[0] - k]                                  # retained subspace only
    ev = np.clip(ev, 1e-12, None)
    return -(float(np.log(ev[0] / ev[-1])) - ctx["log_kappa_identity"])


def observable_P4(dirs, ctx):
    Zc = ctx["Xcal"] - (ctx["Xcal"] @ dirs.T) @ dirs if dirs.shape[0] else ctx["Xcal"]
    pred = ctx["head"].predict(Zc)
    pt = np.bincount(pred, minlength=ctx["n_cls"]).astype(float) + 1e-6; pt /= pt.sum()
    ps = ctx["p_source_prior"]; m = 0.5 * (pt + ps)
    jsd = 0.5 * float((pt * np.log(pt / m)).sum() + (ps * np.log(ps / m)).sum())
    return -jsd


def observable_C3(dirs, ctx):
    """Both source and target pseudo-task contrasts transformed by (I - P_dirs) (same post-deletion frame)."""
    Zc = ctx["Xcal"] - (ctx["Xcal"] @ dirs.T) @ dirs if dirs.shape[0] else ctx["Xcal"]
    pl = ctx["head"].predict(Zc); classes = ctx["classes"]; cos = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            ma, mb = Zc[pl == classes[a]], Zc[pl == classes[b]]
            v_s0 = ctx["src_contrasts"].get((a, b))
            if len(ma) < 2 or len(mb) < 2 or v_s0 is None:
                continue
            v_t = ma.mean(0) - mb.mean(0)
            v_s = v_s0 - (v_s0 @ dirs.T) @ dirs if dirs.shape[0] else v_s0     # SAME transform on source dir
            if np.linalg.norm(v_t) < 1e-8 or np.linalg.norm(v_s) < 1e-8:
                continue
            cos.append(float(v_t @ v_s / (np.linalg.norm(v_t) * np.linalg.norm(v_s))))
    return float(np.mean(cos)) if cos else 0.0


# G1 = primary; G2 = sanity(=G1); G5/P4/C3 = frozen secondary (run in F2.2 after primary is frozen).
OBSERVABLES = {"G1": observable_G1, "G2_sanity": observable_G2_sanity,
               "G5": observable_G5, "P4": observable_P4, "C3": observable_C3}
SECONDARY = ["G5", "P4", "C3"]     # G3,G4,P1,P2,P3,C1,C2 implemented in F2.2; G2 excluded (=G1)


# ============================================================ gated selector + session-macro utility
def source_task_drop(Zs, ys, ds, dirs, seed=0):
    """Source-LOSO held-out bAcc DROP from deleting span(dirs) (>0 means the deletion hurts source task)."""
    from tos_cmi.eval.dg_identifiability import _bacc as _bb
    subs = np.unique(ds); drops = []
    for v in subs:
        tr = ds != v; te = ds == v
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
            continue
        base = _bb(Zs[tr], ys[tr], Zs[te], ys[te], seed)
        got = _bb(Zs[tr] - (Zs[tr] @ dirs.T) @ dirs, ys[tr], Zs[te] - (Zs[te] @ dirs.T) @ dirs, ys[te], seed) \
            if dirs.shape[0] else base
        drops.append(base - got)
    return float(np.mean(drops)) if drops else float("nan")


def g1_select(actions, ctx, Zs, ys, ds, seed=0):
    """S_TX = argmax_G1 over SAFE (source-task drop <= 0.02) & SPECIFIC (G1 > Q95 same-rank ambient random)
    informed actions; identity if none qualify. Returns (selected_action, diagnostics)."""
    # same-rank random G1 null (Q95 per rank) from the ambient random controls
    null_by_rank = {}
    for a in actions:
        if a["kind"] == "random":
            null_by_rank.setdefault(a["rank"], []).append(observable_G1(a["dirs"], ctx))
    q95 = {k: float(np.quantile(v, SPECIFICITY_Q)) for k, v in null_by_rank.items() if v}
    cands = []
    for a in actions:
        if not (a["kind"] == "informed" and a["eligible"] and a["rank"] >= 1):
            continue
        g1 = observable_G1(a["dirs"], ctx)
        drop = source_task_drop(Zs, ys, ds, a["dirs"], seed)
        specific = g1 > q95.get(a["rank"], np.inf)
        safe = np.isfinite(drop) and drop <= TASK_SAFETY_MAX_DROP
        if safe and specific:
            cands.append((g1, a))
    if not cands:
        ident = next(a for a in actions if a["name"] == "identity")
        return ident, {"q95_by_rank": q95, "n_candidates": 0}
    g1, a = max(cands, key=lambda t: t[0])
    return a, {"q95_by_rank": q95, "n_candidates": len(cands), "selected_g1": g1}


def utility(action, Zs, ys, Zt_q, yt_q, session_q, seed=0):
    """Hidden ground-truth utility on T_query (Y_query used ONLY here). Returns (macro, pooled). macro = mean
    over query sessions of [bAcc_session(delete) - bAcc_session(identity)]; pooled = pooled-trial gain."""
    def gain(Zq, yq):
        ident = _bacc(Zs, ys, Zq, yq, seed)
        got = _bacc(apply_action(Zs, action), ys, apply_action(Zq, action), yq, seed)
        return got - ident
    pooled = gain(Zt_q, yt_q)
    sess = np.asarray(session_q); per = []
    for s in np.unique(sess):
        m = sess == s
        if m.sum() >= 4 and len(np.unique(yt_q[m])) >= 2:
            per.append(gain(Zt_q[m], yt_q[m]))
    macro = float(np.mean(per)) if per else pooled
    return float(macro), float(pooled)


# ============================================================ per-fold audit (firewall-traced, all rows kept)
def audit_fold(feat, seed=0, family="cond", max_rank=10, smoke=False, observables=None, n_random_per_rank=50):
    from tos_cmi.eeg.relaxation_ladder import _dense
    Zs = np.asarray(feat["Z_source"], float); ys = np.asarray(feat["y_source"]).astype(int)
    ds = _dense(feat["subj_source"]); Zt = np.asarray(feat["Z_target"], float); yt = np.asarray(feat["y_target"]).astype(int)
    st = feat["session_target"]; n_cls = int(feat["n_cls"]); classes = sorted(np.unique(ys).tolist())
    cal, qry, sinfo = session_split(st, yt, seed)
    Xcal = Zt[cal]; Xq, yq, sq = Zt[qry], yt[qry], np.asarray(st)[qry]
    trace = {"basis_fit_on": "source_only", "targetx_scores_use": "T_cal_X_only",
             "query_x_used_for_selection": False, "query_y_used_for_selection": False,
             "query_x_used_for_outcome": True, "query_y_used_for_outcome": True,
             "target_greedy_in_action_set": False, "random_controls_selectable": False,
             "fallback_used": sinfo["fallback_used"]}
    B = get_candidate_basis(family, False, Zs, ys, ds, max_rank=max_rank, seed=seed)
    if B.shape[0] == 0:
        return None
    head = LogisticRegression(max_iter=300).fit(Zs, ys)
    ev = np.sort(np.linalg.eigvalsh(np.cov(Zs.T) + 1e-12 * np.eye(Zs.shape[1])))[::-1]; ev = np.clip(ev, 1e-12, None)
    src_contrasts = {(a, b): Zs[ys == classes[a]].mean(0) - Zs[ys == classes[b]].mean(0)
                     for a in range(len(classes)) for b in range(a + 1, len(classes))}
    p_src = np.bincount(ys, minlength=n_cls).astype(float); p_src /= p_src.sum()
    ctx = dict(Zs=Zs, mu_s=Zs.mean(0), mu_tcal=Xcal.mean(0), Xcal=Xcal, head=head, n_cls=n_cls, classes=classes,
               src_contrasts=src_contrasts, p_source_prior=p_src, log_kappa_identity=float(np.log(ev[0] / ev[-1])))
    obs = observables or (["G1"] if smoke else (["G1"] + SECONDARY))
    actions = build_actions(B, Zs, ys, ds, seed=seed, smoke=smoke, n_random_per_rank=n_random_per_rank)
    rows = []
    for a in actions:
        sc = {ob: (0.0 if a["rank"] == 0 else float(OBSERVABLES[ob](a["dirs"], ctx))) for ob in obs}
        macro, pooled = utility(a, Zs, ys, Xq, yq, sq, seed)
        rows.append({"action": a["name"], "kind": a["kind"], "rank": a["rank"], "eligible": a["eligible"],
                     "scores": sc, "utility_macro": macro, "utility_pooled": pooled})
    sel, diag = g1_select(actions, ctx, Zs, ys, ds, seed)
    sel_row = next(rw for rw in rows if rw["action"] == sel["name"])
    rand_macros = [rw["utility_macro"] for rw in rows if rw["kind"] == "random"]
    return {"heldout_subject": str(feat["heldout_subject"]), "seed": int(feat.get("seed", seed)),
            "dataset": feat.get("dataset", ""), "session_info": sinfo, "firewall": trace,
            "n_actions": len(rows), "n_informed": sum(rw["kind"] == "informed" for rw in rows),
            "n_random": sum(rw["kind"] == "random" for rw in rows),
            "selected_action": sel["name"], "selected_rank": sel["rank"],
            "delta_tx_macro": sel_row["utility_macro"], "delta_tx_pooled": sel_row["utility_pooled"],
            "delta_random_macro_mean": float(np.mean(rand_macros)) if rand_macros else float("nan"),
            "selector_diag": {"n_candidates": diag["n_candidates"]}, "rows": rows}
