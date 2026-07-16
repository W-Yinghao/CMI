"""Target-X observability audit (Fork 2, AUDIT ONLY; pre-reg + amendments 01/02/03 frozen).

Question: can an UNLABELED-target statistic on a calibration split rank beneficial deletion ACTIONS by their
true (query-set) utility? NOT: build an adapter. Firewall: target-X observables use T_cal X only; T_query X/Y
enter ONLY the hidden-outcome utility. Actions carry three transforms (apply_source / apply_target_cal /
apply_target_query) so transductive baselines (whitening, target-mean-centering) are honest. Typed actions:
informed (selectable), random / baseline / oracle (comparators, never selectable). Primary observable = G1
with a source-task-safety gate and a random-specificity gate; identity fallback. BNCI2015 outcome is session-
macro. Full per-action audit trail is preserved (hashes, gate flags, source task drop) for reconstruction.
Pure numpy + sklearn.
"""
from __future__ import annotations
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression

from tos_cmi.eval.dg_identifiability import (get_candidate_basis, source_greedy_select, _bacc, _select_subset)

PRIMARY = "G1"
TASK_SAFETY_MAX_DROP = 0.02
SPECIFICITY_Q = 0.95
PRIMARY_MAX_RANK = 3        # primary selectable deletions are rank <= 3 (amendment 03 C1)


def _hash(arr):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(arr, float)).tobytes()).hexdigest()[:12]


# ============================================================ session split (unchanged)
def session_split(session_target, y_target, seed=0):
    st = np.asarray(session_target); yt = np.asarray(y_target)
    sessions = sorted(np.unique(st).tolist())
    if len(sessions) >= 2:
        cal = st == sessions[0]; qry = np.isin(st, sessions[1:]); fallback = False
        query_labels = list(sessions[1:])
    else:
        cal = np.zeros(len(yt), bool)
        for c in np.unique(yt):
            idx = np.where(yt == c)[0]; cal[idx[: max(1, len(idx) // 2)]] = True
        qry = ~cal; fallback = True; query_labels = ["<temporal-second-half>"]
    info = dict(cal_sessions=[sessions[0]] if len(sessions) >= 2 else ["<temporal-first-half>"],
                query_sessions=query_labels, n_cal=int(cal.sum()), n_query=int(qry.sum()), fallback_used=bool(fallback))
    return cal, qry, info


# ============================================================ typed actions with 3 transforms
def _orthonormal(rows):
    Vt = np.linalg.svd(np.atleast_2d(rows), full_matrices=False)[2]
    s = np.linalg.svd(np.atleast_2d(rows), full_matrices=False)[1]
    keep = s > 1e-8 * (s.max() if s.size else 1.0)
    return Vt[keep]


def _delete_fn(dirs):
    def f(Z):
        return Z if (dirs is None or dirs.shape[0] == 0) else Z - (Z @ dirs.T) @ dirs
    return f


def make_action(name, kind, rank, dirs=None, eligible=False, apply_source=None, apply_tcal=None, apply_tq=None,
                basis_indices=None, basis_hash=""):
    if dirs is not None and apply_source is None:
        f = _delete_fn(dirs); apply_source = apply_tcal = apply_tq = f
    return dict(name=name, kind=kind, rank=int(rank), dirs=dirs, eligible=bool(eligible),
                apply_source=apply_source, apply_target_cal=apply_tcal, apply_target_query=apply_tq,
                basis_indices=(list(basis_indices) if basis_indices is not None else None),
                basis_hash=basis_hash, projector_hash=(_hash(dirs) if dirs is not None and dirs.shape[0] else "identity"))


def ambient_random_projectors(D, k, n, seed):
    out = []
    for t in range(n):
        rng = np.random.default_rng(90000 + seed * 131 + k * 17 + t)
        Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
        out.append(Q[:, :k].T)
    return out


def build_actions(B, Zs, ys, ds, Xcal, seed=0, smoke=False, n_random_per_rank=50):
    """Full typed action set: informed (selectable, rank<=PRIMARY_MAX_RANK) + random/baseline/oracle
    (comparators, never selectable). Baselines use per-domain transforms (mean-centering, whitening)."""
    from itertools import combinations
    from tos_cmi.eeg.relaxation_ladder import whitening_only
    r, D = B.shape[0], B.shape[1]
    bh = _hash(B)
    kmax = min(2 if smoke else PRIMARY_MAX_RANK, r)
    nrand = (8 if smoke else n_random_per_rank)
    acts = [make_action("identity", "informed", 0, dirs=np.zeros((0, D)), eligible=True, basis_indices=[], basis_hash=bh)]
    for j in range(r):
        acts.append(make_action(f"singleton_{j}", "informed", 1, dirs=B[[j]], eligible=True, basis_indices=[j], basis_hash=bh))
    for k in range(2, kmax + 1):
        for c in combinations(range(r), k):
            acts.append(make_action(f"rank{k}_{'-'.join(map(str, c))}", "informed", k, dirs=_orthonormal(B[list(c)]),
                                    eligible=True, basis_indices=list(c), basis_hash=bh))
    if not smoke:
        S_src = source_greedy_select(Zs, ys, ds, B, seed=seed)
        for m in range(1, min(len(S_src), PRIMARY_MAX_RANK) + 1):
            acts.append(make_action(f"srcgreedy_prefix{m}", "informed", m, dirs=_orthonormal(B[S_src[:m]]),
                                    eligible=True, basis_indices=list(S_src[:m]), basis_hash=bh))
        # source-greedy STANDALONE comparator (full path, not eligible)
        if S_src:
            acts.append(make_action("srcgreedy_standalone", "comparator", len(S_src), dirs=_orthonormal(B[S_src]),
                                    eligible=False, basis_indices=list(S_src), basis_hash=bh))
    informed_ranks = sorted({a["rank"] for a in acts if a["kind"] == "informed" and a["rank"] >= 1})
    for k in informed_ranks:
        for i, Q in enumerate(ambient_random_projectors(D, k, nrand, seed)):
            acts.append(make_action(f"random_r{k}_{i}", "random", k, dirs=Q, eligible=False, basis_hash=bh))
    # baselines (per-domain transforms; NOT selectable)
    mu_s, mu_tcal = Zs.mean(0), Xcal.mean(0)
    acts.append(make_action("mean_centering", "baseline", -1, eligible=False,
                            apply_source=lambda Z, m=mu_s: Z - m, apply_tcal=lambda Z, m=mu_tcal: Z - m,
                            apply_tq=lambda Z, m=mu_tcal: Z - m, basis_hash=bh))
    wfn, _ = whitening_only(Zs)                                    # fit on source, applied to both domains
    acts.append(make_action("whitening", "baseline", -1, eligible=False,
                            apply_source=wfn, apply_tcal=wfn, apply_tq=wfn, basis_hash=bh))
    return acts


# ============================================================ target-X observables (T_cal X only)
def observable_G1(dirs, ctx):
    if dirs is None or dirs.shape[0] == 0:
        return 0.0
    d = ctx["mu_s"] - ctx["mu_tcal"]
    return float(np.sum((dirs @ d) ** 2))


def observable_G2_sanity(dirs, ctx):
    return observable_G1(dirs, ctx)


def observable_G5(dirs, ctx):
    if dirs is None or dirs.shape[0] == 0:
        return 0.0
    k = dirs.shape[0]
    Zc = ctx["Zs"] - (ctx["Zs"] @ dirs.T) @ dirs
    ev = np.sort(np.linalg.eigvalsh(np.cov(Zc.T) + 1e-12 * np.eye(Zc.shape[1])))[::-1][: ctx["Zs"].shape[1] - k]
    ev = np.clip(ev, 1e-12, None)
    return -(float(np.log(ev[0] / ev[-1])) - ctx["log_kappa_identity"])


def observable_P4(dirs, ctx):
    Zc = ctx["Xcal"] - (ctx["Xcal"] @ dirs.T) @ dirs if (dirs is not None and dirs.shape[0]) else ctx["Xcal"]
    pred = ctx["head"].predict(Zc)
    pt = np.bincount(pred, minlength=ctx["n_cls"]).astype(float) + 1e-6; pt /= pt.sum()
    ps = ctx["p_source_prior"]; m = 0.5 * (pt + ps)
    return -0.5 * float((pt * np.log(pt / m)).sum() + (ps * np.log(ps / m)).sum())


def observable_C3(dirs, ctx):
    has = dirs is not None and dirs.shape[0]
    Zc = ctx["Xcal"] - (ctx["Xcal"] @ dirs.T) @ dirs if has else ctx["Xcal"]
    pl = ctx["head"].predict(Zc); classes = ctx["classes"]; cos = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            ma, mb, v_s0 = Zc[pl == classes[a]], Zc[pl == classes[b]], ctx["src_contrasts"].get((a, b))
            if len(ma) < 2 or len(mb) < 2 or v_s0 is None:
                continue
            v_t = ma.mean(0) - mb.mean(0)
            v_s = (v_s0 - (v_s0 @ dirs.T) @ dirs) if has else v_s0
            if np.linalg.norm(v_t) < 1e-8 or np.linalg.norm(v_s) < 1e-8:
                continue
            cos.append(float(v_t @ v_s / (np.linalg.norm(v_t) * np.linalg.norm(v_s))))
    return float(np.mean(cos)) if cos else 0.0


OBSERVABLES = {"G1": observable_G1, "G2_sanity": observable_G2_sanity, "G5": observable_G5,
               "P4": observable_P4, "C3": observable_C3}
SECONDARY = ["G5", "P4", "C3"]     # G3,G4,P1,P2,P3,C1,C2 in F2.2; G2 excluded (=G1)


# ============================================================ gates helpers + session-macro utility
def source_task_drop(Zs, ys, ds, dirs, seed=0):
    from tos_cmi.eval.dg_identifiability import _bacc as _bb
    if dirs is None or dirs.shape[0] == 0:
        return 0.0
    drops = []
    for v in np.unique(ds):
        tr, te = ds != v, ds == v
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
            continue
        base = _bb(Zs[tr], ys[tr], Zs[te], ys[te], seed)
        got = _bb(Zs[tr] - (Zs[tr] @ dirs.T) @ dirs, ys[tr], Zs[te] - (Zs[te] @ dirs.T) @ dirs, ys[te], seed)
        drops.append(base - got)
    return float(np.mean(drops)) if drops else float("nan")


def utility(action, Zs, ys, Zt_q, yt_q, session_q, seed=0):
    """(macro, pooled). Uses the action's per-domain transforms (apply_source on source, apply_target_query on
    target query). macro = mean over query sessions of the per-session gain; pooled = pooled-trial gain."""
    Zs_a = action["apply_source"](Zs)

    def gain(Zq, yq):
        ident = _bacc(Zs, ys, Zq, yq, seed)
        got = _bacc(Zs_a, ys, action["apply_target_query"](Zq), yq, seed)
        return got - ident
    pooled = gain(Zt_q, yt_q)
    sess = np.asarray(session_q); per = []
    for s in np.unique(sess):
        m = sess == s
        if m.sum() >= 4 and len(np.unique(yt_q[m])) >= 2:
            per.append(gain(Zt_q[m], yt_q[m]))
    macro = float(np.mean(per)) if per else pooled
    return float(macro), float(pooled)


def g1_select(actions, ctx, Zs, ys, ds, seed=0):
    """S_TX = argmax G1 over SAFE (source-LOSO drop<=0.02) & SPECIFIC (G1 > Q95 same-rank ambient random)
    informed actions; identity if none. Also stamps every informed action with its gate flags + q95."""
    null_by_rank = {}
    for a in actions:
        if a["kind"] == "random":
            null_by_rank.setdefault(a["rank"], []).append(observable_G1(a["dirs"], ctx))
    q95 = {k: float(np.quantile(v, SPECIFICITY_Q)) for k, v in null_by_rank.items() if v}
    cands = []
    for a in actions:
        if not (a["kind"] == "informed" and a["eligible"]):
            continue
        g1 = observable_G1(a["dirs"], ctx)
        drop = source_task_drop(Zs, ys, ds, a["dirs"], seed) if a["rank"] >= 1 else 0.0
        specific = (a["rank"] >= 1) and (g1 > q95.get(a["rank"], np.inf))
        safe = np.isfinite(drop) and drop <= TASK_SAFETY_MAX_DROP
        a["_g1"], a["_drop"], a["_q95"] = g1, drop, q95.get(a["rank"], float("nan"))
        a["_safe"], a["_specific"] = bool(safe), bool(specific)
        if a["rank"] >= 1 and safe and specific:
            cands.append((g1, a))
    if not cands:
        return next(a for a in actions if a["name"] == "identity"), {"q95_by_rank": q95, "n_candidates": 0}
    return max(cands, key=lambda t: t[0])[1], {"q95_by_rank": q95, "n_candidates": len(cands)}


# ============================================================ per-fold audit (full trail)
def audit_fold(feat, seed=0, family="cond", max_rank=10, smoke=False, phase="primary", n_random_per_rank=50,
               config_hash="", git_sha=""):
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
             "fallback_used": sinfo["fallback_used"], "phase": phase}
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
    obs = ["G1"] if phase == "primary" else (["G1"] + SECONDARY)
    actions = build_actions(B, Zs, ys, ds, Xcal, seed=seed, smoke=smoke, n_random_per_rank=n_random_per_rank)
    # target-hindsight ORACLE (uses T_cal target labels to select; scored on T_query) -> recovery denominator
    bh = _hash(B)
    S_hind = _select_subset(Zs, ys, Xcal, yt[cal], B, "greedy", min(max_rank, PRIMARY_MAX_RANK), seed)
    hind = make_action("target_hindsight", "oracle", len(S_hind),
                       dirs=(_orthonormal(B[S_hind]) if S_hind else np.zeros((0, B.shape[1]))),
                       eligible=False, basis_indices=list(S_hind), basis_hash=bh)
    actions.append(hind)
    sel, diag = g1_select(actions, ctx, Zs, ys, ds, seed)
    rows = []
    for a in actions:
        sc = {ob: (0.0 if a["rank"] == 0 else float(OBSERVABLES[ob](a["dirs"], ctx)))
              for ob in obs if a["dirs"] is not None} if a["dirs"] is not None else {ob: None for ob in obs}
        macro, pooled = utility(a, Zs, ys, Xq, yq, sq, seed)
        rows.append(dict(action=a["name"], kind=a["kind"], rank=a["rank"], eligible=a["eligible"],
                         basis_family=family, basis_rank=int(B.shape[0]), basis_hash=a["basis_hash"],
                         projector_hash=a["projector_hash"], basis_indices=a.get("basis_indices"),
                         G1=sc.get("G1"), scores=sc,
                         source_task_drop=a.get("_drop"), random_q95_same_rank=a.get("_q95"),
                         safe_gate_pass=a.get("_safe"), specificity_gate_pass=a.get("_specific"),
                         utility_macro=macro, utility_pooled=pooled, config_hash=config_hash, git_sha=git_sha))

    def _u(name):
        r = next((rw for rw in rows if rw["action"] == name), None)
        return r["utility_macro"] if r else float("nan")
    rand_macros = [rw["utility_macro"] for rw in rows if rw["kind"] == "random"]
    d_tx = _u(sel["name"]); d_hind = _u("target_hindsight")
    recovery = float(d_tx / d_hind) if (np.isfinite(d_hind) and abs(d_hind) > 1e-9) else float("nan")
    fold = dict(heldout_subject=str(feat["heldout_subject"]), seed=int(feat.get("seed", seed)),
                dataset=feat.get("dataset", ""), session_info=sinfo, firewall=trace,
                n_actions=len(rows), n_informed=sum(rw["kind"] == "informed" for rw in rows),
                n_random=sum(rw["kind"] == "random" for rw in rows),
                selected_action=sel["name"], selected_rank=sel["rank"],
                selected_basis_indices=sel.get("basis_indices"), selected_basis_hash=sel["basis_hash"],
                delta_tx=d_tx, delta_tx_pooled=next((rw["utility_pooled"] for rw in rows if rw["action"] == sel["name"]), float("nan")),
                delta_source_greedy=_u("srcgreedy_standalone"), delta_whitening=_u("whitening"),
                delta_mean_centering=_u("mean_centering"),
                delta_random_same_rank=float(np.mean(rand_macros)) if rand_macros else float("nan"),
                delta_target_hindsight=d_hind, oracle_recovery_ratio=recovery,
                n_candidates=diag["n_candidates"])
    return {"fold": fold, "rows": rows}
