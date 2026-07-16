"""Target-X observability audit (Fork 2, AUDIT ONLY; pre-reg + amendments 01/02/03/04 frozen).

Can an UNLABELED target-cal statistic rank low-rank deletion ACTIONS by their future-session (T_query) utility?
NOT build an adapter. F2.1b: everything geometric is defined in the SOURCE Ledoit-Wolf-WHITENED metric and the
primary basis is the TASK-CONTESTED cond span (row space of the whitened class-centered head), so the selector
can no longer prefer functionally-unused free directions. Actions carry whitened directions U + a raw-space
apply (affine map back). Typed: informed (selectable), random / baseline / oracle / comparator (never
selectable). Primary observable = G1 (whitened source-target-cal mean-discrepancy reduction) with a source-task-
safety gate + a random-specificity gate + identity fallback. The selector is a single SHARED, HASHED rule
(Gate 5 re-runs the SAME rule, not the same action). Session-macro outcome, exact-selected-rank random control,
constrained + unconstrained hindsight, full audit trail. Pure numpy + sklearn.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

from tos_cmi.eval.dg_identifiability import _bacc, _select_subset
from tos_cmi.eval import targetx_metric as M

PRIMARY = "G1"
TASK_SAFETY_MAX_DROP = M.RULE["task_safety_max_drop"]
SPECIFICITY_Q = M.RULE["specificity_quantile"]
PRIMARY_MAX_RANK = M.RULE["max_rank"]
SECONDARY = ["G5", "P4", "C3"]     # implemented but only run in phase='secondary' (F2.2)


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
    return cal, qry, dict(cal_sessions=[sessions[0]] if len(sessions) >= 2 else ["<temporal-first-half>"],
                          query_sessions=query_labels, n_cal=int(cal.sum()), n_query=int(qry.sum()),
                          fallback_used=bool(fallback))


# ============================================================ observables (whitened U + ctx)
def observable_G1(U, ctx):
    """PRIMARY: whitened source-target-cal mean-discrepancy reduction = ||U d~||^2, d~ = A_s(mu_s - mu_tcal)."""
    return 0.0 if (U is None or U.shape[0] == 0) else float(np.sum((U @ ctx["d_white"]) ** 2))


def observable_G2_sanity(U, ctx):
    return observable_G1(U, ctx)


def observable_G5(U, ctx):
    if U is None or U.shape[0] == 0:
        return 0.0
    Zw = ctx["Zs_w"]; Zc = Zw - (Zw @ U.T) @ U
    ev = np.sort(np.linalg.eigvalsh(np.cov(Zc.T) + 1e-12 * np.eye(Zc.shape[1])))[::-1][: Zw.shape[1] - U.shape[0]]
    return -(float(np.log(np.clip(ev, 1e-12, None)[0] / np.clip(ev, 1e-12, None)[-1])) - ctx["log_kappa_identity"])


def observable_P4(U, ctx):
    Xw = ctx["Xcal_w"]; Zc = Xw - (Xw @ U.T) @ U if (U is not None and U.shape[0]) else Xw
    pred = ctx["head_w"].predict(Zc)
    pt = np.bincount(pred, minlength=ctx["n_cls"]).astype(float) + 1e-6; pt /= pt.sum()
    ps = ctx["p_source_prior"]; m = 0.5 * (pt + ps)
    return -0.5 * float((pt * np.log(pt / m)).sum() + (ps * np.log(ps / m)).sum())


def observable_C3(U, ctx):
    has = U is not None and U.shape[0]
    Xw = ctx["Xcal_w"]; Zc = Xw - (Xw @ U.T) @ U if has else Xw
    pl = ctx["head_w"].predict(Zc); classes = ctx["classes"]; cos = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            ma, mb, v_s0 = Zc[pl == classes[a]], Zc[pl == classes[b]], ctx["src_contrasts_w"].get((a, b))
            if len(ma) < 2 or len(mb) < 2 or v_s0 is None:
                continue
            v_t = ma.mean(0) - mb.mean(0); v_s = (v_s0 - (v_s0 @ U.T) @ U) if has else v_s0
            if np.linalg.norm(v_t) < 1e-8 or np.linalg.norm(v_s) < 1e-8:
                continue
            cos.append(float(v_t @ v_s / (np.linalg.norm(v_t) * np.linalg.norm(v_s))))
    return float(np.mean(cos)) if cos else 0.0


OBSERVABLES = {"G1": observable_G1, "G2_sanity": observable_G2_sanity, "G5": observable_G5,
               "P4": observable_P4, "C3": observable_C3}


# ============================================================ actions (whitened U + raw apply)
def _act(name, kind, rank, U, W, eligible=False, basis_label="", basis_indices=None, basis_hash="",
         apply_source=None, apply_tcal=None, apply_tq=None):
    if U is not None and apply_source is None:
        f = M.whitened_delete_fn(U, W); apply_source = apply_tcal = apply_tq = f
    return dict(name=name, kind=kind, rank=int(rank), U=U, eligible=bool(eligible), basis_label=basis_label,
                basis_indices=(list(basis_indices) if basis_indices is not None else None), basis_hash=basis_hash,
                projector_hash=(M._hash(U) if (U is not None and U.shape[0]) else "identity"),
                apply_source=apply_source, apply_target_cal=apply_tcal, apply_target_query=apply_tq)


def build_actions(B_w, W, Zs, ys, ds, Xcal, seed=0, smoke=False, n_random_per_rank=50, basis_label="cond_contested"):
    """Typed actions in the whitened metric. informed (selectable, rank<=3) from the contested basis B_w;
    ambient whitened random per informed rank (not selectable); source-greedy prefixes + standalone (in BOTH
    smoke and full -> parity); baselines whitening + target-mean-centering (per-domain)."""
    from itertools import combinations
    r, D = B_w.shape[0], B_w.shape[1]; bh = M._hash(B_w)
    kmax = min(2 if smoke else PRIMARY_MAX_RANK, r)
    nrand = (8 if smoke else n_random_per_rank)
    acts = [_act("identity", "informed", 0, np.zeros((0, D)), W, eligible=True, basis_label=basis_label,
                 basis_indices=[], basis_hash=bh)]
    for j in range(r):
        acts.append(_act(f"singleton_{j}", "informed", 1, B_w[[j]], W, eligible=True, basis_label=basis_label,
                         basis_indices=[j], basis_hash=bh))
    for k in range(2, kmax + 1):
        for c in combinations(range(r), k):
            acts.append(_act(f"rank{k}_{'-'.join(map(str, c))}", "informed", k, M._orthonormal(B_w[list(c)]), W,
                             eligible=True, basis_label=basis_label, basis_indices=list(c), basis_hash=bh))
    # source-greedy over the SAME contested span, rank<=3 (policy-symmetric comparator; smoke AND full)
    S_src = _source_greedy_whitened(Zs, ys, ds, B_w, W, seed=seed, max_k=PRIMARY_MAX_RANK)
    for m in range(1, len(S_src) + 1):
        acts.append(_act(f"srcgreedy_prefix{m}", "informed", m, M._orthonormal(B_w[S_src[:m]]), W, eligible=True,
                         basis_label=basis_label, basis_indices=list(S_src[:m]), basis_hash=bh))
    # standalone comparator ALWAYS present (empty S_src == source-greedy chose identity -> delta 0, never NaN)
    acts.append(_act("srcgreedy_standalone", "comparator", len(S_src),
                     M._orthonormal(B_w[S_src]) if S_src else np.zeros((0, D)), W,
                     eligible=False, basis_label=basis_label, basis_indices=list(S_src), basis_hash=bh))
    for k in sorted({a["rank"] for a in acts if a["kind"] == "informed" and a["rank"] >= 1}):
        for i, U in enumerate(M.ambient_random_projectors_whitened(D, k, nrand, seed)):
            acts.append(_act(f"random_r{k}_{i}", "random", k, U, W, eligible=False, basis_label="ambient_whitened", basis_hash=bh))
    mu_s, mu_tcal = Zs.mean(0), Xcal.mean(0)
    acts.append(_act("mean_centering", "baseline", -1, None, W, eligible=False,
                     apply_source=lambda Z, m=mu_s: Z - m, apply_tcal=lambda Z, m=mu_tcal: Z - m,
                     apply_tq=lambda Z, m=mu_tcal: Z - m, basis_label="baseline", basis_hash=bh))
    acts.append(_act("whitening", "baseline", -1, None, W, eligible=False,
                     apply_source=lambda Z: M.to_whitened(Z, W), apply_tcal=lambda Z: M.to_whitened(Z, W),
                     apply_tq=lambda Z: M.to_whitened(Z, W), basis_label="baseline", basis_hash=bh))
    return acts


# ============================================================ safety, selection (shared hashed rule), utility
def source_task_drop(Zs, ys, ds, apply_fn, seed=0):
    drops = []
    for v in np.unique(ds):
        tr, te = ds != v, ds == v
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
            continue
        base = _bacc(Zs[tr], ys[tr], Zs[te], ys[te], seed)
        got = _bacc(apply_fn(Zs[tr]), ys[tr], apply_fn(Zs[te]), ys[te], seed)
        drops.append(base - got)
    return float(np.mean(drops)) if drops else float("nan")


def _source_greedy_whitened(Zs, ys, ds, B_w, W, seed=0, max_k=3):
    """Greedy source-LOSO deletion over the whitened contested basis (arbitrary coords), rank<=max_k."""
    r = B_w.shape[0]; S, cur = [], 0.0
    def gain(idx):
        U = M._orthonormal(B_w[idx]) if idx else np.zeros((0, B_w.shape[1]))
        f = M.whitened_delete_fn(U, W); accs = []
        for v in np.unique(ds):
            tr, te = ds != v, ds == v
            if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
                continue
            accs.append(_bacc(f(Zs[tr]), ys[tr], f(Zs[te]), ys[te], seed) - _bacc(Zs[tr], ys[tr], Zs[te], ys[te], seed))
        return float(np.mean(accs)) if accs else float("nan")
    for _ in range(min(max_k, r)):
        cand = [(gain(S + [j]), j) for j in range(r) if j not in S]
        if not cand:
            break
        bm, bj = max(cand, key=lambda x: (x[0] if np.isfinite(x[0]) else -1))
        if not np.isfinite(bm) or bm <= cur + 1e-4:
            break
        S.append(bj); cur = bm
    return S


def g1_select(actions, ctx, Zs, ys, ds, seed=0):
    """SHARED, HASHED selection rule (M.RULE): argmax G1 over SAFE (source-LOSO drop<=0.02) & SPECIFIC
    (G1 > Q95 same-rank ambient random) informed actions; identity fallback. Stamps gate flags on each action."""
    null_by_rank = {}
    for a in actions:
        if a["kind"] == "random":
            null_by_rank.setdefault(a["rank"], []).append(observable_G1(a["U"], ctx))
    q95 = {k: float(np.quantile(v, SPECIFICITY_Q)) for k, v in null_by_rank.items() if v}
    cands = []
    for a in actions:
        if not (a["kind"] == "informed" and a["eligible"]):
            continue
        g1 = observable_G1(a["U"], ctx)
        drop = source_task_drop(Zs, ys, ds, a["apply_source"], seed) if a["rank"] >= 1 else 0.0
        a["_g1"], a["_drop"], a["_q95"] = g1, drop, q95.get(a["rank"], float("nan"))
        a["_safe"] = bool(np.isfinite(drop) and drop <= TASK_SAFETY_MAX_DROP)
        a["_specific"] = bool(a["rank"] >= 1 and g1 > q95.get(a["rank"], np.inf))
        if a["rank"] >= 1 and a["_safe"] and a["_specific"]:
            cands.append((g1, a))
    if not cands:
        return next(a for a in actions if a["name"] == "identity"), {"n_candidates": 0, "rule_hash": M.rule_hash()}
    return max(cands, key=lambda t: t[0])[1], {"n_candidates": len(cands), "rule_hash": M.rule_hash()}


def utility(action, Zs, ys, Zt_q, yt_q, session_q, seed=0):
    Zs_a = action["apply_source"](Zs)
    def gain(Zq, yq):
        return _bacc(Zs_a, ys, action["apply_target_query"](Zq), yq, seed) - _bacc(Zs, ys, Zq, yq, seed)
    pooled = gain(Zt_q, yt_q); sess = np.asarray(session_q); per = []
    for s in np.unique(sess):
        m = sess == s
        if m.sum() >= 4 and len(np.unique(yt_q[m])) >= 2:
            per.append(gain(Zt_q[m], yt_q[m]))
    return (float(np.mean(per)) if per else float(pooled)), float(pooled)


# ============================================================ per-fold audit (whitened, full trail)
def audit_fold(feat, seed=0, family="cond", max_rank=10, smoke=False, phase="primary", n_random_per_rank=50,
               config_hash="", git_sha=""):
    from tos_cmi.eeg.relaxation_ladder import _dense
    Zs = np.asarray(feat["Z_source"], float); ys = np.asarray(feat["y_source"]).astype(int)
    ds = _dense(feat["subj_source"]); Zt = np.asarray(feat["Z_target"], float); yt = np.asarray(feat["y_target"]).astype(int)
    st = feat["session_target"]; n_cls = int(feat["n_cls"]); classes = sorted(np.unique(ys).tolist())
    cal, qry, sinfo = session_split(st, yt, seed)
    Xcal = Zt[cal]; Xq, yq, sq = Zt[qry], yt[qry], np.asarray(st)[qry]
    W = M.source_whitener(Zs)
    Zs_w, Xcal_w = M.to_whitened(Zs, W), M.to_whitened(Xcal, W)
    row_w, null_w = M.whitened_head_rowspace(Zs_w, ys, seed,
                                             W_stored=feat.get("head_W"), A_inv=W["A_inv"] if feat.get("head_W") is not None else None)
    B_cond_w = M.whitened_cond_basis(Zs_w, ys, ds, max_rank=max_rank)
    B_contested = M.project_basis(B_cond_w, row_w)               # PRIMARY (task-used directions)
    B_free = M.project_basis(B_cond_w, null_w)                   # sanitation control (head-null), NOT selectable
    if B_contested.shape[0] == 0:
        B_contested = B_cond_w[: min(1, B_cond_w.shape[0])]      # degenerate guard
    d_white = -M.to_whitened(Xcal.mean(0)[None, :], W)[0]        # A_s(mu_s - mu_tcal)
    head_w = LogisticRegression(max_iter=300).fit(Zs_w, ys)
    ev = np.sort(np.linalg.eigvalsh(np.cov(Zs_w.T) + 1e-12 * np.eye(Zs_w.shape[1])))[::-1]; ev = np.clip(ev, 1e-12, None)
    src_contrasts_w = {(a, b): Zs_w[ys == classes[a]].mean(0) - Zs_w[ys == classes[b]].mean(0)
                       for a in range(len(classes)) for b in range(a + 1, len(classes))}
    p_src = np.bincount(ys, minlength=n_cls).astype(float); p_src /= p_src.sum()
    ctx = dict(Zs_w=Zs_w, Xcal_w=Xcal_w, d_white=d_white, head_w=head_w, n_cls=n_cls, classes=classes,
               src_contrasts_w=src_contrasts_w, p_source_prior=p_src, log_kappa_identity=float(np.log(ev[0] / ev[-1])))
    trace = {"metric": "source_ledoitwolf_whitened", "primary_basis": "cond_contested", "rule_hash": M.rule_hash(),
             "targetx_scores_use": "T_cal_X_only", "query_x_used_for_selection": False,
             "query_y_used_for_selection": False, "query_x_used_for_outcome": True, "query_y_used_for_outcome": True,
             "target_greedy_in_action_set": False, "random_controls_selectable": False,
             "fallback_used": sinfo["fallback_used"], "phase": phase,
             "whitening_hash": W["whitening_hash"], "cov_condition_number": W["cond"],
             "contested_rank": int(B_contested.shape[0]), "free_rank": int(B_free.shape[0]),
             "full_cond_rank": int(B_cond_w.shape[0])}
    actions = build_actions(B_contested, W, Zs, ys, ds, Xcal, seed=seed, smoke=smoke, n_random_per_rank=n_random_per_rank)
    # constrained hindsight (same contested span, rank<=3, source-task-safe) + unconstrained ceiling
    S_hc = _select_subset(Zs_w, ys, Xcal_w, yt[cal], B_contested, "greedy", PRIMARY_MAX_RANK, seed)
    hind_c = _act("hindsight_constrained", "oracle", len(S_hc), M._orthonormal(B_contested[S_hc]) if S_hc else np.zeros((0, Zs.shape[1])), W,
                  eligible=False, basis_label="cond_contested", basis_indices=list(S_hc), basis_hash=M._hash(B_contested))
    S_hu = _select_subset(Zs_w, ys, Xcal_w, yt[cal], B_cond_w, "greedy", max_rank, seed)
    hind_u = _act("hindsight_unconstrained", "oracle", len(S_hu), M._orthonormal(B_cond_w[S_hu]) if S_hu else np.zeros((0, Zs.shape[1])), W,
                  eligible=False, basis_label="cond_full", basis_indices=list(S_hu), basis_hash=M._hash(B_cond_w))
    actions += [hind_c, hind_u]
    sel, diag = g1_select(actions, ctx, Zs, ys, ds, seed)
    obs = ["G1"] if phase == "primary" else (["G1"] + SECONDARY)
    rows = []
    for a in actions:
        sc = {ob: (0.0 if a["rank"] == 0 else float(OBSERVABLES[ob](a["U"], ctx))) for ob in obs} if a["U"] is not None else {ob: None for ob in obs}
        macro, pooled = utility(a, Zs, ys, Xq, yq, sq, seed)
        rows.append(dict(action=a["name"], kind=a["kind"], rank=a["rank"], eligible=a["eligible"],
                         basis_label=a["basis_label"], basis_family=family, basis_hash=a["basis_hash"],
                         projector_hash=a["projector_hash"], basis_indices=a.get("basis_indices"),
                         G1=sc.get("G1"), scores=sc, source_task_drop=a.get("_drop"), random_q95_same_rank=a.get("_q95"),
                         safe_gate_pass=a.get("_safe"), specificity_gate_pass=a.get("_specific"),
                         utility_macro=macro, utility_pooled=pooled, config_hash=config_hash, git_sha=git_sha,
                         rule_hash=M.rule_hash()))
    def _u(name):
        r = next((rw for rw in rows if rw["action"] == name), None); return r["utility_macro"] if r else float("nan")
    ksel = sel["rank"]
    rand_sel = [rw["utility_macro"] for rw in rows if rw["kind"] == "random" and rw["rank"] == ksel] if ksel >= 1 else []
    rand_by_rank = {k: float(np.mean([rw["utility_macro"] for rw in rows if rw["kind"] == "random" and rw["rank"] == k]))
                    for k in (1, 2, 3)}
    fold = dict(heldout_subject=str(feat["heldout_subject"]), seed=int(feat.get("seed", seed)),
                dataset=feat.get("dataset", ""), session_info=sinfo, firewall=trace,
                n_actions=len(rows), n_informed=sum(rw["kind"] == "informed" for rw in rows),
                n_random=sum(rw["kind"] == "random" for rw in rows),
                selected_action=sel["name"], selected_rank=ksel, selected_basis_indices=sel.get("basis_indices"),
                selected_basis_hash=sel["basis_hash"], rule_hash=M.rule_hash(),
                delta_tx=_u(sel["name"]),
                delta_tx_pooled=next((rw["utility_pooled"] for rw in rows if rw["action"] == sel["name"]), float("nan")),
                delta_source_greedy=_u("srcgreedy_standalone"), delta_whitening=_u("whitening"),
                delta_mean_centering=_u("mean_centering"),
                delta_random_selected_rank=(float(np.mean(rand_sel)) if rand_sel else 0.0),
                delta_random_rank1=rand_by_rank[1], delta_random_rank2=rand_by_rank[2], delta_random_rank3=rand_by_rank[3],
                delta_hindsight_constrained=_u("hindsight_constrained"),
                delta_hindsight_unconstrained=_u("hindsight_unconstrained"),
                n_candidates=diag["n_candidates"])
    return {"fold": fold, "rows": rows}
