"""Readout Prior Decomposition per-cell runner (CPU, env c84c). Decisive question: is the anchoring win a genuine
source-head PRIOR (H2 source-centered MAP > H1 hardened zero-centered ridge) or a weak-baseline / optimisation-path /
budget-mismatch artifact? Primary = native Z0, arms H0/H1/H1-W/H2/H3/H4 with FIXED-tau budget-matched source-only tau.
Secondary = high-powered matched-random subspace specificity (Z0/ZI/ZR) at budgets {1,8,Full} for H1/H2. Plus external
headroom diagnostics. Firewall: target QUERY only in the final utility; cal Y only adapts heads; tau + gate source-only.
NO re-inference. Manuscript FROZEN; only the owner stops/redirects a line.

  python -m scripts.run_readout_prior_decomposition --cell-index 0 --out-dir results/cmi_trace_readout_prior
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from sklearn.metrics import balanced_accuracy_score
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump
from tos_cmi.eval import targetx_metric as TM
from tos_cmi.eval import readout_prior as RP
from tos_cmi.eval.readout_calibration import standardize, _std, fit_head, fit_biastemp, session_macro_bacc
from tos_cmi.eval.mechanism_subspace import _del, build_ambient_random_dictionaries, cell_seed
from tos_cmi.eval.targetx_observability import session_split

_HERE = "/home/infres/yinwang/CMI_AAAI_readout_prior/tos_cmi/results/tos_cmi_eeg_frozen"
FEAT_ROOTS = {
    "BNCI2014_001": "/home/infres/yinwang/CMI_AAAI_cmitrace/tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_EEGNet_LOSO",
    "BNCI2015_001": "/home/infres/yinwang/CMI_AAAI_cmitrace/tos_cmi/results/tos_cmi_eeg_frozen/BNCI2015_001_EEGNet_LOSO",
    "Lee2019_MI":   "/home/infres/yinwang/CMI_AAAI_tos/tos_cmi/results/tos_cmi_eeg_frozen/Lee2019_MI_EEGNet_LOSO",
    "BNCI2014_004": "/home/infres/yinwang/CMI_AAAI_tos/tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_004_EEGNet_LOSO",
    # untouched natural-multi-session MI lockboxes (dumped in THIS worktree)
    "Stieger2021":  f"{_HERE}/Stieger2021_EEGNet_LOSO",
    "Shin2017A":    f"{_HERE}/Shin2017A_EEGNet_LOSO",
}
DATASETS = ["BNCI2014_001", "BNCI2015_001", "Lee2019_MI", "BNCI2014_004", "Stieger2021", "Shin2017A"]
BUDGETS = [1, 2, 4, 8, 16, 32, "Full"]
SPEC_BUDGETS = [1, 8, "Full"]           # reduced-resolution specificity (high-powered control is expensive)
DICT_RANK = 8
SRC_RETENTION_TOL = 0.03
N_TAU_DRAWS = 50                         # P0.1: tau-selection draws = target draws (was 10)
MAX_TAU_PSEUDO = 12                      # cap pseudo-subjects used for tau selection (subsample if more)


def enumerate_cells():
    return [(ds, p) for ds in DATASETS for p in sorted(glob.glob(f"{FEAT_ROOTS[ds]}/sub*_erm_lam0_seed*.npz"))]


def _draws(ds, subj, sd, ycal, k, nd):
    if k == "Full":
        return [np.arange(len(ycal))]
    return [RP.balanced_draw(ycal, k, np.random.default_rng(cell_seed(ds, "EEGNet", subj, sd, f"draw{k}", i))) for i in range(nd)]


def _headroom(Zs_w, ys, dsub, Xcal_w, ycal, Xq_w, yq, sq, C):
    """External-headroom mechanism diagnostics per cell."""
    mu, sd = standardize(Zs_w); Xs, Xc, Xq = _std(Zs_w, mu, sd), _std(Xcal_w, mu, sd), _std(Xq_w, mu, sd)
    Ws, bs = fit_head(Xs, ys, C)
    fro_bacc = session_macro_bacc(Ws, bs, Xq, yq, sq)
    # full-cal target head (oracle-ish headroom)
    Wt, bt = fit_head(Xc, ycal, C); full_bacc = session_macro_bacc(Wt, bt, Xq, yq, sq)
    ang = float(np.degrees(np.arccos(np.clip(np.sum(Ws * Wt) / (np.linalg.norm(Ws) * np.linalg.norm(Wt) + 1e-12), -1, 1))))
    from sklearn.metrics import log_loss
    fro_nll = float(log_loss(yq, RP._softmax(Xq @ Ws.T + bs), labels=list(range(C))))
    # calibration Hessian condition number (Fisher proxy): cov of standardized cal features
    cov = np.cov(Xc.T) + 1e-9 * np.eye(Xc.shape[1]); ev = np.linalg.eigvalsh(cov)
    cond = float(ev[-1] / max(ev[0], 1e-12))
    prior_shift = float(np.abs(np.bincount(ycal, minlength=C) / len(ycal) - np.bincount(yq, minlength=C) / len(yq)).sum())
    mean_drift = float(np.linalg.norm(Xc.mean(0) - Xq.mean(0)))
    return dict(frozen_query_bacc=fro_bacc, frozen_query_nll=fro_nll, fullcal_query_bacc=full_bacc,
                fullcal_gain=full_bacc - fro_bacc, angle_Ws_Wt_deg=ang, bias_displacement=float(np.linalg.norm(bs - bt)),
                direction_displacement=float(np.linalg.norm(Ws - Wt)), cal_cov_condition=cond,
                class_prior_shift=prior_shift, cal_query_mean_drift=mean_drift)


def _session_split_policy(ds, session_target, y_target, seed):
    """Dataset-aware cal/query split. Stieger2021 uses the PM-FROZEN protocol (cal=session1,
    primary query=sessions 2-7 session-macro, long-horizon=8-11 reported SEPARATELY, not in the
    primary verdict). All other datasets keep the pre-registered generic split unchanged (NO amendment).
    Returns (cal_mask, primary_query_mask, long_horizon_mask, info)."""
    st = np.asarray(session_target).astype(str)
    if ds == "Stieger2021":
        present = set(st.tolist())
        # cal = session "1" (all 62 subjects have it, verified); fall back to the earliest PRESENT session if a
        # subject were ever missing "1", so cal is never empty. cal_sessions reports the ACTUAL session used.
        cal_sess = "1" if "1" in present else sorted(present, key=int)[0]
        cal = np.isin(st, [cal_sess])
        q_primary = [s for s in ["2", "3", "4", "5", "6", "7"] if s != cal_sess]   # never overlap cal
        q_long = [s for s in ["8", "9", "10", "11"] if s != cal_sess]              # long-horizon (11-session subj)
        qry = np.isin(st, q_primary); lh = np.isin(st, q_long)
        info = dict(cal_sessions=[cal_sess], query_sessions=[q for q in q_primary if q in present],
                    long_horizon_sessions=[q for q in q_long if q in present],
                    n_cal=int(cal.sum()), n_query=int(qry.sum()), n_long_horizon=int(lh.sum()),
                    fallback_used=bool(cal_sess != "1"))
        return cal, qry, lh, info
    cal, qry, info = session_split(session_target, y_target, seed)  # generic (existing datasets + Shin)
    info["long_horizon_sessions"] = []
    return cal, qry, np.zeros(len(st), bool), info


def run_cell(ds, path, n_random=50, n_draws=50, smoke=False):
    f = feat_from_tos_dump(path)
    subj = str(f.get("heldout_subject")); sd = int(f.get("seed", -1))
    if "session_target" not in f:
        return dict(dataset=ds, subject=subj, seed=sd, status="skipped", reason="NO_SESSION_AXIS")
    Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
    dsub = np.asarray(f["subj_source"]).astype(int); Zt = np.asarray(f["Z_target"], float)
    yt = np.asarray(f["y_target"]).astype(int); C = int(f.get("n_cls", len(np.unique(ys))))
    sess_s = np.asarray(f["session_source"]).astype(str)
    W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W); D = Zs.shape[1]
    cal, qry, lh, sinfo = _session_split_policy(ds, f["session_target"], yt, sd)
    st_all = np.asarray(f["session_target"])
    Xcal_w, ycal = TM.to_whitened(Zt[cal], W), yt[cal]
    Xq_w, yq, sq = TM.to_whitened(Zt[qry], W), yt[qry], st_all[qry]
    # long-horizon (Stieger sessions 8-11): SEPARATE readout, not in the primary verdict; only if ≥2 classes present
    has_lh = bool(lh.sum() > 0 and len(np.unique(yt[lh])) >= 2)
    Xlh_w, ylh, slh = (TM.to_whitened(Zt[lh], W), yt[lh], st_all[lh]) if has_lh else (None, np.array([]), None)
    nd = 5 if smoke else n_draws; ntd = 3 if smoke else N_TAU_DRAWS
    draws_by_k = {k: _draws(ds, subj, sd, ycal, k, nd) for k in BUDGETS}

    # native source-standardised space + source head
    mu, sd_ = standardize(Zs_w); Xs_std = _std(Zs_w, mu, sd_)
    Ws, bs = fit_head(Xs_std, ys, C)
    Xcal_std = _std(Xcal_w, mu, sd_); Xq_std = _std(Xq_w, mu, sd_)
    Xlh_std = _std(Xlh_w, mu, sd_) if has_lh else None

    # budget-matched SOURCE-ONLY tau selection + gate (tau selection operates on the native standardised source)
    def rng_fn(dd, di): return np.random.default_rng(cell_seed(ds, "EEGNet", subj, sd, "tau", int(dd), di))
    tau0, taus, gate, tau_curves = {}, {}, {}, {"tau0": {}, "taus": {}}
    for k in BUDGETS:
        t0, d0 = RP.select_tau_budget_matched(Xs_std, ys, dsub, sess_s, C, k, ntd, source_centered=False, rng_fn=rng_fn)
        ts, ds_ = RP.select_tau_budget_matched(Xs_std, ys, dsub, sess_s, C, k, ntd, source_centered=True, rng_fn=rng_fn)
        tau0[str(k)] = float(t0); taus[str(k)] = float(ts)
        tau_curves["tau0"][str(k)] = d0.get("mean_ce", d0); tau_curves["taus"][str(k)] = ds_.get("mean_ce", ds_)  # P0.5 full tau-curve
        g, _ = RP.source_gate(Xs_std, ys, dsub, sess_s, C, k, ts, ntd, rng_fn)
        gate[str(k)] = int(g)

    draws_std = {k: draws_by_k[k] for k in BUDGETS}
    nat_curve, lh_curve, init_pdiff, solver_audit = _arm_curve_native(
        Xs_std, ys, Xq_std, yq, sq, C, Ws, bs, Xcal_std, ycal, draws_std, tau0, taus, gate,
        Xq2_std=Xlh_std, yq2=ylh, sq2=slh)
    if solver_audit["success_rate"] < 0.99:                    # P0.2: fail loud on solver non-convergence
        return dict(dataset=ds, subject=subj, seed=sd, status="failed_solver", solver_audit=solver_audit)

    def _endpoints(curve):
        e = {}
        for k in BUDGETS:
            kk = str(k); c = curve[kk]
            e[kk] = dict(U_H0=c["frozen"], U_H1=c["ridge"], U_H1W=c["ridgeW"], U_H2=c["map"], U_H3=c["bias"], U_H4=c["gate"],
                         dU_center=c["map"] - c["ridge"],                       # policy: H2@taus - H1@tau0 (mixes center + strength)
                         dU_center_t0=c["map_t0"] - c["ridge"],                 # matched-tau: H2@tau0 - H1@tau0 (pure center)
                         dU_center_ts=c["map"] - c["ridge_ts"],                 # matched-tau: H2@taus - H1@taus (pure center)
                         dU_MAP_frozen=c["map"] - c["frozen"], dU_gate_frozen=c["gate"] - c["frozen"], dU_gate_map=c["gate"] - c["map"],
                         dU_init_bacc=c["ridgeW"] - c["ridge"], U_H2_minus_H3=c["map"] - c["bias"],
                         tau0=tau0[kk], taus=taus[kk], gate=gate[kk])
        return e
    # endpoints (native): matched-tau center contrasts isolate CENTER from shrinkage-strength
    ep = _endpoints(nat_curve)
    lh_ep = _endpoints(lh_curve) if lh_curve is not None else None

    # high-powered matched-random subspace specificity (reduced budgets, H1+H2), source-VALIDATION matching metric
    spec = _specificity(Zs_w, ys, dsub, Xcal_w, ycal, Xq_w, yq, sq, C, D, mu, sd_, Ws, bs, tau0, taus, gate,
                        draws_std, ds, subj, sd, n_random, smoke)

    hr = _headroom(Zs_w, ys, dsub, Xcal_w, ycal, Xq_w, yq, sq, C)
    long_horizon = None
    if lh_ep is not None:                                       # Stieger 8-11: SEPARATE, excluded from primary verdict
        long_horizon = dict(sessions=sinfo.get("long_horizon_sessions", []), n_query=int(lh.sum()),
                            endpoints=lh_ep, note="SEPARATE long-horizon readout; NOT in primary L-A..L-D verdict")
    return dict(dataset=ds, subject=subj, seed=sd, status="ok", C=C, n_cal=int(cal.sum()), n_query=int(qry.sum()),
                cal_session=sinfo["cal_sessions"], query_sessions=sinfo["query_sessions"], n_draws=nd, n_tau_draws=ntd,
                init_param_diff=float(init_pdiff), solver_audit=solver_audit, tau_curves=tau_curves,
                endpoints=ep, specificity=spec, headroom=hr, tau0=tau0, taus=taus, gate=gate, long_horizon=long_horizon,
                firewall=dict(source_only_construction=True, tau_source_only=True, gate_source_only=True,
                              Ycal_used_for_head_adapt=True, Yquery_used_for_selection=False, Yquery_used_for_outcome=True))


def _arm_curve_native(Xs_std, ys, Xq_std, yq, sq, C, Ws, bs, Xcal_std, ycal, draws_std, tau0, taus, gate,
                      Xq2_std=None, yq2=None, sq2=None):
    """Native arm curve with MATCHED-tau contrasts (ridge_ts = H1@taus, map_t0 = H2@tau0) so the CENTER effect can be
    isolated from the shrinkage-strength effect, + solver-audit accumulation (P0.2) + parameter-level init diff (P0.3
    already applied to gate). If a SECOND query (Xq2_std,yq2,sq2) is given (Stieger long-horizon sessions 8-11), the SAME
    fitted heads are ALSO evaluated on it (heads fit ONCE — no redundant re-fit), returning a second curve."""
    which = ("frozen", "ridge", "ridgeW", "map", "bias", "gate", "ridge_ts", "map_t0")
    has2 = Xq2_std is not None and len(yq2) > 0
    src_init = np.concatenate([Ws.ravel(), bs])
    u_frozen = session_macro_bacc(Ws, bs, Xq_std, yq, sq)
    u_frozen2 = session_macro_bacc(Ws, bs, Xq2_std, yq2, sq2) if has2 else 0.0
    curve, curve2, init_pdiff = {}, {}, []
    audit = dict(n_fit=0, n_failed=0, max_grad_norm=0.0, grad_norms=[])

    def _fit(xc, yc, k, anchorW, anchorb, tau, init=None):
        W, b, au = RP.fit_ridge_map(xc, yc, C, anchorW, anchorb, tau, init=init)
        audit["n_fit"] += 1; audit["grad_norms"].append(au["grad_norm"]); audit["max_grad_norm"] = max(audit["max_grad_norm"], au["grad_norm"])
        if not au["success"]:
            audit["n_failed"] += 1
        return W, b
    def _ev(W, b, on2):    # session-macro bAcc on the primary (on2=False) or long-horizon (on2=True) query
        return session_macro_bacc(W, b, Xq2_std, yq2, sq2) if on2 else session_macro_bacc(W, b, Xq_std, yq, sq)
    for k in draws_std:
        acc = {a: [] for a in which}; acc2 = {a: [] for a in which}
        for di in draws_std[k]:
            xc, yc = Xcal_std[di], ycal[di]
            if len(np.unique(yc)) < 2:
                for a in which: acc[a].append(u_frozen); acc2[a].append(u_frozen2)
                continue
            acc["frozen"].append(u_frozen); acc2["frozen"].append(u_frozen2)
            W1, b1 = _fit(xc, yc, k, None, None, tau0[str(k)])                                  # H1 @ tau0
            W1w, b1w = _fit(xc, yc, k, None, None, tau0[str(k)], init=src_init)                 # H1-W @ tau0 (init audit)
            Wm, bm = _fit(xc, yc, k, Ws, bs, taus[str(k)])                                       # H2 @ taus
            W1s, b1s = _fit(xc, yc, k, None, None, taus[str(k)])                                 # H1 @ taus (matched-tau)
            Wm0, bm0 = _fit(xc, yc, k, Ws, bs, tau0[str(k)])                                     # H2 @ tau0 (matched-tau)
            Wt, bt = fit_biastemp(xc, yc, C, Ws)
            init_pdiff.append(float(np.linalg.norm(W1 - W1w) + np.linalg.norm(b1 - b1w)))
            heads = dict(ridge=(W1, b1), ridgeW=(W1w, b1w), map=(Wm, bm), ridge_ts=(W1s, b1s), map_t0=(Wm0, bm0), bias=(Wt, bt))
            for a, (W, b) in heads.items():
                acc[a].append(_ev(W, b, False))
                if has2: acc2[a].append(_ev(W, b, True))
            acc["gate"].append(_ev(Wm, bm, False) if gate[str(k)] else u_frozen)
            if has2: acc2["gate"].append(_ev(Wm, bm, True) if gate[str(k)] else u_frozen2)
        curve[str(k)] = {a: float(np.mean(acc[a])) for a in which}
        if has2: curve2[str(k)] = {a: float(np.mean(acc2[a])) for a in which}
    audit["median_grad_norm"] = float(np.median(audit["grad_norms"])) if audit["grad_norms"] else 0.0
    audit["success_rate"] = float(1.0 - audit["n_failed"] / max(1, audit["n_fit"])); audit.pop("grad_norms")
    return curve, (curve2 if has2 else None), (float(np.mean(init_pdiff)) if init_pdiff else 0.0), audit


def _src_val_bacc(Zs_wd, ys, dsub, C):
    """Source-VALIDATION bAcc of a (deleted) representation: leave-one-source-subject-out mean bAcc (cheap subset)."""
    subs = np.unique(dsub); accs = []
    for v in subs:                                    # P0.4: ALL source subjects (full source-LOSO)
        tr, te = dsub != v, dsub == v
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0: continue
        mu, sd = standardize(Zs_wd[tr]); Wv, bv = fit_head(_std(Zs_wd[tr], mu, sd), ys[tr], C)
        accs.append(balanced_accuracy_score(ys[te], (_std(Zs_wd[te], mu, sd) @ Wv.T + bv).argmax(1)))
    return float(np.mean(accs)) if accs else float("nan")


def _rep_gain(Zs_wd, ys, Xcal_wd, ycal, Zq_wd, yq, sq, C, tau0, taus, gate, draws_std, arms=("ridge", "map")):
    """G_h(rep,k) = U_arm - U_frozen for a (deleted) representation, at SPEC_BUDGETS."""
    mu, sd = standardize(Zs_wd); Xs = _std(Zs_wd, mu, sd); Ws, bs = fit_head(Xs, ys, C)
    Xc = _std(Xcal_wd, mu, sd); Xq = _std(Zq_wd, mu, sd); u0 = session_macro_bacc(Ws, bs, Xq, yq, sq)
    out = {}
    for k in SPEC_BUDGETS:
        g = {a: [] for a in arms}
        for di in draws_std[k]:
            xc, yc = Xc[di], ycal[di]
            if len(np.unique(yc)) < 2:
                for a in arms: g[a].append(0.0)
                continue
            if "ridge" in arms:
                W1, b1, _ = RP.fit_ridge_map(xc, yc, C, None, None, tau0[str(k)]); g["ridge"].append(session_macro_bacc(W1, b1, Xq, yq, sq) - u0)
            if "map" in arms:
                Wm, bm, _ = RP.fit_ridge_map(xc, yc, C, Ws, bs, taus[str(k)]); g["map"].append(session_macro_bacc(Wm, bm, Xq, yq, sq) - u0)
        out[str(k)] = {a: float(np.mean(g[a])) for a in arms}
    return out


SPEC_MAX_SOURCE = 20000                  # skip the matched-random specificity above this source size (see below)


def _specificity(Zs_w, ys, dsub, Xcal_w, ycal, Xq_w, yq, sq, C, D, mu, sd_, Ws, bs, tau0, taus, gate, draws_std, ds, subj, sd, n_random, smoke):
    # The specificity control samples up to 5000 random projectors, each scored by a FULL source-LOSO
    # (_src_val_bacc: one head-fit per source subject). On a large source (Stieger ~225k trials, 60+ subjects)
    # that is O(days) per cell. It is a SECONDARY subspace-causality diagnostic and is NOT a verdict input
    # (routing uses only the primary AULC endpoints), so we SKIP it above SPEC_MAX_SOURCE and keep it where cheap
    # (Shin + the 4 dev/context datasets). The subspace question is already answered (NULL) on those 5 datasets.
    if len(ys) > SPEC_MAX_SOURCE:
        return dict(status="skipped_large_source", n_source=int(len(ys)), spec_max_source=SPEC_MAX_SOURCE,
                    note="matched-random specificity = O(days) at this source size; NOT a verdict input (routing "
                         "uses only primary AULC endpoints); subspace-causal question already NULL on the 5 smaller datasets")
    B = TM.whitened_cond_basis(Zs_w, ys, dsub, max_rank=DICT_RANK); r = B.shape[0]
    if r == 0:
        return dict(status="EMPTY_B_COND")
    inf_srcval = _src_val_bacc(_del(Zs_w, B), ys, dsub, C)
    gh_informed = _rep_gain(_del(Zs_w, B), ys, _del(Xcal_w, B), ycal, _del(Xq_w, B), yq, sq, C, tau0, taus, gate, draws_std)
    need = 8 if smoke else 50; cap = 200 if smoke else 5000
    matched = []; tried = 0
    while len(matched) < need and tried < cap:
        Q = build_ambient_random_dictionaries(D, r, 1, cell_seed(ds, "EEGNet", subj, sd, "rnd", tried))[0]
        if abs(_src_val_bacc(_del(Zs_w, Q), ys, dsub, C) - inf_srcval) <= SRC_RETENTION_TOL:
            matched.append(Q)
        tried += 1
    rand_gains = [_rep_gain(_del(Zs_w, Q), ys, _del(Xcal_w, Q), ycal, _del(Xq_w, Q), yq, sq, C, tau0, taus, gate, draws_std) for Q in matched]
    dgh = {}
    for k in SPEC_BUDGETS:
        for a in ("ridge", "map"):
            er = float(np.mean([rg[str(k)][a] for rg in rand_gains])) if rand_gains else float("nan")
            dgh[f"{a}_k{k}"] = dict(informed=gh_informed[str(k)][a], random_mean=er, specific=gh_informed[str(k)][a] - er)
    return dict(status="ok", rank=int(r), n_matched=len(matched), n_tried=tried, informed_srcval=inf_srcval, dGh_specific=dgh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-index", type=int); ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--out-dir", default="results/cmi_trace_readout_prior")
    ap.add_argument("--n-random", type=int, default=50); ap.add_argument("--n-draws", type=int, default=50); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cells = enumerate_cells()
    if a.list_cells:
        [print(f"{i}\t{ds}\t{Path(p).name}") for i, (ds, p) in enumerate(cells)]; print(f"# {len(cells)}"); return
    ds, path = cells[a.cell_index]
    print(f"[ro-prior] cell {a.cell_index} {ds} {Path(path).name}", flush=True)
    row = run_cell(ds, path, n_random=a.n_random, n_draws=a.n_draws, smoke=a.smoke)
    outd = Path(a.out_dir); (outd / "cells").mkdir(parents=True, exist_ok=True)
    stem = f"cell_{a.cell_index:03d}_{ds}_sub{row.get('subject')}_seed{row.get('seed')}"
    (outd / "cells" / f"{stem}.json").write_text(json.dumps(row, indent=2, default=float))
    (outd / "cells" / f"{stem}.done").write_text(row.get("status", "?") + "\n")
    if row.get("status") == "ok":
        e = row["endpoints"]
        print(f"  init_param_diff={row['init_param_diff']:.2e} (init-invariance) | spec matched={row['specificity'].get('n_matched')}/{row['specificity'].get('n_tried')}", flush=True)
        print("  dU_center(H2-H1): " + " ".join(f"k{k}={e[str(k)]['dU_center']:+.3f}" for k in BUDGETS), flush=True)
        print("  dU_MAP-frozen   : " + " ".join(f"k{k}={e[str(k)]['dU_MAP_frozen']:+.3f}" for k in BUDGETS), flush=True)
        print("  dU_gate-frozen  : " + " ".join(f"k{k}={e[str(k)]['dU_gate_frozen']:+.3f}" for k in BUDGETS), flush=True)


if __name__ == "__main__":
    main()
