"""
CSC Route B3-P2.4 — calibrated paired certifier (`pc_centered_calibrated`). Method LOCKED (pc basis,
centered +-0.5 coding, rank 3, C 0.5, min_confirm_pairs 20); P2.4 adds ONLY safety + calibration, per the
reviewer:

  1. PAIR-INTEGRITY guard (batch-level): pair_integrity = n_complete_pairs / n_subjects_total computed on
     the WHOLE target batch (not just queried/complete subjects). CONCEPT_CONFIRMED requires
     pair_integrity >= PAIR_INTEGRITY_MIN; else INVALID_PAIR_STRUCTURE. -> closes the missing_pair leak.
  2. EFFECTIVE-AUDIT guard: an eligible complete pair = subject with BOTH conditions AND both cells with
     >= MIN_EPOCHS_PER_CONDITION epochs. Confirmation needs n_eligible_complete_pairs >= min_confirm_pairs.
     -> closes the unequal_epochs_extreme leak.
  3. CLASS-BALANCED subject->condition->class->epoch loss + weights (w ∝ 1/(|C_s||Y_sc|n_scy), sum=#subj)
     + per-condition class-coverage gate. -> strips label/prior composition from the concept evidence.
  4. CROSS-FITTED paired T (subject-grouped folds; standardise + PC basis fit on TRAIN, evaluated on
     held-out; parametric-bootstrap null rerun through the same cross-fit pipeline). -> removes in-sample
     overfit creep. n_folds fixed (no sweep).
  5. (B3-P2.4c) FIXED-MARGIN h0 BOOTSTRAP null (sample_h0_fixed_condition_margins): preserves per-condition
     class margins so label/prior COMPOSITION cannot inflate T. + STUDENTIZED SUBJECT-CONSISTENCY GATE:
     CONCEPT_CONFIRMED requires fixed-margin mean-T p<=alpha AND studentized Z=mean(delta_s)/se(delta_s)
     p<=alpha AND LCB(delta_s)>0 -- so a few subjects' noise improvements cannot confirm.
  6. (B3-P2.4d) CROSS-BUDGET ALPHA-SPENDING: the certifier may confirm at n_decision_budgets positive
     budgets (m=20,30), so alpha_budget = alpha_family/n_decision_budgets (=0.025) is applied to BOTH the
     mean-T and studentized p-gates AND the LCB is taken at 1-alpha_budget (97.5%). Deterministic
     Bonferroni; no sweep. Fixes the per-budget m=30 control edge + random_label borderline.

DEVELOPMENT only; NO freeze/confirmatory; NO real EEG. calibration_version below is logged.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression

from .paired_conditional_test import condition_code
from .paired_certifier import (CONCEPT_CONFIRMED, NO_CONCEPT_EVIDENCE, NEED_MORE_LABELS,
                               INVALID_PAIR, UNIDENTIFIABLE)

CALIBRATION_VERSION = "p24d_cross_budget_alpha_spending_studentized_fixed_margin"
PAIR_INTEGRITY_MIN = 0.95
MIN_EPOCHS_PER_CONDITION = 8
N_FOLDS = 3


def pair_integrity_stats(D, groups):
    g = np.asarray(groups); D = np.asarray(D)
    subs = np.unique(g)
    complete = [s for s in subs if len(np.unique(D[g == s])) >= 2]
    n_total = len(subs); n_complete = len(complete)
    integ = n_complete / n_total if n_total else 0.0
    return dict(n_subjects_total=int(n_total), n_complete_pairs=int(n_complete),
                pair_integrity=float(integ), missing_condition_fraction=float(1.0 - integ))


def eligible_complete_pairs(D, groups, min_epochs=MIN_EPOCHS_PER_CONDITION):
    """Subjects with BOTH conditions and each condition cell having >= min_epochs epochs."""
    g = np.asarray(groups); D = np.asarray(D)
    out = []
    for s in np.unique(g):
        sm = g == s; conds = np.unique(D[sm])
        if len(conds) >= 2 and all(int((sm & (D == c)).sum()) >= min_epochs for c in conds):
            out.append(int(s))
    return out


def class_balanced_weights(Y, D, groups):
    """w_sce ∝ 1/(|C_s| * |Y_sc| * n_scy); sum over a subject == 1 -> total == #subjects (so the sklearn
    L2 scale is preserved AND frequent classes/conditions cannot dominate the fit)."""
    Y = np.asarray(Y); D = np.asarray(D); g = np.asarray(groups); w = np.zeros(len(Y))
    for s in np.unique(g):
        sm = g == s; conds = np.unique(D[sm]); Cs = len(conds)
        for c in conds:
            cm = sm & (D == c); ys = np.unique(Y[cm]); Ysc = len(ys)
            for y in ys:
                ym = cm & (Y == y); n = int(ym.sum())
                if n:
                    w[ym] = 1.0 / (Cs * Ysc * n)
    return w


def cb_subject_losses(nll, Y, D, groups):
    """Per-subject class-balanced nested loss: subject -> mean over conditions -> mean over classes ->
    mean over that cell's epochs. Returns dict subject -> loss."""
    nll = np.asarray(nll, float); Y = np.asarray(Y); D = np.asarray(D); g = np.asarray(groups)
    out = {}
    for s in np.unique(g):
        sm = g == s; cond_losses = []
        for c in np.unique(D[sm]):
            cm = sm & (D == c); ys = np.unique(Y[cm])
            cond_losses.append(np.mean([nll[cm & (Y == y)].mean() for y in ys]))
        out[int(s)] = float(np.mean(cond_losses))
    return out


def per_condition_classes(Y, D):
    Y = np.asarray(Y); D = np.asarray(D)
    return {int(c): sorted(int(v) for v in np.unique(Y[D == c])) for c in np.unique(D)}


def _features(Z, D, coding, mu, sd, Vr):
    Zs = (Z - mu) / sd
    cv = condition_code(D, coding)[:, None]
    return np.hstack([Zs, cv]), np.hstack([Zs, cv, cv * (Zs @ Vr)])


def _nll(clf, X, y, cl):
    p = np.clip(clf.predict_proba(X), 1e-12, 1.0)
    full = np.full((len(X), len(cl)), 1e-12)
    cli = {c: j for j, c in enumerate(cl)}
    for j, c in enumerate(clf.classes_):
        full[:, cli[c]] = p[:, j]
    yi = np.searchsorted(cl, y)
    return -np.log(full[np.arange(len(y)), yi])


def _fit(X, y, C, w):
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X, y, sample_weight=w)


def _make_folds(subs, n_folds, seed):
    subs = np.array(sorted(subs)); rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subs))
    folds = [list(subs[perm[i::n_folds]]) for i in range(n_folds)]
    h = hashlib.sha1(str([sorted(f) for f in folds]).encode()).hexdigest()[:12]
    return folds, h


def _prep_folds(Z, D, g, folds, coding, rank, C):
    """Precompute per-fold (Z-only) standardise + PC basis + design matrices (train/heldout)."""
    prep = []
    for f in folds:
        ho = np.isin(g, f); tr = ~ho
        if tr.sum() == 0 or ho.sum() == 0:
            return None
        wz = class_balanced_weights(np.zeros(tr.sum()), D[tr], g[tr])  # Z-only standardise weight (label-free)
        wz = wz if wz.sum() > 0 else np.ones(tr.sum())
        W = wz.sum()
        mu = (wz[:, None] * Z[tr]).sum(0) / W
        sd = np.sqrt(np.clip((wz[:, None] * (Z[tr] - mu) ** 2).sum(0) / W, 0, None)) + 1e-8
        Zs_tr = (Z[tr] - mu) / sd
        Vt = np.linalg.svd(np.sqrt(wz)[:, None] * Zs_tr, full_matrices=False)[2]
        Vr = Vt[:max(1, rank)].T
        X0_tr, X1_tr = _features(Z[tr], D[tr], coding, mu, sd, Vr)
        X0_ho, X1_ho = _features(Z[ho], D[ho], coding, mu, sd, Vr)
        prep.append(dict(tr=tr, ho=ho, X0_tr=X0_tr, X1_tr=X1_tr, X0_ho=X0_ho, X1_ho=X1_ho))
    return prep


def _t_quantile(p, df):
    """t_{p, df} quantile (scipy if available; else a Cornish-Fisher-ish normal approx)."""
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(p, max(int(df), 1)))
    except Exception:
        from math import sqrt
        z = 1.6448536269514722 if abs(p - 0.95) < 1e-6 else _norm_ppf(p)
        df = max(int(df), 1)
        return float(z + (z ** 3 + z) / (4 * df))  # small-sample bump


def _norm_ppf(p):
    # Acklam-style rational approx (sufficient here); avoids hard scipy dependency
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _studentize(deltas, eps=1e-9):
    """Subject-consistency summary of per-subject improvements delta_s: mean, sd, se, studentized Z =
    mean/(se+eps), and one-sided 95% lower confidence bound mean - t_{.95,S-1}*se. Random/noise labels give
    inconsistent (high-variance, ~0-mean) delta_s -> small Z & LCB<=0; real concept -> consistent delta_s>0."""
    v = np.asarray(list(deltas.values()), float); S = len(v)
    mean = float(v.mean()) if S else 0.0
    sd = float(v.std(ddof=1)) if S > 1 else 0.0
    se = sd / np.sqrt(S) if S else 0.0
    Z = mean / (se + eps)
    lcb = mean - _t_quantile(0.95, S - 1) * se
    return dict(mean=mean, sd=sd, se=float(se), Z=float(Z), lcb=float(lcb), S=int(S))


def _T_cv(prep, Yvec, D, g, cl, C):
    """Cross-fitted class-balanced per-subject improvements delta_s = cb_loss_h0 - cb_loss_h1 on held-out
    folds. Returns (T_mean, ok, deltas_dict); ok=False if any fold degenerates."""
    d0, d1 = {}, {}
    for p in prep:
        tr, ho = p["tr"], p["ho"]
        ytr = Yvec[tr]
        if len(np.unique(ytr)) < 2:
            return float("nan"), False, {}
        w_tr = class_balanced_weights(ytr, D[tr], g[tr])
        try:
            h0 = _fit(p["X0_tr"], ytr, C, w_tr)
            h1 = _fit(p["X1_tr"], ytr, C, w_tr)
        except Exception:
            return float("nan"), False, {}
        n0 = _nll(h0, p["X0_ho"], Yvec[ho], cl)
        n1 = _nll(h1, p["X1_ho"], Yvec[ho], cl)
        d0.update(cb_subject_losses(n0, Yvec[ho], D[ho], g[ho]))
        d1.update(cb_subject_losses(n1, Yvec[ho], D[ho], g[ho]))
    deltas = {s: d0[s] - d1[s] for s in d0}
    return float(np.mean(list(deltas.values()))), True, deltas


def sample_h0_fixed_condition_margins(logp0, D, y0_idx, rng, n_swaps, return_diag=False):
    """B3-P2.4b null draw: within-condition Metropolis label swaps that PRESERVE the observed
    condition x class counts EXACTLY, with acceptance ∝ p0 (so Y* ~ p0(Y|Z,C) restricted to fixed
    per-condition margins). Returns class-INDEX labels. Per-condition label composition/prior is thus a
    nuisance the null holds fixed, not concept evidence; the Z-dependent shared boundary stays via p0.
    return_diag=True also returns mixing diagnostics (acceptance rate, changed-label fraction) -- this does
    NOT change the swap behaviour, only instruments it (the audit uses it; the test path does not)."""
    y0 = np.asarray(y0_idx); y = y0.copy(); D = np.asarray(D)
    conds = np.unique(D); cond_idx = {int(c): np.where(D == c)[0] for c in conds}
    n_prop = n_acc = 0
    for _ in range(int(n_swaps)):
        c = conds[rng.integers(len(conds))]; idx = cond_idx[int(c)]
        if len(idx) < 2:
            continue
        i = idx[rng.integers(len(idx))]; j = idx[rng.integers(len(idx))]
        if y[i] == y[j]:
            continue
        n_prop += 1
        delta = (logp0[i, y[j]] + logp0[j, y[i]]) - (logp0[i, y[i]] + logp0[j, y[j]])
        if np.log(rng.random() + 1e-300) < delta:
            y[i], y[j] = y[j], y[i]; n_acc += 1
    if return_diag:
        chg = float(np.mean(y != y0))
        cw = {int(c): float(np.mean(y[D == c] != y0[D == c])) for c in conds}
        diag = dict(n_swaps=int(n_swaps), n_proposals=int(n_prop), n_accepted=int(n_acc),
                    acceptance_rate=(n_acc / n_prop if n_prop else 0.0), changed_fraction=chg,
                    condition_changed_fraction=cw)
        return y, diag
    return y


def _h0_full_logp(Z, Y, D, g, cl, coding, C):
    """log p0(Y|Z,C) from a single full-audit class-balanced h0 fit, columns aligned to cl."""
    w = class_balanced_weights(Y, D, g); W = w.sum()
    mu = (w[:, None] * Z).sum(0) / W
    sd = np.sqrt(np.clip((w[:, None] * (Z - mu) ** 2).sum(0) / W, 0, None)) + 1e-8
    Zs = (Z - mu) / sd; cv = condition_code(D, coding)[:, None]
    h0 = _fit(np.hstack([Zs, cv]), Y, C, w)
    pr = np.clip(h0.predict_proba(np.hstack([Zs, cv])), 1e-12, 1.0)
    full = np.full((len(Z), len(cl)), 1e-12); cli = {c: j for j, c in enumerate(cl)}
    for j, c in enumerate(h0.classes_):
        full[:, cli[c]] = pr[:, j]
    return np.log(full / full.sum(1, keepdims=True))


def _bootstrap(prep, D, g, cl, C, T, Z_obs, draw_fn, n_boot, invalid_frac_max, min_classes):
    """Run B replicates of a draw_fn -> Y*; recompute cross-fit (T*, studentized Z*); conservative invalid
    accounting (invalid charged extreme to BOTH p's). p = (1 + #{T*>=T} + #invalid)/(1+B); p_stud likewise
    on the studentized subject-consistency statistic Z* = mean(delta*)/(se(delta*)+eps)."""
    ge, ge_s, n_inv, n_fail, tstars, zstars = 1, 1, 0, 0, [], []
    for _ in range(n_boot):
        Ystar, failed = draw_fn()
        if failed:
            n_fail += 1; n_inv += 1; ge += 1; ge_s += 1; continue
        if any(len(np.unique(Ystar[D == c])) < min_classes for c in np.unique(D)):
            n_inv += 1; ge += 1; ge_s += 1; continue
        Ts, ok, dstar = _T_cv(prep, Ystar, D, g, cl, C)
        if not ok:
            n_inv += 1; ge += 1; ge_s += 1; continue
        Zs = _studentize(dstar)["Z"]
        tstars.append(Ts); zstars.append(Zs); ge += int(Ts >= T); ge_s += int(Zs >= Z_obs)
    return dict(p=ge / (n_boot + 1), p_stud=ge_s / (n_boot + 1), n_inv=int(n_inv), n_fail=int(n_fail),
                nmean=float(np.mean(tstars)) if tstars else float("nan"),
                nsd=float(np.std(tstars)) if tstars else float("nan"),
                snmean=float(np.mean(zstars)) if zstars else float("nan"),
                snsd=float(np.std(zstars)) if zstars else float("nan"),
                estimable=bool(n_inv <= invalid_frac_max * n_boot))


def paired_cv_test(Z, Y, D, groups, condition_coding="centered", rank=3, C=0.5, n_folds=N_FOLDS,
                   min_epochs=MIN_EPOCHS_PER_CONDITION, n_boot=200, seed=0, invalid_frac_max=0.20,
                   min_classes=2, null_mode="fixed_margin", also_standard=True, n_swaps=None,
                   null_generator="full_audit"):
    """Cross-fitted class-balanced paired conditional-change test on ELIGIBLE complete pairs only.
    null_mode="fixed_margin" (B3-P2.4b, PRIMARY): null preserves per-condition class margins (so label/prior
    composition cannot inflate T); "parametric" = P2.4a standard null. also_standard reports the standard
    null too (DIAGNOSTIC only). Returns T_cv, p_value_cv (primary), fixed_margin_null_p, standard_null_p,
    valid, reason, null_mean_cv/null_sd_cv, n_boot_invalid, n_sampler_failures, margin_preserved,
    sampler_seed, null_version, would_confirm_under_standard_null, n_eligible, fold_hash, per_condition_classes."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); g = np.asarray(groups)
    base = dict(T_cv=float("nan"), p_value_cv=1.0, fixed_margin_null_p=float("nan"),
                standard_null_p=float("nan"), valid=False, null_mean_cv=float("nan"),
                null_sd_cv=float("nan"), n_boot_invalid=0, n_sampler_failures=0, margin_preserved=False,
                sampler_seed=int(seed + 777), null_version="condition_matched_fixed_margin_h0_bootstrap",
                would_confirm_under_standard_null=False, n_eligible=0, fold_hash=None,
                per_condition_classes={}, studentized_p_value=1.0, studentized_stat=float("nan"),
                subject_consistency_lcb=float("nan"), mean_delta=float("nan"), sd_delta=float("nan"),
                se_delta=float("nan"), n_subject_deltas=0, delta_subjects=[],
                studentized_null_mean=float("nan"), studentized_null_sd=float("nan"))
    elig = eligible_complete_pairs(D, g, min_epochs)
    if len(elig) < n_folds * 2:
        return {**base, "n_eligible": len(elig), "reason": f"only {len(elig)} eligible complete pairs"}
    m = np.isin(g, elig)
    Z, Y, D, g = Z[m], Y[m], D[m], g[m]
    pcc = per_condition_classes(Y, D)
    if any(len(v) < min_classes for v in pcc.values()):
        return {**base, "n_eligible": len(elig), "per_condition_classes": pcc,
                "reason": "a condition lacks >=min_classes among eligible pairs"}
    cl = np.array(sorted(np.unique(Y)))
    folds, fhash = _make_folds(elig, n_folds, seed)
    prep = _prep_folds(Z, D, g, folds, condition_coding, rank, C)
    if prep is None:
        return {**base, "n_eligible": len(elig), "fold_hash": fhash, "reason": "fold prep degenerate"}
    T, ok, deltas = _T_cv(prep, Y, D, g, cl, C)
    if not ok:
        return {**base, "n_eligible": len(elig), "fold_hash": fhash, "T_cv": T,
                "reason": "observed cross-fit degenerate"}
    st = _studentize(deltas); Z_obs = st["Z"]   # subject-consistency studentized statistic + LCB

    # standard (parametric) null draw: per-fold held-out Y* ~ fold-train-h0 (P2.4a) -- DIAGNOSTIC.
    h0_draw = []
    for p in prep:
        w_tr = class_balanced_weights(Y[p["tr"]], D[p["tr"]], g[p["tr"]])
        h0 = _fit(p["X0_tr"], Y[p["tr"]], C, w_tr)
        pr = np.clip(h0.predict_proba(p["X0_ho"]), 1e-12, 1.0)
        full = np.full((p["ho"].sum(), len(cl)), 1e-12); cli = {c: j for j, c in enumerate(cl)}
        for j, c in enumerate(h0.classes_):
            full[:, cli[c]] = pr[:, j]
        h0_draw.append(full / full.sum(1, keepdims=True))
    rng_std = np.random.default_rng(seed + 1)

    def draw_std():
        Ystar = Y.copy()
        for p, prob in zip(prep, h0_draw):
            cum = np.cumsum(prob, axis=1); u = rng_std.random(len(prob))
            Ystar[p["ho"]] = cl[(u[:, None] > cum).sum(1)]
        return Ystar, False

    # fixed-margin null draw (PRIMARY): preserve per-condition class counts exactly.
    # null_generator: "full_audit" = single full-audit h0 (default, P2.4b); "fold_local" = each point's p0
    # from its OWN fold's train-h0 (audit-3 comparison; reuses the standard-null per-fold probs).
    if null_generator == "fold_local":
        logp0 = np.zeros((len(Y), len(cl)))
        for p, prob in zip(prep, h0_draw):
            logp0[p["ho"]] = np.log(np.clip(prob, 1e-12, 1.0))
    else:
        logp0 = _h0_full_logp(Z, Y, D, g, cl, condition_coding, C)
    y0_idx = np.searchsorted(cl, Y)
    sampler_seed = int(seed + 777); rng_fm = np.random.default_rng(sampler_seed)
    nsw = int(n_swaps) if n_swaps else max(20 * len(Y), 300)

    def draw_fm():
        ys = sample_h0_fixed_condition_margins(logp0, D, y0_idx, rng_fm, nsw)
        return cl[ys], False

    res_fm = _bootstrap(prep, D, g, cl, C, T, Z_obs, draw_fm, n_boot, invalid_frac_max, min_classes)
    res_std = _bootstrap(prep, D, g, cl, C, T, Z_obs, draw_std, n_boot, invalid_frac_max, min_classes) \
        if also_standard else None
    # margin-preservation self-check on one draw
    chk = sample_h0_fixed_condition_margins(logp0, D, y0_idx, np.random.default_rng(sampler_seed + 1), nsw)
    margin_ok = all(np.array_equal(np.bincount(y0_idx[D == c], minlength=len(cl)),
                                   np.bincount(chk[D == c], minlength=len(cl))) for c in np.unique(D))
    primary = res_fm if null_mode == "fixed_margin" else res_std
    out = dict(base)
    out.update(T_cv=float(T), p_value_cv=primary["p"], fixed_margin_null_p=res_fm["p"],
               standard_null_p=(res_std["p"] if res_std else float("nan")),
               null_mean_cv=primary["nmean"], null_sd_cv=primary["nsd"], n_boot_invalid=primary["n_inv"],
               n_sampler_failures=res_fm["n_fail"], margin_preserved=bool(margin_ok),
               sampler_seed=sampler_seed, n_eligible=int(len(elig)), fold_hash=fhash,
               per_condition_classes=pcc,
               would_confirm_under_standard_null=bool(res_std and res_std["p"] <= 0.05),
               studentized_p_value=primary["p_stud"], studentized_stat=float(Z_obs),
               subject_consistency_lcb=float(st["lcb"]), mean_delta=float(st["mean"]),
               sd_delta=float(st["sd"]), se_delta=float(st["se"]), n_subject_deltas=int(st["S"]),
               delta_subjects=[float(v) for v in deltas.values()],
               studentized_null_mean=primary["snmean"], studentized_null_sd=primary["snsd"])
    if not primary["estimable"]:
        out.update(valid=False, reason=f"null not estimable: {primary['n_inv']}/{n_boot} invalid")
        return out
    out.update(valid=True, reason=f"{len(elig)} eligible pairs")
    return out


def certify_paired_calibrated(Z, Y, D, G, m, alpha=0.05, decide_n=20, min_pairs=4, min_confirm_pairs=20,
                              pair_integrity_min=PAIR_INTEGRITY_MIN, min_epochs=MIN_EPOCHS_PER_CONDITION,
                              rank=3, C=0.5, n_folds=N_FOLDS, n_boot=200, seed=0,
                              alpha_family=0.05, n_decision_budgets=2):
    """B3-P2.4 calibrated certifier. Batch-level pair-integrity guard + eligible-complete-pair (min-epochs)
    guard + class-balanced cross-fitted test. A positive CONCEPT_CONFIRMED requires: pair_integrity >=
    pair_integrity_min, >= min_confirm_pairs ELIGIBLE complete pairs queried, valid cross-fit, p_cv<=alpha.
    Returns the certificate state + the full P2.4 per-cluster log fields."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); G = np.asarray(G)
    stats = pair_integrity_stats(D, G)
    elig_all = eligible_complete_pairs(D, G, min_epochs)
    log = dict(m=int(m), calibration_version=CALIBRATION_VERSION, h1_basis="pc", condition_coding="centered",
               pair_integrity=stats["pair_integrity"], n_subjects_total=stats["n_subjects_total"],
               n_complete_pairs=stats["n_complete_pairs"],
               missing_condition_fraction=stats["missing_condition_fraction"],
               n_eligible_complete_pairs=len(elig_all), min_epochs_per_condition=int(min_epochs),
               min_confirm_pairs=int(min_confirm_pairs), n_folds=int(n_folds), valid=False,
               p_value=float("nan"), T=float("nan"), would_confirm_without_guard=False)
    if m <= 0:
        log.update(state=UNIDENTIFIABLE, reason="m=0: Z-only triage cannot confirm"); return log
    if stats["pair_integrity"] < pair_integrity_min:          # GUARD 1 (closes missing_pair leak)
        log.update(state=INVALID_PAIR,
                   reason=f"pair_integrity {stats['pair_integrity']:.2f} < {pair_integrity_min}")
        return log
    if len(elig_all) < min_pairs:                             # GUARD 2 (closes unequal_epochs leak)
        log.update(state=INVALID_PAIR, reason=f"{len(elig_all)} eligible complete pairs < {min_pairs}")
        return log
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.array(sorted(elig_all)), size=min(int(m), len(elig_all)), replace=False)
    mask = np.isin(G, pick); Zq, Yq, Dq, Gq = Z[mask], Y[mask], D[mask], G[mask]
    n_q = int(len(pick))
    log.update(n_queried_subjects=n_q,
               n_labeled_subject_conditions=len({(int(s), int(c)) for s, c in zip(Gq, Dq)}),
               n_labeled_epochs=int(len(Yq)))
    t = paired_cv_test(Zq, Yq, Dq, Gq, condition_coding="centered", rank=rank, C=C, n_folds=n_folds,
                       min_epochs=min_epochs, n_boot=n_boot, seed=seed,
                       null_mode="fixed_margin", also_standard=True)
    # B3-P2.4d: CROSS-BUDGET ALPHA-SPENDING. The certifier may confirm at n_decision_budgets positive
    # budgets (m=20,30) -> Bonferroni alpha_budget = alpha_family / n_decision_budgets (=0.025), applied to
    # BOTH p-gates AND the lower-confidence-bound level (1 - alpha_budget = 0.975). Deterministic, no sweep.
    alpha_budget = alpha_family / max(int(n_decision_budgets), 1)
    lcb_level = 1.0 - alpha_budget
    S = int(t.get("n_subject_deltas", 0) or 0)
    lcb_budget = (float(t.get("mean_delta", float("nan")))
                  - _t_quantile(lcb_level, max(S - 1, 1)) * float(t.get("se_delta", 0.0) or 0.0))
    # P2.4c diagnostic (alpha_family=0.05, 95% LCB) -- kept only to show what alpha-spending removes
    would_c = bool(t["valid"] and t["p_value_cv"] <= alpha_family
                   and t["studentized_p_value"] <= alpha_family and t["subject_consistency_lcb"] > 0)
    # P2.4d DECISION (alpha_budget=0.025, 97.5% LCB)
    would_meanT = bool(t["valid"] and t["p_value_cv"] <= alpha_budget)
    stud_ok = bool(t["valid"] and t["studentized_p_value"] <= alpha_budget and lcb_budget > 0)
    would = would_meanT and stud_ok
    size_ok = bool(n_q >= min_confirm_pairs and t["n_eligible"] >= min_confirm_pairs)
    old_decision = (CONCEPT_CONFIRMED if (would_c and size_ok)
                    else NEED_MORE_LABELS if (would_c or n_q < decide_n) else NO_CONCEPT_EVIDENCE)
    log.update(alpha_family=float(alpha_family), alpha_budget=float(alpha_budget), lcb_level=float(lcb_level),
               positive_decision_budgets=[20, 30], subject_consistency_lcb_budget=float(lcb_budget),
               old_p24c_decision=old_decision,
               valid=bool(t["valid"]), p_value=float(t["p_value_cv"]), T=float(t["T_cv"]),
               observed_T=float(t["T_cv"]), T_cv=float(t["T_cv"]), p_value_cv=float(t["p_value_cv"]),
               fixed_margin_null_p=float(t["fixed_margin_null_p"]), standard_null_p=float(t["standard_null_p"]),
               studentized_p_value=float(t["studentized_p_value"]), studentized_stat=float(t["studentized_stat"]),
               subject_consistency_lcb=float(t["subject_consistency_lcb"]), mean_delta=float(t["mean_delta"]),
               sd_delta=float(t["sd_delta"]), se_delta=float(t["se_delta"]),
               n_subject_deltas=int(t["n_subject_deltas"]), studentized_null_mean=t["studentized_null_mean"],
               studentized_null_sd=t["studentized_null_sd"], delta_subjects=t.get("delta_subjects", []),
               would_confirm_under_standard_null=bool(t["would_confirm_under_standard_null"]),
               null_version=t["null_version"], n_sampler_failures=int(t["n_sampler_failures"]),
               margin_preserved=bool(t["margin_preserved"]), sampler_seed=int(t["sampler_seed"]),
               null_mean=t["null_mean_cv"], null_sd=t["null_sd_cv"], null_mean_cv=t["null_mean_cv"],
               null_sd_cv=t["null_sd_cv"], n_boot_invalid=t["n_boot_invalid"],
               n_eligible_queried=int(t["n_eligible"]), fold_hash=t["fold_hash"],
               class_cell_counts_by_condition=t["per_condition_classes"],
               class_balance_gate_status=("ok" if t["valid"] else "blocked"),
               would_confirm_without_guard=would_meanT,
               old_decision_without_studentized_gate=old_decision, reason=t["reason"])
    if not t["valid"]:
        log["state"] = NEED_MORE_LABELS
    elif would and size_ok:
        log["state"] = CONCEPT_CONFIRMED
    elif would:
        log["state"] = NEED_MORE_LABELS                       # consistent+significant but too few eligible
    elif would_meanT and size_ok:
        # mean-T significant @ alpha_budget but the studentized/LCB consistency gate rejected -> refuse.
        log["state"] = NO_CONCEPT_EVIDENCE
        log["reason"] = ("STUDENTIZED_FIXED_MARGIN_NULL_NOT_SIG" if t["studentized_p_value"] > alpha_budget
                         else "SUBJECT_CONSISTENCY_GATE_NOT_MET")
    elif would_c and size_ok:
        # P2.4c (alpha=0.05) WOULD have confirmed, but the cross-budget alpha-spending (0.025) removed it.
        log["state"] = NO_CONCEPT_EVIDENCE
        log["reason"] = "CROSS_BUDGET_ALPHA_SPENDING_NOT_MET"
    elif n_q >= decide_n:
        log["state"] = NO_CONCEPT_EVIDENCE
        if t["would_confirm_under_standard_null"]:
            log["reason"] = "PRIOR_COMPOSITION_MATCHED_NULL_NOT_SIG"
    else:
        log["state"] = NEED_MORE_LABELS
    log["new_decision_with_studentized_gate"] = log["state"]
    log["new_p24d_decision"] = log["state"]
    return log
