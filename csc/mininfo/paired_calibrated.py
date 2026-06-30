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
     overfit creep (random_label etc.). n_folds fixed (no sweep).

DEVELOPMENT only; NO freeze/confirmatory; NO real EEG. calibration_version below is logged.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression

from .paired_conditional_test import condition_code
from .paired_certifier import (CONCEPT_CONFIRMED, NO_CONCEPT_EVIDENCE, NEED_MORE_LABELS,
                               INVALID_PAIR, UNIDENTIFIABLE)

CALIBRATION_VERSION = "p24_pair_integrity_classbalanced_crossfit"
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


def _T_cv(prep, Yvec, D, g, cl, C):
    """Cross-fitted class-balanced T = mean_subject (cb_loss_h0 - cb_loss_h1) on held-out folds. Returns
    (T, ok) where ok=False if any fold degenerates (a class missing in a train cell etc.)."""
    d0, d1 = {}, {}
    for p in prep:
        tr, ho = p["tr"], p["ho"]
        ytr = Yvec[tr]
        if len(np.unique(ytr)) < 2:
            return float("nan"), False
        w_tr = class_balanced_weights(ytr, D[tr], g[tr])
        try:
            h0 = _fit(p["X0_tr"], ytr, C, w_tr)
            h1 = _fit(p["X1_tr"], ytr, C, w_tr)
        except Exception:
            return float("nan"), False
        n0 = _nll(h0, p["X0_ho"], Yvec[ho], cl)
        n1 = _nll(h1, p["X1_ho"], Yvec[ho], cl)
        d0.update(cb_subject_losses(n0, Yvec[ho], D[ho], g[ho]))
        d1.update(cb_subject_losses(n1, Yvec[ho], D[ho], g[ho]))
    subs = list(d0)
    return float(np.mean([d0[s] - d1[s] for s in subs])), True


def paired_cv_test(Z, Y, D, groups, condition_coding="centered", rank=3, C=0.5, n_folds=N_FOLDS,
                   min_epochs=MIN_EPOCHS_PER_CONDITION, n_boot=200, seed=0, invalid_frac_max=0.20,
                   min_classes=2):
    """Cross-fitted, class-balanced paired conditional-change test on ELIGIBLE complete pairs only.
    Returns T_cv, p_value_cv, valid, reason, null_mean_cv, null_sd_cv, n_boot_invalid, n_eligible,
    fold_hash, per_condition_classes."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); g = np.asarray(groups)
    base = dict(T_cv=float("nan"), p_value_cv=1.0, valid=False, null_mean_cv=float("nan"),
                null_sd_cv=float("nan"), n_boot_invalid=0, n_eligible=0, fold_hash=None,
                per_condition_classes={})
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
    T, ok = _T_cv(prep, Y, D, g, cl, C)
    if not ok:
        return {**base, "n_eligible": len(elig), "fold_hash": fhash, "T_cv": T,
                "reason": "observed cross-fit degenerate"}
    # parametric-bootstrap null: per fold, draw held-out Y* ~ (h0 fit on that fold's TRAIN), assemble Y*,
    # rerun the SAME cross-fit pipeline.
    h0_draw = []
    for p in prep:
        w_tr = class_balanced_weights(Y[p["tr"]], D[p["tr"]], g[p["tr"]])
        h0 = _fit(p["X0_tr"], Y[p["tr"]], C, w_tr)
        pr = np.clip(h0.predict_proba(p["X0_ho"]), 1e-12, 1.0)
        full = np.full((p["ho"].sum(), len(cl)), 1e-12)
        cli = {c: j for j, c in enumerate(cl)}
        for j, c in enumerate(h0.classes_):
            full[:, cli[c]] = pr[:, j]
        h0_draw.append(full / full.sum(1, keepdims=True))
    rng = np.random.default_rng(seed + 1)
    ge, n_inv, tstars = 1, 0, []
    for _ in range(n_boot):
        Ystar = Y.copy()
        for p, prob in zip(prep, h0_draw):
            cum = np.cumsum(prob, axis=1); u = rng.random(len(prob))
            Ystar[p["ho"]] = cl[(u[:, None] > cum).sum(1)]
        if any(len(np.unique(Ystar[D == c])) < min_classes for c in np.unique(D)):
            n_inv += 1; ge += 1; continue
        Ts, ok = _T_cv(prep, Ystar, D, g, cl, C)
        if not ok:
            n_inv += 1; ge += 1; continue
        tstars.append(Ts); ge += int(Ts >= T)
    nmean = float(np.mean(tstars)) if tstars else float("nan")
    nsd = float(np.std(tstars)) if tstars else float("nan")
    if n_inv > invalid_frac_max * n_boot:
        return {**base, "n_eligible": len(elig), "fold_hash": fhash, "T_cv": T, "n_boot_invalid": int(n_inv),
                "null_mean_cv": nmean, "null_sd_cv": nsd, "per_condition_classes": pcc,
                "reason": f"null not estimable: {n_inv}/{n_boot} invalid"}
    return dict(T_cv=float(T), p_value_cv=ge / (n_boot + 1), valid=True, reason=f"{len(elig)} eligible pairs",
                null_mean_cv=nmean, null_sd_cv=nsd, n_boot_invalid=int(n_inv),
                n_eligible=int(len(elig)), fold_hash=fhash, per_condition_classes=pcc)


def certify_paired_calibrated(Z, Y, D, G, m, alpha=0.05, decide_n=20, min_pairs=4, min_confirm_pairs=20,
                              pair_integrity_min=PAIR_INTEGRITY_MIN, min_epochs=MIN_EPOCHS_PER_CONDITION,
                              rank=3, C=0.5, n_folds=N_FOLDS, n_boot=200, seed=0):
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
                       min_epochs=min_epochs, n_boot=n_boot, seed=seed)
    would = bool(t["valid"] and t["p_value_cv"] <= alpha)
    log.update(valid=bool(t["valid"]), p_value=float(t["p_value_cv"]), T=float(t["T_cv"]),
               observed_T=float(t["T_cv"]), T_cv=float(t["T_cv"]), p_value_cv=float(t["p_value_cv"]),
               null_mean=t["null_mean_cv"], null_sd=t["null_sd_cv"], null_mean_cv=t["null_mean_cv"],
               null_sd_cv=t["null_sd_cv"], n_boot_invalid=t["n_boot_invalid"],
               n_eligible_queried=int(t["n_eligible"]), fold_hash=t["fold_hash"],
               class_cell_counts_by_condition=t["per_condition_classes"],
               class_balance_gate_status=("ok" if t["valid"] else "blocked"),
               would_confirm_without_guard=would, reason=t["reason"])
    if not t["valid"]:
        log["state"] = NEED_MORE_LABELS
    elif would and n_q >= min_confirm_pairs and t["n_eligible"] >= min_confirm_pairs:
        log["state"] = CONCEPT_CONFIRMED
    elif would:
        log["state"] = NEED_MORE_LABELS                       # significant but too few eligible pairs
    elif n_q >= decide_n:
        log["state"] = NO_CONCEPT_EVIDENCE
    else:
        log["state"] = NEED_MORE_LABELS
    return log
