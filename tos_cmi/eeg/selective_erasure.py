"""CMI-Trace Relaxation Ladder Stage 5 — SOURCE-ONLY selective erasure with identity fallback.

Four fixed policies (G0..G3). All gate decisions are made from SOURCE-ONLY diagnostics BEFORE any target
scoring; thresholds are NOT optimized against target outcomes (EXPLORATORY, not preregistered confirmation).
A refused gate deploys IDENTITY (no erasure). The gated deployment target effect vs the always-identity
baseline is hypothesis H5.

  G0 always_erase                 : diagnostic only.
  G1 source competence + safety   : erase iff (source task clears chance) AND (source internal-CV task drop
                                     after erasure <= 0.02) AND (linear subject decode is materially reduced).
  G2 shared task direction        : G1 AND task-direction consistency exceeds its source-only null (p<=0.05).
  G3 subject-specific erasure      : G2 AND informed deletion reduces source subject decodability MORE than a
                                     matched-rank random deletion (source-only, favorable direction).
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

from tos_cmi.eeg.relaxation_ladder import lw_leace_full, random_removal, fresh_head_bacc, _dense

TASK_DROP_MAX = 0.02
SUBJECT_REDUCE_MIN = 0.02          # linear subject decode must fall by >= this to count as "materially reduced"
CONSISTENCY_ALPHA = 0.05


def default_consistency(Z, y, subj):
    """Source-only task-direction consistency + permutation p (Stage 4). Binary -> pairwise-cosine test;
    multiclass -> macro over class pairs with the median per-pair permutation p (conservative summary)."""
    from cmi.eval.task_direction_consistency import (direction_consistency_binary,
                                                     direction_consistency_multiclass)
    classes = sorted(int(c) for c in np.unique(y))
    if len(classes) < 2:
        return float("nan"), float("nan")
    if len(classes) == 2:
        r = direction_consistency_binary(Z, y, subj, classes[1], classes[0])
        return float(r["mean_pairwise_cosine"]), float(r["perm_p"])
    r = direction_consistency_multiclass(Z, y, subj, classes)
    ps = [v["perm_p"] for v in r["per_pair"].values() if np.isfinite(v.get("perm_p", np.nan))]
    return float(r["macro_avg_consistency"]), (float(np.median(ps)) if ps else float("nan"))


def _linear_subject_decode(Z, subj, seed=0):
    """Source-only linear subject decodability (grouped is impossible within-subject; use a stratified split)."""
    subj = _dense(subj)
    if len(np.unique(subj)) < 2:
        return float("nan")
    rng = np.random.default_rng(seed); idx = rng.permutation(len(Z)); cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    if len(np.unique(subj[tr])) < 2 or len(te) == 0:
        return float("nan")
    clf = LogisticRegression(max_iter=300).fit(Z[tr], subj[tr])
    return float((clf.predict(Z[te]) == subj[te]).mean())


def _source_cv_task_bacc(Z, y, subj, seed=0, n_splits=5):
    """Source internal subject-grouped CV task balanced accuracy (no target)."""
    groups = _dense(subj); n = min(n_splits, len(np.unique(groups)))
    if n < 2 or len(np.unique(y)) < 2:
        return float("nan")
    accs = []
    for tr, te in GroupKFold(n_splits=n).split(Z, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        accs.append(fresh_head_bacc(Z[tr], y[tr], Z[te], y[te], head="logreg", seed=seed))
    return float(np.mean(accs)) if accs else float("nan")


def source_diagnostics(feat, seed=0, n_random=20, consistency_fn=None):
    """All SOURCE-ONLY diagnostics the gates use. Never touches target. `consistency_fn(Z,y,subj)->(stat,p)`
    is the optional task-direction-consistency source-only null test (Stage 4)."""
    Zs, ys, ds = feat["Z_source"], feat["y_source"], feat["subj_source"]
    n_cls = int(feat.get("n_cls", len(np.unique(ys))))
    chance = 1.0 / n_cls
    # eraser fit on source only (the strict, deployable regime)
    fn_lw, rank = lw_leace_full(Zs, ds)
    Zs_lw = fn_lw(Zs)
    src_task_full = _source_cv_task_bacc(Zs, ys, ds, seed=seed)
    src_task_lw = _source_cv_task_bacc(Zs_lw, ys, ds, seed=seed)
    subj_full = _linear_subject_decode(Zs, ds, seed=seed)
    subj_lw = _linear_subject_decode(Zs_lw, ds, seed=seed)
    # matched-rank random deletion (source-only) subject decode, averaged
    subj_rand = np.nanmean([_linear_subject_decode(random_removal(Zs.shape[1], rank, 1000 * seed + d)(Zs), ds, seed=seed)
                            for d in range(n_random)]) if rank > 0 else subj_full
    cfn = consistency_fn or default_consistency
    try:
        cons_stat, cons_p = cfn(Zs, ys, ds)
    except Exception:
        cons_stat, cons_p = float("nan"), float("nan")
    return dict(
        rank=int(rank), chance=float(chance),
        source_task_full=float(src_task_full), source_task_after_erasure=float(src_task_lw),
        source_cv_task_drop=float((src_task_full - src_task_lw) if np.isfinite(src_task_full) and np.isfinite(src_task_lw) else np.nan),
        subject_decode_full=float(subj_full), subject_decode_after_erasure=float(subj_lw),
        subject_decode_random=float(subj_rand),
        subject_decode_reduction=float((subj_full - subj_lw) if np.isfinite(subj_full) and np.isfinite(subj_lw) else np.nan),
        subject_reduction_vs_random=float((subj_rand - subj_lw) if np.isfinite(subj_rand) and np.isfinite(subj_lw) else np.nan),
        task_direction_consistency=float(cons_stat), task_direction_consistency_p=float(cons_p))


def gate_decision(diag, policy):
    """Return (accepted: bool, refusal_reason: str) for policy in {G0,G1,G2,G3} from source-only `diag`."""
    if policy == "G0":
        return True, ""                                    # always erase (diagnostic)
    reasons = []
    # G1 conditions
    if not (np.isfinite(diag["source_task_full"]) and diag["source_task_full"] > diag["chance"]):
        reasons.append("source_task_below_chance")
    if not (np.isfinite(diag["source_cv_task_drop"]) and diag["source_cv_task_drop"] <= TASK_DROP_MAX):
        reasons.append(f"source_task_drop>{TASK_DROP_MAX}")
    if not (np.isfinite(diag["subject_decode_reduction"]) and diag["subject_decode_reduction"] >= SUBJECT_REDUCE_MIN):
        reasons.append("subject_decode_not_materially_reduced")
    if policy == "G1":
        return (len(reasons) == 0), ";".join(reasons)
    # G2 adds shared-task-direction
    if not (np.isfinite(diag["task_direction_consistency_p"]) and diag["task_direction_consistency_p"] <= CONSISTENCY_ALPHA):
        reasons.append("task_direction_not_consistent")
    if policy == "G2":
        return (len(reasons) == 0), ";".join(reasons)
    # G3 adds subject-specificity (beats matched-rank random, source-only)
    if not (np.isfinite(diag["subject_reduction_vs_random"]) and diag["subject_reduction_vs_random"] > 0):
        reasons.append("erasure_not_subject_specific_vs_random")
    if policy == "G3":
        return (len(reasons) == 0), ";".join(reasons)
    raise ValueError(f"unknown policy {policy}")


def gated_target_effect(feat, policy, diag, seed=0, head_regime="logreg"):
    """Apply the source-only gate, then deploy erasure-if-accepted-else-identity, and score the target
    (fresh head, strict source-only L1 regime). Returns the gated target bAcc, the identity baseline, and
    the decision. Target Y enters only this final scoring."""
    accepted, reason = gate_decision(diag, policy)
    Zs, ys, ds = feat["Z_source"], feat["y_source"], feat["subj_source"]
    Zt, yt = feat["Z_target"], feat["y_target"]
    identity_bacc = fresh_head_bacc(Zs, ys, Zt, yt, head=head_regime, seed=seed)
    if accepted:
        fn, _ = lw_leace_full(Zs, ds)
        gated_bacc = fresh_head_bacc(fn(Zs), ys, fn(Zt), yt, head=head_regime, seed=seed)
    else:
        gated_bacc = identity_bacc
    return dict(policy=policy, accepted=bool(accepted), refusal_reason=reason,
                gated_target_bacc=float(gated_bacc), identity_target_bacc=float(identity_bacc),
                gated_minus_identity=float(gated_bacc - identity_bacc))
