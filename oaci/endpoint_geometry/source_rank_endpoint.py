"""C31 Q3 — source-rank endpoint specificity (RED-TEAM corrected). The C30 source-visible rank factor (frozen probe
SCORE, and R_src) grades EACH endpoint label by within-target + pooled AUC + direction-agnostic strength +
sign-consistency. RED-TEAM caveats folded in: (1) the frozen probe SCORE is trained on the accuracy competence label
(label == accuracy_good), so "ranks accuracy best" is partly BY CONSTRUCTION, not a discovered endpoint property;
(2) the accuracy-vs-calibration strength gap is tested with a 9-target cluster-bootstrap CI — "accuracy-specific" is
asserted only if that CI excludes 0. At the primary margin it does NOT (accuracy is indistinguishable from overall
calibration within per-target noise); only accuracy-vs-ECE is distinguishable."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc
from . import artifact_loader, schema
from .artifact_loader import _v

_TARGETS = ("accuracy_good", "nll_good", "ece_good", "calibration_good", "joint_good", "pareto_good")


def _per_target_aucs(rows, factor_key, label_key) -> dict:
    per = {}
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t]
        y = np.array([_v(r, label_key) for r in g], dtype=float)
        x = np.array([_v(r, factor_key) for r in g], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y); yb = y[ok]
        if ok.sum() > 2 and yb.std() > 1e-9 and 0 < yb.sum() < ok.sum() and x[ok].std() > 1e-9:
            per[t] = _auc(yb.astype(int), x[ok])
    return per


def _bootstrap_gap_ci(aucs_a, aucs_b, n=2000, seed=None):
    """Cluster-bootstrap the strength gap |mean(a)-0.5| - |mean(b)-0.5| over the shared targets."""
    ts = sorted(set(aucs_a) & set(aucs_b))
    if len(ts) < 3:
        return {"gap": None, "ci_lo": None, "ci_hi": None, "excludes_zero": None, "frac_positive": None}
    a = np.array([aucs_a[t] for t in ts]); b = np.array([aucs_b[t] for t in ts])
    gap = abs(a.mean() - 0.5) - abs(b.mean() - 0.5)
    rng = np.random.RandomState(schema.PERM_SEED if seed is None else seed)
    boot = []
    for _ in range(n):
        idx = rng.randint(0, len(ts), len(ts))
        boot.append(abs(a[idx].mean() - 0.5) - abs(b[idx].mean() - 0.5))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"gap": float(gap), "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "frac_positive": float(np.mean(np.array(boot) > 0))}


def source_rank_endpoint(rows) -> dict:
    out = {}
    for factor in (schema.SCORE_KEY, "R_src"):
        per = {}
        for lab in _TARGETS:
            wt = artifact_loader.within_target_auc(rows, factor, lab)
            pooled = artifact_loader.pooled_auc(rows, factor, lab)
            sc = artifact_loader.sign_consistency(rows, factor, lab)
            per[lab] = {"within_target_auc": wt, "pooled_auc": pooled, "rank_strength": artifact_loader.rank_strength(wt),
                        "sign_consistency": sc["sign_consistency"], "transfers": sc["transfers"]}
        out[factor] = per
    score = out[schema.SCORE_KEY]
    acc_s = score["accuracy_good"]["rank_strength"] or 0
    cal_s = score["calibration_good"]["rank_strength"] or 0          # calibration_good's OWN strength (not max-nll)
    joint_s = score["joint_good"]["rank_strength"] or 0

    # by-construction check: the frozen probe SCORE is trained to rank the accuracy competence label
    n_mismatch = sum(1 for r in rows if _v(r, schema.LABEL_KEY) is not None and _v(r, "accuracy_good") is not None
                     and int(r[schema.LABEL_KEY]) != int(r["accuracy_good"]))
    by_construction = bool(n_mismatch == 0)

    # 9-target cluster-bootstrap on the strength gap (accuracy vs calibration_good, and accuracy vs ECE)
    a_auc = _per_target_aucs(rows, "score", "accuracy_good")
    c_auc = _per_target_aucs(rows, "score", "calibration_good")
    e_auc = _per_target_aucs(rows, "score", "ece_good")
    gap_calib = _bootstrap_gap_ci(a_auc, c_auc)
    gap_ece = _bootstrap_gap_ci(a_auc, e_auc)

    # "accuracy-specific" only if the accuracy-vs-CALIBRATION gap CI excludes 0 (it does NOT at the primary margin)
    accuracy_specific = bool(gap_calib["excludes_zero"] and gap_calib["gap"] and gap_calib["gap"] > 0)
    calibration_biased = bool(gap_calib["excludes_zero"] and gap_calib["gap"] and gap_calib["gap"] < 0)
    accuracy_vs_ece_distinguishable = bool(gap_ece["excludes_zero"] and gap_ece["gap"] and gap_ece["gap"] > 0)
    joint_predicted = bool(joint_s is not None and (score["joint_good"]["within_target_auc"] or 0) >= schema.RANK_SIGNAL_MIN)

    if accuracy_specific:
        note = ("the source rank is ACCURACY-SPECIFIC (acc-vs-calibration strength gap %.3f, 95%% CI [%.3f, %.3f] "
                "excludes 0)" % (gap_calib["gap"], gap_calib["ci_lo"], gap_calib["ci_hi"]))
    else:
        note = ("the source rank is accuracy-ALIGNED BY CONSTRUCTION (probe trained on label==accuracy_good, "
                "%d/%d mismatches) and NOT distinguishably accuracy-specific vs calibration: acc strength %.3f vs "
                "calibration_good %.3f, gap %.3f 95%% CI [%.3f, %.3f] INCLUDES 0; the only distinguishable contrast "
                "is accuracy vs ECE (gap %.3f, CI [%.3f, %.3f], excludes 0: %s)"
                % (n_mismatch, len(rows), acc_s, cal_s, gap_calib["gap"] or 0, gap_calib["ci_lo"] or 0,
                   gap_calib["ci_hi"] or 0, gap_ece["gap"] or 0, gap_ece["ci_lo"] or 0, gap_ece["ci_hi"] or 0,
                   accuracy_vs_ece_distinguishable))

    return {"per_factor": out, "score_accuracy_strength": acc_s, "score_calibration_strength": cal_s,
            "score_ece_strength": score["ece_good"]["rank_strength"], "score_joint_strength": joint_s,
            "source_rank_accuracy_specific": accuracy_specific, "source_rank_calibration_biased": calibration_biased,
            "accuracy_vs_calibration_gap_ci": gap_calib, "accuracy_vs_ece_gap_ci": gap_ece,
            "accuracy_vs_ece_distinguishable": accuracy_vs_ece_distinguishable,
            "accuracy_aligned_by_construction": by_construction, "label_accuracy_good_mismatches": n_mismatch,
            "source_rank_predicts_joint": joint_predicted, "note": note}
