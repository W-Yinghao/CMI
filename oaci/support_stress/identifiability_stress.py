"""C18-P — per-regime identifiability stress. TWO probes, kept distinct (per the recomputable-column split):

  identity_probe   : S0 all-column atlas from the EXTRACTED authoritative S0 scalars -> must reproduce C17
                     Case III (G1) and AUC 0.602 (G2), oracle rho ~ +0.120 (G3), no strong scalar (G4).
  mask_stress_probe: per regime S0-S7, RECOMPUTABLE-column atlas (masked recompute; static training-log cols
                     excluded) -> LOTO AUC + permutation baseline. The recomputable-column S0 is a DECLARED
                     new baseline (it differs from 0.602 because the 3 weak scalar signals were static).

Target labels are joined POST HOC (G5) and every fit is finite-filtered (G6, in the C17 probe)."""
from __future__ import annotations

from ..identifiability.multivariate_probe import multivariate_probe
from ..identifiability.univariate import univariate_identifiability
from . import feature_inventory, schema
from . import source_signal_recompute as ssr


def _nan_out_static(rows):
    """Set static training-log src__ columns to NaN so the probe's all-NaN-column drop removes them: the
    mask-stress probe uses ONLY recomputable-under-mask features."""
    static = set("src__" + s for s in schema.STATIC_TRAINING_LOG_ONLY)
    out = []
    for r in rows:
        rr = dict(r)
        for c in static:
            rr[c] = float("nan")
        out.append(rr)
    return out


def _assert_no_target_features(rows) -> None:
    """G5: no tgt__ column may enter the feature set (targets are labels only)."""
    if rows:
        feat_like = [k for k in rows[0] if k.startswith("src__")]
        if any(k.startswith("tgt__") for k in feat_like):
            raise ValueError("target column leaked into source feature set")


def identity_probe(extract_dir, c10_dir, *, n_perm=None, folds=None) -> dict:
    rows = ssr.build_identity_atlas(extract_dir, c10_dir, folds=folds)
    _assert_no_target_features(rows)
    u = univariate_identifiability(rows)
    m = multivariate_probe(rows) if n_perm is None else multivariate_probe(rows, n_perm=n_perm)
    auc, rho = m["loto_auc"], u["oracle_signal_spearman_bacc"]
    return {"n_rows": len(rows), "loto_auc": auc, "beats_permutation": m["beats_permutation"],
            "permutation_p": m["permutation_p"], "univariate_verdict": u["univariate_verdict"],
            "oracle_spearman_bacc": rho, "n_strong": u["n_strong_accuracy_signals"],
            "n_weak": u["n_weak_accuracy_signals"], "max_abs_accuracy_spearman": u["max_abs_accuracy_spearman"],
            "G1_s0_reproduces_case_iii": bool(u["univariate_verdict"] == "weak_accuracy_needs_multivariate"
                                              and m["beats_permutation"]),
            "G2_auc_reproduces_0602": bool(auc is not None and abs(auc - schema.C17_LOTO_AUC) <= schema.G2_AUC_TOL),
            "G3_oracle_rho_reproduces": bool(rho is not None
                                             and abs(rho - schema.C17_ORACLE_SPEARMAN_BACC) <= schema.G3_RHO_TOL),
            "G4_no_strong_scalar": bool(u["n_strong_accuracy_signals"] == 0)}


def mask_stress_probe(extract_dir, c10_dir, regime, *, boundary_classes, n_perturb=2, n_perm=None, folds=None) -> dict:
    rows = ssr.build_regime_atlas(extract_dir, c10_dir, regime, boundary_classes=boundary_classes,
                                  n_perturb=n_perturb, folds=folds)
    _assert_no_target_features(rows)
    feature_inventory.assert_only_recomputable_used(["src__" + s for s in feature_inventory.recomputable_features()])
    rows_r = _nan_out_static(rows)
    u = univariate_identifiability(rows_r)
    m = multivariate_probe(rows_r) if n_perm is None else multivariate_probe(rows_r, n_perm=n_perm)
    return {"regime": regime, "n_rows": len(rows), "n_used": m["n_used"], "n_features": m["n_features"],
            "loto_auc": m["loto_auc"], "loso_auc": m["loso_auc"], "permutation_p": m["permutation_p"],
            "permutation_mean_auc": m["permutation_mean_auc"], "beats_permutation": m["beats_permutation"],
            "base_rate": m["base_rate"], "univariate_verdict": u["univariate_verdict"],
            "n_weak_accuracy": u["n_weak_accuracy_signals"], "max_abs_accuracy_spearman": u["max_abs_accuracy_spearman"]}


def run_all(extract_dir, c10_dir, *, boundary_classes, n_perturb=2, n_perm=None, folds=None) -> dict:
    identity = identity_probe(extract_dir, c10_dir, n_perm=n_perm, folds=folds)
    per_regime = {r: mask_stress_probe(extract_dir, c10_dir, r, boundary_classes=boundary_classes,
                                       n_perturb=n_perturb, n_perm=n_perm, folds=folds)
                  for r in schema.REGIME_ORDER}
    return {"identity_probe": identity, "recomputable_column_s0_baseline": per_regime["S0_full_support"],
            "mask_stress_by_regime": per_regime,
            "note": ("identity_probe = extracted all-column S0 (reproduces C17 0.602); mask_stress = "
                     "recomputable-column atlas per regime (declared new S0 baseline; static scalars excluded).")}
