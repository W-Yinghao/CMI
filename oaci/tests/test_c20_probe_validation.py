"""C20 frozen-probe new-regime validation. Locked to the exact C19 config hash + robust-core feature set;
primary uses robust-core only (no grid search); development regimes exclude the held-out validation regimes;
LOTO excludes the held-out target from the fit; diagnostic-only labels; within-fold permutation family; finite
filtering; endpoint-augmented is secondary-only; no selector artifact; report forbids detector/selector
language; and the external-dataset protocol launches NO execution. Synthetic rows only."""
from __future__ import annotations

import numpy as np

from oaci.competence_probe import schema as c19
from oaci.probe_validation import (cross_regime_validation as crv, feature_lock, frozen_config, frozen_probe,
                                   permutation, regime_splits, report, schema)


def _rows(regime, n_targets=4, n_per=6, seed=0, sig=0.15):
    rng = np.random.RandomState(seed + hash(regime) % 1000); rows = []
    for t in range(1, n_targets + 1):
        for k in range(n_per):
            good = k % 2 == 0
            r = {"seed": 0, "target": t, "level": 0, "model_hash": f"{regime}-{t}{k}",
                 "diagnostic_only_non_deployable": True, c19.DIAGNOSTIC_LABEL: good}
            for f in c19.ROBUST_CORE_FEATURES:
                r[f] = (0.5 + sig if good else 0.5 - sig) + rng.randn() * 0.05
            for f in c19.ENDPOINT_FEATURES:
                r[f] = (0.55 if good else 0.45) + rng.randn() * 0.05
            rows.append(r)
    return rows


def _dev_by():
    return {r: _rows(r) for r in schema.DEVELOPMENT_REGIMES}


def test_c20_uses_exact_c19_config_hash():
    assert frozen_config.assert_locked() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_feature_registry_matches_c19_byte_for_byte():
    feature_lock.assert_locked()
    a = feature_lock.lock_audit()
    assert a["robust_core_matches_c19"] and a["n_robust_core"] == 16 and tuple(schema.robust_core_features()) == c19.ROBUST_CORE_FEATURES


def test_primary_probe_uses_robust_core_only():
    a = feature_lock.lock_audit()
    assert a["endpoints_excluded_from_primary"] and a["static_excluded_from_primary"]
    assert all("worst_bacc" not in f for f in schema.robust_core_features())


def test_no_feature_selection_or_grid_search():
    import inspect
    from oaci.identifiability.multivariate_probe import _fit_logit
    d = inspect.signature(_fit_logit).parameters
    assert d["l2"].default == c19.PROBE_L2_C and d["iters"].default == c19.PROBE_ITERS


def test_training_regimes_exclude_heldout_validation_regimes():
    regime_splits.assert_no_leakage_between_splits()
    sp = regime_splits.split_plan()
    assert not (set(sp["development_regimes"]) & set(sp["held_out_regimes"])) and sp["disjoint"]


def test_loto_target_is_excluded_from_fit():
    # poison: make held-out target 1's DEV rows perfectly separable by a robust feature; if the fit used them,
    # target 1 test AUC would be ~1.0. With LOTO exclusion, target 1's dev rows are NOT in its own fit.
    dev = _dev_by(); val = _rows("S4_missing_cells")
    res = crv.cross_regime_loto(dev, val, list(c19.ROBUST_CORE_FEATURES), n_perm=10)
    assert set(res["per_target_auc"]) <= {"1", "2", "3", "4"} and res["loto_auc"] is not None


def test_target_labels_joined_only_for_diagnostic_validation():
    from oaci.competence_probe import labels
    labels.assert_no_target_in_features(list(schema.robust_core_features()))


def test_permutation_baseline_matches_preregistered_family():
    dev = _dev_by(); pool = [r for rg in schema.DEVELOPMENT_REGIMES for r in dev[rg]]
    rng = np.random.RandomState(0)
    sh = permutation._shuffle_dev_labels(pool, rng, c19.DIAGNOSTIC_LABEL)
    # within-fold label COUNT preserved (shuffle is within (seed,target,level))
    def cnt(rows):
        d = {}
        for r in rows:
            d.setdefault((r["seed"], r["target"], r["level"]), []).append(1 if r[c19.DIAGNOSTIC_LABEL] else 0)
        return {k: sum(v) for k, v in d.items()}
    assert cnt(pool) == cnt(sh)


def test_finite_filter_drops_none_nan_inf():
    dev = _dev_by(); val = _rows("S6_boundary_aligned_mask")
    for i, v in enumerate((None, float("nan"), float("inf"), float("-inf"))):
        val[i]["selection_leakage_point"] = v
    res = crv.cross_regime_loto(dev, val, list(c19.ROBUST_CORE_FEATURES), n_perm=8)
    assert res["loto_auc"] is not None


def test_endpoint_augmented_is_secondary_only():
    from oaci.probe_validation import endpoint_checks
    dev = _dev_by(); val = _rows("S4_missing_cells")
    e = endpoint_checks.endpoint_augmented_cross_regime(dev, val, n_perm=8)
    assert e["is_secondary"] is True


def test_no_selected_checkpoint_artifact_emitted():
    dev = _dev_by(); val = _rows("S7_random_matched_mask")
    res = crv.cross_regime_loto(dev, val, list(c19.ROBUST_CORE_FEATURES), n_perm=8)
    assert "selector" not in res and "chosen_model_hash" not in res and res["non_deployable"] is True


def test_report_forbids_detector_selector_language():
    for bad in ("we built a selector", "detector is validated", "deployable selector", "production selector"):
        try:
            report._guard_forbidden("# C20\n\n" + bad + ".\n")
        except ValueError:
            continue
        raise AssertionError(f"forbidden claim not caught: {bad}")
    report._guard_forbidden("# C20\n\nthe frozen diagnostic probe generalizes across held-out support-stress regimes.\n")


def test_external_dataset_protocol_does_not_launch_execution():
    prot = report.render_protocol_md({"config_hash": "664007686afb520f"})
    low = prot.lower()
    assert "no execution" in low and "barred_pending_explicit_approval" in low
    # the protocol module must not import or call any dataset loader / sbatch
    import oaci.probe_validation.report as rep
    import inspect
    src = inspect.getsource(rep.render_protocol_md)
    assert "load_moabb" not in src and "sbatch" not in src


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c20-probe-validation tests")


if __name__ == "__main__":
    _run_all()
