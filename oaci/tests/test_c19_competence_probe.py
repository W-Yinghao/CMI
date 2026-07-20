"""C19 pre-registered source-only competence probe. Frozen feature registry (robust-core / endpoint /
static-excluded); estimability gate reason-codes; diagnostic-only labels joined post-hoc; LOTO + within-fold
permutation; fixed regularization (no grid search); finite-filter; no selector artifact; forbidden-claim
guard; and pre-registration config matches the executed config. Synthetic rows only (no real-data peek)."""
from __future__ import annotations

import inspect

import numpy as np

from oaci.competence_probe import estimability_gate, labels, probe, report, schema, validation


def _synth(n_targets=4, n_per=8, seed=0, endpoint_nan_frac=0.0):
    rng = np.random.RandomState(seed); rows = []
    for t in range(1, n_targets + 1):
        for k in range(n_per):
            good = k % 2 == 0
            r = {"seed": 0, "target": t, "level": 0, "model_hash": f"m{t}{k}",
                 "diagnostic_only_non_deployable": True, schema.DIAGNOSTIC_LABEL: good}
            for f in schema.ROBUST_CORE_FEATURES:
                r[f] = (0.6 if good else 0.4) + rng.randn() * 0.05
            for f in schema.ENDPOINT_FEATURES:
                r[f] = (float("nan") if rng.rand() < endpoint_nan_frac else (0.55 if good else 0.45) + rng.randn() * 0.05)
            for f in schema.STATIC_EXCLUDED:
                r[f] = float(rng.randn())
            rows.append(r)
    return rows


def test_feature_registry_is_frozen_before_probe_fit():
    # exactly 16 robust-core (7 conf-geom x 2 roles + 2 leakage), 2 endpoint, 4 static; disjoint families
    assert len(schema.ROBUST_CORE_FEATURES) == 16 and len(schema.ENDPOINT_FEATURES) == 2
    assert not (set(schema.ROBUST_CORE_FEATURES) & set(schema.ENDPOINT_FEATURES))
    assert not (set(schema.ROBUST_CORE_FEATURES) & set(schema.STATIC_EXCLUDED))


def test_static_training_columns_excluded_from_primary_probe():
    assert set(schema.STATIC_EXCLUDED) == {"R_src", "balanced_err", "train_surrogate", "epoch"}
    assert all(f not in schema.ROBUST_CORE_FEATURES for f in schema.STATIC_EXCLUDED)


def test_fragile_accuracy_endpoints_excluded_from_primary_probe():
    assert all(f not in schema.ROBUST_CORE_FEATURES for f in schema.ENDPOINT_FEATURES)
    assert all("worst_bacc" not in f for f in schema.ROBUST_CORE_FEATURES)


def test_endpoint_estimability_gate_reason_codes_missing_features():
    rows = _synth(endpoint_nan_frac=1.0)                     # all endpoints NaN
    g = estimability_gate.gate_summary(rows, list(schema.ROBUST_CORE_FEATURES), list(schema.ENDPOINT_FEATURES))
    assert g["endpoint_nonestimable_rate"] == 1.0 and g["scored_rate"] == 0.0
    # robust-core alone: all scored (robust features finite)
    g2 = estimability_gate.gate_summary(rows, list(schema.ROBUST_CORE_FEATURES))
    assert g2["scored_rate"] == 1.0
    # a NaN robust feature -> insufficient_finite
    bad = _synth(); bad[0]["source_guard_nll"] = float("nan")
    assert estimability_gate.score_status(bad[0], list(schema.ROBUST_CORE_FEATURES)) == "abstained_insufficient_finite_features"


def test_target_labels_joined_only_for_diagnostic_validation():
    labels.assert_no_target_in_features(list(schema.ROBUST_CORE_FEATURES))
    labels.assert_no_target_in_features(list(schema.ENDPOINT_FEATURES))
    try:
        labels.assert_no_target_in_features(["tgt__target_bacc_good"])
    except ValueError:
        pass
    else:
        raise AssertionError("target column must be rejected from the feature set")
    assert labels.assert_diagnostic_only(_synth()) is None


def test_probe_uses_loto_validation_and_permutation_baseline():
    m = probe.run_probe(_synth(), list(schema.ROBUST_CORE_FEATURES), n_perm=25)
    assert set(m["per_target_auc"]) <= {"1", "2", "3", "4"}          # per held-out TARGET (LOTO)
    assert m["permutation_p"] is not None and m["permutation_mean_auc"] is not None
    assert m["loto_auc"] is not None and m["non_deployable"] is True


def test_probe_has_fixed_regularization_no_grid_search():
    from oaci.identifiability.multivariate_probe import _fit_logit
    d = inspect.signature(_fit_logit).parameters
    assert d["l2"].default == schema.PROBE_L2_C and d["iters"].default == schema.PROBE_ITERS and d["lr"].default == schema.PROBE_LR


def test_finite_filter_drops_none_nan_inf():
    rows = _synth()
    for i, v in enumerate((None, float("nan"), float("inf"), float("-inf"))):
        rows[i]["selection_leakage_point"] = v
    m = probe.run_probe(rows, list(schema.ROBUST_CORE_FEATURES), n_perm=15)
    assert m["loto_auc"] is not None                                 # non-finite rows dropped, not crashing


def test_no_selected_checkpoint_artifact_emitted():
    m = probe.run_probe(_synth(), list(schema.ROBUST_CORE_FEATURES), n_perm=15)
    assert "selector" not in m and "chosen_model_hash" not in m and m["non_deployable"] is True


def test_report_forbids_deployable_selector_claim():
    for bad in ("we built a selector", "deployable target-free selector", "OACI is rescued",
                "target oracle is deployable"):
        try:
            report._guard_forbidden("# C19\n\n" + bad + ".\n")
        except ValueError:
            continue
        raise AssertionError(f"forbidden claim not caught: {bad}")
    report._guard_forbidden("# C19\n\na pre-registered low-freedom diagnostic probe recovers weak competence information.\n")


def test_c19_preregistration_matches_executed_config():
    c = schema.frozen_config()
    assert c["l2_C"] == schema.PROBE_L2_C and c["n_perm"] == schema.N_PERM and c["validation"] == "leave_one_target_out"
    h1 = report._config_hash(); h2 = report._config_hash()
    assert h1 == h2 and len(h1) == 16                                # deterministic pre-registration hash


def test_taxonomy_diagnostic_only_and_no_detector_overclaim():
    # robust-core passes on all success regimes -> recovers; but with high per-target spread -> heterogeneous
    robust_by = {r: {"passes": True} for r in schema.SUCCESS_REGIMES}
    hetero = {"heterogeneous": True, "spread": 0.5, "min": 0.35, "max": 0.85}
    t = report._taxonomy({"primary_success": True}, robust_by, {}, hetero)
    assert t["case_label"] == schema.CASE_HETEROGENEOUS and t["diagnostic_only_non_deployable"]
    assert "detector" not in t["next_science"].lower() or "not" in t["next_science"].lower()


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c19-competence-probe tests")


if __name__ == "__main__":
    _run_all()
