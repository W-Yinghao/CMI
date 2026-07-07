"""Project A Step 12 — tests for the harm-attribution extractor.

Builds fake audited run dirs and checks the extractor separates R0 / R1 / oracle features, marks the
oracle gain non-identifiable, records missing diagnostics explicitly, and never lets an oracle field
leak into the predictor feature groups. Run:

    python -m h2cmi.tests.test_observability_harm_attribution
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.harm_attribution import (ORACLE_KEYS, _PER_TRIAL_KEYS, build_table,
                                                  extract_run)


def _write_run(root, dataset, target, seed, gain, K=4, status="ok", with_r1_block=False):
    d = Path(root) / f"dataset={dataset}_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    raw = {
        "strict_dg": {"balanced_acc": 0.40},
        "offline_tta": {
            "identity": {"balanced_acc": 0.42},
            "adapt": {"balanced_acc": 0.42 + gain},
            "delta_adapt": {"d_balanced_acc": gain},
            "per_domain_pi_T": {"0": [1.0 / K] * K},
            "per_domain_tta_diagnostics": {"0": {"transform_norm": 0.5, "condition_number": 2.0,
                                                 "delta_density_nll": 0.1, "prior_shift": 0.05,
                                                 "pred_disagreement": 0.2}},
        },
        "leakage": {"site": {"I_hat": 0.0}, "subject": {"I_hat": 0.3, "cond_dom_acc": 0.2},
                    "session": {"I_hat": 0.8}},
        "gate_info": {"mean_pseudo_gain": -0.1},
    }
    if with_r1_block:
        raw["r1_diagnostics"] = {
            "target_prediction_entropy_identity": 0.5, "target_confidence_mean_adapt": 0.6,
            "identity_adapt_prediction_disagreement": 0.15, "source_target_mmd_rbf": 0.02,
            "target_off_source_mass_proxy": 0.1, "target_prior_entropy_hat": 1.3,
            "source_target_centroid_distance": None}          # one null -> recorded missing
        raw["r1_diagnostics_missing"] = {"source_target_centroid_distance": "null (repr failed)"}
        raw["per_trial_oracle_predictions"] = {"y_true": [0, 1], "identity_pred": [0, 0],
                                               "adapt_pred": [0, 1], "identity_confidence": [0.6, 0.6],
                                               "adapt_confidence": [0.7, 0.7], "domain": [0, 0]}
    (d / "raw_results.json").write_text(json.dumps(raw))
    (d / "run_manifest.json").write_text(json.dumps(
        {"status": status, "dataset": dataset, "target_subject": target, "seed": seed,
         "n_classes": K}))
    return d


def test_harm_table_separates_r0_r1_oracle_features():
    with tempfile.TemporaryDirectory() as root:
        rec = extract_run(_write_run(root, "BNCI2014_001", 1, 0, gain=-0.05))
        assert {"r0_features", "r1_features", "oracle_fields", "missing_diagnostics"} <= set(rec)
        assert "source_leakage_subject_I_hat" in rec["r0_features"]
        assert "target_prior_entropy_hat" in rec["r1_features"]
        assert "offline_tta_gain_bacc" in rec["oracle_fields"]
        feat = set(rec["r0_features"]) | set(rec["r1_features"])
        assert feat.isdisjoint(ORACLE_KEYS), "an oracle key leaked into the feature groups"


def test_harm_table_marks_oracle_gain_not_identifiable():
    with tempfile.TemporaryDirectory() as root:
        rec = extract_run(_write_run(root, "BNCI2014_001", 1, 0, gain=-0.05))
        cb = rec["claim_boundary"]
        assert cb["oracle_target_gain_identifiable"] is False
        assert cb["used_for_retrospective_evaluation_only"] is True
        assert rec["oracle_fields"]["target_harmed"] is True         # gain -0.05 < 0
        assert extract_run(_write_run(root, "BNCI2014_004", 2, 0, gain=0.03, K=2))[
            "oracle_fields"]["target_harmed"] is False               # gain +0.03 >= 0


def test_harm_table_missing_diagnostics_are_explicit():
    with tempfile.TemporaryDirectory() as root:
        rec = extract_run(_write_run(root, "BNCI2014_001", 1, 0, gain=-0.05))
        for k in ("tta_confidence_mean", "tta_entropy_mean", "target_support_proxy",
                  "target_marginal_shift_proxy"):
            assert k in rec["missing_diagnostics"], f"{k} not recorded as missing"


def test_harm_table_never_uses_oracle_gain_as_predictor_feature():
    # the oracle gain/harm keys are declared on the denylist and appear only in oracle_fields
    assert {"offline_tta_gain_bacc", "target_harmed"} <= ORACLE_KEYS
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0, gain=-0.05)
        _write_run(root, "BNCI2014_001", 2, 0, gain=0.02)
        _write_run(root, "BNCI2014_001", 3, 0, gain=-0.01, status="skipped")  # excluded
        rows = build_table([root])
        assert len(rows) == 2                                        # skipped run dropped
        for rec in rows:
            feat = set(rec["r0_features"]) | set(rec["r1_features"])
            assert feat.isdisjoint(ORACLE_KEYS)


def test_harm_table_r1_features_include_new_diagnostics_when_present():
    with tempfile.TemporaryDirectory() as root:
        rec = extract_run(_write_run(root, "BNCI2014_001", 1, 0, gain=-0.05, with_r1_block=True))
        assert rec["r1_source"] == "instrumented_r1_diagnostics"
        for k in ("source_target_mmd_rbf", "target_prediction_entropy_identity",
                  "target_off_source_mass_proxy", "target_prior_entropy_hat"):
            assert k in rec["r1_features"], k


def test_harm_table_records_missing_when_old_runs_lack_new_diagnostics():
    with tempfile.TemporaryDirectory() as root:
        legacy = extract_run(_write_run(root, "BNCI2014_001", 1, 0, gain=-0.05, with_r1_block=False))
        assert legacy["r1_source"] == "legacy_per_domain"
        assert "tta_confidence_mean" in legacy["missing_diagnostics"]
    with tempfile.TemporaryDirectory() as root2:
        rich = extract_run(_write_run(root2, "BNCI2014_001", 2, 0, gain=-0.05, with_r1_block=True))
        assert "source_target_centroid_distance" in rich["missing_diagnostics"]  # null field recorded


def test_harm_attribution_does_not_use_per_trial_oracle_predictions_as_r0_or_r1_feature():
    with tempfile.TemporaryDirectory() as root:
        rec = extract_run(_write_run(root, "BNCI2014_001", 1, 0, gain=-0.05, with_r1_block=True))
        feat = set(rec["r0_features"]) | set(rec["r1_features"])
        assert feat.isdisjoint(_PER_TRIAL_KEYS), "a per-trial oracle key leaked into the features"
        assert feat.isdisjoint(ORACLE_KEYS)


def test_harm_table_md_renders_for_instrumented_and_legacy_rows():
    # regression: write_md must not KeyError when r1_features come from the instrumented block
    from h2cmi.observability.harm_attribution import build_table, write_md
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0, gain=-0.05, with_r1_block=True)     # instrumented
        _write_run(root, "BNCI2014_004", 2, 0, gain=0.02, K=2, with_r1_block=False)  # legacy
        rows = build_table([root])
        text = write_md(rows, Path(root) / "t.md")
        assert "harm attribution table" in text and "\r" not in text


def test_r1_prediction_diagnostics_are_label_free():
    # the harness R1 prediction diagnostics take PREDICTIONS only — permuting labels can't change them
    import numpy as np
    from h2cmi.eval.harness import _prediction_diagnostics
    from h2cmi.run_real_audited import _prior_diagnostics, _representation_diagnostics
    proba_id = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    proba_ad = np.array([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]])
    d1 = _prediction_diagnostics(proba_id, proba_ad, 2)
    assert d1 == _prediction_diagnostics(proba_id, proba_ad, 2)   # deterministic, no label input
    assert "target_confidence_mean_identity" in d1
    Zs = np.random.RandomState(0).randn(30, 4); Zt = np.random.RandomState(1).randn(20, 4)
    rep, miss = _representation_diagnostics(Zs, Zt, seed=0)
    assert set(rep) == {"source_target_mmd_rbf", "source_target_centroid_distance",
                        "target_knn_distance_mean", "target_off_source_mass_proxy"} and not miss
    pr = _prior_diagnostics({"0": [0.2, 0.8]}, [0.5, 0.5])
    assert pr["target_prior_shift_l1_from_source"] == round(abs(0.2 - 0.5) + abs(0.8 - 0.5), 6)


ALL_TESTS = [
    test_harm_table_separates_r0_r1_oracle_features,
    test_harm_table_marks_oracle_gain_not_identifiable,
    test_harm_table_missing_diagnostics_are_explicit,
    test_harm_table_never_uses_oracle_gain_as_predictor_feature,
    test_harm_table_r1_features_include_new_diagnostics_when_present,
    test_harm_table_records_missing_when_old_runs_lack_new_diagnostics,
    test_harm_attribution_does_not_use_per_trial_oracle_predictions_as_r0_or_r1_feature,
    test_harm_table_md_renders_for_instrumented_and_legacy_rows,
    test_r1_prediction_diagnostics_are_label_free,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} HARM-ATTRIBUTION TESTS PASSED")


if __name__ == "__main__":
    run()
