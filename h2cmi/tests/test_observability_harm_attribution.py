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

from h2cmi.observability.harm_attribution import ORACLE_KEYS, build_table, extract_run


def _write_run(root, dataset, target, seed, gain, K=4, status="ok"):
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


ALL_TESTS = [
    test_harm_table_separates_r0_r1_oracle_features,
    test_harm_table_marks_oracle_gain_not_identifiable,
    test_harm_table_missing_diagnostics_are_explicit,
    test_harm_table_never_uses_oracle_gain_as_predictor_feature,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} HARM-ATTRIBUTION TESTS PASSED")


if __name__ == "__main__":
    run()
