"""Project A Step 12 — tests for the retrospective harm predictor.

Builds a fake harm-attribution table and checks the oracle harm label never enters the feature
matrix, R0 uses only source features, R1 adds target-unlabeled features (no labels), and results are
labelled retrospective-not-identifiability. Run:

    python -m h2cmi.tests.test_observability_harm_predictor
"""
from __future__ import annotations

from h2cmi.observability.harm_attribution import ORACLE_KEYS
from h2cmi.observability.harm_predictor import build_summary


def _row(dataset, target, seed, harmed, leak, ent):
    g = -0.05 if harmed else 0.02
    return {"dataset": dataset, "target_subject": target, "seed": seed, "n_classes": 4,
            "offline_tta_harmed": harmed,
            "r0_features": {"source_leakage_subject_I_hat": leak,
                            "source_mean_pseudo_gain": -0.1 - 0.01 * target},
            "r1_features": {"target_prior_entropy_hat": ent,
                            "tta_transform_norm_mean": 0.5 + 0.02 * seed + 0.01 * target},
            "oracle_fields": {"offline_tta_gain_bacc": g, "target_harmed": harmed},
            "missing_diagnostics": []}


def _table():
    rows = [_row("D1", t, s, harmed=(t != 3), leak=0.1 * t + 0.03 * s, ent=1.0 + 0.1 * t)
            for t in (1, 2, 3) for s in (0, 1)]
    return {"runs": rows}


def test_oracle_harm_label_not_in_predictor_features():
    s = build_summary(_table())
    for fs in s["feature_sets"].values():
        for feat in fs["features"]:
            assert feat not in ORACLE_KEYS, f"oracle key {feat} used as feature"
    assert s["oracle_never_a_feature"] is True


def test_r0_predictor_uses_only_r0_features():
    s = build_summary(_table())
    r0 = set(s["feature_sets"]["R0_source_only"]["features"])
    assert r0 and all(f.startswith("source_") for f in r0), r0


def test_r1_predictor_uses_no_target_labels():
    s = build_summary(_table())
    r1 = set(s["feature_sets"]["R1_target_unlabeled"]["features"])
    assert r1.isdisjoint(ORACLE_KEYS)
    assert any(f.startswith("target_") or f.startswith("tta_") for f in r1), r1


def test_predictor_summary_labels_results_retrospective_not_identifiable():
    s = build_summary(_table())
    for fs in s["feature_sets"].values():
        assert "retrospective" in fs["claim"].lower() and "identif" in fs["claim"].lower()
    assert "retrospective" in s["claim_boundary"].lower()
    assert s["majority_baseline_balanced_acc"] == 0.5


ALL_TESTS = [
    test_oracle_harm_label_not_in_predictor_features,
    test_r0_predictor_uses_only_r0_features,
    test_r1_predictor_uses_no_target_labels,
    test_predictor_summary_labels_results_retrospective_not_identifiable,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} HARM-PREDICTOR TESTS PASSED")


if __name__ == "__main__":
    run()
