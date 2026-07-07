"""Project A Step 13/14 — tests for the real minimal-label curves (coverage-decomposed).

Builds fake run dirs with per_trial_oracle_predictions and checks: k=0 is R1 non-identifiable with
NULL accuracy (not 0.5); k>0 is an R2 labeled slice under an iid sampling contract; the metric is
decomposed into coverage (decisive_rate) vs accuracy (conditional/unconditional); conditional accuracy
is defined only when decisive; the module never claims full-target identification; oracle labels are
used only for the R2 slice. Run:

    python -m h2cmi.tests.test_real_minimal_label_curves
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.real_minimal_labels import _curve


def _write_run(root, dataset, target, seed):
    # 100 trials with a clear (noisy) accuracy HARM: 40 d=-1, 40 d=0, 20 d=+1 -> mean -0.2
    y = [0] * 100
    id_pred = [0] * 40 + [0] * 40 + [1] * 20          # correct, correct, wrong
    ad_pred = [1] * 40 + [0] * 40 + [0] * 20          # wrong, correct, correct
    d = Path(root) / f"dataset={dataset}_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    raw = {"per_trial_oracle_predictions": {
        "target_trial_index": list(range(100)), "y_true": y, "identity_pred": id_pred,
        "adapt_pred": ad_pred, "identity_confidence": [0.6] * 100, "adapt_confidence": [0.6] * 100,
        "domain": [0] * 100}}
    (d / "raw_results.json").write_text(json.dumps(raw))
    (d / "run_manifest.json").write_text(json.dumps(
        {"status": "ok", "dataset": dataset, "target_subject": target, "seed": seed, "n_classes": 2}))
    return d


def test_k0_has_null_accuracy_not_random_guess():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 8, 32], repeats=50, seed=0)
        k0 = s["per_k"]["0"]
        assert k0["identified_status"] == "not_identified_R1"
        assert k0["unconditional_correct_rate"] is None        # NULL, not 0.5
        assert k0["conditional_accuracy_given_decisive"] is None
        assert k0["decisive_rate"] == 0.0 and k0["abstention_rate"] == 1.0
        assert s["k0_status"] == "not_identified_R1"


def test_minimal_label_curve_decomposes_accuracy_and_coverage():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 8, 64], repeats=80, seed=0)
        r = s["per_k"]["64"]
        for key in ("decisive_rate", "unconditional_correct_rate",
                    "conditional_accuracy_given_decisive", "abstention_rate", "coverage"):
            assert key in r
        # coverage + abstention == 1
        assert abs(r["decisive_rate"] + r["abstention_rate"] - 1.0) < 1e-6
        assert "harm_sign_accuracy_deprecated_per_k" in s      # deprecated alias present, not primary
        assert "oracle_full_target_sign_distribution" in s["baselines"]


def test_conditional_accuracy_defined_only_when_decisive():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 64], repeats=80, seed=0)
        assert s["per_k"]["0"]["conditional_accuracy_given_decisive"] is None      # 0 decisive -> null
        r = s["per_k"]["64"]
        if r["decisive_rate"] > 0:
            assert r["conditional_accuracy_given_decisive"] is not None
            assert 0.0 <= r["conditional_accuracy_given_decisive"] <= 1.0


def test_real_minimal_labels_k_positive_is_r2_labeled_slice():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 8, 64], repeats=80, seed=0)
        r = s["per_k"]["64"]
        assert r["identified_status"] == "r2_labeled_slice_under_iid_sampling_contract"
        assert "sampling contract" in r["claim_boundary"].lower()
        assert s["per_k"]["64"]["decisive_rate"] >= s["per_k"]["8"]["decisive_rate"]  # coverage rises with k


def test_real_minimal_labels_does_not_claim_full_target_identification():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 32], repeats=20, seed=0)
        blob = (s["claim_boundary"] + " " + " ".join(v["claim_boundary"]
                for v in s["per_k"].values())).lower()
        assert "full target risk identified" not in blob
        assert "k labels identify full target" not in blob
        assert s["claim_boundary_ok"] is True


def test_real_minimal_labels_uses_oracle_labels_only_for_r2_slice_and_evaluation():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 8], repeats=10, seed=0)
        assert s["oracle_labels_used_only_for_r2_slice_and_evaluation"] is True
        assert s["n_runs"] == 1


ALL_TESTS = [
    test_k0_has_null_accuracy_not_random_guess,
    test_minimal_label_curve_decomposes_accuracy_and_coverage,
    test_conditional_accuracy_defined_only_when_decisive,
    test_real_minimal_labels_k_positive_is_r2_labeled_slice,
    test_real_minimal_labels_does_not_claim_full_target_identification,
    test_real_minimal_labels_uses_oracle_labels_only_for_r2_slice_and_evaluation,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} REAL-MINIMAL-LABEL TESTS PASSED")


if __name__ == "__main__":
    run()
