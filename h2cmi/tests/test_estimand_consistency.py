"""Project A Step 17 — tests for estimand-consistent harm control.

Run:  python -m h2cmi.tests.test_estimand_consistency
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.estimand_consistency import build_summary


def _write_run(root, dataset, target, seed, kind):
    if kind == "acc_benefit_bacc_harm":
        # imbalanced: 90 class-0, 10 class-1. adapt predicts all 0 -> accuracy UP, bAcc DOWN vs identity.
        y = [0] * 90 + [1] * 10
        idp = [0] * 70 + [1] * 20 + [1] * 8 + [0] * 2          # acc 0.78, bAcc ~0.789
        adp = [0] * 100                                        # acc 0.90, bAcc 0.5
    else:
        y = [0] * 60 + [1] * 60
        idp, adp = [1] * 120, list(y)                          # adapt clearly better on both
    d = Path(root) / f"dataset={dataset}_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_results.json").write_text(json.dumps({"per_trial_oracle_predictions": {
        "y_true": y, "identity_pred": idp, "adapt_pred": adp}}))
    (d / "run_manifest.json").write_text(json.dumps(
        {"status": "ok", "dataset": dataset, "target_subject": target, "seed": seed, "n_classes": 2}))
    return d


def _summary(root):
    _write_run(root, "BNCI2014_001", 1, 0, "acc_benefit_bacc_harm")
    _write_run(root, "BNCI2014_004", 2, 0, "benefit")
    return build_summary([root], [0, 1, 2, 32, "full"], [0.0], 30, ["iid", "class_balanced"], seed=0)


def _cell(s, estimand, sampling, k, policy, tau=0.0):
    return next(c for c in s["cells"] if c["estimand"] == estimand and c["sampling"] == sampling
                and c["k"] == k and c["policy"] == policy and c["tau"] == tau)


def test_accuracy_and_bacc_gain_can_disagree():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert s["runs_accuracy_benefit_bacc_harm"] >= 1                # the imbalanced run
        assert s["cross_estimand_sign_agreement"] < 1.0
        assert s["accuracy_benefit_rate"] != s["bacc_benefit_rate"]


def test_iid_bacc_estimator_abstains_when_class_missing():
    with tempfile.TemporaryDirectory() as root:
        c = _cell(_summary(root), "balanced_accuracy_gain", "iid", "2", "ci_three_way")
        assert c["missing_class_rate"] > 0                              # small iid slice misses a class


def test_class_balanced_sampling_requires_c13():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert _cell(s, "balanced_accuracy_gain", "class_balanced", "32", "ci_three_way")["requires_contract"] == "C13"
        assert _cell(s, "accuracy_gain", "class_balanced", "32", "ci_three_way")["requires_contract"] is None


def test_k_less_than_n_classes_bacc_abstains():
    with tempfile.TemporaryDirectory() as root:
        c = _cell(_summary(root), "balanced_accuracy_gain", "class_balanced", "1", "ci_three_way")
        assert c["adaptation_coverage"] == 0.0 and c["abstention_rate"] == 1.0   # k<K -> under budget


def test_policy_summary_separates_estimands():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert {c["estimand"] for c in s["cells"]} == {"accuracy_gain", "balanced_accuracy_gain"}


def test_no_cross_estimand_policy_claims():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert s["accuracy_policy_controls_bacc"] is False
        assert "different target functionals" in s["claim_boundary"].lower()


ALL_TESTS = [
    test_accuracy_and_bacc_gain_can_disagree,
    test_iid_bacc_estimator_abstains_when_class_missing,
    test_class_balanced_sampling_requires_c13,
    test_k_less_than_n_classes_bacc_abstains,
    test_policy_summary_separates_estimands,
    test_no_cross_estimand_policy_claims,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} ESTIMAND-CONSISTENCY TESTS PASSED")


if __name__ == "__main__":
    run()
