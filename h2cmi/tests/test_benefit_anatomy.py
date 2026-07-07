"""Project A Step 16 — tests for oracle-only benefit anatomy.

Run:  python -m h2cmi.tests.test_benefit_anatomy
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.benefit_anatomy import build_summary


def _write_run(root, dataset, target, seed, kind):
    y = [0] * 60 + [1] * 60                                     # two classes for balanced accuracy
    if kind == "harm":                                         # identity better than adapt
        idp, adp = list(y), [1] * 60 + [1] * 60
    else:                                                      # benefit: adapt better than identity
        idp, adp = [1] * 60 + [0] * 60, list(y)
    d = Path(root) / f"dataset={dataset}_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_results.json").write_text(json.dumps({"per_trial_oracle_predictions": {
        "y_true": y, "identity_pred": idp, "adapt_pred": adp}}))
    (d / "run_manifest.json").write_text(json.dumps(
        {"status": "ok", "dataset": dataset, "target_subject": target, "seed": seed, "n_classes": 2}))
    return d


def _summary(root):
    _write_run(root, "BNCI2014_001", 1, 0, "benefit")
    _write_run(root, "BNCI2014_001", 1, 1, "benefit")
    _write_run(root, "BNCI2014_004", 2, 0, "harm")
    return build_summary([root])


def test_benefit_anatomy_marks_oracle_only():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert all(r["oracle_only"] is True for r in s["runs"])
        assert "oracle-only" in s["claim_boundary"].lower()


def test_benefit_anatomy_counts_benefit_harm_near_zero():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert s["n_beneficial"] + s["n_harmful"] + s["n_near_zero"] == s["n_runs"] == 3
        assert s["n_beneficial"] == 2 and s["n_harmful"] == 1
        assert s["benefit_rate"] == round(2 / 3, 4)


def test_benefit_anatomy_groups_by_dataset_target_seed():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        assert "BNCI2014_001" in s["per_dataset"] and "BNCI2014_004" in s["per_dataset"]
        assert "BNCI2014_001:1" in s["per_target"] and s["per_target"]["BNCI2014_001:1"]["n_seeds"] == 2
        assert s["per_target"]["BNCI2014_001:1"]["sign_consistent"] is True    # both seeds beneficial
        assert set(s["per_seed"]) == {"0", "1"}


def test_benefit_anatomy_does_not_export_features_as_r0_or_r1_observable():
    with tempfile.TemporaryDirectory() as root:
        s = _summary(root)
        cb = s["claim_boundary"].lower()
        assert "not deployment-observable under r0/r1" in cb
        assert "never as a predictor or deployable signal" in cb


ALL_TESTS = [
    test_benefit_anatomy_marks_oracle_only,
    test_benefit_anatomy_counts_benefit_harm_near_zero,
    test_benefit_anatomy_groups_by_dataset_target_seed,
    test_benefit_anatomy_does_not_export_features_as_r0_or_r1_observable,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} BENEFIT-ANATOMY TESTS PASSED")


if __name__ == "__main__":
    run()
