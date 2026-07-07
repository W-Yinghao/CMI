"""Project A Step 18 — tests for the TTA harm-mechanism decomposition.

Run:  python -m h2cmi.tests.test_harm_mechanisms
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from h2cmi.observability.harm_mechanisms import _per_run, build_summary


def _write_run(root, dataset, target, seed, y, ip, ap):
    d = Path(root) / f"dataset={dataset}_target={target}_seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_results.json").write_text(json.dumps({"per_trial_oracle_predictions": {
        "y_true": y, "identity_pred": ip, "adapt_pred": ap}}))
    (d / "run_manifest.json").write_text(json.dumps(
        {"status": "ok", "dataset": dataset, "target_subject": target, "seed": seed}))
    return d


def _mixed_run():
    # class 0: identity all wrong->adapt all right (recall +1); class 1: identity all right->adapt all
    # wrong (recall -1). Mixed class effects; prior-dependent sign.
    y = [0] * 10 + [1] * 10
    ip = [1] * 10 + [1] * 10          # class0 predicted 1 (wrong); class1 predicted 1 (right)
    ap = [0] * 10 + [0] * 10          # class0 predicted 0 (right); class1 predicted 0 (wrong)
    return y, ip, ap


def test_gain_loss_decomposition_matches_accuracy_gain():
    m = {"dataset": "D", "target_subject": 1, "seed": 0}
    y, ip, ap = _mixed_run()
    r = _per_run({"y_true": y, "identity_pred": ip, "adapt_pred": ap}, m)
    # net_gain_from_gain_loss must equal accuracy_gain exactly
    assert abs(r["net_gain_from_gain_loss"] - r["accuracy_gain"]) < 1e-9
    assert abs(r["gained_correct_rate"] - r["lost_correct_rate"] - r["accuracy_gain"]) < 1e-9


def test_bacc_gain_equals_mean_per_class_recall_delta():
    m = {"dataset": "D", "target_subject": 1, "seed": 0}
    y, ip, ap = _mixed_run()
    r = _per_run({"y_true": y, "identity_pred": ip, "adapt_pred": ap}, m)
    deltas = [r["per_class"][c]["recall_delta"] for c in r["per_class"]]
    assert abs(r["bacc_gain"] - sum(deltas) / len(deltas)) < 1e-9
    # here class deltas are +1 and -1 -> bAcc gain 0
    assert abs(r["bacc_gain"]) < 1e-9


def test_mixed_class_effect_detects_positive_and_negative_class_deltas():
    m = {"dataset": "D", "target_subject": 1, "seed": 0}
    y, ip, ap = _mixed_run()
    hc = _per_run({"y_true": y, "identity_pred": ip, "adapt_pred": ap}, m)["harm_channel_summary"]
    assert hc["mixed_class_effects"] is True and hc["prior_dependent_possible"] is True
    assert hc["worst_class_recall_delta"] < 0 < hc["best_class_recall_delta"]


def test_uniform_effect_is_not_mixed():
    # both classes improve -> not mixed, not prior-dependent
    m = {"dataset": "D", "target_subject": 1, "seed": 0}
    y = [0] * 10 + [1] * 10
    ip = [1] * 5 + [0] * 5 + [0] * 5 + [1] * 5     # both classes ~half wrong
    ap = [0] * 10 + [1] * 10                        # both classes fully correct
    hc = _per_run({"y_true": y, "identity_pred": ip, "adapt_pred": ap}, m)["harm_channel_summary"]
    assert hc["mixed_class_effects"] is False and hc["prior_dependent_possible"] is False


def test_harm_mechanism_marked_oracle_evaluation_only():
    with tempfile.TemporaryDirectory() as root:
        y, ip, ap = _mixed_run()
        _write_run(root, "D", 1, 0, y, ip, ap)
        s = build_summary([root])
        assert s["oracle_labels_used_only_for_mechanism_and_evaluation"] is True
        assert s["runs"][0]["oracle_evaluation_only"] is True
        assert "oracle" in s["runs"][0]["claim_boundary"].lower()


def test_no_r0_or_r1_claim_from_harm_mechanism():
    with tempfile.TemporaryDirectory() as root:
        y, ip, ap = _mixed_run()
        _write_run(root, "D", 1, 0, y, ip, ap)
        s = build_summary([root])
        blob = (s["scope"] + " " + s["claim_boundary"]).lower()
        assert "not r0/r1 identif" in blob or "no target functional" in blob or "identifies no" in blob
        assert "not sota" in s["scope"].lower() or "no sota" in s["claim_boundary"].lower()


ALL_TESTS = [
    test_gain_loss_decomposition_matches_accuracy_gain,
    test_bacc_gain_equals_mean_per_class_recall_delta,
    test_mixed_class_effect_detects_positive_and_negative_class_deltas,
    test_uniform_effect_is_not_mixed,
    test_harm_mechanism_marked_oracle_evaluation_only,
    test_no_r0_or_r1_claim_from_harm_mechanism,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} HARM-MECHANISM TESTS PASSED")


if __name__ == "__main__":
    run()
