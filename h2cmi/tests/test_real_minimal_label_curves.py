"""Project A Step 13 — tests for the real minimal-label curves.

Builds fake run dirs with per_trial_oracle_predictions and checks k=0 is the R1 non-identifiability
boundary, k>0 is an R2 labeled slice under an iid sampling contract, the module never claims
full-target identification, and oracle labels are used only for the R2 slice / evaluation. Run:

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


def test_real_minimal_labels_k0_is_r1_nonidentified():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 8, 32], repeats=50, seed=0)
        k0 = s["per_k"]["0"]
        assert k0["identified_status"] == "not_identified_R1"
        assert k0["harm_sign_accuracy"] == 0.5 and k0["abstention_rate"] == 1.0
        assert s["k0_status"] == "not_identified_R1"


def test_real_minimal_labels_k_positive_is_r2_labeled_slice():
    with tempfile.TemporaryDirectory() as root:
        _write_run(root, "BNCI2014_001", 1, 0)
        s = _curve([root], [0, 8, 64], repeats=80, seed=0)
        r = s["per_k"]["64"]
        assert r["identified_status"] == "r2_labeled_slice_under_iid_sampling_contract"
        assert "sampling contract" in r["claim_boundary"].lower()
        # harm-sign accuracy rises with k (more labels -> more decisive-correct)
        assert s["per_k"]["64"]["harm_sign_accuracy"] >= s["per_k"]["8"]["harm_sign_accuracy"]


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
    test_real_minimal_labels_k0_is_r1_nonidentified,
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
