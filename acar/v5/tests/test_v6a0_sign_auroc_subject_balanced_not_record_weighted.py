"""Guard (V6-A0a2): the primary sign-AUROC is SUBJECT-BALANCED (per-record weight 1/n_records(subject)), NOT record-weighted. On a
batch-imbalanced fixture the two differ sharply. Uses _auroc directly (sklearn roc_auc_score); gated on sklearn."""
from __future__ import annotations
import numpy as np
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5.tests._util import ok, has_sklearn


def _fixture():
    # subject A: 100 records, PERFECTLY separated; 10 small subjects (1 record each), MAXIMALLY wrong.
    groups = ["A"] * 100 + [f"s{i}" for i in range(10)]
    y = [1] * 50 + [0] * 50 + [1] * 5 + [0] * 5
    scores = [1.0] * 50 + [0.0] * 50 + [0.0] * 5 + [1.0] * 5     # A perfect; small: pos->0, neg->1 (wrong)
    return np.asarray(y, int), np.asarray(scores, float), np.asarray(groups, dtype=object)


def test_subject_balanced_differs_from_record_weighted():
    if not has_sklearn():
        ok("sklearn absent — AUROC weighting path skipped")
        return
    y, scores, groups = _fixture()
    rec = SP._auroc(y, scores, None)                            # record-weighted (A's 100 records dominate)
    bal = SP._auroc(y, scores, SP.subject_weights(groups))     # subject-balanced (each subject weight 1)
    assert rec >= 0.60 and bal < 0.60, f"record={rec:.3f} (should pass) vs subject-balanced={bal:.3f} (should fail)"
    assert rec > bal + 0.2, f"the two must differ sharply on batch-imbalanced data: rec={rec:.3f} bal={bal:.3f}"
    # subject_weights: each subject totals weight 1
    w = SP.subject_weights(groups)
    assert abs(w[:100].sum() - 1.0) < 1e-9 and abs(w[100:101].sum() - 1.0) < 1e-9
    ok("primary sign-AUROC is subject-balanced (each subject weight 1); differs sharply from record-weighted on imbalanced data")


def main():
    print("ACAR v5 V6-A0a2 guard: sign-AUROC subject-balanced, not record-weighted")
    test_subject_balanced_differs_from_record_weighted()
    print("ALL V6A0A2-AUROC-SUBJECT-BALANCED GUARDS PASS")


if __name__ == "__main__":
    main()
