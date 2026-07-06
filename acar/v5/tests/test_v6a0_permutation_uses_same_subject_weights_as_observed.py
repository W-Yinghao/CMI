"""Guard (V6-A0a2): the permutation null computes its AUROC with the SAME subject-balanced weights as the observed statistic —
so observed and null are comparable. Monkeypatches _oof_scores/_auroc to capture the weights (no sklearn)."""
from __future__ import annotations
import numpy as np
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5.tests._util import ok

_PAIRED = ("d_entropy", "d_margin", "flip_rate", "JS", "Bures", "post_sep", "n_eff")


def _feat():
    return {"per_action": {a: {k: 0.0 for k in _PAIRED} for a in SP.PRIMARY_ACTIONS},
            "source_confidence": 0.5, "batch_entropy": 0.5, "batch_size": 32}


def _records():
    recs = []
    for si in range(24):                                     # 24 subjects, 2 records each -> all size-2 -> 24 permutable (>=20)
        for bi in range(2):
            recs.append({"subject_key": f"s{si:02d}", "batch_id": bi, "action_id": "matched_coral", "provenance": "native",
                         "features": _feat(), "beneficial": (si + bi) % 2})
    return recs, ["native"]


def test_observed_and_null_use_subject_weights():
    recs, prov = _records()
    captured = []
    orig_oof, orig_auroc = SP._oof_scores, SP._auroc
    SP._oof_scores = lambda X, y, groups, seed: np.zeros(len(y))
    SP._auroc = lambda y, scores, weights=None: (captured.append(weights), 0.5)[1]
    try:
        _, _, groups = SP.design_matrix(recs, prov)
        expected_w = SP.subject_weights(groups)
        SP.primary_sign_auroc(recs, prov, seed=0)                       # observed: 1 weighted call
        SP.permutation_pvalue(recs, prov, 0.5, seed=0, n_perm=3)        # null: 3 weighted calls (24 permutable -> not short-circuit)
    finally:
        SP._oof_scores, SP._auroc = orig_oof, orig_auroc
    assert len(captured) == 4, f"expected 1 observed + 3 null AUROC calls, got {len(captured)}"
    for w in captured:
        assert w is not None, "AUROC must be computed WEIGHTED (subject-balanced), not unweighted"
        assert np.allclose(np.asarray(w, float), expected_w), "observed and null must use the SAME subject weights"
    ok("observed AND permutation-null AUROC both use the same subject-balanced weights")


def main():
    print("ACAR v5 V6-A0a2 guard: permutation uses same subject weights as observed")
    test_observed_and_null_use_subject_weights()
    print("ALL V6A0A2-PERM-SAME-WEIGHTS GUARDS PASS")


if __name__ == "__main__":
    main()
