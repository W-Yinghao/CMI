"""Guard (V6-A0a2): even with enough permutable subjects, too few VALID permutations (< PERM_MIN_VALID) forces perm_p to 1.0
(reason insufficient_valid_permutations). sklearn-gated (this path runs the OOF model)."""
from __future__ import annotations
import numpy as np
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5.tests._util import ok, has_sklearn

_PAIRED = ("d_entropy", "d_margin", "flip_rate", "JS", "Bures", "post_sep", "n_eff")


def test_insufficient_valid_permutations_forces_perm_p_one():
    if not has_sklearn():
        ok("sklearn absent — valid-permutation-count path skipped")
        return
    r = np.random.RandomState(0)
    recs = []
    for si in range(24):                                     # 24 subjects size-2 -> 24 permutable (>=20, not underpowered)
        for bi in range(2):
            feats = {"per_action": {a: {k: float(r.randn()) for k in _PAIRED} for a in SP.PRIMARY_ACTIONS},
                     "source_confidence": float(r.rand()), "batch_entropy": float(r.rand()), "batch_size": 32}
            recs.append({"subject_key": f"s{si:02d}", "batch_id": bi, "action_id": "matched_coral", "provenance": "native",
                         "features": feats, "beneficial": int(r.rand() < 0.5)})
    out = SP.permutation_pvalue(recs, ["native"], observed_auroc=0.7, seed=0, n_perm=6)   # only 6 perms << 900
    assert out["reason"] == "insufficient_valid_permutations", out["reason"]
    assert out["perm_p_subject_block"] == 1.0 and out["n_perm_valid"] < SP.PERM_MIN_VALID
    assert out["n_permutable_subjects"] == 24
    ok("too few valid permutations (< 900) -> perm_p=1.0 (insufficient_valid_permutations)")


def main():
    print("ACAR v5 V6-A0a2 guard: valid permutation count required")
    test_insufficient_valid_permutations_forces_perm_p_one()
    print("ALL V6A0A2-PERM-VALID-COUNT GUARDS PASS")


if __name__ == "__main__":
    main()
