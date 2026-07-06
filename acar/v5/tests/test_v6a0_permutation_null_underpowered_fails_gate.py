"""Guard (V6-A0a2): a subject-block null with too few permutable subjects is UNDERPOWERED -> perm_p forced to 1.0 (reason
permutation_null_underpowered), so the continuation gate cannot pass on a degenerate null. Short-circuits before sklearn."""
from __future__ import annotations
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5 import v6_a0_report as RPT

_PAIRED = ("d_entropy", "d_margin", "flip_rate", "JS", "Bures", "post_sep", "n_eff")


def _feat():
    return {"per_action": {a: {k: 0.0 for k in _PAIRED} for a in SP.PRIMARY_ACTIONS},
            "source_confidence": 0.5, "batch_entropy": 0.5, "batch_size": 32}


def test_underpowered_null_forces_perm_p_one_and_gate_stops():
    # only 6 subjects -> n_permutable = 6 < PERM_MIN_PERMUTABLE (20) -> underpowered (short-circuit, no sklearn)
    recs = [{"subject_key": f"s{si}", "batch_id": bi, "action_id": "matched_coral", "provenance": "native",
             "features": _feat(), "beneficial": (si + bi) % 2} for si in range(6) for bi in range(2)]
    out = SP.permutation_pvalue(recs, ["native"], observed_auroc=0.99, seed=0, n_perm=1000)
    assert out["perm_p_subject_block"] == 1.0 and out["reason"] == "permutation_null_underpowered"
    assert out["n_permutable_subjects"] == 6 and out["n_perm_valid"] == 0
    from acar.v5.tests._util import ok
    # even with everything else perfect, the underpowered perm_p=1.0 makes the gate STOP
    m = {"oracle_red_upper": 0.9, "beneficial_coverage_subject_macro": 0.9,
         "sign_auroc_subject_balanced": 0.99, "perm_p_subject_block": out["perm_p_subject_block"]}
    assert RPT.continuation_gate({"PD": m, "SCZ": m})[0] == RPT.STOP
    # defense-in-depth: a NaN observed AUROC -> perm_p non-evaluable (never a spurious 'significant' p in the diagnostic report)
    nan_out = SP.permutation_pvalue(recs, ["native"], observed_auroc=float("nan"), seed=0, n_perm=1000)
    assert nan_out["perm_p_subject_block"] == 1.0 and nan_out["reason"] == "observed_auroc_non_evaluable"
    ok("underpowered subject-block null -> perm_p=1.0 (permutation_null_underpowered) -> gate STOP; NaN observed -> non-evaluable")


def main():
    print("ACAR v5 V6-A0a2 guard: underpowered permutation null fails the gate")
    test_underpowered_null_forces_perm_p_one_and_gate_stops()
    print("ALL V6A0A2-PERM-UNDERPOWERED GUARDS PASS")


if __name__ == "__main__":
    main()
