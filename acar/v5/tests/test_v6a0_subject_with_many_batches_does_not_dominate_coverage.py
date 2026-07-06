"""Guard (V6-A0a2): one batch-rich beneficial subject cannot make disease-level coverage pass — the subject-macro coverage stays
balanced (< 0.15, gate FAILS) even though the batch-weighted coverage would pass (>= 0.15). Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import v6_a0_action_viability as AV
from acar.v5 import v6_a0_report as RPT
from acar.v5.tests._util import ok


def _rec(sk, best_negative):
    return {"subject_key": sk, "delta_r": {"matched_coral": (-1.0 if best_negative else 1.0), "spdim": 1.0, "t3a": 1.0}}


def test_batch_rich_subject_does_not_dominate_coverage_gate():
    # 1 batch-rich subject (41 beneficial batches) + 9 sparse subjects (1 batch each, none beneficial)
    records = [_rec("rich", True) for _ in range(41)] + [_rec(f"s{i}", False) for i in range(9)]
    env = AV.oracle_envelope(records)
    # subject-macro = (1.0 + 9*0.0)/10 = 0.10 < 0.15 -> the gate metric FAILS
    assert abs(env["beneficial_coverage_subject_macro"] - 0.10) < 1e-9
    assert env["beneficial_coverage_subject_macro"] < RPT.COVERAGE_MIN
    # batch-weighted = 41/50 = 0.82 >= 0.15 -> would have WRONGLY passed
    assert env["beneficial_coverage_batch_weighted"] >= RPT.COVERAGE_MIN
    # confirm the gate uses the subject-macro (fails) not the batch-weighted (would pass)
    m = {"oracle_red_upper": 0.5, "beneficial_coverage_subject_macro": env["beneficial_coverage_subject_macro"],
         "sign_auroc_subject_balanced": 0.9, "perm_p_subject_block": 0.001}
    assert RPT.continuation_gate({"PD": m, "SCZ": m})[0] == RPT.STOP
    ok("a batch-rich beneficial subject cannot dominate the coverage gate (subject-macro 0.10 < 0.15 -> STOP)")


def main():
    print("ACAR v5 V6-A0a2 guard: batch-rich subject does not dominate coverage")
    test_batch_rich_subject_does_not_dominate_coverage_gate()
    print("ALL V6A0A2-COVERAGE-NO-DOMINATE GUARDS PASS")


if __name__ == "__main__":
    main()
