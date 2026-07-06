"""Guard (V6-A0a2): primary beneficial_coverage is SUBJECT-MACRO (mean of per-subject beneficial fractions), NOT batch-weighted.
The two differ when subjects have different batch counts. Synthetic, torch-free (hand-built ΔR records; no sklearn)."""
from __future__ import annotations
from acar.v5 import v6_a0_action_viability as AV
from acar.v5.tests._util import ok


def _rec(sk, best_negative):
    dr = {"matched_coral": (-1.0 if best_negative else 1.0), "spdim": 1.0, "t3a": 1.0}
    return {"subject_key": sk, "delta_r": dr}


def test_subject_macro_vs_batch_weighted_differ():
    # subject A: 8 eligible batches ALL beneficial (frac 1.0); subject B: 2 batches NONE beneficial (frac 0.0)
    records = [_rec("A", True) for _ in range(8)] + [_rec("B", False) for _ in range(2)]
    env = AV.oracle_envelope(records)
    assert env["beneficial_coverage_subject_macro"] == 0.5, env["beneficial_coverage_subject_macro"]   # mean(1.0, 0.0)
    assert env["beneficial_coverage_batch_weighted"] == 0.8                                             # 8/10 (A dominates)
    assert env["beneficial_coverage_subject_macro"] != env["beneficial_coverage_batch_weighted"]
    assert env["n_subjects_with_eligible"] == 2 and env["n_eligible_batches"] == 10
    # empty -> NaN (gate then fails)
    empty = AV.oracle_envelope([])
    assert empty["beneficial_coverage_subject_macro"] != empty["beneficial_coverage_subject_macro"]     # NaN
    assert empty["oracle_red_upper"] != empty["oracle_red_upper"]                                       # NaN
    ok("beneficial_coverage_subject_macro = mean per-subject fraction (not batch-weighted); empty -> NaN")


def main():
    print("ACAR v5 V6-A0a2 guard: beneficial coverage is subject-macro")
    test_subject_macro_vs_batch_weighted_differ()
    print("ALL V6A0A2-COVERAGE-SUBJECT-MACRO GUARDS PASS")


if __name__ == "__main__":
    main()
