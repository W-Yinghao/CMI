"""Guard (V6-A0a2): one record-rich subject with strong apparent signal cannot force the subject-balanced primary AUROC to pass —
the many equal-weighted small subjects (with opposite signal) hold it below 0.60. sklearn-gated."""
from __future__ import annotations
import numpy as np
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5 import v6_a0_report as RPT
from acar.v5.tests._util import ok, has_sklearn


def test_record_rich_subject_does_not_dominate_auroc_gate():
    if not has_sklearn():
        ok("sklearn absent — AUROC domination path skipped")
        return
    groups = np.asarray(["A"] * 100 + [f"s{i}" for i in range(10)], dtype=object)
    y = np.asarray([1] * 50 + [0] * 50 + [1] * 5 + [0] * 5, int)
    scores = np.asarray([1.0] * 50 + [0.0] * 50 + [0.0] * 5 + [1.0] * 5, float)   # A perfect; 10 small subjects wrong
    bal = SP._auroc(y, scores, SP.subject_weights(groups))
    assert bal < RPT.AUROC_MIN, f"subject-balanced AUROC {bal:.3f} must stay below 0.60 despite the record-rich perfect subject"
    # the gate (which uses sign_auroc_subject_balanced) must FAIL on this
    m = {"oracle_red_upper": 0.5, "beneficial_coverage_subject_macro": 0.5,
         "sign_auroc_subject_balanced": bal, "perm_p_subject_block": 0.001}
    assert RPT.continuation_gate({"PD": m, "SCZ": m})[0] == RPT.STOP
    ok("a record-rich perfect subject cannot dominate the subject-balanced AUROC gate (stays < 0.60 -> STOP)")


def main():
    print("ACAR v5 V6-A0a2 guard: record-rich subject does not dominate AUROC")
    test_record_rich_subject_does_not_dominate_auroc_gate()
    print("ALL V6A0A2-AUROC-NO-DOMINATE GUARDS PASS")


if __name__ == "__main__":
    main()
