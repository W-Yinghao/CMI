"""Guard (V6-A0a2): the continuation gate reads EXACTLY the subject-balanced field names; a report using the OLD batch/record-level
names is treated as missing metrics (gate STOP), so there is no ambiguity about which metrics fed the V6-A0b decision. Torch-free."""
from __future__ import annotations
from acar.v5 import v6_a0_report as RPT
from acar.v5.tests._util import ok


def test_gate_uses_subject_balanced_field_names():
    assert RPT._REQUIRED_METRICS == ("oracle_red_upper", "beneficial_coverage_subject_macro",
                                     "sign_auroc_subject_balanced", "perm_p_subject_block")
    passing = {"oracle_red_upper": 0.5, "beneficial_coverage_subject_macro": 0.4,
               "sign_auroc_subject_balanced": 0.7, "perm_p_subject_block": 0.01}
    assert RPT.continuation_gate({"PD": passing, "SCZ": passing})[0] == RPT.CONTINUE
    # OLD field names -> treated as missing -> STOP (no silent fall-through to a batch-weighted metric)
    old_names = {"oracle_red_upper": 0.5, "beneficial_coverage": 0.4, "sign_auroc": 0.7, "perm_p": 0.01}
    dec, detail = RPT.continuation_gate({"PD": old_names, "SCZ": old_names})
    assert dec == RPT.STOP and detail["PD"]["pass"] is False
    # a full report built with the subject-balanced names validates + carries no forbidden key
    rep = RPT.build_v6a0_report(per_disease_eval={"PD": passing, "SCZ": passing})
    RPT.validate_v6a0_report(rep)
    assert rep["decision"] == RPT.CONTINUE
    ok("continuation gate uses subject-balanced field names exactly; old names -> missing -> STOP")


def main():
    print("ACAR v5 V6-A0a2 guard: gate uses subject-balanced field names")
    test_gate_uses_subject_balanced_field_names()
    print("ALL V6A0A2-GATE-FIELD-NAMES GUARDS PASS")


if __name__ == "__main__":
    main()
