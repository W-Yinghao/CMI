"""Guard (V6-A0a): the continuation gate is EVAL-primary and requires BOTH diseases to pass ALL FOUR sub-gates; and the report
schema forbids any selection/routing/gate-pass/later-stage key. Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import v6_a0_report as RPT
from acar.v5.tests._util import ok, expect_raises

_PASS = {"oracle_red_upper": 0.30, "beneficial_coverage": 0.40, "sign_auroc": 0.70, "perm_p": 0.01}


def _both(pd_over=None, scz_over=None):
    pd = dict(_PASS, **(pd_over or {}))
    scz = dict(_PASS, **(scz_over or {}))
    return {"PD": pd, "SCZ": scz}


def test_gate_requires_both_and_all_four():
    assert RPT.continuation_gate(_both())[0] == RPT.CONTINUE
    # each sub-gate, boundary-correct, one disease at a time -> STOP
    for over in ({"oracle_red_upper": 0.02},              # strict > 0.02 -> 0.02 fails
                 {"beneficial_coverage": 0.149},          # >= 0.15 -> 0.149 fails
                 {"sign_auroc": 0.599},                   # >= 0.60 -> 0.599 fails
                 {"sign_auroc": float("nan")},            # NaN AUROC fails
                 {"perm_p": 0.051}):                      # <= 0.05 -> 0.051 fails
        assert RPT.continuation_gate(_both(pd_over=over))[0] == RPT.STOP, f"PD {over} should STOP"
        assert RPT.continuation_gate(_both(scz_over=over))[0] == RPT.STOP, f"SCZ {over} should STOP"
    # boundaries that PASS: coverage exactly 0.15, auroc exactly 0.60, perm_p exactly 0.05, red_upper just over 0.02
    assert RPT.continuation_gate(_both(pd_over={"beneficial_coverage": 0.15, "sign_auroc": 0.60, "perm_p": 0.05,
                                                "oracle_red_upper": 0.0201}))[0] == RPT.CONTINUE
    # a missing disease -> STOP
    assert RPT.continuation_gate({"PD": _PASS})[0] == RPT.STOP
    ok("V6_CONTINUE requires BOTH diseases to pass all four sub-gates (EVAL-primary); else V6_STOP")


def test_report_schema_forbids_selection_and_later_keys():
    rep = RPT.build_v6a0_report(per_disease_eval=_both(), descriptive={"per_action": {"t3a_auroc": 0.5}},
                                accounting={"n_eval_eligible_batches": 100}, provenance_tags=["native"])
    RPT.validate_v6a0_report(rep)
    assert rep["decision"] == RPT.CONTINUE and rep["exploratory"] is True and rep["primary_split"] == "EVAL"
    # any forbidden key anywhere (top-level or nested) fails closed
    for bad_key in ("candidate_id", "selected_candidate_id", "threshold", "route", "g4", "stage4", "external", "lockbox"):
        polluted = dict(rep)
        polluted["descriptive_fit_cal_and_secondary"] = {"nested": {bad_key: 1}}
        expect_raises(RPT.V6A0ReportError, lambda p=polluted: RPT.validate_v6a0_report(p))
    ok("V6-A0 report forbids candidate_id/threshold/route/G1-G6/Stage-4/external/lockbox anywhere (fail-closed)")


def main():
    print("ACAR v5 V6-A0a guard: continue gate requires both diseases; report schema fail-closed")
    test_gate_requires_both_and_all_four()
    test_report_schema_forbids_selection_and_later_keys()
    print("ALL V6A0-CONTINUE-GATE GUARDS PASS")


if __name__ == "__main__":
    main()
