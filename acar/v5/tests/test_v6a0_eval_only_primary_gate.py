"""Guard (V6-A0a): the audit reads EVAL subjects only for the primary continuation gate; FIT/CAL are descriptive and cannot set
the decision. Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import stage2_action_records as AR
from acar.v5 import v6_a0_action_viability as AV
from acar.v5 import v6_a0_report as RPT
from acar.v5.tests._util import ok, v6a0_eval_fold, expect_raises


def test_collect_processes_eval_subjects_only():
    fold = v6a0_eval_fold([
        ("PD/ds002778/sub-t0", "train", 64, 0), ("PD/ds002778/sub-v0", "val", 64, 1),
        ("PD/ds002778/sub-c0", "cal", 64, 0),
        ("PD/ds002778/sub-e0", "eval", 64, 0), ("PD/ds002778/sub-e1", "eval", 64, 1),
    ])
    records, acct = AV.collect_eval_records([fold], AR.synthetic_action_provider)
    subs = {r["subject_key"] for r in records}
    assert subs == {"PD/ds002778/sub-e0", "PD/ds002778/sub-e1"}, f"only EVAL subjects allowed, got {subs}"
    assert acct["n_eval_subjects"] == 2
    ok("collect_eval_records uses EVAL subjects only (FIT/CAL never enter the action envelope)")


def test_gate_reads_eval_only_descriptive_cannot_flip():
    passing = {d: {"oracle_red_upper": 0.30, "beneficial_coverage": 0.40, "sign_auroc": 0.70, "perm_p": 0.01}
               for d in ("PD", "SCZ")}
    # a 'failing' FIT/CAL descriptive block must NOT change the decision (gate reads per_disease_eval only)
    rep = RPT.build_v6a0_report(per_disease_eval=passing,
                                descriptive={"FIT": {"note": "worse"}, "CAL": {"note": "worse"}, "per_action": {"t3a": 0.5}})
    assert rep["decision"] == RPT.CONTINUE and rep["primary_split"] == "EVAL"
    # validate rejects any report whose primary split is not EVAL
    bad = dict(rep)
    bad["primary_split"] = "CAL"
    expect_raises(RPT.V6A0ReportError, lambda: RPT.validate_v6a0_report(bad))
    ok("continuation gate reads EVAL metrics only; primary_split must be EVAL; FIT/CAL descriptive cannot flip the decision")


def main():
    print("ACAR v5 V6-A0a guard: EVAL-only primary gate")
    test_collect_processes_eval_subjects_only()
    test_gate_reads_eval_only_descriptive_cannot_flip()
    print("ALL V6A0-EVAL-ONLY GUARDS PASS")


if __name__ == "__main__":
    main()
