"""C14c — method closure table (data-driven from the gate outcomes) + battery payload assembly. The closure
table is a project-level guard: it records which method hypotheses are CLOSED as control objectives (so the
work does not loop back to 'just tune the selector'), and it explicitly forbids the OACI-v2 selector."""
from __future__ import annotations

from .schema import (ANTITRANSFER_DETECTED, K2_STOP, ORACLE_FAIL)


def build_method_closure_table(gates) -> list:
    """gates: dict gate_id -> gate record. Entries are derived from the gate statuses so the table can never
    contradict the evidence."""
    g3, g4, g5 = gates["G3_endpoint_transfer"], gates["G4_oracle_rescue"], gates["G5_source_target_transfer"]
    oaci_closed = g3["status"] == K2_STOP and g4["status"] == ORACLE_FAIL
    src_closed = g5["status"] == ANTITRANSFER_DETECTED
    rows = [
        {"method_hypothesis": "OACI conditional-domain leakage-control",
         "evidence": "C8 K1 weak-nominal/non-multiplicity + K2 stop; C10 source_audit oracle fails to rescue",
         "status": "closed_as_control_objective" if oaci_closed else "open",
         "next_allowed_action": "keep support-aware leakage + K1 as MEASUREMENT only; NO OACI-v2 selector, "
                                "NO adversary tuning"},
        {"method_hypothesis": "SRC source-endpoint control",
         "evidence": "C12 anti-transfer (source worst-domain NLL improves ~1 nat, target NLL worsens); "
                     "target NLL blowup; level-1 ERM fallback",
         "status": "closed_as_control_objective" if src_closed else "open",
         "next_allowed_action": "no further source-side endpoint control without a NEW transfer diagnostic"},
        {"method_hypothesis": "global_lpc / uniform (posterior-KL / uniform alignment)",
         "evidence": "baselines / failure modes; never beat ERM on target worst-domain",
         "status": "closed_as_positive_method",
         "next_allowed_action": "retain as stress-test baselines only"},
        {"method_hypothesis": "support-aware extractable leakage (L_Q^ov) + K1 permutation null",
         "evidence": "identifies non-identifiable cells + quantifies held-out audit leakage with an honest null",
         "status": "retained_as_measurement",
         "next_allowed_action": "use inside the falsification battery; NOT as a control objective"},
    ]
    return rows


def forbids_oaci_v2_selector(closure_table) -> bool:
    """The closure table must explicitly forbid an OACI-v2 selector recommendation."""
    for r in closure_table:
        if r["method_hypothesis"].startswith("OACI") and r["status"] == "closed_as_control_objective" \
                and "NO OACI-v2" in r["next_allowed_action"]:
            return True
    return False
