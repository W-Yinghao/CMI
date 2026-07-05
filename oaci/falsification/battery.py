"""C14 — battery orchestrator. Loads the committed C8 / C10 / C12 evidence, runs the six gates + the
source->target instability diagnostics + the method-closure table, and decides the battery verdict. Pure
aggregation of already-committed artifacts: NO GPU, NO retraining, NO target used for selection."""
from __future__ import annotations

import json
import os

from . import gates as _g
from .antitransfer import g5_source_target_transfer
from .oracle import g4_oracle_rescue
from .payloads import build_method_closure_table
from .schema import (ANTITRANSFER_DETECTED, CONTROL_INCONCLUSIVE, CONTROL_SUPPORTED, FALSIFIED_ANTITRANSFER,
                     FALSIFIED_NO_ENDPOINT, FALSIFIED_ORACLE, FALSIFIED_SELECTION, INTEGRITY_OK, INVALID_EVIDENCE,
                     K1_DETECTED, K2_GAIN, K2_STOP, ORACLE_FAIL, ORACLE_RESCUE, SELECTION_OPTIMISM_PRESENT)
from .transfer import harm_localization, instability_metrics, transfer_correlations

_C8 = "C8_BNCI001_LOSO_SEEDS012_K1K2.json"
_C10 = "C10_OACI_FAILURE_DIAGNOSTICS.json"
_C12 = "C12_SRC_STRESS_REPLICATION.json"


def load_evidence(report_dir) -> dict:
    def rd(name):
        p = os.path.join(report_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"falsification battery requires {name} in {report_dir}")
        return json.load(open(p))
    return {"c8": rd(_C8), "c10": rd(_C10), "c12": rd(_C12)}


def final_verdict(gate_map) -> dict:
    g0, g1, g2, g3, g4, g5 = (gate_map["G0_integrity"], gate_map["G1_selection_optimism"],
                              gate_map["G2_heldout_leakage"], gate_map["G3_endpoint_transfer"],
                              gate_map["G4_oracle_rescue"], gate_map["G5_source_target_transfer"])
    if g0["status"] != INTEGRITY_OK:
        return {"control_hypothesis_status": INVALID_EVIDENCE, "falsification_reasons": [],
                "reason": "G0 integrity failed — downstream evidence is untrustworthy"}
    reasons = []
    if g3["status"] == K2_STOP:
        reasons.append(FALSIFIED_NO_ENDPOINT)
    if g4["status"] == ORACLE_FAIL:
        reasons.append(FALSIFIED_ORACLE)
    if g5["status"] == ANTITRANSFER_DETECTED:
        reasons.append(FALSIFIED_ANTITRANSFER)
    # selection optimism is only the PRIMARY falsification when nothing deeper fired (e.g. no endpoint/oracle data)
    if not reasons and g1["status"] == SELECTION_OPTIMISM_PRESENT and g2["status"] != K1_DETECTED:
        reasons.append(FALSIFIED_SELECTION)
    if reasons:
        status = "falsified"
    elif g3["status"] == K2_GAIN and g4["status"] == ORACLE_RESCUE:
        status = CONTROL_SUPPORTED
    else:
        status = CONTROL_INCONCLUSIVE
    return {"control_hypothesis_status": status, "falsification_reasons": reasons,
            "reason": ("; ".join(reasons) if reasons else status)}


def run_battery(c8, c10, c12) -> dict:
    gate_list = [
        _g.g0_integrity(c8, c10, c12), _g.g1_selection_optimism(c10), _g.g2_heldout_leakage(c8),
        _g.g3_endpoint_transfer(c8), g4_oracle_rescue(c10), g5_source_target_transfer(c12, c10["part1_transfer"]),
    ]
    gate_map = {g["gate"]: g for g in gate_list}
    cells = c12.get("cells", [])
    diagnostics = {"transfer_correlations": transfer_correlations(cells, c10["part1_transfer"]),
                   "instability": instability_metrics(cells), "harm_localization": harm_localization(cells)}
    closure = build_method_closure_table(gate_map)
    verdict = final_verdict(gate_map)
    return {"battery": "C14_EEG_DG_Falsification_Battery",
            "sources": {"C8": _C8, "C10": _C10, "C12": _C12},
            "gates": gate_map, "gate_order": [g["gate"] for g in gate_list],
            "diagnostics": diagnostics, "method_closure_table": closure, "verdict": verdict,
            "notice": ("OACI / SRC are NOT control methods. Support-aware leakage + K1/K2 + oracle replay + "
                       "anti-transfer diagnostics are a MEASUREMENT / FALSIFICATION instrument.")}


def build_from_reports(report_dir) -> dict:
    ev = load_evidence(report_dir)
    return run_battery(ev["c8"], ev["c10"], ev["c12"])
