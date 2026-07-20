"""C14 falsification battery — gate/verdict state constants + a tiny Gate record. States are frozen strings
so downstream tools and the report can pattern-match them."""
from __future__ import annotations

# ---- gate ids ----
G0, G1, G2, G3, G4, G5 = "G0_integrity", "G1_selection_optimism", "G2_heldout_leakage", \
    "G3_endpoint_transfer", "G4_oracle_rescue", "G5_source_target_transfer"

# ---- G0 integrity ----
INTEGRITY_OK = "integrity_ok"
INVALID_EVIDENCE = "invalid_evidence"

# ---- G1 selection optimism ----
SELECTION_OPTIMISM_PRESENT = "selection_optimism_present"
NO_SELECTION_OPTIMISM = "no_selection_optimism"

# ---- G2 held-out leakage (K1) ----
K1_DETECTED = "heldout_leakage_reduction_detected"
K1_STOP = "stop_no_detectable_heldout_leakage_reduction"
K1_WEAK = "weak_nominal_nonmultiplicity_signal"

# ---- G3 endpoint transfer (K2) ----
K2_GAIN = "reproducible_endpoint_gain"
K2_STOP = "stop_no_reproducible_gain"
K2_MIXED = "mixed_or_unstable_endpoint_effect"

# ---- G4 oracle rescue ----
ORACLE_RESCUE = "oracle_rescues_trajectory"
ORACLE_FAIL = "oracle_fails_to_rescue"

# ---- G5 source->target transfer ----
ANTITRANSFER_DETECTED = "source_target_antitransfer_detected"
NO_ANTITRANSFER = "no_source_target_antitransfer"

# ---- battery verdicts ----
CONTROL_SUPPORTED = "control_hypothesis_supported"
CONTROL_INCONCLUSIVE = "control_hypothesis_inconclusive"
FALSIFIED_SELECTION = "falsified_by_selection_optimism"
FALSIFIED_NO_ENDPOINT = "falsified_by_no_endpoint_transfer"
FALSIFIED_ORACLE = "falsified_by_oracle_failure"
FALSIFIED_ANTITRANSFER = "falsified_by_source_target_antitransfer"


def gate(gate_id, status, **fields) -> dict:
    return {"gate": gate_id, "status": status, **fields}
