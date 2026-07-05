"""C15 manuscript-spine generator: 4 grounded claims, protocol steps cover G0-G5, evidence chain covers
C8..C14, limitation table avoids over-claims, canonical serializable. Synthetic minimal report dicts (only
the fields the generator reads)."""
from __future__ import annotations

from oaci.artifacts.canonical_json import canonical_json_bytes
from oaci.confirmatory.c15_spine import (build_claim_evidence, evidence_chain, limitation_boundary, protocol_steps)


def _reports():
    c8 = {"k1_overall": {"k1_sweep_status": "stop_no_detectable_heldout_leakage_reduction",
                         "n_leakage_reduction_detected": 11, "n_tests": 54,
                         "multiplicity": {"n_tests": 54, "n_bh_survive": 0}},
          "k2": {"k2_status": "stop_no_reproducible_gain"},
          "k2_agg": {"worst_domain_bacc": {"n_improved": 2, "n_harmed": 4, "mean": -0.0049}}}
    c10 = {"part1_transfer": {"selection_to_audit_optimism": {
                "delta_selection_leakage": {"mean": -0.3261}, "delta_audit_leakage": {"mean": 0.0076},
                "corr_selection_vs_audit_delta": {"pearson": {"r": 0.0045}}}},
           "part2_selector_replay": {"final_case": "C_oracle_also_fails", "oracle_reproducible": False,
                                     "s0_current_k2": "stop_no_reproducible_gain",
                                     "identity": {"n_all_match": 216, "n_checks": 216, "total_argmax_flips": 0}}}
    c12 = {"verdict": {"verdict": "stop_SRC_pivot_measurement_only", "n_source_improved_not_transferred": 6,
                       "n_target_nll_blowup": 6, "n_cells": 12}}
    gates = {g: {"status": s} for g, s in [
        ("G0_integrity", "integrity_ok"), ("G1_selection_optimism", "selection_optimism_present"),
        ("G2_heldout_leakage", "weak_nominal_nonmultiplicity_signal"),
        ("G3_endpoint_transfer", "stop_no_reproducible_gain"), ("G4_oracle_rescue", "oracle_fails_to_rescue"),
        ("G5_source_target_transfer", "source_target_antitransfer_detected")]}
    gates["G5_source_target_transfer"]["source_nll_to_target_nll_pearson"] = -0.947
    c14 = {"gates": gates, "diagnostics": {"instability": {"ATI_NLL": 1.0, "source_target_instability_score": 1.0}},
           "verdict": {"control_hypothesis_status": "falsified",
                       "falsification_reasons": ["falsified_by_no_endpoint_transfer", "falsified_by_oracle_failure",
                                                 "falsified_by_source_target_antitransfer"]}}
    return {"C8": c8, "C10": c10, "C12": c12, "C14": c14}


def test_build_claim_evidence_has_4_claims_grounded():
    cem = build_claim_evidence(_reports())
    assert [c["id"] for c in cem["claims"]] == ["C1_measurement", "C2_control_failure_localization",
                                                "C3_anti_transfer", "C4_framework"]
    assert cem["reviewer_hardened"] is True and cem["genuine_evidence_gaps_future_work"]
    for c in cem["claims"]:
        assert c["status"].startswith("supported") and len(c["evidence"]) >= 3 and len(c["caveats"]) >= 2
    # the real committed numbers must appear in the evidence (grounded, not transcribed)
    blob = str(cem)
    assert "-0.3261" in blob and "0 BH survivors / 54" in blob and "pearson -0.947" in blob


def test_protocol_steps_covers_g0_to_g5():
    steps = protocol_steps(_reports())
    assert [s[0] for s in steps] == ["G0_integrity", "G1_selection_optimism", "G2_heldout_leakage",
                                     "G3_endpoint_transfer", "G4_oracle_rescue", "G5_source_target_transfer"]
    assert all(s[3] for s in steps)                              # every gate has a BNCI001 outcome


def test_evidence_chain_covers_c8_to_c14():
    chain = evidence_chain(_reports())
    assert [row[0] for row in chain] == ["C8", "C10a", "C10b", "C12", "C14"]


def test_limitation_table_does_not_overclaim():
    lim = limitation_boundary()
    joined = " ".join(col for row in lim for col in row).lower()
    # the "what is NOT claimed" column must explicitly disclaim the forbidden over-claims
    assert "do not claim all eeg dg fails" in joined
    assert "every dg penalty" in joined and "support-aware invariance is useless" in joined


def test_c15_claim_map_canonical_serializable():
    cem = build_claim_evidence(_reports())
    blob = canonical_json_bytes(cem)
    assert blob and b'"claims"' in blob and b'"do_not_claim"' in blob


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c15-spine tests")


if __name__ == "__main__":
    _run_all()
