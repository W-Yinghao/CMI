"""Compute-equivalence assertions for H200_SSL_CONT/TRUE/SHUFFLED."""

from typing import Dict, Mapping

from star_eeg.objectives.alternating_schedule import build_compute_match_contract


def evaluate_compute_match(contract: Mapping[str, object] = None) -> Dict[str, object]:
    selected = dict(contract or build_compute_match_contract())
    repeat = build_compute_match_contract(int(selected["total_optimizer_steps"]))
    checks = dict(selected.get("checks", {}))
    checks.update({
        "schedule_deterministic": selected.get("compute_match_hash") == repeat.get("compute_match_hash"),
        "scheduler_defined_by_optimizer_step": selected.get("unit") == "optimizer_step",
        "primary_is_fixed_final_step": selected.get("primary_checkpoint_selection") == "fixed_final_optimizer_step",
        "source_val_does_not_select_primary": selected.get("diagnostic_best_pretrain_val_selects_primary") is False,
    })
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "compute_match_hash": selected.get("compute_match_hash"),
    }
