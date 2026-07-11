"""Checkpoint immutability and start-artifact readiness checks."""

from typing import Dict, Mapping


def evaluate_checkpoint_immutability(inventory: Mapping[str, object]) -> Dict[str, object]:
    entries = list(inventory.get("entries", []))
    payload_entries = [entry for entry in entries if entry.get("kind") != "random_reference_config"]
    h200 = [entry for entry in entries if entry.get("tag") in {"H200_s0", "H200_s1"}]
    checks = {
        "all_declared_entries_present": len(entries) == 10,
        "all_payload_paths_exist": all(entry.get("exists") is True for entry in payload_entries),
        "repeated_sha_identical": all(entry.get("repeated_sha_identical") is True for entry in entries),
        "inventory_paths_stable": all(entry.get("resolved_path_stable") is True for entry in entries),
        "all_strict_reload_pass": all(entry.get("strict_reload_pass") is True for entry in entries),
        "h200_count_exact": len(h200) == 2,
        "h200_complete_provenance": len(h200) == 2 and all(
            entry.get("training_complete") is True and entry.get("provenance_complete") is True
            for entry in h200
        ),
        "h200_only_star_starts": {
            entry.get("tag") for entry in entries if entry.get("usable_as_star_start")
        } == {"H200_s0", "H200_s1"},
    }
    infrastructure_pass = all(value for key, value in checks.items() if key != "h200_complete_provenance")
    h200_ready = bool(inventory.get("h200_start_checkpoints_ready")) and checks["h200_complete_provenance"]
    return {
        "status": "PASS" if infrastructure_pass else "FAIL",
        "h200_start_checkpoints_ready": h200_ready,
        "star01_artifact_status": "READY_FOR_PM_REVIEW" if h200_ready else "STAR_01_BLOCKED_ARTIFACT_SUPPLY",
        "checks": checks,
    }
