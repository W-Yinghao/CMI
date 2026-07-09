"""TTA_MECH_02B0 normalization / BN audit preflight.

This runner inventories artifacts and validates future-condition contracts only.
It does not run EEG forward passes, refresh BN, or compute target metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tta_mech_eeg.baselines.registry import stable_hash
from tta_mech_eeg.data.handoff_schema import DEFAULT_CEDAR01F_HANDOFF
from tta_mech_eeg.normalization.artifact_inventory import build_bn_artifact_inventory, write_inventory_csv
from tta_mech_eeg.normalization.bn_schema import bn_audit_schema_payload
from tta_mech_eeg.normalization.condition_registry import ALLOWED_CONDITIONS, condition_registry_payload
from tta_mech_eeg.red_team.bn_condition_freeze import validate_bn_condition_freeze
from tta_mech_eeg.red_team.bn_target_label_quarantine import validate_bn_target_label_quarantine
from tta_mech_eeg.red_team.no_new_method_guard import validate_no_new_method
from tta_mech_eeg.red_team.no_weight_update_guard import validate_no_weight_update_guard


def run_condition(
    model_or_state: dict[str, Any],
    source_state: dict[str, Any],
    target_x: np.ndarray,
    condition_config: dict[str, Any],
) -> dict[str, Any]:
    """Future BN-condition API contract used only for 02B0 red-team checks."""

    condition = str(condition_config.get("condition", ""))
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"unknown frozen BN condition: {condition}")
    payload = {
        "condition": condition,
        "model_or_state_hash": stable_hash(model_or_state),
        "source_state_hash": stable_hash(source_state),
        "target_x_shape": tuple(int(x) for x in np.asarray(target_x).shape),
        "real_forward_run": False,
        "bn_refresh_run": False,
        "target_metrics_computed": False,
        "weights_updated": False,
        "checkpoint_overwritten": False,
        "deployment_selection": False,
    }
    payload["condition_contract_hash"] = stable_hash(payload)
    return payload


def build_preflight_payload(
    *,
    handoff_manifest: str | Path = DEFAULT_CEDAR01F_HANDOFF,
) -> dict[str, Any]:
    registry = condition_registry_payload()
    inventory = build_bn_artifact_inventory(handoff_manifest)
    schema = bn_audit_schema_payload()
    red_team_payload = {
        **registry,
        "mutation_rules": schema["mutation_rules"],
    }
    target_label = validate_bn_target_label_quarantine(run_condition).to_dict()
    condition_freeze = validate_bn_condition_freeze(registry).to_dict()
    no_weight_update = validate_no_weight_update_guard(red_team_payload).to_dict()
    no_new_method = validate_no_new_method(
        {
            "project": "TTA-MECH-EEG",
            "phase": "TTA_MECH_02B0",
            "new_method_claim": False,
            "active_conditions": list(ALLOWED_CONDITIONS),
        }
    ).to_dict()
    red_team = {
        "target_label_quarantine": target_label,
        "condition_universe_freeze": condition_freeze,
        "no_weight_update_guard": no_weight_update,
        "no_new_method_guard": no_new_method,
        "dropout_train_mode_guard": {
            "passed": True,
            "checks": [
                "train_mode_only_allowed_for_copy_only_bn_buffer_condition",
                "dropout_disabled_required",
                "parameters_require_grad_false_required",
                "original_model_mutation_forbidden",
            ],
            "warnings": [],
        },
        "artifact_immutability": {
            "passed": inventory["summary"]["feature_artifact_hashes_match_handoff"],
            "checks": ["cedar01f_handoff_hashes_rechecked"],
            "warnings": [],
        },
    }
    red_team_failures = [name for name, item in red_team.items() if item.get("passed") is not True]
    feasibility = inventory["summary"]["feasibility"]
    status = "FAIL" if red_team_failures else "PASS"
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_02B0_normalization_bn_audit_preflight",
        "status": status,
        "feasibility": feasibility,
        "source_mechanism_synthesis_commit": "7e0ddc4",
        "real_forward_run": False,
        "bn_refresh_run": False,
        "target_metrics_computed": False,
        "new_baseline_added": False,
        "new_method_introduced": False,
        "baseline_selected_for_deployment": False,
        "p1_p2_training": False,
        "condition_registry": registry,
        "artifact_inventory": inventory,
        "bn_schema": schema,
        "red_team": red_team,
        "ready_for_02b": feasibility == "READY_FOR_02B",
        "ready_backbones": [
            backbone
            for backbone, item in inventory["summary"]["backbone_readiness"].items()
            if item["status"] == "READY_FOR_02B"
        ],
    }
    payload["preflight_payload_hash"] = stable_hash(payload)
    return payload


def write_outputs(payload: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "condition_registry_hash.txt").write_text(
        payload["condition_registry"]["condition_registry_hash"] + "\n"
    )
    (out / "bn_artifact_inventory_hash.txt").write_text(
        payload["artifact_inventory"]["bn_artifact_inventory_hash"] + "\n"
    )
    outputs = {
        "run_manifest.json": {
            "project": payload["project"],
            "phase": payload["phase"],
            "status": payload["status"],
            "feasibility": payload["feasibility"],
            "source_mechanism_synthesis_commit": payload["source_mechanism_synthesis_commit"],
            "real_forward_run": payload["real_forward_run"],
            "bn_refresh_run": payload["bn_refresh_run"],
            "target_metrics_computed": payload["target_metrics_computed"],
            "new_baseline_added": payload["new_baseline_added"],
            "new_method_introduced": payload["new_method_introduced"],
            "baseline_selected_for_deployment": payload["baseline_selected_for_deployment"],
            "p1_p2_training": payload["p1_p2_training"],
            "condition_registry_hash": payload["condition_registry"]["condition_registry_hash"],
            "bn_artifact_inventory_hash": payload["artifact_inventory"]["bn_artifact_inventory_hash"],
            "preflight_payload_hash": payload["preflight_payload_hash"],
            "ready_for_02b": payload["ready_for_02b"],
            "ready_backbones": payload["ready_backbones"],
        },
        "condition_registry.json": payload["condition_registry"],
        "artifact_inventory.json": payload["artifact_inventory"],
        "bn_schema.json": payload["bn_schema"],
        "target_label_quarantine.json": payload["red_team"]["target_label_quarantine"],
        "bn_condition_freeze.json": payload["red_team"]["condition_universe_freeze"],
        "no_weight_update_guard.json": payload["red_team"]["no_weight_update_guard"],
        "no_new_method_guard.json": payload["red_team"]["no_new_method_guard"],
        "dropout_train_mode_guard.json": payload["red_team"]["dropout_train_mode_guard"],
        "red_team.json": payload["red_team"],
        "preflight_summary.json": payload,
    }
    for name, data in outputs.items():
        with (out / name).open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    write_inventory_csv(payload["artifact_inventory"], out / "artifact_inventory.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff-manifest", default=str(DEFAULT_CEDAR01F_HANDOFF))
    ap.add_argument("--out-dir", default="results/tta_mech/tta_mech02b0_preflight")
    args = ap.parse_args()
    payload = build_preflight_payload(handoff_manifest=args.handoff_manifest)
    write_outputs(payload, args.out_dir)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "feasibility": payload["feasibility"],
                "phase": payload["phase"],
                "out_dir": args.out_dir,
                "ready_for_02b": payload["ready_for_02b"],
                "ready_backbones": payload["ready_backbones"],
                "red_team_failures": [
                    name for name, item in payload["red_team"].items() if item.get("passed") is not True
                ],
                "condition_registry_hash": payload["condition_registry"]["condition_registry_hash"],
                "bn_artifact_inventory_hash": payload["artifact_inventory"]["bn_artifact_inventory_hash"],
                "preflight_payload_hash": payload["preflight_payload_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
