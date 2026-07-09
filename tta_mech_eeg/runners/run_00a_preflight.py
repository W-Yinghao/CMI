"""TTA_MECH_00A artifact-inventory and replay-harness preflight.

This runner is intentionally synthetic/read-only. It inventories artifact
availability and validates harness contracts. It does not run real EEG replay
or compute target metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tta_mech_eeg.audit_axes.schema import audit_axis_schema_payload
from tta_mech_eeg.baselines.registry import baseline_registry, registry_payload, stable_hash
from tta_mech_eeg.data.artifact_inventory import build_artifact_inventory
from tta_mech_eeg.data.handoff_schema import DEFAULT_CEDAR01F_HANDOFF, handoff_schema_payload
from tta_mech_eeg.red_team.baseline_universe_freeze import validate_baseline_universe
from tta_mech_eeg.red_team.no_new_method_guard import validate_no_new_method
from tta_mech_eeg.red_team.replay_determinism import validate_replay_determinism
from tta_mech_eeg.red_team.target_label_quarantine import validate_target_label_contract


def adapt_or_replay(source_state: dict[str, Any], target_x: np.ndarray, baseline: str) -> dict[str, Any]:
    """Toy deterministic replay harness contract.

    This dry-run accepts source state, target features, and a frozen baseline
    name only. It intentionally has no target-label argument.
    """

    names = {entry.name for entry in baseline_registry()}
    if baseline not in names:
        raise ValueError(f"unknown frozen baseline: {baseline}")
    target_x = np.asarray(target_x, dtype=np.float64)
    if target_x.ndim != 2:
        raise ValueError("target_x must be 2D")
    payload = {
        "baseline": baseline,
        "source_state_hash": stable_hash(source_state),
        "target_x_shape": tuple(int(x) for x in target_x.shape),
        "target_x_sum": float(target_x.sum()),
        "target_x_mean": float(target_x.mean()),
        "target_x_std": float(target_x.std()),
        "real_eeg_replay_run": False,
        "target_metrics_computed": False,
    }
    payload["toy_output_hash"] = stable_hash(payload)
    return payload


def _toy_inputs(seed: int) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    source_state = {
        "source_state_kind": "toy_summary_for_contract_only",
        "n_features": 4,
        "n_classes": 3,
        "source_prior": [0.34, 0.33, 0.33],
    }
    target_x = rng.normal(size=(12, 4))
    return source_state, target_x


def build_preflight_payload(
    *,
    handoff_manifest: str | Path = DEFAULT_CEDAR01F_HANDOFF,
    seed: int = 0,
) -> dict[str, Any]:
    baseline_payload = registry_payload()
    audit_schema = audit_axis_schema_payload()
    handoff_payload = handoff_schema_payload(handoff_manifest)
    artifact_inventory = build_artifact_inventory(handoff_manifest)

    source_state, target_x = _toy_inputs(seed)
    left = adapt_or_replay(source_state, target_x, "ERM_NO_ADAPT")
    right = adapt_or_replay(source_state, target_x, "ERM_NO_ADAPT")

    target_contract = validate_target_label_contract(adapt_or_replay)
    determinism = validate_replay_determinism(left, right)
    baseline_freeze = validate_baseline_universe(baseline_payload)
    active_method_config = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A",
        "new_method_claim": False,
        "active_baselines": baseline_payload["allowed_baselines"],
        "audit_axes": [axis["name"] for axis in audit_schema["axes"]],
    }
    no_new_method = validate_no_new_method(active_method_config)

    red_team = {
        "baseline_universe_freeze": baseline_freeze.to_dict(),
        "target_label_quarantine_contract": target_contract.to_dict(),
        "replay_determinism": determinism.to_dict(),
        "no_new_method_guard": no_new_method.to_dict(),
    }
    status = "PASS" if all(item.get("passed") for item in red_team.values()) else "FAIL"
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_00A_artifact_inventory_replay_harness_preflight",
        "status": status,
        "design_baseline_commit": "cccd8c1",
        "real_eeg_replay_run": False,
        "target_metrics_computed": False,
        "baseline_selected": False,
        "new_method_introduced": False,
        "baseline_registry": baseline_payload,
        "artifact_inventory": artifact_inventory,
        "handoff_schema": handoff_payload,
        "audit_axis_schema": audit_schema,
        "toy_replay": left,
        "red_team": red_team,
    }
    payload["preflight_payload_hash"] = stable_hash(payload)
    return payload


def write_outputs(payload: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "run_manifest.json": {
            "project": payload["project"],
            "phase": payload["phase"],
            "status": payload["status"],
            "design_baseline_commit": payload["design_baseline_commit"],
            "real_eeg_replay_run": payload["real_eeg_replay_run"],
            "target_metrics_computed": payload["target_metrics_computed"],
            "baseline_selected": payload["baseline_selected"],
            "new_method_introduced": payload["new_method_introduced"],
            "baseline_registry_hash": payload["baseline_registry"]["baseline_registry_hash"],
            "artifact_inventory_hash": payload["artifact_inventory"]["artifact_inventory_hash"],
            "preflight_payload_hash": payload["preflight_payload_hash"],
        },
        "baseline_registry.json": payload["baseline_registry"],
        "artifact_inventory.json": payload["artifact_inventory"],
        "handoff_schema.json": payload["handoff_schema"],
        "audit_axis_schema.json": payload["audit_axis_schema"],
        "toy_replay_determinism.json": payload["red_team"]["replay_determinism"],
        "target_label_quarantine_contract.json": payload["red_team"]["target_label_quarantine_contract"],
        "no_new_method_guard.json": payload["red_team"]["no_new_method_guard"],
        "red_team.json": payload["red_team"],
        "preflight_summary.json": payload,
    }
    for name, data in outputs.items():
        with (out / name).open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff-manifest", default=str(DEFAULT_CEDAR01F_HANDOFF))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/tta_mech/tta_mech00a_preflight")
    args = ap.parse_args()
    payload = build_preflight_payload(
        handoff_manifest=args.handoff_manifest,
        seed=args.seed,
    )
    write_outputs(payload, args.out_dir)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "phase": payload["phase"],
                "out_dir": args.out_dir,
                "real_eeg_replay_run": payload["real_eeg_replay_run"],
                "target_metrics_computed": payload["target_metrics_computed"],
                "baseline_registry_hash": payload["baseline_registry"]["baseline_registry_hash"],
                "artifact_inventory_hash": payload["artifact_inventory"]["artifact_inventory_hash"],
                "preflight_payload_hash": payload["preflight_payload_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
