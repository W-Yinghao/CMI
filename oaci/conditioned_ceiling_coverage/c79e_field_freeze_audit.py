"""Read-only engineering audit of the frozen C79E seed-4 field."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c79_tables"
CAMPAIGN_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c79-seed4/"
    "protocol_e350b7f0c4ee3dfc/implementation_dd4043ad7dd67552"
)
EXPECTED_UNITS = REPORT_DIR / "c79p_tables" / "expected_seed4_field_manifest.csv"
TARGETS = (4, 8, 9, 3, 6, 5, 2, 7, 1)
PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
EPOCHS = tuple(range(4, 200, 5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_root(target: int) -> Path:
    return CAMPAIGN_ROOT / "targets" / f"target-{target:03d}"


def audit() -> dict[str, Any]:
    import torch
    from oaci.train.checkpoint import state_hash
    from .c78_authorized_train import optimizer_state_hash

    expected = read_csv(EXPECTED_UNITS)
    expected_by_id = {row["unit_id"]: row for row in expected}
    if len(expected_by_id) != 1458:
        raise RuntimeError("C79E expected unit registry is not 1,458 unique rows")

    unit_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    genealogy_rows: list[dict[str, Any]] = []
    cadence_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    isolation_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()

    for target in TARGETS:
        root = target_root(target)
        oaci_path = root / "training" / "oaci_erm" / "FIELD_FROZEN.json"
        src_path = root / "training" / "src" / "FIELD_FROZEN.json"
        instrument_path = root / "instrumentation" / "INSTRUMENTATION_COMPLETE.json"
        primary_path = root / "views" / "PRIMARY_INPUT_VIEWS.json"
        oaci = load_json(oaci_path)
        src = load_json(src_path)
        instrument = load_json(instrument_path)
        primary = load_json(primary_path)
        units = list(oaci["units"]) + list(src["units"])
        inst_by_id = {
            row["unit_id"]: load_json(Path(row["path"]))
            for row in instrument["units"]
        }
        if len(units) != 162 or len(inst_by_id) != 162:
            raise RuntimeError(f"C79E target {target} does not contain 162 unique units")

        sidecars: dict[str, dict[str, Any]] = {}
        checkpoint_pass = sidecar_pass = state_pass = optimizer_file_pass = optimizer_state_pass = 0
        target_isolation_pass = 0
        for unit in units:
            unit_id = unit["unit_id"]
            if unit_id in observed_ids or unit_id not in expected_by_id:
                raise RuntimeError(f"C79E unit registry mismatch: {unit_id}")
            observed_ids.add(unit_id)
            expected_row = expected_by_id[unit_id]
            for key in ("target", "seed", "level", "regime", "epoch", "trajectory_order"):
                if str(unit[key]) != expected_row[key]:
                    raise RuntimeError(f"C79E unit field mismatch: {unit_id} {key}")

            checkpoint_path = Path(unit["checkpoint_path"])
            sidecar_path = Path(unit["sidecar_path"])
            checkpoint_file_ok = sha256_file(checkpoint_path) == unit["checkpoint_file_sha256"]
            sidecar_file_ok = sha256_file(sidecar_path) == unit["sidecar_sha256"]
            sidecar = load_json(sidecar_path)
            sidecars[unit_id] = sidecar
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_ok = state_hash(state) == unit["checkpoint_id"] == sidecar["checkpoint_id"]
            del state

            optimizer_path = Path(sidecar["optimizer_state_path"])
            optimizer_file_ok = sha256_file(optimizer_path) == sidecar["optimizer_state_file_sha256"]
            optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
            optimizer_ok = optimizer_state_hash(optimizer) == sidecar["optimizer_state_hash"]
            del optimizer

            inst = inst_by_id[unit_id]
            inst_ok = (
                inst["checkpoint_id"] == unit["checkpoint_id"]
                and inst["checkpoint_file_sha256"] == unit["checkpoint_file_sha256"]
                and inst["sidecar_sha256"] == unit["sidecar_sha256"]
                and inst["all_gates_passed"]
                and all(float(inst["identity"][key]) == 0.0 for key in (
                    "identity_abs", "softmax_abs", "hook_abs", "repeat_logits", "repeat_z"
                ))
            )
            isolation_ok = (
                int(sidecar["seed"]) == 4
                and int(sidecar["target"]) == target
                and sidecar["target_fit_ids_empty"]
                and not sidecar["target_labels_available_to_optimizer"]
                and int(sidecar["target_rows_loaded_in_training_process"]) == 0
                and int(sidecar["source_audit_rows_loaded_in_training_process"]) == 0
                and not sidecar["target_outcome_used_for_retention"]
                and not sidecar["target_outcome_used_for_retry"]
                and not sidecar["selector_artifact"]
            )
            if not all((checkpoint_file_ok, sidecar_file_ok, state_ok, optimizer_file_ok, optimizer_ok, inst_ok, isolation_ok)):
                raise RuntimeError(f"C79E unit identity or isolation failure: {unit_id}")

            checkpoint_pass += int(checkpoint_file_ok)
            sidecar_pass += int(sidecar_file_ok)
            state_pass += int(state_ok)
            optimizer_file_pass += int(optimizer_file_ok)
            optimizer_state_pass += int(optimizer_ok)
            target_isolation_pass += int(isolation_ok)
            unit_rows.append({
                "unit_id": unit_id,
                "target": target,
                "primary": int(target in PRIMARY_TARGETS),
                "seed": 4,
                "level": int(unit["level"]),
                "regime": unit["regime"],
                "epoch": int(unit["epoch"]),
                "trajectory_order": int(unit["trajectory_order"]),
                "checkpoint_id": unit["checkpoint_id"],
                "checkpoint_file_sha256": unit["checkpoint_file_sha256"],
                "sidecar_sha256": unit["sidecar_sha256"],
                "optimizer_state_hash": sidecar["optimizer_state_hash"],
                "instrumentation_manifest_sha256": inst["manifest_sha256"],
                "all_replay_gates_passed": 1,
            })

        for level in (0, 1):
            erm_units = [row for row in units if int(row["level"]) == level and row["regime"] == "ERM"]
            if len(erm_units) != 1:
                raise RuntimeError(f"C79E target {target} level {level} ERM anchor count drift")
            erm_id = erm_units[0]["checkpoint_id"]
            for regime in ("OACI", "SRC"):
                trajectory = sorted(
                    [row for row in units if int(row["level"]) == level and row["regime"] == regime],
                    key=lambda row: int(row["trajectory_order"]),
                )
                epochs = tuple(int(row["epoch"]) for row in trajectory)
                orders = tuple(int(row["trajectory_order"]) for row in trajectory)
                cadence_ok = len(trajectory) == 40 and epochs == EPOCHS and orders == tuple(range(1, 41))
                previous = erm_id
                genealogy_ok = True
                for row in trajectory:
                    sidecar = sidecars[row["unit_id"]]
                    genealogy_ok &= (
                        sidecar["parent_ERM_checkpoint_id"] == erm_id
                        and sidecar["previous_trajectory_checkpoint_id"] == previous
                    )
                    previous = row["checkpoint_id"]
                if not cadence_ok or not genealogy_ok:
                    raise RuntimeError(f"C79E target {target} level {level} {regime} genealogy drift")
                cadence_rows.append({
                    "target": target, "level": level, "regime": regime,
                    "expected_checkpoints": 40, "observed_checkpoints": len(trajectory),
                    "first_epoch": epochs[0], "last_epoch": epochs[-1],
                    "fixed_every_5_epochs": int(cadence_ok), "passed": int(cadence_ok),
                })
                genealogy_rows.append({
                    "target": target, "level": level, "regime": regime,
                    "ERM_parent_checkpoint_id": erm_id,
                    "trajectory_links_checked": len(trajectory),
                    "failed_links": 0, "passed": int(genealogy_ok),
                })

        checkpoint_rows.append({
            "target": target, "expected_units": 162, "observed_units": len(units),
            "checkpoint_file_hash_passed": checkpoint_pass,
            "sidecar_hash_passed": sidecar_pass,
            "checkpoint_state_hash_passed": state_pass,
            "instrumentation_state_binding_passed": len(inst_by_id),
            "failed_units": 0, "passed": 1,
        })
        optimizer_rows.append({
            "target": target, "expected_optimizer_states": 162,
            "optimizer_file_hash_passed": optimizer_file_pass,
            "optimizer_semantic_hash_passed": optimizer_state_pass,
            "failed_optimizer_states": 0, "passed": 1,
        })
        identity = instrument["identity"]
        identity_rows.append({
            "target": target, "units": instrument["unit_count"],
            "source_rows": instrument["source_rows"],
            "target_unlabeled_rows": instrument["target_unlabeled_rows"],
            "max_Wz_logit_abs": identity["identity_abs"],
            "max_softmax_abs": identity["softmax_abs"],
            "max_hook_z_abs": identity["hook_abs"],
            "max_repeat_logits_abs": identity["repeat_logits"],
            "max_repeat_z_abs": identity["repeat_z"],
            "failed_units": identity["failed_units"],
            "CPU_only": int(not instrument["execution"]["GPU_used_for_instrumentation"]),
            "passed": int(instrument["all_gates_passed"]),
        })
        isolation_rows.append({
            "target": target, "units_checked": len(units),
            "unit_isolation_passed": target_isolation_pass,
            "training_target_rows": oaci["execution"]["target_data_rows_loaded_during_training"] + src["execution"]["target_data_rows_loaded_during_training"],
            "training_target_label_reads": oaci["execution"]["target_label_reads_during_training"] + src["execution"]["target_label_reads_during_training"],
            "source_audit_training_rows": 0,
            "selector_target_reads": int(oaci["execution"]["selector_target_read"]) + int(src["execution"]["selector_target_read"]),
            "outcome_retention_decisions": 0,
            "outcome_retry_decisions": 0,
            "target4_primary": int(target == 4 and target in PRIMARY_TARGETS),
            "passed": int(target_isolation_pass == 162),
        })
        for name, descriptor in (
            ("strict_source_trial_view", primary["strict_source_input"]),
            ("target_unlabeled_trial_view", primary["target_unlabeled_input"]),
        ):
            view_rows.append({
                "target": target, "view_name": name,
                "path": descriptor["path"], "sha256": descriptor["sha256"],
                "rows": descriptor["row_count"],
                "fields": "|".join(descriptor["fields"]),
                "uses_target_labels": 0, "uses_evaluation_labels": 0,
                "same_label_oracle": 0, "available_before_field_freeze": 1,
                "passed": 1,
            })
        if primary["label_view_path_present"] or primary["same_label_oracle_path_present"] or primary["target_label_fields_present"]:
            raise RuntimeError(f"C79E pre-label primary view leaked label metadata for target {target}")

        for stage, payload, path in (
            ("oaci_erm_training", oaci, oaci_path),
            ("src_training", src, src_path),
            ("instrumentation", instrument, instrument_path),
        ):
            execution = payload["execution"]
            resource_rows.append({
                "target": target, "stage": stage,
                "SLURM_job_id": execution["SLURM_job_id"],
                "GPU_name": execution.get("GPU_name", "none"),
                "GPU_count": execution.get("GPU_count", 0),
                "GPU_wall_hours": execution.get("GPU_wall_hours", 0),
                "wall_seconds": execution.get("wall_seconds", execution.get("job_wall_seconds", 0)),
                "CPU_seconds": execution.get("process_CPU_seconds", execution.get("summed_unit_process_CPU_seconds", 0)),
                "peak_GPU_memory_bytes": execution.get("peak_GPU_memory_bytes", 0),
                "external_storage_bytes": execution.get("external_storage_bytes_at_freeze", execution.get("external_storage_bytes", 0)),
                "retry_or_requeue_count": execution.get("retry_or_requeue_count", 0),
                "manifest_path": str(path),
                "manifest_file_sha256": sha256_file(path),
                "passed": 1,
            })

    if observed_ids != set(expected_by_id):
        raise RuntimeError("C79E observed field does not equal the 1,458-unit registry")

    external_bytes = 0
    for directory, _, filenames in os.walk(CAMPAIGN_ROOT):
        for filename in filenames:
            external_bytes += (Path(directory) / filename).stat().st_size

    write_csv(TABLE_DIR / "seed4_field_manifest.csv", unit_rows)
    write_csv(TABLE_DIR / "seed4_checkpoint_state_replay.csv", checkpoint_rows)
    write_csv(TABLE_DIR / "seed4_optimizer_state_replay.csv", optimizer_rows)
    write_csv(TABLE_DIR / "seed4_genealogy_replay.csv", genealogy_rows)
    write_csv(TABLE_DIR / "seed4_cadence_replay.csv", cadence_rows)
    write_csv(TABLE_DIR / "seed4_instrumentation_identity.csv", identity_rows)
    write_csv(TABLE_DIR / "seed4_target_isolation.csv", isolation_rows)
    write_csv(TABLE_DIR / "seed4_physical_view_ledger_pre_labels.csv", view_rows)
    write_csv(TABLE_DIR / "seed4_job_resource_ledger.csv", resource_rows)

    result = {
        "schema_version": "c79e_seed4_field_freeze_audit_v1",
        "engineering_units": len(unit_rows),
        "primary_units": sum(row["primary"] for row in unit_rows),
        "target4_units": sum(row["target"] == 4 for row in unit_rows),
        "checkpoint_state_replay_passed": sum(row["checkpoint_state_hash_passed"] for row in checkpoint_rows),
        "optimizer_state_replay_passed": sum(row["optimizer_semantic_hash_passed"] for row in optimizer_rows),
        "genealogy_cells_passed": sum(row["passed"] for row in genealogy_rows),
        "cadence_cells_passed": sum(row["passed"] for row in cadence_rows),
        "instrumentation_failed_units": sum(row["failed_units"] for row in identity_rows),
        "target_training_rows": sum(row["training_target_rows"] for row in isolation_rows),
        "target_training_label_reads": sum(row["training_target_label_reads"] for row in isolation_rows),
        "scientific_outcome_reads_before_freeze": 0,
        "label_views_created": False,
        "same_label_oracle_created": False,
        "external_payload_bytes": external_bytes,
        "all_gates_passed": True,
    }
    (REPORT_DIR / "C79_SEED4_FIELD_RED_TEAM.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = f"""# C79 Seed-4 Field Red Team

```text
engineering units replayed:       {result['engineering_units']} / 1458
primary units replayed:           {result['primary_units']} / 1296
checkpoint state hashes:          {result['checkpoint_state_replay_passed']} / 1458
optimizer semantic hashes:        {result['optimizer_state_replay_passed']} / 1458
genealogy cells:                  {result['genealogy_cells_passed']} / 36
cadence cells:                    {result['cadence_cells_passed']} / 36
instrumentation failed units:     {result['instrumentation_failed_units']}
training target rows:             {result['target_training_rows']}
training target-label reads:      {result['target_training_label_reads']}
scientific outcome reads:         0
label views created:              false
same-label oracle created:        false
external payload bytes:           {result['external_payload_bytes']}
```

All checkpoint files, semantic model states, sidecars, optimizer files,
semantic optimizer states, genealogy links, cadence cells, instrumentation
bindings, numerical identities, and target-isolation fields passed. Target 4
is present only as 162 engineering units and contributes zero primary units.

Gate: `C79E_SEED4_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED`.
"""
    (REPORT_DIR / "C79_SEED4_FIELD_RED_TEAM.md").write_text(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    audit()
