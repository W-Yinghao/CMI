"""Generate the additive C84 V2 protocols and metadata-only audit tables."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import c84_dataset_registry as v1
from . import c84_dataset_registry_v2 as registry
from . import c84_fixed_zoo_protocol as c84p
from .c84r_montage_repair import (
    CLASS_MAPPING_VERSION,
    COMMON_CHANNELS,
    EPOCH_RULE,
    INTERFACE_ID,
    MONTAGE_SHA256,
    SAMPLE_RATE_HZ,
    UNIT_ID_SALT,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r_tables"
CREATED_AT_UTC = "2026-07-13T21:59:52Z"
REPAIR_COMMIT = "482a725abc6bf1f0e5d33be76ea17d37bcfaa6c3"
REPAIR_PROTOCOL_SHA256 = "a6a1fd85ef1b7520a55ef8e075933d08bf6639cbf89bbcf761dec2a753ab1c91"
HISTORICAL_HEAD = "df95f1375f1883dd706a63f65ee9b6313fa1a779"
DATASET_ORDER = c84p.DATASET_ORDER
PANELS = c84p.PANELS
SEEDS = c84p.SEEDS
LEVELS = c84p.LEVELS
REGIMES = c84p.REGIMES
OACI_EPOCHS = c84p.OACI_EPOCHS
TOTAL_UNITS = 1944
TOTAL_PHASES = 72
TOTAL_CONTEXTS = 944
TOTAL_CANDIDATE_CONTEXTS = 76464
CANARY_UNITS = 243
CANARY_PHASES = 9
CANARY_TARGETS = {"Lee2019_MI": 19, "Cho2017": 24, "PhysionetMI": 106}
COMMON_BUDGETS = (1, 2, 4, 8, "FULL")
EXTENDED_BUDGETS = (16, 32)
PRIMARY_ZERO_METHODS = c84p.PRIMARY_ZERO_METHODS
METHOD_REGISTRY_SHA256 = c84p.METHOD_REGISTRY_SHA256
SUCCESS_GATE = "C84_COMMON_20_CHANNEL_MONTAGE_REPAIRED_CANARY_LOCKED_READY_FOR_PI_AUTHORIZATION"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def write_sha(path: str | Path, digest: str, target_name: str) -> None:
    Path(path).write_text(f"{digest}  {target_name}\n", encoding="ascii")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"refusing empty C84R table: {path}")
    fieldnames = list(fields or rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"C84R CSV schema mismatch: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def candidate_unit_identity(
    dataset: str,
    panel: str,
    seed: int,
    level: int,
    regime: str,
    epoch: int,
    trajectory_order: int,
) -> dict[str, Any]:
    return {
        "identity_salt": UNIT_ID_SALT,
        "milestone": "C84",
        "dataset": dataset,
        "source_panel": panel,
        "training_seed": int(seed),
        "level": int(level),
        "regime": regime,
        "epoch": int(epoch),
        "trajectory_order": int(trajectory_order),
        "interface_id": INTERFACE_ID,
        "montage_sha256": MONTAGE_SHA256,
        "epoch_rule": EPOCH_RULE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "class_mapping_version": CLASS_MAPPING_VERSION,
    }


def candidate_unit_id(**kwargs: Any) -> str:
    return "c84v2_" + hashlib.sha256(canonical_bytes(candidate_unit_identity(**kwargs))).hexdigest()[:32]


def candidate_units() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for panel in PANELS:
            for seed in SEEDS:
                for level in LEVELS:
                    specs = [("ERM", 199, 0)]
                    specs.extend(("OACI", epoch, order) for order, epoch in enumerate(OACI_EPOCHS, start=1))
                    specs.extend(("SRC", epoch, order) for order, epoch in enumerate(OACI_EPOCHS, start=1))
                    for regime, epoch, order in specs:
                        kwargs = dict(dataset=dataset, panel=panel, seed=seed, level=level,
                                      regime=regime, epoch=epoch, trajectory_order=order)
                        identity = candidate_unit_identity(**kwargs)
                        rows.append({
                            "unit_id": candidate_unit_id(**kwargs),
                            **identity,
                            "canary_subset": int(panel == "A" and seed == 5 and level == 0),
                        })
    if len(rows) != TOTAL_UNITS or len({row["unit_id"] for row in rows}) != TOTAL_UNITS:
        raise RuntimeError("C84 V2 candidate identity universe is not 1,944 unique units")
    old_ids = {row["unit_id"] for row in c84p.candidate_units()}
    if old_ids & {row["unit_id"] for row in rows}:
        raise RuntimeError("C84 V2 reused a historical blocked unit ID")
    return rows


def subject_partition_rows() -> list[dict[str, Any]]:
    old_path = REPORT_DIR / "c84p_tables/subject_partition_registry.csv"
    with old_path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    rows = []
    for row in old_rows:
        rows.append({
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "partition": row["partition"],
            "within_panel_role": row["within_panel_role"],
            "subject_hash": row["subject_hash"],
            "source_audit_hash": row["source_audit_hash"],
            "v1_row_identity_sha256": hashlib.sha256(canonical_bytes(row)).hexdigest(),
            "changed_by_montage_repair": 0,
        })
    if len(rows) != 214:
        raise RuntimeError("C84R subject partition replay is not 214 rows")
    return rows


def channel_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        ordered = registry.ordered_dataset_channels(dataset)
        for order, channel in enumerate(COMMON_CHANNELS):
            rows.append({
                "dataset": dataset,
                "order": order,
                "canonical_channel": channel,
                "native_channel": ordered[order],
                "available": 1,
                "scientific_alias": 0,
                "interpolated": 0,
                "zero_filled": 0,
                "dataset_specific_mask": 0,
                "montage_sha256": MONTAGE_SHA256,
            })
    return rows


def dataset_registry_rows() -> list[dict[str, Any]]:
    validation = registry.validate_registry()
    rows = []
    for dataset in DATASET_ORDER:
        spec = registry.DATASETS[dataset]
        partition = registry.partition_subjects(spec)
        rows.append({
            "dataset": dataset,
            "schema_version": registry.SCHEMA_VERSION,
            "interface_id": INTERFACE_ID,
            "channels_available": len(registry.ordered_dataset_channels(dataset)),
            "channels_required": len(COMMON_CHANNELS),
            "channel_order_exact": 1,
            "montage_sha256": MONTAGE_SHA256,
            "source_panel_A": len(partition["source_panel_A"]),
            "source_panel_B": len(partition["source_panel_B"]),
            "targets": len(partition["targets"]),
            "excluded_subjects": "|".join(map(str, spec.excluded_subjects)) or "NONE",
            "registry_validation_pass": int(validation["ready"]),
            "real_EEG_arrays_loaded": 0,
            "real_labels_read": 0,
        })
    return rows


def migration_rows() -> list[dict[str, Any]]:
    old = c84p.candidate_units()
    new = candidate_units()
    if len(old) != len(new):
        raise RuntimeError("C84 candidate migration row count drift")
    rows = []
    for old_row, new_row in zip(old, new):
        key_old = tuple(old_row[key] for key in ("dataset", "panel", "seed", "level", "regime", "epoch", "trajectory_order"))
        key_new = (new_row["dataset"], new_row["source_panel"], new_row["training_seed"], new_row["level"],
                   new_row["regime"], new_row["epoch"], new_row["trajectory_order"])
        if key_old != key_new:
            raise RuntimeError("C84 migration alignment drift")
        rows.append({
            "dataset": old_row["dataset"], "panel": old_row["panel"], "seed": old_row["seed"],
            "level": old_row["level"], "regime": old_row["regime"], "epoch": old_row["epoch"],
            "trajectory_order": old_row["trajectory_order"], "historical_blocked_unit_id": old_row["unit_id"],
            "v2_unit_id": new_row["unit_id"], "identity_changed": int(old_row["unit_id"] != new_row["unit_id"]),
            "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256,
        })
    if not all(row["identity_changed"] for row in rows):
        raise RuntimeError("not every historical C84 unit identity changed")
    return rows


def canary_view_rows() -> list[dict[str, Any]]:
    values = (
        ("source_training_view", 1, 1, 0, 0, "training"),
        ("source_audit_view", 1, 1, 0, 0, "engineering_source_audit_only"),
        ("target_unlabeled_view", 1, 0, 1, 0, "instrumentation_without_y"),
        ("target_construction_view", 0, 0, 0, 0, "not_provisioned_in_C84C"),
        ("target_evaluation_view", 0, 0, 0, 0, "not_provisioned_in_C84C"),
        ("same_label_oracle_view", 0, 0, 0, 0, "physically_unreachable"),
    )
    return [{
        "view": view, "C84C_provisioned": provisioned, "source_y_access": source_y,
        "target_X_access": target_x, "target_y_access": target_y, "purpose": purpose,
        "target_y_indexed": 0, "target_y_hashed": 0, "target_y_logged": 0,
        "target_y_retention_or_retry": 0,
    } for view, provisioned, source_y, target_x, target_y, purpose in values]


def build_external_protocol() -> dict[str, Any]:
    return {
        "schema_version": "c84_multidataset_external_validity_protocol_v2",
        "milestone": "C84R",
        "status": "LOCKED_20_CHANNEL_INTERFACE_FUTURE_STAGES_SEPARATELY_AUTHORIZED",
        "created_at_utc": CREATED_AT_UTC,
        "supersession": {
            "historical_C84P_HEAD": HISTORICAL_HEAD,
            "repair_protocol_commit": REPAIR_COMMIT,
            "repair_protocol_sha256": REPAIR_PROTOCOL_SHA256,
            "historical_objects_rewritten": False,
        },
        "epistemic_status": {
            "prospective_to_all_C84_real_data_and_outcomes": True,
            "availability_only_repair": True,
            "outcome_dependent_choice": False,
            "external_cohorts": True,
            "universal_EEG_validity": False,
        },
        "interface": registry.dataset_registry_payload(),
        "subject_partition": {
            "salt": registry.SUBJECT_PARTITION_SALT,
            "source_panel_A": 16, "source_panel_B": 16,
            "source_training_per_panel": 12, "source_audit_per_panel": 4,
            "targets": {code: len(registry.partition_subjects(registry.DATASETS[code])["targets"]) for code in DATASET_ORDER},
            "unchanged_from_C84P": True,
        },
        "candidate_field": {
            "identity_salt": UNIT_ID_SALT, "interface_bound_unit_ids": True,
            "panels": list(PANELS), "seeds": list(SEEDS), "levels": list(LEVELS),
            "units_per_zoo": 81, "units_per_dataset": 648, "total_units": TOTAL_UNITS,
            "training_phases": TOTAL_PHASES, "target_contexts": TOTAL_CONTEXTS,
            "candidate_context_evaluations": TOTAL_CANDIDATE_CONTEXTS,
            "target_specific_retraining": False,
        },
        "selectors": {
            "method_registry_sha256": METHOD_REGISTRY_SHA256,
            "primary_zero_label": list(PRIMARY_ZERO_METHODS), "strict_source": "S1",
            "labeled_primary": "Q0_B1", "new_method_or_retuning": False,
        },
        "budgets": {"primary": list(COMMON_BUDGETS), "Lee_Cho_secondary": list(EXTENDED_BUDGETS), "FULL_cell_specific": True},
        "inference": {"unchanged_from_C84P": True, "registry_sha256": "11ec3b37e8538c75e7ebae994b8af30c8d2acdddbe7ec989fe0f85e2718bafeb"},
        "authorization": {
            "fresh_direct_authorization_per_stage": True, "magic_token_required": False,
            "prior_conversational_C84_authorization_active": False,
            "C84C_authorized": False, "C84F_authorized": False, "C84S_authorized": False,
        },
        "resource_envelopes": {"GPU_phase_hours": 250, "external_payload_TiB": 2.0, "Git_max_file_MiB": 50},
    }


def build_canary_protocol(external_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "c84_canary_protocol_v2", "milestone": "C84C",
        "status": "LOCKED_PROTOCOL_IMPLEMENTATION_LOCK_REQUIRED_NOT_AUTHORIZED",
        "parent_external_protocol_sha256": external_sha,
        "repair_protocol_sha256": REPAIR_PROTOCOL_SHA256,
        "scope": {
            "datasets": list(DATASET_ORDER), "source_panel": "A", "training_seed": 5, "level": 0,
            "target_subjects": CANARY_TARGETS, "units_per_dataset": 81,
            "total_units": CANARY_UNITS, "training_phases": CANARY_PHASES,
            "engineering_only": True,
        },
        "interface": {"id": INTERFACE_ID, "channels": list(COMMON_CHANNELS), "montage_sha256": MONTAGE_SHA256,
                      "epoch_rule": EPOCH_RULE, "sample_rate_hz": SAMPLE_RATE_HZ, "input_shape": [20, 480]},
        "checks": ["dataset_access", "license_path", "source_subject_identity", "event_mapping", "channel_order",
                   "native_sfreq", "anti_aliased_resampling", "input_shape", "finite_values_units",
                   "source_train_audit_isolation", "target_y_stripping", "training_determinism", "checkpoint_cadence",
                   "optimizer_state_sidecar", "logit_probability_hook_identity", "storage_runtime"],
        "forbidden_outputs": ["target_accuracy", "target_calibration", "target_regret", "target_label_counts",
                              "selector_scores", "Q1", "Q2", "label_budget_frontier", "cross_dataset_science"],
        "target_unlabeled_payload_fields": ["X", "stable_trial_id", "target_subject_id", "session", "run", "dataset_id"],
        "target_y_operations": {"index": False, "hash": False, "summarize": False, "log": False,
                                "retention": False, "retry": False},
        "reuse_in_C84F": "only_if_243_complete_all_engineering_gates_exact_ID_hash_no_outcome_decision",
        "fresh_direct_PI_authorization_required": True,
    }


def build_field_protocol(external_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "c84_field_generation_protocol_v2", "milestone": "C84F",
        "status": "LOCKED_PROTOCOL_ONLY_NO_EXECUTION_LOCK_NOT_AUTHORIZED",
        "parent_external_protocol_sha256": external_sha,
        "field": {"total_units": TOTAL_UNITS, "training_phases": TOTAL_PHASES,
                  "canary_reusable_units": CANARY_UNITS, "remaining_units_after_valid_canary": 1701,
                  "remaining_phases_after_valid_canary": 63, "target_contexts": TOTAL_CONTEXTS,
                  "candidate_context_evaluations": TOTAL_CANDIDATE_CONTEXTS},
        "candidate_identity": {"salt": UNIT_ID_SALT, "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256},
        "historical_implementations": {"ERM": "stage1_final_anchor", "OACI": "40_checkpoint_trajectory",
                                       "SRC": "C11_negative_control_smooth_temperature_0.1",
                                       "epochs": 200, "checkpoint_every": 5},
        "isolation": {"training_target_rows": 0, "training_target_labels": 0,
                      "target_outcome_retention": 0, "target_outcome_retry": 0},
        "fresh_direct_PI_authorization_after_canary_review": True,
        "scope_specific_execution_lock_created_in_C84R": False,
    }


def build_science_protocol(external_sha: str) -> dict[str, Any]:
    historical = read_json(REPORT_DIR / "C84_SCIENTIFIC_ANALYSIS_PROTOCOL.json")
    return {
        **{key: value for key, value in historical.items() if key not in {"schema_version", "status", "parent_protocol_sha256", "open_blocker"}},
        "schema_version": "c84_scientific_analysis_protocol_v2",
        "status": "LOCKED_PROTOCOL_ONLY_NO_EXECUTION_LOCK_NOT_AUTHORIZED",
        "parent_external_protocol_sha256": external_sha,
        "interface_id": INTERFACE_ID,
        "montage_sha256": MONTAGE_SHA256,
        "scientific_contract_changed_from_C84P": False,
        "fresh_direct_PI_authorization_after_full_field_freeze": True,
        "scope_specific_execution_lock_created_in_C84R": False,
    }


def table_rows() -> dict[str, list[dict[str, Any]]]:
    units = candidate_units()
    return {
        "dataset_registry_v2_replay.csv": dataset_registry_rows(),
        "candidate_unit_id_migration.csv": migration_rows(),
        "candidate_field_arithmetic.csv": [
            {"scope": "C84C", "units": 243, "training_phases": 9, "formula": "3*81_and_3*3", "interface_id": INTERFACE_ID},
            {"scope": "C84F_remaining_after_reusable_C84C", "units": 1701, "training_phases": 63, "formula": "1944-243_and_72-9", "interface_id": INTERFACE_ID},
            {"scope": "C84_complete", "units": len(units), "training_phases": 72, "formula": "3*2*2*2*81_and_3*2*2*2*3", "interface_id": INTERFACE_ID},
        ],
        "canary_scope_registry.csv": [{
            "dataset": dataset, "panel": "A", "seed": 5, "level": 0, "units": 81, "training_phases": 3,
            "target_subject": CANARY_TARGETS[dataset], "engineering_only": 1, "scientific_metrics": 0,
        } for dataset in DATASET_ORDER],
        "canary_target_registry.csv": [{
            "dataset": dataset, "target_subject": subject,
            "partition_first_target": registry.partition_subjects(registry.DATASETS[dataset])["targets"][0],
            "partition_replay_pass": int(subject == registry.partition_subjects(registry.DATASETS[dataset])["targets"][0]),
            "target_y_access": 0,
        } for dataset, subject in CANARY_TARGETS.items()],
        "canary_view_access_matrix.csv": canary_view_rows(),
        "target_label_isolation_contract.csv": [{
            "operation": operation, "allowed": 0, "fail_closed": 1,
            "implementation_requirement": "target label object discarded without indexing/hash/summary/logging",
        } for operation in ("index_target_y", "hash_target_y", "summarize_target_y", "log_target_y",
                            "retain_by_target_y", "retry_by_target_y", "pass_target_y_to_training", "pass_target_y_to_instrumentation")],
        "resource_estimate.csv": [
            {"scope": "C84C", "resource": "candidate_units", "estimate": "243", "unit": "units", "envelope": "fixed", "within_envelope": 1},
            {"scope": "C84C", "resource": "GPU_phase_hours_safety", "estimate": "7.109241", "unit": "hours", "envelope": "250", "within_envelope": 1},
            {"scope": "C84C", "resource": "CPU_instrumentation", "estimate": "2.0", "unit": "million_row_upper_bound", "envelope": "planning", "within_envelope": 1},
            {"scope": "C84C", "resource": "download_bytes", "estimate": str(180 * 1024**3), "unit": "bytes_upper_bound", "envelope": str(2 * 1024**4), "within_envelope": 1},
            {"scope": "C84C", "resource": "external_payload", "estimate": "70", "unit": "GiB", "envelope": "2048", "within_envelope": 1},
            {"scope": "C84C", "resource": "calendar_wall", "estimate": "18", "unit": "hours", "envelope": "planning", "within_envelope": 1},
            {"scope": "C84F_remaining", "resource": "candidate_units", "estimate": "1701", "unit": "units", "envelope": "fixed", "within_envelope": 1},
            {"scope": "C84F_remaining", "resource": "GPU_phase_hours_safety", "estimate": "49.764684", "unit": "hours", "envelope": "250", "within_envelope": 1},
            {"scope": "C84F_remaining", "resource": "training_phases", "estimate": "63", "unit": "phases", "envelope": "fixed", "within_envelope": 1},
            {"scope": "C84_complete", "resource": "candidate_units", "estimate": "1944", "unit": "units", "envelope": "fixed", "within_envelope": 1},
            {"scope": "C84_complete", "resource": "training_phases", "estimate": "72", "unit": "phases", "envelope": "fixed", "within_envelope": 1},
            {"scope": "C84_complete", "resource": "GPU_phase_hours_safety", "estimate": "56.873925", "unit": "hours", "envelope": "250", "within_envelope": 1},
            {"scope": "C84_complete", "resource": "download_plus_derived", "estimate": "630", "unit": "GiB", "envelope": "2048", "within_envelope": 1},
        ],
        "risk_register.csv": [{
            "risk": risk, "blocking": 0, "status": "CLOSED_BY_C84R_LOCKED_CONTROL",
            "control": control, "real_outcome_access": 0,
        } for risk, control in (
            ("historical_21_channel_protocol_rewritten", "additive V2 filenames and supersession ledger"),
            ("FCz_reintroduced", "exact montage digest validation"),
            ("Fz_substitution", "Fz forbidden by V2 channel contract"),
            ("channel_interpolation", "interpolation false and fail-closed"),
            ("dataset_specific_mask", "all datasets exact 20/20 order"),
            ("subject_partition_drift", "214-row bit-for-bit replay"),
            ("old_unit_ID_reused", "1,944 migration rows require changed identity"),
            ("target_labels_reaching_canary", "target-unlabeled payload schema excludes y"),
            ("unauthorized_C84C", "guard precedes loader import"),
            ("C84F_or_C84S_lock_created", "C84R creates C84C lock only"),
            ("raw_EEG_or_weights_in_git", "payload hygiene scan"),
            ("C34S_regression_omitted", "leading-numeric suite parser"),
        )],
    }


def generate() -> dict[str, Any]:
    validation = registry.validate_registry()
    if not validation["ready"]:
        raise RuntimeError(f"C84 V2 registry failed: {validation}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in table_rows().items():
        write_csv(TABLE_DIR / name, rows)

    external_path = REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json"
    write_json(external_path, build_external_protocol())
    external_sha = sha256_file(external_path)
    write_sha(external_path.with_suffix(".sha256"), external_sha, external_path.name)
    builders = (
        ("C84_CANARY_PROTOCOL_V2", build_canary_protocol),
        ("C84_FIELD_GENERATION_PROTOCOL_V2", build_field_protocol),
        ("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2", build_science_protocol),
    )
    hashes = {}
    for stem, builder in builders:
        path = REPORT_DIR / f"{stem}.json"
        write_json(path, builder(external_sha))
        digest = sha256_file(path)
        write_sha(path.with_suffix(".sha256"), digest, path.name)
        hashes[stem] = digest
    hashes["C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2"] = external_sha
    write_csv(TABLE_DIR / "stage_protocol_hash_replay.csv", [{
        "protocol": stem, "path": f"oaci/reports/{stem}.json", "sha256": digest,
        "hash_replay_pass": int(sha256_file(REPORT_DIR / f"{stem}.json") == digest),
        "authorized": 0,
    } for stem, digest in sorted(hashes.items())])
    return {
        "schema_version": "c84r_v2_protocol_generation_v1",
        "protocol_hashes": hashes,
        "candidate_units_planned": len(candidate_units()),
        "candidate_units_created": 0,
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "dataset_downloads": 0,
        "C84C_authorized": False,
        "C84F_or_C84S_execution_locks_created": 0,
    }


def main() -> int:
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
