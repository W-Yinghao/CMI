"""Additive C84R exact-intersection montage repair protocol.

This module is metadata-only.  It preserves the blocked C84P artifacts and
creates a new 20-channel protocol identity before any real C84 adapter exists.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import c84_dataset_registry as v1


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r_tables"
CREATED_AT_UTC = "2026-07-13T21:54:23Z"
C84P_HEAD = "df95f1375f1883dd706a63f65ee9b6313fa1a779"

INTERFACE_ID = "C84_LEFT_RIGHT_20CH_160HZ_0_3S_V2"
UNIT_ID_SALT = "C84_FIXED_ZOO_LEFT_RIGHT_20CH_V2"
CLASS_MAPPING_VERSION = "C84_LEFT_RIGHT_CLASS_MAPPING_V1"
EPOCH_RULE = "half_open_[0.0,3.0)_480_samples"
SAMPLE_RATE_HZ = 160
COMMON_CHANNELS = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)
MONTAGE_SHA256 = "988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04"

HISTORICAL_PROTOCOLS = {
    "external": (
        "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL.json",
        "ebfe9933ac838af22cc6553f81ba87d806996e99ce33158c7a5c30b9a1f5e824",
    ),
    "canary": (
        "C84_CANARY_PROTOCOL.json",
        "bacc511dec01c2141470e689e62b6664089ab2ce7b78255b46acd14446cbfffd",
    ),
    "field": (
        "C84_FIELD_GENERATION_PROTOCOL.json",
        "cbb515772ad4257b59e64a499c0af909c6bfabe54c7a2c3f3226e4cccd6d6f15",
    ),
    "science": (
        "C84_SCIENTIFIC_ANALYSIS_PROTOCOL.json",
        "3c6810f99c1e69cd4e0758ff3ea2ca81799d06c0d18dc7d29e4d128a3ef4590c",
    ),
}

UNCHANGED_REGISTRIES = {
    "subject_partitions": ("c84p_tables/subject_partition_registry.csv", "bb98564730b4e9cd4fc9c2326c5c30943a2dcf7e47bc1ce09ae9cefc68555d63"),
    "selector_registry": ("c84p_tables/selector_registry_replay.csv", "d589437e40812350eec44bdfbf1b75c52f10ef41e0e3ca5868e07844b0228e68"),
    "budgets": ("c84p_tables/common_and_extended_budget_registry.csv", "86d39c9ae3a40ea071447aca7bdeed98da064cc7a9f8c9ec91f2188bf097c65b"),
    "inference": ("c84p_tables/inference_registry.csv", "11ec3b37e8538c75e7ebae994b8af30c8d2acdddbe7ec989fe0f85e2718bafeb"),
}


class C84RMontageError(RuntimeError):
    """Raised when an interface differs from the locked 20-channel repair."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def montage_bytes(channels: Sequence[str]) -> bytes:
    return json.dumps(list(channels), separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def write_sha(path: str | Path, digest: str, target_name: str) -> None:
    Path(path).write_text(f"{digest}  {target_name}\n")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"refusing empty C84R table: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"C84R table schema drift: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_intersection() -> tuple[str, ...]:
    available = set.intersection(*(set(spec.native_eeg_channels) for spec in v1.DATASETS.values()))
    return tuple(channel for channel in v1.PRIMARY_CHANNELS if channel in available)


def validate_montage(
    channels: Sequence[str],
    *,
    dataset_channels: Mapping[str, Sequence[str]] | None = None,
    substituted_channels: Mapping[str, str] | None = None,
    interpolation: bool = False,
    zero_fill: bool = False,
    dataset_specific_mask: bool = False,
) -> dict[str, Any]:
    observed = tuple(channels)
    if observed != COMMON_CHANNELS:
        raise C84RMontageError("C84R channel list/order differs from exact 20-channel interface")
    if sha256_bytes(montage_bytes(observed)) != MONTAGE_SHA256:
        raise C84RMontageError("C84R montage digest mismatch")
    if substituted_channels:
        raise C84RMontageError("C84R channel substitution is forbidden")
    if interpolation:
        raise C84RMontageError("C84R channel interpolation is forbidden")
    if zero_fill:
        raise C84RMontageError("C84R channel zero filling is forbidden")
    if dataset_specific_mask:
        raise C84RMontageError("C84R dataset-specific masks are forbidden")
    available = dataset_channels or {
        code: spec.native_eeg_channels for code, spec in v1.DATASETS.items()
    }
    for dataset in v1.DATASETS:
        if dataset not in available:
            raise C84RMontageError(f"C84R channel availability missing for {dataset}")
        if any(channel not in set(available[dataset]) for channel in COMMON_CHANNELS):
            raise C84RMontageError(f"C84R {dataset} does not provide the exact interface")
    return {
        "interface_id": INTERFACE_ID,
        "channels": list(observed),
        "channel_count": len(observed),
        "montage_sha256": MONTAGE_SHA256,
        "all_datasets_complete": True,
        "substitution": False,
        "interpolation": False,
        "zero_fill": False,
        "dataset_specific_mask": False,
    }


def _verify_historical_objects() -> dict[str, str]:
    observed: dict[str, str] = {}
    for role, (name, expected) in HISTORICAL_PROTOCOLS.items():
        digest = sha256_file(REPORT_DIR / name)
        if digest != expected:
            raise C84RMontageError(f"historical C84P {role} protocol hash mismatch")
        observed[role] = digest
    for role, (relative, expected) in UNCHANGED_REGISTRIES.items():
        digest = sha256_file(REPORT_DIR / relative)
        if digest != expected:
            raise C84RMontageError(f"historical C84P {role} registry hash mismatch")
    return observed


def build_repair_protocol() -> dict[str, Any]:
    historical = _verify_historical_objects()
    if exact_intersection() != COMMON_CHANNELS:
        raise C84RMontageError("locked repair is not the exact requested-channel intersection")
    montage = validate_montage(COMMON_CHANNELS)
    return {
        "schema_version": "c84r_common_montage_repair_protocol_v1",
        "milestone": "C84R",
        "created_at_utc": CREATED_AT_UTC,
        "status": "LOCKED_REPAIR_DECISION_IMPLEMENTATION_PENDING",
        "historical_C84P": {
            "HEAD": C84P_HEAD,
            "final_gate": "C84_DATASET_CHANNEL_EVENT_RESOURCE_OR_PROTOCOL_RECONCILIATION_REQUIRED",
            "protocols": {
                role: {"path": name, "sha256": digest, "preserved": True, "operative_for_execution": False}
                for role, (name, digest) in HISTORICAL_PROTOCOLS.items()
            },
            "history_rewritten": False,
        },
        "repair_decision": {
            **montage,
            "reason": "exact_cross_dataset_availability_intersection",
            "dropped_from_original": ["FCz"],
            "Fz_in_primary_track": False,
            "Fz_substitution": False,
            "spatial_interpolation": False,
            "outcome_dependent_choice": False,
            "availability_only": True,
        },
        "interface": {
            "id": INTERFACE_ID,
            "unit_identity_salt": UNIT_ID_SALT,
            "epoch_rule": EPOCH_RULE,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "input_shape": [20, 480],
            "class_mapping_version": CLASS_MAPPING_VERSION,
        },
        "pre_repair_protected_state": {
            "real_EEG_arrays_loaded": 0,
            "real_labels_read": 0,
            "dataset_downloads": 0,
            "training_or_forward_runs": 0,
            "GPU_jobs": 0,
            "real_candidate_units": 0,
            "selector_results": 0,
        },
        "unchanged_scientific_objects": {
            "subject_partitions": True,
            "source_train_audit_splits": True,
            "method_registry": True,
            "candidate_count": True,
            "budgets": True,
            "inference": True,
            "registry_hashes": {
                role: {"path": relative, "sha256": digest}
                for role, (relative, digest) in UNCHANGED_REGISTRIES.items()
            },
        },
        "candidate_identity_repair": {
            "old_planned_units_realized": 0,
            "old_planned_unit_ids_operative": False,
            "new_unit_count": 1944,
            "new_ids_must_bind_interface": True,
            "same_identity_function_C84C_C84F": True,
        },
        "authorization": {
            "C84C_active_authorization": False,
            "C84F_active_authorization": False,
            "C84S_active_authorization": False,
            "old_conversational_statement_active": False,
            "fresh_direct_stage_authorization_required": True,
            "magic_token_required": False,
        },
        "next_stage": "mechanical_V2_implementation_and_C84C_execution_lock_only",
    }


def _supersession_rows() -> list[dict[str, Any]]:
    rows = []
    for role, (name, digest) in HISTORICAL_PROTOCOLS.items():
        rows.append({
            "role": role,
            "historical_path": f"oaci/reports/{name}",
            "historical_sha256": digest,
            "hash_replay": "PASS",
            "historical_status": "BLOCKED_21_CHANNEL_OBJECT",
            "edited_in_place": 0,
            "operative_for_future_execution": 0,
            "superseded_additively": 1,
        })
    return rows


def _intersection_rows() -> list[dict[str, Any]]:
    rows = []
    for order, channel in enumerate(v1.PRIMARY_CHANNELS, start=1):
        availability = {
            dataset: int(channel in spec.native_eeg_channels)
            for dataset, spec in v1.DATASETS.items()
        }
        retained = int(all(availability.values()))
        rows.append({
            "original_order": order,
            "channel": channel,
            **availability,
            "in_exact_intersection": retained,
            "repair_action": "RETAIN" if retained else "DROP_ALL_DATASETS",
            "substitute_with_Fz": 0,
            "interpolate": 0,
        })
    return rows


def _subject_identity_rows() -> list[dict[str, Any]]:
    old_path = REPORT_DIR / "c84p_tables/subject_partition_registry.csv"
    with old_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{
        "dataset": row["dataset"],
        "subject_id": row["subject_id"],
        "partition": row["partition"],
        "within_panel_role": row["within_panel_role"],
        "subject_hash": row["subject_hash"],
        "source_audit_hash": row["source_audit_hash"],
        "C84P_row_identity": sha256_bytes(canonical_bytes(row)),
        "C84R_recomputed_identity": sha256_bytes(canonical_bytes(row)),
        "bit_for_bit_replay": 1,
    } for row in rows]


def generate_repair_protocol() -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    protocol_path = REPORT_DIR / "C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.json"
    write_json(protocol_path, build_repair_protocol())
    protocol_sha = sha256_file(protocol_path)
    write_sha(
        REPORT_DIR / "C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.sha256",
        protocol_sha,
        protocol_path.name,
    )
    write_csv(TABLE_DIR / "protocol_supersession_ledger.csv", _supersession_rows())
    write_csv(TABLE_DIR / "channel_intersection_replay.csv", _intersection_rows())
    write_csv(TABLE_DIR / "channel_order_registry.csv", [{
        "order": index,
        "channel": channel,
        "interface_id": INTERFACE_ID,
        "montage_sha256": MONTAGE_SHA256,
        "Fz": int(channel == "Fz"),
        "FCz": int(channel == "FCz"),
    } for index, channel in enumerate(COMMON_CHANNELS, start=1)])
    write_csv(TABLE_DIR / "montage_repair_decision.csv", [{
        "decision_id": "C84R_EXACT_INTERSECTION_001",
        "old_channel_count": 21,
        "new_channel_count": 20,
        "dropped_channel": "FCz",
        "reason": "exact_cross_dataset_availability_intersection",
        "canonical_json_sha256": MONTAGE_SHA256,
        "Fz_substitution": 0,
        "interpolation": 0,
        "zero_fill": 0,
        "dataset_specific_mask": 0,
        "outcome_dependent": 0,
        "real_access_before_decision": 0,
    }])
    write_csv(TABLE_DIR / "subject_partition_identity_replay.csv", _subject_identity_rows())
    write_csv(TABLE_DIR / "failure_reason_ledger.csv", [{
        "failure_id": "C84P_CHANNEL_001",
        "source_milestone": "C84P",
        "root_cause": "Lee2019_MI lacks requested FCz",
        "scientific_outcome_access": 0,
        "repair": "drop_FCz_from_all_datasets_exact_intersection",
        "repair_changes_subjects_methods_budgets_inference": 0,
        "status": "CLOSED_BY_PROSPECTIVE_C84R_DECISION",
    }])
    timing_path = REPORT_DIR / "C84R_PROTOCOL_TIMING_AUDIT.md"
    timing_path.write_text(
        "# C84R Protocol Timing Audit\n\n"
        "The accepted C84P HEAD is `df95f1375f1883dd706a63f65ee9b6313fa1a779`. "
        "Its committed final red team records zero real EEG arrays, labels, downloads, "
        "training/forward runs, GPU jobs, candidate units and selector outcomes.\n\n"
        "The C84R repair is availability-only and was instantiated before any real C84 adapter. "
        "It drops FCz from every dataset, uses no Fz substitution or interpolation, and changes "
        "no subject partition, method, candidate count, budget or inference rule.\n\n"
        "The four historical 21-channel hashes remain content-valid, preserved and non-operative. "
        "This repair protocol must be committed before the C84C real adapter and V2 execution lock.\n\n"
        "```text\n"
        "real EEG access before repair:       0\n"
        "real label access before repair:     0\n"
        "outcome-dependent decisions:         0\n"
        "history rewritten:                    0\n"
        "active C84C/F/S authorization:        0\n"
        "```\n"
    )
    return {
        "protocol_sha256": protocol_sha,
        "montage_sha256": MONTAGE_SHA256,
        "channels": len(COMMON_CHANNELS),
        "historical_hashes_replayed": len(HISTORICAL_PROTOCOLS),
        "subject_rows_replayed": len(_subject_identity_rows()),
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "gate": "C84R_REPAIR_PROTOCOL_COMMIT_READY",
    }


def main() -> int:
    print(json.dumps(generate_repair_protocol(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
