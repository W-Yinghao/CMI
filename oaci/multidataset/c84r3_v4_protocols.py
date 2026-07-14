"""Generate additive C84C V4 and C84F V4 protocol objects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import c84r3_canary_runtime_repair as runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
CREATED_AT_UTC = "2026-07-14T00:30:00Z"
REPAIR_PROTOCOL_COMMIT = "1c523a4749444136a00b502204b0ed06cac0e5d2"
REPAIR_PROTOCOL_SHA256 = "cdbdb9a25dc29b6a37ac9eb65f130f44efa120042dfb7ddb140cf3db103ec196"
IMPLEMENTATION_COMMIT = "10c60d92f61dd091fef7a08f686a7ce85d99eb07"
HISTORICAL_CANARY_V3_SHA256 = "34cf9e9daca2578ed22c64345e014c0b9fa08b31c4c04939ba13c112c5f57dac"
HISTORICAL_FIELD_V3_SHA256 = "1a6d39443194501d8b09bd44fa87fd77a5665b5a215ccec7c9ec0b3ef865af81"
HISTORICAL_LOCK_V2_SHA256 = "2e38dcd63c02a887b1dcf7eaa26749709dbfb5187373de7808efae21afb0285b"
FAILED_ATTEMPT_SHA256 = "10b05a8989501a3f7af913be74e57a72ef633f88e20a8d3ccabd59abb5673469"
EXTERNAL_V2_SHA256 = "522e6fe8372f8c73741ed146a27068076db8c3d7087f4c4a36760fe0328b7c2f"
SCIENCE_V2_SHA256 = "dc33b22527352bd42989c26f6771b4a49dc1443d458962587ca3d70ad76dd631"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checked(name: str, expected_sha256: str) -> dict[str, Any]:
    path = REPORT_DIR / name
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"C84R3 predecessor hash drift: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_protocol(name: str, payload: dict[str, Any]) -> str:
    path = REPORT_DIR / name
    path.write_bytes(canonical_bytes(payload) + b"\n")
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def build_canary_protocol() -> dict[str, Any]:
    prior = _read_checked("C84_CANARY_PROTOCOL_V3.json", HISTORICAL_CANARY_V3_SHA256)
    return {
        **prior,
        "schema_version": "c84_canary_protocol_v4",
        "status": "LOCKED_PROTOCOL_REPLACEMENT_LOCK_REQUIRED_NOT_AUTHORIZED",
        "created_at_utc": CREATED_AT_UTC,
        "supersession": {
            **prior["supersession"],
            "historical_C84C_protocol_V3_sha256": HISTORICAL_CANARY_V3_SHA256,
            "historical_C84C_lock_V2_sha256": HISTORICAL_LOCK_V2_SHA256,
            "historical_V3_authorization_consumed": True,
            "historical_V3_objects_execution_authority": False,
            "C84R3_repair_protocol_commit": REPAIR_PROTOCOL_COMMIT,
            "C84R3_repair_protocol_sha256": REPAIR_PROTOCOL_SHA256,
            "C84R3_implementation_commit": IMPLEMENTATION_COMMIT,
            "historical_objects_rewritten": False,
        },
        "engineering_failure_895366": {
            "failed_attempt_report_sha256": FAILED_ATTEMPT_SHA256,
            "failed_stage": "training_and_instrumentation:Lee2019_MI",
            "observed_Wz_plus_b_max_abs_error": 2.86102294921875e-6,
            "softmax_repeat_logits_repeat_z_max_abs_error": 0.0,
            "complete_units": 0,
            "target_y_access": 0,
            "target_scientific_metrics": 0,
            "authorization_consumed": True,
            "failed_external_root_preserved": True,
            "failed_artifact_reuse_allowed": False,
        },
        "instrumentation": {
            **prior["instrumentation"],
            "linear_z_classifier_logits_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "softmax_repeat_logits_repeat_z_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
            "tolerance_scope": "float32_1040_term_z_classifier_linear_reconstruction_only",
            "scientific_metric_or_threshold_changed": False,
        },
        "persisted_replay": {
            **prior["persisted_replay"],
            "linear_z_classifier_logits_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "strict_identity_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
            "same_split_tolerance_contract_as_in_memory": True,
        },
        "attempt_ledger": {
            **prior["attempt_ledger"],
            "historical_failed_attempt_preserved": True,
            "replacement_attempt_uses_new_content_addressed_root": True,
        },
        "authorization": {
            **prior["authorization"],
            "fresh_direct_PI_authorization_required": True,
            "historical_authorization_reusable": False,
            "failed_authorization_reused": False,
            "record_schema": "c84c_direct_pi_authorization_record_v3",
            "record_path": "oaci/reports/C84C_PI_AUTHORIZATION_RECORD_V3.json",
        },
        "retry": {
            "retrain_all_243_units": True,
            "reuse_failed_checkpoint_optimizer_or_instrumentation": False,
            "new_external_root": str(runtime.DEFAULT_EXTERNAL_ROOT),
            "outcome_driven_retry": False,
        },
        "protected_state": {
            "prior_real_EEG_views_materialized": 3,
            "prior_source_label_arrays_read": 2,
            "prior_target_y_access": 0,
            "prior_target_scientific_metrics": 0,
            "prior_complete_units": 0,
            "replacement_real_EEG_access": 0,
            "replacement_training_forward_GPU_jobs": 0,
            "replacement_candidate_units_created": 0,
            "replacement_authorization_record_present": False,
        },
    }


def build_field_protocol(canary_sha256: str) -> dict[str, Any]:
    prior = _read_checked("C84_FIELD_GENERATION_PROTOCOL_V3.json", HISTORICAL_FIELD_V3_SHA256)
    return {
        **{key: value for key, value in prior.items() if key not in {
            "parent_canary_protocol_v3_sha256",
            "scope_specific_execution_lock_created_in_C84R2",
        }},
        "schema_version": "c84_field_generation_protocol_v4",
        "status": "LOCKED_PROTOCOL_ONLY_NO_EXECUTION_LOCK_NOT_AUTHORIZED",
        "parent_canary_protocol_v4_sha256": canary_sha256,
        "historical_field_protocol_v3_sha256": HISTORICAL_FIELD_V3_SHA256,
        "C84R3_repair_protocol_sha256": REPAIR_PROTOCOL_SHA256,
        "scientific_field_scope_changed": False,
        "instrumentation_replay_tolerances": {
            "linear_z_classifier_logits_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "softmax_repeat_logits_repeat_z_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
        },
        "canary_reuse": {
            **prior["canary_reuse"],
            "C84C_V3_failed_partial_root_reusable": False,
            "C84C_V4_complete_manifest_required": True,
            "C84C_V4_units_must_use_exact_C84F_candidate_IDs": True,
            "failed_attempt_895366_units_reusable": 0,
            "replacement_units_required": 243,
        },
        "fresh_direct_PI_authorization_after_canary_review": True,
        "scope_specific_execution_lock_created_in_C84R3": False,
    }


def generate() -> dict[str, Any]:
    if sha256_file(runtime.REPAIR_PROTOCOL_PATH) != REPAIR_PROTOCOL_SHA256:
        raise RuntimeError("C84R3 repair protocol hash drift")
    _read_checked("C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json", EXTERNAL_V2_SHA256)
    _read_checked("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.json", SCIENCE_V2_SHA256)
    canary_sha = _write_protocol("C84_CANARY_PROTOCOL_V4.json", build_canary_protocol())
    field_sha = _write_protocol("C84_FIELD_GENERATION_PROTOCOL_V4.json", build_field_protocol(canary_sha))
    return {
        "canary_protocol_v4_sha256": canary_sha,
        "field_protocol_v4_sha256": field_sha,
        "external_protocol_v2_sha256": EXTERNAL_V2_SHA256,
        "science_protocol_v2_sha256": SCIENCE_V2_SHA256,
        "scientific_interface_changed": False,
        "real_data_access_during_generation": 0,
        "C84C_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
