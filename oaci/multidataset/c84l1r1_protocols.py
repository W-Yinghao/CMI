"""Build additive C84L1C V2 canary and V6 field protocols."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
CREATED_AT_UTC = "2026-07-14T16:58:31Z"
REPAIR_SHA256 = "2e199f6f63dffd1b02c1e31102ed189e31bf6e4961465394230f8e9de1d4ddf0"
FAILED_JOB = 895928
FAILED_MANIFEST_SHA256 = "ba67a4a0f8a516085b3eb020c353c401c2eafdd1981eb880c5c63587ac31b091"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(name: str) -> dict[str, Any]:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def build_canary_protocol() -> dict[str, Any]:
    prior_path = REPORT_DIR / "C84_LEVEL1_CANARY_PROTOCOL_V1.json"
    protocol = deepcopy(_read(prior_path.name))
    protocol.update({
        "schema_version": "c84_level1_canary_protocol_v2",
        "created_at_utc": CREATED_AT_UTC,
        "parent_canary_protocol_v1_sha256": sha256_file(prior_path),
        "numerical_repair_protocol_sha256": REPAIR_SHA256,
        "status": "LOCKED_REPLACEMENT_PROTOCOL_IMPLEMENTATION_AND_EXECUTION_LOCK_REQUIRED_NOT_AUTHORIZED",
    })
    protocol["authorization"] = {
        **protocol["authorization"],
        "record_path": "oaci/reports/C84L1C_PI_AUTHORIZATION_RECORD_V2.json",
        "historical_authorization_reusable": False,
        "fresh_direct_PI_authorization_required": True,
    }
    protocol["instrumentation_replay_tolerances"] = {
        "linear_z_classifier_logits_abs_tolerance": 2e-5,
        "softmax_repeat_logits_repeat_z_abs_tolerance": 1e-6,
        "linear_scope": "float32_1040_term_CPU_GPU_classifier_reconstruction_only",
    }
    protocol["historical_failed_attempt"] = {
        "job_id": FAILED_JOB,
        "partial_manifest_sha256": FAILED_MANIFEST_SHA256,
        "complete_units": 73,
        "target_y_access": 0,
        "target_scientific_metrics": 0,
        "authorization_consumed": True,
        "authorization_reusable": False,
        "partial_artifacts_reusable": False,
    }
    protocol["replacement_execution"] = {
        "fresh_external_content_addressed_root": True,
        "retrain_units": 243,
        "reuse_failed_checkpoints_optimizer_sidecars_or_instrumentation": False,
        "training_data_candidate_IDs_and_intervention_unchanged": True,
        "C84F": False,
        "C84S": False,
    }
    return protocol


def build_field_protocol(canary_v2_sha256: str) -> dict[str, Any]:
    prior_path = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V5.json"
    protocol = deepcopy(_read(prior_path.name))
    protocol.update({
        "schema_version": "c84_field_generation_protocol_v6",
        "parent_field_protocol_v5_sha256": sha256_file(prior_path),
        "parent_level1_canary_protocol_v2_sha256": canary_v2_sha256,
        "C84L1R1_numerical_repair_protocol_sha256": REPAIR_SHA256,
        "status": "LOCKED_PROTOCOL_ONLY_REPAIRED_C84L1C_REVIEW_REQUIRED_NO_C84F_LOCK_NOT_AUTHORIZED",
    })
    protocol["instrumentation_replay_tolerances"] = {
        "linear_z_classifier_logits_abs_tolerance": 2e-5,
        "softmax_repeat_logits_repeat_z_abs_tolerance": 1e-6,
    }
    protocol["canary_reuse"].update({
        "C84L1C_failed_job_895928_partial_artifacts_reusable": False,
        "C84L1C_failed_job_895928_complete_units_reusable": 0,
        "C84L1C_replacement_must_retrain_all_units": 243,
        "C84L1C_V2_complete_manifest_required": True,
    })
    return protocol


def _write_protocol(stem: str, payload: dict[str, Any]) -> str:
    path = REPORT_DIR / f"{stem}.json"
    path.write_bytes(canonical_bytes(payload) + b"\n")
    digest = sha256_file(path)
    (REPORT_DIR / f"{stem}.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def generate() -> dict[str, str]:
    canary_sha = _write_protocol("C84_LEVEL1_CANARY_PROTOCOL_V2", build_canary_protocol())
    field_sha = _write_protocol("C84_FIELD_GENERATION_PROTOCOL_V6", build_field_protocol(canary_sha))
    return {
        "canary_protocol_v2_sha256": canary_sha,
        "field_protocol_v6_sha256": field_sha,
        "repair_protocol_sha256": REPAIR_SHA256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate repaired C84L1 protocol family")
    parser.parse_args(argv)
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
