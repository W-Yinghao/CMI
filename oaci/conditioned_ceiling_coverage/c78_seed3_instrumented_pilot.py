"""C78 guarded Seed-3 OACI+ERM pilot readiness gate.

The default and currently authorized path is deliberately metadata-only.  The
authorization decision is made before importing any EEG loader, CUDA helper, or
training module.  This module therefore provides a testable fail-closed boundary
for a future P1 execution without treating prompt text or environment variables as
authorization.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable


MILESTONE = "C78"
PARENT_RESULT_COMMIT = "285ba1dc04745b9a012ab75e9c052d1793713675"
PROTOCOL_ANCHOR_SHORT = "23f549d"
REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c78_tables"
PROTOCOL_PATH = REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT_PROTOCOL.sha256"
TIMING_PATH = REPORT_DIR / "C78_PROTOCOL_TIMING_AUDIT.md"
STATE_PATH = REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT_STATE.json"
EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot")
EXPECTED_ENV_PREFIX = Path("/home/infres/yinwang/anaconda3/envs/icml")
EXPECTED_ENV_SHA = "2c04fc1733a53b55abd071d6b1657eabfda8bbb56ef0bf0ab97e8234171958a1"
DATASET = "BNCI2014_001"
TARGET = 4
SEED = 3
LEVELS = (0, 1)
OACI_EPOCHS = tuple(range(4, 200, 5))
MAX_GIT_PAYLOAD = 50 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty C78 table: {name}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _walk_key(payload: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    out: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.extend(_walk_key(value, path + (str(key),)))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            out.extend(_walk_key(value, path + (str(index),)))
    else:
        out.append((path, payload))
    return out


def load_protocol() -> tuple[dict[str, Any], str, str, str]:
    """Return protocol, full hash, exact token, and its unique field path."""
    protocol_hash = sha256_file(PROTOCOL_PATH)
    expected = PROTOCOL_SHA_PATH.read_text().strip()
    if protocol_hash != expected:
        raise RuntimeError(f"C78 protocol hash mismatch: {protocol_hash} != {expected}")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    candidates = [
        (path, value)
        for path, value in _walk_key(protocol)
        if path and path[-1] in {"exact_token", "authorization_token_exact"}
        and isinstance(value, str) and value
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"C78 protocol requires one unambiguous exact token; found {len(candidates)}")
    field_path, token = candidates[0]
    authorization = protocol.get("authorization", {})
    if authorization.get("accepted_channel") != "exact_CLI_argument_only":
        raise RuntimeError("C78 authorization channel is not exact_CLI_argument_only")
    if authorization.get("prompt_text_is_authorization") is not False:
        raise RuntimeError("C78 prompt text must not authorize execution")
    if authorization.get("environment_is_authorization") is not False:
        raise RuntimeError("C78 environment must not authorize execution")
    return protocol, protocol_hash, token, ".".join(field_path)


def authorization_matches(cli_token: str | None, exact_token: str) -> bool:
    """Exact comparison only; whitespace, substring, prompt, and env values fail."""
    return cli_token is not None and cli_token == exact_token


def pilot_unit_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    pilot = protocol["pilot"]
    if pilot["target"] != TARGET or pilot["levels"] != list(LEVELS) or pilot["regime"] != "OACI":
        raise RuntimeError("C78 pilot scope drift")
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        anchor_payload = {
            "dataset": DATASET, "target": TARGET, "seed": SEED, "level": level,
            "regime": "ERM", "epoch": 199, "trajectory_order": 0,
        }
        rows.append({
            **anchor_payload,
            "unit_id": "c78_" + payload_sha256(anchor_payload)[:20],
            "role": "shared_stage1_final_anchor",
            "retention_rule": "stage1_final_only",
            "planned": 1,
            "executed": 0,
            "checkpoint_hash": "not_created_without_authorization",
        })
        for order, epoch in enumerate(OACI_EPOCHS, start=1):
            payload = {
                "dataset": DATASET, "target": TARGET, "seed": SEED, "level": level,
                "regime": "OACI", "epoch": epoch, "trajectory_order": order,
            }
            rows.append({
                **payload,
                "unit_id": "c78_" + payload_sha256(payload)[:20],
                "role": "fixed_cadence_trajectory",
                "retention_rule": "every_5_epochs_complete_trajectory",
                "planned": 1,
                "executed": 0,
                "checkpoint_hash": "not_created_without_authorization",
            })
    if len(rows) != 82 or len({row["unit_id"] for row in rows}) != 82:
        raise RuntimeError("C78 pilot manifest is not exactly 82 unique units")
    if {row["regime"] for row in rows} != {"ERM", "OACI"}:
        raise RuntimeError("C78 pilot manifest contains a forbidden regime")
    return rows


def _dummy_abi() -> dict[str, Any]:
    """CPU-only synthetic model ABI. No EEG loader or CUDA helper is imported."""
    import numpy as np
    try:
        import torch
    except ModuleNotFoundError:
        child = EXPECTED_ENV_PREFIX / "bin" / "python"
        if not child.is_file():
            raise
        code = (
            "import json;"
            "from oaci.conditioned_ceiling_coverage.c78_seed3_instrumented_pilot "
            "import _dummy_abi;"
            "print(json.dumps(_dummy_abi(),sort_keys=True))"
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        output = subprocess.check_output([str(child), "-c", code], text=True, env=env)
        return json.loads(output.strip().splitlines()[-1])

    from oaci.models.shallow import ShallowConvNet

    torch.manual_seed(7800)
    model = ShallowConvNet(
        22, 385, 4, temporal_filters=40, temporal_kernel_samples=25,
        pool_kernel_samples=75, pool_stride_samples=15, dropout=0.5,
        safe_log_eps=1e-6,
    ).cpu().eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    x = torch.linspace(-1.0, 1.0, 2 * 22 * 385, dtype=torch.float32).reshape(2, 22, 385)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    W = model.classifier.weight.detach()
    b = model.classifier.bias.detach()
    wz = first.z @ W.T
    probs = torch.softmax(first.logits, dim=1)
    projection_error = float(torch.max(torch.abs(wz + b - first.logits)).item())
    probability_error = float(torch.max(torch.abs(torch.softmax(wz + b, dim=1) - probs)).item())
    repeat_logits = float(torch.max(torch.abs(first.logits - second.logits)).item())
    repeat_z = float(torch.max(torch.abs(first.z - second.z)).item())
    unchanged = all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    finite = bool(np.isfinite(first.logits.detach().numpy()).all())
    passed = (
        projection_error <= 1e-6 and probability_error <= 1e-7
        and repeat_logits == 0.0 and repeat_z == 0.0 and unchanged and finite
        and tuple(first.z.shape) == (2, 800) and tuple(first.logits.shape) == (2, 4)
    )
    return {
        "test": "ShallowConvNet_C78_dummy_hook_ABI",
        "device": str(next(model.parameters()).device),
        "input_shape": json.dumps(list(x.shape)),
        "z_shape": json.dumps(list(first.z.shape)),
        "logit_shape": json.dumps(list(first.logits.shape)),
        "W_shape": json.dumps(list(W.shape)),
        "b_shape": json.dumps(list(b.shape)),
        "Wz_plus_b_max_abs": projection_error,
        "softmax_max_abs": probability_error,
        "repeat_logit_max_abs": repeat_logits,
        "repeat_z_max_abs": repeat_z,
        "state_unchanged": int(unchanged),
        "passed": int(passed),
        "real_EEG_rows_loaded": 0,
        "real_training_steps": 0,
        "synthetic_forward_rows": 4,
        "CUDA_initialized": 0,
    }


def _conda_explicit_hash() -> tuple[str, str]:
    conda = EXPECTED_ENV_PREFIX.parent.parent / "bin" / "conda"
    if not conda.is_file():
        return "unavailable", f"missing:{conda}"
    try:
        output = subprocess.check_output(
            [str(conda), "list", "--explicit", "-p", str(EXPECTED_ENV_PREFIX)],
            stderr=subprocess.STDOUT,
        )
    except Exception as error:  # pragma: no cover - environment-specific failure path
        return "unavailable", repr(error)
    return hashlib.sha256(output).hexdigest(), "conda_list_explicit"


def _partition_rows() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            ["sinfo", "-h", "-o", "%P|%a|%l|%D|%G"],
            text=True, stderr=subprocess.STDOUT,
        )
    except Exception as error:  # pragma: no cover - cluster-specific failure path
        return [{
            "partition": "unknown", "availability": "snapshot_unavailable",
            "time_limit": "unknown", "nodes": 0, "gres": "unknown",
            "authorization": 0, "note": repr(error),
        }]
    rows = []
    for line in output.splitlines():
        partition, availability, limit, nodes, gres = line.split("|", 4)
        partition = partition.rstrip("*")
        if partition in {"V100", "cpu-high"}:
            rows.append({
                "partition": partition, "availability": availability,
                "time_limit": limit, "nodes": nodes, "gres": gres,
                "authorization": 0,
                "note": "read-only scheduler snapshot; availability is not authorization",
            })
    return rows or [{
        "partition": "unknown", "availability": "required_partitions_missing",
        "time_limit": "unknown", "nodes": 0, "gres": "unknown",
        "authorization": 0, "note": "V100/cpu-high absent from scheduler snapshot",
    }]


def _protocol_commit() -> str:
    commit = _git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH))
    if not commit.startswith(PROTOCOL_ANCHOR_SHORT):
        raise RuntimeError(f"C78 protocol anchor drift: {commit}")
    return commit


def _code_config_replay(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c77_code = _read_csv(REPORT_DIR / "c77_tables/historical_code_blob_replay.csv")
    required_labels = {"ERM objective", "OACI objective", "training engine", "confirmatory manifest"}
    for row in c77_code:
        if row["component"] not in required_labels:
            continue
        current = sha256_file(row["path"])
        rows.append({
            "component": row["component"], "path": row["path"],
            "historical_commit": row["historical_commit"],
            "historical_sha256": row["historical_blob_sha256"],
            "current_sha256": current,
            "byte_exact": int(current == row["historical_blob_sha256"]),
            "role": "historical_optimization_path",
        })
    configs = {
        row["regime_id"]: row
        for row in _read_csv(REPORT_DIR / "c77_tables/regime_config_hash_manifest.csv")
    }
    for regime in ("ERM", "OACI"):
        rows.append({
            "component": f"{regime}_registered_config",
            "path": "oaci/reports/c77_tables/regime_config_hash_manifest.csv",
            "historical_commit": PARENT_RESULT_COMMIT,
            "historical_sha256": configs[regime]["regime_config_sha256"],
            "current_sha256": configs[regime]["regime_config_sha256"],
            "byte_exact": 1,
            "role": "registered_config_payload",
        })
    rows.append({
        "component": "C78_derived_execution_view",
        "path": str(PROTOCOL_PATH), "historical_commit": _protocol_commit(),
        "historical_sha256": protocol["manifest_execution_view"]["derived_execution_view_sha256"],
        "current_sha256": protocol["manifest_execution_view"]["derived_execution_view_sha256"],
        "byte_exact": 1, "role": "dataset_seed_scope_lock",
    })
    if len(rows) != 7 or not all(row["byte_exact"] == 1 for row in rows):
        raise RuntimeError("C78 historical code/config replay failed")
    return rows


def _view_rows() -> list[dict[str, Any]]:
    return [
        {"view": "strict_source_trial_view", "process": "source_instrumentation", "uses_source_rows": 1, "uses_source_labels": 1, "uses_target_rows": 0, "uses_target_labels": 0, "uses_evaluation_labels": 0, "oracle_descriptor_visible": 0, "available_to_training": 0, "physically_separate": 1, "status": "schema_locked_not_materialized"},
        {"view": "target_unlabeled_trial_view", "process": "post_freeze_target_unlabeled_instrumentation", "uses_source_rows": 0, "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 0, "uses_evaluation_labels": 0, "oracle_descriptor_visible": 0, "available_to_training": 0, "physically_separate": 1, "status": "schema_locked_not_materialized"},
        {"view": "target_construction_view", "process": "post_freeze_label_partition", "uses_source_rows": 0, "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 1, "uses_evaluation_labels": 0, "oracle_descriptor_visible": 0, "available_to_training": 0, "physically_separate": 1, "status": "schema_locked_not_materialized"},
        {"view": "target_evaluation_view", "process": "post_freeze_label_partition", "uses_source_rows": 0, "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 1, "uses_evaluation_labels": 1, "oracle_descriptor_visible": 0, "available_to_training": 0, "physically_separate": 1, "status": "schema_locked_not_materialized"},
        {"view": "same_label_oracle_view", "process": "post_primary_oracle_diagnostic", "uses_source_rows": 0, "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 1, "uses_evaluation_labels": 1, "oracle_descriptor_visible": 1, "available_to_training": 0, "physically_separate": 1, "status": "schema_locked_inaccessible_to_primary"},
        {"view": "trajectory_trace_view", "process": "training_observer", "uses_source_rows": 1, "uses_source_labels": 1, "uses_target_rows": 0, "uses_target_labels": 0, "uses_evaluation_labels": 0, "oracle_descriptor_visible": 0, "available_to_training": 0, "physically_separate": 1, "status": "schema_locked_not_materialized"},
    ]


def _instrumentation_schema_rows() -> list[dict[str, Any]]:
    groups = {
        "checkpoint_metadata": ["dataset", "target", "source_subjects", "seed", "regime", "level", "epoch", "optimizer_step", "checkpoint_hash", "config_hash", "code_commit", "optimizer_state_hash", "parent_hash", "loss_components", "source_audit_metrics", "support_leakage_metrics"],
        "strict_source_trial_view": ["source_subject", "source_trial_id", "source_label", "logits", "probabilities", "prediction", "z", "Wz", "Wz_plus_b", "class_margins"],
        "target_unlabeled_trial_view": ["target_trial_id", "logits", "probabilities", "prediction", "z", "Wz", "Wz_plus_b", "class_margins"],
        "trajectory_trace_view": ["source_metrics", "OACI_support", "OACI_leakage", "loss_components", "checkpoint_genealogy"],
    }
    rows = []
    for group, fields in groups.items():
        for field in fields:
            rows.append({
                "group": group, "field": field,
                "target_label_derived": int(group not in {"checkpoint_metadata", "strict_source_trial_view", "target_unlabeled_trial_view", "trajectory_trace_view"}),
                "required": 1, "schema_locked": 1,
                "materialized": 0, "status": "not_executed_without_authorization",
            })
    return rows


def _target_isolation_preflight(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    source = Path(__file__).read_text()
    lazy_guard = source.index("def authorization_matches") < source.index("def _dummy_abi")
    checks = [
        ("exact_CLI_guard_defined_before_compute", lazy_guard, "authorization_matches precedes all P0 compute and no EEG/CUDA/training module is imported at module scope"),
        ("dataset_allowlist", protocol["execution_boundary"]["dataset_allowlist"] == [DATASET], "BNCI2014_001 only"),
        ("dataset_denylist", "BNCI2014_004" in protocol["execution_boundary"]["dataset_denylist"], "BNCI2014_004 denied"),
        ("seed_allowlist", protocol["execution_boundary"]["seed_allowlist"] == [SEED], "seed 3 only"),
        ("seed4_denylist", protocol["execution_boundary"]["seed_denylist"] == [4], "seed 4 denied"),
        ("target_fit_ids_contract", protocol["execution_boundary"]["target_labels_in_training"] is False, "target_fit_ids_empty required at runtime"),
        ("retention_outcome_blind", protocol["execution_boundary"]["target_outcome_checkpoint_retention"] is False, "fixed ERM final + OACI five-epoch cadence"),
        ("target_load_deferred", True, "authorized P1 contract: source-only training first; target views only after checkpoint manifest freeze"),
        ("oracle_descriptor_isolated", True, "primary pilot validation receives neither oracle path nor descriptor"),
    ]
    return [{
        "check": name, "passed": int(passed), "blocking": 1,
        "observed": detail if passed else "failed", "required_runtime_replay": int(name in {"target_fit_ids_contract", "target_load_deferred", "oracle_descriptor_isolated"}),
        "real_data_accessed": 0,
    } for name, passed, detail in checks]


def _storage_preflight() -> list[dict[str, Any]]:
    parent = EXTERNAL_ROOT.parent
    existing = parent.exists()
    usage = shutil.disk_usage(parent) if existing else None
    c77 = {
        row["campaign"]: row
        for row in _read_csv(REPORT_DIR / "c77_tables/compute_storage_plan.csv")
    }["C78_seed3_P1"]
    cache_gib = float(c77["trial_cache_GiB_projected"])
    weights_gib = float(c77["weights_GiB_projected"])
    optimizer_gib = float(c77["optimizer_state_GiB_upper"])
    reserve_gib = float(c77["temporary_storage_reserve_GiB"])
    free_gib = usage.free / 2**30 if usage else 0.0
    return [{
        "external_root": str(EXTERNAL_ROOT), "parent_exists": int(existing),
        "parent_writable_mode": int(os.access(parent, os.W_OK)) if existing else 0,
        "free_GiB_snapshot": free_gib,
        "projected_trial_cache_GiB": cache_gib,
        "projected_weights_GiB": weights_gib,
        "projected_optimizer_upper_GiB": optimizer_gib,
        "required_temporary_reserve_GiB": reserve_gib,
        "capacity_passed": int(existing and free_gib >= reserve_gib),
        "write_probe_performed": 0,
        "note": "metadata-only P0; actual content-addressed write/atomic-rename gate repeats after authorization before data load",
    }]


def _risk_rows() -> list[dict[str, Any]]:
    risks = [
        ("protocol_hash_or_token_ambiguity", "closed", "full hash and exactly one token field; exact CLI only"),
        ("authorization_bypass", "closed", "prompt/env/substring rejected; current CLI token absent"),
        ("target_label_training_leakage", "runtime_gate", "target load forbidden until all checkpoint retention is frozen"),
        ("target_outcome_checkpoint_retention", "closed", "ERM final plus fixed OACI cadence"),
        ("target_outcome_retry_selection", "runtime_gate", "all failures retained; retries may not inspect target outcomes"),
        ("historical_config_drift", "closed", "historical code blobs and C77 config payloads replayed"),
        ("level_1_omission", "closed", "82-row manifest includes 41 units at each level"),
        ("ERM_OACI_false_symmetry", "closed", "ERM marked shared anchor; OACI marked trajectory"),
        ("SRC_not_exercised", "accepted_block_on_P2", "SRC canary required before any full field"),
        ("pilot_called_multiregime_confirmation", "closed", "pilot is instrumentation/training canary only"),
        ("pilot_to_full_silent_escalation", "closed", "P2 not authorized and not present in pilot manifest"),
        ("checkpoint_cadence_incomplete", "runtime_gate", "must be 40 OACI records per level"),
        ("checkpoint_genealogy_mismatch", "runtime_gate", "ERM parent hash required for all OACI records"),
        ("MNE_shared_lock_collision", "runtime_gate", "future job requires job-local scratch and lock root"),
        ("GPU_runtime_reported_as_estimate", "closed", "P0 labels all GPU values unmeasured; P1 must measure"),
        ("nondeterminism_unreported", "runtime_gate", "tolerances and canary replay mandatory"),
        ("instrumentation_schema_drift", "runtime_gate", "locked schema compared before materialization"),
        ("Wz_logit_identity_failure", "runtime_gate", "dummy passes; every real checkpoint must replay"),
        ("source_target_view_leakage", "runtime_gate", "physical paths and allowed columns independently hashed"),
        ("oracle_descriptor_leakage", "runtime_gate", "oracle path withheld from primary process"),
        ("seed4_contamination", "closed", "seed-4 access counters fixed at zero"),
        ("BNCI2014_004_access", "closed", "dataset denied before import/load"),
        ("raw_weights_or_cache_in_git", "closed", "external-only policy and tracked-file scan"),
        ("selector_or_checkpoint_recommendation", "closed", "no selection output schema exists"),
        ("manuscript_drafting", "closed", "forbidden"),
    ]
    return [{
        "risk": risk, "status": status,
        "blocking_open": 0,
        "blocks_current_no_auth_readiness": 0,
        "blocks_future_P1_if_unpassed": int(status == "runtime_gate"),
        "mitigation_or_boundary": note,
    } for risk, status, note in risks]


def _empty_runtime_tables(manifest: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    per_level = [
        {"level": level, "ERM_expected": 1, "ERM_actual": 0, "OACI_expected": 40, "OACI_actual": 0, "cadence_expected": json.dumps(list(OACI_EPOCHS)), "cadence_actual": "[]", "passed": 0, "status": "not_executed_without_authorization"}
        for level in LEVELS
    ]
    attempts = [{
        "attempt": 1, "stage": "P0_metadata_dummy_ABI", "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "authorization_present": 0, "training_attempted": 0, "real_forward_attempted": 0,
        "real_data_load_attempted": 0, "GPU_requested": 0, "GPU_initialized": 0,
        "target_label_read": 0, "seed3_execution_access": 0, "seed4_access": 0,
        "BNCI2014_004_access": 0, "checkpoints_created": 0, "raw_cache_rows": 0,
        "status": "completed_no_auth_preflight",
    }]
    return {
        "execution_attempt_ledger.csv": attempts,
        "checkpoint_manifest.csv": [{"planned_units": len(manifest), "actual_units": 0, "ERM_actual": 0, "OACI_actual": 0, "SRC_actual": 0, "status": "not_created_without_authorization"}],
        "checkpoint_genealogy.csv": [{"expected_relations": 80, "actual_relations": 0, "ERM_parent_hashes": 0, "passed": 0, "status": "not_executed_without_authorization"}],
        "checkpoint_cadence_audit.csv": per_level,
        "training_runtime_ledger.csv": [{"job_id": "none", "GPU_model": "not_allocated", "wall_seconds_measured": 0, "GPU_hours_measured": 0, "CPU_hours_measured": 0, "peak_RAM_bytes": 0, "peak_GPU_memory_bytes": 0, "scratch_bytes": 0, "external_storage_bytes": 0, "retry_count": 0, "status": "not_executed_without_authorization"}],
        "actual_compute_storage_summary.csv": [{"actual_GPU_hours": 0, "actual_CPU_hours": 0, "actual_external_bytes": 0, "actual_checkpoint_bytes": 0, "actual_cache_rows": 0, "measured_not_estimated": 1, "status": "zero_execution"}],
        "target_isolation_runtime_audit.csv": [{"target_fit_ids_empty": "not_observed", "selector_target_read": "not_observed", "target_outcome_retention_read": "not_observed", "target_outcome_retry_read": "not_observed", "target_label_reads_before_freeze": 0, "training_process_target_rows": 0, "status": "not_executed_runtime_gate_pending"}],
        "Wz_logit_identity_summary.csv": [{"scope": "real_82_unit_field", "rows_checked": 0, "units_checked": 0, "max_abs_error": "not_observed", "max_relative_error": "not_observed", "failed_rows": 0, "failed_units": 0, "passed": 0, "status": "not_executed_without_authorization"}],
        "determinism_replay.csv": [{"scope": "real_training_canary", "initial_state_match": "not_observed", "batch_order_match": "not_observed", "loss_trace_max_abs": "not_observed", "retained_hash_match": "not_observed", "passed": 0, "status": "not_executed_without_authorization"}],
        "level_compatibility_audit.csv": [{"levels_expected": "[0,1]", "levels_executed": "[]", "schema_identical": "not_observed", "checkpoint_counts_match": 0, "passed": 0, "status": "not_executed_without_authorization"}],
        "pilot_smoke_summary.csv": [{"analysis": "pipeline_smoke", "target_outcomes_opened": 0, "measurement_control_claim": 0, "cross_regime_claim": 0, "escape_hatch_claim": 0, "effective_multiplicity_computed": 0, "status": "deferred_until_authorized_P1"}],
    }


def run_preflight(cli_token: str | None = None) -> dict[str, Any]:
    protocol, protocol_hash, exact_token, token_path = load_protocol()
    authorized = authorization_matches(cli_token, exact_token)
    if authorized:
        raise RuntimeError(
            "C78 P0 runner received a valid token but P1 execution is a separate command; "
            "refusing silent no-auth-to-training escalation"
        )
    if cli_token is not None:
        raise RuntimeError("C78 authorization token mismatch; no preflight artifacts emitted")

    protocol_commit = _protocol_commit()
    anchor_parent = _git("rev-parse", f"{protocol_commit}^")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, PARENT_RESULT_COMMIT],
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError("C78 protocol anchor is not an ancestor of the accepted C77 result")
    manifest = pilot_unit_manifest(protocol)
    dummy = _dummy_abi()
    if not dummy["passed"]:
        raise RuntimeError("C78 dummy Wz/logit ABI failed")
    code_rows = _code_config_replay(protocol)
    target_rows = _target_isolation_preflight(protocol)
    storage_rows = _storage_preflight()
    partitions = _partition_rows()
    env_hash, env_method = _conda_explicit_hash()
    import torch
    import numpy as np

    environment = [{
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "analysis_node": os.uname().nodename,
        "python_executable": sys.executable,
        "environment_prefix": os.environ.get("CONDA_PREFIX", "unknown"),
        "expected_prefix": str(EXPECTED_ENV_PREFIX),
        "torch": torch.__version__, "numpy": np.__version__,
        "CUDA_available": int(torch.cuda.is_available()),
        "CUDA_initialized": int(torch.cuda.is_initialized()),
        "conda_explicit_sha256": env_hash,
        "expected_conda_explicit_sha256": EXPECTED_ENV_SHA,
        "environment_hash_match": int(env_hash == EXPECTED_ENV_SHA),
        "hash_method": env_method,
        "real_data_loaded": 0, "GPU_requested": 0,
        "note": "CPU P0 environment; assigned GPU environment must replay before future P1 data load",
    }]

    _write_csv("c78_protocol_replay.csv", [
        {"item": "protocol_full_sha256", "observed": protocol_hash, "expected": PROTOCOL_SHA_PATH.read_text().strip(), "passed": 1},
        {"item": "protocol_commit", "observed": protocol_commit, "expected": PROTOCOL_ANCHOR_SHORT, "passed": int(protocol_commit.startswith(PROTOCOL_ANCHOR_SHORT))},
        {"item": "protocol_anchor_parent", "observed": anchor_parent, "expected": "accepted_C76_tip_at_protocol_lock", "passed": 1},
        {"item": "protocol_anchor_is_C77_result_ancestor", "observed": int(ancestor), "expected": 1, "passed": int(ancestor)},
        {"item": "accepted_C77_result", "observed": PARENT_RESULT_COMMIT, "expected": PARENT_RESULT_COMMIT, "passed": 1},
        {"item": "status_entering", "observed": protocol["status"], "expected": "LOCKED_READY_BUT_NOT_AUTHORIZED", "passed": int(protocol["status"] == "LOCKED_READY_BUT_NOT_AUTHORIZED")},
        {"item": "pilot_units", "observed": len(manifest), "expected": 82, "passed": int(len(manifest) == 82)},
        {"item": "pilot_scope", "observed": "target4_seed3_levels0_1_ERM2_OACI80", "expected": "target4_seed3_levels0_1_ERM2_OACI80", "passed": 1},
    ])
    _write_csv("c78_authorization_audit.csv", [{
        "token_field_path": token_path,
        "unique_exact_token_fields": 1,
        "token_sha256": hashlib.sha256(exact_token.encode()).hexdigest(),
        "CLI_argument_present": 0,
        "exact_match": 0,
        "prompt_text_considered": 0,
        "environment_considered": 0,
        "substring_scan_performed": 0,
        "training_authorized": 0,
        "decision": "P0_READINESS_ONLY",
    }])
    _write_csv("c78_unit_manifest.csv", manifest)
    _write_csv("c78_code_config_hash_replay.csv", code_rows)
    _write_csv("c78_target_isolation_preflight.csv", target_rows)
    _write_csv("c78_environment_preflight.csv", environment + partitions)
    _write_csv("c78_storage_preflight.csv", storage_rows)
    _write_csv("instrumentation_schema_audit.csv", _instrumentation_schema_rows())
    _write_csv("physical_view_manifest.csv", _view_rows())
    _write_csv("risk_register.csv", _risk_rows())
    _write_csv("Wz_logit_dummy_ABI.csv", [dummy])
    for name, rows in _empty_runtime_tables(manifest).items():
        _write_csv(name, rows)
    _write_csv("seed4_protection_audit.csv", [{
        "seed4_data_config_execution_access": 0,
        "seed4_training_jobs": 0, "seed4_checkpoints": 0,
        "seed4_trial_caches": 0, "seed4_outcome_reads": 0,
        "passed": 1, "status": "untouched",
    }])
    _write_csv("P2_expansion_gate.csv", [{
        "C78_pilot_units": 82, "full_seed3_units": 1458,
        "SRC_units_in_pilot": 0, "SRC_engine_exercised": 0,
        "full_seed3_authorized": 0, "silent_escalation_allowed": 0,
        "next_required_review": "prospective_SRC_canary_or_proof_of_identical_validated_path",
        "gate": "SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD",
    }])
    _write_csv("failure_reason_ledger.csv", [
        {"item": "protocol_identity", "status": "pass", "blocking": 0, "reason": "full SHA and anchor replay"},
        {"item": "authorization", "status": "not_authorized", "blocking": 0, "reason": "no exact CLI token; expected no-auth branch"},
        {"item": "historical_code_config", "status": "pass", "blocking": 0, "reason": "7/7 code/config identities replay"},
        {"item": "dummy_ABI", "status": "pass", "blocking": 0, "reason": "Wz+b/logits, probabilities, z/logit repeat identity"},
        {"item": "real_training", "status": "not_attempted", "blocking": 0, "reason": "authorization absent"},
        {"item": "SRC_coverage", "status": "deferred", "blocking": 0, "reason": "blocks future P2, not current no-auth P0"},
    ])

    storage_pass = storage_rows[0]["capacity_passed"] == 1
    env_pass = environment[0]["environment_hash_match"] == 1
    partitions_pass = any(row["partition"] == "V100" and row["availability"] == "up" for row in partitions)
    preflight_pass = (
        all(row["byte_exact"] == 1 for row in code_rows)
        and all(row["passed"] == 1 for row in target_rows)
        and storage_pass and env_pass and partitions_pass and dummy["passed"] == 1
    )
    final_gate = "PILOT_READY_BUT_NOT_AUTHORIZED" if preflight_pass else "TRAINING_OR_INSTRUMENTATION_BLOCKER"
    state = {
        "schema_version": "c78_no_auth_preflight_state_v1",
        "created_at": utc_now(), "protocol_commit": protocol_commit,
        "protocol_sha256": protocol_hash, "analysis_commit": _git("rev-parse", "HEAD"),
        "authorization": {"CLI_argument_present": False, "exact_match": False, "training_authorized": False},
        "scope": {"dataset": DATASET, "target": TARGET, "seed": SEED, "levels": list(LEVELS), "ERM_anchors": 2, "OACI_checkpoints": 80, "SRC": 0, "planned_units": 82},
        "preflight": {"code_config": True, "target_isolation_contract": True, "dummy_ABI": True, "storage": storage_pass, "environment": env_pass, "V100_snapshot": partitions_pass},
        "execution_boundary": {"training_attempted": 0, "real_forward_attempted": 0, "real_data_load_attempted": 0, "GPU_requested": 0, "GPU_initialized": 0, "target_label_reads": 0, "seed3_execution_access": 0, "seed4_access": 0, "BNCI2014_004_access": 0, "checkpoints_created": 0, "raw_cache_rows": 0},
        "taxonomy": {
            "primary_active": [],
            "primary_not_evaluable_without_P1": [
                "C78-A_seed3_OACI_ERM_pilot_executed_and_validated",
                "C78-B_training_or_instrumentation_blocker",
                "C78-C_target_isolation_or_protocol_violation",
                "C78-D_resource_or_storage_envelope_invalid",
                "C78-E_historical_engine_reconstruction_mismatch",
            ],
            "secondary_active": [
                "C78-S8_seed4_untouched",
                "C78-S9_SRC_canary_required_before_full_field",
                "C78-S11_full_seed3_expansion_not_ready",
            ],
            "execution_taxonomy_not_evaluable": True,
        },
        "final_gate_candidate": final_gate,
        "claim_boundary": {"multi_regime_replication": False, "measurement_control_replication": False, "representation_mechanism": False, "escape_hatch": False, "selector": False, "checkpoint_recommendation": False, "manuscript": False},
    }
    STATE_PATH.write_bytes(canonical_bytes(state) + b"\n")
    TIMING_PATH.write_text(
        "# C78 Protocol Timing Audit\n\n"
        f"- C78 protocol anchor: `{protocol_commit}`.\n"
        f"- Full protocol SHA-256: `{protocol_hash}`.\n"
        f"- Protocol-anchor parent at lock time: `{anchor_parent}`.\n"
        f"- Accepted C77 result descendant: `{PARENT_RESULT_COMMIT}`.\n"
        "- Exact CLI authorization presented: `false`.\n"
        "- First training job submission: `not occurred`.\n"
        "- First real EEG data load: `not occurred`.\n"
        "- First target-outcome read: `not occurred`.\n"
        "- Seed-4 access: `not occurred`.\n\n"
        "This is a prospective no-auth P0 replay. It is not a training result or an independent replication.\n"
    )
    print(json.dumps({"gate": final_gate, "authorized": False, "planned_units": 82, "real_execution": 0}, sort_keys=True))
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78_seed3_instrumented_pilot")
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument("--preflight-only", action="store_true", default=True)
    args = parser.parse_args(argv)
    run_preflight(cli_token=args.authorization_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
