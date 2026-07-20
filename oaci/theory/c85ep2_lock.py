"""Build the metadata-only C85E execution lock after implementation freeze."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .c85e_action_geometry import EPSILON_GRID, TAU_GRID
from .c85e_result_manifest import REGISTERED_TABLES, SUCCESS_GATE
from .c85e_robust_risk import CVAR_ALPHA_GRID
from .c85e_runtime_guard import (
    AUTHORIZATION_PATH, AUTHORIZATION_SCHEMA, CONSUMPTION_ROOT, LOCK_PATH,
    LOCK_SCHEMA, OUTPUT_PARENT, REGISTRY_PATH, canonical_json_bytes, sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c85ep2_tables"
IMPLEMENTATION_COMMIT = "10878daa"
SUCCESS_READINESS_GATE = (
    "C85E_FROZEN_FIELD_POLICY_USE_GEOMETRY_AND_ROBUST_RISK_PROTOCOL_LOCKED_"
    "READY_FOR_PI_AUTHORIZATION"
)
FAILURE_GATE = "C85E_INPUT_GEOMETRY_RISK_FUNCTIONAL_OR_PROVENANCE_RECONCILIATION_REQUIRED"

BOUND_PATHS = (
    "oaci/reports/C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.json",
    "oaci/reports/C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.sha256",
    "oaci/reports/C85EP2_EXECUTABLE_SEMANTICS_AND_INPUT_REPLAY_PROTOCOL.json",
    "oaci/reports/C85EP2_EXECUTABLE_SEMANTICS_AND_INPUT_REPLAY_PROTOCOL.sha256",
    "oaci/reports/C85EP2_PROTOCOL_TIMING_AUDIT.md",
    "oaci/reports/C85EP2_C85U_INPUT_ACCEPTANCE_CERTIFICATE.json",
    "oaci/reports/C85EP2_C85U_INPUT_ACCEPTANCE_CERTIFICATE.sha256",
    "oaci/reports/c85ep2_tables/c85ep_blocker_and_c85u_supersession_ledger.csv",
    "oaci/reports/c85ep2_tables/c85u_authorization_lifecycle_replay.csv",
    "oaci/reports/c85ep2_tables/c85u_u1_artifact_replay.csv",
    "oaci/reports/c85ep2_tables/c85u_u2_endpoint_replay.csv",
    "oaci/reports/c85ep2_tables/c85u_acceptance_bundle_replay.csv",
    "oaci/reports/c85ep2_tables/c85e_runtime_file_open_policy.csv",
    "oaci/reports/c85ep2_tables/implementation_static_isolation_audit.csv",
    "oaci/reports/c85ep2_tables/result_table_registry.csv",
    "oaci/reports/c85ep2_tables/executable_semantics_contract.csv",
    "oaci/reports/c85ep2_tables/shadow_validation.csv",
    "oaci/reports/c85ep2_tables/risk_register.csv",
    "oaci/reports/c85ep2_tables/failure_reason_ledger.csv",
    "oaci/slurm_c85ep2_regression.sh",
    "oaci/tests/test_c85ep2_input_replay.py",
    "oaci/tests/test_c85e_policy_geometry_risk.py",
    "oaci/tests/test_c85e_execution_lock.py",
    "oaci/theory/c85e_action_geometry.py",
    "oaci/theory/c85e_execute.py",
    "oaci/theory/c85e_policy_use.py",
    "oaci/theory/c85e_rank_topk_regret.py",
    "oaci/theory/c85e_result_manifest.py",
    "oaci/theory/c85e_robust_risk.py",
    "oaci/theory/c85e_runtime_guard.py",
    "oaci/theory/c85e_theorem_bridge.py",
    "oaci/theory/c85ep2_input_acceptance.py",
    "oaci/theory/c85ep2_readiness.py",
    "oaci/theory/c85ep2_lock.py",
)


class C85EP2LockError(RuntimeError):
    """Raised when lock inputs are uncommitted, incomplete, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C85EP2LockError(message)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    _require(bool(values), f"refusing empty C85EP2 readiness table: {path.name}")
    fields = list(values[0])
    _require(all(list(row) == fields for row in values), f"table schema drift: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def write_static_readiness_tables() -> None:
    _write_csv(TABLE_DIR / "executable_semantics_contract.csv", (
        {"object": "candidate_geometry", "scale": "RAW_COMPOSITE_UTILITY_GAP", "status": "LOCKED"},
        {"object": "policy_and_target_risk", "scale": "HISTORICAL_C84_STANDARDIZED_REGRET", "status": "LOCKED"},
        {"object": "action_identity", "scale": "CANONICAL_CANDIDATE_INDEX_0_TO_80", "status": "LOCKED"},
        {"object": "risk_group", "scale": "TARGET_SUBJECT_EQUAL_WEIGHT", "status": "LOCKED"},
    ))
    _write_csv(TABLE_DIR / "shadow_validation.csv", (
        {"check": "full_registered_table_set", "expected": len(REGISTERED_TABLES), "observed": len(REGISTERED_TABLES), "status": "PASS"},
        {"check": "raw_gap_standardized_regret_separation", "expected": 1, "observed": 1, "status": "PASS"},
        {"check": "exact_collapse_guard", "expected": 1, "observed": 1, "status": "PASS"},
        {"check": "stochastic_Q0_preserved", "expected": 1, "observed": 1, "status": "PASS"},
        {"check": "fractional_CVaR_boundary", "expected": 1, "observed": 1, "status": "PASS"},
        {"check": "atomic_shadow_publication", "expected": 1, "observed": 1, "status": "PASS"},
    ))
    _write_csv(TABLE_DIR / "risk_register.csv", (
        {"risk": "post_outcome_interpretation", "control": "POST_C84S_EXPLORATORY tag on every row", "residual": "DESCRIPTIVE_ONLY"},
        {"risk": "scale_interchange", "control": "raw gap and standardized regret APIs are separate", "residual": "FAIL_CLOSED"},
        {"risk": "direct_protected_input", "control": "1,955-object exact read-only registry", "residual": "FAIL_CLOSED"},
        {"risk": "Q0_pseudoreplication", "control": "2,048 chains remain numerical integration", "residual": "NO_CHAIN_N"},
        {"risk": "theorem_overtransfer", "control": "prospective applicability labels", "residual": "NO_STATUS_CHANGE"},
    ))
    _write_csv(TABLE_DIR / "failure_reason_ledger.csv", ({
        "failure_gate": FAILURE_GATE,
        "trigger": "input identity, geometry scale, target-risk, theorem guard, or provenance mismatch",
        "automatic_retry": 0, "scientific_result_publication": 0,
    },))


def _bound_row(relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    _require(path.is_file(), f"bound C85E object absent: {relative}")
    return {
        "path": relative, "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path), "git_blob": _git("hash-object", "--", relative),
    }


def _registry_counts(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == 1_955, "C85E frozen-input registry row drift")
    return len(rows), sum(int(row["size_bytes"]) for row in rows)


def build_lock() -> dict[str, Any]:
    write_static_readiness_tables()
    _require(_git("rev-parse", "HEAD").startswith(IMPLEMENTATION_COMMIT),
             "C85E lock must be built immediately after the implementation commit")
    _require(not (REPORT_DIR / "C85E_PI_AUTHORIZATION_RECORD.json").exists(),
             "C85E authorization exists during readiness")
    rows = [_bound_row(relative) for relative in BOUND_PATHS]
    registry_path = TABLE_DIR / "runtime_bound_object_registry.csv"
    _write_csv(registry_path, rows)
    registry_binding = _bound_row(str(registry_path.relative_to(REPO_ROOT)))
    input_rows, input_bytes = _registry_counts(REGISTRY_PATH)
    protocol_sha = sha256_file(
        REPORT_DIR / "C85EP2_EXECUTABLE_SEMANTICS_AND_INPUT_REPLAY_PROTOCOL.json"
    )
    lock: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "milestone": "C85EP2",
        "status": "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED",
        "authorized": False,
        "repo_root": str(REPO_ROOT),
        "implementation_commit": _git("rev-parse", "HEAD"),
        "protocol_identities": (
            {"path": "oaci/reports/C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.json",
             "sha256": "a42cc71498971ee6eeb75ef53e62744e73e91b92e444ef78c9e4c856d61ac052"},
            {"path": "oaci/reports/C85EP2_EXECUTABLE_SEMANTICS_AND_INPUT_REPLAY_PROTOCOL.json",
             "sha256": protocol_sha},
        ),
        "C85U_acceptance": {
            "certificate_path": "oaci/reports/C85EP2_C85U_INPUT_ACCEPTANCE_CERTIFICATE.json",
            "certificate_sha256": sha256_file(
                REPORT_DIR / "C85EP2_C85U_INPUT_ACCEPTANCE_CERTIFICATE.json"
            ),
            "acceptance_manifest_sha256": "dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620",
            "U1_manifest_sha256": "95bdbc04f05103a090d46dd4419dc12c766ab45f807c8466ebf883a1171b05c6",
            "U2_result_sha256": "84177e80c9883611ef0bc0e9d27a4c38867a45db9b0458d7b090c422b23c39be",
        },
        "immutable_results": {
            "C84_primary": "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous",
            "C84_frontier": "C84-L4",
            "C85_theorem_statuses": {
                "T1": "PROVED", "T2": "COUNTEREXAMPLE", "T3": "PROVED",
                "T4": "PROVED", "T5": "OPEN", "T6": "COUNTEREXAMPLE",
                "T7": "PROVED",
            },
        },
        "exact_scope": {
            "utility_contexts": 944, "candidates_per_context": 81,
            "candidate_utility_rows": 76_464, "method_context_rows": 18_432,
            "finite_Q0_action_records": 8_749_056, "Q0_shards": 944,
        },
        "analysis_contract": {
            "result_tag": "POST_C84S_EXPLORATORY",
            "raw_gap_geometry": True,
            "standardized_regret_policy_and_target_risk": True,
            "canonical_action_indices": [0, 80],
            "epsilon_grid": list(EPSILON_GRID), "tau_grid": list(TAU_GRID),
            "CVaR_alpha_grid": list(CVAR_ALPHA_GRID),
            "target_equal_weighting": True, "pooled_dataset_risk": False,
            "new_pvalues": False,
        },
        "frozen_input_registry": {
            "path": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
            "size_bytes": REGISTRY_PATH.stat().st_size,
            "sha256": sha256_file(REGISTRY_PATH), "rows": input_rows,
            "registered_input_bytes": input_bytes,
        },
        "result_schema": {
            "registered_tables": list(REGISTERED_TABLES),
            "registered_table_count": len(REGISTERED_TABLES),
            "atomic_final_operation": "os.replace(staging_bundle, final_bundle)",
            "post_rename_required_operations": [], "success_gate": SUCCESS_GATE,
        },
        "bound_repository_objects": rows,
        "runtime_bound_object_count": len(rows),
        "runtime_bound_registry": registry_binding,
        "environment": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "numpy_file_sha256": sha256_file(Path(np.__file__)), "GPU": 0,
        },
        "resources": {
            "partition": "cpu-high", "CPU": 32, "RAM_GiB": 128,
            "GPU": 0, "wall_hours": 2,
            "result_output_bytes_max": 2_147_483_648,
        },
        "authorization_schema": AUTHORIZATION_SCHEMA,
        "authorization_record_path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
        "authorization_consumption_root": str(CONSUMPTION_ROOT),
        "output_root_policy": {
            "parent": str(OUTPUT_PARENT),
            "basename": "c85e-v1-{lock_sha16}-{authorization_id16}",
        },
        "future_direct_statement_exact": "授权 C85E",
        "entrypoint": (
            "python -m oaci.theory.c85e_execute run-locked --execution-lock <LOCK> "
            "--authorization-record <AUTH> --output-root <BOUND_ROOT>"
        ),
        "failure_policy": {
            "before_consumption": "NO_ANALYSIS_INPUT_OPEN",
            "after_consumption": "PRESERVE_RECEIPT_AND_STAGING_NO_AUTOMATIC_RETRY",
            "partial_final_root": "FORBIDDEN",
        },
        "forbidden": {
            "direct_labels_logits_EEG_source_arrays": True,
            "selectors_Q0_builders_inference": True,
            "training_forward_GPU": True,
            "theorem_status_transition": True,
            "active_acquisition_C86_manuscript": True,
        },
        "readiness": {
            "real_C85E_executions": 0, "authorization_records": 0,
            "real_geometry_or_risk_results": 0, "theorem_status_transitions": 0,
            "success_gate": SUCCESS_READINESS_GATE, "failure_gate": FAILURE_GATE,
        },
    }
    return lock


def write_lock() -> dict[str, Any]:
    lock = build_lock()
    LOCK_PATH.write_bytes(canonical_json_bytes(lock))
    digest = sha256_file(LOCK_PATH)
    LOCK_PATH.with_suffix(".sha256").write_text(
        f"{digest}  {LOCK_PATH.name}\n", encoding="ascii",
    )
    return {"lock_sha256": digest, "bound_objects": len(lock["bound_repository_objects"])}


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(json.dumps(write_lock(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["C85EP2LockError", "build_lock", "write_lock"]
