"""Atomic C85T result publication and artifact identity replay."""
from __future__ import annotations

from contextlib import AbstractContextManager
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4

from .c85_decision_experiments import DecisionContractError


RESULT_SCHEMA = "c85t_decision_theory_proof_and_synthetic_result_v1"
MANIFEST_SCHEMA = "c85t_result_artifact_manifest_v1"
ATTEMPT_SCHEMA = "c85t_execution_attempt_ledger_v1"
SUCCESS_GATE = (
    "C85T_DECISION_THEORY_PROOF_AUDIT_AND_SYNTHETIC_VALIDATION_COMPLETE_"
    "C85E_PROTOCOL_REVIEW_REQUIRED"
)
READINESS_SUCCESS_GATE = (
    "C85T_PROOF_AND_SYNTHETIC_EXECUTION_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION"
)
READINESS_FAILURE_GATE = (
    "C85T_RNG_ESTIMAND_PROOF_STATUS_OR_EXECUTION_PROVENANCE_RECONCILIATION_REQUIRED"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_attempt_ledger(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise DecisionContractError("C85T attempt ledger already exists")
    payload = {"schema_version": ATTEMPT_SCHEMA, **value}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


class AtomicResultWriter(AbstractContextManager["AtomicResultWriter"]):
    """Write a complete result in staging and publish it with one rename."""

    def __init__(self, output_root: Path, *, failure_injection: str | None = None):
        self.output_root = output_root.resolve()
        self.staging_root = self.output_root.with_name(
            f".{self.output_root.name}.staging-{uuid4().hex}"
        )
        self.failure_injection = failure_injection
        self._published = False
        self.failed_root: Path | None = None

    def __enter__(self) -> "AtomicResultWriter":
        if self.output_root.exists():
            raise DecisionContractError("C85T output root must be absent")
        self.staging_root.mkdir(parents=True, exist_ok=False)
        return self

    def path(self, relative: str | Path) -> Path:
        path = (self.staging_root / relative).resolve()
        if self.staging_root not in path.parents and path != self.staging_root:
            raise DecisionContractError("result path escapes staging root")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, value: Any) -> Path:
        path = self.path(relative)
        path.write_bytes(canonical_json_bytes(value))
        return path

    def write_text(self, relative: str | Path, value: str) -> Path:
        path = self.path(relative)
        path.write_text(value)
        return path

    def _artifact_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.staging_root.rglob("*")):
            if not path.is_file() or path.name == "C85T_RESULT_ARTIFACT_MANIFEST.json":
                continue
            rows.append(
                {
                    "path": str(path.relative_to(self.staging_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return rows

    def publish(self, result: dict[str, Any]) -> Path:
        if self._published:
            raise DecisionContractError("C85T staging root was already published")
        if result.get("schema_version") != RESULT_SCHEMA:
            raise DecisionContractError("C85T result schema drifted")
        if result.get("final_gate") != SUCCESS_GATE:
            raise DecisionContractError("C85T success gate drifted")
        if self.failure_injection == "before_result":
            raise RuntimeError("C85T_SHADOW_FAILURE_BEFORE_RESULT")
        self.write_json("C85T_RESULT.json", result)
        rows = self._artifact_rows()
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "created_at_utc": utc_now(),
            "artifact_count": len(rows),
            "artifacts": rows,
        }
        if self.failure_injection == "before_manifest":
            raise RuntimeError("C85T_SHADOW_FAILURE_BEFORE_MANIFEST")
        self.write_json("C85T_RESULT_ARTIFACT_MANIFEST.json", manifest)
        replay_manifest(self.staging_root)
        if self.failure_injection == "before_publish":
            raise RuntimeError("C85T_SHADOW_FAILURE_BEFORE_PUBLISH")
        os.replace(self.staging_root, self.output_root)
        self._published = True
        return self.output_root

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if not self._published and self.staging_root.exists():
            if exc_type is None:
                shutil.rmtree(self.staging_root)
            else:
                self.failed_root = self.output_root.with_name(
                    f"{self.output_root.name}.failed-{uuid4().hex}"
                )
                os.replace(self.staging_root, self.failed_root)
        return False


def replay_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "C85T_RESULT_ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        raise DecisionContractError("C85T artifact manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DecisionContractError("C85T artifact manifest schema drifted")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise DecisionContractError("C85T artifact manifest count drifted")
    observed_paths: set[str] = set()
    for row in rows:
        relative = row["path"]
        if relative in observed_paths:
            raise DecisionContractError("duplicate C85T artifact path")
        observed_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise DecisionContractError(f"C85T artifact is absent: {relative}")
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise DecisionContractError(f"C85T artifact identity drift: {relative}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    }
    if actual != observed_paths:
        raise DecisionContractError("C85T artifact manifest coverage drifted")
    return manifest


def _git_blob(repo_root: Path, relative: str) -> str:
    return subprocess.run(
        ["git", "hash-object", "--", relative],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _readiness_bound_paths(repo_root: Path) -> list[str]:
    reports = repo_root / "oaci" / "reports"
    paths: set[Path] = {
        repo_root / "oaci" / "theory" / "__init__.py",
        repo_root / "oaci" / "theory" / "c85_decision_experiments.py",
        repo_root / "oaci" / "theory" / "c85_robust_risk.py",
        repo_root / "oaci" / "theory" / "c85_policy_collapse.py",
        repo_root / "oaci" / "theory" / "c85_lower_bound_contracts.py",
        repo_root / "oaci" / "theory" / "c85_synthetic_contract.py",
        repo_root / "oaci" / "theory" / "c85r_synthetic_semantic_repair.py",
        repo_root / "oaci" / "theory" / "c85t_rng.py",
        repo_root / "oaci" / "theory" / "c85t_exact_scenarios.py",
        repo_root / "oaci" / "theory" / "c85t_monte_carlo.py",
        repo_root / "oaci" / "theory" / "c85t_proofs.py",
        repo_root / "oaci" / "theory" / "c85t_result_manifest.py",
        repo_root / "oaci" / "theory" / "c85t_execute.py",
        repo_root / "oaci" / "multidataset" / "c84r_regression_suite.py",
        repo_root / "oaci" / "slurm_c85tl_regression.sh",
        repo_root / "oaci" / "tests" / "test_c85_decision_theory_protocol.py",
        repo_root / "oaci" / "tests" / "test_c85_synthetic_contract.py",
        repo_root / "oaci" / "tests" / "test_c85r_synthetic_semantic_repair.py",
        repo_root / "oaci" / "tests" / "test_c85r_protocol_lock.py",
        repo_root / "oaci" / "tests" / "test_c85t_shadow_execution.py",
        repo_root / "oaci" / "tests" / "test_c85tl_execution_lock.py",
    }
    fixed_reports = (
        "C84A_PM_REALIZED_POLICY_USE_ADDENDUM.md",
        "C84A_PM_REALIZED_POLICY_USE_ADDENDUM.json",
        "C84A_PM_REALIZED_POLICY_USE_ADDENDUM.sha256",
        "C85_TPAMI_DECISION_THEORY_PROTOCOL.json",
        "C85_TPAMI_DECISION_THEORY_PROTOCOL.sha256",
        "C85_PROTOCOL_TIMING_AUDIT.md",
        "C85P_PROTOCOL_READINESS.md",
        "C85P_FINAL_REPORT_RED_TEAM.md",
        "C85P_REGRESSION_VERIFICATION.md",
        "C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.json",
        "C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.sha256",
        "C85R_PROTOCOL_TIMING_AUDIT.md",
        "C85R_PROTOCOL_READINESS.md",
        "C85R_FINAL_REPORT_RED_TEAM.md",
        "C85R_REGRESSION_VERIFICATION.md",
        "C85R_OVERALL_REPORT.md",
        "C85R_OVERALL_REPORT.json",
        "C85R_OVERALL_REPORT.sha256",
        "C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json",
        "C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.sha256",
        "C85T_PROTOCOL_TIMING_AUDIT.md",
    )
    paths.update(reports / name for name in fixed_reports)
    for directory in ("c85p_tables", "c85r_tables", "c85tl_tables"):
        for path in (reports / directory).glob("*"):
            if path.is_file() and path.name != "runtime_bound_object_registry.csv":
                paths.add(path)
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise DecisionContractError(f"readiness bound file is absent: {missing[0]}")
    return sorted(str(path.relative_to(repo_root)) for path in paths)


def materialize_readiness_lock(
    *, repo_root: Path, implementation_commit: str, created_at_utc: str
) -> dict[str, Any]:
    """Materialize the prospective lock without executing a registered scenario."""

    reports = repo_root / "oaci" / "reports"
    registry_path = reports / "c85tl_tables" / "runtime_bound_object_registry.csv"
    lock_path = reports / "C85T_EXECUTION_LOCK.json"
    sidecar_path = reports / "C85T_EXECUTION_LOCK.sha256"
    if any(path.exists() for path in (registry_path, lock_path, sidecar_path)):
        raise DecisionContractError("C85T readiness registry/lock must be created once")
    forbidden = (
        reports / "C85T_PI_AUTHORIZATION_RECORD.json",
        reports / "C85T_RESULT.json",
        reports / "C85_SYNTHETIC_SCIENTIFIC_RESULT.json",
        reports / "c85t_proofs",
    )
    if any(path.exists() for path in forbidden):
        raise DecisionContractError("C85T execution or proof artifact already exists")

    rows: list[dict[str, Any]] = []
    for relative in _readiness_bound_paths(repo_root):
        path = repo_root / relative
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "git_blob": _git_blob(repo_root, relative),
            }
        )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "size_bytes", "sha256", "git_blob")
        )
        writer.writeheader()
        writer.writerows(rows)

    protocol_path = reports / "C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json"
    c85p_path = reports / "C85_TPAMI_DECISION_THEORY_PROTOCOL.json"
    c85r_path = reports / "C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.json"
    v2_path = reports / "c85r_tables" / "synthetic_generator_contract_v2.json"
    registry_identity = {
        "path": str(registry_path.relative_to(repo_root)),
        "size_bytes": registry_path.stat().st_size,
        "sha256": sha256_file(registry_path),
        "git_blob": _git_blob(repo_root, str(registry_path.relative_to(repo_root))),
    }
    lock = {
        "schema_version": "c85t_execution_lock_v1",
        "milestone": "C85TL",
        "created_at_utc": created_at_utc,
        "status": "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED",
        "authorized": False,
        "implementation_commit": implementation_commit,
        "execution_lock_commit_binding": "DISCOVER_FROM_GIT_PATH_AND_BIND_IN_FUTURE_AUTHORIZATION",
        "c85p_protocol_sha256": sha256_file(c85p_path),
        "c85r_repair_protocol_sha256": sha256_file(c85r_path),
        "v2_generator_sha256": sha256_file(v2_path),
        "operationalization_protocol_sha256": sha256_file(protocol_path),
        "runtime_bound_object_count": len(rows),
        "runtime_bound_registry": registry_identity,
        "bound_repository_objects": rows,
        "environment": {
            "prefix": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact",
            "python": "3.13.7",
            "numpy_runtime": "2.4.4",
            "numpy_metadata_first_match": "2.3.3",
            "numpy_dual_metadata_bound": True,
            "bit_generator": "PCG64DXSM",
            "GPU": 0,
        },
        "rng": {
            "namespace": "C85_SYNTHETIC_V1",
            "seed": "low64_SHA256_little_endian",
            "replicates": 4096,
            "canonical_action_order": True,
            "parallel_nondeterministic_reduction": False,
        },
        "scenario_execution_modes": {
            "exact_only": ["S0", "S1", "S2", "S3", "S4", "S5", "S8", "S10"],
            "exact_plus_4096_MC": ["S6", "S7", "S9"],
            "exact_authoritative_where_available": True,
        },
        "S9": {
            "draw_order": "51_L_then_46_H",
            "passive_prefix": [51, 13],
            "neyman_prefix": [18, 46],
            "all_action_estimators": True,
            "selection": "canonical_first_argmin",
            "top2": "stable_estimated_top2_contains_true_best",
            "analytic_variance_authoritative": True,
            "universal_active_superiority_claim": False,
        },
        "proofs": {
            "theorems": [f"T{i}" for i in range(1, 8)],
            "entering_status": "OPEN",
            "simulation_can_prove": False,
            "citation_alone_can_prove": False,
            "independent_PASS_required": True,
            "T5_may_remain_OPEN": True,
            "proof_files_created_at_readiness": 0,
            "status_transitions_at_readiness": 0,
        },
        "result": {
            "schema": RESULT_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "atomic_staging_and_rename": True,
            "failed_staging_preserved": True,
            "successful_C85T_gate": SUCCESS_GATE,
        },
        "resources": {
            "CPU": 1,
            "GPU": 0,
            "RAM_GiB_envelope": 8,
            "wall_minutes_envelope": 30,
            "external_storage_MiB_envelope": 64,
            "replicate_reduction": "serial_canonical_order",
            "runtime_scope_reduction_allowed": False,
        },
        "authorization_record_path": "oaci/reports/C85T_PI_AUTHORIZATION_RECORD.json",
        "authorization_schema": "c85t_direct_pi_authorization_record_v1",
        "future_direct_statement": "授权 C85T",
        "forbidden": {
            "S0_S10_execution_before_fresh_authorization": True,
            "real_project_data": True,
            "active_acquisition": True,
            "C85E": True,
            "new_data_or_model_zoo": True,
            "manuscript_work": True,
        },
        "readiness_success_gate": READINESS_SUCCESS_GATE,
        "readiness_failure_gate": READINESS_FAILURE_GATE,
    }
    lock_path.write_bytes(canonical_json_bytes(lock))
    lock_sha = sha256_file(lock_path)
    sidecar_path.write_text(f"{lock_sha}  {lock_path.name}\n")
    return {
        "lock_path": str(lock_path),
        "lock_sha256": lock_sha,
        "runtime_bound_object_count": len(rows),
        "runtime_registry_sha256": registry_identity["sha256"],
        "registered_scenarios_executed": 0,
        "proofs_completed": 0,
        "theorem_status_transitions": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-readiness-lock")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--implementation-commit", required=True)
    build.add_argument("--created-at-utc", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-readiness-lock":
        value = materialize_readiness_lock(
            repo_root=args.repo_root.resolve(),
            implementation_commit=args.implementation_commit,
            created_at_utc=args.created_at_utc,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    raise DecisionContractError("unknown C85TL lock materializer command")


if __name__ == "__main__":
    raise SystemExit(main())
