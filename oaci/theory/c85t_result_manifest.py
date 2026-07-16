"""Atomic C85T result publication and artifact identity replay."""
from __future__ import annotations

from contextlib import AbstractContextManager
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4
import zipfile

import numpy as np

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
RESULT_SCHEMA_V2 = "c85t_synthetic_validation_and_proof_candidates_result_v2"
MANIFEST_SCHEMA_V2 = "c85t_atomic_result_manifest_v2"
SUCCESS_GATE_V2 = (
    "C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_"
    "C85V_REVIEW_REQUIRED"
)
PROOF_FILENAMES_V2 = (
    "T1_blackwell_monotonicity.md",
    "T2_restricted_policy_counterexamples.md",
    "T3_policy_collapse.md",
    "T4_two_state_lecam_regret_bound.md",
    "T5_fano_extension.md",
    "T6_mean_tail_counterexample.md",
    "T7_near_optimal_union_bound.md",
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


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write non-object NPY members with fixed ZIP metadata and canonical order."""

    if not arrays:
        raise DecisionContractError("deterministic NPZ requires at least one array")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DecisionContractError("deterministic NPZ path must be fresh")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise DecisionContractError("invalid NPZ member name")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise DecisionContractError("NPZ arrays must be non-object and finite")
            buffer = BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(array), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())


def read_deterministic_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in sorted(loaded.files)}
    if any(value.dtype.hasobject or not np.isfinite(value).all() for value in arrays.values()):
        raise DecisionContractError("persisted NPZ contains object or nonfinite arrays")
    return arrays


def _required_v2_artifacts() -> set[str]:
    required = {
        "exact_scenario_results.json",
        "monte_carlo_summary.json",
        "S6_replicates.npz",
        "S7_replicates.npz",
        "S9_replicates.npz",
        "S9_raw_draw_digest_registry.csv",
        "proof_candidate_dispositions.csv",
        "authorization_consumed.json",
        "C85T_RESULT.json",
    }
    required.update(f"c85t_proof_candidates/{name}" for name in PROOF_FILENAMES_V2)
    return required


def _validate_v2_result_root(root: Path, result: dict[str, Any]) -> dict[str, int]:
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "C85T_RESULT_ARTIFACT_MANIFEST.json"
    }
    missing = _required_v2_artifacts() - actual
    if missing:
        raise DecisionContractError(f"C85T V2 required artifact is absent: {min(missing)}")
    from .c85t_monte_carlo import (
        _summarize_s9_arrays_v2,
        summarize_near_replicates_v2,
    )

    summaries = json.loads((root / "monte_carlo_summary.json").read_text())
    population = np.asarray(summaries["S9_population_mean_losses"], dtype="<f8")
    logical_near_rows = 0
    for scenario_id in ("S6", "S7"):
        arrays = read_deterministic_npz(root / f"{scenario_id}_replicates.npz")
        replay = summarize_near_replicates_v2(
            scenario_id, arrays, summaries[scenario_id]["geometry"]
        )
        if replay != summaries[scenario_id]:
            raise DecisionContractError(
                f"{scenario_id} aggregate does not replay from saved arrays"
            )
        logical_near_rows += int(arrays["replicate_id"].size)
    s9_arrays = read_deterministic_npz(root / "S9_replicates.npz")
    s9_replay = _summarize_s9_arrays_v2(s9_arrays, population)
    for key in ("analytic_variance", "universal_active_superiority_claim"):
        s9_replay[key] = summaries["S9"][key]
    if s9_replay != summaries["S9"]:
        raise DecisionContractError("S9 aggregate does not replay from saved arrays")
    with (root / "S9_raw_draw_digest_registry.csv").open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if len(raw_rows) != 4096 or [int(row["replicate_id"]) for row in raw_rows] != list(range(4096)):
        raise DecisionContractError("S9 raw-draw digest registry coverage drifted")
    with (root / "proof_candidate_dispositions.csv").open(newline="") as handle:
        dispositions = list(csv.DictReader(handle))
    if len(dispositions) != 7 or any(row["formal_status"] != "OPEN" for row in dispositions):
        raise DecisionContractError("proof-candidate formal status must remain OPEN")
    if result.get("scenario_count") != 11:
        raise DecisionContractError("C85T V2 scenario count drifted")
    if result.get("formal_theorem_statuses") != {f"T{i}": "OPEN" for i in range(1, 8)}:
        raise DecisionContractError("C85T V2 theorem status transitioned")
    if result.get("real_project_data_access") != 0 or result.get("active_acquisition") != 0:
        raise DecisionContractError("C85T V2 protected counter is nonzero")
    return {
        "scenario_results": 11,
        "S6_S7_logical_replicate_rows": logical_near_rows,
        "S9_logical_replicate_design_rows": 8192,
        "S9_raw_draw_digest_rows": len(raw_rows),
        "proof_candidates": len(dispositions),
        "formal_theorem_status_OPEN": 7,
    }


class AtomicResultWriterV2(AbstractContextManager["AtomicResultWriterV2"]):
    """Atomic V2 writer that validates persisted replicates before publication."""

    def __init__(self, output_root: Path, *, failure_injection: str | None = None):
        self.output_root = output_root.resolve()
        self.staging_root = self.output_root.with_name(
            f".{self.output_root.name}.staging-{uuid4().hex}"
        )
        self.failure_injection = failure_injection
        self._published = False
        self.failed_root: Path | None = None

    def __enter__(self) -> "AtomicResultWriterV2":
        if self.output_root.exists():
            raise DecisionContractError("C85T V2 output root must be absent")
        self.staging_root.mkdir(parents=True, exist_ok=False)
        return self

    def path(self, relative: str | Path) -> Path:
        path = (self.staging_root / relative).resolve()
        if self.staging_root not in path.parents and path != self.staging_root:
            raise DecisionContractError("C85T V2 result path escapes staging")
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

    def write_npz(self, relative: str | Path, arrays: dict[str, np.ndarray]) -> Path:
        path = self.path(relative)
        write_deterministic_npz(path, arrays)
        return path

    def _artifact_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(path.relative_to(self.staging_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(self.staging_root.rglob("*"))
            if path.is_file()
            and path.name != "C85T_RESULT_ARTIFACT_MANIFEST.json"
        ]

    def publish(
        self,
        result: dict[str, Any],
        *,
        manifest_completed_callback: Any = None,
        atomic_publish_callback: Any = None,
    ) -> Path:
        if result.get("schema_version") != RESULT_SCHEMA_V2:
            raise DecisionContractError("C85T V2 result schema drifted")
        if result.get("final_gate") != SUCCESS_GATE_V2:
            raise DecisionContractError("C85T V2 success gate drifted")
        if self.failure_injection == "before_result":
            raise RuntimeError("C85T_V2_SHADOW_FAILURE_BEFORE_RESULT")
        self.write_json("C85T_RESULT.json", result)
        counts = _validate_v2_result_root(self.staging_root, result)
        rows = self._artifact_rows()
        manifest = {
            "schema_version": MANIFEST_SCHEMA_V2,
            "created_at_utc": utc_now(),
            "artifact_count": len(rows),
            "counts": counts,
            "artifacts": rows,
        }
        if self.failure_injection == "before_manifest":
            raise RuntimeError("C85T_V2_SHADOW_FAILURE_BEFORE_MANIFEST")
        self.write_json("C85T_RESULT_ARTIFACT_MANIFEST.json", manifest)
        replay_manifest_v2(self.staging_root)
        if manifest_completed_callback is not None:
            manifest_completed_callback(
                sha256_file(
                    self.staging_root / "C85T_RESULT_ARTIFACT_MANIFEST.json"
                )
            )
        if self.failure_injection == "before_publish":
            raise RuntimeError("C85T_V2_SHADOW_FAILURE_BEFORE_PUBLISH")
        os.replace(self.staging_root, self.output_root)
        self._published = True
        if atomic_publish_callback is not None:
            atomic_publish_callback(
                sha256_file(self.output_root / "C85T_RESULT_ARTIFACT_MANIFEST.json")
            )
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


def replay_manifest_v2(root: Path) -> dict[str, Any]:
    path = root / "C85T_RESULT_ARTIFACT_MANIFEST.json"
    if not path.is_file():
        raise DecisionContractError("C85T V2 manifest is absent")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA_V2:
        raise DecisionContractError("C85T V2 manifest schema drifted")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise DecisionContractError("C85T V2 manifest count drifted")
    observed: set[str] = set()
    for row in rows:
        relative = row["path"]
        if relative in observed:
            raise DecisionContractError("duplicate C85T V2 artifact path")
        observed.add(relative)
        artifact = root / relative
        if not artifact.is_file() or artifact.stat().st_size != row["size_bytes"]:
            raise DecisionContractError(f"C85T V2 artifact is absent: {relative}")
        if sha256_file(artifact) != row["sha256"]:
            raise DecisionContractError(f"C85T V2 artifact hash drifted: {relative}")
    actual = {
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file() and item.name != path.name
    }
    if observed != actual:
        raise DecisionContractError("C85T V2 manifest coverage drifted")
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
