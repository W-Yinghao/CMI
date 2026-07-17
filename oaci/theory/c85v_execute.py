"""Single authorized Stage-A to Stage-B to adjudication C85V coordinator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
import uuid

from .c85_decision_experiments import DecisionContractError
from .c85v_adjudication import freeze_adjudication
from .c85v_result_manifest import AtomicC85VResultBundle, RESULT_SCHEMA, SUCCESS_GATE
from .c85v_stage_a_derivation import (
    freeze_stage_a_derivations,
    load_primary_source_ids,
    load_review_obligations,
)
from .c85v_stage_b_candidate_audit import freeze_stage_b_comparisons
from .c85v_statement_registry import (
    C85T_BUNDLE_ROOT,
    THEOREM_IDS,
    canonical_json_bytes,
    load_candidate_identities,
    load_registered_statements,
    load_review_protocol,
    sha256_file,
    validate_c85t_control_identity,
)


AUTHORIZATION_SCHEMA = "c85v_direct_pi_authorization_record_v1"
CONSUMPTION_SCHEMA = "c85v_authorization_consumption_receipt_v1"
LOCK_SCHEMA = "c85v_execution_lock_v1"
LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replay_execution_lock(lock_path: Path) -> tuple[dict[str, Any], str, Path]:
    lock_path = lock_path.resolve()
    sidecar = lock_path.with_suffix(".sha256")
    if not lock_path.is_file() or not sidecar.is_file():
        raise DecisionContractError("C85V execution lock or sidecar is absent")
    lock_sha = sha256_file(lock_path)
    if sidecar.read_text().split()[0] != lock_sha:
        raise DecisionContractError("C85V execution-lock sidecar drifted")
    lock = json.loads(lock_path.read_text())
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != LOCK_STATUS
        or lock.get("authorized") is not False
    ):
        raise DecisionContractError("C85V execution-lock contract drifted")
    repo_root = Path(str(lock["repo_root"])).resolve()
    if lock_path != repo_root / "oaci/reports/C85V_EXECUTION_LOCK.json":
        raise DecisionContractError("C85V execution-lock path drifted")
    rows = lock.get("bound_repository_objects")
    if not isinstance(rows, list) or lock.get("runtime_bound_object_count") != len(rows):
        raise DecisionContractError("C85V bound-object count drifted")
    for row in rows:
        relative = str(row["path"])
        path = repo_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
            or _git(repo_root, "hash-object", "--", relative) != row.get("git_blob")
        ):
            raise DecisionContractError(f"C85V bound repository object drifted: {relative}")
    registry = lock.get("runtime_bound_registry")
    if not isinstance(registry, dict):
        raise DecisionContractError("C85V runtime registry identity is absent")
    registry_path = repo_root / str(registry.get("path"))
    if (
        not registry_path.is_file()
        or registry_path.stat().st_size != registry.get("size_bytes")
        or sha256_file(registry_path) != registry.get("sha256")
        or _git(repo_root, "hash-object", "--", str(registry.get("path")))
        != registry.get("git_blob")
    ):
        raise DecisionContractError("C85V runtime registry identity drifted")
    return lock, lock_sha, repo_root


def _replay_external_review_objects(lock: Mapping[str, Any]) -> None:
    rows = lock.get("bound_external_objects")
    if not isinstance(rows, list) or len(rows) != 13:
        raise DecisionContractError("C85V external review-object coverage drifted")
    for row in rows:
        path = Path(str(row.get("path"))).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise DecisionContractError(f"C85V external review object drifted: {row.get('object')}")


def expected_output_root(
    lock: Mapping[str, Any], authorization_id: str, lock_sha: str
) -> Path:
    parent = Path(str(lock["output_root_policy"]["parent"])).resolve()
    basename = str(lock["output_root_policy"]["basename"]).format(
        lock_sha16=lock_sha[0:16],
        authorization_id16=authorization_id.replace("-", "")[0:16],
    )
    return parent / basename


def _validate_authorization(
    *,
    authorization_path: Path,
    output_root: Path,
    lock: Mapping[str, Any],
    lock_sha: str,
    lock_commit: str,
    repo_root: Path,
) -> tuple[dict[str, Any], str]:
    path = authorization_path.resolve()
    expected_path = repo_root / "oaci/reports/C85V_PI_AUTHORIZATION_RECORD.json"
    if path != expected_path or not path.is_file():
        raise DecisionContractError("C85V authorization must use the committed canonical path")
    if _git(repo_root, "ls-files", "--error-unmatch", str(path.relative_to(repo_root))) == "":
        raise DecisionContractError("C85V authorization record is not tracked")
    record = json.loads(path.read_text())
    required = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": "授权 C85V",
        "authorized_stage": "C85V",
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "c85t_result_sha256": lock["frozen_c85t"]["result_sha256"],
        "c85t_result_manifest_sha256": lock["frozen_c85t"]["result_manifest_sha256"],
        "C85E": False,
        "active_acquisition": False,
        "real_data": False,
        "new_data_or_model_zoo": False,
        "manuscript": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise DecisionContractError(f"C85V authorization binding drifted: {key}")
    authorization_id = record.get("authorization_id")
    if not isinstance(authorization_id, str) or len(authorization_id.replace("-", "")) < 32:
        raise DecisionContractError("C85V authorization ID is invalid")
    expected_root = expected_output_root(lock, authorization_id, lock_sha)
    if Path(str(record.get("output_root"))).resolve() != expected_root or output_root.resolve() != expected_root:
        raise DecisionContractError("C85V authorization output-root binding drifted")
    authorization_commit = _git(
        repo_root, "log", "-1", "--format=%H", "--", str(path.relative_to(repo_root))
    )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock_commit, authorization_commit],
        cwd=repo_root,
        check=False,
    ).returncode != 0:
        raise DecisionContractError("C85V authorization does not follow the execution lock")
    return record, sha256_file(path)


def _consume_authorization(
    *,
    lock: Mapping[str, Any],
    record: Mapping[str, Any],
    authorization_sha: str,
    lock_sha: str,
    lock_commit: str,
    output_root: Path,
    head: str,
    attempt_id: str,
) -> tuple[dict[str, Any], Path]:
    consumption_root = Path(str(lock["authorization_consumption_root"])).resolve()
    consumption_root.mkdir(parents=True, exist_ok=True)
    receipt_path = consumption_root / f"{authorization_sha}.json"
    receipt = {
        "schema_version": CONSUMPTION_SCHEMA,
        "authorized_stage": "C85V",
        "direct_explicit_PI_authorization": True,
        "authorization_file_sha256": authorization_sha,
        "authorization_id": record["authorization_id"],
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "output_root": str(output_root.resolve()),
        "attempt_id": attempt_id,
        "HEAD": head,
    }
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(canonical_json_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(consumption_root)
    return receipt, receipt_path


def _write_retention_ledger(
    path: Path,
    identities: Mapping[str, Any],
    bundle_root: Path,
) -> None:
    rows = [
        {
            "theorem_id": theorem_id,
            "external_path": str((bundle_root / identities[theorem_id].relative_path).resolve()),
            "sha256": identities[theorem_id].sha256,
            "retained": 1,
            "overwritten": 0,
        }
        for theorem_id in THEOREM_IDS
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_locked(
    *,
    execution_lock: Path,
    authorization_record: Path,
    output_root: Path,
) -> dict[str, Any]:
    lock, lock_sha, repo_root = replay_execution_lock(execution_lock)
    load_review_protocol(repo_root)
    head = _git(repo_root, "rev-parse", "HEAD")
    if (
        _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != "oaci"
        or head != _git(repo_root, "rev-parse", "origin/oaci")
        or _git(repo_root, "status", "--porcelain")
    ):
        raise DecisionContractError("C85V execution requires clean oaci HEAD == origin/oaci")
    lock_commit = _git(
        repo_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        str(execution_lock.resolve().relative_to(repo_root)),
    )
    record, authorization_sha = _validate_authorization(
        authorization_path=authorization_record,
        output_root=output_root,
        lock=lock,
        lock_sha=lock_sha,
        lock_commit=lock_commit,
        repo_root=repo_root,
    )
    validate_c85t_control_identity(C85T_BUNDLE_ROOT)
    attempt_id = uuid.uuid4().hex
    identity = {
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "authorization_file_sha256": authorization_sha,
        "attempt_id": attempt_id,
        "output_root": str(output_root.resolve()),
    }
    bundle = AtomicC85VResultBundle(
        output_root=output_root,
        attempt_id=attempt_id,
        identity=identity,
    )
    bundle.append_event("PREFLIGHT_STARTED")
    bundle.append_event("PREFLIGHT_COMPLETED")
    receipt, receipt_path = _consume_authorization(
        lock=lock,
        record=record,
        authorization_sha=authorization_sha,
        lock_sha=lock_sha,
        lock_commit=lock_commit,
        output_root=output_root,
        head=head,
        attempt_id=attempt_id,
    )
    bundle.write_json("authorization_consumed.json", receipt)
    bundle.append_event("AUTHORIZATION_CONSUMED", sha256_file(receipt_path))
    statements = load_registered_statements(repo_root)
    obligations = load_review_obligations(repo_root)
    sources = load_primary_source_ids(repo_root)
    identities = load_candidate_identities(repo_root)
    bundle.append_event("STAGE_A_STARTED")
    stage_a_root = bundle.staging_root / "stage_a"
    freeze_stage_a_derivations(
        statements=statements,
        obligations=obligations,
        available_source_ids=sources,
        output_root=stage_a_root,
        review_mode="REGISTERED_C85V",
    )
    bundle.append_event(
        "STAGE_A_COMPLETED",
        sha256_file(stage_a_root / "C85V_STAGE_A_DERIVATION_MANIFEST.json"),
    )
    _replay_external_review_objects(lock)
    bundle.append_event("STAGE_B_STARTED")
    stage_b_root = bundle.staging_root / "stage_b"
    freeze_stage_b_comparisons(
        stage_a_root=stage_a_root,
        candidate_bundle_root=C85T_BUNDLE_ROOT,
        exact_results_path=C85T_BUNDLE_ROOT / "exact_scenario_results.json",
        output_root=stage_b_root,
        statements=statements,
        identities=identities,
        review_mode="REGISTERED_C85V",
    )
    bundle.append_event(
        "STAGE_B_COMPLETED",
        sha256_file(stage_b_root / "C85V_STAGE_B_COMPARISON_MANIFEST.json"),
    )
    bundle.append_event("ADJUDICATION_STARTED")
    adjudication_root = bundle.staging_root / "adjudication"
    freeze_adjudication(
        stage_a_root=stage_a_root,
        stage_b_root=stage_b_root,
        output_root=adjudication_root,
        review_mode="REGISTERED_C85V",
        authorization_receipt=receipt,
    )
    bundle.append_event(
        "ADJUDICATION_COMPLETED",
        sha256_file(adjudication_root / "C85V_ADJUDICATION_MANIFEST.json"),
    )
    literature_source = repo_root / "oaci/reports/c85vp_tables/primary_literature_registry.csv"
    (bundle.staging_root / "primary_literature_registry.csv").write_bytes(
        literature_source.read_bytes()
    )
    _write_retention_ledger(
        bundle.staging_root / "proof_candidate_retention_ledger.csv",
        identities,
        C85T_BUNDLE_ROOT,
    )
    with (adjudication_root / "formal_theorem_status_registry.csv").open(newline="") as handle:
        statuses = {
            row["theorem_id"]: row["formal_status"] for row in csv.DictReader(handle)
        }
    result = {
        "schema_version": RESULT_SCHEMA,
        **identity,
        "c85t_result_sha256": lock["frozen_c85t"]["result_sha256"],
        "c85t_result_manifest_sha256": lock["frozen_c85t"]["result_manifest_sha256"],
        "theorem_count": 7,
        "formal_theorem_statuses": statuses,
        "monte_carlo_reruns": 0,
        "real_data_access": 0,
        "active_acquisition": 0,
        "C85E_authorized": False,
        "manuscript_modified": False,
        "final_gate": SUCCESS_GATE,
    }
    return bundle.prepare_and_publish(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-locked")
    run.add_argument("--execution-lock", type=Path, required=True)
    run.add_argument("--authorization-record", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_locked(
        execution_lock=args.execution_lock,
        authorization_record=args.authorization_record,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
