"""Single fail-closed coordinator for a future authorized C85T execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .c85_decision_experiments import DecisionContractError
from .c85r_synthetic_semantic_repair import (
    EXPECTED_V2_CONTRACT_SHA256,
    V2_CONTRACT_PATH,
    validate_locked_contracts,
)
from .c85t_exact_scenarios import execute_registered_exact_scenarios
from .c85t_monte_carlo import execute_registered_monte_carlo
from .c85t_proofs import execute_proof_pipeline
from .c85t_result_manifest import (
    AtomicResultWriter,
    RESULT_SCHEMA,
    SUCCESS_GATE,
    canonical_json_bytes,
    sha256_file,
    write_attempt_ledger,
)
from .c85t_rng import validate_environment


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
DEFAULT_LOCK_PATH = REPORT_DIR / "C85T_EXECUTION_LOCK.json"
DEFAULT_LOCK_SIDECAR = REPORT_DIR / "C85T_EXECUTION_LOCK.sha256"
OPERATIONALIZATION_PROTOCOL = REPORT_DIR / "C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json"
AUTHORIZATION_SCHEMA = "c85t_direct_pi_authorization_record_v1"
LOCK_SCHEMA = "c85t_execution_lock_v1"
LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
ATTEMPT_STAGE_ORDER = (
    "PREFLIGHT_REPLAY",
    "AUTHORIZATION_CONSUMPTION",
    "EXACT_SCENARIOS",
    "MONTE_CARLO",
    "PROOF_ARTIFACTS_AND_AUDIT",
    "ATOMIC_RESULT_PUBLICATION",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sidecar_digest(sidecar: Path, expected_name: str) -> str:
    rows = [line.split(maxsplit=1) for line in sidecar.read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][1].strip() != expected_name:
        raise DecisionContractError(f"invalid SHA-256 sidecar: {sidecar}")
    return rows[0][0]


def replay_execution_lock(lock_path: Path) -> tuple[dict[str, Any], str]:
    lock_path = lock_path.resolve()
    sidecar = lock_path.with_suffix(".sha256")
    if not lock_path.is_file() or not sidecar.is_file():
        raise DecisionContractError("C85T execution lock or sidecar is absent")
    lock_sha = sha256_file(lock_path)
    if _sidecar_digest(sidecar, lock_path.name) != lock_sha:
        raise DecisionContractError("C85T execution lock self-hash drifted")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("status") != LOCK_STATUS:
        raise DecisionContractError("C85T execution lock schema/status drifted")
    if lock.get("authorized") is not False:
        raise DecisionContractError("readiness lock must remain unauthorized")
    if lock.get("v2_generator_sha256") != EXPECTED_V2_CONTRACT_SHA256:
        raise DecisionContractError("C85T lock V2 generator identity drifted")
    if sha256_file(V2_CONTRACT_PATH) != EXPECTED_V2_CONTRACT_SHA256:
        raise DecisionContractError("C85T V2 generator bytes drifted")
    if lock.get("operationalization_protocol_sha256") != sha256_file(OPERATIONALIZATION_PROTOCOL):
        raise DecisionContractError("C85T operationalization protocol drifted")
    registry = lock.get("runtime_bound_registry")
    if not isinstance(registry, dict):
        raise DecisionContractError("C85T runtime-bound registry identity is absent")
    registry_path = REPO_ROOT / registry["path"]
    if (
        not registry_path.is_file()
        or registry_path.stat().st_size != registry["size_bytes"]
        or sha256_file(registry_path) != registry["sha256"]
        or _run_git("hash-object", "--", registry["path"]) != registry["git_blob"]
    ):
        raise DecisionContractError("C85T runtime-bound registry drifted")
    return lock, lock_sha


def replay_bound_repository_objects(lock: Mapping[str, Any]) -> dict[str, int]:
    rows = lock.get("bound_repository_objects")
    if not isinstance(rows, list) or not rows:
        raise DecisionContractError("C85T bound object registry is empty")
    bytes_replayed = 0
    observed: set[str] = set()
    for row in rows:
        relative = row["path"]
        if relative in observed:
            raise DecisionContractError("duplicate C85T bound path")
        observed.add(relative)
        path = REPO_ROOT / relative
        current_matches = (
            path.is_file()
            and path.stat().st_size == row["size_bytes"]
            and sha256_file(path) == row["sha256"]
            and _run_git("hash-object", "--", relative) == row["git_blob"]
        )
        if current_matches:
            bytes_replayed += path.stat().st_size
            continue
        if lock.get("schema_version") != LOCK_SCHEMA:
            raise DecisionContractError(f"C85T bound bytes drifted: {relative}")
        # The V1 lock is superseded. Replay its immutable archived Git blob rather
        # than treating post-repair working-tree bytes as executable V1 bytes.
        archived_size = int(_run_git("cat-file", "-s", row["git_blob"]))
        if archived_size != row["size_bytes"]:
            raise DecisionContractError(f"C85T archived blob drifted: {relative}")
        bytes_replayed += archived_size
    return {"object_count": len(rows), "bytes_replayed": bytes_replayed}


def replay_repository_state(lock: Mapping[str, Any], lock_path: Path) -> dict[str, str]:
    if _run_git("branch", "--show-current") != "oaci":
        raise DecisionContractError("C85T requires branch oaci")
    if _run_git("status", "--porcelain"):
        raise DecisionContractError("C85T requires a clean worktree")
    head = _run_git("rev-parse", "HEAD")
    origin = _run_git("rev-parse", "origin/oaci")
    if head != origin:
        raise DecisionContractError("C85T requires HEAD == origin/oaci")
    lock_commit = _run_git(
        "log", "-1", "--format=%H", "--", str(lock_path.resolve().relative_to(REPO_ROOT))
    )
    if not lock_commit:
        raise DecisionContractError("C85T execution lock is not committed")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock_commit, head],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestry.returncode != 0:
        raise DecisionContractError("C85T execution-lock commit is not an ancestor")
    implementation_commit = lock["implementation_commit"]
    implementation_ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, lock_commit],
        cwd=REPO_ROOT,
        check=False,
    )
    if implementation_ancestry.returncode != 0:
        raise DecisionContractError("C85T implementation does not precede the lock")
    return {
        "branch": "oaci",
        "HEAD": head,
        "origin_oaci": origin,
        "execution_lock_commit": lock_commit,
        "implementation_commit": implementation_commit,
    }


def replay_authorization(
    authorization_path: Path,
    lock: Mapping[str, Any],
    lock_sha: str,
    lock_commit: str,
) -> tuple[dict[str, Any], str]:
    expected = (REPO_ROOT / lock["authorization_record_path"]).resolve()
    if authorization_path.resolve() != expected:
        raise DecisionContractError("C85T authorization path is not lock-bound")
    if not authorization_path.is_file():
        raise DecisionContractError("fresh direct C85T authorization record is absent")
    value = json.loads(authorization_path.read_text())
    required = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": "授权 C85T",
        "authorized_stage": "C85T",
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "C85E": False,
        "active_acquisition": False,
        "real_data": False,
        "manuscript": False,
    }
    for key, expected_value in required.items():
        if value.get(key) != expected_value:
            raise DecisionContractError(f"C85T authorization binding drifted: {key}")
    return value, sha256_file(authorization_path)


def preflight(
    *, lock_path: Path, authorization_path: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise DecisionContractError("C85T output root must be fresh")
    if output_root.with_name(f"{output_root.name}.attempt").exists():
        raise DecisionContractError("C85T attempt root must be fresh")
    locked = validate_locked_contracts()
    lock, lock_sha = replay_execution_lock(lock_path)
    bound = replay_bound_repository_objects(lock)
    repository = replay_repository_state(lock, lock_path)
    environment = validate_environment(strict_prefix=True)
    authorization, authorization_sha = replay_authorization(
        authorization_path,
        lock,
        lock_sha,
        repository["execution_lock_commit"],
    )
    if locked["v2_sha256"] != lock["v2_generator_sha256"]:
        raise DecisionContractError("C85T semantic-contract replay drifted")
    return {
        "lock": lock,
        "lock_sha256": lock_sha,
        "bound_objects": bound,
        "repository": repository,
        "environment": environment,
        "authorization": authorization,
        "authorization_sha256": authorization_sha,
        "contract": locked["v2"],
    }


def _proof_status_csv(statuses: Mapping[str, str]) -> str:
    lines = ["theorem_id,historical_status,C85T_status,status_transition"]
    for theorem_id in (f"T{i}" for i in range(1, 8)):
        status = statuses[theorem_id]
        lines.append(f"{theorem_id},OPEN,{status},{int(status != 'OPEN')}")
    return "\n".join(lines) + "\n"


def run_real(*, lock_path: Path, output_root: Path) -> dict[str, Any]:
    raise DecisionContractError(
        "historical C85T V1 coordinator is superseded and cannot execute"
    )
    # Unreachable historical code is retained below solely as Git history context.
    lock_preview, _ = replay_execution_lock(lock_path)
    authorization_path = REPO_ROOT / lock_preview["authorization_record_path"]
    replay = preflight(
        lock_path=lock_path,
        authorization_path=authorization_path,
        output_root=output_root,
    )
    attempt_root = output_root.with_name(f"{output_root.name}.attempt")
    attempt_root.mkdir(parents=True, exist_ok=False)
    attempt_path = attempt_root / "C85T_EXECUTION_ATTEMPT.json"
    write_attempt_ledger(
        attempt_path,
        {
            "created_at_utc": _utc_now(),
            "stage_order": ATTEMPT_STAGE_ORDER,
            "current_stage": "AUTHORIZATION_CONSUMPTION",
            "lock_sha256": replay["lock_sha256"],
            "authorization_sha256": replay["authorization_sha256"],
            "registered_scenario_execution_started": False,
            "real_project_data_access": 0,
        },
    )
    consumed = {
        "schema_version": "c85t_authorization_consumption_receipt_v1",
        "consumed_at_utc": _utc_now(),
        "authorization_sha256": replay["authorization_sha256"],
        "execution_lock_sha256": replay["lock_sha256"],
        "authorized_stage": "C85T",
        "real_project_data_access": 0,
    }
    (attempt_root / "authorization_consumed.json").write_bytes(canonical_json_bytes(consumed))

    try:
        exact = execute_registered_exact_scenarios(
            replay["contract"], capability=None
        )
        monte_carlo = execute_registered_monte_carlo(
            replay["contract"], capability=None
        )
        protocol = json.loads(OPERATIONALIZATION_PROTOCOL.read_text())
        with AtomicResultWriter(output_root) as writer:
            writer.write_json("exact_scenario_results.json", exact)
            writer.write_json("monte_carlo_results.json", monte_carlo)
            writer.write_json("authorization_consumed.json", consumed)
            proof_dir = writer.staging_root / "c85t_proofs"
            statuses = execute_proof_pipeline(
                statements=protocol["proof_statements"],
                exact_results=exact,
                output_dir=proof_dir,
                capability=None,
            )
            writer.write_text("theorem_status_transitions.csv", _proof_status_csv(statuses))
            result = {
                "schema_version": RESULT_SCHEMA,
                "created_at_utc": _utc_now(),
                "final_gate": SUCCESS_GATE,
                "execution_lock_sha256": replay["lock_sha256"],
                "authorization_sha256": replay["authorization_sha256"],
                "scenario_count": 11,
                "monte_carlo_replicates": {"S6": 4096, "S7": 4096, "S9": 4096},
                "theorem_statuses": statuses,
                "T5_may_remain_OPEN": True,
                "real_project_data_access": 0,
                "active_acquisition": 0,
                "C85E_authorized": False,
                "manuscript_modified": False,
            }
            writer.publish(result)
        completion = {
            "schema_version": "c85t_execution_completion_receipt_v1",
            "completed_at_utc": _utc_now(),
            "output_root": str(output_root.resolve()),
            "result_sha256": sha256_file(output_root / "C85T_RESULT.json"),
            "manifest_sha256": sha256_file(output_root / "C85T_RESULT_ARTIFACT_MANIFEST.json"),
            "final_gate": SUCCESS_GATE,
        }
        (attempt_root / "completion.json").write_bytes(canonical_json_bytes(completion))
        return completion
    except BaseException as error:
        failure = {
            "schema_version": "c85t_execution_failure_receipt_v1",
            "failed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "authorization_consumed": True,
            "real_project_data_access": 0,
            "automatic_retry": False,
        }
        (attempt_root / "failure.json").write_bytes(canonical_json_bytes(failure))
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-locked")
    run.add_argument("--execution-lock", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run-locked":
        result = run_real(
            lock_path=args.execution_lock,
            output_root=args.output_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise DecisionContractError("unknown C85T coordinator command")


if __name__ == "__main__":
    raise SystemExit(main())
