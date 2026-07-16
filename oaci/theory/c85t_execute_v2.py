"""Fail-closed C85T V2 coordinator; C85TR1 never invokes ``run-locked``."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85r_synthetic_semantic_repair import validate_locked_contracts
from .c85t_exact_scenarios import as_fraction, execute_registered_exact_scenarios
from .c85t_execution_guard import (
    AppendOnlyLifecycleLedger,
    authorization_binding_sha256,
    canonical_json_bytes,
    consume_authorization_once,
    replay_lifecycle,
    sha256_file,
    validate_authorization_record,
)
from .c85t_monte_carlo import (
    _summarize_s9_arrays_v2,
    execute_registered_monte_carlo,
    summarize_near_replicates_v2,
)
from .c85t_proofs import execute_proof_candidate_pipeline_v2
from .c85t_result_manifest import (
    AtomicResultWriterV2,
    RESULT_SCHEMA_V2,
    SUCCESS_GATE_V2,
    read_deterministic_npz,
)
from .c85t_rng import validate_environment


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
DEFAULT_LOCK_PATH = REPORT_DIR / "C85T_EXECUTION_LOCK_V2.json"
LOCK_SCHEMA = "c85t_execution_lock_v2"
LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sidecar_digest(path: Path, expected_name: str) -> str:
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line]
    if len(rows) != 1 or rows[0][1].strip() != expected_name:
        raise DecisionContractError("invalid C85T V2 SHA-256 sidecar")
    return rows[0][0]


def replay_execution_lock_v2(lock_path: Path) -> tuple[dict[str, Any], str]:
    lock_path = lock_path.resolve()
    sidecar = lock_path.with_suffix(".sha256")
    if not lock_path.is_file() or not sidecar.is_file():
        raise DecisionContractError("C85T V2 execution lock or sidecar is absent")
    lock_sha = sha256_file(lock_path)
    if _sidecar_digest(sidecar, lock_path.name) != lock_sha:
        raise DecisionContractError("C85T V2 execution lock self-hash drifted")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("status") != LOCK_STATUS:
        raise DecisionContractError("C85T V2 lock schema/status drifted")
    if lock.get("authorized") is not False:
        raise DecisionContractError("C85T V2 readiness lock must be unauthorized")
    for identity_key, relative_key in (
        ("c85p_protocol_sha256", "c85p_protocol_path"),
        ("c85r_repair_protocol_sha256", "c85r_repair_protocol_path"),
        ("v2_generator_sha256", "v2_generator_path"),
        ("c85tl_operationalization_sha256", "c85tl_operationalization_path"),
        ("c85tr1_repair_protocol_sha256", "c85tr1_repair_protocol_path"),
    ):
        path = REPO_ROOT / lock[relative_key]
        if not path.is_file() or sha256_file(path) != lock[identity_key]:
            raise DecisionContractError(f"C85T V2 protocol identity drifted: {relative_key}")
    registry = lock.get("runtime_bound_registry")
    if not isinstance(registry, dict):
        raise DecisionContractError("C85T V2 runtime registry binding is absent")
    path = REPO_ROOT / registry["path"]
    if (
        not path.is_file()
        or path.stat().st_size != registry["size_bytes"]
        or sha256_file(path) != registry["sha256"]
        or _git("hash-object", "--", registry["path"]) != registry["git_blob"]
    ):
        raise DecisionContractError("C85T V2 runtime registry drifted")
    return lock, lock_sha


def replay_bound_repository_objects_v2(lock: Mapping[str, Any]) -> dict[str, int]:
    rows = lock.get("bound_repository_objects")
    if not isinstance(rows, list) or not rows:
        raise DecisionContractError("C85T V2 bound object registry is empty")
    observed: set[str] = set()
    total = 0
    for row in rows:
        relative = row["path"]
        if relative in observed:
            raise DecisionContractError("duplicate C85T V2 bound path")
        observed.add(relative)
        path = REPO_ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or sha256_file(path) != row["sha256"]
            or _git("hash-object", "--", relative) != row["git_blob"]
        ):
            raise DecisionContractError(f"C85T V2 bound object drifted: {relative}")
        total += path.stat().st_size
    if len(rows) != lock.get("runtime_bound_object_count"):
        raise DecisionContractError("C85T V2 bound object count drifted")
    return {"object_count": len(rows), "bytes_replayed": total}


def replay_repository_state_v2(
    lock: Mapping[str, Any], lock_path: Path
) -> dict[str, str]:
    if _git("branch", "--show-current") != "oaci":
        raise DecisionContractError("C85T V2 requires branch oaci")
    if _git("status", "--porcelain"):
        raise DecisionContractError("C85T V2 requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/oaci")
    if head != origin:
        raise DecisionContractError("C85T V2 requires HEAD == origin/oaci")
    lock_commit = _git(
        "log", "-1", "--format=%H", "--", str(lock_path.resolve().relative_to(REPO_ROOT))
    )
    if not lock_commit:
        raise DecisionContractError("C85T V2 lock is not committed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock_commit, head],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise DecisionContractError("C85T V2 lock commit is not an ancestor of HEAD")
    if subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            lock["implementation_commit"],
            lock_commit,
        ],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise DecisionContractError("C85T V2 implementation does not precede lock")
    return {
        "branch": "oaci",
        "HEAD": head,
        "origin_oaci": origin,
        "execution_lock_commit": lock_commit,
    }


def replay_authorization_v2(
    path: Path,
    *,
    lock: Mapping[str, Any],
    lock_sha: str,
    lock_commit: str,
) -> tuple[dict[str, Any], str]:
    expected = (REPO_ROOT / lock["authorization_record_path"]).resolve()
    if path.resolve() != expected:
        raise DecisionContractError("C85T V2 authorization path is not lock-bound")
    if not path.is_file():
        raise DecisionContractError("fresh direct C85T V2 authorization record is absent")
    record = json.loads(path.read_text())
    authorization_sha = authorization_binding_sha256(record)
    return (
        validate_authorization_record(
            record,
            authorization_sha256=authorization_sha,
            lock_sha256=lock_sha,
            lock_commit=lock_commit,
            output_parent=lock["output_root_policy"]["parent"],
            consumption_root=lock["authorization_consumption_root"],
        ),
        authorization_sha,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise DecisionContractError("cannot write empty C85T V2 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_real(*, lock_path: Path, output_root: Path) -> dict[str, Any]:
    lock, lock_sha = replay_execution_lock_v2(lock_path)
    authorization_path = REPO_ROOT / lock["authorization_record_path"]
    if not authorization_path.is_file():
        raise DecisionContractError("fresh direct C85T V2 authorization record is absent")
    authorization_preview = json.loads(authorization_path.read_text())
    authorization_sha = authorization_binding_sha256(authorization_preview)
    authorization_id = str(authorization_preview.get("authorization_id", ""))
    attempt_id = uuid4().hex
    attempt_root = output_root.with_name(
        f"{output_root.name}.attempt-{authorization_id.replace('-', '')[:16]}"
    )
    if output_root.exists() or attempt_root.exists():
        raise DecisionContractError("C85T V2 output and attempt roots must be fresh")
    attempt_root.mkdir(parents=True, exist_ok=False)
    ledger = AppendOnlyLifecycleLedger(
        attempt_root / "C85T_V2_LIFECYCLE.jsonl",
        authorization_sha256=authorization_sha,
        lock_sha256=lock_sha,
        attempt_id=attempt_id,
    )
    consumed = False
    try:
        ledger.append("PREFLIGHT_STARTED")
        locked = validate_locked_contracts()
        bound = replay_bound_repository_objects_v2(lock)
        repository = replay_repository_state_v2(lock, lock_path)
        environment = validate_environment(strict_prefix=True)
        authorization, observed_authorization_sha = replay_authorization_v2(
            authorization_path,
            lock=lock,
            lock_sha=lock_sha,
            lock_commit=repository["execution_lock_commit"],
        )
        if observed_authorization_sha != authorization_sha:
            raise DecisionContractError("authorization bytes changed during preflight")
        if Path(authorization["output_root"]).resolve() != output_root.resolve():
            raise DecisionContractError("CLI output root differs from authorization binding")
        if locked["v2_sha256"] != lock["v2_generator_sha256"]:
            raise DecisionContractError("C85T V2 semantic contract drifted")
        preflight_receipt = {
            "schema_version": "c85t_v2_preflight_receipt_v1",
            "bound_objects": bound,
            "repository": repository,
            "environment": environment,
            "authorization_sha256": authorization_sha,
            "authorization_file_sha256": sha256_file(authorization_path),
            "output_root": str(output_root.resolve()),
        }
        preflight_path = attempt_root / "preflight_completed.json"
        preflight_path.write_bytes(canonical_json_bytes(preflight_receipt))
        ledger.append(
            "PREFLIGHT_COMPLETED",
            artifact_or_receipt_sha256=sha256_file(preflight_path),
        )
        consumption, capability = consume_authorization_once(
            record=authorization,
            authorization_sha256=authorization_sha,
            lock_sha256=lock_sha,
            lock_commit=repository["execution_lock_commit"],
            output_root=output_root,
            attempt_id=attempt_id,
            head=repository["HEAD"],
        )
        consumed = True
        consumption_path = attempt_root / "authorization_consumed.json"
        consumption_path.write_bytes(canonical_json_bytes(consumption))
        consumption_sha = sha256_file(consumption_path)
        ledger.append(
            "AUTHORIZATION_CONSUMED",
            artifact_or_receipt_sha256=consumption_sha,
        )
        protocol = json.loads(
            (REPO_ROOT / lock["c85tl_operationalization_path"]).read_text()
        )
        with AtomicResultWriterV2(output_root) as writer:
            writer.write_json("authorization_consumed.json", consumption)
            ledger.append("EXACT_SCENARIOS_STARTED")
            exact = execute_registered_exact_scenarios(
                locked["v2"], capability=capability
            )
            exact_path = writer.write_json("exact_scenario_results.json", exact)
            ledger.append(
                "EXACT_SCENARIOS_COMPLETED",
                artifact_or_receipt_sha256=sha256_file(exact_path),
            )
            ledger.append("MONTE_CARLO_STARTED")
            monte = execute_registered_monte_carlo(
                locked["v2"], capability=capability
            )
            summaries: dict[str, Any] = {}
            for scenario_id in ("S6", "S7"):
                npz = writer.write_npz(
                    f"{scenario_id}_replicates.npz",
                    monte[scenario_id]["arrays"],
                )
                reloaded = read_deterministic_npz(npz)
                summaries[scenario_id] = summarize_near_replicates_v2(
                    scenario_id,
                    reloaded,
                    monte[scenario_id]["summary"]["geometry"],
                )
            s9_npz = writer.write_npz("S9_replicates.npz", monte["S9"]["arrays"])
            s9_reloaded = read_deterministic_npz(s9_npz)
            s9_scenario = {row["id"]: row for row in locked["v2"]["scenarios"]}["S9"]
            population = [
                float(as_fraction(value))
                for value in s9_scenario["population_mean_losses"]
            ]
            summaries["S9"] = _summarize_s9_arrays_v2(
                s9_reloaded, np.asarray(population, dtype="<f8")
            )
            summaries["S9"]["analytic_variance"] = monte["S9"]["summary"][
                "analytic_variance"
            ]
            summaries["S9"]["universal_active_superiority_claim"] = False
            summaries["S9_population_mean_losses"] = population
            digest_path = writer.path("S9_raw_draw_digest_registry.csv")
            _write_csv(digest_path, monte["S9"]["raw_draw_digest_rows"])
            summary_path = writer.write_json("monte_carlo_summary.json", summaries)
            ledger.append(
                "MONTE_CARLO_COMPLETED",
                artifact_or_receipt_sha256=sha256_file(summary_path),
            )
            ledger.append("PROOF_CANDIDATES_STARTED")
            dispositions = execute_proof_candidate_pipeline_v2(
                statements=protocol["proof_statements"],
                exact_results=exact,
                output_dir=writer.path("c85t_proof_candidates"),
                dispositions_path=writer.path("proof_candidate_dispositions.csv"),
                capability=capability,
            )
            disposition_sha = sha256_file(
                writer.path("proof_candidate_dispositions.csv")
            )
            ledger.append(
                "PROOF_CANDIDATES_COMPLETED",
                artifact_or_receipt_sha256=disposition_sha,
            )
            result = {
                "schema_version": RESULT_SCHEMA_V2,
                "final_gate": SUCCESS_GATE_V2,
                "execution_lock_sha256": lock_sha,
                "authorization_sha256": authorization_sha,
                "scenario_count": 11,
                "S6_S7_logical_replicate_rows": 8192,
                "S9_logical_replicate_design_rows": 8192,
                "S9_raw_draw_digest_rows": 4096,
                "proof_candidate_dispositions": dispositions,
                "formal_theorem_statuses": {f"T{i}": "OPEN" for i in range(1, 8)},
                "real_project_data_access": 0,
                "active_acquisition": 0,
                "C85E_authorized": False,
                "manuscript_modified": False,
            }
            ledger.append("MANIFEST_STARTED")
            writer.publish(
                result,
                manifest_completed_callback=lambda digest: ledger.append(
                    "MANIFEST_COMPLETED",
                    artifact_or_receipt_sha256=digest,
                ),
                atomic_publish_callback=lambda digest: ledger.append(
                    "ATOMIC_PUBLISH_COMPLETED",
                    artifact_or_receipt_sha256=digest,
                ),
            )
        replay_lifecycle(ledger.path)
        completion = {
            "schema_version": "c85t_v2_execution_completion_receipt_v1",
            "output_root": str(output_root.resolve()),
            "result_sha256": sha256_file(output_root / "C85T_RESULT.json"),
            "manifest_sha256": sha256_file(
                output_root / "C85T_RESULT_ARTIFACT_MANIFEST.json"
            ),
            "final_gate": SUCCESS_GATE_V2,
        }
        (attempt_root / "completion.json").write_bytes(canonical_json_bytes(completion))
        return completion
    except BaseException as error:
        try:
            ledger.append(
                "FAILED",
                failure={
                    "primary_exception_type": type(error).__name__,
                    "primary_exception_message": str(error),
                },
            )
        finally:
            failure = {
                "schema_version": "c85t_v2_execution_failure_receipt_v1",
                "authorization_consumed": consumed,
                "last_completed_stage": ledger.last_completed_stage,
                "primary_exception_type": type(error).__name__,
                "primary_exception_message": str(error),
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
        print(
            json.dumps(
                run_real(
                    lock_path=args.execution_lock.resolve(),
                    output_root=args.output_root.resolve(),
                ),
                sort_keys=True,
            )
        )
        return 0
    raise DecisionContractError("unknown C85T V2 coordinator command")


if __name__ == "__main__":
    raise SystemExit(main())
