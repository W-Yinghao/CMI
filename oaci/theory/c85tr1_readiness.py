"""C85TR1 shadow calibration, contract tables, and prospective V2 lock builder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85t_execution_guard import LIFECYCLE_EVENTS, canonical_json_bytes
from .c85t_monte_carlo import (
    _summarize_s9_arrays_v2,
    simulate_full_information_designs_v2,
    simulate_near_optimal_selection_v2,
    summarize_near_replicates_v2,
)
from .c85t_result_manifest import read_deterministic_npz, write_deterministic_npz
from .c85t_rng import (
    canonical_int64_sha256,
    deterministic_seed,
    draw_s9_rademacher_int64,
)


PROTOCOL_SHA256 = "9c0a7084a7ddd83ef96b8d7f95faf89138829729c0acc5c3d6baeb0ef87ab13d"
PROTOCOL_COMMIT = "46442b281d61d00a575fae17685648b749659263"
HISTORICAL_LOCK_SHA256 = "4a289a46040b10855c6f23def53c328bdce0a8b1c71b7e90523887b6c1db7991"
LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
SUCCESS_GATE = (
    "C85T_EXECUTION_GUARD_RNG_REPLICATE_PERSISTENCE_AND_PROOF_REVIEW_"
    "REPAIRED_V2_LOCK_READY_FOR_PI_AUTHORIZATION"
)
FAILURE_GATE = (
    "C85T_AUTHORIZATION_RNG_PERSISTENCE_PROOF_INDEPENDENCE_OR_"
    "LIFECYCLE_RECONCILIATION_REQUIRED"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise DecisionContractError(f"refusing empty C85TR1 table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _combined_int64_sha(low: np.ndarray, high: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(low, dtype="<i8").tobytes())
    digest.update(np.asarray(high, dtype="<i8").tobytes())
    return digest.hexdigest()


def _shadow_replicate_replay_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in ("SHADOW_RADEMACHER_A", "SHADOW_RADEMACHER_B"):
        for replicate in (0, 1, 4095):
            low, high = draw_s9_rademacher_int64(fixture, replicate)
            rows.append(
                {
                    "fixture_id": fixture,
                    "replicate_id": replicate,
                    "seed_low64": deterministic_seed(fixture, replicate),
                    "dtype": low.dtype.str,
                    "L_count": low.size,
                    "H_count": high.size,
                    "L_sha256": canonical_int64_sha256(low),
                    "H_sha256": canonical_int64_sha256(high),
                    "combined_sha256": _combined_int64_sha(low, high),
                    "registered_scenario": 0,
                    "scientific_result": 0,
                }
            )
    return rows


def _shadow_aggregate_replay_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="c85tr1-shadow-") as temporary:
        root = Path(temporary)
        for label, utilities in (
            ("S6_SCHEMA_SHADOW", (1.0, 0.98, 0.7)),
            ("S7_SCHEMA_SHADOW", (1.0, 0.8, 0.3)),
        ):
            in_memory, arrays = simulate_near_optimal_selection_v2(
                scenario_id="SHADOW_NORMAL_A",
                utilities=utilities,
                epsilon=0.03,
                tau=0.1,
                pairwise_sigma=0.05,
            )
            path = root / f"{label}.npz"
            write_deterministic_npz(path, arrays)
            loaded = read_deterministic_npz(path)
            replay = summarize_near_replicates_v2(
                "SHADOW_NORMAL_A", loaded, in_memory["geometry"]
            )
            rows.append(
                {
                    "fixture": label,
                    "logical_rows": 4096,
                    "saved_array_sha256": _sha(path),
                    "aggregate_exact_replay": int(replay == in_memory),
                    "registered_scenario": 0,
                    "scientific_result": 0,
                }
            )
        in_memory, arrays, digests = simulate_full_information_designs_v2(
            scenario_id="SHADOW_RADEMACHER_A",
            stratum_masses=(0.7, 0.3),
            sigmas=(0.03, 0.15),
            passive_allocation=(51, 13),
            neyman_allocation=(18, 46),
            population_mean_losses=(0.2, 0.24, 0.6, 0.8),
            action1_offset=0.04,
        )
        path = root / "S9_SCHEMA_SHADOW.npz"
        write_deterministic_npz(path, arrays)
        loaded = read_deterministic_npz(path)
        replay = _summarize_s9_arrays_v2(
            loaded, np.asarray((0.2, 0.24, 0.6, 0.8), dtype="<f8")
        )
        for key in ("analytic_variance", "universal_active_superiority_claim"):
            replay[key] = in_memory[key]
        rows.append(
            {
                "fixture": "S9_SCHEMA_SHADOW",
                "logical_rows": 8192,
                "saved_array_sha256": _sha(path),
                "aggregate_exact_replay": int(replay == in_memory),
                "registered_scenario": 0,
                "scientific_result": 0,
            }
        )
        if len(digests) != 4096:
            raise DecisionContractError("shadow S9 digest coverage drifted")
    return rows


def materialize_contract_tables(repo_root: Path) -> dict[str, Any]:
    reports = repo_root / "oaci" / "reports"
    tables = reports / "c85tr1_tables"
    if tables.exists():
        raise DecisionContractError("C85TR1 table directory must be fresh")
    tables.mkdir(parents=True)
    _write_csv(
        tables / "historical_lock_supersession.csv",
        [
            {
                "object": "C85T_EXECUTION_LOCK.json",
                "commit": "9d414ebb889b2cfc3fefa19fa98d7ea5ca9fd691",
                "sha256": HISTORICAL_LOCK_SHA256,
                "authorization_record": "ABSENT",
                "authorization_consumed": 0,
                "registered_execution": 0,
                "status": "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION",
                "historical_bytes_modified": 0,
            }
        ],
    )
    _write_csv(
        tables / "S9_rng_dtype_reconciliation.csv",
        [
            {"object": "historical_protocol", "dtype": "int64", "operative_V2": 1, "disposition": "PRESERVED"},
            {"object": "historical_implementation", "dtype": "uint8", "operative_V2": 0, "disposition": "SUPERSEDED"},
            {"object": "V2_raw_draw", "dtype": "little-endian int64", "operative_V2": 1, "disposition": "LOCKED"},
            {"object": "V2_post_digest_conversion", "dtype": "int8", "operative_V2": 1, "disposition": "ALLOWED_AFTER_RAW_DIGEST"},
        ],
    )
    _write_csv(tables / "shadow_rademacher_int64_replay.csv", _shadow_replicate_replay_rows())
    _write_csv(
        tables / "monte_carlo_interval_contract_v2.csv",
        [
            {"estimand": "top_1_probability", "raw_interval": "mean+/-1.96*MC_SE", "reported_interval": "clip(raw,0,1)", "interval_clipped_field": 1, "value_clipped": 0},
            {"estimand": "outside_A_epsilon_probability", "raw_interval": "mean+/-1.96*MC_SE", "reported_interval": "clip(raw,0,1)", "interval_clipped_field": 1, "value_clipped": 0},
            {"estimand": "mean_regret", "raw_interval": "mean+/-1.96*MC_SE", "reported_interval": "raw", "interval_clipped_field": 0, "value_clipped": 0},
        ],
    )
    replicate_rows: list[dict[str, Any]] = []
    for artifact, logical_rows, fields in (
        ("S6_replicates.npz", 4096, (("replicate_id", "uint16"), ("selected_action", "uint16"), ("top1", "uint8"), ("outside_A_epsilon", "uint8"), ("selection_regret", "float64"))),
        ("S7_replicates.npz", 4096, (("replicate_id", "uint16"), ("selected_action", "uint16"), ("top1", "uint8"), ("outside_A_epsilon", "uint8"), ("selection_regret", "float64"))),
        ("S9_replicates.npz", 8192, (("replicate_id_per_design", "uint16"), ("selected_action_per_design", "uint8"), ("correct_best_per_design", "uint8"), ("top2_coverage_per_design", "uint8"), ("selection_regret_per_design", "float64"), ("D_hat_per_design", "float64"), ("paired_endpoint_arrays", "float64"))),
    ):
        for field, dtype in fields:
            replicate_rows.append({"artifact": artifact, "field": field, "dtype": dtype, "length_per_array": 4096, "logical_rows": logical_rows, "exact_reload_required": 1})
    _write_csv(tables / "replicate_artifact_schema_v2.csv", replicate_rows)
    _write_csv(tables / "aggregate_from_saved_array_replay.csv", _shadow_aggregate_replay_rows())
    _write_csv(
        tables / "authorization_single_use_contract.csv",
        [
            {"property": "schema", "value": "c85t_direct_pi_authorization_record_v2", "blocking": 1},
            {"property": "authorization_SHA", "value": "canonical record hash with self-referential ledger path normalized", "blocking": 1},
            {"property": "consumption", "value": "O_CREAT|O_EXCL", "blocking": 1},
            {"property": "output_root", "value": "exact absolute content-addressed root", "blocking": 1},
            {"property": "ledger_path", "value": "exact absolute path derived from authorization SHA-256", "blocking": 1},
            {"property": "post_failure_reuse", "value": "forbidden", "blocking": 1},
        ],
    )
    _write_csv(
        tables / "authorization_failure_truth_table.csv",
        [
            {"case": "fresh authorization fresh exact root", "consume": 1, "execute": 1, "result": "PASS"},
            {"case": "receipt already exists same root", "consume": 0, "execute": 0, "result": "FAIL"},
            {"case": "receipt already exists different root", "consume": 0, "execute": 0, "result": "FAIL"},
            {"case": "authorization output differs CLI", "consume": 0, "execute": 0, "result": "FAIL"},
            {"case": "lock SHA or commit differs", "consume": 0, "execute": 0, "result": "FAIL"},
            {"case": "protected field true", "consume": 0, "execute": 0, "result": "FAIL"},
        ],
    )
    _write_csv(
        tables / "runtime_capability_contract.csv",
        [
            {"property": "constructor", "contract": "module-private sentinel", "static_string_sufficient": 0, "blocking": 1},
            {"property": "issuance", "contract": "after atomic authorization consumption", "static_string_sufficient": 0, "blocking": 1},
            {"property": "binding", "contract": "authorization SHA|lock SHA|attempt ID|output root", "static_string_sufficient": 0, "blocking": 1},
            {"property": "cross_attempt_reuse", "contract": "fail closed", "static_string_sufficient": 0, "blocking": 1},
            {"property": "shadow_fixture", "contract": "no registered capability", "static_string_sufficient": 0, "blocking": 1},
        ],
    )
    _write_csv(
        tables / "C85T_C85V_stage_separation.csv",
        [
            {"stage": "C85T", "synthetic_execution": 1, "proof_candidates": 1, "independent_proof_review": 0, "formal_status_transition": 0, "Monte_Carlo_rerun": 1},
            {"stage": "C85V", "synthetic_execution": 0, "proof_candidates": 0, "independent_proof_review": 1, "formal_status_transition": 1, "Monte_Carlo_rerun": 0},
        ],
    )
    _write_csv(
        tables / "proof_candidate_disposition_schema.csv",
        [
            {"field": "theorem_id", "allowed": "T1..T7", "required": 1},
            {"field": "historical_status", "allowed": "OPEN", "required": 1},
            {"field": "candidate_disposition", "allowed": "PROPOSED_PROOF|PROPOSED_COUNTEREXAMPLE|INCOMPLETE_OPEN|PROPOSED_INVALIDATION", "required": 1},
            {"field": "formal_status", "allowed": "OPEN", "required": 1},
            {"field": "check_class", "allowed": "PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY", "required": 1},
        ],
    )
    _write_csv(
        tables / "lifecycle_event_schema_v2.csv",
        [
            {"sequence": index, "stage": stage, "append_only": 1, "authorization_SHA": 1, "lock_SHA": 1, "attempt_ID": 1, "artifact_SHA_when_applicable": 1, "failure_last_stage": int(stage == "FAILED")}
            for index, stage in enumerate(LIFECYCLE_EVENTS)
        ],
    )
    _write_csv(
        tables / "result_manifest_v2_contract.csv",
        [
            {"object": "scenario_results", "required_count": 11, "formal_status": "OPEN"},
            {"object": "S6_S7_logical_replicate_rows", "required_count": 8192, "formal_status": "OPEN"},
            {"object": "S9_logical_replicate_design_rows", "required_count": 8192, "formal_status": "OPEN"},
            {"object": "S9_raw_draw_digest_rows", "required_count": 4096, "formal_status": "OPEN"},
            {"object": "proof_candidates", "required_count": 7, "formal_status": "OPEN"},
        ],
    )
    _write_csv(
        tables / "risk_register.csv",
        [
            {"risk": "authorization replay", "control": "global O_EXCL receipt and exact-root binding", "residual": "external filesystem availability", "status": "CONTROLLED"},
            {"risk": "RNG byte drift", "control": "int64 raw digest and environment binding", "residual": "environment drift fails closed", "status": "CONTROLLED"},
            {"risk": "aggregate-only evidence", "control": "persist and reload every replicate array", "residual": "none within locked schema", "status": "CONTROLLED"},
            {"risk": "same-process proof approval", "control": "C85T/C85V stage separation", "residual": "future C85V quality", "status": "DEFERRED_TO_C85V"},
        ],
    )
    _write_csv(
        tables / "failure_reason_ledger.csv",
        [
            {"reason": "historical static capability", "historical_blocker": 1, "V2_control": "private consumed-authorization capability", "readiness_status": "REPAIRED"},
            {"reason": "historical uint8 S9 draws", "historical_blocker": 1, "V2_control": "little-endian int64 raw draw freeze", "readiness_status": "REPAIRED"},
            {"reason": "historical aggregate-only persistence", "historical_blocker": 1, "V2_control": "NPZ replicate arrays and exact replay", "readiness_status": "REPAIRED"},
            {"reason": "historical proof auto-transition", "historical_blocker": 1, "V2_control": "formal OPEN through C85T", "readiness_status": "REPAIRED"},
        ],
    )
    return {
        "table_count": len(list(tables.glob("*.csv"))),
        "shadow_registered_draws": 0,
        "registered_scenario_results": 0,
        "proof_candidates": 0,
        "theorem_status_transitions": 0,
    }


def _bound_paths(repo_root: Path) -> list[str]:
    reports = repo_root / "oaci" / "reports"
    historical_lock = json.loads((reports / "C85T_EXECUTION_LOCK.json").read_text())
    paths = {row["path"] for row in historical_lock["bound_repository_objects"]}
    paths.update(
        {
            "oaci/reports/C85T_EXECUTION_LOCK.json",
            "oaci/reports/C85T_EXECUTION_LOCK.sha256",
            "oaci/reports/C85TR1_EXECUTION_GUARD_RNG_PERSISTENCE_AND_PROOF_REVIEW_PROTOCOL.json",
            "oaci/reports/C85TR1_EXECUTION_GUARD_RNG_PERSISTENCE_AND_PROOF_REVIEW_PROTOCOL.sha256",
            "oaci/reports/C85TR1_PROTOCOL_TIMING_AUDIT.md",
            "oaci/theory/c85t_execution_guard.py",
            "oaci/theory/c85t_execute_v2.py",
            "oaci/theory/c85tr1_readiness.py",
            "oaci/tests/test_c85tr1_execution_guard.py",
            "oaci/tests/test_c85tr1_replicate_persistence.py",
            "oaci/tests/test_c85tr1_lock.py",
            "oaci/slurm_c85tr1_regression.sh",
        }
    )
    for path in (reports / "c85tr1_tables").glob("*.csv"):
        if path.name != "runtime_bound_object_registry.csv":
            paths.add(str(path.relative_to(repo_root)))
    missing = [relative for relative in sorted(paths) if not (repo_root / relative).is_file()]
    if missing:
        raise DecisionContractError(f"C85TR1 bound object is absent: {missing[0]}")
    return sorted(paths)


def build_execution_lock_v2(
    repo_root: Path, *, implementation_commit: str, created_at_utc: str
) -> dict[str, Any]:
    reports = repo_root / "oaci" / "reports"
    tables = reports / "c85tr1_tables"
    registry_path = tables / "runtime_bound_object_registry.csv"
    lock_path = reports / "C85T_EXECUTION_LOCK_V2.json"
    sidecar = reports / "C85T_EXECUTION_LOCK_V2.sha256"
    if any(path.exists() for path in (registry_path, lock_path, sidecar)):
        raise DecisionContractError("C85T V2 lock objects must be created once")
    if _git(repo_root, "rev-parse", "HEAD") != implementation_commit:
        raise DecisionContractError("implementation commit must equal HEAD at V2 lock build")
    if _git(repo_root, "status", "--porcelain"):
        raise DecisionContractError("V2 lock build requires a clean worktree")
    protocol_path = reports / "C85TR1_EXECUTION_GUARD_RNG_PERSISTENCE_AND_PROOF_REVIEW_PROTOCOL.json"
    historical_lock_path = reports / "C85T_EXECUTION_LOCK.json"
    if _sha(protocol_path) != PROTOCOL_SHA256:
        raise DecisionContractError("C85TR1 repair protocol hash drifted")
    if _sha(historical_lock_path) != HISTORICAL_LOCK_SHA256:
        raise DecisionContractError("historical C85T lock bytes drifted")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, implementation_commit],
        cwd=repo_root,
        check=False,
    ).returncode:
        raise DecisionContractError("C85TR1 protocol does not precede implementation")
    forbidden = (
        reports / "C85T_V2_PI_AUTHORIZATION_RECORD.json",
        reports / "C85T_RESULT.json",
        reports / "c85t_proof_candidates",
    )
    if any(path.exists() for path in forbidden):
        raise DecisionContractError("C85T V2 authorization or result exists at lock build")
    rows: list[dict[str, Any]] = []
    for relative in _bound_paths(repo_root):
        path = repo_root / relative
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
                "git_blob": _git(repo_root, "hash-object", "--", relative),
            }
        )
    _write_csv(registry_path, rows)
    registry_identity = {
        "path": str(registry_path.relative_to(repo_root)),
        "size_bytes": registry_path.stat().st_size,
        "sha256": _sha(registry_path),
        "git_blob": _git(repo_root, "hash-object", "--", str(registry_path.relative_to(repo_root))),
    }
    identities = {
        "c85p_protocol_path": "oaci/reports/C85_TPAMI_DECISION_THEORY_PROTOCOL.json",
        "c85r_repair_protocol_path": "oaci/reports/C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.json",
        "v2_generator_path": "oaci/reports/c85r_tables/synthetic_generator_contract_v2.json",
        "c85tl_operationalization_path": "oaci/reports/C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json",
        "c85tr1_repair_protocol_path": "oaci/reports/C85TR1_EXECUTION_GUARD_RNG_PERSISTENCE_AND_PROOF_REVIEW_PROTOCOL.json",
    }
    lock: dict[str, Any] = {
        "schema_version": "c85t_execution_lock_v2",
        "milestone": "C85TR1",
        "created_at_utc": created_at_utc,
        "status": LOCK_STATUS,
        "authorized": False,
        "implementation_commit": implementation_commit,
        "execution_lock_commit_binding": "DISCOVER_FROM_GIT_PATH_AND_BIND_IN_FUTURE_AUTHORIZATION",
        **identities,
        "c85p_protocol_sha256": _sha(repo_root / identities["c85p_protocol_path"]),
        "c85r_repair_protocol_sha256": _sha(repo_root / identities["c85r_repair_protocol_path"]),
        "v2_generator_sha256": _sha(repo_root / identities["v2_generator_path"]),
        "c85tl_operationalization_sha256": _sha(repo_root / identities["c85tl_operationalization_path"]),
        "c85tr1_repair_protocol_sha256": _sha(repo_root / identities["c85tr1_repair_protocol_path"]),
        "historical_execution_lock": {
            "path": "oaci/reports/C85T_EXECUTION_LOCK.json",
            "sha256": HISTORICAL_LOCK_SHA256,
            "status": "SUPERSEDED_BEFORE_AUTHORIZATION_OR_REGISTERED_EXECUTION",
        },
        "runtime_bound_object_count": len(rows),
        "runtime_bound_registry": registry_identity,
        "bound_repository_objects": rows,
        "environment": {
            "prefix": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact",
            "python": "3.13.7",
            "numpy_runtime": "2.4.4",
            "numpy_metadata_first_match": "2.3.3",
            "bit_generator": "PCG64DXSM",
            "GPU": 0,
        },
        "rng": {
            "namespace": "C85_SYNTHETIC_V1",
            "raw_S9_dtype": "<i8",
            "draw_order": "51_L_then_46_H",
            "replicates": 4096,
            "static_string_authorizes": False,
        },
        "replicate_persistence": {
            "S6_rows": 4096,
            "S7_rows": 4096,
            "S9_replicate_design_rows": 8192,
            "S9_raw_digest_rows": 4096,
            "aggregate_from_reloaded_arrays": True,
        },
        "proof_governance": {
            "C85T_formal_status": "OPEN",
            "proof_candidates": 7,
            "automatic_transition": False,
            "independent_review_stage": "C85V",
            "C85V_authorized": False,
        },
        "result": {
            "schema": "c85t_synthetic_validation_and_proof_candidates_result_v2",
            "manifest_schema": "c85t_atomic_result_manifest_v2",
            "success_gate": "C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED",
            "atomic": True,
        },
        "authorization_record_path": "oaci/reports/C85T_V2_PI_AUTHORIZATION_RECORD.json",
        "authorization_schema": "c85t_direct_pi_authorization_record_v2",
        "authorization_consumption_root": "/projects/EEG-foundation-model/yinghao/oaci-c85t-authorization-consumption-v2",
        "output_root_policy": {
            "parent": "/projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v2",
            "basename": "c85t-v2-{lock_sha16}-{authorization_id16}",
            "exact_absolute_binding_required": True,
        },
        "entrypoint": "python -m oaci.theory.c85t_execute_v2 run-locked --execution-lock <V2_LOCK> --output-root <EXACT_AUTHORIZED_ROOT>",
        "lifecycle_schema": "c85t_append_only_lifecycle_ledger_v2",
        "resources": {"CPU": 1, "GPU": 0, "RAM_GiB": 8, "wall_minutes": 30, "storage_MiB": 64},
        "readiness": {
            "registered_S0_S10_draws": 0,
            "canonical_proof_artifacts": 0,
            "theorem_status_transitions": 0,
            "authorization_records": 0,
            "success_gate": SUCCESS_GATE,
            "failure_gate": FAILURE_GATE,
        },
        "forbidden": {
            "real_project_data": True,
            "active_acquisition": True,
            "C85E": True,
            "new_data_or_model_zoo": True,
            "manuscript_work": True,
        },
    }
    lock_path.write_bytes(canonical_json_bytes(lock))
    digest = _sha(lock_path)
    sidecar.write_text(f"{digest}  {lock_path.name}\n")
    return {
        "lock_path": str(lock_path),
        "lock_sha256": digest,
        "runtime_bound_object_count": len(rows),
        "registered_S0_S10_draws": 0,
        "proof_candidates": 0,
        "theorem_status_transitions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    tables = commands.add_parser("build-contract-tables")
    tables.add_argument("--repo-root", type=Path, required=True)
    lock = commands.add_parser("build-execution-lock-v2")
    lock.add_argument("--repo-root", type=Path, required=True)
    lock.add_argument("--implementation-commit", required=True)
    lock.add_argument("--created-at-utc", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-contract-tables":
        result = materialize_contract_tables(args.repo_root.resolve())
    else:
        result = build_execution_lock_v2(
            args.repo_root.resolve(),
            implementation_commit=args.implementation_commit,
            created_at_utc=args.created_at_utc,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
