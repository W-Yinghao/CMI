"""Shadow-only helpers for C85TR2 governance tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

import numpy as np

from oaci.theory.c85t_execution_context_v3 import (
    authorization_binding_sha256,
    canonical_json_bytes,
    expected_consumption_path,
    expected_output_root,
    sha256_file,
)
from oaci.theory.c85t_exact_scenarios import near_optimal_geometry
from oaci.theory.c85t_monte_carlo import (
    _summarize_s9_arrays_v2,
    summarize_near_replicates_v2,
)
from oaci.theory.c85t_proofs import PROOF_FILENAMES
from oaci.theory.c85t_semantic_replay_v3 import RESULT_SCHEMA_V3, SUCCESS_GATE_V3


def _run(root: Path, *args: str) -> str:
    return subprocess.run(
        list(args), cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _run(root, "git", *args)


def create_shadow_authorized_repository(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _run(tmp_path, "git", "init", "--bare", str(remote))
    _run(repo, "git", "init", "-b", "oaci")
    _git(repo, "config", "user.name", "C85TR2 Shadow")
    _git(repo, "config", "user.email", "shadow@example.invalid")
    (repo / "oaci/reports").mkdir(parents=True)
    (repo / "oaci/theory").mkdir(parents=True)
    implementation = repo / "oaci/theory/shadow_v3_runtime.py"
    implementation.write_text("SHADOW_ONLY = True\n")
    protocol = repo / "oaci/reports/SHADOW_PROTOCOL.json"
    protocol.write_bytes(canonical_json_bytes({"schema_version": "shadow_protocol_v1"}))
    generator = repo / "oaci/reports/SHADOW_GENERATOR.json"
    generator.write_bytes(canonical_json_bytes({"schema_version": "shadow_generator_v1"}))
    operationalization = repo / "oaci/reports/SHADOW_OPERATIONALIZATION.json"
    operationalization.write_bytes(
        canonical_json_bytes(
            {"proof_statements": {f"T{i}": f"Shadow theorem statement T{i}." for i in range(1, 8)}}
        )
    )
    bound_paths = (
        "oaci/theory/shadow_v3_runtime.py",
        "oaci/reports/SHADOW_PROTOCOL.json",
        "oaci/reports/SHADOW_GENERATOR.json",
        "oaci/reports/SHADOW_OPERATIONALIZATION.json",
    )
    registry = repo / "oaci/reports/SHADOW_RUNTIME_REGISTRY.csv"
    with registry.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "size_bytes", "sha256", "git_blob")
        )
        writer.writeheader()
        for relative in bound_paths:
            path = repo / relative
            writer.writerow(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "git_blob": _git(repo, "hash-object", "--", relative),
                }
            )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add shadow C85TR2 runtime")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    bound_rows = []
    for relative in bound_paths:
        path = repo / relative
        bound_rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "git_blob": _git(repo, "hash-object", "--", relative),
            }
        )
    registry_identity = {
        "path": "oaci/reports/SHADOW_RUNTIME_REGISTRY.csv",
        "size_bytes": registry.stat().st_size,
        "sha256": sha256_file(registry),
        "git_blob": _git(repo, "hash-object", "--", "oaci/reports/SHADOW_RUNTIME_REGISTRY.csv"),
    }
    output_parent = tmp_path / "outputs"
    consumption_root = tmp_path / "consumption"
    lock = {
        "schema_version": "c85t_execution_lock_v3",
        "milestone": "C85TR2_SHADOW",
        "status": "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED",
        "authorized": False,
        "execution_scope": "SHADOW_READINESS_ONLY",
        "implementation_commit": implementation_commit,
        "protocol_identities": [
            {
                "path": "oaci/reports/SHADOW_PROTOCOL.json",
                "sha256": sha256_file(protocol),
            }
        ],
        "runtime_bound_object_count": len(bound_rows),
        "runtime_bound_registry": registry_identity,
        "bound_repository_objects": bound_rows,
        "environment": {"enforce_exact": False},
        "authorization_record_path": "oaci/reports/C85T_V3_PI_AUTHORIZATION_RECORD.json",
        "authorization_consumption_root": str(consumption_root.resolve()),
        "output_root_policy": {
            "parent": str(output_parent.resolve()),
            "basename": "c85t-v3-{lock_sha16}-{authorization_id16}",
        },
        "v2_generator_path": "oaci/reports/SHADOW_GENERATOR.json",
        "c85tl_operationalization_path": "oaci/reports/SHADOW_OPERATIONALIZATION.json",
    }
    lock_path = repo / "oaci/reports/C85T_EXECUTION_LOCK_V3.json"
    lock_path.write_bytes(canonical_json_bytes(lock))
    lock_sha = sha256_file(lock_path)
    lock_path.with_suffix(".sha256").write_text(f"{lock_sha}  {lock_path.name}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Lock shadow C85T V3 execution")
    lock_commit = _git(repo, "rev-parse", "HEAD")
    authorization_id = str(uuid4())
    output_root = expected_output_root(output_parent, lock_sha, authorization_id)
    record = {
        "schema_version": "c85t_direct_pi_authorization_record_v3",
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": "\u6388\u6743 C85T",
        "authorized_stage": "C85T",
        "authorization_id": authorization_id,
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "output_root": str(output_root),
        "consumption_ledger_path": "PLACEHOLDER",
        "C85V": False,
        "C85E": False,
        "active_acquisition": False,
        "real_data": False,
        "new_data_or_model_zoo": False,
        "manuscript": False,
    }
    binding_sha = authorization_binding_sha256(record)
    record["consumption_ledger_path"] = str(
        expected_consumption_path(consumption_root, binding_sha)
    )
    authorization_path = repo / lock["authorization_record_path"]
    authorization_path.write_bytes(canonical_json_bytes(record))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Authorize shadow C85T V3 governance fixture")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "oaci")
    return {
        "repo": repo,
        "lock_path": lock_path,
        "authorization_path": authorization_path,
        "output_root": output_root,
        "external_receipt_path": Path(record["consumption_ledger_path"]),
        "lock_sha": lock_sha,
        "lock_commit": lock_commit,
        "binding_sha": binding_sha,
        "authorization_id": authorization_id,
        "statements": json.loads(operationalization.read_text())["proof_statements"],
    }


def shadow_contract() -> dict[str, Any]:
    scenarios = []
    for index in range(11):
        row: dict[str, Any] = {"id": f"S{index}", "shadow_fixture": True}
        if index in (6, 7):
            row.update(
                {
                    "utilities": [[1.0, 0.5]],
                    "epsilon": 0.1,
                    "tau": 0.2,
                    "pairwise_sigma": 0.3,
                }
            )
        if index == 9:
            row["population_mean_losses"] = [0.3, 0.35, 0.65, 0.85]
        scenarios.append(row)
    return {"schema_version": "c85tr2_shadow_contract_v1", "scenarios": scenarios}


def shadow_exact() -> dict[str, Any]:
    geometry = near_optimal_geometry((1.0, 0.5), 0.1, 0.2, 0.3)
    value = {
        "S0": {"optimal_constant_risk": "0"},
        "S1": {
            "coarse_registered_risk": "0",
            "rich_unrestricted_risk": "0",
            "rich_registered_risk": "1",
        },
        "S2": {"action_divergence": "0", "registered_risk": "0", "reference_risk": "0"},
        "S3": {"selected_action": 0, "regret": 0.0, "spearman": 1.0},
        "S4": {"top4_localization": 1, "selected_regret": "0"},
        "S5": {
            "candidate_open_lower": "13/20",
            "candidate_open_upper": "1",
            "endpoint_policy": "both endpoints excluded",
            "status": "SHADOW_SCHEMA_ONLY",
        },
        "S6": geometry,
        "S7": geometry,
        "S8": {
            "identified_set_infinity_diameter": "1",
            "optimal_randomized_action_distribution": ["1/2", "1/2"],
            "minimax_regret": "1/4",
            "extreme_point_constraint_slacks": ["0", "0"],
            "active_constraints": ["shadow"],
            "pure_action_minimax_regret": "1/2",
            "randomization_gain": "1/4",
        },
        "S9": {
            "population_means": ["3/10", "7/20", "13/20", "17/20"],
            "passive_allocation": [51, 13],
            "neyman_allocation": [18, 46],
            "passive_analytic_variance": "1/1000",
            "neyman_analytic_variance": "1/2000",
        },
        "S10": {
            "coarse_policy": [1, 1],
            "coarse_risk": "11/40",
            "historical_rich_risk": "11/40",
            "rich_unrestricted_risk": "0",
            "v2_rich_risk": "3/5",
            "rich_gap": "3/5",
            "reversal": "13/40",
        },
    }
    return json.loads(canonical_json_bytes(value))


def shadow_near_arrays() -> dict[str, np.ndarray]:
    return {
        "replicate_id": np.arange(4096, dtype="<u2"),
        "selected_action": np.zeros(4096, dtype="<u2"),
        "top1": np.ones(4096, dtype=np.uint8),
        "outside_A_epsilon": np.zeros(4096, dtype=np.uint8),
        "selection_regret": np.zeros(4096, dtype="<f8"),
    }


def shadow_s9_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for design in ("passive", "neyman"):
        arrays[f"{design}_replicate_id"] = np.arange(4096, dtype="<u2")
        arrays[f"{design}_selected_action"] = np.zeros(4096, dtype=np.uint8)
        arrays[f"{design}_correct_best"] = np.ones(4096, dtype=np.uint8)
        arrays[f"{design}_top2_coverage"] = np.ones(4096, dtype=np.uint8)
        arrays[f"{design}_selection_regret"] = np.zeros(4096, dtype="<f8")
        arrays[f"{design}_D_hat"] = np.full(4096, 0.05, dtype="<f8")
    for endpoint in ("selection_regret", "correct_best", "top2_coverage", "D_hat"):
        arrays[f"paired_passive_minus_neyman_{endpoint}"] = np.zeros(4096, dtype="<f8")
    return arrays


def shadow_digest_rows() -> list[dict[str, Any]]:
    rows = []
    for replicate in range(4096):
        low = hashlib.sha256(f"shadow-L-{replicate}".encode()).hexdigest()
        high = hashlib.sha256(f"shadow-H-{replicate}".encode()).hexdigest()
        rows.append(
            {
                "replicate_id": replicate,
                "L_sha256": low,
                "H_sha256": high,
                "combined_sha256": hashlib.sha256((low + high).encode()).hexdigest(),
                "dtype": "<i8",
                "L_count": 51,
                "H_count": 46,
            }
        )
    return rows


def populate_shadow_bundle(bundle: Any, context: Any, statements: dict[str, str]) -> dict[str, Any]:
    exact = shadow_exact()
    contract = shadow_contract()
    context._lifecycle.append("EXACT_SCENARIOS_STARTED")
    exact_path = bundle.write_json("exact_scenario_results.json", exact)
    context._lifecycle.append(
        "EXACT_SCENARIOS_COMPLETED", artifact_or_receipt_sha256=sha256_file(exact_path)
    )
    context._lifecycle.append("MONTE_CARLO_STARTED")
    near = shadow_near_arrays()
    geometry = exact["S6"]
    summaries = {
        "S6": summarize_near_replicates_v2("S6", near, geometry),
        "S7": summarize_near_replicates_v2("S7", near, geometry),
    }
    bundle.write_npz("S6_replicates.npz", near)
    bundle.write_npz("S7_replicates.npz", near)
    s9 = shadow_s9_arrays()
    bundle.write_npz("S9_replicates.npz", s9)
    population = np.asarray([0.3, 0.35, 0.65, 0.85], dtype="<f8")
    summaries["S9"] = _summarize_s9_arrays_v2(s9, population)
    summaries["S9"]["analytic_variance"] = {
        "passive_d_hat_variance": 0.001,
        "neyman_d_hat_variance": 0.0005,
    }
    summaries["S9"]["universal_active_superiority_claim"] = False
    summaries["S9_population_mean_losses"] = population.tolist()
    summary_path = bundle.write_json("monte_carlo_summary.json", summaries)
    rows = shadow_digest_rows()
    digest_path = bundle.path("S9_raw_draw_digest_registry.csv")
    with digest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    context._lifecycle.append(
        "MONTE_CARLO_COMPLETED", artifact_or_receipt_sha256=sha256_file(summary_path)
    )
    context._lifecycle.append("PROOF_CANDIDATES_STARTED")
    proof_dir = bundle.path("c85t_proof_candidates")
    proof_dir.mkdir()
    dispositions = []
    for theorem_id, filename in PROOF_FILENAMES.items():
        statement_sha = hashlib.sha256(statements[theorem_id].encode()).hexdigest()
        text = (
            f"# {theorem_id} Proof Candidate\n\n"
            f"## Exact Statement\n\n{statements[theorem_id]}\n\n"
            f"Statement SHA-256: `{statement_sha}`\n\n"
            "## Assumptions\n\n- Shadow assumption.\n\n"
            "## Proof Candidate Or Counterexample\n\nShadow schema fixture.\n\n"
            "## Boundary Cases\n\n- Shadow boundary.\n\n"
            "## Candidate Disposition\n\n`INCOMPLETE_OPEN`\n\n"
            "## Proof Candidate Schema And Internal Consistency\n\n"
            "Check class: `PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY`\n\n"
            "This is not an independent proof review and cannot transition theorem status.\n\n"
            "## Formal Status\n\n`OPEN`\n"
        )
        path = proof_dir / filename
        path.write_text(text)
        dispositions.append(
            {
                "theorem_id": theorem_id,
                "historical_status": "OPEN",
                "candidate_disposition": "INCOMPLETE_OPEN",
                "formal_status": "OPEN",
                "check_class": "PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY",
                "proof_candidate_sha256": sha256_file(path),
            }
        )
    disposition_path = bundle.path("proof_candidate_dispositions.csv")
    with disposition_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(dispositions[0]))
        writer.writeheader()
        writer.writerows(dispositions)
    context._lifecycle.append(
        "PROOF_CANDIDATES_COMPLETED",
        artifact_or_receipt_sha256=sha256_file(disposition_path),
    )
    result = {
        "schema_version": RESULT_SCHEMA_V3,
        "final_gate": SUCCESS_GATE_V3,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_id": context.authorization_id,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "HEAD": context.head,
        "scenario_count": 11,
        "S6_S7_logical_replicate_rows": 8192,
        "S9_logical_replicate_design_rows": 8192,
        "S9_raw_draw_digest_rows": 4096,
        "proof_candidate_count": 7,
        "formal_theorem_statuses": {f"T{i}": "OPEN" for i in range(1, 8)},
        "real_project_data_access": 0,
        "active_acquisition": 0,
        "C85V_authorized": False,
        "C85E_authorized": False,
        "manuscript_modified": False,
    }
    return {
        "contract": contract,
        "exact": exact,
        "s9_arrays": s9,
        "digest_rows": rows,
        "result": result,
    }
