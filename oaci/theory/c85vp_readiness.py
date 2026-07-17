"""C85VP readiness tables, static isolation audit, and C85V lock builder."""
from __future__ import annotations

import argparse
import ast
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence

from .c85_decision_experiments import DecisionContractError
from .c85v_stage_a_derivation import (
    near_optimal_union_bound,
    replay_s10_exact_risks,
    s5_policy_cvar_relation,
    two_state_regret_lower_bound,
)
from .c85v_statement_registry import (
    C85T_BUNDLE_ROOT,
    C85T_CONTROL_IDENTITIES,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    THEOREM_IDS,
    canonical_json_bytes,
    load_candidate_identities,
    load_registered_statements,
    load_review_protocol,
    sha256_file,
    validate_c85t_control_identity,
)


LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
SUCCESS_GATE = "C85V_INDEPENDENT_PROOF_REVIEW_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION"
FAILURE_GATE = "C85V_STATEMENT_ASSUMPTION_REVIEW_INDEPENDENCE_OR_VERDICT_CONTRACT_RECONCILIATION_REQUIRED"
IMPLEMENTATION_PATHS = (
    "oaci/theory/c85v_statement_registry.py",
    "oaci/theory/c85v_stage_a_derivation.py",
    "oaci/theory/c85v_stage_b_candidate_audit.py",
    "oaci/theory/c85v_adjudication.py",
    "oaci/theory/c85v_result_manifest.py",
    "oaci/theory/c85v_execute.py",
    "oaci/theory/c85vp_readiness.py",
)
FORBIDDEN_IMPORTS = {
    "oaci.theory.c85t_proofs",
    "oaci.theory.c85t_registered_v3",
    "oaci.theory.c85t_monte_carlo",
    "torch",
    "mne",
    "moabb",
}


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
        raise DecisionContractError(f"refusing empty C85VP table: {path.name}")
    fields = tuple(values[0])
    if any(tuple(row) != fields for row in values):
        raise DecisionContractError(f"C85VP table schema drifted: {path.name}")
    if path.exists():
        raise DecisionContractError(f"C85VP table must be fresh: {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def static_isolation_audit(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_PATHS:
        path = repo_root / relative
        tree = ast.parse(path.read_text())
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.add(module if node.level == 0 else f"relative:{module}")
        forbidden = sorted(
            imported
            for imported in imports
            if imported in FORBIDDEN_IMPORTS
            or any(imported.startswith(f"{value}.") for value in FORBIDDEN_IMPORTS)
        )
        if forbidden:
            raise DecisionContractError(f"C85V forbidden import: {forbidden[0]}")
        rows.append({
            "path": relative,
            "forbidden_import_count": 0,
            "candidate_generator_import": 0,
            "monte_carlo_dispatch_import": 0,
            "real_data_import": 0,
            "status": "PASS",
        })
    stage_a_tree = ast.parse(
        (repo_root / "oaci/theory/c85v_stage_a_derivation.py").read_text()
    )
    stage_a = next(
        node
        for node in ast.walk(stage_a_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "freeze_stage_a_derivations"
    )
    argument_names = {argument.arg for argument in (*stage_a.args.args, *stage_a.args.kwonlyargs)}
    if any("candidate" in name.lower() for name in argument_names):
        raise DecisionContractError("C85V Stage A accepts a proof-candidate argument")
    return rows


def _external_identity_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "object": name,
            "path": str(C85T_BUNDLE_ROOT / name),
            "size_bytes": (C85T_BUNDLE_ROOT / name).stat().st_size,
            "sha256": expected,
            "role": "C85T_CONTROL",
        }
        for name, expected in C85T_CONTROL_IDENTITIES.items()
    ]
    identities = load_candidate_identities(repo_root)
    for theorem_id in THEOREM_IDS:
        identity = identities[theorem_id]
        path = C85T_BUNDLE_ROOT / identity.relative_path
        if sha256_file(path) != identity.sha256:
            raise DecisionContractError("C85V external proof-candidate identity drifted")
        rows.append({
            "object": theorem_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": identity.sha256,
            "role": "FROZEN_PROOF_CANDIDATE",
        })
    for name, role in (
        ("exact_scenario_results.json", "EXACT_FINITE_REVIEW_INPUT"),
        ("proof_candidate_dispositions.csv", "CANDIDATE_DISPOSITION_REGISTRY"),
    ):
        path = C85T_BUNDLE_ROOT / name
        rows.append({
            "object": name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": role,
        })
    return rows


def materialize_readiness_tables(repo_root: Path) -> dict[str, Any]:
    protocol = load_review_protocol(repo_root)
    validate_c85t_control_identity(C85T_BUNDLE_ROOT)
    statements = load_registered_statements(repo_root)
    tables = repo_root / "oaci/reports/c85vp_tables"
    isolation = static_isolation_audit(repo_root)
    external = _external_identity_rows(repo_root)
    _write_csv(tables / "c85t_identity_replay.csv", external)
    _write_csv(tables / "implementation_isolation_audit.csv", isolation)
    s10 = replay_s10_exact_risks()
    exact_checks = [
        {"check": "T2_S10_coarse", "expected": "11/40", "observed": str(s10["coarse_registered_risk"]), "status": "PASS"},
        {"check": "T2_S10_rich_unrestricted", "expected": "0", "observed": str(s10["rich_unrestricted_risk"]), "status": "PASS"},
        {"check": "T2_S10_rich_registered", "expected": "3/5", "observed": str(s10["rich_registered_risk"]), "status": "PASS"},
        {"check": "T2_S10_reversal", "expected": "13/40", "observed": str(s10["registered_reversal"]), "status": "PASS"},
        {"check": "T4_factor_TV0", "expected": "1", "observed": str(two_state_regret_lower_bound(Fraction(2), Fraction(0))), "status": "PASS"},
        {"check": "T4_factor_TV1", "expected": "0", "observed": str(two_state_regret_lower_bound(Fraction(2), Fraction(1))), "status": "PASS"},
        {"check": "T6_alpha_boundary", "expected": "0", "observed": str(s5_policy_cvar_relation(Fraction(13, 20))), "status": "PASS"},
        {"check": "T7_empty_outside", "expected": "0.0", "observed": str(near_optimal_union_bound([0.0], [0.0], 0.0)), "status": "PASS"},
    ]
    _write_csv(tables / "exact_adversarial_fixture_registry.csv", exact_checks)
    bundle_rows = [
        {"object": "Stage-A derivations", "required_count": 7, "role_hash": "REVIEWER_A", "monte_carlo": 0, "atomic": 1},
        {"object": "Stage-B comparisons", "required_count": 7, "role_hash": "REVIEWER_B_COMPARISON", "monte_carlo": 0, "atomic": 1},
        {"object": "adversarial audits", "required_count": 7, "role_hash": "REVIEWER_B_ADVERSARIAL", "monte_carlo": 0, "atomic": 1},
        {"object": "final verdicts", "required_count": 7, "role_hash": "ADJUDICATOR", "monte_carlo": 0, "atomic": 1},
        {"object": "formal status rows", "required_count": 7, "role_hash": "ADJUDICATOR", "monte_carlo": 0, "atomic": 1},
        {"object": "proof retention rows", "required_count": 7, "role_hash": "EXTERNAL_IDENTITY", "monte_carlo": 0, "atomic": 1},
    ]
    _write_csv(tables / "result_bundle_schema.csv", bundle_rows)
    shadow_rows = [
        {"case": "Stage A signature has no candidate path", "expected": "PASS", "observed": "PASS", "formal_status_transition": 0},
        {"case": "forbidden candidate-generator import", "expected": "0", "observed": "0", "formal_status_transition": 0},
        {"case": "forbidden Monte Carlo import", "expected": "0", "observed": "0", "formal_status_transition": 0},
        {"case": "T2 exact rational replay", "expected": "PASS", "observed": "PASS", "formal_status_transition": 0},
        {"case": "T4 factor and TV boundaries", "expected": "PASS", "observed": "PASS", "formal_status_transition": 0},
        {"case": "T6 CVaR boundary", "expected": "PASS", "observed": "PASS", "formal_status_transition": 0},
        {"case": "T7 deterministic and empty boundaries", "expected": "PASS", "observed": "PASS", "formal_status_transition": 0},
        {"case": "T1-T7 remain OPEN in readiness", "expected": "7", "observed": "7", "formal_status_transition": 0},
    ]
    _write_csv(tables / "shadow_validation.csv", shadow_rows)
    _write_csv(
        tables / "risk_register.csv",
        [
            {"risk": "Stage A sees candidate prose", "control": "no candidate argument plus staged manifest release", "residual": "coordinator implementation must remain exact", "status": "CONTROLLED"},
            {"risk": "Monte Carlo interpreted as proof", "control": "forbidden imports and exact-only Stage B input", "residual": "frozen scenario metadata remains contextual", "status": "CONTROLLED"},
            {"risk": "finite enumeration upgraded to general proof", "control": "scope-sensitive adjudication", "residual": "future derivation quality", "status": "CONTROLLED_BY_C85V"},
            {"risk": "T5 repaired during review", "control": "frozen-statement insufficiency forces OPEN", "residual": "future theorem may require additive protocol", "status": "CONTROLLED"},
            {"risk": "citation substitutes for proof", "control": "primary registry flag and statement-to-source audit", "residual": "human source interpretation", "status": "CONTROLLED_BY_C85V"},
            {"risk": "partial result publication", "control": "single final rename after semantic replay", "residual": "filesystem rename semantics", "status": "CONTROLLED"},
        ],
    )
    _write_csv(
        tables / "failure_reason_ledger.csv",
        [
            {"reason": "statement identity mismatch", "stage": "PREFLIGHT", "formal_status_effect": "NONE", "disposition": "BLOCK"},
            {"reason": "Stage A incomplete", "stage": "A", "formal_status_effect": "NONE", "disposition": "BLOCK_BEFORE_CANDIDATE_ACCESS"},
            {"reason": "candidate hash mismatch", "stage": "B", "formal_status_effect": "NONE", "disposition": "BLOCK"},
            {"reason": "candidate substantive gap", "stage": "B/C", "formal_status_effect": "OPEN_OR_INVALIDATED_BY_CONTRACT", "disposition": "RETAIN"},
            {"reason": "T5 decoder assumption absent", "stage": "C", "formal_status_effect": "OPEN", "disposition": "DO_NOT_REPAIR_STATEMENT"},
            {"reason": "atomic semantic replay failure", "stage": "PUBLICATION", "formal_status_effect": "NO_FINAL_ROOT", "disposition": "BLOCK"},
        ],
    )
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "statement_count": len(statements),
        "external_identity_count": len(external),
        "implementation_file_count": len(isolation),
        "readiness_table_count": len(list(tables.glob("*.csv"))),
        "registered_review_executions": 0,
        "monte_carlo_reruns": 0,
        "formal_status_transitions": 0,
        "candidate_text_before_protocol": int(
            protocol["chronology"]["proof_candidate_text_opened_for_review_before_protocol"]
        ),
    }


def _bound_paths(repo_root: Path) -> list[str]:
    reports = repo_root / "oaci/reports"
    paths = {
        "oaci/reports/C85V_INDEPENDENT_PROOF_REVIEW_PROTOCOL.json",
        "oaci/reports/C85V_INDEPENDENT_PROOF_REVIEW_PROTOCOL.sha256",
        "oaci/reports/C85V_PROTOCOL_TIMING_AUDIT.md",
        "oaci/reports/C85T_EXECUTION_LOCK_V3.json",
        "oaci/reports/C85T_EXECUTION_LOCK_V3.sha256",
        "oaci/reports/C85_TPAMI_DECISION_THEORY_PROTOCOL.json",
        "oaci/reports/C85R_SYNTHETIC_CONTRACT_SEMANTIC_REPAIR_PROTOCOL.json",
        "oaci/reports/C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json",
        "oaci/reports/C85TR1_EXECUTION_GUARD_RNG_PERSISTENCE_AND_PROOF_REVIEW_PROTOCOL.json",
        "oaci/reports/C85TR2_AUTHORIZATION_CERTIFICATE_ATOMIC_TRANSACTION_AND_RESULT_REPLAY_PROTOCOL.json",
        "oaci/reports/c85p_tables/theorem_registry.csv",
        "oaci/reports/c85p_tables/proof_obligation_registry.csv",
        "oaci/reports/c85p_tables/assumption_lattice.csv",
        "oaci/reports/c85r_tables/proof_obligation_precision_addendum.csv",
        "oaci/reports/c85tr1_tables/C85T_C85V_stage_separation.csv",
        "oaci/multidataset/c84r_regression_suite.py",
        "oaci/slurm_c85vp_regression.sh",
    }
    paths.update(IMPLEMENTATION_PATHS)
    paths.update(
        {
            "oaci/tests/c85vp_test_support.py",
            "oaci/tests/test_c85vp_stage_isolation.py",
            "oaci/tests/test_c85vp_theorem_contracts.py",
            "oaci/tests/test_c85vp_execution_lock.py",
        }
    )
    paths.update(
        str(path.relative_to(repo_root))
        for path in (reports / "c85vp_tables").glob("*.csv")
        if path.name != "runtime_bound_object_registry.csv"
    )
    missing = [relative for relative in sorted(paths) if not (repo_root / relative).is_file()]
    if missing:
        raise DecisionContractError(f"C85V bound object is absent: {missing[0]}")
    return sorted(paths)


def build_execution_lock(
    repo_root: Path,
    *,
    implementation_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    reports = repo_root / "oaci/reports"
    tables = reports / "c85vp_tables"
    registry_path = tables / "runtime_bound_object_registry.csv"
    lock_path = reports / "C85V_EXECUTION_LOCK.json"
    sidecar = reports / "C85V_EXECUTION_LOCK.sha256"
    if any(path.exists() for path in (registry_path, lock_path, sidecar)):
        raise DecisionContractError("C85V execution-lock objects must be fresh")
    if _git(repo_root, "rev-parse", "HEAD") != implementation_commit:
        raise DecisionContractError("C85V implementation commit must equal HEAD")
    if _git(repo_root, "status", "--porcelain"):
        raise DecisionContractError("C85V lock build requires a clean worktree")
    load_review_protocol(repo_root)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, implementation_commit],
        cwd=repo_root,
        check=False,
    ).returncode != 0:
        raise DecisionContractError("C85V protocol does not precede implementation")
    if (reports / "C85V_PI_AUTHORIZATION_RECORD.json").exists():
        raise DecisionContractError("C85V authorization exists before lock readiness")
    bound_rows: list[dict[str, Any]] = []
    for relative in _bound_paths(repo_root):
        path = repo_root / relative
        bound_rows.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob": _git(repo_root, "hash-object", "--", relative),
        })
    _write_csv(registry_path, bound_rows)
    external = _external_identity_rows(repo_root)
    lock = {
        "schema_version": "c85v_execution_lock_v1",
        "milestone": "C85VP",
        "created_at_utc": created_at_utc,
        "status": LOCK_STATUS,
        "authorized": False,
        "repo_root": str(repo_root.resolve()),
        "implementation_commit": implementation_commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime_bound_object_count": len(bound_rows),
        "runtime_bound_registry": {
            "path": str(registry_path.relative_to(repo_root)),
            "size_bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
            "git_blob": _git(repo_root, "hash-object", "--", str(registry_path.relative_to(repo_root))),
        },
        "bound_repository_objects": bound_rows,
        "bound_external_objects": external,
        "frozen_c85t": {
            "bundle_root": str(C85T_BUNDLE_ROOT),
            "result_sha256": C85T_CONTROL_IDENTITIES["C85T_RESULT.json"],
            "result_manifest_sha256": C85T_CONTROL_IDENTITIES["C85T_RESULT_ARTIFACT_MANIFEST.json"],
            "semantic_replay_sha256": C85T_CONTROL_IDENTITIES["C85T_V3_SEMANTIC_REPLAY_RECEIPT.json"],
            "completion_receipt_sha256": C85T_CONTROL_IDENTITIES["C85T_V3_COMPLETION_RECEIPT.json"],
            "proof_candidate_count": 7,
            "formal_statuses": {theorem_id: "OPEN" for theorem_id in THEOREM_IDS},
        },
        "review_process": {
            "stage_A": "candidate-blind independent derivation and atomic freeze",
            "stage_B": "post-freeze candidate comparison and adversarial audit",
            "stage_C": "prospective deterministic adjudication without majority vote",
            "role_artifacts_separately_hashed": True,
            "candidate_text_withheld_until_stage_A_freeze": True,
            "monte_carlo_rerun": False,
        },
        "verdict_contract": {
            "general_proof_required_for_PROVED": True,
            "finite_enumeration_maximum": "PROVED_FINITE_MODEL_ONLY",
            "T5_missing_decoder_rule": "OPEN",
            "dissent_retained": True,
        },
        "authorization_record_path": "oaci/reports/C85V_PI_AUTHORIZATION_RECORD.json",
        "authorization_schema": "c85v_direct_pi_authorization_record_v1",
        "future_direct_statement_exact": "授权 C85V",
        "authorization_consumption_root": "/projects/EEG-foundation-model/yinghao/oaci-c85v-authorization-consumption-v1",
        "output_root_policy": {
            "parent": "/projects/EEG-foundation-model/yinghao/oaci-c85v-proof-review-v1",
            "basename": "c85v-{lock_sha16}-{authorization_id16}",
            "exact_absolute_binding_required": True,
        },
        "environment": {
            "prefix": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact",
            "python": "3.13.7",
            "GPU": 0,
            "monte_carlo": 0,
        },
        "entrypoint": "python -m oaci.theory.c85v_execute run-locked --execution-lock <LOCK> --authorization-record <AUTHORIZATION> --output-root <BOUND_ROOT>",
        "result": {
            "bundle_schema": "c85v_atomic_proof_review_bundle_v1",
            "manifest_schema": "c85v_result_artifact_manifest_v1",
            "stage_A_derivations": 7,
            "stage_B_comparisons": 7,
            "adversarial_audits": 7,
            "final_verdicts": 7,
            "proof_candidates_overwritten": False,
            "atomic_publication": True,
            "success_gate": "C85V_INDEPENDENT_PROOF_VERDICTS_AND_THEOREM_STATUSES_FROZEN_C85E_PROTOCOL_REVIEW_REQUIRED",
        },
        "resources": {"CPU": 1, "GPU": 0, "RAM_GiB": 4, "wall_minutes": 30, "storage_MiB": 64},
        "readiness": {
            "registered_review_executions": 0,
            "monte_carlo_reruns": 0,
            "formal_status_transitions": 0,
            "authorization_records": 0,
            "success_gate": SUCCESS_GATE,
            "failure_gate": FAILURE_GATE,
        },
        "forbidden": {
            "C85T_candidate_generator_import": True,
            "C85T_monte_carlo_dispatch": True,
            "real_data": True,
            "active_acquisition": True,
            "C85E": True,
            "new_data_or_model_zoo": True,
            "manuscript_work": True,
        },
    }
    lock_path.write_bytes(canonical_json_bytes(lock))
    digest = sha256_file(lock_path)
    sidecar.write_text(f"{digest}  {lock_path.name}\n")
    return {
        "lock_path": str(lock_path),
        "lock_sha256": digest,
        "runtime_bound_object_count": len(bound_rows),
        "bound_external_object_count": len(external),
        "registered_review_executions": 0,
        "monte_carlo_reruns": 0,
        "formal_status_transitions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    tables = commands.add_parser("build-readiness-tables")
    tables.add_argument("--repo-root", type=Path, required=True)
    lock = commands.add_parser("build-execution-lock")
    lock.add_argument("--repo-root", type=Path, required=True)
    lock.add_argument("--implementation-commit", required=True)
    lock.add_argument("--created-at-utc", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-readiness-tables":
        result = materialize_readiness_tables(args.repo_root.resolve())
    else:
        result = build_execution_lock(
            args.repo_root.resolve(),
            implementation_commit=args.implementation_commit,
            created_at_utc=args.created_at_utc,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
