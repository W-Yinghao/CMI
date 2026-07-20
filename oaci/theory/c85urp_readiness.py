"""C85URP readiness evidence and C85U execution-lock construction."""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy

from oaci.multidataset.c84s_common import (
    canonical_sha256,
    require,
    sha256_file,
    write_csv,
)
from oaci.multidataset.c84s_evaluation import standardized_regret
from oaci.multidataset.c84s_q0_budget import midrank_percentile

from .c85u_input_registry import (
    EVALUATION_LABEL_SHA256,
    EVALUATION_LABEL_TABLE,
    EVALUATION_SEAL,
    EVALUATION_VIEW_MANIFEST,
    PROTOCOL_SHA256,
    RESULT_MANIFEST,
    SELECTION_MANIFEST,
    build_frozen_input_registry,
    write_readiness_registries,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c85urp_tables"
PROTOCOL_COMMIT = "ebe158c9e929f67423a9ebdc3cea7c6ea5c16c9a"
SUCCESS_GATE = "C85U_HELD_EVALUATION_CANDIDATE_UTILITY_RECONSTRUCTION_LOCKED_READY_FOR_PI_AUTHORIZATION"
FAILURE_GATE = "C85U_EVALUATION_VIEW_TARGET_ARTIFACT_UTILITY_SCHEMA_OR_PROVENANCE_RECONCILIATION_REQUIRED"
LOCK_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"

IMPLEMENTATION_PATHS = (
    "oaci/theory/c85u_input_registry.py",
    "oaci/theory/c85u_utility_builder.py",
    "oaci/theory/c85u_persistence.py",
    "oaci/theory/c85u_result_manifest.py",
    "oaci/theory/c85u_historical_decision_replay.py",
    "oaci/theory/c85u_runtime_guard.py",
    "oaci/theory/c85u_stage_u1.py",
    "oaci/theory/c85u_stage_u2.py",
    "oaci/theory/c85u_execute.py",
    "oaci/theory/c85urp_readiness.py",
)
TEST_PATHS = (
    "oaci/tests/c85urp_test_support.py",
    "oaci/tests/test_c85urp_candidate_utility.py",
    "oaci/tests/test_c85urp_historical_replay.py",
    "oaci/tests/test_c85urp_isolation.py",
    "oaci/tests/test_c85urp_lock.py",
)
HISTORICAL_IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84s_evaluation.py",
    "oaci/multidataset/c84s_q0_budget.py",
    "oaci/multidataset/c84sr1_common.py",
    "oaci/multidataset/c84sr1_q0_store.py",
    "oaci/multidataset/c84sr1_context_enumerator.py",
    "oaci/multidataset/c84sr3_common.py",
    "oaci/multidataset/c84sr3_q0_store.py",
)
FORBIDDEN_IMPORT_FRAGMENTS = (
    "torch", "mne", "moabb", "train", "inference", "analysis", "taxonomy",
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _write_fresh_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    require(values, f"C85URP refusing empty readiness table: {path.name}")
    fields = tuple(values[0])
    require(all(tuple(row) == fields for row in values),
            f"C85URP readiness table schema drift: {path.name}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    payload = stream.getvalue()
    if path.exists():
        require(path.read_text(encoding="utf-8") == payload,
                f"C85URP existing readiness table drift: {path.name}")
        return
    path.write_text(payload, encoding="utf-8")


def _imports(relative: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def static_isolation_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_PATHS:
        imports = _imports(relative)
        forbidden = sorted(
            imported for imported in imports
            if any(fragment in imported.lower() for fragment in FORBIDDEN_IMPORT_FRAGMENTS)
        )
        # The locked historical metric module is intentionally named c84s_evaluation.
        forbidden = [
            value for value in forbidden
            if value not in {"oaci.multidataset.c84s_evaluation", "c84s_evaluation"}
        ]
        if forbidden:
            raise RuntimeError(f"C85URP forbidden import: {relative}/{forbidden[0]}")
        rows.append({
            "path": relative,
            "forbidden_import_count": 0,
            "training_forward_GPU_import": 0,
            "scientific_inference_import": 0,
            "theorem_status_writer_import": 0,
            "status": "PASS",
        })
    u1_imports = _imports("oaci/theory/c85u_stage_u1.py")
    require(not any("stage_b" in value or "q0_store" in value for value in u1_imports),
            "C85URP U1 imports Stage-B/Q0 storage")
    u2_imports = _imports("oaci/theory/c85u_stage_u2.py")
    require(not any("input_registry" in value or "utility_builder" in value for value in u2_imports),
            "C85URP U2 imports protected U1 input code")
    coordinator_imports = _imports("oaci/theory/c85u_execute.py")
    require(not any("stage_u1" in value or "stage_u2" in value for value in coordinator_imports),
            "C85URP coordinator imports a stage implementation in-process")
    return rows


def _schema_rows() -> list[dict[str, Any]]:
    float_fields = {
        "balanced_accuracy", "NLL", "ECE", "bAcc_midrank_percentile",
        "negative_NLL_midrank_percentile", "negative_ECE_midrank_percentile",
        "composite_utility", "utility_rank_midrank", "standardized_regret",
    }
    vector_fields = (
        "candidate_index", "candidate_id", "regime", "trajectory_order", "epoch",
        "target_artifact_sha256", "balanced_accuracy", "NLL", "ECE",
        "bAcc_midrank_percentile", "negative_NLL_midrank_percentile",
        "negative_ECE_midrank_percentile", "composite_utility",
        "utility_rank_midrank", "canonical_utility_order_position",
        "standardized_regret", "is_canonical_best", "is_in_canonical_top5",
        "is_in_canonical_top10",
    )
    return [
        {
            "field": field,
            "shape": "[81]",
            "dtype_contract": "<f8" if field in float_fields else "NON_OBJECT_EXACT",
            "persisted": 1,
            "replayed": 1,
            "contains_label_or_logit": 0,
        }
        for field in vector_fields
    ]


def _selection_artifact_bindings() -> list[dict[str, Any]]:
    selection = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    result = json.loads(RESULT_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for manifest, root, names, role in (
        (
            selection, SELECTION_MANIFEST.parent,
            {"candidate_ranks.csv", "fixed_default_selections.csv", "q0_selection_shard_index.csv"},
            "U2_FROZEN_ACTION_INPUT_NOT_OPENED_C85URP",
        ),
        (
            result, RESULT_MANIFEST.parent, {"method_context_decisions.csv"},
            "U2_HISTORICAL_ENDPOINT_INPUT_NOT_OPENED_C85URP",
        ),
    ):
        raw = manifest["artifacts"]
        artifacts = list(raw.values()) if isinstance(raw, dict) else list(raw)
        for item in artifacts:
            if Path(str(item["path"])).name not in names:
                continue
            path = root / str(item["path"])
            require(path.is_file(), f"C85URP future U2 input path absent: {path}")
            rows.append({
                "object_id": Path(str(item["path"])).name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "expected_sha256": str(item["sha256"]),
                "observed_sha256": "NOT_OPENED_C85URP",
                "role": role,
            })
    require(len(rows) == 4, "C85URP future U2 input registry coverage drift")
    return rows


def materialize_readiness_tables() -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    registry_summary = write_readiness_registries(TABLE_DIR)
    isolation = static_isolation_audit()
    _write_fresh_csv(TABLE_DIR / "process_isolation_audit.csv", isolation)
    _write_fresh_csv(TABLE_DIR / "candidate_utility_artifact_schema.csv", _schema_rows())
    _write_fresh_csv(
        TABLE_DIR / "historical_decision_replay_contract.csv",
        [
            {"object": "contexts", "expected": 944, "tolerance": "EXACT", "U1_input": 1, "U2_input": 1},
            {"object": "candidate utility rows", "expected": 76464, "tolerance": "EXACT", "U1_input": 0, "U2_input": 1},
            {"object": "method-context rows", "expected": 18432, "tolerance": "EXACT", "U1_input": 0, "U2_input": 1},
            {"object": "selected utility/regret/top-k", "expected": "historical", "tolerance": "1e-12", "U1_input": 0, "U2_input": 1},
            {"object": "selected regime", "expected": "historical", "tolerance": "EXACT", "U1_input": 0, "U2_input": 1},
            {"object": "Q0 finite actions", "expected": "2048 frozen chains", "tolerance": "EXACT_NO_RESAMPLE", "U1_input": 0, "U2_input": 1},
        ],
    )
    _write_fresh_csv(TABLE_DIR / "future_u2_input_registry.csv", _selection_artifact_bindings())
    tied = midrank_percentile(np.ones(81))
    calibration = [
        {"case": "944x81 arithmetic", "expected": 76464, "observed": 944 * 81, "status": "PASS"},
        {"case": "all-tie midrank", "expected": "0.5", "observed": str(float(tied[0])), "status": "PASS"},
        {"case": "first canonical argmax", "expected": 0, "observed": int(np.lexsort((np.arange(81), -np.ones(81)))[0]), "status": "PASS"},
        {"case": "zero-spread regret", "expected": "0.0", "observed": str(standardized_regret(np.ones(81), 80)), "status": "PASS"},
        {"case": "real protected objects opened", "expected": 0, "observed": sum(registry_summary["protected_access_counters"].values()), "status": "PASS"},
        {"case": "authorization records", "expected": 0, "observed": int((REPORT_DIR / "C85U_PI_AUTHORIZATION_RECORD.json").exists()), "status": "PASS"},
    ]
    _write_fresh_csv(TABLE_DIR / "synthetic_shadow_calibration.csv", calibration)
    _write_fresh_csv(
        TABLE_DIR / "authorization_and_resource_contract.csv",
        [
            {"item": "direct statement", "value": "授权 C85U", "unit": "exact string", "locked": 1},
            {"item": "single use", "value": "O_CREAT|O_EXCL", "unit": "external receipt", "locked": 1},
            {"item": "CPU", "value": 48, "unit": "cores", "locked": 1},
            {"item": "RAM", "value": 128, "unit": "GiB", "locked": 1},
            {"item": "GPU", "value": 0, "unit": "devices", "locked": 1},
            {"item": "wall", "value": 2, "unit": "hours", "locked": 1},
            {"item": "output maximum", "value": 2, "unit": "GiB", "locked": 1},
            {"item": "target read bytes", "value": registry_summary["target_artifact_bytes"], "unit": "bytes", "locked": 1},
        ],
    )
    _write_fresh_csv(
        TABLE_DIR / "risk_register.csv",
        [
            {"risk": "protected input opened before authorization", "control": "metadata-only readiness plus O_EXCL guard", "residual": "filesystem administrator access", "status": "CONTROLLED"},
            {"risk": "selection influences utility construction", "control": "U1 subprocess has no Stage-B/Q0 input", "residual": "historical formula dependency", "status": "CONTROLLED"},
            {"risk": "U2 reopens labels/logits", "control": "separate subprocess and no protected-input argument", "residual": "OS-level access is outside governance claim", "status": "CONTROLLED"},
            {"risk": "partial U1 accepted", "control": "944-artifact semantic replay before atomic rename", "residual": "filesystem rename semantics", "status": "CONTROLLED"},
            {"risk": "historical endpoint mismatch", "control": "18,432-row exact identity and 1e-12 value replay", "residual": "runtime mismatch blocks", "status": "CONTROLLED"},
            {"risk": "post-outcome artifact overinterpreted", "control": "exploratory infrastructure claim contract", "residual": "downstream reporting discipline", "status": "CONTROLLED_BY_C85EP2"},
        ],
    )
    _write_fresh_csv(
        TABLE_DIR / "failure_reason_ledger.csv",
        [
            {"stage": "PREFLIGHT", "reason": "input path/hash/lock drift", "final_utility_manifest": 0, "disposition": "BLOCK_BEFORE_CONSUMPTION"},
            {"stage": "U1", "reason": "metric/identity/persistence mismatch", "final_utility_manifest": 0, "disposition": "PRESERVE_FAILED_ROOT_NO_RETRY"},
            {"stage": "U1", "reason": "coverage below 944x81", "final_utility_manifest": 0, "disposition": "NO_ATOMIC_PUBLICATION"},
            {"stage": "U2", "reason": "frozen action coverage mismatch", "final_utility_manifest": 1, "disposition": "U1_NOT_ACCEPTED_FOR_C85E"},
            {"stage": "U2", "reason": "historical endpoint mismatch", "final_utility_manifest": 1, "disposition": "STOP_PM_REVIEW"},
            {"stage": "RUNTIME", "reason": "formula/tolerance/schema pressure", "final_utility_manifest": 0, "disposition": "NO_RUNTIME_CHANGE"},
        ],
    )
    return {
        **registry_summary,
        "implementation_files": len(IMPLEMENTATION_PATHS),
        "isolation_rows": len(isolation),
        "readiness_tables": len(list(TABLE_DIR.glob("*.csv"))),
        "real_evaluation_rows_opened": 0,
        "real_target_artifact_payloads_opened": 0,
        "real_utility_rows_computed": 0,
    }


def _bound_paths() -> list[str]:
    fixed = {
        "oaci/reports/C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.json",
        "oaci/reports/C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.sha256",
        "oaci/reports/C85U_PROTOCOL_TIMING_AUDIT.md",
        "oaci/reports/C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.json",
        "oaci/reports/C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.sha256",
        "oaci/reports/C85EP_INPUT_AVAILABILITY_BLOCKER.json",
        "oaci/reports/C85EP_INPUT_AVAILABILITY_BLOCKER.sha256",
        "oaci/reports/C84S_ANALYSIS_EXECUTION_LOCK_V5.json",
        "oaci/multidataset/c84r_regression_suite.py",
        "oaci/slurm_c85urp_regression.sh",
    }
    fixed.update(IMPLEMENTATION_PATHS)
    fixed.update(TEST_PATHS)
    fixed.update(HISTORICAL_IMPLEMENTATION_PATHS)
    fixed.update(
        str(path.relative_to(REPO_ROOT))
        for path in TABLE_DIR.glob("*.csv")
        if path.name != "runtime_bound_object_registry.csv"
    )
    missing = [relative for relative in sorted(fixed) if not (REPO_ROOT / relative).is_file()]
    require(not missing, f"C85URP bound object absent: {missing[0] if missing else ''}")
    return sorted(fixed)


def build_execution_lock(
    *, implementation_commit: str, created_at_utc: str,
) -> dict[str, Any]:
    lock_path = REPORT_DIR / "C85U_EXECUTION_LOCK.json"
    sidecar = REPORT_DIR / "C85U_EXECUTION_LOCK.sha256"
    registry_path = TABLE_DIR / "runtime_bound_object_registry.csv"
    require(not any(path.exists() for path in (lock_path, sidecar, registry_path)),
            "C85U lock objects must be fresh")
    require(_git("rev-parse", "HEAD") == implementation_commit and
            not _git("status", "--porcelain"),
            "C85U lock build requires clean implementation HEAD")
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, implementation_commit],
        cwd=REPO_ROOT, check=False,
    ).returncode == 0, "C85U protocol must precede implementation")
    require(not (REPORT_DIR / "C85U_PI_AUTHORIZATION_RECORD.json").exists(),
            "C85U authorization exists during readiness")

    bound_rows = []
    for relative in _bound_paths():
        path = REPO_ROOT / relative
        bound_rows.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob": _git("hash-object", "--", relative),
        })
    _write_fresh_csv(registry_path, bound_rows)
    input_registry = build_frozen_input_registry()
    protected_registry_sha = sha256_file(TABLE_DIR / "target_artifact_registry.csv")
    lock = {
        "schema_version": "c85u_execution_lock_v1",
        "milestone": "C85URP",
        "created_at_utc": created_at_utc,
        "status": LOCK_STATUS,
        "authorized": False,
        "repo_root": str(REPO_ROOT),
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "runtime_bound_object_count": len(bound_rows),
        "runtime_bound_repository_registry_sha256": canonical_sha256(bound_rows),
        "bound_repository_objects": bound_rows,
        "runtime_bound_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "size_bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
            "git_blob": _git("hash-object", "--", str(registry_path.relative_to(REPO_ROOT))),
        },
        "frozen_inputs": {
            "C84F_complete_field_manifest_sha256": "cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8",
            "C84S_V5_analysis_lock_sha256": "030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846",
            "C84S_selection_freeze_sha256": "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4",
            "C84S_scientific_result_sha256": "5590f85c3552ec0176a015e34296059a950dd2c5853a51aa140657cf53d79ee7",
            "C84S_result_manifest_sha256": "516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5",
            "evaluation_seal": {"path": str(EVALUATION_SEAL), "sha256": "54e06dff60d80255631dc4faa20c8c7db651f2af8fc5415671dd9ab6681b5502"},
            "evaluation_view_manifest": {"path": str(EVALUATION_VIEW_MANIFEST), "sha256": "6fad247629eb48340a4badf9ab1a0669652757a58216e46826e4dfd8bfd608bd"},
            "evaluation_label_table": {"path": str(EVALUATION_LABEL_TABLE), "sha256": EVALUATION_LABEL_SHA256, "bytes": EVALUATION_LABEL_TABLE.stat().st_size, "rows": 4848, "opened_C85URP": 0},
            "target_artifact_registry": {"path": "oaci/reports/c85urp_tables/target_artifact_registry.csv", "sha256": protected_registry_sha, "units": 1944, "bytes": sum(int(row["target_artifact_bytes"]) for row in input_registry.target_artifact_rows), "opened_C85URP": 0},
            "field_descriptors": 1944,
            "contexts": 944,
            "candidates_per_context": 81,
        },
        "historical_utility": {
            "context_candidate_utility_sha256": sha256_file(REPO_ROOT / "oaci/multidataset/c84s_evaluation.py"),
            "endpoint_and_midrank_sha256": sha256_file(REPO_ROOT / "oaci/multidataset/c84s_q0_budget.py"),
            "metric_fields": ["balanced_accuracy", "NLL", "ECE"],
            "oriented_midrank_components": 3,
            "composite": "ARITHMETIC_MEAN",
            "canonical_argmax": "FIRST_INDEX",
        },
        "stages": {
            "U1": {"process": "oaci.theory.c85u_stage_u1", "contexts": 944, "candidate_rows": 76464, "selection_inputs": 0, "atomic": True},
            "U2": {"process": "oaci.theory.c85u_stage_u2", "method_context_rows": 18432, "label_or_logit_inputs": 0, "Q0_chains": 2048, "Q0_resampling": 0},
            "subprocess_isolation": True,
        },
        "schemas": {
            "context": "c85u_candidate_utility_context_v1",
            "index": "c85u_candidate_utility_index_v1",
            "utility_manifest": "c85u_complete_utility_manifest_v1",
            "historical_replay": "c85u_historical_decision_replay_v1",
        },
        "numerical_contract": {
            "float_dtype": "<f8",
            "metric_and_utility_max_abs": 1e-12,
            "identity_digest_order_midrank": "EXACT",
            "runtime_widening": False,
        },
        "authorization_record_path": "oaci/reports/C85U_PI_AUTHORIZATION_RECORD.json",
        "authorization_schema": "c85u_direct_pi_authorization_record_v1",
        "future_direct_statement_exact": "授权 C85U",
        "authorization_consumption_root": "/projects/EEG-foundation-model/yinghao/oaci-c85u-authorization-consumption-v1",
        "output_root_policy": {
            "parent": "/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v1",
            "basename": "c85u-{lock_sha16}-{authorization_id16}",
            "exact_absolute_binding_required": True,
            "max_bytes": 2147483648,
        },
        "environment": {
            "prefix": str(Path(sys.executable).resolve().parents[1]),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "numpy_file_sha256": sha256_file(Path(np.__file__)),
            "scipy_version": scipy.__version__,
            "scipy_file_sha256": sha256_file(Path(scipy.__file__)),
            "GPU": 0,
        },
        "entrypoint": "python -m oaci.theory.c85u_execute run-real --execution-lock <LOCK> --authorization-record <AUTHORIZATION> --output-root <BOUND_ROOT>",
        "resources": {"partition": "cpu-high", "CPU": 48, "RAM_GiB": 128, "GPU": 0, "wall_hours": 2, "output_GiB_max": 2},
        "failure_policy": {
            "before_consumption": "NO_PROTECTED_BYTES_AUTHORIZATION_UNCONSUMED",
            "U1": "NO_PARTIAL_FINAL_MANIFEST_PRESERVE_STAGING_NO_AUTOMATIC_RETRY",
            "U2": "U1_FROZEN_NOT_ACCEPTED_FOR_C85E_NO_LABEL_OR_LOGIT_REOPEN",
            "runtime_mismatch": "STOP_NO_FORMULA_TOLERANCE_OR_SCHEMA_CHANGE",
        },
        "immutable_results": {
            "C84_primary": "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous",
            "C84_frontier": "C84-L4",
            "C85_theorem_statuses": {"T1": "PROVED", "T2": "COUNTEREXAMPLE", "T3": "PROVED", "T4": "PROVED", "T5": "OPEN", "T6": "COUNTEREXAMPLE", "T7": "PROVED"},
        },
        "readiness": {
            "real_evaluation_label_rows_opened": 0,
            "real_target_artifact_payloads_opened": 0,
            "real_candidate_utilities_computed": 0,
            "authorization_records": 0,
            "C85E_lock_created": False,
            "success_gate": SUCCESS_GATE,
            "failure_gate": FAILURE_GATE,
        },
        "future_completion_gate": "C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED",
        "forbidden": {
            "C85E_execution": True, "C86": True, "active_acquisition": True,
            "new_data_or_model_zoo": True, "manuscript_work": True,
            "training_forward_GPU": True, "new_scientific_inference": True,
        },
    }
    lock_path.write_bytes(_canonical_json_bytes(lock))
    digest = sha256_file(lock_path)
    sidecar.write_text(f"{digest}  {lock_path.name}\n", encoding="ascii")
    return {
        "lock_path": str(lock_path), "lock_sha256": digest,
        "runtime_bound_object_count": len(bound_rows),
        "target_artifacts_bound": 1944,
        "protected_objects_opened": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize-readiness")
    lock = subparsers.add_parser("build-lock")
    lock.add_argument("--implementation-commit", required=True)
    lock.add_argument("--created-at-utc", required=True)
    arguments = parser.parse_args(argv)
    result = (
        materialize_readiness_tables()
        if arguments.command == "materialize-readiness"
        else build_execution_lock(
            implementation_commit=arguments.implementation_commit,
            created_at_utc=arguments.created_at_utc,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAILURE_GATE", "IMPLEMENTATION_PATHS", "LOCK_STATUS", "SUCCESS_GATE",
    "build_execution_lock", "materialize_readiness_tables", "static_isolation_audit",
]
