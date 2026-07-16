"""Fail-closed runtime and execution-lock support for future C84S."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from . import c84s_analysis as analysis
from .c84s_common import (
    C84SContractError, canonical_sha256, read_csv, read_json, require,
    sha256_file, write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84sl_tables"
PROTOCOL_PATH = REPORT_DIR / "C84S_ANALYSIS_OPERATIONALIZATION_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C84S_ANALYSIS_OPERATIONALIZATION_PROTOCOL.sha256"
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C84SL_END_TO_END_RESULT_FREEZE_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C84SL_END_TO_END_RESULT_FREEZE_REPAIR_PROTOCOL.sha256"
FRONTIER_CLARIFICATION_PATH = REPORT_DIR / "C84SL_LABEL_FRONTIER_STABILITY_CLARIFICATION.json"
FRONTIER_CLARIFICATION_SHA_PATH = REPORT_DIR / "C84SL_LABEL_FRONTIER_STABILITY_CLARIFICATION.sha256"
SCIENCE_V3_PATH = REPORT_DIR / "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json"
METHOD_REGISTRY_PATH = REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json"
SELECTOR_REPLAY_PATH = REPORT_DIR / "c84p_tables/selector_registry_replay.csv"
COMPLETE_FIELD_MANIFEST_PATH = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2/"
    "lock_f0c369ee273352b47e36/C84F_COMPLETE_FIELD_MANIFEST.json"
)
HISTORICAL_LOCK_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK.json"
HISTORICAL_LOCK_SHA_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK.sha256"
LOCK_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V2.json"
LOCK_SHA_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V2.sha256"
AUTHORIZATION_PATH = REPORT_DIR / "C84S_PI_AUTHORIZATION_RECORD.json"
LOCK_READY_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
EXPECTED = {
    "operationalization": "abf56676901bd7e5f484ffe4f4bb49de625d5c7b87cc34c0e3ab2bdb39361c5e",
    "end_to_end_repair": "b2d52a3bfdcb89b8a8db5d1a5501fb7b24a22ad860dbe1d6da2c5e6d77ca189c",
    "frontier_stability": "020251ded9dbe4688ef08e9854875fc06b789120f4697c661b94f96eeef66fca",
    "historical_lock": "e17e4da14b60ac77ca0ec8bec80a2ca249cda014baf5460cfd64627294f2047b",
    "science_v3": "bf6c7f718413b4b2ac2ad9786aa2e47dc045a536e7237d5d8c0464b6598130b8",
    "method_registry": "ef48ecf7fcc55188b78b0878d86f07f6239fe4f6c88bbc854829b3a1c7a1a120",
    "selector_replay": "d589437e40812350eec44bdfbf1b75c52f10ef41e0e3ca5868e07844b0228e68",
    "complete_field": "cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8",
    "target_raw": "9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd",
    "target_registry": "52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8",
}
LOCKED_ENVIRONMENT = {
    "conda_prefix": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact",
    "python": "3.13.7",
    "distributions": {
        "numpy": "2.3.3", "scipy": "1.16.2", "moabb": "1.5.0", "mne": "1.11.0",
    },
}
LOADER_SOURCE_IDENTITIES = (
    {"distribution": "moabb", "path": "moabb/datasets/Lee2019.py", "sha256": "a0234b81923fed15e4a221e011399f76a83873cd43d598ad5c8c71ba54678a6f"},
    {"distribution": "moabb", "path": "moabb/datasets/gigadb.py", "sha256": "42e2ef372762cb86aab11a886e1707675477ac776e0468448233de7a4ba71e32"},
    {"distribution": "moabb", "path": "moabb/datasets/physionet_mi.py", "sha256": "a8abe8097870d804a2d78f500f3c6820962c1c3402f53368e92e7a91068b84ba"},
    {"distribution": "moabb", "path": "moabb/paradigms/motor_imagery.py", "sha256": "f941a3f17c1bca4211045c28f7df3704c9d428ef689dff2410a478b5bf68651e"},
)
IMPLEMENTATION_PATHS = (
    "oaci/multidataset/c84s_common.py",
    "oaci/multidataset/c84s_label_views.py",
    "oaci/multidataset/c84s_selectors.py",
    "oaci/multidataset/c84s_q0_budget.py",
    "oaci/multidataset/c84s_selection_freeze.py",
    "oaci/multidataset/c84s_evaluation.py",
    "oaci/multidataset/c84s_inference.py",
    "oaci/multidataset/c84s_taxonomy.py",
    "oaci/multidataset/c84s_analysis.py",
    "oaci/multidataset/c84s_synthetic_end_to_end.py",
    "oaci/multidataset/c84s_runtime_guard.py",
    "oaci/multidataset/c84sl_readiness.py",
)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def verify_protocol_inputs() -> dict[str, Any]:
    sidecar = PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(sidecar == EXPECTED["operationalization"], "operative C84S protocol sidecar drift")
    repair_sidecar = REPAIR_PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(repair_sidecar == EXPECTED["end_to_end_repair"], "C84SL repair protocol sidecar drift")
    frontier_sidecar = FRONTIER_CLARIFICATION_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(frontier_sidecar == EXPECTED["frontier_stability"], "C84SL frontier clarification sidecar drift")
    historical_sidecar = HISTORICAL_LOCK_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(historical_sidecar == EXPECTED["historical_lock"], "historical C84S lock sidecar drift")
    paths = {
        "operationalization": PROTOCOL_PATH,
        "end_to_end_repair": REPAIR_PROTOCOL_PATH,
        "frontier_stability": FRONTIER_CLARIFICATION_PATH,
        "historical_lock": HISTORICAL_LOCK_PATH,
        "science_v3": SCIENCE_V3_PATH,
        "method_registry": METHOD_REGISTRY_PATH,
        "selector_replay": SELECTOR_REPLAY_PATH,
        "complete_field": COMPLETE_FIELD_MANIFEST_PATH,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    for name, digest in observed.items():
        require(digest == EXPECTED[name], f"C84S frozen input hash drift: {name}")
    manifest = read_json(COMPLETE_FIELD_MANIFEST_PATH)
    require(manifest["gate"] == "C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
            "C84F complete-field gate drift")
    require(manifest["target_raw_manifest_sha256"] == EXPECTED["target_raw"], "target raw-manifest drift")
    require(manifest["target_trial_registry_sha256"] == EXPECTED["target_registry"], "target trial-registry drift")
    require(len(manifest["field_descriptors"]) == 1944, "complete-field descriptor count drift")
    for field in (
        "target_construction_labels", "target_evaluation_labels", "same_label_oracle",
        "selector_scores", "scientific_statistics",
    ):
        require(manifest[field] == 0, f"C84F protected field is nonzero: {field}")
    require(manifest["C84S_authorized"] is False, "C84F manifest unexpectedly authorizes C84S")
    return {"hashes": observed, "manifest": manifest}


def target_artifact_registry(*, verify_bytes: bool) -> list[dict[str, Any]]:
    manifest = verify_protocol_inputs()["manifest"]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for descriptor in manifest["field_descriptors"]:
        unit_id = str(descriptor["unit_id"])
        for kind, key in (
            ("target_artifact", "complete_target_unlabeled"),
            ("context_digest_sidecar", "target_context_digest_index"),
        ):
            identity = descriptor[key]
            path = Path(identity["path"])
            expected = str(identity["sha256"])
            pair = (unit_id, kind)
            require(pair not in seen, "duplicate target artifact identity")
            seen.add(pair)
            observed = sha256_file(path) if verify_bytes else expected
            require(observed == expected, f"target artifact byte hash drift: {unit_id}/{kind}")
            rows.append({
                "unit_id": unit_id, "artifact_kind": kind, "path": str(path),
                "bytes": path.stat().st_size, "expected_sha256": expected,
                "observed_sha256": observed, "byte_replay_pass": int(observed == expected),
                "verification_mode": "sha256_file_read" if verify_bytes else "manifest_identity_only",
            })
    require(len(rows) == 3888 and len(seen) == 3888, "target artifact registry coverage drift")
    return rows


def static_isolation_audit() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    forbidden_all = {"torch", "mne", "moabb", "cupy"}
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        require(path.is_file(), f"C84S implementation file absent: {relative}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            str(node.module or "")
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        blocked = sorted(
            name for name in imports
            if any(name == token or name.startswith(token + ".") for token in forbidden_all)
        )
        require(not blocked, f"training/forward/GPU package import in {relative}: {blocked}")
        checks.append({"path": relative, "check": "no_training_forward_GPU_import", "pass": 1})
    label_source = (REPO_ROOT / "oaci/multidataset/c84s_label_views.py").read_text(encoding="utf-8")
    require("c84s_selectors" not in label_source and "complete_target_unlabeled" not in label_source,
            "label provisioner imports/references candidate artifacts")
    selector_source = (REPO_ROOT / "oaci/multidataset/c84s_selectors.py").read_text(encoding="utf-8")
    require("c84s_label_views" not in selector_source and "evaluation_label" not in selector_source,
            "zero-label selector imports target label view")
    evaluation_source = (REPO_ROOT / "oaci/multidataset/c84s_evaluation.py").read_text(encoding="utf-8")
    require("c84s_selectors" not in evaluation_source and "freeze_selection(" not in evaluation_source,
            "evaluation process can modify selection")
    analysis_source = (REPO_ROOT / "oaci/multidataset/c84s_analysis.py").read_text(encoding="utf-8")
    require(
        "c84s_selectors" not in analysis_source
        and "c84s_label_views" not in analysis_source
        and "freeze_selection(" not in analysis_source,
        "Stage-C analysis can alter selection or provision labels",
    )
    checks.extend([
        {"path": "c84s_label_views.py", "check": "no_candidate_artifact_import", "pass": 1},
        {"path": "c84s_selectors.py", "check": "no_target_label_view_import", "pass": 1},
        {"path": "c84s_evaluation.py", "check": "no_selection_mutation_callable", "pass": 1},
        {"path": "c84s_analysis.py", "check": "immutable_selection_and_no_label_provisioning", "pass": 1},
    ])
    return checks


def repository_object_registry(paths: Sequence[str] = IMPLEMENTATION_PATHS) -> list[dict[str, Any]]:
    head = _git("rev-parse", "HEAD")
    rows = []
    for relative in paths:
        path = REPO_ROOT / relative
        require(path.is_file(), f"runtime-bound repository object absent: {relative}")
        blob = _git("rev-parse", f"HEAD:{relative}")
        rows.append({
            "path": relative, "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "blob": blob, "commit": head,
        })
    return rows


def verify_bound_repository_objects(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for identity in lock["runtime_bound_repository_objects"]:
        path = REPO_ROOT / identity["path"]
        require(path.is_file(), f"bound repository object absent: {identity['path']}")
        require(sha256_file(path) == identity["sha256"], f"bound file SHA drift: {identity['path']}")
        require(_git("rev-parse", f"HEAD:{identity['path']}") == identity["blob"],
                f"bound Git blob drift: {identity['path']}")
        rows.append({"path": identity["path"], "replay_pass": 1})
    return rows


def verify_lock_self(path: Path = LOCK_PATH, sidecar: Path = LOCK_SHA_PATH) -> str:
    require(path.is_file() and sidecar.is_file(), "C84S analysis lock is absent")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    observed = sha256_file(path)
    require(observed == expected, "C84S analysis-lock SHA drift")
    return observed


def verify_environment_and_loader_sources() -> dict[str, Any]:
    require(".".join(map(str, sys.version_info[:3])) == LOCKED_ENVIRONMENT["python"],
            "C84S Python version drift")
    require(str(Path(sys.prefix).resolve()) == str(Path(LOCKED_ENVIRONMENT["conda_prefix"]).resolve()),
            "C84S Conda prefix drift")
    versions = {}
    for distribution, expected in LOCKED_ENVIRONMENT["distributions"].items():
        observed = importlib.metadata.version(distribution)
        require(observed == expected, f"C84S distribution version drift: {distribution}")
        versions[distribution] = observed
    identities = []
    for identity in LOADER_SOURCE_IDENTITIES:
        distribution = importlib.metadata.distribution(identity["distribution"])
        path = Path(distribution.locate_file(identity["path"]))
        require(path.is_file(), f"C84S loader source absent: {identity['path']}")
        observed = sha256_file(path)
        require(observed == identity["sha256"], f"C84S loader source drift: {identity['path']}")
        identities.append({**identity, "observed_sha256": observed})
    return {"versions": versions, "loader_sources": identities}


def verify_selection_input_artifacts(manifest: Mapping[str, Any]) -> dict[str, int]:
    counts = {"source_audit": 0, "training_sidecar": 0}
    for descriptor in manifest["field_descriptors"]:
        for key in counts:
            identity = descriptor[key]
            path = Path(identity["path"])
            require(path.is_file() and sha256_file(path) == identity["sha256"],
                    f"C84S selection input artifact drift: {descriptor['unit_id']}/{key}")
            counts[key] += 1
    require(counts == {"source_audit": 1944, "training_sidecar": 1944},
            "C84S selection input artifact coverage drift")
    return counts


def verify_authorization_record(lock: Mapping[str, Any], lock_sha: str, path: Path = AUTHORIZATION_PATH) -> dict[str, Any]:
    require(path.is_file(), "C84S direct PI authorization record is absent")
    record = read_json(path)
    required = {
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84S",
        "analysis_lock_sha256": lock_sha,
        "operationalization_protocol_sha256": EXPECTED["operationalization"],
        "end_to_end_repair_protocol_sha256": EXPECTED["end_to_end_repair"],
        "frontier_stability_protocol_sha256": EXPECTED["frontier_stability"],
        "complete_field_manifest_sha256": EXPECTED["complete_field"],
        "same_label_oracle": False,
        "training": False,
        "forward": False,
        "GPU": False,
        "C85": False,
    }
    for key, value in required.items():
        require(record.get(key) == value, f"C84S authorization binding drift: {key}")
    require(lock["status"] == LOCK_READY_STATUS, "C84S lock is not authorization-ready")
    return record


def verify_clean_synced_branch() -> str:
    require(_git("status", "--porcelain") == "", "C84S runtime requires a clean worktree")
    require(_git("branch", "--show-current") == "oaci", "C84S runtime requires branch oaci")
    head = _git("rev-parse", "HEAD")
    require(head == _git("rev-parse", "origin/oaci"), "C84S runtime HEAD differs from origin/oaci")
    return head


def pre_label_access_guard() -> dict[str, Any]:
    """Future real entrypoint guard; never called during C84SL readiness."""
    lock_sha = verify_lock_self()
    lock = read_json(LOCK_PATH)
    head = verify_clean_synced_branch()
    inputs = verify_protocol_inputs()
    verify_bound_repository_objects(lock)
    registry = read_csv(TABLE_DIR / "target_artifact_registry_replay.csv")
    require(sha256_file(TABLE_DIR / "target_artifact_registry_replay.csv") == lock["external_field"]["target_artifact_registry_sha256"],
            "target artifact registry SHA drift")
    require(len(registry) == 3888 and all(row["byte_replay_pass"] == "1" for row in registry),
            "target artifact byte-replay registry incomplete")
    verify_selection_input_artifacts(inputs["manifest"])
    verify_environment_and_loader_sources()
    verify_authorization_record(lock, lock_sha)
    return {
        "head": head, "lock_sha256": lock_sha,
        "bound_files": len(lock["runtime_bound_repository_objects"]),
        "authorized_stage": "C84S", "C84S_authorized": True,
    }


def build_execution_lock(*, implementation_commit: str) -> dict[str, Any]:
    require(_git("rev-parse", "HEAD") == implementation_commit, "implementation commit is not current HEAD")
    require(not AUTHORIZATION_PATH.exists(), "C84S authorization exists before lock")
    inputs = verify_protocol_inputs()
    static_checks = static_isolation_audit()
    artifact_registry_path = TABLE_DIR / "target_artifact_registry_replay.csv"
    require(artifact_registry_path.is_file(), "target artifact registry is absent")
    artifact_rows = read_csv(artifact_registry_path)
    require(len(artifact_rows) == 3888 and all(row["byte_replay_pass"] == "1" for row in artifact_rows),
            "target artifact registry byte replay is incomplete")
    repository_rows = repository_object_registry()
    from .c84s_common import write_csv
    repository_registry_path = TABLE_DIR / "runtime_bound_object_registry.csv"
    repository_registry_sha = write_csv(repository_registry_path, repository_rows)
    readiness_tables = {
        path.name: sha256_file(path)
        for path in sorted(TABLE_DIR.glob("*.csv"))
    }
    lock = {
        "schema_version": "c84s_analysis_execution_lock_v2",
        "milestone": "C84SL",
        "status": LOCK_READY_STATUS,
        "chronology": {
            "operationalization_protocol_commit": _git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(REPO_ROOT))),
            "end_to_end_repair_protocol_commit": _git("log", "-1", "--format=%H", "--", str(REPAIR_PROTOCOL_PATH.relative_to(REPO_ROOT))),
            "frontier_stability_protocol_commit": _git("log", "-1", "--format=%H", "--", str(FRONTIER_CLARIFICATION_PATH.relative_to(REPO_ROOT))),
            "implementation_commit": implementation_commit,
            "protocol_precedes_implementation": True,
            "real_label_access_at_lock": 0,
            "real_selector_scores_at_lock": 0,
            "real_scientific_statistics_at_lock": 0,
        },
        "protocols": {
            "operationalization": EXPECTED["operationalization"],
            "end_to_end_repair": EXPECTED["end_to_end_repair"],
            "frontier_stability": EXPECTED["frontier_stability"],
            "scientific_v3": EXPECTED["science_v3"],
            "method_registry": EXPECTED["method_registry"],
            "selector_replay": EXPECTED["selector_replay"],
        },
        "external_field": {
            "complete_manifest_path": str(COMPLETE_FIELD_MANIFEST_PATH),
            "complete_manifest_sha256": EXPECTED["complete_field"],
            "model_field_manifest_sha256": inputs["manifest"]["model_field_manifest_sha256"],
            "target_raw_manifest_sha256": EXPECTED["target_raw"],
            "target_trial_registry_sha256": EXPECTED["target_registry"],
            "target_artifact_registry_path": str(artifact_registry_path.relative_to(REPO_ROOT)),
            "target_artifact_registry_sha256": sha256_file(artifact_registry_path),
            "target_artifacts": 1944,
            "context_digest_sidecars": 1944,
            "candidate_context_slices": 76464,
        },
        "runtime_bound_repository_objects": repository_rows,
        "runtime_bound_object_registry": {
            "path": str(repository_registry_path.relative_to(REPO_ROOT)),
            "sha256": repository_registry_sha,
            "rows": len(repository_rows),
        },
        "readiness_table_hashes": readiness_tables,
        "environment": {**LOCKED_ENVIRONMENT, "GPU_required": False},
        "loader_source_identities": list(LOADER_SOURCE_IDENTITIES),
        "static_isolation_audit_sha256": canonical_sha256(static_checks),
        "analysis_contract": {
            "label_split": "C84_TARGET_SPLIT_V1",
            "selection_freeze_before_evaluation": True,
            "Q0_chains": 2048,
            "Q0_RNG": "PCG64_SHA256_low64",
            "Q0_FULL_deterministic_across_chains": True,
            "maxT_draws": 65536,
            "principal_cluster": "target_subject",
            "same_method_across_datasets": True,
            "level_heterogeneity_precedence": True,
            "LOTO_thresholds": {"Lee2019_MI": 17, "Cho2017": 15, "PhysionetMI": 57},
            "taxonomy": ["C84-E", "C84-D", "C84-A", "C84-B", "C84-C"],
            "label_frontier": ["C84-L1", "C84-L2", "C84-L3", "C84-L4"],
            "method_context_rows": 18608,
            "Stage_C_table_schemas": len(analysis.RESULT_TABLE_FIELDS),
            "atomic_result_freeze": True,
            "synthetic_uses_production_entrypoint": True,
        },
        "authorization": {
            "record_path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
            "record_present_at_lock": False,
            "fresh_direct_statement": "授权 C84S",
            "authorization_inherited": False,
        },
        "historical_lock_supersession": {
            "path": str(HISTORICAL_LOCK_PATH.relative_to(REPO_ROOT)),
            "sha256": EXPECTED["historical_lock"],
            "preserved": True,
            "operative_for_future_C84S": False,
            "authorization_consumed": False,
        },
        "forbidden": {
            "training": True, "forward": True, "GPU": True,
            "checkpoint_load": True, "new_method": True, "retuning": True,
            "same_label_oracle": True, "evaluation_before_selection_freeze": True,
            "C85": True,
        },
        "success_gate": "C84S_MULTIDATASET_LABEL_VIEWS_SELECTION_INFERENCE_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION",
        "failure_gate": "C84S_LABEL_VIEW_SELECTOR_INFERENCE_TAXONOMY_OR_PROVENANCE_RECONCILIATION_REQUIRED",
    }
    digest = write_json(LOCK_PATH, lock)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")
    return {"lock": lock, "sha256": digest}
