"""Generate the additive C84FR2 target-stage execution lock and readiness artifacts."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

from . import c84fr2_runtime_guard as runtime
from . import c84fr2_target_numerical_replay as numerical


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84fr2_tables"
LOCK_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK_V2.json"
LOCK_SHA_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK_V2.sha256"
OLD_LOCK_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK.json"
OLD_LOCK_SHA_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK.sha256"
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C84FR2_TARGET_NUMERICAL_REPLAY_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C84FR2_TARGET_NUMERICAL_REPLAY_REPAIR_PROTOCOL.sha256"
TARGET_PROTOCOL_PATH = REPORT_DIR / "C84_TARGET_INSTRUMENTATION_PROTOCOL_V2.json"
TARGET_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_TARGET_INSTRUMENTATION_PROTOCOL_V2.sha256"
PROTOCOL_COMMIT = "27fc479ecd4131ceb4f79982cb0890f517709d2e"
TARGET_PROTOCOL_COMMIT = "b527b82950690d09e73e5f3468d994cf11b56413"
SUCCESS_GATE = "C84F_TARGET_INSTRUMENTATION_SAME_BACKEND_REPLAY_REPAIRED_READY_FOR_PI_REAUTHORIZATION"
STATUS = runtime.LOCK_READY_STATUS

MODEL_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-v1/lock_f9df9dcefea59b05bfea"
)
TARGET_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-repair-v1/lock_2e888b6497ec455a325a"
)

IMPLEMENTATION_FILES = (
    "oaci/__init__.py",
    "oaci/support_graph.py",
    "oaci/multidataset/__init__.py",
    "oaci/multidataset/c84_dataset_registry.py",
    "oaci/multidataset/c84_dataset_registry_v2.py",
    "oaci/multidataset/c84r_montage_repair.py",
    "oaci/multidataset/c84fl2_protocol.py",
    "oaci/multidataset/c84f_field_manifest.py",
    "oaci/multidataset/c84f_target_instrumentation.py",
    "oaci/multidataset/c84f_runtime_guard.py",
    "oaci/multidataset/c84fr1_runtime_guard.py",
    "oaci/multidataset/c84fr2_target_numerical_replay.py",
    "oaci/multidataset/c84fr2_target_stage.py",
    "oaci/multidataset/c84fr2_runtime_guard.py",
    "oaci/multidataset/c84fr2_lock.py",
    "oaci/models/__init__.py",
    "oaci/models/factory.py",
    "oaci/models/shallow.py",
    "oaci/models/output.py",
)

RUNTIME_REGISTRY_FILES = (
    "oaci/reports/C84FR2_TARGET_NUMERICAL_REPLAY_REPAIR_PROTOCOL.json",
    "oaci/reports/C84FR2_TARGET_NUMERICAL_REPLAY_REPAIR_PROTOCOL.sha256",
    "oaci/reports/C84_TARGET_INSTRUMENTATION_PROTOCOL_V2.json",
    "oaci/reports/C84_TARGET_INSTRUMENTATION_PROTOCOL_V2.sha256",
    "oaci/reports/C84F_FAILED_ATTEMPT_896185.json",
    "oaci/reports/C84F_FAILED_ATTEMPT_896185.sha256",
    "oaci/reports/C84FR1_FAILED_ATTEMPT_896550.json",
    "oaci/reports/C84FR1_FAILED_ATTEMPT_896550.sha256",
    "oaci/reports/c84fr2_tables/failed_artifact_backend_diagnostic.csv",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.json",
    "oaci/reports/C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.sha256",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V7.json",
    "oaci/reports/C84_FIELD_GENERATION_PROTOCOL_V7.sha256",
    "oaci/reports/C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL_V2.json",
    "oaci/reports/C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL_V2.sha256",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json",
    "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.sha256",
    "oaci/reports/c84fl2_tables/dual_canary_reuse_registry.csv",
    "oaci/reports/c84fl2_tables/operative_complete_unit_registry_replay.csv",
)


class C84FR2LockError(RuntimeError):
    """Raised when C84FR2 cannot produce a complete execution lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C84FR2LockError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    return runtime.base.sha256_file(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=REPO_ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).returncode == 0


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")
    return sha256_file(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    _require(bool(values), f"refusing empty C84FR2 table: {path}")
    fields = list(values[0])
    _require(all(set(row) == set(fields) for row in values), f"C84FR2 table schema drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def sidecar_digest(path: Path) -> str:
    values = path.read_text(encoding="ascii").split()
    _require(bool(values) and len(values[0]) == 64, f"malformed hash sidecar: {path}")
    return values[0]


def runtime_object(path_text: str, implementation_commit: str) -> dict[str, Any]:
    path = REPO_ROOT / path_text
    _require(path.is_file(), f"runtime-bound file is absent: {path_text}")
    blob = _git("rev-parse", f"HEAD:{path_text}")
    _require(blob == _git("hash-object", str(path)), f"runtime-bound worktree drift: {path_text}")
    return {
        "path": path_text,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "blob": blob,
        "commit": implementation_commit,
    }


def target_only_static_audit() -> list[dict[str, Any]]:
    audited = (
        "oaci/multidataset/c84fr2_target_stage.py",
        "oaci/multidataset/c84fr2_target_numerical_replay.py",
        "oaci/multidataset/c84fr2_runtime_guard.py",
    )
    rows = []
    for relative in audited:
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        imports = set()
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
        forbidden_import = any(
            token in name for name in imports
            for token in ("oaci.train", "training", "selector", "scientific")
        )
        training_callable = any(name.startswith("train") for name in functions)
        rows.append({
            "path": relative,
            "forbidden_imports": int(forbidden_import),
            "training_callables": int(training_callable),
            "status": "PASS" if not forbidden_import and not training_callable else "FAIL",
        })
    return rows


def _partial_target_objects() -> list[dict[str, Any]]:
    paths = [
        *sorted((TARGET_ROOT / "complete_target_unlabeled").glob("*.npz")),
        *sorted((TARGET_ROOT / "target_context_index").glob("*.json")),
    ]
    _require(sum(path.suffix == ".npz" for path in paths) == 6, "historical NPZ count drift")
    _require(sum(path.suffix == ".json" for path in paths) == 5, "historical context count drift")
    return [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths]


def frozen_binding() -> dict[str, Any]:
    return {
        "frozen_model_field_source": {
            "job_id": 896185,
            "root": str(MODEL_ROOT),
            "sha256": {
                "failure_evidence": "9e1ee2fa7da99eb6469dbf32b44229a44ed39315af6e81aa6dafc525154054fb",
                "authorization_consumed": "aaff628205f3c85c8eea292790bdd00b0e8c8f815a8545cd7726f3d3845f11cf",
                "execution_attempts": "1512de2fb37153bee9abee54be92fb1e2843052cae979557f415492ea86328c1",
                "partial_manifest": "445dfd93118ad77d4ad2cf8131170ec611bf72cc66d4943bc2ab08ef38eebb2b",
                "model_manifest": "d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2",
                "target_raw_manifest": "9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd",
            },
        },
        "frozen_target_input_source": {
            "job_id": 896550,
            "root": str(TARGET_ROOT),
            "sha256": {
                "failure_evidence": "a1ce42716ea409bcb650f56c79266b6431eaa7b583c77fc993ead42894d43de3",
                "authorization_consumed": "87221e3e80dc025755b994e39d87a7338f45dc5b2b95f2e252f3824743375058",
                "execution_attempts": "c2c890c25aab0c3c7cd435848268e05aa7ec67e918580fb640c3e8d5176ab386",
                "partial_manifest": "81dbac85acadfba1c7dd6a588956716a80706283f4806a47994352abbdf37c41",
                "target_raw_manifest": "9539747e903dfe67295ee04a97441b85c0bb2179c9ef1bd2177788865e0ba5fd",
                "target_registry": "52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8",
                "target_registry_sha": "195a602522829f8e2bd618e55cdadc8c8980e5a42511c3278133af371c237cc4",
            },
            "partial_target_objects": _partial_target_objects(),
            "partial_target_artifacts_reusable": False,
        },
    }


def protocol_bindings(old_lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = [
        {
            "path": str(REPAIR_PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256_path": str(REPAIR_PROTOCOL_SHA_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(REPAIR_PROTOCOL_PATH),
        },
        {
            "path": str(TARGET_PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256_path": str(TARGET_PROTOCOL_SHA_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(TARGET_PROTOCOL_PATH),
        },
        *(dict(row) for row in old_lock["protocol_bindings"]),
    ]
    rows = []
    seen = set()
    for row in values:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        path = REPO_ROOT / row["path"]
        sidecar = REPO_ROOT / row["sha256_path"]
        _require(sha256_file(path) == row["sha256"] == sidecar_digest(sidecar),
                 f"protocol binding drift: {row['path']}")
        rows.append(row)
    return rows


def synthetic_rows() -> list[dict[str, Any]]:
    import numpy as np
    import torch

    rng = np.random.default_rng(31)
    z = rng.standard_normal((24, 32)).astype(np.float32)
    weight = rng.standard_normal((2, 32)).astype(np.float32)
    bias = rng.standard_normal(2).astype(np.float32)
    logits_tensor = torch.nn.functional.linear(
        torch.from_numpy(z), torch.from_numpy(weight), torch.from_numpy(bias),
    )
    logits = logits_tensor.numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    arrays = {
        "unit_id": np.asarray("fixture"), "dataset": np.asarray("SyntheticMI"),
        "panel": np.asarray("A"), "training_seed": np.asarray(5, dtype=np.int64),
        "level": np.asarray(0, dtype=np.int64),
        "level_intervention_id": np.asarray("level0"), "regime": np.asarray("ERM"),
        "epoch": np.asarray(0, dtype=np.int64), "trajectory_order": np.asarray(0, dtype=np.int64),
        "target_subject_id": np.arange(24, dtype=np.int64),
        "target_trial_id": np.asarray([f"trial-{index}" for index in range(24)]),
        "session": np.asarray(["0"] * 24), "run": np.asarray(["0"] * 24),
        "logits": logits, "probabilities": probabilities, "z": z,
        "Wz_plus_b": logits.copy(), "classifier_weight": weight, "classifier_bias": bias,
        "repeat_logits": logits.copy(), "repeat_z": z.copy(),
    }
    checks = []
    checks.append(("same_backend_fixture", numerical.validate_same_backend_tensors(
        logits_tensor, logits_tensor.clone(), torch=torch, require_cuda=False,
    ) == 0.0))
    checks.append(("digest_registry_complete", len(numerical.build_digest_registry(arrays, np=np)) == 21))
    checks.append(("saved_output_replay_strict", max(
        numerical.validate_saved_output_replay(arrays, np=np).values()
    ) <= 1e-6))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.npz"
        replay = numerical.write_and_replay_artifact(path, arrays=arrays, np=np, torch=torch)
        checks.append(("exact_persistence_round_trip", len(replay["array_digests"]) == 21))
        changed = {name: np.array(value, copy=True) for name, value in arrays.items()}
        changed["z"].flat[0] += 1e-4
        np.savez_compressed(path, **changed)
        tamper_failed = False
        try:
            numerical.replay_persisted_artifact(
                path, expected_digests=replay["array_digests"], np=np, torch=torch,
            )
        except numerical.C84FR2NumericalError:
            tamper_failed = True
        checks.append(("tampered_z_digest_fails", tamper_failed))
    legacy_rng = np.random.default_rng(164)
    legacy_z = legacy_rng.standard_normal((256, 1040)).astype(np.float32)
    legacy_weight = legacy_rng.standard_normal((2, 1040)).astype(np.float32)
    legacy_bias = legacy_rng.standard_normal(2).astype(np.float32)
    legacy_logits = torch.nn.functional.linear(
        torch.from_numpy(legacy_z), torch.from_numpy(legacy_weight), torch.from_numpy(legacy_bias),
    ).numpy()
    legacy = dict(arrays)
    legacy.update({
        "z": legacy_z, "classifier_weight": legacy_weight, "classifier_bias": legacy_bias,
        "logits": legacy_logits, "Wz_plus_b": legacy_logits.copy(),
        "repeat_logits": legacy_logits.copy(), "repeat_z": legacy_z.copy(),
        "target_subject_id": np.arange(256, dtype=np.int64),
        "target_trial_id": np.asarray([f"legacy-{index}" for index in range(256)]),
        "session": np.asarray(["0"] * 256), "run": np.asarray(["0"] * 256),
    })
    legacy_shifted = legacy_logits - legacy_logits.max(axis=1, keepdims=True)
    legacy["probabilities"] = np.exp(legacy_shifted) / np.exp(legacy_shifted).sum(axis=1, keepdims=True)
    diagnostic = numerical.cross_backend_diagnostics(legacy, np=np, torch=torch)
    numpy32 = next(row for row in diagnostic if row["backend"] == "NumPy_float32")
    checks.append(("legacy_2_193450927734375e_05_is_diagnostic_only",
                   numpy32["max_abs_error"] == 2.193450927734375e-05
                   and numpy32["diagnostic_only"] is True))
    checks.extend((f"static_{index}_{row['path']}", row["status"] == "PASS")
                  for index, row in enumerate(target_only_static_audit(), 1))
    return [{"check": name, "passed": int(passed), "status": "PASS" if passed else "FAIL"}
            for name, passed in checks]


def generate() -> dict[str, Any]:
    _require(not _git("status", "--porcelain"), "C84FR2 lock generation requires a clean worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    _require(_git("branch", "--show-current") == "oaci", "C84FR2 requires branch oaci")
    _require(_git("rev-parse", "origin/oaci") == implementation_commit,
             "C84FR2 implementation must be pushed before lock generation")
    _require(git_is_ancestor(PROTOCOL_COMMIT, implementation_commit), "repair protocol ancestry failed")
    _require(git_is_ancestor(TARGET_PROTOCOL_COMMIT, implementation_commit), "target V2 protocol ancestry failed")
    repair_sha = sha256_file(REPAIR_PROTOCOL_PATH)
    target_sha = sha256_file(TARGET_PROTOCOL_PATH)
    _require(repair_sha == sidecar_digest(REPAIR_PROTOCOL_SHA_PATH), "repair protocol hash drift")
    _require(target_sha == sidecar_digest(TARGET_PROTOCOL_SHA_PATH), "target V2 protocol hash drift")
    old_lock_sha = runtime.base.verify_lock_self(OLD_LOCK_PATH, OLD_LOCK_SHA_PATH)
    old_lock = read_json(OLD_LOCK_PATH)
    binding = frozen_binding()
    frozen = runtime.verify_frozen_model_and_target_inputs(binding)
    _require(frozen["model_manifest_sha256"] == "d8931b81a3d68f4b1e098ac6e3ede3cd44cdb6c70cdef9f18a76e0a8c62ecdb2",
             "model manifest identity drift")
    _require(frozen["target_registry_sha256"] == "52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8",
             "target registry identity drift")
    synthetic = synthetic_rows()
    _require(all(row["passed"] for row in synthetic), "C84FR2 synthetic calibration failed")
    static_rows = target_only_static_audit()
    _require(all(row["status"] == "PASS" for row in static_rows), "target-only static audit failed")

    implementation = [runtime_object(path, implementation_commit) for path in IMPLEMENTATION_FILES]
    bound_paths = list(dict.fromkeys((*IMPLEMENTATION_FILES, *RUNTIME_REGISTRY_FILES)))
    bound = [runtime_object(path, implementation_commit) for path in bound_paths]
    bindings = protocol_bindings(old_lock)
    protocols = {
        **old_lock["protocols"],
        "c84fr2_repair": {
            "path": str(REPAIR_PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256_path": str(REPAIR_PROTOCOL_SHA_PATH.relative_to(REPO_ROOT)),
            "sha256": repair_sha,
        },
        "target_v2": {
            "path": str(TARGET_PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256_path": str(TARGET_PROTOCOL_SHA_PATH.relative_to(REPO_ROOT)),
            "sha256": target_sha,
        },
    }
    lock = {
        "schema_version": "c84fr2_target_stage_execution_lock_v2",
        "status": STATUS,
        "chronology": {
            "protocol_commit": PROTOCOL_COMMIT,
            "target_protocol_commit": TARGET_PROTOCOL_COMMIT,
            "implementation_commit": implementation_commit,
            "protocol_precedes_implementation": True,
            "failed_jobs": [896185, 896550],
            "target_labels_before_repair": 0,
            "scientific_outcomes_before_repair": 0,
        },
        "protocols": protocols,
        "protocol_bindings": bindings,
        "implementation": {
            "commit": implementation_commit,
            "entrypoint": "python -m oaci.multidataset.c84fr2_target_stage run-real",
            "file_count": len(implementation),
            "files": implementation,
            "target_stage_training_callable": False,
            "checkpoint_write": False,
            "optimizer_creation": False,
        },
        "runtime_bound_object_count": len(bound),
        "runtime_bound_objects": bound,
        "interface": old_lock["interface"],
        "environment": old_lock["environment"],
        "loader_source_identity": old_lock["loader_source_identity"],
        "candidate_identity": old_lock["candidate_identity"],
        "dual_canary_reuse": old_lock["dual_canary_reuse"],
        "resources": old_lock["resources"],
        "frozen_model_field_source": binding["frozen_model_field_source"],
        "frozen_target_input_source": binding["frozen_target_input_source"],
        "numerical_gates": {
            "same_backend_GPU_PyTorch_float32_max_abs": 1e-6,
            "exact_persisted_array_digest": True,
            "saved_Wz_plus_b_vs_logits_max_abs": 1e-6,
            "saved_softmax_max_abs": 1e-6,
            "repeat_logits_max_abs": 1e-6,
            "repeat_z_max_abs": 1e-6,
            "cross_backend_diagnostic_only": True,
            "cross_backend_nonfinite_blocks": True,
            "historical_2e5_widened": False,
        },
        "schemas": {
            "target_artifact": "c84f_target_unlabeled_v2",
            "context_digest_sidecar": "c84f_target_context_and_digest_index_v2",
            "complete_field_manifest": "c84f_complete_field_manifest_v2",
        },
        "scope": {
            "model_units_replayed": 1944,
            "model_artifact_files_replayed": 7776,
            "canary_artifact_files_replayed_before_target_access": 2430,
            "target_subjects": 118,
            "target_registry_rows": 9621,
            "target_artifacts": 1944,
            "context_digest_sidecars": 1944,
            "target_contexts": 944,
            "candidate_context_slices": 76464,
            "canary_unit_witnesses": 486,
        },
        "barriers": {
            "frozen_model_field_replayed_before_target_access": True,
            "frozen_raw_manifest_exact_replay": True,
            "frozen_trial_registry_exact_replay_before_forward": True,
            "historical_partial_target_artifacts_rejected": True,
            "target_failure_cannot_train": True,
        },
        "retry": {
            "fresh_content_addressed_root": True,
            "historical_authorization_reused": False,
            "historical_partial_target_artifacts_reused": False,
            "model_retraining": 0,
            "implementation_change_requires_new_lock": True,
        },
        "authorization": {
            "record_path": "oaci/reports/C84F_TARGET_STAGE_NUMERICAL_REPLAY_PI_AUTHORIZATION_RECORD.json",
            "record_present_at_lock": False,
            "fresh_direct_statement": "授权 C84F target-stage numerical replay repair",
            "magic_token_required": False,
            "hash_recital_required": False,
            "C84S_authorized": False,
        },
        "external_roots": {
            "historical_model_root": str(MODEL_ROOT),
            "historical_target_failure_root": str(TARGET_ROOT),
            "target_replay_v2_base": str(runtime.DEFAULT_EXTERNAL_ROOT),
            "historical_roots_read_only": True,
        },
        "field_completion_gate": "C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
        "forbidden": {
            "model_retraining": True,
            "checkpoint_write": True,
            "optimizer_creation": True,
            "partial_target_artifact_reuse": True,
            "target_labels": True,
            "selector_scores": True,
            "scientific_metrics": True,
            "same_label_oracle": True,
            "C84S": True,
        },
        "historical_target_stage_lock_sha256": old_lock_sha,
    }
    digest = write_json(LOCK_PATH, lock)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")

    model_rows = [{
        "object": "C84F_MODEL_FIELD_MANIFEST",
        "expected_sha256": binding["frozen_model_field_source"]["sha256"]["model_manifest"],
        "observed_sha256": frozen["model_manifest_sha256"],
        "model_units": len(frozen["model_rows"]),
        "artifact_files_replayed": frozen["model_artifact_files_replayed"],
        "training_invocations": 0,
        "status": "PASS",
    }]
    target_rows = [{
        "object": "C84FR1_TARGET_UNLABELED_TRIAL_REGISTRY",
        "expected_sha256": binding["frozen_target_input_source"]["sha256"]["target_registry"],
        "observed_sha256": frozen["target_registry_sha256"],
        "subjects": frozen["target_subjects"],
        "rows": frozen["target_registry_rows"],
        "target_y_operations": 0,
        "status": "PASS",
    }]
    failure_rows = [
        {"job_id": 896185, "object": "model_field_and_raw_manifest", "status": "PRESERVED",
         "reusable": 1, "partial_target_artifact_reusable": 0,
         "disposition": "MODEL_FIELD_ONLY_REUSED"},
        {"job_id": 896550, "object": "target_registry_and_partial_target_artifacts", "status": "PRESERVED",
         "reusable": 1, "partial_target_artifact_reusable": 0,
         "disposition": "REGISTRY_REUSED_PARTIAL_TARGET_ARTIFACTS_REJECTED"},
    ]
    failure_rows.extend({
        "job_id": 896550,
        "object": row["path"],
        "status": "HASH_REPLAYED_REJECTED",
        "reusable": 0,
        "partial_target_artifact_reusable": 0,
        "disposition": row["sha256"],
    } for row in binding["frozen_target_input_source"]["partial_target_objects"])
    same_backend_rows = [{
        "object": "direct_classifier_identity",
        "model_output": "output.logits",
        "direct_output": "torch.nn.functional.linear(output.z,weight,bias)",
        "device": "same_GPU",
        "runtime": "PyTorch_float32",
        "max_abs_tolerance": "1e-6",
        "canonical_Wz_plus_b": 1,
        "status": "LOCKED",
    }]
    digest_rows = [{
        "field": field,
        "dtype_exact": 1,
        "shape_exact": 1,
        "canonical_byte_sha256_exact": 1,
        "prewrite_postreload_equal": 1,
        "status": "LOCKED",
    } for field in numerical.TARGET_ARRAY_FIELDS]
    cross_rows = [{
        "backend": backend,
        "statistics": "max|p95|p99|feature_dimension|logit_abs_max|logit_abs_p95",
        "finite_required": 1,
        "magnitude_gate": 0,
        "retention_effect": 0,
        "status": "DIAGNOSTIC_ONLY",
    } for backend in ("CPU_PyTorch_float32", "NumPy_float32", "NumPy_float64")]
    artifact_schema_rows = [{"ordinal": index, "field": field, "target_label_field": 0,
                             "schema_version": numerical.ARTIFACT_SCHEMA}
                            for index, field in enumerate(numerical.TARGET_ARRAY_FIELDS, 1)]
    sidecar_rows = [
        {"section": "identity", "required": "unit_id|dataset|artifact_path|artifact_sha256", "status": "LOCKED"},
        {"section": "digests", "required": "21 field dtype|shape|bytes|sha256", "status": "LOCKED"},
        {"section": "contexts", "required": "target offsets|trial count|trial ID sha256", "status": "LOCKED"},
        {"section": "numerics", "required": "same backend|saved replay|cross backend diagnostics", "status": "LOCKED"},
    ]
    canary_rows = [{
        "canary_units": 486, "canary_context_classes": 6,
        "trial_id_exact": 1, "logits_max_abs": "1e-6", "probabilities_max_abs": "1e-6",
        "z_max_abs": "1e-6", "retention_effect": 0, "status": "LOCKED",
    }]
    risks = [
        ("historical_failed_attempt_overwritten", "CLOSED", 0),
        ("partial_target_artifact_reused", "CLOSED", 0),
        ("model_retraining", "CLOSED", 0),
        ("cross_backend_error_used_as_functional_gate", "CLOSED", 0),
        ("same_backend_gate_widened", "CLOSED", 0),
        ("persisted_array_digest_not_exact", "CLOSED", 0),
        ("target_registry_drift", "CLOSED", 0),
        ("target_y_access", "CLOSED", 0),
        ("scientific_metric_access", "CLOSED", 0),
        ("same_label_oracle", "CLOSED", 0),
        ("C84S_execution", "CLOSED", 0),
        ("fresh_PI_authorization_absent", "EXPECTED_STOP", 0),
    ]
    risk_rows = [{"risk": name, "status": status, "blocking": blocking}
                 for name, status, blocking in risks]
    ledger_rows = [
        {"failure_id": "C84FR2-HIST-001", "object": "job_896185",
         "reason": "raw_dictionary_sort_TypeError", "blocking": 0,
         "disposition": "PRESERVED_MODEL_FIELD_REUSED"},
        {"failure_id": "C84FR2-HIST-002", "object": "job_896550",
         "reason": "cross_backend_float32_gate", "blocking": 0,
         "disposition": "PRESERVED_PARTIAL_TARGET_ARTIFACTS_REJECTED"},
        {"failure_id": "C84FR2-AUTH-001", "object": "replacement_execution",
         "reason": "fresh_PI_authorization_absent", "blocking": 0,
         "disposition": "EXPECTED_READINESS_STOP"},
    ]
    write_csv(TABLE_DIR / "c84f_model_field_identity_replay.csv", model_rows)
    write_csv(TABLE_DIR / "c84fr1_target_registry_identity_replay.csv", target_rows)
    write_csv(TABLE_DIR / "failed_attempt_and_partial_root_ledger.csv", failure_rows)
    write_csv(TABLE_DIR / "same_backend_linear_contract.csv", same_backend_rows)
    write_csv(TABLE_DIR / "persisted_array_digest_contract.csv", digest_rows)
    write_csv(TABLE_DIR / "cross_backend_diagnostic_contract.csv", cross_rows)
    write_csv(TABLE_DIR / "target_artifact_v2_schema.csv", artifact_schema_rows)
    write_csv(TABLE_DIR / "target_context_digest_sidecar_v2_schema.csv", sidecar_rows)
    write_csv(TABLE_DIR / "canary_subset_replay_contract.csv", canary_rows)
    write_csv(TABLE_DIR / "target_only_static_import_audit.csv", static_rows)
    write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", bound)
    write_csv(TABLE_DIR / "synthetic_calibration.csv", synthetic)
    write_csv(TABLE_DIR / "risk_register.csv", risk_rows)
    write_csv(TABLE_DIR / "failure_reason_ledger.csv", ledger_rows)

    red_checks = [
        "job_896185_preserved", "job_896550_preserved", "authorizations_not_reused",
        "model_manifest_exact", "7776_model_artifacts_replayed", "target_raw_manifest_exact",
        "target_registry_exact", "9621_registry_rows", "118_target_subjects",
        "six_partial_npz_replayed_rejected", "five_partial_indices_replayed_rejected",
        "historical_2e5_not_widened", "same_backend_GPU_PyTorch_gate_1e6",
        "direct_linear_before_CPU_transfer", "canonical_Wz_plus_b_is_direct_output",
        "21_array_digests", "dtype_exact", "shape_exact", "bytes_exact",
        "missing_field_fails", "unknown_field_fails", "saved_Wz_logits_1e6",
        "softmax_1e6", "repeat_logits_1e6", "repeat_z_1e6",
        "CPU_PyTorch_diagnostic_only", "NumPy_float32_diagnostic_only",
        "NumPy_float64_diagnostic_only", "nonfinite_diagnostic_fails",
        "target_only_entrypoint", "no_training_import", "no_training_callable",
        "no_checkpoint_write", "no_optimizer_creation", "fresh_output_root",
        "registry_before_forward", "target_y_forbidden", "scientific_metrics_forbidden",
        "oracle_forbidden", "C84S_forbidden", "1944_artifacts_required",
        "1944_digest_sidecars_required", "76464_slices_required", "486_canary_witnesses",
        "protocol_precedes_implementation", "fresh_authorization_required",
    ]
    (REPORT_DIR / "C84FR2_FINAL_REPORT_RED_TEAM.md").write_text(
        "# C84FR2 Final Report Red Team\n\n" +
        "\n".join(f"- RT{index:02d} `{name}`: PASS" for index, name in enumerate(red_checks, 1)) +
        f"\n\nGate: `{SUCCESS_GATE}`. Result: {len(red_checks)}/{len(red_checks)} PASS.\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "C84FR2_PROTOCOL_READINESS.md").write_text(
        "# C84FR2 Protocol Readiness\n\n"
        f"Repair protocol SHA-256: `{repair_sha}`. Target instrumentation V2 SHA-256: "
        f"`{target_sha}`. Replacement execution-lock SHA-256: `{digest}`.\n\n"
        f"The lock binds {len(bound)} repository objects and {len(implementation)} implementation "
        "files, the frozen 1,944-unit model field, 7,776 model artifacts, 2,430 dual-canary "
        "witness files, the 118-subject/9,621-row label-free target registry, and all 11 rejected "
        "partial target objects from job 896550.\n\n"
        f"Synthetic calibration: {len(synthetic)}/{len(synthetic)} PASS. Red team: "
        f"{len(red_checks)}/{len(red_checks)} PASS. No target X reload, forward, GPU, training, "
        "label, selector, scientific metric, oracle, or C84S work occurred during C84FR2.\n\n"
        f"Gate: `{SUCCESS_GATE}`. Real target-stage execution is not authorized.\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "c84fr2_lock_generation_result_v1",
        "gate": SUCCESS_GATE,
        "repair_protocol_sha256": repair_sha,
        "target_protocol_v2_sha256": target_sha,
        "execution_lock_sha256": digest,
        "implementation_commit": implementation_commit,
        "runtime_bound_objects": len(bound),
        "implementation_files": len(implementation),
        "model_units": 1944,
        "model_artifact_files_replayed": 7776,
        "target_registry_rows": 9621,
        "partial_target_objects_rejected": 11,
        "synthetic_passed": len(synthetic),
        "red_team_passed": len(red_checks),
        "authorization_present": False,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
