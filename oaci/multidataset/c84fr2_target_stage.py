"""Target-only C84FR2 execution using same-backend and exact persistence gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from . import c84f_field_manifest as manifests
from . import c84f_target_instrumentation as target_inputs
from . import c84fl2_protocol as field_protocol
from . import c84fr2_target_numerical_replay as numerical


FIELD_GATE = "C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED"
COMPLETE_MANIFEST_SCHEMA = "c84f_complete_field_manifest_v2"
COMPLETE_MANIFEST_NAME = "C84F_COMPLETE_FIELD_MANIFEST.json"
COMPLETE_MANIFEST_SHA_NAME = "C84F_COMPLETE_FIELD_MANIFEST.sha256"


class C84FR2TargetStageError(RuntimeError):
    """Raised when the target-only V2 execution deviates from its lock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C84FR2TargetStageError(message)


def _protected_loader_objects() -> tuple[dict[str, Any], tuple[Any, Any, Any, Any]]:
    from moabb.datasets import Cho2017, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery

    objects = {
        "moabb.datasets.Lee2019_MI": Lee2019_MI,
        "moabb.datasets.Cho2017": Cho2017,
        "moabb.datasets.PhysionetMI": PhysionetMI,
        "moabb.paradigms.MotorImagery": MotorImagery,
    }
    return objects, (Lee2019_MI, Cho2017, PhysionetMI, MotorImagery)


def target_raw_manifest_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    values.sort(key=lambda row: (str(row["dataset"]), str(row["path"])))
    return {
        "schema_version": "c84f_target_raw_input_manifest_v1",
        "files": values,
        "file_count": len(values),
        "target_labels": 0,
    }


def require_exact_json_payload(
    observed: Mapping[str, Any], historical_path: str | Path, *, object_name: str,
) -> dict[str, Any]:
    historical = manifests.read_json(historical_path)
    if manifests.canonical_bytes(dict(observed)) != manifests.canonical_bytes(historical):
        raise C84FR2TargetStageError(f"{object_name} differs from the frozen C84FR1 object")
    return {
        "path": str(historical_path),
        "sha256": manifests.sha256_file(historical_path),
        "exact_replay": True,
    }


def require_exact_trial_registry(
    trial_rows: Sequence[Mapping[str, Any]], historical_path: str | Path,
) -> dict[str, Any]:
    historical = manifests.read_json(historical_path)
    manifests.validate_target_trial_rows(trial_rows)
    if manifests.canonical_bytes(list(trial_rows)) != manifests.canonical_bytes(historical.get("trials", ())):
        raise C84FR2TargetStageError("target trial registry rows/order differ from frozen C84FR1 registry")
    if len(trial_rows) != 9621 or int(historical["complete_gate"]["target_subjects"]) != 118:
        raise C84FR2TargetStageError("frozen target trial registry arithmetic drift")
    return {
        "path": str(historical_path),
        "sha256": manifests.sha256_file(historical_path),
        "rows": len(trial_rows),
        "subjects": 118,
        "exact_replay": True,
    }


def _copy_exact(source: str | Path, destination: str | Path, *, expected_sha256: str) -> str:
    source_path = Path(source)
    target = Path(destination)
    _require(source_path.is_file(), f"frozen source object is absent: {source_path}")
    _require(not target.exists(), f"replacement object overwrite is forbidden: {target}")
    _require(manifests.sha256_file(source_path) == expected_sha256, "frozen source hash drift")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.copying")
    _require(not temporary.exists(), f"stale copy staging file exists: {temporary}")
    try:
        shutil.copyfile(source_path, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _require(manifests.sha256_file(temporary) == expected_sha256, "copied object hash drift")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifests.sha256_file(target)


def _model_factory() -> Any:
    from oaci.models import build_model

    return build_model(
        "shallow_convnet", in_chans=20, in_times=480, n_classes=2,
        temporal_filters=40, temporal_kernel_samples=25,
        pool_kernel_samples=75, pool_stride_samples=15,
        dropout=0.5, safe_log_eps=1e-6,
    )


def _load_checkpoint_state(path: Path, torch: Any) -> Mapping[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    _require(isinstance(state, Mapping) and bool(state), f"checkpoint state is invalid: {path}")
    return state


def _require_checkpoint_classifier_identity(model: Any, state: Mapping[str, Any], *, torch: Any) -> None:
    required = ("classifier.weight", "classifier.bias")
    _require(all(name in state for name in required), "checkpoint classifier tensors are absent")
    observed = (
        model.classifier.weight.detach().cpu(),
        model.classifier.bias.detach().cpu(),
    )
    expected = tuple(state[name].detach().cpu() for name in required)
    _require(
        all(left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
            for left, right in zip(observed, expected)),
        "checkpoint-loaded classifier byte identity failed",
    )


def _require_artifact_identity(
    arrays: Mapping[str, Any], unit: Mapping[str, Any], trial_ids: Sequence[str], *, np: Any,
) -> None:
    expected = {
        "unit_id": unit["unit_id"],
        "dataset": unit["dataset"],
        "panel": unit["panel"],
        "training_seed": int(unit["training_seed"]),
        "level": int(unit["level"]),
        "level_intervention_id": unit["level_intervention_id"],
        "regime": unit["regime"],
        "epoch": int(unit["epoch"]),
        "trajectory_order": int(unit["trajectory_order"]),
    }
    mismatch = {
        name: (np.asarray(arrays[name]).item(), value)
        for name, value in expected.items()
        if str(np.asarray(arrays[name]).item()) != str(value)
    }
    _require(not mismatch, f"target artifact scalar identity drift: {mismatch}")
    observed_trials = tuple(np.asarray(arrays["target_trial_id"]).astype(str))
    _require(observed_trials == tuple(map(str, trial_ids)), "target artifact trial order drift")
    rows = len(observed_trials)
    for name in ("target_subject_id", "session", "run", "logits", "probabilities", "z",
                 "Wz_plus_b", "repeat_logits", "repeat_z"):
        _require(int(np.asarray(arrays[name]).shape[0]) == rows,
                 f"target artifact row-count drift: {name}")


def _forward_same_backend(model: Any, X: Any, *, torch: Any) -> dict[str, Any]:
    logits_parts = []
    probability_parts = []
    feature_parts = []
    direct_parts = []
    maximum = 0.0
    with torch.inference_mode():
        for start in range(0, X.shape[0], 1024):
            tensor = torch.as_tensor(X[start:start + 1024], device="cuda:0")
            output = model(tensor)
            direct = torch.nn.functional.linear(
                output.z, model.classifier.weight, model.classifier.bias,
            )
            maximum = max(maximum, numerical.validate_same_backend_tensors(
                output.logits, direct, torch=torch, require_cuda=True,
            ))
            logits_parts.append(output.logits.detach().cpu())
            probability_parts.append(torch.softmax(output.logits, dim=1).detach().cpu())
            feature_parts.append(output.z.detach().cpu())
            direct_parts.append(direct.detach().cpu())
    return {
        "logits": torch.cat(logits_parts),
        "probabilities": torch.cat(probability_parts),
        "z": torch.cat(feature_parts),
        "Wz_plus_b": torch.cat(direct_parts),
        "same_backend_max_abs_error": maximum,
    }


def _concatenate_views(views: Sequence[Any], *, np: Any) -> dict[str, Any]:
    _require(bool(views), "cannot instrument an empty target population")
    return {
        "X": np.concatenate([view.X for view in views], axis=0),
        "subject": np.concatenate([
            np.full(len(view.trial_id), view.subject_id, dtype=np.int64) for view in views
        ]),
        "trial_id": tuple(value for view in views for value in view.trial_id),
        "session": tuple(value for view in views for value in view.session),
        "run": tuple(value for view in views for value in view.run),
    }


def _context_rows(views: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    start = 0
    for view in views:
        end = start + len(view.trial_id)
        rows.append({
            "dataset": view.dataset,
            "target_subject_id": int(view.subject_id),
            "row_start_inclusive": start,
            "row_end_exclusive": end,
            "trial_count": end - start,
            "trial_id_sha256": hashlib.sha256(
                manifests.canonical_bytes(list(view.trial_id))
            ).hexdigest(),
        })
        start = end
    return rows


def instrument_unit_v2(
    *,
    unit: Mapping[str, Any],
    views: Sequence[Any],
    artifact_path: Path,
    context_digest_path: Path,
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    combined = _concatenate_views(views, np=np)
    model = _model_factory()
    checkpoint_state = _load_checkpoint_state(Path(unit["checkpoint_path"]), torch)
    model.load_state_dict(checkpoint_state)
    _require_checkpoint_classifier_identity(model, checkpoint_state, torch=torch)
    model.eval().to("cuda:0")
    first = _forward_same_backend(model, combined["X"], torch=torch)
    repeat = _forward_same_backend(model, combined["X"], torch=torch)
    weight = model.classifier.weight.detach().cpu()
    bias = model.classifier.bias.detach().cpu()
    arrays = {
        "unit_id": np.asarray(unit["unit_id"]),
        "dataset": np.asarray(unit["dataset"]),
        "panel": np.asarray(unit["panel"]),
        "training_seed": np.asarray(int(unit["training_seed"]), dtype=np.int64),
        "level": np.asarray(int(unit["level"]), dtype=np.int64),
        "level_intervention_id": np.asarray(unit["level_intervention_id"]),
        "regime": np.asarray(unit["regime"]),
        "epoch": np.asarray(int(unit["epoch"]), dtype=np.int64),
        "trajectory_order": np.asarray(int(unit["trajectory_order"]), dtype=np.int64),
        "target_subject_id": combined["subject"],
        "target_trial_id": np.asarray(combined["trial_id"], dtype=str),
        "session": np.asarray(combined["session"], dtype=str),
        "run": np.asarray(combined["run"], dtype=str),
        "logits": first["logits"].numpy(),
        "probabilities": first["probabilities"].numpy(),
        "z": first["z"].numpy(),
        "Wz_plus_b": first["Wz_plus_b"].numpy(),
        "classifier_weight": weight.numpy(),
        "classifier_bias": bias.numpy(),
        "repeat_logits": repeat["logits"].numpy(),
        "repeat_z": repeat["z"].numpy(),
    }
    _require_artifact_identity(arrays, unit, combined["trial_id"], np=np)
    persisted = numerical.write_and_replay_artifact(
        artifact_path, arrays=arrays, np=np, torch=torch,
    )
    same_backend_max = max(
        float(first["same_backend_max_abs_error"]),
        float(repeat["same_backend_max_abs_error"]),
    )
    contexts = _context_rows(views)
    sidecar = {
        "schema_version": numerical.CONTEXT_DIGEST_SCHEMA,
        "artifact_schema_version": numerical.ARTIFACT_SCHEMA,
        "unit_id": str(unit["unit_id"]),
        "dataset": str(unit["dataset"]),
        "artifact_path": str(artifact_path),
        "artifact_sha256": persisted["artifact_sha256"],
        "array_digests": persisted["array_digests"],
        "same_backend_functional_identity": {
            "backend": "GPU_PyTorch_float32",
            "max_abs_error": same_backend_max,
            "tolerance": numerical.SAME_BACKEND_TOLERANCE,
            "passed": True,
        },
        "saved_output_replay": persisted["saved_output_replay"],
        "cross_backend_diagnostics": persisted["cross_backend_diagnostics"],
        "contexts": contexts,
        "context_count": len(contexts),
        "trial_rows": int(persisted["rows"]),
        "target_label_fields": 0,
        "target_y_operations": 0,
    }
    _require(not context_digest_path.exists(), "context/digest sidecar overwrite is forbidden")
    sidecar_sha = manifests.write_json_atomic(context_digest_path, sidecar)
    return persisted, {**sidecar, "path": str(context_digest_path), "sha256": sidecar_sha}


def _replay_canary_subset(
    *, complete_path: str | Path, canary_path: str | Path, canary_subject: int, np: Any,
) -> dict[str, Any]:
    return target_inputs.replay_canary_subset(
        complete_path=complete_path,
        canary_path=canary_path,
        canary_subject=canary_subject,
        np=np,
    )


def _aggregate_numerics(descriptors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    same_backend = 0.0
    saved = {
        "saved_Wz_plus_b_vs_logits_max_abs_error": 0.0,
        "saved_softmax_max_abs_error": 0.0,
        "repeat_logits_max_abs_error": 0.0,
        "repeat_z_max_abs_error": 0.0,
    }
    cross = {
        backend: {"max_abs_error": 0.0, "max_p95_abs_error": 0.0, "max_p99_abs_error": 0.0}
        for backend in ("CPU_PyTorch_float32", "NumPy_float32", "NumPy_float64")
    }
    for row in descriptors:
        numerical_identity = row["numerical_identity"]
        same_backend = max(same_backend, float(numerical_identity["same_backend_max_abs_error"]))
        for name in saved:
            saved[name] = max(saved[name], float(numerical_identity["saved_output_replay"][name]))
        for diagnostic in numerical_identity["cross_backend_diagnostics"]:
            backend = str(diagnostic["backend"])
            _require(backend in cross and bool(diagnostic["finite"]), "cross-backend diagnostic drift")
            cross[backend]["max_abs_error"] = max(
                cross[backend]["max_abs_error"], float(diagnostic["max_abs_error"]),
            )
            cross[backend]["max_p95_abs_error"] = max(
                cross[backend]["max_p95_abs_error"], float(diagnostic["p95_abs_error"]),
            )
            cross[backend]["max_p99_abs_error"] = max(
                cross[backend]["max_p99_abs_error"], float(diagnostic["p99_abs_error"]),
            )
    return {
        "same_backend_max_abs_error": same_backend,
        "same_backend_tolerance": numerical.SAME_BACKEND_TOLERANCE,
        **saved,
        "strict_replay_tolerance": numerical.STRICT_REPLAY_TOLERANCE,
        "cross_backend_diagnostic_only": True,
        "cross_backend_summary": cross,
    }


def instrument_complete_field_v2(
    *,
    model_rows: Sequence[Mapping[str, Any]],
    views: Mapping[str, Sequence[Any]],
    reuse_rows: Mapping[str, Mapping[str, Any]],
    output_root: str | Path,
    model_manifest_path: str | Path,
    model_manifest_sha_path: str | Path,
    torch: Any,
    np: Any,
    ledger: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_inputs.require_model_field_barrier(model_manifest_path, model_manifest_sha_path)
    _require(len(model_rows) == 1944, "complete V2 target instrumentation requires 1,944 models")
    root = Path(output_root)
    descriptors = []
    context_slices = 0
    canary_units = 0
    for unit in model_rows:
        dataset_views = tuple(views[unit["dataset"]])
        artifact_path = root / "complete_target_unlabeled_v2" / f"{unit['unit_id']}.npz"
        sidecar_path = root / "target_context_digest_index_v2" / f"{unit['unit_id']}.json"
        persisted, sidecar = instrument_unit_v2(
            unit=unit,
            views=dataset_views,
            artifact_path=artifact_path,
            context_digest_path=sidecar_path,
            torch=torch,
            np=np,
        )
        historical = reuse_rows.get(str(unit["unit_id"]))
        if historical is None:
            witness = {"required": True, "applicable": False, "passed": True}
        else:
            witness = _replay_canary_subset(
                complete_path=artifact_path,
                canary_path=historical["canary_target_path"],
                canary_subject=field_protocol.CANARY_TARGETS[unit["dataset"]],
                np=np,
            )
            canary_units += 1
        descriptors.append({
            "unit_id": str(unit["unit_id"]),
            "checkpoint": {"path": unit["checkpoint_path"], "sha256": unit["checkpoint_sha256"]},
            "optimizer": {"path": unit["optimizer_path"], "sha256": unit["optimizer_sha256"]},
            "training_sidecar": {"path": unit["sidecar_path"], "sha256": unit["sidecar_sha256"]},
            "source_audit": {"path": unit["source_audit_path"], "sha256": unit["source_audit_sha256"]},
            "complete_target_unlabeled": {
                "path": str(artifact_path), "sha256": persisted["artifact_sha256"],
            },
            "target_context_digest_index": {"path": str(sidecar_path), "sha256": sidecar["sha256"]},
            "interface_id": field_protocol.INTERFACE_ID,
            "level_intervention_id": unit["level_intervention_id"],
            "model_reuse_provenance": unit["reuse_provenance"],
            "target_artifact_provenance": "C84FR2",
            "canary_subset_replay": witness,
            "failed_attempt_provenance": {
                "jobs_preserved": [896185, 896550],
                "partial_target_artifact_reused": False,
            },
            "numerical_identity": {
                "same_backend_max_abs_error": float(sidecar["same_backend_functional_identity"]["max_abs_error"]),
                "saved_output_replay": dict(persisted["saved_output_replay"]),
                "cross_backend_diagnostics": list(persisted["cross_backend_diagnostics"]),
            },
        })
        context_slices += len(dataset_views)
        ledger.increment("target_unlabeled_artifacts")
        ledger.increment("target_context_slices", len(dataset_views))
        if historical is not None:
            ledger.increment("canary_contexts_replayed")
        if len(descriptors) % 81 == 0:
            ledger.publish_partial_manifest("IN_PROGRESS")
    _require(
        len(descriptors) == 1944 and context_slices == 76464 and canary_units == 486,
        f"complete V2 target arithmetic drift: {len(descriptors)}/{context_slices}/{canary_units}",
    )
    numerics = _aggregate_numerics(descriptors)
    _require(numerics["same_backend_max_abs_error"] <= numerical.SAME_BACKEND_TOLERANCE,
             "aggregate same-backend gate failed")
    for name in (
        "saved_Wz_plus_b_vs_logits_max_abs_error", "saved_softmax_max_abs_error",
        "repeat_logits_max_abs_error", "repeat_z_max_abs_error",
    ):
        _require(numerics[name] <= numerical.STRICT_REPLAY_TOLERANCE, f"aggregate strict gate failed: {name}")
    return descriptors, {
        "target_artifacts": 1944,
        "context_digest_sidecars": 1944,
        "target_contexts": 944,
        "candidate_context_slices": 76464,
        "canary_unit_witnesses": 486,
        "target_label_fields": 0,
        "target_y_operations": 0,
        "target_scientific_metrics": 0,
        "training_invocations": 0,
        **numerics,
    }


def publish_complete_field_manifest_v2(
    root: str | Path,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    model_manifest_sha256: str,
    target_raw_manifest_sha256: str,
    target_trial_registry_sha256: str,
    instrumentation: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(len(descriptors) == 1944, "V2 complete manifest requires 1,944 descriptors")
    unit_ids = [str(row["unit_id"]) for row in descriptors]
    _require(len(set(unit_ids)) == 1944, "V2 complete manifest unit IDs are not unique")
    _require(int(instrumentation.get("target_artifacts", -1)) == 1944, "V2 artifact gate incomplete")
    _require(int(instrumentation.get("context_digest_sidecars", -1)) == 1944,
             "V2 digest-sidecar gate incomplete")
    _require(int(instrumentation.get("candidate_context_slices", -1)) == 76464,
             "V2 context-slice gate incomplete")
    for row in descriptors:
        for name in ("complete_target_unlabeled", "target_context_digest_index"):
            identity = row[name]
            path = Path(identity["path"])
            _require(path.is_file() and manifests.sha256_file(path) == identity["sha256"],
                     f"V2 descriptor object replay failed: {row['unit_id']}/{name}")
        _require(row["failed_attempt_provenance"]["partial_target_artifact_reused"] is False,
                 "historical partial target artifact entered V2 field")
    payload = {
        "schema_version": COMPLETE_MANIFEST_SCHEMA,
        "execution_identity": dict(execution_identity),
        "model_field_manifest_sha256": model_manifest_sha256,
        "target_raw_manifest_sha256": target_raw_manifest_sha256,
        "target_trial_registry_sha256": target_trial_registry_sha256,
        "field_descriptors": [dict(row) for row in descriptors],
        "instrumentation_gate": dict(instrumentation),
        "target_construction_labels": 0,
        "target_evaluation_labels": 0,
        "same_label_oracle": 0,
        "selector_scores": 0,
        "scientific_statistics": 0,
        "C84S_authorized": False,
        "gate": FIELD_GATE,
    }
    directory = Path(root)
    path = directory / COMPLETE_MANIFEST_NAME
    sidecar = directory / COMPLETE_MANIFEST_SHA_NAME
    _require(not path.exists() and not sidecar.exists(), "V2 complete manifest overwrite is forbidden")
    digest = manifests.write_json_atomic(path, payload)
    manifests.write_hash_sidecar(sidecar, digest)
    return {**payload, "path": str(path), "sha256": digest, "sha256_path": str(sidecar)}


def run_real(*, authorization_path: Path, output_root: Path) -> dict[str, Any]:
    from . import c84fr2_runtime_guard as runtime

    binding = runtime.require_authorization_and_lock(
        authorization_path=authorization_path, output_root=output_root,
    )
    consumption = runtime.consume_authorization(binding)
    ledger = runtime.ExecutionAttemptLedger(Path(binding["run_root"]), consumption)
    try:
        ledger.stage("protected_package_imports_and_runtime_versions")
        import numpy as np
        import torch
        import mne
        import moabb

        ledger.increment("package_imports", 4)
        runtime.verify_protected_runtime_versions(binding["lock"], torch=torch, mne=mne, moabb=moabb)
        ledger.stage("CUDA_and_determinism")
        ledger.increment("CUDA_checks")
        if not torch.cuda.is_available() or os.environ.get("SLURM_JOB_ID") is None:
            raise C84FR2TargetStageError("C84FR2 requires an authorized Slurm CUDA allocation")
        torch.use_deterministic_algorithms(True, warn_only=False)
        _require(torch.are_deterministic_algorithms_enabled(), "torch deterministic algorithms are disabled")

        ledger.stage("loader_source_identity_replay")
        runtime.verify_loader_source_files(binding["lock"])
        objects, loader_classes = _protected_loader_objects()
        runtime.verify_loader_runtime_objects(binding["lock"], objects)
        ledger.increment("loader_source_replays")
        ledger.increment("dataset_loader_imports")

        frozen = binding["frozen_replay"]
        model_rows = sorted(frozen["model_rows"], key=lambda row: str(row["unit_id"]))
        ledger.stage("frozen_model_and_target_registry_replay_no_training")
        _require(len(model_rows) == 1944 and frozen["model_artifact_files_replayed"] == 7776,
                 "frozen model-field replay is incomplete")
        ledger.increment("model_field_units", 1944)
        _require(not ledger.counters["training_phases_started"] and not ledger.counters["training_phases_completed"],
                 "target-only V2 stage invoked training")
        ledger.publish_partial_manifest("FROZEN_INPUTS_REPLAYED_NO_TRAINING")

        ledger.stage("target_X_reload_and_exact_non_label_registry_replay")
        views, raw_target_files = target_inputs.load_complete_target_views(
            model_manifest_path=frozen["model_manifest_path"],
            model_manifest_sha_path=frozen["model_manifest_sha_path"],
            loader_classes=loader_classes,
            np=np,
            ledger=ledger,
        )
        raw_payload = target_raw_manifest_payload(raw_target_files)
        raw_replay = require_exact_json_payload(
            raw_payload, frozen["target_raw_manifest_path"], object_name="target raw manifest",
        )
        trial_rows = target_inputs.target_trial_registry_rows(views)
        trial_replay = require_exact_trial_registry(trial_rows, frozen["target_registry_path"])
        ledger.increment("target_registry_trials", len(trial_rows))

        raw_destination = Path(binding["run_root"]) / "C84F_TARGET_RAW_INPUT_MANIFEST.json"
        registry_destination = Path(binding["run_root"]) / manifests.TARGET_REGISTRY_NAME
        registry_sha_destination = Path(binding["run_root"]) / manifests.TARGET_REGISTRY_SHA_NAME
        _copy_exact(frozen["target_raw_manifest_path"], raw_destination,
                    expected_sha256=raw_replay["sha256"])
        _copy_exact(frozen["target_registry_path"], registry_destination,
                    expected_sha256=trial_replay["sha256"])
        _copy_exact(frozen["target_registry_sha_path"], registry_sha_destination,
                    expected_sha256=manifests.sha256_file(frozen["target_registry_sha_path"]))

        ledger.stage("complete_target_unlabeled_v2_instrumentation_no_training")
        reuse_rows = runtime.base.read_csv(field_protocol.TABLE_DIR / "dual_canary_reuse_registry.csv")
        reuse_map = {row["unit_id"]: row for row in reuse_rows}
        descriptors, instrumentation = instrument_complete_field_v2(
            model_rows=model_rows,
            views=views,
            reuse_rows=reuse_map,
            output_root=binding["run_root"],
            model_manifest_path=frozen["model_manifest_path"],
            model_manifest_sha_path=frozen["model_manifest_sha_path"],
            torch=torch,
            np=np,
            ledger=ledger,
        )

        ledger.stage("atomic_complete_field_manifest_v2")
        complete = publish_complete_field_manifest_v2(
            binding["run_root"],
            descriptors,
            model_manifest_sha256=frozen["model_manifest_sha256"],
            target_raw_manifest_sha256=raw_replay["sha256"],
            target_trial_registry_sha256=trial_replay["sha256"],
            instrumentation=instrumentation,
            execution_identity={
                "execution_lock_sha256": binding["lock_sha256"],
                "authorization_consumption_sha256": consumption["sha256"],
                "historical_failed_jobs": [896185, 896550],
                "model_retraining": 0,
            },
        )
        ledger.complete(complete["sha256"])
        return {
            "schema_version": "c84fr2_target_stage_execution_result_v1",
            "gate": FIELD_GATE,
            "complete_manifest_path": complete["path"],
            "complete_manifest_sha256": complete["sha256"],
            "model_units": 1944,
            "target_artifacts": 1944,
            "context_digest_sidecars": 1944,
            "candidate_context_slices": 76464,
            "target_labels": 0,
            "scientific_metrics": 0,
            "model_retraining": 0,
            "C84S_authorized": False,
        }
    except Exception as exc:
        ledger.fail(exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C84FR2 target-only numerical replay repair")
    subparsers = parser.add_subparsers(dest="command", required=True)
    real = subparsers.add_parser("run-real", help="run authorized V2 target-field completion")
    real.add_argument("--authorization-record", type=Path, required=True)
    real.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run-real":
        print(json.dumps(run_real(
            authorization_path=args.authorization_record,
            output_root=args.output_root,
        ), sort_keys=True))
        return 0
    raise C84FR2TargetStageError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
