"""Authorized target-only recovery for the frozen C84F model field.

This module has no training entrypoint. It replays the immutable model field,
reloads only target-unlabeled X, writes the canonical trial registry, and then
performs the originally locked complete-field instrumentation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import c84f_field_manifest as manifests
from . import c84f_target_instrumentation as target_stage
from . import c84fl2_protocol as protocol
from . import c84fr1_runtime_guard as runtime


class C84FR1TargetStageError(RuntimeError):
    """Raised when the target-only recovery deviates from its lock."""


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


def require_exact_historical_raw_manifest(
    observed: Mapping[str, Any], historical_path: str | Path,
) -> dict[str, Any]:
    historical = runtime.read_json(historical_path)
    if manifests.canonical_bytes(dict(observed)) != manifests.canonical_bytes(historical):
        raise C84FR1TargetStageError("target raw-input manifest differs from failed job 896185")
    return {
        "path": str(historical_path),
        "sha256": manifests.sha256_file(historical_path),
        "file_count": int(historical["file_count"]),
        "exact_replay": True,
    }


def run_real(
    *,
    authorization_path: Path = runtime.AUTHORIZATION_RECORD_PATH,
    output_root: Path = runtime.DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
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
            raise C84FR1TargetStageError("target-stage repair requires an authorized Slurm CUDA allocation")
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not torch.are_deterministic_algorithms_enabled():
            raise C84FR1TargetStageError("torch deterministic algorithms are not enabled")

        ledger.stage("loader_source_identity_replay")
        runtime.verify_loader_source_files(binding["lock"])
        ledger.increment("loader_source_replays")
        objects, loader_classes = _protected_loader_objects()
        ledger.increment("dataset_loader_imports")
        runtime.verify_loader_runtime_objects(binding["lock"], objects)

        frozen = binding["frozen_model_field_replay"]
        model_rows = sorted(frozen["model_rows"], key=lambda row: str(row["unit_id"]))
        ledger.stage("frozen_model_field_replay_no_training")
        if len(model_rows) != 1944 or frozen["model_artifact_files_replayed"] != 7776:
            raise C84FR1TargetStageError("frozen model-field replay is incomplete")
        ledger.increment("model_field_units", len(model_rows))
        if ledger.counters["training_phases_started"] or ledger.counters["training_phases_completed"]:
            raise C84FR1TargetStageError("target-only recovery invoked training")
        ledger.publish_partial_manifest("MODEL_FIELD_REPLAYED_NO_TRAINING")

        ledger.stage("complete_target_unlabeled_registry_repair")
        views, raw_target_files = target_stage.load_complete_target_views(
            model_manifest_path=frozen["model_manifest_path"],
            model_manifest_sha_path=frozen["model_manifest_sha_path"],
            loader_classes=loader_classes,
            np=np,
            ledger=ledger,
        )
        raw_payload = target_raw_manifest_payload(raw_target_files)
        historical_raw = require_exact_historical_raw_manifest(
            raw_payload, frozen["target_raw_manifest_path"],
        )
        raw_path = Path(binding["run_root"]) / "C84F_TARGET_RAW_INPUT_MANIFEST.json"
        raw_sha = manifests.write_json_atomic(raw_path, raw_payload)
        if raw_sha != historical_raw["sha256"]:
            raise C84FR1TargetStageError("re-emitted target raw manifest hash differs from historical input")

        trial_rows = target_stage.target_trial_registry_rows(views)
        ledger.increment("target_registry_trials", len(trial_rows))
        target_registry = manifests.publish_target_trial_registry(
            binding["run_root"],
            trial_rows,
            model_manifest_path=frozen["model_manifest_path"],
            model_manifest_sha_path=frozen["model_manifest_sha_path"],
            execution_identity={
                "execution_lock_sha256": binding["lock_sha256"],
                "target_raw_input_manifest_sha256": raw_sha,
                "historical_failed_job": 896185,
                "model_retraining": 0,
            },
        )

        ledger.stage("complete_target_unlabeled_instrumentation_no_training")
        reuse_rows = runtime.base.read_csv(protocol.TABLE_DIR / "dual_canary_reuse_registry.csv")
        reuse_map = {row["unit_id"]: row for row in reuse_rows}
        descriptors, instrumentation = target_stage.instrument_complete_field(
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
        if instrumentation["training_invocations"] != 0:
            raise C84FR1TargetStageError("target instrumentation reported a training invocation")

        ledger.stage("atomic_complete_field_manifest")
        complete = manifests.publish_complete_field_manifest(
            binding["run_root"],
            descriptors,
            operative_unit_ids=[row["unit_id"] for row in model_rows],
            model_manifest_path=frozen["model_manifest_path"],
            model_manifest_sha_path=frozen["model_manifest_sha_path"],
            target_registry_path=target_registry["path"],
            target_registry_sha_path=target_registry["sha256_path"],
            instrumentation_summary=instrumentation,
            execution_identity={
                "execution_lock_sha256": binding["lock_sha256"],
                "authorization_consumption_sha256": consumption["sha256"],
                "historical_failed_job": 896185,
                "frozen_model_field_manifest_sha256": frozen["model_manifest_sha256"],
                "model_retraining": 0,
            },
        )
        ledger.complete(complete["sha256"])
        return {
            "schema_version": "c84fr1_target_stage_execution_result_v1",
            "gate": protocol.FIELD_GATE,
            "complete_manifest_path": complete["path"],
            "complete_manifest_sha256": complete["sha256"],
            "frozen_model_field_manifest_sha256": frozen["model_manifest_sha256"],
            "model_units": 1944,
            "model_retraining": 0,
            "target_contexts": 944,
            "candidate_context_slices": 76464,
            "target_labels": 0,
            "scientific_metrics": 0,
            "C84S_authorized": False,
        }
    except Exception as exc:
        ledger.fail(exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C84F target-stage canonical registry repair")
    subparsers = parser.add_subparsers(dest="command", required=True)
    real = subparsers.add_parser("run-real", help="run the authorized target-only field completion")
    real.add_argument("--authorization-record", type=Path, required=True)
    real.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run-real":
        print(json.dumps(run_real(
            authorization_path=args.authorization_record,
            output_root=args.output_root,
        ), sort_keys=True))
        return 0
    raise C84FR1TargetStageError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

