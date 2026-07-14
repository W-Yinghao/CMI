"""Replacement C84L1C engineering canary after the FP32 replay repair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from . import c84c_real_canary_v2 as instrumentation_base
from . import c84l1_canary as base
from . import c84l1r1_runtime_repair as runtime


AUTHORIZATION_RECORD_PATH = runtime.AUTHORIZATION_RECORD_PATH
DEFAULT_EXTERNAL_ROOT = runtime.DEFAULT_EXTERNAL_ROOT


class C84L1CanaryV2Error(runtime.C84L1R1RuntimeError):
    """Raised when the replacement level-1 canary fails closed."""


def _instrument_and_replay(
    state: Mapping[str, Any],
    source_audit: instrumentation_base.SourceView,
    target: instrumentation_base.TargetUnlabeledView,
    unit: Mapping[str, Any],
    root: Path,
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = instrumentation_base._model_factory()
    model.load_state_dict(state)
    model.eval().to("cuda:0")
    source_logits, _ = instrumentation_base._forward_model(model, source_audit.X, torch)
    target_logits, target_z = instrumentation_base._forward_model(model, target.X, torch)
    with torch.inference_mode():
        source_probabilities = torch.softmax(source_logits, dim=1)
        target_probabilities = torch.softmax(target_logits, dim=1)
        classifier_weight = model.classifier.weight.detach().cpu()
        classifier_bias = model.classifier.bias.detach().cpu()
        reconstructed = target_z @ classifier_weight.T + classifier_bias
        repeat_output = model(torch.as_tensor(target.X, device="cuda:0"))
        repeat_logits = repeat_output.logits.detach().cpu()
        repeat_z = repeat_output.z.detach().cpu()
    errors = {
        "Wz_plus_b_max_error": float(torch.max(torch.abs(reconstructed - target_logits))),
        "softmax_max_error": float(torch.max(torch.abs(torch.softmax(target_logits, dim=1) - target_probabilities))),
        "repeat_logits_max_error": float(torch.max(torch.abs(repeat_logits - target_logits))),
        "repeat_z_max_error": float(torch.max(torch.abs(repeat_z - target_z))),
    }
    identity_validation = runtime.validate_instrumentation_errors(errors)

    identity = {
        "dataset": unit["dataset"], "panel": unit["source_panel"],
        "seed": unit["training_seed"], "level": unit["level"], "unit_id": unit["unit_id"],
    }
    source_path = root / "source_audit" / f"{unit['unit_id']}.npz"
    instrumentation_base._atomic_save_npz(
        source_path, np,
        logits=source_logits.numpy(), probabilities=source_probabilities.numpy(),
        source_class_label=np.asarray(source_audit.y, dtype=np.int64),
        source_domain_id=np.asarray(source_audit.subject_id, dtype=np.int64),
        source_trial_id=np.asarray(source_audit.trial_id, dtype=str),
        dataset=np.asarray(identity["dataset"]), panel=np.asarray(identity["panel"]),
        seed=np.asarray(identity["seed"], dtype=np.int64), level=np.asarray(identity["level"], dtype=np.int64),
        unit_id=np.asarray(identity["unit_id"]),
    )
    source_replay = runtime.replay_source_audit_artifact(
        source_path, expected_identity=identity, expected_trial_ids=source_audit.trial_id,
        expected_labels=source_audit.y, expected_domains=source_audit.subject_id, np=np,
    )
    source_descriptor = {
        **source_replay, "bytes": source_path.stat().st_size,
        "fields": sorted(runtime.prior.SOURCE_AUDIT_FIELDS), "target_label_fields": 0,
        "scientific_metrics": 0,
    }

    target_path = root / "target_unlabeled" / f"{unit['unit_id']}.npz"
    instrumentation_base._atomic_save_npz(
        target_path, np,
        logits=target_logits.numpy(), probabilities=target_probabilities.numpy(), z=target_z.numpy(),
        Wz_plus_b=reconstructed.numpy(), classifier_weight=classifier_weight.numpy(),
        classifier_bias=classifier_bias.numpy(), repeat_logits=repeat_logits.numpy(), repeat_z=repeat_z.numpy(),
        target_trial_id=np.asarray(target.trial_id, dtype=str), dataset=np.asarray(identity["dataset"]),
        target_subject_id=np.asarray(target.target_subject_id, dtype=np.int64), unit_id=np.asarray(identity["unit_id"]),
    )
    target_replay = runtime.replay_target_unlabeled_artifact(
        target_path,
        expected_identity={
            "dataset": identity["dataset"], "unit_id": identity["unit_id"],
            "target_subject_id": target.target_subject_id,
        },
        expected_trial_ids=target.trial_id,
        np=np,
        linear_tolerance=runtime.LINEAR_REPLAY_ABS_TOLERANCE,
    )
    target_descriptor = {
        **target_replay, "bytes": target_path.stat().st_size,
        "fields": sorted(runtime.prior.TARGET_UNLABELED_FIELDS), "target_label_fields": 0,
        "linear_replay_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
        "strict_identity_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
        "in_memory_identity_validation_pass": identity_validation["validation_pass"],
        **errors,
    }
    return source_descriptor, target_descriptor


def _configure_base() -> None:
    base.runtime = runtime
    base.persisted = sys.modules[__name__]


def run_real(
    *,
    authorization_path: Path = AUTHORIZATION_RECORD_PATH,
    output_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
    _configure_base()
    binding = runtime.require_authorization_and_lock(
        authorization_path=authorization_path, output_root=output_root,
    )
    consumption = runtime.consume_authorization(binding)
    ledger = runtime.ExecutionAttemptLedger(Path(binding["run_root"]), consumption)
    try:
        ledger.stage("package_imports_and_exact_version_replay")
        import numpy as np
        import torch
        import mne
        import moabb

        ledger.increment("package_imports", 4)
        runtime.verify_protected_runtime_versions(binding["lock"], torch=torch, mne=mne, moabb=moabb)
        ledger.stage("CUDA_and_determinism_check")
        ledger.increment("CUDA_checks")
        if not torch.cuda.is_available() or os.environ.get("SLURM_JOB_ID") is None:
            raise C84L1CanaryV2Error("replacement C84L1C requires an authorized Slurm CUDA allocation")
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not torch.are_deterministic_algorithms_enabled():
            raise C84L1CanaryV2Error("torch deterministic algorithms are not enabled")

        ledger.stage("loader_source_identity_replay")
        runtime.verify_loader_source_files(binding["lock"])
        ledger.increment("loader_source_replays")
        objects, Lee2019_MI, Cho2017, PhysionetMI, MotorImagery = instrumentation_base._protected_loader_objects()
        ledger.increment("dataset_loader_imports")
        runtime.verify_loader_runtime_objects(binding["lock"], objects)
        loader_classes = (Lee2019_MI, Cho2017, PhysionetMI, MotorImagery)

        datasets = []
        for dataset in base.protocol.historical.DATASET_ORDER:
            ledger.stage(f"dataset_access:{dataset}")
            source, audit, target = instrumentation_base._load_canary_views(
                dataset, np, loader_classes, ledger,
            )
            ledger.stage(f"level1_training_and_instrumentation:{dataset}")
            datasets.append(base._run_dataset_training(
                dataset, source, audit, target, Path(binding["run_root"]) / dataset,
                binding, torch, np, ledger,
            ))
            ledger.publish_partial_manifest("IN_PROGRESS")

        rows = [row for dataset in datasets for row in dataset["units"]]
        complete = runtime.validate_complete_level1_canary_gate(rows)
        manifest = {
            "schema_version": "c84l1c_complete_engineering_manifest_v2",
            "execution_lock_v2_sha256": binding["lock_sha256"],
            "canary_protocol_v2_sha256": binding["lock"]["canary_protocol"]["sha256"],
            "repair_protocol_sha256": binding["lock"]["repair_protocol"]["sha256"],
            "authorization_consumption_sha256": consumption["sha256"],
            "historical_failed_job": 895928,
            "failed_authorization_reused": False,
            "failed_partial_artifacts_reused": False,
            "linear_replay_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "strict_identity_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
            "datasets": datasets,
            "complete_gate": complete,
            "unit_count": len(rows),
            "training_phases": 9,
            "source_audit_artifacts": ledger.counters["source_audit_artifacts"],
            "target_unlabeled_artifacts": ledger.counters["target_unlabeled_artifacts"],
            "target_label_access": ledger.counters["target_y_accesses"],
            "target_scientific_metrics": ledger.counters["target_scientific_metrics"],
            "C84F_authorized": False,
            "C84S_authorized": False,
        }
        manifest_path = Path(binding["run_root"]) / "C84L1C_COMPLETE_ENGINEERING_MANIFEST.json"
        runtime.write_json_atomic(manifest_path, manifest)
        manifest_sha = runtime.sha256_file(manifest_path)
        ledger.stage("manifest_publication")
        ledger.complete(manifest_sha)
        return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha}
    except Exception as exc:
        ledger.fail(exc)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repaired C84 level-1 engineering canary")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-contract")
    subparsers.add_parser("schema-dry-run")
    real = subparsers.add_parser("run-real")
    real.add_argument("--authorization-record", type=Path, default=AUTHORIZATION_RECORD_PATH)
    real.add_argument("--output-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    args = parser.parse_args(argv)
    if args.command == "show-contract":
        print(json.dumps({
            "stage": "C84L1C", "level": 1, "units": 243, "training_phases": 9,
            "linear_replay_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "strict_identity_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
            "fresh_authorization_required": True, "failed_artifacts_reused": False,
            "C84F": False, "C84S": False, "target_y_access": 0, "scientific_metrics": 0,
        }, sort_keys=True))
        return 0
    if args.command == "schema-dry-run":
        payload = base.synthetic_schema_dry_run()
        print(json.dumps({
            **payload,
            "schema_version": "c84l1c_repaired_schema_dry_run_v1",
            "linear_replay_abs_tolerance": runtime.LINEAR_REPLAY_ABS_TOLERANCE,
            "strict_identity_abs_tolerance": runtime.STRICT_IDENTITY_ABS_TOLERANCE,
            "failed_partial_artifacts_reused": False,
        }, sort_keys=True))
        return 0
    print(json.dumps(run_real(
        authorization_path=args.authorization_record, output_root=args.output_root,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
