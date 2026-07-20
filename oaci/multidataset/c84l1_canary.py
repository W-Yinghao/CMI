"""Authorized future C84 level-1 engineering canary adapter.

The module is import-safe in C84L1P. Protected numerical and dataset packages
are imported only after a fresh authorization has been consumed and an attempt
ledger has been created.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import c84c_real_canary as legacy
from . import c84c_real_canary_v2 as base
from . import c84c_real_canary_v3 as persisted
from . import c84l1_intervention as intervention
from . import c84l1_protocols as protocol
from . import c84l1_runtime_guard as runtime
from .c84r_montage_repair import EPOCH_RULE, INTERFACE_ID, MONTAGE_SHA256


AUTHORIZATION_RECORD_PATH = runtime.AUTHORIZATION_RECORD_PATH
DEFAULT_EXTERNAL_ROOT = runtime.DEFAULT_EXTERNAL_ROOT
SOURCE_PANEL = "A"
TRAINING_SEED = 5
LEVEL = 1
CHECKPOINT_EPOCHS = tuple(range(4, 200, 5))


class C84L1CanaryError(runtime.C84L1RuntimeError):
    """Raised on any level-1 engineering-contract failure."""


@dataclass(frozen=True)
class TrainingBundle:
    source: base.SourceView
    application: intervention.InterventionApplication
    data: Any
    support: Any
    stage1_plan: Any
    stage2_plan: Any
    oaci_plan: Any
    full_plan: Any
    engine_config: Any
    population_signature: str

    @property
    def plans(self) -> tuple[Any, Any, Any, Any]:
        return self.stage1_plan, self.stage2_plan, self.oaci_plan, self.full_plan

    @property
    def plan_hashes(self) -> tuple[str, str, str, str]:
        return tuple(str(plan.plan_hash) for plan in self.plans)


SIDECAR_FIELDS = frozenset({
    "schema_version", "unit_id", "dataset", "panel", "seed", "level", "regime", "epoch",
    "trajectory_order", "interface_id", "montage_sha256", "epoch_rule", "checkpoint",
    "optimizer", "source_audit", "target_unlabeled", "model_state_hash",
    "parent_ERM_model_state_hash", "previous_trajectory_model_state_hash", "genealogy_rule",
    "support_hash", "target_subject", "target_fit_ids", "training_target_rows",
    "training_target_labels", "source_audit_rows_used_in_training", "target_outcome_retention",
    "target_outcome_retry", "target_scientific_metrics", "level_intervention_id",
    "level_intervention_registry_sha256", "deleted_source_subject", "deleted_class",
    "pre_deletion_trial_count", "post_deletion_trial_count", "deleted_trial_id_sha256",
    "intervention_population_signature_sha256", "intervention_support_graph_sha256",
    "training_population_signature", "plan_hashes", "paired_model_init_hash",
    "paired_model_init_pass", "level0_plan_replay_pass", "accepted_level0_model_registry_sha256",
})


def _subset_source(source: base.SourceView, indices: Sequence[int], np: Any) -> base.SourceView:
    rows = np.asarray(tuple(indices), dtype=np.int64)
    return base.SourceView(
        X=source.X[rows],
        y=source.y[rows],
        trial_id=tuple(source.trial_id[index] for index in indices),
        subject_id=tuple(source.subject_id[index] for index in indices),
        session=tuple(source.session[index] for index in indices),
        run=tuple(source.run[index] for index in indices),
        dataset_id=source.dataset_id,
        role=source.role,
        interface=source.interface,
    )


def materialize_training_bundle(
    source: base.SourceView,
    *,
    dataset: str,
    panel: str,
    training_seed: int,
    level: int,
    torch: Any,
    np: Any,
) -> TrainingBundle:
    """Apply the level intervention before support and plan materialization."""
    from oaci.config import SamplerConfig
    from oaci.data.plan_materialize import (
        materialize_full_domain_alignment_plan,
        materialize_oaci_alignment_plan,
        materialize_stage1_task_plan,
        materialize_stage2_task_plan,
    )
    from oaci.data.plan_sampler import UnitIndex
    from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior
    from oaci.train.data import TrainingData, population_signature_hash
    from oaci.train.engine import EngineConfig

    applied = intervention.apply_level_intervention(
        dataset=dataset,
        panel=panel,
        level=level,
        source_subjects=source.subject_id,
        source_labels=source.y,
        source_trial_ids=source.trial_id,
    )
    selected = _subset_source(source, applied.keep_indices, np)
    domain_names = sorted(set(selected.subject_id))
    domain_map = {subject: index for index, subject in enumerate(domain_names)}
    domains = np.asarray([domain_map[value] for value in selected.subject_id], dtype=np.int64)
    groups = tuple(
        f"{dataset}|subject={selected.subject_id[index]}|session={selected.session[index]}|run={selected.run[index]}"
        for index in range(len(selected.trial_id))
    )
    mass = np.ones(len(selected.trial_id), dtype=np.float64)
    data = TrainingData(
        X=torch.as_tensor(selected.X, dtype=torch.float32),
        y=torch.as_tensor(selected.y, dtype=torch.long),
        sample_id=tuple(selected.trial_id),
        sample_mass=torch.as_tensor(mass, dtype=torch.float32),
        n_classes=2,
        d=torch.as_tensor(domains, dtype=torch.long),
        group=groups,
    ).validate()
    counts = counts_from_labels(domains, selected.y, n_domains=len(domain_names), n_classes=2)
    support = build_support_graph(
        counts,
        protocol.MIN_CELL_SUPPORT,
        cell_mass=counts.astype(float),
        reference_prior=empirical_class_prior(counts),
        domain_names=[str(value) for value in domain_names],
        class_names=["left_hand", "right_hand"],
    ).validate()
    index = UnitIndex(data.sample_id, selected.y, domains, groups, data.sample_id, mass)
    population = population_signature_hash(data)
    sampler = SamplerConfig(
        task_batch_size=256,
        adv_microbatch_size=256,
        adv_accumulation_steps=4,
        min_per_eligible_cell=protocol.MIN_CELL_SUPPORT,
        steps_per_epoch=20,
        replacement_mode="auto",
        seed=int(training_seed),
    )
    stage1 = materialize_stage1_task_plan(index, population, 200, 1, 256, training_seed, "auto")
    stage2 = materialize_stage2_task_plan(index, population, 200, 20, 256, training_seed, "auto")
    oaci = materialize_oaci_alignment_plan(
        index, support, population, 60, 4000, 5, protocol.MIN_CELL_SUPPORT, 256,
        training_seed, accumulation_steps=4, replacement_mode="auto",
    )
    full = materialize_full_domain_alignment_plan(
        index, population, 60, 4000, 5, protocol.MIN_CELL_SUPPORT, 256,
        training_seed, accumulation_steps=4, replacement_mode="auto",
    )
    engine = EngineConfig(
        metric="balanced_ce", epsilon=0.03, numerical_tol=1e-4,
        stage1_epochs=200, stage1_steps_per_epoch=1, stage2_epochs=200,
        steps_per_epoch=20, warmup_steps=60, critic_steps=5, checkpoint_every=5,
        guard_chunk_size=1024, optimizer_name="adam", weight_decay=0.0,
        lr_stage1=0.005, lr_encoder=0.01, lr_critic=0.01, dual_lr=0.5,
        lambda_init=0.3, lambda_max=20.0, lambda_floor=0.0,
        gradient_clip=0.0, critic_gradient_clip=0.0,
        deterministic_algorithms=True, stage2_bn_mode="frozen_erm_running_stats",
        base_seed=int(training_seed),
    )
    if sampler.seed != training_seed or engine.base_seed != training_seed:
        raise C84L1CanaryError("training seed was not propagated to sampler and engine")
    return TrainingBundle(selected, applied, data, support, stage1, stage2, oaci, full, engine, population)


def _candidate_rows(dataset: str) -> list[dict[str, Any]]:
    rows = protocol.read_csv(protocol.TABLE_DIR / "level1_candidate_id_registry.csv")
    selected = []
    for row in rows:
        if row["dataset"] == dataset and row["panel"] == SOURCE_PANEL and int(row["training_seed"]) == TRAINING_SEED:
            selected.append({
                **row,
                "training_seed": int(row["training_seed"]),
                "level": int(row["level"]),
                "epoch": int(row["epoch"]),
                "trajectory_order": int(row["trajectory_order"]),
                "source_panel": row["panel"],
            })
    if len(selected) != 81 or any(row["level"] != LEVEL for row in selected):
        raise C84L1CanaryError(f"{dataset} does not have exactly 81 operative canary IDs")
    return selected


def _model_init_hash(dataset: str, training_seed: int, torch: Any) -> str:
    from oaci.train.checkpoint import state_hash
    from oaci.train.rng import derive_seed, forked_rng

    with forked_rng(derive_seed(training_seed, dataset, "model_init"), torch.device("cuda:0")):
        model = base._model_factory()
    return state_hash({key: value.detach().cpu() for key, value in model.state_dict().items()})


def _accepted_level0_replay(
    dataset: str,
    source: base.SourceView,
    binding: Mapping[str, Any],
    torch: Any,
    np: Any,
) -> tuple[TrainingBundle, str]:
    level0 = materialize_training_bundle(
        source, dataset=dataset, panel=SOURCE_PANEL, training_seed=TRAINING_SEED,
        level=0, torch=torch, np=np,
    )
    expected = binding["lock"]["accepted_C84C_level0"]["datasets"][dataset]
    if list(level0.plan_hashes) != list(expected["plan_hashes"]):
        raise C84L1CanaryError(f"accepted C84C level-0 plan replay failed for {dataset}")
    return level0, str(expected["model_unit_registry_sha256"])


def _run_dataset_training(
    dataset: str,
    source: base.SourceView,
    source_audit: base.SourceView,
    target: base.TargetUnlabeledView,
    root: Path,
    binding: Mapping[str, Any],
    torch: Any,
    np: Any,
    ledger: runtime.ExecutionAttemptLedger,
) -> dict[str, Any]:
    from oaci.methods.oaci import OACIObjective
    from oaci.methods.source_robust import SRCObjective
    from oaci.train.checkpoint import state_hash
    from oaci.train.engine import InvocationRegistry, train_stage1, train_stage2
    from oaci.train.rng import derive_seed, forked_rng

    level0, accepted_model_registry = _accepted_level0_replay(dataset, source, binding, torch, np)
    level1 = materialize_training_bundle(
        source, dataset=dataset, panel=SOURCE_PANEL, training_seed=TRAINING_SEED,
        level=LEVEL, torch=torch, np=np,
    )
    if level0.plan_hashes == level1.plan_hashes:
        raise C84L1CanaryError("level-specific plans did not bind the changed population")
    level0_init = _model_init_hash(dataset, TRAINING_SEED, torch)
    level1_init = _model_init_hash(dataset, TRAINING_SEED, torch)
    if level0_init != level1_init:
        raise C84L1CanaryError("paired level-0/level-1 model initialization differs")

    units = _candidate_rows(dataset)
    by_key = {(row["regime"], row["epoch"]): row for row in units}
    output_units = []
    ledger.increment("training_phases_started", 3)
    with legacy._OptimizerCapture(root / "optimizer_states") as capture:
        capture.begin("ERM")
        with forked_rng(derive_seed(TRAINING_SEED, dataset, "model_init"), torch.device("cuda:0")):
            model = base._model_factory()
        erm = train_stage1(
            model, level1.data, level1.stage1_plan, level1.engine_config, torch.device("cuda:0"),
            InvocationRegistry(), f"C84L1C|{dataset}|A|seed5|level1",
        )
        ledger.increment("training_phases_completed")
        capture.begin("OACI")
        oaci = train_stage2(
            base._model_factory, erm, level1.data, OACIObjective(level1.support, adv_hidden=16),
            level1.stage2_plan, level1.oaci_plan, level1.engine_config, torch.device("cuda:0"),
        )
        ledger.increment("training_phases_completed")
        capture.begin("SRC")
        src = train_stage2(
            base._model_factory, erm, level1.data,
            SRCObjective(2, level1.support.n_domains, smooth_temperature=0.1),
            level1.stage2_plan, level1.full_plan, level1.engine_config, torch.device("cuda:0"),
        )
        ledger.increment("training_phases_completed")
        if tuple(record.epoch for record in oaci.trajectory) != CHECKPOINT_EPOCHS:
            raise C84L1CanaryError("OACI checkpoint cadence drift")
        if tuple(record.epoch for record in src.trajectory) != CHECKPOINT_EPOCHS:
            raise C84L1CanaryError("SRC checkpoint cadence drift")
        records = [("ERM", erm.checkpoint, 0, capture.descriptor("ERM", 0))]
        records.extend(("OACI", record, order, capture.descriptor("OACI", order))
                       for order, record in enumerate(oaci.trajectory, start=1))
        records.extend(("SRC", record, order, capture.descriptor("SRC", order))
                       for order, record in enumerate(src.trajectory, start=1))
        previous_hash = {"ERM": erm.checkpoint.model_hash, "OACI": erm.checkpoint.model_hash,
                         "SRC": erm.checkpoint.model_hash}
        for regime, record, order, optimizer_descriptor in records:
            unit = by_key[(regime, int(record.epoch))]
            checkpoint_path = root / "checkpoints" / f"{unit['unit_id']}.pt"
            checkpoint = legacy._save_torch_state(checkpoint_path, record.model_state, torch)
            checkpoint_replay = runtime.replay_checkpoint(
                checkpoint_path, expected_file_sha256=checkpoint["sha256"],
                expected_state_hash=record.model_hash, torch=torch, state_hash_fn=state_hash,
            )
            optimizer_replay = runtime.replay_optimizer_state(
                optimizer_descriptor, phase=regime, trajectory_order=order, torch=torch,
            )
            source_descriptor, target_descriptor = persisted._instrument_and_replay(
                record.model_state, source_audit, target, unit, root, torch, np,
            )
            ledger.increment("source_audit_artifacts")
            ledger.increment("target_unlabeled_artifacts")
            sidecar = {
                "schema_version": "c84l1c_candidate_sidecar_v1",
                "unit_id": unit["unit_id"], "dataset": dataset, "panel": SOURCE_PANEL,
                "seed": TRAINING_SEED, "level": LEVEL, "regime": regime,
                "epoch": int(record.epoch), "trajectory_order": order,
                "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256,
                "epoch_rule": EPOCH_RULE,
                "checkpoint": {**checkpoint, "replay": checkpoint_replay},
                "optimizer": {**optimizer_descriptor, "replay": optimizer_replay},
                "source_audit": source_descriptor, "target_unlabeled": target_descriptor,
                "model_state_hash": record.model_hash,
                "parent_ERM_model_state_hash": erm.checkpoint.model_hash,
                "previous_trajectory_model_state_hash": previous_hash[regime],
                "genealogy_rule": "shared_level1_ERM_parent_then_fixed_regime_trajectory_order",
                "support_hash": level1.support.support_hash(), "target_subject": target.target_subject_id,
                "target_fit_ids": [], "training_target_rows": 0, "training_target_labels": 0,
                "source_audit_rows_used_in_training": 0, "target_outcome_retention": 0,
                "target_outcome_retry": 0, "target_scientific_metrics": 0,
                "level_intervention_id": protocol.LEVEL1_ID,
                "level_intervention_registry_sha256": unit["level_intervention_registry_sha256"],
                "deleted_source_subject": level1.application.deleted_source_subject,
                "deleted_class": level1.application.deleted_class,
                "pre_deletion_trial_count": level1.application.pre_trial_count,
                "post_deletion_trial_count": level1.application.post_trial_count,
                "deleted_trial_id_sha256": level1.application.deleted_trial_id_sha256,
                "intervention_population_signature_sha256": level1.application.population_signature_sha256,
                "intervention_support_graph_sha256": level1.application.support_graph_sha256,
                "training_population_signature": level1.population_signature,
                "plan_hashes": {
                    "stage1": level1.plan_hashes[0], "stage2": level1.plan_hashes[1],
                    "OACI": level1.plan_hashes[2], "SRC": level1.plan_hashes[3],
                },
                "paired_model_init_hash": level1_init, "paired_model_init_pass": True,
                "level0_plan_replay_pass": True,
                "accepted_level0_model_registry_sha256": accepted_model_registry,
            }
            if set(sidecar) != SIDECAR_FIELDS or legacy.SCIENTIFIC_OUTPUT_KEYS & set(sidecar):
                raise C84L1CanaryError("C84L1C sidecar field contract drift")
            sidecar_path = root / "sidecars" / f"{unit['unit_id']}.json"
            runtime.write_json_atomic(sidecar_path, sidecar)
            sidecar_replay = runtime.replay_sidecar(
                sidecar_path, expected_fields=SIDECAR_FIELDS,
                expected_identity={"unit_id": unit["unit_id"], "dataset": dataset,
                                   "regime": regime, "epoch": int(record.epoch),
                                   "trajectory_order": order},
            )
            output_units.append({
                "unit_id": unit["unit_id"], "dataset": dataset, "regime": regime,
                "epoch": int(record.epoch), "trajectory_order": order,
                "model_state_hash": record.model_hash,
                "parent_ERM_model_state_hash": erm.checkpoint.model_hash,
                "previous_trajectory_model_state_hash": previous_hash[regime],
                "checkpoint_sha256": checkpoint["sha256"], "checkpoint_replay_pass": True,
                "optimizer_sha256": optimizer_descriptor["file_sha256"], "optimizer_replay_pass": True,
                "source_audit_sha256": source_descriptor["sha256"], "source_audit_replay_pass": True,
                "target_unlabeled_sha256": target_descriptor["sha256"], "target_unlabeled_replay_pass": True,
                "sidecar_sha256": sidecar_replay["sha256"], "sidecar_replay_pass": True,
                "support_replay_pass": True, "paired_model_init_pass": True,
                "level0_plan_replay_pass": True, "target_y_access": 0,
                "target_scientific_metrics": 0,
            })
            ledger.increment("complete_units")
            previous_hash[regime] = record.model_hash
    if len(output_units) != 81:
        raise C84L1CanaryError(f"{dataset} produced {len(output_units)} units instead of 81")
    return {
        "dataset": dataset,
        "units": output_units,
        "unit_count": 81,
        "level": LEVEL,
        "level_intervention": level1.application.as_dict(),
        "level0_plan_hashes": list(level0.plan_hashes),
        "level1_plan_hashes": list(level1.plan_hashes),
        "paired_model_init_hash": level1_init,
        "source_training_subjects": sorted(set(source.subject_id)),
        "source_audit_subjects": sorted(set(source_audit.subject_id)),
        "target_subjects": [target.target_subject_id],
        "source_training_pre_deletion_trials": level1.application.pre_trial_count,
        "source_training_post_deletion_trials": level1.application.post_trial_count,
        "source_audit_trial_count": len(source_audit.trial_id),
        "target_unlabeled_trial_count": len(target.trial_id),
        "target_label_access": 0,
        "scientific_metrics": 0,
    }


def run_real(
    *,
    authorization_path: Path = AUTHORIZATION_RECORD_PATH,
    output_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
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
            raise C84L1CanaryError("C84L1C requires an authorized Slurm CUDA allocation")
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not torch.are_deterministic_algorithms_enabled():
            raise C84L1CanaryError("torch deterministic algorithms are not enabled")

        ledger.stage("loader_source_identity_replay")
        runtime.verify_loader_source_files(binding["lock"])
        ledger.increment("loader_source_replays")
        objects, Lee2019_MI, Cho2017, PhysionetMI, MotorImagery = base._protected_loader_objects()
        ledger.increment("dataset_loader_imports")
        runtime.verify_loader_runtime_objects(binding["lock"], objects)
        loader_classes = (Lee2019_MI, Cho2017, PhysionetMI, MotorImagery)

        datasets = []
        for dataset in protocol.historical.DATASET_ORDER:
            ledger.stage(f"dataset_access:{dataset}")
            source, audit, target = base._load_canary_views(dataset, np, loader_classes, ledger)
            ledger.stage(f"level1_training_and_instrumentation:{dataset}")
            datasets.append(_run_dataset_training(
                dataset, source, audit, target, Path(binding["run_root"]) / dataset,
                binding, torch, np, ledger,
            ))
            ledger.publish_partial_manifest("IN_PROGRESS")

        rows = [row for dataset in datasets for row in dataset["units"]]
        complete = runtime.validate_complete_level1_canary_gate(rows)
        manifest = {
            "schema_version": "c84l1c_complete_engineering_manifest_v1",
            "execution_lock_sha256": binding["lock_sha256"],
            "canary_protocol_v1_sha256": binding["lock"]["canary_protocol"]["sha256"],
            "repair_protocol_sha256": binding["lock"]["repair_protocol"]["sha256"],
            "authorization_consumption_sha256": consumption["sha256"],
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


def synthetic_schema_dry_run() -> dict[str, Any]:
    canary_units = 0
    deletion_cells = []
    for dataset in protocol.historical.DATASET_ORDER:
        fixture = intervention.synthetic_source_panel(dataset, "A")
        applied = intervention.apply_level_intervention(
            dataset=dataset, panel="A", level=1,
            source_subjects=fixture["subjects"], source_labels=fixture["labels"],
            source_trial_ids=fixture["trial_ids"],
        )
        if len(applied.post_cell_counts) != 23:
            raise C84L1CanaryError("synthetic canary support graph is not 23 cells")
        canary_units += len(_candidate_rows(dataset))
        deletion_cells.append({
            "dataset": dataset,
            "subject": applied.deleted_source_subject,
            "class": applied.deleted_class,
            "deleted_rows": len(applied.deleted_indices),
            "post_cells": len(applied.post_cell_counts),
        })
    return {
        "schema_version": "c84l1c_schema_dry_run_v1",
        "datasets": 3,
        "candidate_units": canary_units,
        "training_phases": 9,
        "deletion_cells": deletion_cells,
        "target_y_access": 0,
        "scientific_metrics": 0,
        "real_EEG_access": 0,
        "training_forward_GPU": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C84 fixed-panel level-1 engineering canary")
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
            "fresh_authorization_required": True, "C84F": False, "C84S": False,
            "target_y_access": 0, "scientific_metrics": 0,
        }, sort_keys=True))
        return 0
    if args.command == "schema-dry-run":
        print(json.dumps(synthetic_schema_dry_run(), sort_keys=True))
        return 0
    print(json.dumps(run_real(
        authorization_path=args.authorization_record, output_root=args.output_root,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
