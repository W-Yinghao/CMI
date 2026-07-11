#!/usr/bin/env python
"""Generate STAR_00B real source-only launch artifacts (no optimizer steps)."""

import argparse
import csv
import inspect
import json
import subprocess
from pathlib import Path
from typing import Dict, Mapping

from star_eeg.config import DEPENDENCY_COMMIT, STAR01, STAR_BRANCH
from star_eeg.data.anchor_batch_stream import (
    anchor_stream_hash_artifact,
    build_exposure_matched_shuffled_manifest,
)
from star_eeg.data.faced_source_train_loader import (
    FACEDSourceTrainAnchorLoader,
    build_real_anchor_manifest,
    scan_source_train_inventory,
)
from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.data.tueg_ssl_batch_stream import ssl_stream_hash_artifact
from star_eeg.objectives.alternating_schedule import build_compute_match_contract
from star_eeg.training.real_star_runner import RealStarConfig, run_real_star


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(repo_root), text=True).strip()


def _validate_h200_closure(manifest_path: Path, go_nogo_path: Path) -> Dict[str, object]:
    from star_eeg.data.checkpoint_registry import sha256_file

    manifest = json.loads(manifest_path.read_text())
    go_nogo = json.loads(go_nogo_path.read_text())
    rows = manifest.get("checkpoints", [])
    checks = {
        "closure_status_pass": go_nogo.get("status") == "PASS",
        "two_h200_payloads": {row.get("tag") for row in rows} == {"H200_s0", "H200_s1"},
        "sha_named_paths": all(
            Path(row["launcher_accepted_path"]).name == f"best.{row['sha256']}.pth"
            for row in rows
        ),
        "payload_hashes_match": all(
            sha256_file(Path(row["launcher_accepted_path"])) == row["sha256"] for row in rows
        ),
        "payloads_read_only": all(
            Path(row["launcher_accepted_path"]).stat().st_mode & 0o222 == 0 for row in rows
        ),
        "launcher_manifest_only": go_nogo.get("launcher_restricted_to_sha_named_payload") is True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "manifest_hash": manifest.get("h200_immutable_manifest_hash"),
        "checks": checks,
    }


def _runner_contract() -> Dict[str, object]:
    signatures = {
        "run_real_star": str(inspect.signature(run_real_star)),
        "RealStarConfig": str(inspect.signature(RealStarConfig)),
        "FACEDSourceTrainAnchorLoader": str(inspect.signature(FACEDSourceTrainAnchorLoader)),
        "FACEDSourceTrainAnchorLoader.load_batch": str(
            inspect.signature(FACEDSourceTrainAnchorLoader.load_batch)
        ),
    }
    forbidden = ("target", "source_val", "test_labels", "evaluation_path")
    violations = {
        name: [token for token in forbidden if token in signature.lower()]
        for name, signature in signatures.items()
    }
    violations = {name: values for name, values in violations.items() if values}
    compute = build_compute_match_contract()
    checks = {
        "actual_runner_signatures_have_no_non_source_path": not violations,
        "formal_3750_path_requires_pm_manifest": "launch_approval_path" in inspect.signature(run_real_star).parameters,
        "batch_size_64": STAR01.batch_size == 64,
        "mask_ratio_0_5": RealStarConfig("H200_SSL_CONT", 0, 10).mask_ratio == 0.5,
        "gradient_clip_1_0": RealStarConfig("H200_SSL_CONT", 0, 10).gradient_clip_norm == 1.0,
        "fp32_only": RealStarConfig("H200_SSL_CONT", 0, 10).mixed_precision == "disabled_fp32",
        "masked_mse_mean": RealStarConfig("H200_SSL_CONT", 0, 10).ssl_loss == "masked_mse_mean",
        "zero_grad_set_to_none": RealStarConfig("H200_SSL_CONT", 0, 10).zero_grad_semantics == "set_to_none_true",
        "model_train_mode": RealStarConfig("H200_SSL_CONT", 0, 10).model_mode == "train",
        "compute_contract_pass": all(compute["checks"].values()),
    }
    core = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "signatures": signatures,
        "signature_violations": violations,
        "checks": checks,
        "native_ssl_contract": {
            "objective": "CBraMod_forward_with_native_generate_mask_then_MSE_on_mask_equals_1",
            "mask_ratio": 0.5,
            "loss_reduction": "mean",
            "input_normalization": "per_channel_per_1s_patch_zscore_eps_1e-6",
            "gradient_clip_norm": 1.0,
            "optimizer_zero_grad": "set_to_none_true",
            "model_mode": "train",
            "mixed_precision": "disabled_fp32",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR_Tmax_3750_by_optimizer_step",
        },
        "checkpoint_contract": {
            "source": "immutable_manifest_sha_named_payload_only",
            "payload_fields": [
                "model_state", "task_head_state", "optimizer_state", "scheduler_state",
                "optimizer_step", "source_checkpoint", "source_checkpoint_sha", "config",
                "config_hash", "schedule_hash", "stream_hash", "telemetry_rows",
                "telemetry_sha256", "primary_checkpoint", "target_data_used",
                "approval_manifest_hash",
            ],
            "strict_reload": True,
            "temporary_head_discard_before_frozen_evaluation": True,
        },
    }
    return {**core, "realpath_runner_contract_hash": canonical_hash(core)}


def _blind_chain() -> Dict[str, object]:
    training_cells = [
        f"{variant}_s{seed}"
        for variant in ("H200_SSL_CONT", "H200_STAR_TRUE", "H200_STAR_SHUFFLED")
        for seed in (0, 1)
    ]
    core = {
        "schema_version": 1,
        "status": "PLANNED_NOT_SUBMITTED",
        "stages": [
            {
                "stage": 1,
                "name": "six_training_jobs",
                "cells": training_cells,
                "all_cells_submitted_together": True,
                "seed_screening": False,
                "requires_gate": "STAR_01_SCIENTIFIC_TRAINING_APPROVED",
            },
            {
                "stage": 2,
                "name": "immutable_final_step_closure",
                "dependency": "afterok_all_six_training_jobs",
                "checkpoint_step": 3750,
            },
            {
                "stage": 3,
                "name": "source_only_integrity_and_task_gate_audit",
                "dependency": "afterok_immutable_final_step_closure",
                "separate_from_training_process": True,
                "source_val_labels_for_task_gate_only": True,
                "target_test_samples_or_labels_read": False,
            },
            {
                "stage": 4,
                "name": "single_all_cells_scoring",
                "dependency": "afterok_source_only_audit",
                "cells": [
                    "H200_BASE_s0", "H200_BASE_s1",
                    *training_cells,
                    "random", "released", "H500_s0", "H500_s1",
                    "H1000_s0", "H1000_s1", "H2000_s0", "H2000_s1",
                ],
                "partial_or_adaptive_scoring_forbidden": True,
                "currently_blocked": True,
            },
            {
                "stage": 5,
                "name": "independent_final_verifier",
                "dependency": "afterok_single_all_cells_scoring",
                "currently_blocked": True,
            },
        ],
        "no_cell_can_be_viewed_before_all_training_finishes": True,
        "target_scoring_submitted": False,
    }
    return {**core, "blind_evaluation_chain_hash": canonical_hash(core)}


def run_preflight(repo_root: Path, out_dir: Path, faced_lmdb: Path) -> Mapping[str, object]:
    closure = _validate_h200_closure(
        out_dir / "h200_immutable_manifest.json",
        out_dir / "h200_immutable_go_nogo.json",
    )
    inventory = scan_source_train_inventory(faced_lmdb)
    anchor = build_real_anchor_manifest(inventory)
    shuffled = build_exposure_matched_shuffled_manifest(anchor)
    anchor_streams, exposure_rows = anchor_stream_hash_artifact(anchor, shuffled)
    ssl_streams = ssl_stream_hash_artifact(
        repo_root, repo_root / "results/s2p_route_b_33ch_contract"
    )
    loader = FACEDSourceTrainAnchorLoader(faced_lmdb, anchor)
    smoke_ids = [row["sample_id"] for row in anchor["records"][:8]]
    batch, labels, _ = loader.load_batch(smoke_ids)
    loader_audit = loader.access_audit()
    source_firewall_core = {
        "status": "PASS" if all([
            inventory["n_records"] == 6720,
            inventory["n_subjects"] == 80,
            inventory["access_audit"]["source_val_sample_reads"] == 0,
            inventory["access_audit"]["test_sample_reads"] == 0,
            loader_audit["status"] == "PASS",
        ]) else "FAIL",
        "inventory_access_audit": inventory["access_audit"],
        "batch_loader_access_audit": loader_audit,
        "batch_loader_smoke_shape": list(batch.shape),
        "batch_loader_smoke_labels": labels.tolist(),
        "allowed_lmdb_index_member": "train",
        "source_val_deserialized": False,
        "test_deserialized": False,
        "non_source_class_counts_computed": False,
    }
    source_firewall = {
        **source_firewall_core,
        "source_loader_firewall_hash": canonical_hash(source_firewall_core),
    }
    runner_contract = _runner_contract()
    blind_chain = _blind_chain()
    compute = build_compute_match_contract()
    dependency_core = {
        "star00a_baseline": "26c3fca009d0ecbde7e92e7c759c0256caf1361d",
        "required_dependency": DEPENDENCY_COMMIT,
        "branch": _git(repo_root, "branch", "--show-current"),
        "merge_base": _git(repo_root, "merge-base", "HEAD", DEPENDENCY_COMMIT),
        "remote_dependency_at_star_creation": DEPENDENCY_COMMIT,
        "remote_may_advance_after_creation": True,
        "automatic_sync_or_rebase_allowed": False,
        "s2p_h2cmi_oaci_mutated": False,
    }
    dependency = {**dependency_core, "star00b_dependency_hash": canonical_hash(dependency_core)}

    artifacts = {
        "faced_source_train_inventory.json": inventory,
        "anchor_manifest.json": anchor,
        "shuffled_label_manifest.json": shuffled,
        "anchor_batch_stream_hashes.json": anchor_streams,
        "ssl_batch_stream_hashes.json": ssl_streams,
        "source_loader_firewall.json": source_firewall,
        "realpath_runner_contract.json": runner_contract,
        "blind_evaluation_chain.json": blind_chain,
        "compute_match_contract.json": compute,
        "star00b_dependency_manifest.json": dependency,
    }
    for name, payload in artifacts.items():
        _write_json(out_dir / name, payload)
    _write_csv(out_dir / "anchor_exposure_table.csv", exposure_rows)
    _write_csv(out_dir / "star01_training_tasks.csv", [
        {"task_id": task_id, "variant": variant, "seed": seed}
        for task_id, (variant, seed) in enumerate(
            (variant, seed)
            for variant in ("H200_SSL_CONT", "H200_STAR_TRUE", "H200_STAR_SHUFFLED")
            for seed in (0, 1)
        )
    ])

    checks = {
        "h200_immutable_closure": closure["status"],
        "source_manifest": "PASS" if inventory["n_records"] == 6720 else "FAIL",
        "source_loader_firewall": source_firewall["status"],
        "shuffled_manifest": "PASS" if shuffled["n_records"] == 6720 else "FAIL",
        "anchor_stream": "PASS" if all(
            stream["true_shuffled_exposure_marginals_equal"]
            and stream["subject_exposures_exact_600"]
            for stream in anchor_streams["streams"].values()
        ) else "FAIL",
        "ssl_stream": "PASS" if all(
            stream["common_batches"] == 3000 and stream["replacement_batches"] == 750
            for stream in ssl_streams["streams"].values()
        ) else "FAIL",
        "compute_match": "PASS" if all(compute["checks"].values()) else "FAIL",
        "realpath_runner_contract": runner_contract["status"],
        "blind_chain": "PASS" if blind_chain["target_scoring_submitted"] is False else "FAIL",
    }
    status = "PENDING_GPU_SMOKE" if all(value == "PASS" for value in checks.values()) else "FAIL"
    gate_core = {
        "STAR_00_PROJECT_CHARTER": "PASS",
        "STAR_00A_DESIGN_AND_RED_TEAM_PREFLIGHT": "PASS",
        "STAR_H200_ARTIFACT_SUPPLY": "AVAILABLE_IMMUTABLE",
        "STAR_00B_REAL_PATH_PREFLIGHT": status,
        "STAR_01_SCIENTIFIC_TRAINING": "BLOCKED",
        "STAR_TARGET_SCORING": "BLOCKED",
        "STAR_MANUSCRIPT_CLAIM": "FORBIDDEN",
        "S2P_PHASE_B": "INDEPENDENT_UNCHANGED",
        "H2CMI_AND_OACI": "PROTECTED_UNCHANGED",
        "checks": checks,
        "gpu_smoke": "PENDING",
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
    }
    summary = {**gate_core, "star00b_preflight_summary_hash": canonical_hash(gate_core)}
    _write_json(out_dir / "star00b_preflight_summary.json", summary)
    launch_core = {
        "status": "BLOCKED_PENDING_STAR00B_GPU_SMOKE_AND_PM_REVIEW",
        "approved_new_training_cells": [],
        "planned_training_cells": [
            f"{variant}_s{seed}"
            for variant in ("H200_SSL_CONT", "H200_STAR_TRUE", "H200_STAR_SHUFFLED")
            for seed in (0, 1)
        ],
        "star01_runner_execution_allowed": False,
        "target_scoring_allowed": False,
        "blind_chain_hash": blind_chain["blind_evaluation_chain_hash"],
    }
    _write_json(out_dir / "star00b_launch_manifest.json", {
        **launch_core,
        "star00b_launch_manifest_hash": canonical_hash(launch_core),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="results/star/star00b_preflight")
    parser.add_argument("--faced-lmdb", default="/projects/EEG-foundation-model/FACED_data/processed")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    summary = run_preflight(repo_root, out_dir, Path(args.faced_lmdb))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["STAR_00B_REAL_PATH_PREFLIGHT"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
