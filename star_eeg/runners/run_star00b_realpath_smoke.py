#!/usr/bin/env python
"""Run the approved H200_s0 B/C/D ten-step real CUDA smoke."""

import argparse
import hashlib
import json
from pathlib import Path

from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.objectives.alternating_schedule import SSL_CONT, STAR_SHUFFLED, STAR_TRUE
from star_eeg.training.real_star_runner import RealStarConfig, run_real_star


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_runtime_memory_estimate(summaries, smoke_steps: int):
    observed = {}
    projected_hours = []
    for variant, row in summaries.items():
        seconds_per_step = row["wall_seconds"] / smoke_steps
        projection = seconds_per_step * 3750 / 3600
        projected_hours.append(projection)
        observed[variant] = {
            "ten_step_wall_seconds": row["wall_seconds"],
            "naive_linear_projection_hours": projection,
            "peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"],
        }
    core = {
        "basis": "sequential_ten_step_real_A40_smoke",
        "observed": observed,
        "cold_cache_and_one_time_setup_confounded": True,
        "variant_runtime_comparison_valid": False,
        "rough_per_cell_hours_lower": min(projected_hours),
        "rough_per_cell_hours_upper": max(projected_hours),
        "recommended_slurm_time_limit_hours": 24,
        "recommended_gpu_memory_gib": 24,
        "observed_peak_gpu_memory_gib": max(
            row["peak_gpu_memory_bytes"] for row in summaries.values()
        ) / (1024 ** 3),
        "selection_use": False,
        "planning_only": True,
    }
    return {**core, "runtime_memory_estimate_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="results/star/star00b_preflight")
    parser.add_argument(
        "--runtime-root",
        default="/home/infres/yinwang/CMI_AAAI_star_runtime/results/star00b_realpath_smoke",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps != 10:
        raise ValueError("STAR_00B smoke is frozen at exactly 10 optimizer steps")
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    runtime_root = Path(args.runtime_root)
    execution_files = [
        repo_root / "star_eeg/training/real_star_runner.py",
        repo_root / "star_eeg/data/faced_source_train_loader.py",
        repo_root / "star_eeg/data/anchor_batch_stream.py",
        repo_root / "star_eeg/data/tueg_ssl_batch_stream.py",
        repo_root / "star_eeg/objectives/alternating_schedule.py",
        repo_root / "star_eeg/runners/run_star00b_realpath_smoke.py",
        repo_root / "star_eeg/slurm/star00b_realpath_smoke.sbatch",
    ]
    execution_source_hashes = {
        str(path.relative_to(repo_root)): _file_sha256(path) for path in execution_files
    }
    common = {
        "repo_root": repo_root,
        "immutable_manifest_path": out_dir / "h200_immutable_manifest.json",
        "immutable_go_nogo_path": out_dir / "h200_immutable_go_nogo.json",
        "anchor_manifest_path": out_dir / "anchor_manifest.json",
        "shuffled_manifest_path": out_dir / "shuffled_label_manifest.json",
        "faced_lmdb_path": Path("/projects/EEG-foundation-model/FACED_data/processed"),
        "contract_dir": repo_root / "results/s2p_route_b_33ch_contract",
        "device_name": args.device,
    }
    summaries = {}
    for variant in (SSL_CONT, STAR_TRUE, STAR_SHUFFLED):
        summaries[variant] = run_real_star(
            config=RealStarConfig(variant=variant, model_seed=0, optimizer_steps=args.steps),
            runtime_output_dir=runtime_root / variant,
            **common,
        )

    b, c, d = summaries[SSL_CONT], summaries[STAR_TRUE], summaries[STAR_SHUFFLED]
    checks = {
        "start_model_state_hash_b_c_d_equal": len({
            b["start_model_state_hash"], c["start_model_state_hash"], d["start_model_state_hash"]
        }) == 1,
        "model_update_scope_b_c_d_equal": len({
            b["model_update_scope_hash"], c["model_update_scope_hash"], d["model_update_scope_hash"]
        }) == 1,
        "common_ssl_batch_ids_b_c_d_equal": b["common_ssl_batch_id_hashes"] == c["common_ssl_batch_id_hashes"] == d["common_ssl_batch_id_hashes"],
        "common_ssl_batch_tensors_b_c_d_equal": b["common_ssl_batch_tensor_hashes"] == c["common_ssl_batch_tensor_hashes"] == d["common_ssl_batch_tensor_hashes"],
        "anchor_x_ids_c_d_equal": c["anchor_x_id_hashes"] == d["anchor_x_id_hashes"],
        "anchor_x_tensors_c_d_equal": c["anchor_x_tensor_hashes"] == d["anchor_x_tensor_hashes"],
        "true_and_shuffled_label_streams_differ": c["anchor_label_hashes"] != d["anchor_label_hashes"],
        "b_has_two_replacement_ssl_slots": len(b["replacement_ssl_batch_id_hashes"]) == 2,
        "c_d_have_two_anchor_slots": len(c["anchor_x_id_hashes"]) == len(d["anchor_x_id_hashes"]) == 2,
        "all_losses_finite": all(row["losses_finite"] for row in summaries.values()),
        "all_gradients_finite": all(row["gradients_finite"] for row in summaries.values()),
        "all_parameter_deltas_finite": all(row["parameter_deltas_finite"] for row in summaries.values()),
        "all_checkpoints_strict_reload": all(row["checkpoint_strict_reload_pass"] for row in summaries.values()),
        "all_source_starts_unchanged": all(row["source_checkpoint_unchanged"] for row in summaries.values()),
        "b_temporary_head_unchanged": b["temporary_head_unchanged"],
        "c_d_temporary_heads_updated": not c["temporary_head_unchanged"] and not d["temporary_head_unchanged"],
        "source_only_loader_firewall": all(row["faced_loader_access_audit"]["status"] == "PASS" for row in summaries.values()),
        "same_h200_s0_source_sha": len({row["source_checkpoint_sha_before"] for row in summaries.values()}) == 1,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    telemetry_core = {
        "phase": "STAR_00B_realpath_smoke",
        "steps_per_variant": args.steps,
        "model_seed": 0,
        "execution_source_hashes": execution_source_hashes,
        "summaries": summaries,
    }
    telemetry = {**telemetry_core, "realpath_smoke_telemetry_hash": canonical_hash(telemetry_core)}
    _write_json(out_dir / "realpath_smoke_telemetry.json", telemetry)
    summary_core = {
        "phase": "STAR_00B_realpath_smoke",
        "status": status,
        "gpu_names": sorted({row["gpu_name"] for row in summaries.values()}),
        "steps_per_variant": args.steps,
        "variants": [SSL_CONT, STAR_TRUE, STAR_SHUFFLED],
        "model_seed": 0,
        "checks": checks,
        "peak_gpu_memory_bytes": {
            variant: row["peak_gpu_memory_bytes"] for variant, row in summaries.items()
        },
        "wall_seconds": {variant: row["wall_seconds"] for variant, row in summaries.items()},
        "source_checkpoint_sha": b["source_checkpoint_sha_before"],
        "execution_source_hashes": execution_source_hashes,
        "scientific_training_run": False,
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
    }
    smoke_summary = {**summary_core, "realpath_smoke_summary_hash": canonical_hash(summary_core)}
    _write_json(out_dir / "realpath_smoke_summary.json", smoke_summary)

    _write_json(
        out_dir / "runtime_memory_estimate.json",
        build_runtime_memory_estimate(summaries, args.steps),
    )

    existing = json.loads((out_dir / "star00b_preflight_summary.json").read_text())
    final_core = {
        key: value for key, value in existing.items()
        if key != "star00b_preflight_summary_hash"
    }
    final_core["gpu_smoke"] = status
    final_core["STAR_00B_REAL_PATH_PREFLIGHT"] = "PASS" if status == "PASS" else "FAIL"
    final_core["STAR_01_SCIENTIFIC_TRAINING"] = "BLOCKED_PENDING_PM_REVIEW"
    final_core["formal_3750_step_training_run"] = False
    final_core["target_metrics_computed"] = False
    _write_json(out_dir / "star00b_preflight_summary.json", {
        **final_core,
        "star00b_preflight_summary_hash": canonical_hash(final_core),
    })
    launch_core = {
        "status": "STAR00B_PASS_PENDING_PM_STAR01_APPROVAL" if status == "PASS" else "BLOCKED_REALPATH_SMOKE_FAILED",
        "approved_new_training_cells": [],
        "planned_training_cells": [
            f"{variant}_s{seed}"
            for variant in (SSL_CONT, STAR_TRUE, STAR_SHUFFLED)
            for seed in (0, 1)
        ],
        "star01_runner_execution_allowed": False,
        "target_scoring_allowed": False,
        "realpath_smoke_summary_hash": smoke_summary["realpath_smoke_summary_hash"],
        "requires_new_pm_gate": True,
    }
    _write_json(out_dir / "star00b_launch_manifest.json", {
        **launch_core,
        "star00b_launch_manifest_hash": canonical_hash(launch_core),
    })
    print(json.dumps(smoke_summary, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
