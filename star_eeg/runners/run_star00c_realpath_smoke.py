#!/usr/bin/env python
"""Run the final-code STAR_00C ten-step persistent CUDA smoke."""

import argparse
import hashlib
import json
from pathlib import Path

from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.objectives.alternating_schedule import (
    SSL_CONT,
    STAR_SHUFFLED,
    STAR_TRUE,
)
from star_eeg.training.persistence import (
    claim_attempt_directory,
    sha256_file,
    tree_is_read_only,
    validate_telemetry_file,
)
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


def _hash_valid(path: Path, field: str) -> bool:
    payload = json.loads(path.read_text())
    core = {key: value for key, value in payload.items() if key != field}
    return canonical_hash(core) == payload.get(field)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out-dir", default="results/star/star00c_preflight")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps != 10:
        raise ValueError("STAR_00C final-code smoke is frozen at ten steps")
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    runtime_root = Path(args.runtime_root)
    execution_files = (
        "star_eeg/training/real_star_runner.py",
        "star_eeg/training/persistence.py",
        "star_eeg/training/approval_lock.py",
        "star_eeg/runners/run_star01_train.py",
        "star_eeg/runners/submit_star01a_blind_chain.py",
        "star_eeg/artifacts/close_star01a_finals.py",
        "star_eeg/slurm/star01_train_array.sbatch",
        "star_eeg/slurm/star01a_final_closure.sbatch",
        "star_eeg/runners/run_star00c_realpath_smoke.py",
        "star_eeg/slurm/star00c_realpath_smoke.sbatch",
    )
    execution_source_hashes = {
        relative: _file_sha256(repo_root / relative) for relative in execution_files
    }
    common = {
        "repo_root": repo_root,
        "immutable_manifest_path": repo_root / "results/star/star00b_preflight/h200_immutable_manifest.json",
        "immutable_go_nogo_path": repo_root / "results/star/star00b_preflight/h200_immutable_go_nogo.json",
        "anchor_manifest_path": repo_root / "results/star/star00b_preflight/anchor_manifest.json",
        "shuffled_manifest_path": repo_root / "results/star/star00b_preflight/shuffled_label_manifest.json",
        "faced_lmdb_path": Path("/projects/EEG-foundation-model/FACED_data/processed"),
        "contract_dir": repo_root / "results/s2p_route_b_33ch_contract",
        "device_name": args.device,
        "attempt_id": "attempt_00",
    }
    summaries = {}
    persistence_rows = []
    for variant in (SSL_CONT, STAR_TRUE, STAR_SHUFFLED):
        attempt_dir = runtime_root / variant / "attempt_00"
        result = run_real_star(
            config=RealStarConfig(
                variant=variant,
                model_seed=0,
                optimizer_steps=args.steps,
            ),
            runtime_output_dir=attempt_dir,
            **common,
        )
        summaries[variant] = result
        telemetry = validate_telemetry_file(
            attempt_dir / "telemetry.jsonl", args.steps
        )
        completion = json.loads((attempt_dir / "completion.json").read_text())
        no_overwrite_rejected = False
        try:
            claim_attempt_directory(attempt_dir, "attempt_00")
        except FileExistsError:
            no_overwrite_rejected = True
        persistence_rows.append({
            "variant": variant,
            "attempt_dir": str(attempt_dir.resolve()),
            "telemetry_rows": telemetry["rows"],
            "telemetry_sha256": telemetry["sha256"],
            "run_manifest_hash_valid": _hash_valid(
                attempt_dir / "run_manifest.json", "run_manifest_hash"
            ),
            "execution_manifest_hash_valid": _hash_valid(
                attempt_dir / "execution_manifest.json", "execution_manifest_hash"
            ),
            "run_summary_hash_valid": _hash_valid(
                attempt_dir / "run_summary.json", "run_summary_hash"
            ),
            "completion_hash_valid": _hash_valid(
                attempt_dir / "completion.json", "completion_hash"
            ),
            "completion_checks_pass": all(completion["checks"].values()),
            "checkpoint_sha256": sha256_file(
                attempt_dir / "checkpoints/step_0010.pth"
            ),
            "attempt_tree_read_only": tree_is_read_only(attempt_dir),
            "same_attempt_reuse_rejected": no_overwrite_rejected,
        })

    b, c, d = summaries[SSL_CONT], summaries[STAR_TRUE], summaries[STAR_SHUFFLED]
    checks = {
        "start_model_state_hash_b_c_d_equal": len({
            b["start_model_state_hash"],
            c["start_model_state_hash"],
            d["start_model_state_hash"],
        }) == 1,
        "model_update_scope_b_c_d_equal": len({
            b["model_update_scope_hash"],
            c["model_update_scope_hash"],
            d["model_update_scope_hash"],
        }) == 1,
        "common_ssl_batch_ids_b_c_d_equal": (
            b["common_ssl_batch_id_hashes"]
            == c["common_ssl_batch_id_hashes"]
            == d["common_ssl_batch_id_hashes"]
        ),
        "common_ssl_batch_tensors_b_c_d_equal": (
            b["common_ssl_batch_tensor_hashes"]
            == c["common_ssl_batch_tensor_hashes"]
            == d["common_ssl_batch_tensor_hashes"]
        ),
        "anchor_x_ids_c_d_equal": c["anchor_x_id_hashes"] == d["anchor_x_id_hashes"],
        "anchor_x_tensors_c_d_equal": c["anchor_x_tensor_hashes"] == d["anchor_x_tensor_hashes"],
        "true_shuffled_labels_differ": c["anchor_label_hashes"] != d["anchor_label_hashes"],
        "b_has_two_replacement_ssl_slots": len(
            b["replacement_ssl_batch_id_hashes"]
        ) == 2,
        "c_d_have_two_anchor_slots": (
            len(c["anchor_x_id_hashes"]) == len(d["anchor_x_id_hashes"]) == 2
        ),
        "b_temporary_head_unchanged": b["temporary_head_unchanged"],
        "c_d_temporary_heads_updated": (
            not c["temporary_head_unchanged"]
            and not d["temporary_head_unchanged"]
        ),
        "same_h200_s0_source_sha": len({
            row["source_checkpoint_sha_before"] for row in summaries.values()
        }) == 1,
        "all_losses_gradients_deltas_finite": all(
            row["losses_finite"]
            and row["gradients_finite"]
            and row["parameter_deltas_finite"]
            for row in summaries.values()
        ),
        "all_checkpoint_strict_reload": all(
            row["checkpoint_strict_reload_pass"] for row in summaries.values()
        ),
        "all_sources_unchanged": all(
            row["source_checkpoint_unchanged"] for row in summaries.values()
        ),
        "all_completion_markers_valid": all(
            row["completion_checks_pass"] for row in persistence_rows
        ),
        "all_ten_telemetry_rows_persisted": all(
            row["telemetry_rows"] == 10 for row in persistence_rows
        ),
        "all_manifest_hashes_valid": all(
            row["run_manifest_hash_valid"]
            and row["execution_manifest_hash_valid"]
            and row["run_summary_hash_valid"]
            and row["completion_hash_valid"]
            for row in persistence_rows
        ),
        "all_attempt_trees_frozen": all(
            row["attempt_tree_read_only"] for row in persistence_rows
        ),
        "same_attempt_reuse_rejected": all(
            row["same_attempt_reuse_rejected"] for row in persistence_rows
        ),
        "update_accounting_exact": (
            b["encoder_updates"] == 10
            and b["full_reconstruction_updates"] == 10
            and b["anchor_updates"] == 0
            and c["encoder_updates"] == d["encoder_updates"] == 10
            and c["full_reconstruction_updates"]
            == d["full_reconstruction_updates"]
            == 8
            and c["anchor_updates"] == d["anchor_updates"] == 2
        ),
        "comparison_not_claimed_flop_matched": all(
            row["strict_flop_matched"] is False for row in summaries.values()
        ),
        "source_only_loader_firewall": all(
            row["faced_loader_access_audit"]["source_val_sample_reads"] == 0
            and row["faced_loader_access_audit"]["test_sample_reads"] == 0
            for row in summaries.values()
        ),
    }
    persistence_core = {
        "schema_version": 1,
        "phase": "STAR_00C_PERSISTENT_REALPATH_SMOKE",
        "rows": persistence_rows,
    }
    persistence_artifact = {
        **persistence_core,
        "smoke_persistence_index_hash": canonical_hash(persistence_core),
    }
    _write_json(
        out_dir / "smoke_persistence_index.json", persistence_artifact
    )
    summary_core = {
        "schema_version": 1,
        "phase": "STAR_00C_FINAL_CODE_REALPATH_SMOKE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "steps_per_variant": 10,
        "model_seed": 0,
        "variants": [SSL_CONT, STAR_TRUE, STAR_SHUFFLED],
        "checks": checks,
        "execution_source_hashes": execution_source_hashes,
        "peak_gpu_memory_bytes": {
            variant: row["peak_gpu_memory_bytes"]
            for variant, row in summaries.items()
        },
        "wall_seconds": {
            variant: row["wall_seconds"] for variant, row in summaries.items()
        },
        "formal_3750_step_training_run": False,
        "target_metrics_computed": False,
    }
    summary = {
        **summary_core,
        "star00c_realpath_smoke_summary_hash": canonical_hash(summary_core),
    }
    _write_json(out_dir / "realpath_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
