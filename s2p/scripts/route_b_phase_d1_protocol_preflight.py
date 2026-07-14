#!/usr/bin/env python
"""Build the metadata-only Phase-D1 nested-trajectory protocol package.

This script allocates no tensors from the EEG recordings and invokes no trainer,
SLURM command, downstream task, or optimizer. It reads corpus metadata, constructs
exact nested manifests, hashes deterministic initial states, and freezes schedules.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WINDOWS_PER_HOUR = 120
BATCH_SIZE = 64
P_LOW = 18_750
P_HIGH = 93_750
VAL_CADENCE = 1_875
BASE_LR = 5e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 5e-2
MASK_RATIO = 0.5
SUBSET_LEVELS = (200, 1000)
SUBSET_SEEDS = (0, 1)
INIT_SEEDS = (0, 1)
NESTING_CONTRACT = "S2P-D1-nested-v1"
STREAM_CONTRACT = "S2P-D1-stream-v1"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def derived_seed(namespace, *parts):
    text = "|".join([namespace, *(str(part) for part in parts)])
    value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return value % (2**31 - 1)


def allocation_seed(subset_seed, group_id):
    return derived_seed(NESTING_CONTRACT, f"subset={subset_seed}", f"group={group_id}")


def stream_seed(subset_seed, init_seed):
    return derived_seed(STREAM_CONTRACT, f"subset={subset_seed}", f"init={init_seed}")


def loader_seed_root(stream_seed_value):
    return derived_seed("S2P-D1-loader-v1", stream_seed_value)


def mask_seed_root(stream_seed_value):
    return derived_seed("S2P-D1-mask-v1", stream_seed_value)


def ordered_frame(frame, seed):
    frame = frame.sort_values("recording_id").reset_index(drop=True)
    order = np.arange(len(frame))
    np.random.default_rng(seed).shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


def allocate_rows(frame, target_windows, seed, group_id):
    ordered = ordered_frame(frame, seed)
    rows = []
    remaining = int(target_windows)
    for rank, (_, record) in enumerate(ordered.iterrows()):
        if remaining <= 0:
            break
        take = min(int(record["avail_w"]), remaining)
        if take <= 0:
            continue
        rows.append(
            {
                "allocation_rank": rank,
                "group_id": group_id,
                "reference_scheme": record["reference_scheme"],
                "subject": int(record["subject"]),
                "recording_id": int(record["recording_id"]),
                "filepath_relative": str(record["filepath"]),
                "window_start": 0,
                "window_stop_exclusive": take,
                "take_windows": take,
                "available_windows": int(record["avail_w"]),
            }
        )
        remaining -= take
    if remaining:
        raise RuntimeError(f"{group_id}: allocation shortfall={remaining}")
    return rows


def identities(rows):
    result = set()
    for row in rows:
        result.update(
            (
                int(row["subject"]),
                int(row["recording_id"]),
                row["filepath_relative"],
                index,
            )
            for index in range(
                int(row["window_start"]), int(row["window_stop_exclusive"])
            )
        )
    return result


def hash_identities(values):
    h = hashlib.sha256()
    for subject, recording, filepath, index in sorted(values):
        h.update(f"{subject}|{recording}|{filepath}|{index}\n".encode())
    return h.hexdigest()


def model_state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0")
        h.update(str(array.dtype).encode() + b"\0")
        h.update(str(tuple(array.shape)).encode() + b"\0")
        h.update(array.tobytes())
    return h.hexdigest()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def cosine_lr(completed_updates):
    return MIN_LR + (BASE_LR - MIN_LR) * (
        1.0 + math.cos(math.pi * completed_updates / P_HIGH)
    ) / 2.0


def gpu_name(log_path):
    match = re.search(r"^(.+?),\s+\d+ MiB$", Path(log_path).read_text(), re.MULTILINE)
    return match.group(1) if match else "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cbramod-root", type=Path, required=True)
    parser.add_argument("--b1-launch-root", type=Path, required=True)
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=Path("results/s2p_route_b_33ch_contract"),
    )
    parser.add_argument(
        "--immutable-manifest",
        type=Path,
        default=Path(
            "results/s2p_route_b_phase_b_checkpoint_closure/"
            "phase_b_checkpoint_immutable_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/s2p_route_b_phase_d1_protocol"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    contract_dir = (repo_root / args.contract_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    cbramod_root = args.cbramod_root.resolve()
    b1_results = (
        args.b1_launch_root.resolve() / "results" / "s2p_route_b_33ch_b1"
    )
    immutable_manifest = (repo_root / args.immutable_manifest).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root / "s2p" / "scripts"))
    sys.path.insert(0, str(cbramod_root))
    import route_b_33ch_loader as route_loader
    from models.cbramod import CBraMod

    metadata = route_loader._meta()
    val_subjects = set(map(int, route_loader.fixed_val_subjects()))
    train = metadata[~metadata["subject"].isin(val_subjects)].copy()
    group_mix = pd.read_csv(contract_dir / "route_b_group_mix_by_budget.csv")
    targets = {
        budget: {
            row.group_id: int(row.take_windows)
            for row in group_mix[group_mix["budget_h"].astype(int) == budget].itertuples()
        }
        for budget in SUBSET_LEVELS
    }
    if set(targets[200]) != set(targets[1000]):
        raise RuntimeError("200h and 1000h group sets differ")

    manifest_rows = []
    group_rows = []
    exposure_rows = []
    selection_sets = {}
    selection_hashes = {}
    val_subject_hash = sha256_json(sorted(val_subjects))

    for subset_seed_value in SUBSET_SEEDS:
        allocations = {budget: [] for budget in SUBSET_LEVELS}
        for group_id in sorted(targets[1000]):
            frame = train[train["group_id"] == group_id]
            seed = allocation_seed(subset_seed_value, group_id)
            group_allocations = {}
            for budget in SUBSET_LEVELS:
                group_allocations[budget] = allocate_rows(
                    frame, targets[budget][group_id], seed, group_id
                )
                allocations[budget].extend(group_allocations[budget])
            low_map = {
                int(row["recording_id"]): int(row["take_windows"])
                for row in group_allocations[200]
            }
            for budget in SUBSET_LEVELS:
                realized = sum(
                    int(row["take_windows"]) for row in group_allocations[budget]
                )
                group_rows.append(
                    {
                        "subset_seed": subset_seed_value,
                        "unique_data_h": budget,
                        "group_id": group_id,
                        "reference_scheme": group_allocations[budget][0][
                            "reference_scheme"
                        ],
                        "target_windows": targets[budget][group_id],
                        "realized_windows": realized,
                        "target_proportion": f"{targets[budget][group_id] / (budget * WINDOWS_PER_HOUR):.12f}",
                        "realized_proportion": f"{realized / (budget * WINDOWS_PER_HOUR):.12f}",
                        "absolute_proportion_error": "0.000000000000",
                        "allocation_seed": seed,
                        "exact_group_quota_pass": "true",
                    }
                )
            for budget in SUBSET_LEVELS:
                for row in group_allocations[budget]:
                    core_windows = (
                        int(row["take_windows"])
                        if budget == 200
                        else min(
                            int(row["take_windows"]),
                            low_map.get(int(row["recording_id"]), 0),
                        )
                    )
                    output = {
                        "subset_seed": subset_seed_value,
                        "unique_data_h": budget,
                        "subset_contract": NESTING_CONTRACT,
                        "allocation_seed": seed,
                        **row,
                        "core_u200_windows_in_row": core_windows,
                        "hours": f"{int(row['take_windows']) / WINDOWS_PER_HOUR:.9f}",
                    }
                    output["manifest_row_sha256"] = sha256_json(output)
                    manifest_rows.append(output)

        for budget in SUBSET_LEVELS:
            chosen = identities(allocations[budget])
            expected = budget * WINDOWS_PER_HOUR
            if len(chosen) != expected:
                raise RuntimeError(
                    f"subset={subset_seed_value} H={budget}: {len(chosen)} != {expected}"
                )
            if {int(row["subject"]) for row in allocations[budget]} & val_subjects:
                raise RuntimeError("train/validation subject overlap")
            selection_sets[(subset_seed_value, budget)] = chosen
            selection_hashes[(subset_seed_value, budget)] = hash_identities(chosen)

            by_subject = defaultdict(lambda: {"windows": 0, "records": set(), "groups": set()})
            for row in allocations[budget]:
                item = by_subject[int(row["subject"])]
                item["windows"] += int(row["take_windows"])
                item["records"].add(int(row["recording_id"]))
                item["groups"].add(row["group_id"])
            for subject, item in sorted(by_subject.items()):
                exposure_rows.append(
                    {
                        "subset_seed": subset_seed_value,
                        "unique_data_h": budget,
                        "subject": subject,
                        "windows": item["windows"],
                        "hours": f"{item['windows'] / WINDOWS_PER_HOUR:.9f}",
                        "recordings": len(item["records"]),
                        "group_ids": ";".join(sorted(item["groups"])),
                    }
                )

    overlap_rows = []
    for subset_seed_value in SUBSET_SEEDS:
        low = selection_sets[(subset_seed_value, 200)]
        high = selection_sets[(subset_seed_value, 1000)]
        intersection = low & high
        low_subjects = {item[0] for item in low}
        high_subjects = {item[0] for item in high}
        low_records = {item[1] for item in low}
        high_records = {item[1] for item in high}
        extension = high - low
        overlap_rows.append(
            {
                "comparison": "within_subset_seed_u200_in_u1000",
                "subset_seed_left": subset_seed_value,
                "subset_seed_right": subset_seed_value,
                "unique_data_h_left": 200,
                "unique_data_h_right": 1000,
                "left_windows": len(low),
                "right_windows": len(high),
                "intersection_windows": len(intersection),
                "extension_windows": len(extension),
                "left_window_sha256": selection_hashes[(subset_seed_value, 200)],
                "right_window_sha256": selection_hashes[(subset_seed_value, 1000)],
                "intersection_window_sha256": hash_identities(intersection),
                "extension_window_sha256": hash_identities(extension),
                "left_nested_in_right": "true" if low <= high else "false",
                "left_subjects": len(low_subjects),
                "right_subjects": len(high_subjects),
                "subject_overlap": len(low_subjects & high_subjects),
                "left_recordings": len(low_records),
                "right_recordings": len(high_records),
                "recording_overlap": len(low_records & high_records),
                "exact_nesting_pass": "true" if low <= high else "false",
            }
        )
    for budget in SUBSET_LEVELS:
        left = selection_sets[(0, budget)]
        right = selection_sets[(1, budget)]
        intersection = left & right
        overlap_rows.append(
            {
                "comparison": "cross_subset_seed_same_u",
                "subset_seed_left": 0,
                "subset_seed_right": 1,
                "unique_data_h_left": budget,
                "unique_data_h_right": budget,
                "left_windows": len(left),
                "right_windows": len(right),
                "intersection_windows": len(intersection),
                "extension_windows": "NA",
                "left_window_sha256": selection_hashes[(0, budget)],
                "right_window_sha256": selection_hashes[(1, budget)],
                "intersection_window_sha256": hash_identities(intersection),
                "extension_window_sha256": "NA",
                "left_nested_in_right": "true" if left <= right else "false",
                "left_subjects": len({item[0] for item in left}),
                "right_subjects": len({item[0] for item in right}),
                "subject_overlap": len(
                    {item[0] for item in left} & {item[0] for item in right}
                ),
                "left_recordings": len({item[1] for item in left}),
                "right_recordings": len({item[1] for item in right}),
                "recording_overlap": len(
                    {item[1] for item in left} & {item[1] for item in right}
                ),
                "exact_nesting_pass": "not_applicable",
            }
        )

    write_csv(output_dir / "phase_d1_nested_subset_manifest.csv", manifest_rows)
    write_csv(output_dir / "phase_d1_subset_overlap_verification.csv", overlap_rows)
    write_csv(output_dir / "phase_d1_subject_exposure.csv", exposure_rows)
    write_csv(output_dir / "phase_d1_group_mixture.csv", group_rows)

    cbramod_sources = {
        relative: sha256_file(cbramod_root / relative)
        for relative in (
            "models/cbramod.py",
            "models/criss_cross_transformer.py",
            "utils/util.py",
        )
    }
    model_source_sha = sha256_json(cbramod_sources)
    initial_rows = []
    pair_hashes = {}
    for subset_seed_value in SUBSET_SEEDS:
        for init_seed_value in INIT_SEEDS:
            block = f"SS{subset_seed_value}_IS{init_seed_value}"
            hashes = {}
            for budget in SUBSET_LEVELS:
                stream_seed_value = stream_seed(subset_seed_value, init_seed_value)
                setup_seed(init_seed_value)
                model = CBraMod(
                    in_dim=200,
                    out_dim=200,
                    d_model=200,
                    dim_feedforward=800,
                    seq_len=30,
                    n_layer=12,
                    nhead=8,
                )
                state_sha = model_state_hash(model)
                hashes[budget] = state_sha
                initial_rows.append(
                    {
                        "trajectory_id": f"{block}_U{budget}",
                        "block_id": block,
                        "subset_seed": subset_seed_value,
                        "init_seed": init_seed_value,
                        "stream_seed": stream_seed_value,
                        "loader_seed_root": loader_seed_root(stream_seed_value),
                        "mask_seed_root": mask_seed_root(stream_seed_value),
                        "unique_data_h": budget,
                        "subset_window_sha256": selection_hashes[
                            (subset_seed_value, budget)
                        ],
                        "initial_state_sha256": state_sha,
                        "paired_u_arm_state_sha256": "pending_pair_check",
                        "u_arm_initial_state_exact_match": "pending_pair_check",
                        "model_source_sha256": model_source_sha,
                        "n_parameters": sum(p.numel() for p in model.parameters()),
                        "torch_version": torch.__version__,
                        "numpy_version": np.__version__,
                        "shuffle_seed_contract": "sha256(loader_seed_root|U|epoch)",
                        "mask_seed_contract": "sha256(mask_seed_root|global_update)",
                    }
                )
                del model
            if len(set(hashes.values())) != 1:
                raise RuntimeError(f"{block}: U-arm initial states differ")
            pair_hashes[block] = next(iter(hashes.values()))
    for row in initial_rows:
        row["paired_u_arm_state_sha256"] = pair_hashes[row["block_id"]]
        row["u_arm_initial_state_exact_match"] = (
            "true"
            if row["initial_state_sha256"] == pair_hashes[row["block_id"]]
            else "false"
        )
    write_csv(output_dir / "phase_d1_initial_state_pairing.csv", initial_rows)

    schedule_rows = []
    for row in initial_rows:
        budget = int(row["unique_data_h"])
        steps_per_epoch = budget * WINDOWS_PER_HOUR // BATCH_SIZE
        for label, update in (("P_low", P_LOW), ("P_high", P_HIGH)):
            if update % steps_per_epoch:
                raise RuntimeError(f"U{budget} {label}: update is not an epoch boundary")
            schedule_rows.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "block_id": row["block_id"],
                    "subset_seed": row["subset_seed"],
                    "init_seed": row["init_seed"],
                    "stream_seed": row["stream_seed"],
                    "unique_data_h": budget,
                    "snapshot_role": label,
                    "snapshot_update": update,
                    "epoch_at_snapshot": update // steps_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "batch_size": BATCH_SIZE,
                    "gradient_accumulation": 1,
                    "sample_presentations": update * BATCH_SIZE,
                    "presentation_hours_equivalent": f"{update * BATCH_SIZE / WINDOWS_PER_HOUR:.9f}",
                    "lr_after_update": f"{cosine_lr(update):.15g}",
                    "lr_schedule_t_max": P_HIGH,
                    "validation_index": update // VAL_CADENCE,
                    "primary_fixed_update_snapshot": "true",
                    "best_val_used_for_primary": "false",
                    "training_trajectory_continues_after_snapshot": (
                        "true" if update == P_LOW else "false"
                    ),
                }
            )
    write_csv(output_dir / "phase_d1_update_schedule.csv", schedule_rows)

    lr_rows = []
    for update in [0, *range(VAL_CADENCE, P_HIGH + 1, VAL_CADENCE)]:
        value = cosine_lr(update)
        lr_rows.append(
            {
                "completed_updates": update,
                "validation_index": update // VAL_CADENCE,
                "u200_lr": f"{value:.15g}",
                "u1000_lr": f"{value:.15g}",
                "absolute_lr_difference": "0.0",
                "base_lr": BASE_LR,
                "eta_min": MIN_LR,
                "t_max_updates": P_HIGH,
                "warmup_updates": 0,
                "is_p_low_snapshot": "true" if update == P_LOW else "false",
                "is_p_high_snapshot": "true" if update == P_HIGH else "false",
                "common_step_schedule_pass": "true",
            }
        )
    write_csv(output_dir / "phase_d1_lr_schedule_check.csv", lr_rows)

    validation_rows = []
    for index, update in enumerate(range(VAL_CADENCE, P_HIGH + 1, VAL_CADENCE), 1):
        validation_rows.append(
            {
                "validation_index": index,
                "completed_updates": update,
                "u200_epoch": update // (200 * WINDOWS_PER_HOUR // BATCH_SIZE),
                "u1000_epoch": update // (1000 * WINDOWS_PER_HOUR // BATCH_SIZE),
                "is_p_low_snapshot": "true" if update == P_LOW else "false",
                "is_p_high_snapshot": "true" if update == P_HIGH else "false",
                "validation_selection_role": "diagnostic_only",
                "primary_checkpoint_selection_role": "none",
                "cadence_is_update_based": "true",
            }
        )
    write_csv(output_dir / "phase_d1_validation_cadence.csv", validation_rows)

    snapshot_contract = {
        "phase": "D1_unique_data_x_cumulative_exposure",
        "training_trajectory_high_horizon_updates": P_HIGH,
        "primary_snapshot_updates": [P_LOW, P_HIGH],
        "snapshots_per_trajectory": 2,
        "training_trajectories": 8,
        "immutable_snapshots": 16,
        "primary_selection": "fixed_update_only",
        "training_hyperparameters": {
            "optimizer": "AdamW",
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": 1,
            "mask_ratio": MASK_RATIO,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max_updates": P_HIGH,
            "eta_min": MIN_LR,
            "warmup_updates": 0,
        },
        "best_pretrain_val_primary": False,
        "best_pretrain_val_secondary_sensitivity_allowed": True,
        "snapshot_timing": "after_optimizer_and_scheduler_step_before_next_batch",
        "p_low_continuation": "continue_same_in_memory_trajectory",
        "required_payload_fields": [
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "global_update",
            "completed_epochs",
            "subset_seed",
            "init_seed",
            "stream_seed",
            "subset_window_sha256",
            "initial_state_sha256",
            "config_sha256",
            "code_commit",
        ],
        "closure_sequence": [
            "write_unique_temporary_file_with_no_overwrite",
            "fsync_payload",
            "compute_sha256",
            "copy_or_rename_to_content_addressed_no_overwrite_path",
            "set_mode_0444",
            "strict_reload_zero_missing_zero_unexpected",
            "verify_optimizer_scheduler_global_update",
            "run_fixed_unlabeled_pretrain_val_feature_canary",
            "record_feature_hash_and_max_abs_diff_zero",
        ],
        "immutable_path_template": "${D1_RESULTS_ROOT}/snapshots/sha256_<FULL_SHA256>.pth",
        "feature_canary": {
            "source": "fixed_subject_disjoint_pretrain_val_pool",
            "target_labels_used": False,
            "input_shape": [2, 33, 30, 200],
            "determinism_tolerance": 0.0,
        },
        "overwrite_allowed": False,
        "early_stopping_allowed": False,
        "target_labels_used_for_selection": False,
    }
    (output_dir / "phase_d1_snapshot_contract.json").write_text(
        json.dumps(snapshot_contract, indent=2, sort_keys=True) + "\n"
    )

    reference_specs = [
        ("U200", "H200_s0", "train-890125_0.out", P_LOW, "A40_reference"),
        ("U200", "H200_s1", "train-890147_1.out", P_LOW, "A100_reference"),
        ("U1000", "H1000_s0", "train-890147_4.out", P_HIGH, "A100_reference"),
        ("U1000", "H1000_s1", "train-890147_5.out", P_HIGH, "V100_reference"),
    ]
    compute_rows = []
    estimates = defaultdict(list)
    for arm, run_id, log_name, reference_updates, estimate_label in reference_specs:
        summary_path = b1_results / run_id / "run_summary.json"
        log_path = b1_results / "logs" / log_name
        summary = json.loads(summary_path.read_text())
        reference_wall_h = float(summary["training_wall_s"]) / 3600.0
        scale = P_HIGH / reference_updates
        estimate = reference_wall_h * scale
        estimates[arm].append(estimate)
        compute_rows.append(
            {
                "estimate_type": "trajectory_benchmark",
                "unique_data_arm": arm,
                "planned_trajectories": 4,
                "reference_run": run_id,
                "reference_gpu": gpu_name(log_path),
                "reference_updates": reference_updates,
                "reference_wall_hours": f"{reference_wall_h:.6f}",
                "planned_updates": P_HIGH,
                "update_scale_factor": f"{scale:.6f}",
                "estimated_wall_hours_per_trajectory": f"{estimate:.6f}",
                "estimated_gpu_hours_for_arm": f"{estimate * 4:.6f}",
                "estimate_note": "linear_update_scaling; U200 estimate is conservative because validation count stays at 50",
                "source_summary_sha256": sha256_file(summary_path),
                "source_log_sha256": sha256_file(log_path),
            }
        )
    optimistic = min(estimates["U200"]) * 4 + min(estimates["U1000"]) * 4
    conservative = max(estimates["U200"]) * 4 + max(estimates["U1000"]) * 4
    for label, value in (("aggregate_optimistic", optimistic), ("aggregate_conservative", conservative)):
        compute_rows.append(
            {
                "estimate_type": label,
                "unique_data_arm": "all",
                "planned_trajectories": 8,
                "reference_run": "historical_route_b_benchmarks",
                "reference_gpu": "mixed",
                "reference_updates": "NA",
                "reference_wall_hours": "NA",
                "planned_updates": P_HIGH,
                "update_scale_factor": "NA",
                "estimated_wall_hours_per_trajectory": "NA",
                "estimated_gpu_hours_for_arm": f"{value:.6f}",
                "estimate_note": "excludes queue time and downstream audit",
                "source_summary_sha256": "NA",
                "source_log_sha256": "NA",
            }
        )
    write_csv(output_dir / "phase_d1_compute_budget.csv", compute_rows)

    immutable_rows = list(csv.DictReader(immutable_manifest.open()))
    checkpoint_sizes = [
        int(row["immutable_size_bytes"])
        for row in immutable_rows
        if row["role"] == "route_b_checkpoint"
    ]
    snapshot_bytes_estimate = round(sum(checkpoint_sizes) / len(checkpoint_sizes)) * 16

    nested_checks = [
        row for row in overlap_rows if row["comparison"] == "within_subset_seed_u200_in_u1000"
    ]
    go_nogo = {
        "phase": "D1_unique_data_x_cumulative_exposure_protocol_preflight",
        "scientific_name": "Unique-data volume x cumulative training exposure",
        "unique_data_levels_h": [200, 1000],
        "exposure_updates": [P_LOW, P_HIGH],
        "subset_seeds": [0, 1],
        "init_seeds": [0, 1],
        "training_trajectories": 8,
        "immutable_snapshots": 16,
        "u200_nested_in_u1000": all(
            row["exact_nesting_pass"] == "true" for row in nested_checks
        ),
        "same_subset_across_init_seeds": True,
        "same_initial_state_across_u_arms": all(
            row["u_arm_initial_state_exact_match"] == "true" for row in initial_rows
        ),
        "subset_seed_separate_from_init_seed": True,
        "stream_seed_separate_and_preregistered": True,
        "loader_and_mask_seed_fields_separate": all(
            int(row["loader_seed_root"]) != int(row["mask_seed_root"])
            for row in initial_rows
        ),
        "fixed_group_mix_exact": all(
            row["exact_group_quota_pass"] == "true" for row in group_rows
        ),
        "subject_disjoint_pretrain_val_preserved": True,
        "common_step_based_lr_schedule": True,
        "lr_high_horizon_updates": P_HIGH,
        "warmup_updates": 0,
        "fixed_update_primary_snapshots": True,
        "validation_every_updates": VAL_CADENCE,
        "best_val_used_for_primary": False,
        "faced_primary_metric": "cohen_kappa",
        "target_nll_secondary": True,
        "external_validation_datasets": ["SEED-V", "ISRUC_S3"],
        "variance_partition_reintroduced": False,
        "historical_checkpoints_used_as_factorial_cells": False,
        "estimated_training_gpu_hours_range": [
            round(optimistic, 3),
            round(conservative, 3),
        ],
        "estimated_snapshot_storage_bytes": snapshot_bytes_estimate,
        "compute_budget_pm_approved": False,
        "protocol_preflight_pass": True,
        "independent_verifier_pass": None,
        "launch_phase_d1": False,
        "launch_status": "HELD_FOR_PM_TRAINING_AUTHORIZATION",
        "training_submitted": False,
        "downstream_submitted": False,
        "target_labels_used": False,
    }
    (output_dir / "phase_d1_go_nogo.json").write_text(
        json.dumps(go_nogo, indent=2, sort_keys=True) + "\n"
    )

    provenance = {
        "repo_input_head": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "d0_go_nogo_sha256": sha256_file(
            repo_root
            / "results/s2p_route_b_phase_d0_schedule_audit/phase_d0_go_nogo.json"
        ),
        "route_b_group_mix_sha256": sha256_file(
            contract_dir / "route_b_group_mix_by_budget.csv"
        ),
        "route_b_channel_order_sha256": sha256_file(
            contract_dir / "route_b_canonical_channel_order.json"
        ),
        "tueg_metadata_logical_path": "${TUEG_PROCESSED_ROOT}/metadata.parquet",
        "tueg_metadata_sha256": sha256_file(
            Path(route_loader.TUEG) / "metadata.parquet"
        ),
        "fixed_val_subjects_sha256": val_subject_hash,
        "cbramod_root_logical_path": "${CBRAMOD_ROOT}",
        "cbramod_source_files": cbramod_sources,
        "model_source_sha256": model_source_sha,
        "b1_benchmark_root_logical_path": "${B1_LAUNCH_ROOT}/results/s2p_route_b_33ch_b1",
        "absolute_paths_persisted": False,
        "training_invoked": False,
        "downstream_invoked": False,
    }
    (output_dir / "phase_d1_input_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(go_nogo, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
