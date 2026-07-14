#!/usr/bin/env python
"""Independent fail-closed verifier for the Phase-D1 protocol package."""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


WINDOWS_PER_HOUR = 120
BATCH_SIZE = 64
P_LOW = 18_750
P_HIGH = 93_750
VAL_CADENCE = 1_875
BASE_LR = 5e-4
MIN_LR = 1e-5


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
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


def identities(rows):
    values = set()
    duplicate_count = 0
    for row in rows:
        for index in range(int(row["window_start"]), int(row["window_stop_exclusive"])):
            value = (
                int(row["subject"]),
                int(row["recording_id"]),
                row["filepath_relative"],
                index,
            )
            if value in values:
                duplicate_count += 1
            values.add(value)
    return values, duplicate_count


def hash_identities(values):
    h = hashlib.sha256()
    for subject, recording, filepath, index in sorted(values):
        h.update(f"{subject}|{recording}|{filepath}|{index}\n".encode())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cbramod-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("results/s2p_route_b_phase_d1_protocol"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    protocol_dir = (repo_root / args.protocol_dir).resolve()
    cbramod_root = args.cbramod_root.resolve()
    sys.path.insert(0, str(cbramod_root))
    from models.cbramod import CBraMod

    manifest = read_csv(protocol_dir / "phase_d1_nested_subset_manifest.csv")
    overlaps = read_csv(protocol_dir / "phase_d1_subset_overlap_verification.csv")
    exposures = read_csv(protocol_dir / "phase_d1_subject_exposure.csv")
    groups = read_csv(protocol_dir / "phase_d1_group_mixture.csv")
    initial = read_csv(protocol_dir / "phase_d1_initial_state_pairing.csv")
    schedule = read_csv(protocol_dir / "phase_d1_update_schedule.csv")
    lr_rows = read_csv(protocol_dir / "phase_d1_lr_schedule_check.csv")
    validation = read_csv(protocol_dir / "phase_d1_validation_cadence.csv")
    compute = read_csv(protocol_dir / "phase_d1_compute_budget.csv")
    snapshot = json.loads((protocol_dir / "phase_d1_snapshot_contract.json").read_text())
    go_nogo = json.loads((protocol_dir / "phase_d1_go_nogo.json").read_text())

    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    by_cell = defaultdict(list)
    for row in manifest:
        by_cell[(int(row["subset_seed"]), int(row["unique_data_h"]))].append(row)
    sets = {}
    duplicate_failures = []
    count_failures = []
    for key, rows in sorted(by_cell.items()):
        values, duplicates = identities(rows)
        sets[key] = values
        expected = key[1] * WINDOWS_PER_HOUR
        if duplicates:
            duplicate_failures.append(f"{key}:{duplicates}")
        if len(values) != expected or sum(int(row["take_windows"]) for row in rows) != expected:
            count_failures.append(f"{key}:{len(values)}/{expected}")
    check("four_subset_manifests", set(by_cell) == {(0, 200), (0, 1000), (1, 200), (1, 1000)}, str(sorted(by_cell)))
    check("no_duplicate_windows", not duplicate_failures, ";".join(duplicate_failures))
    check("exact_window_counts", not count_failures, ";".join(count_failures))

    nesting_failures = []
    for subset_seed in (0, 1):
        low = sets[(subset_seed, 200)]
        high = sets[(subset_seed, 1000)]
        if not low <= high or len(high - low) != 96_000:
            nesting_failures.append(f"subset_seed={subset_seed}")
        reported = next(
            row
            for row in overlaps
            if row["comparison"] == "within_subset_seed_u200_in_u1000"
            and int(row["subset_seed_left"]) == subset_seed
        )
        if (
            reported["left_window_sha256"] != hash_identities(low)
            or reported["right_window_sha256"] != hash_identities(high)
            or reported["extension_window_sha256"] != hash_identities(high - low)
        ):
            nesting_failures.append(f"subset_seed={subset_seed}:hash")
    check("exact_nested_u200_in_u1000", not nesting_failures, ";".join(nesting_failures))

    group_failures = []
    group_counts = Counter()
    for row in manifest:
        group_counts[(int(row["subset_seed"]), int(row["unique_data_h"]), row["group_id"])] += int(row["take_windows"])
    for row in groups:
        key = (int(row["subset_seed"]), int(row["unique_data_h"]), row["group_id"])
        if group_counts[key] != int(row["target_windows"]) or row["exact_group_quota_pass"] != "true":
            group_failures.append(str(key))
    check("fixed_group_mix_exact", not group_failures, ";".join(group_failures))

    exposure_counts = Counter()
    for row in manifest:
        exposure_counts[(int(row["subset_seed"]), int(row["unique_data_h"]), int(row["subject"]))] += int(row["take_windows"])
    exposure_failures = []
    for row in exposures:
        key = (int(row["subset_seed"]), int(row["unique_data_h"]), int(row["subject"]))
        if exposure_counts[key] != int(row["windows"]):
            exposure_failures.append(str(key))
    check("subject_exposure_replay", not exposure_failures, ";".join(exposure_failures))

    initial_failures = []
    initial_by_block = defaultdict(list)
    for row in initial:
        initial_by_block[row["block_id"]].append(row)
    recomputed_hash = {}
    for init_seed in (0, 1):
        setup_seed(init_seed)
        model = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        recomputed_hash[init_seed] = model_state_hash(model)
        del model
    for block, rows in initial_by_block.items():
        if len(rows) != 2 or {int(row["unique_data_h"]) for row in rows} != {200, 1000}:
            initial_failures.append(f"{block}:arms")
            continue
        hashes = {row["initial_state_sha256"] for row in rows}
        if len(hashes) != 1:
            initial_failures.append(f"{block}:pair")
        if len({row["loader_seed_root"] for row in rows}) != 1:
            initial_failures.append(f"{block}:loader_seed_pair")
        if len({row["mask_seed_root"] for row in rows}) != 1:
            initial_failures.append(f"{block}:mask_seed_pair")
        for row in rows:
            if row["loader_seed_root"] == row["mask_seed_root"]:
                initial_failures.append(f"{row['trajectory_id']}:seed_domain_collision")
            if row["initial_state_sha256"] != recomputed_hash[int(row["init_seed"])]:
                initial_failures.append(f"{row['trajectory_id']}:recompute")
            expected_subset_hash = hash_identities(
                sets[(int(row["subset_seed"]), int(row["unique_data_h"]))]
            )
            if row["subset_window_sha256"] != expected_subset_hash:
                initial_failures.append(f"{row['trajectory_id']}:subset")
    check("fully_crossed_initial_pairing", len(initial_by_block) == 4 and not initial_failures, ";".join(initial_failures))

    schedule_failures = []
    if len(schedule) != 16:
        schedule_failures.append(f"rows={len(schedule)}")
    for row in schedule:
        budget = int(row["unique_data_h"])
        update = int(row["snapshot_update"])
        steps_per_epoch = budget * WINDOWS_PER_HOUR // BATCH_SIZE
        expected_epoch = update // steps_per_epoch
        if update not in (P_LOW, P_HIGH) or update % steps_per_epoch:
            schedule_failures.append(f"{row['trajectory_id']}:{update}:boundary")
        if int(row["epoch_at_snapshot"]) != expected_epoch:
            schedule_failures.append(f"{row['trajectory_id']}:{update}:epoch")
        if int(row["sample_presentations"]) != update * BATCH_SIZE:
            schedule_failures.append(f"{row['trajectory_id']}:{update}:presentations")
        if int(row["lr_schedule_t_max"]) != P_HIGH:
            schedule_failures.append(f"{row['trajectory_id']}:{update}:tmax")
        expected_lr = MIN_LR + (BASE_LR - MIN_LR) * (
            1 + math.cos(math.pi * update / P_HIGH)
        ) / 2
        if abs(float(row["lr_after_update"]) - expected_lr) > 1e-15:
            schedule_failures.append(f"{row['trajectory_id']}:{update}:lr")
        if row["best_val_used_for_primary"] != "false":
            schedule_failures.append(f"{row['trajectory_id']}:{update}:selection")
    check("sixteen_fixed_update_snapshots", not schedule_failures, ";".join(schedule_failures))

    lr_failures = []
    if len(lr_rows) != 51:
        lr_failures.append(f"rows={len(lr_rows)}")
    for row in lr_rows:
        update = int(row["completed_updates"])
        expected = MIN_LR + (BASE_LR - MIN_LR) * (
            1 + math.cos(math.pi * update / P_HIGH)
        ) / 2
        if abs(float(row["u200_lr"]) - expected) > 1e-15 or float(row["absolute_lr_difference"]) != 0:
            lr_failures.append(str(update))
    check("common_high_horizon_lr", not lr_failures, ";".join(lr_failures))

    validation_failures = []
    if len(validation) != 50:
        validation_failures.append(f"rows={len(validation)}")
    for index, row in enumerate(validation, 1):
        if int(row["completed_updates"]) != index * VAL_CADENCE:
            validation_failures.append(f"index={index}")
        if row["validation_selection_role"] != "diagnostic_only":
            validation_failures.append(f"index={index}:selection")
    check("fixed_update_validation_cadence", not validation_failures, ";".join(validation_failures))

    check(
        "snapshot_fail_closed_contract",
        snapshot["primary_snapshot_updates"] == [P_LOW, P_HIGH]
        and snapshot["primary_selection"] == "fixed_update_only"
        and not snapshot["best_pretrain_val_primary"]
        and not snapshot["overwrite_allowed"]
        and not snapshot["early_stopping_allowed"]
        and not snapshot["target_labels_used_for_selection"],
        "fixed updates, no overwrite, no early stop, no target labels",
    )
    aggregate_estimates = {
        row["estimate_type"]: float(row["estimated_gpu_hours_for_arm"])
        for row in compute
        if row["estimate_type"].startswith("aggregate_")
    }
    check(
        "compute_budget_present",
        set(aggregate_estimates) == {"aggregate_optimistic", "aggregate_conservative"}
        and aggregate_estimates["aggregate_optimistic"]
        < aggregate_estimates["aggregate_conservative"],
        str(aggregate_estimates),
    )
    check(
        "launch_remains_held",
        go_nogo["protocol_preflight_pass"]
        and not go_nogo["compute_budget_pm_approved"]
        and not go_nogo["launch_phase_d1"]
        and not go_nogo["training_submitted"]
        and not go_nogo["downstream_submitted"],
        go_nogo["launch_status"],
    )

    failures = [row["check"] for row in checks if not row["pass"]]
    result = {
        "phase": "D1_protocol_preflight_independent_verification",
        "checks": checks,
        "failures": failures,
        "pass": not failures,
        "training_launched": False,
        "downstream_launched": False,
        "phase_d1_training_authorized": False,
    }
    (protocol_dir / "phase_d1_protocol_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    redteam = {
        "phase": "D1_unique_data_x_cumulative_exposure_design_redteam",
        "nested_windows": "PASS" if not nesting_failures else "FAIL",
        "fixed_group_mix": "PASS" if not group_failures else "FAIL",
        "fully_crossed_subset_init": "PASS" if not initial_failures else "FAIL",
        "common_high_horizon_lr": "PASS" if not lr_failures else "FAIL",
        "fixed_update_primary_snapshots": "PASS" if not schedule_failures else "FAIL",
        "validation_cadence": "PASS" if not validation_failures else "FAIL",
        "historical_checkpoint_reuse": "FORBIDDEN_IN_PRIMARY_FACTORIAL",
        "variance_partition": "PERMANENTLY_EXCLUDED",
        "remaining_scientific_limits": [
            "U changes windows, subjects, recordings, and population breadth together",
            "P changes updates, presentations, repeated exposure, and mask draws together",
            "two subset seeds and two initialization seeds provide limited training-level replication",
            "P_high continuation shares trajectory history with P_low by design",
        ],
        "blockers": failures,
        "verdict": "PASS_PROTOCOL_ONLY_TRAINING_HELD" if not failures else "FAIL_CLOSED",
        "launch_phase_d1": False,
    }
    (protocol_dir / "phase_d1_redteam_verdict.json").write_text(
        json.dumps(redteam, indent=2, sort_keys=True) + "\n"
    )

    go_nogo["independent_verifier_pass"] = not failures
    (protocol_dir / "phase_d1_go_nogo.json").write_text(
        json.dumps(go_nogo, indent=2, sort_keys=True) + "\n"
    )

    artifact_rows = []
    for path in sorted(protocol_dir.iterdir()):
        if not path.is_file() or path.name == "phase_d1_artifact_manifest.csv":
            continue
        artifact_rows.append(
            {
                "artifact": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (protocol_dir / "phase_d1_artifact_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact", "size_bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(artifact_rows)

    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
