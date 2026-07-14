#!/usr/bin/env python
"""Independent verifier for the Route-B Phase-D0 schedule audit."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch


WINDOWS_PER_HOUR = 120


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_events(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def window_identity(rows):
    identities = set()
    payload = []
    for row in rows:
        subject = int(row["subject"])
        recording_id = int(row["recording_id"])
        filepath = row["filepath"]
        take_windows = int(row["take_windows"])
        payload.append(
            (subject, recording_id, filepath, take_windows, row["group_id"])
        )
        identities.update(
            (subject, recording_id, filepath, index) for index in range(take_windows)
        )
    return identities, sha256_json(sorted(payload))


def validation_sha(rows):
    payload = sorted(
        (int(row["subject"]), int(row["recording_id"]), int(row["take_windows"]))
        for row in rows
    )
    return sha256_json(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("results/s2p_route_b_phase_d0_schedule_audit"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--b1-launch-root", type=Path, required=True)
    parser.add_argument(
        "--immutable-manifest",
        type=Path,
        default=Path(
            "results/s2p_route_b_phase_b_checkpoint_closure/"
            "phase_b_checkpoint_immutable_manifest.csv"
        ),
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=Path("results/s2p_route_b_33ch_contract"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    audit_dir = (repo_root / args.audit_dir).resolve()
    b1_results = (
        args.b1_launch_root.resolve() / "results" / "s2p_route_b_33ch_b1"
    )
    immutable_manifest = (repo_root / args.immutable_manifest).resolve()
    contract_dir = (repo_root / args.contract_dir).resolve()
    sys.path.insert(0, str(repo_root / "s2p" / "scripts"))
    import route_b_33ch_loader as route_loader

    schedule = read_csv(audit_dir / "phase_d0_run_schedule.csv")
    checkpoints = read_csv(audit_dir / "phase_d0_checkpoint_selection.csv")
    subsets = read_csv(audit_dir / "phase_d0_subset_identity.csv")
    overlaps = read_csv(audit_dir / "phase_d0_subset_overlap.csv")
    mapping = read_csv(audit_dir / "phase_d0_d1_cell_mapping.csv")
    reuse = read_csv(audit_dir / "phase_d0_reuse_eligibility.csv")
    verdict = json.loads((audit_dir / "phase_d0_go_nogo.json").read_text())

    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    check("run_count", len(schedule) == 8, f"observed={len(schedule)} expected=8")
    check(
        "checkpoint_count",
        len(checkpoints) == 8,
        f"observed={len(checkpoints)} expected=8",
    )
    check("subset_count", len(subsets) == 8, f"observed={len(subsets)} expected=8")
    check("d1_cell_count", len(mapping) == 8, f"observed={len(mapping)} expected=8")

    formula_failures = []
    immutable_rows = {
        row["tag"]: row
        for row in read_csv(immutable_manifest)
        if row["role"] == "route_b_checkpoint"
    }
    for row in schedule:
        tag = row["tag"]
        summary_path = b1_results / tag / "run_summary.json"
        log_path = b1_results / tag / "train_log.jsonl"
        summary = json.loads(summary_path.read_text())
        events = read_events(log_path)
        model_events = [event for event in events if event.get("event") == "model"]
        epoch_events = [event for event in events if event.get("event") == "epoch"]
        manifest_row = immutable_rows[tag]
        checkpoint_path = Path(manifest_row["immutable_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        windows = int(row["unique_windows"])
        batch = int(row["batch_size"])
        epochs = int(row["epochs_completed"])
        steps_per_epoch = windows // batch
        updates = steps_per_epoch * epochs
        presentations = updates * batch
        selected_epoch = int(row["selected_epoch"])
        expected = {
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_updates": updates,
            "total_sample_presentations": presentations,
            "nominal_sample_presentations": windows * epochs,
            "selected_optimizer_updates": selected_epoch * steps_per_epoch,
            "selected_sample_presentations": selected_epoch * steps_per_epoch * batch,
        }
        for field, value in expected.items():
            if int(row[field]) != value:
                formula_failures.append(f"{row['tag']}:{field}")
        expected_hours = presentations / WINDOWS_PER_HOUR
        if abs(float(row["presentation_hours_equivalent"]) - expected_hours) > 1e-6:
            formula_failures.append(f"{row['tag']}:presentation_hours_equivalent")
        if int(row["lr_t_max_updates"]) != updates:
            formula_failures.append(f"{row['tag']}:lr_t_max_updates")
        if len(model_events) != 1 or int(model_events[0]["steps_per_epoch"]) != steps_per_epoch:
            formula_failures.append(f"{tag}:raw_model_event")
        if len(epoch_events) != epochs or int(summary["epochs"]) != epochs:
            formula_failures.append(f"{tag}:raw_epoch_count")
        if sha256_file(summary_path) != row["run_summary_sha256"]:
            formula_failures.append(f"{tag}:summary_sha")
        if sha256_file(log_path) != row["train_log_sha256"]:
            formula_failures.append(f"{tag}:log_sha")
        if sha256_file(checkpoint_path) != row["immutable_checkpoint_sha256"]:
            formula_failures.append(f"{tag}:checkpoint_sha")
        if int(checkpoint["epoch"]) != selected_epoch:
            formula_failures.append(f"{tag}:checkpoint_epoch")
        scheduler = checkpoint["scheduler_state"]
        if int(scheduler["T_max"]) != updates:
            formula_failures.append(f"{tag}:checkpoint_tmax")
        if int(scheduler["last_epoch"]) != selected_epoch * steps_per_epoch:
            formula_failures.append(f"{tag}:checkpoint_scheduler_step")
    check(
        "schedule_formula_replay",
        not formula_failures,
        "failures=" + ";".join(formula_failures),
    )

    tags_schedule = {row["tag"] for row in schedule}
    tags_checkpoints = {row["tag"] for row in checkpoints}
    tags_subsets = {row["tag"] for row in subsets}
    check(
        "tag_join",
        tags_schedule == tags_checkpoints == tags_subsets,
        f"schedule={sorted(tags_schedule)}",
    )

    subset_by_tag = {row["tag"]: row for row in subsets}
    rebuilt_sets = {}
    raw_subset_failures = []
    for row in schedule:
        tag = row["tag"]
        budget_h = int(row["budget_h"])
        seed = int(row["seed"])
        cell = route_loader.build_route_b_cell(
            budget_h, seed, contract_dir=str(contract_dir)
        )
        identities, selection_sha = window_identity(cell["train"])
        rebuilt_sets[(budget_h, seed)] = identities
        recorded = subset_by_tag[tag]
        if selection_sha != recorded["window_selection_sha256"]:
            raw_subset_failures.append(f"{tag}:selection_sha")
        if validation_sha(cell["val"]) != recorded["validation_selection_sha256"]:
            raw_subset_failures.append(f"{tag}:validation_sha")
        if cell["manifest"]["selected_subjects_sha"] != recorded["selected_subjects_sha_recorded"]:
            raw_subset_failures.append(f"{tag}:subject_sha")
        checkpoint = torch.load(Path(immutable_rows[tag]["immutable_path"]), map_location="cpu")
        if checkpoint["route_b_manifest"]["selected_subjects_sha"] != cell["manifest"]["selected_subjects_sha"]:
            raw_subset_failures.append(f"{tag}:checkpoint_subset")
    check(
        "raw_subset_rebuild",
        not raw_subset_failures,
        "failures=" + ";".join(raw_subset_failures),
    )

    h200_h1000 = [
        row
        for row in overlaps
        if row["comparison_type"] == "same_seed_budget_pair"
        and row["left_tag"].startswith("H200_")
        and row["right_tag"].startswith("H1000_")
    ]
    check(
        "h200_h1000_comparisons_present",
        len(h200_h1000) == 2,
        f"observed={len(h200_h1000)} expected=2",
    )
    check(
        "h200_not_nested",
        len(h200_h1000) == 2
        and all(row["left_nested_in_right"] == "false" for row in h200_h1000),
        ";".join(
            f"{row['left_tag']}->{row['right_tag']}={row['left_coverage_fraction']}"
            for row in h200_h1000
        ),
    )
    overlap_replay_failures = []
    for row in overlaps:
        left_budget = int(row["left_tag"].split("_")[0][1:])
        left_seed = int(row["left_tag"].split("_s")[1])
        right_budget = int(row["right_tag"].split("_")[0][1:])
        right_seed = int(row["right_tag"].split("_s")[1])
        left = rebuilt_sets[(left_budget, left_seed)]
        right = rebuilt_sets[(right_budget, right_seed)]
        intersection = len(left & right)
        if intersection != int(row["intersection_windows"]):
            overlap_replay_failures.append(
                f"{row['left_tag']}->{row['right_tag']}:intersection"
            )
        if (left <= right) != (row["left_nested_in_right"] == "true"):
            overlap_replay_failures.append(
                f"{row['left_tag']}->{row['right_tag']}:nested"
            )
    check(
        "raw_overlap_replay",
        not overlap_replay_failures,
        "failures=" + ";".join(overlap_replay_failures),
    )
    cross_seed = [
        row for row in overlaps if row["comparison_type"] == "same_budget_seed_pair"
    ]
    check(
        "subsets_differ_across_seeds",
        len(cross_seed) == 4 and all(row["identical"] == "false" for row in cross_seed),
        f"comparisons={len(cross_seed)}",
    )
    check(
        "validation_pool_fixed",
        len({row["validation_selection_sha256"] for row in subsets}) == 1,
        "distinct_validation_hashes="
        + str(len({row["validation_selection_sha256"] for row in subsets})),
    )

    expected_cells = {
        "U200_P_low": (200, 50, 18750, 1200000),
        "U1000_P_low": (1000, 10, 18750, 1200000),
        "U200_P_high": (200, 250, 93750, 6000000),
        "U1000_P_high": (1000, 50, 93750, 6000000),
    }
    mapping_failures = []
    grouped = {}
    for row in mapping:
        grouped.setdefault(row["cell_id"], []).append(row)
        expected = expected_cells.get(row["cell_id"])
        if expected is None:
            mapping_failures.append(f"unknown:{row['cell_id']}")
            continue
        actual = (
            int(row["unique_data_h"]),
            int(row["epochs"]),
            int(row["total_optimizer_updates"]),
            int(row["total_sample_presentations"]),
        )
        if actual != expected:
            mapping_failures.append(f"{row['cell_id']}_s{row['init_seed']}:{actual}")
        if int(row["lr_t_max_updates"]) != expected[2]:
            mapping_failures.append(f"{row['cell_id']}:lr_t_max")
        if int(row["validation_cadence_updates"]) != 1875:
            mapping_failures.append(f"{row['cell_id']}:validation_cadence")
    if any(len(rows) != 2 or {row["init_seed"] for row in rows} != {"0", "1"} for rows in grouped.values()):
        mapping_failures.append("seed_pairing")
    check("d1_factorial_replay", not mapping_failures, "failures=" + ";".join(mapping_failures))

    check(
        "reuse_fail_closed",
        len(reuse) == 8 and all(row["reuse_recommended"] == "false" for row in reuse),
        f"rows={len(reuse)}",
    )
    check(
        "verdict_no_launch",
        verdict["phase_d0_pass"]
        and not verdict["phase_d1_training_authorized"]
        and not verdict["training_launched_by_phase_d0"]
        and not verdict["fine_tuning_launched_by_phase_d0"]
        and not verdict["codebrain_forensic_launched_by_phase_d0"],
        "D0 pass with all compute launch flags false",
    )
    check(
        "verdict_matches_subsets",
        not verdict["existing_h200_nested_in_h1000"]
        and not verdict["same_data_subset_across_training_seeds"]
        and verdict["fixed_pretrain_val_pool_identical"],
        "subset and validation dispositions agree",
    )

    failures = [row["check"] for row in checks if not row["pass"]]
    result = {
        "phase": "D0_route_b_schedule_and_presentations_audit",
        "verifier": "independent_formula_and_contract_replay",
        "checks": checks,
        "failures": failures,
        "pass": not failures,
        "phase_d1_training_authorized": False,
        "training_launched": False,
    }
    verification_path = audit_dir / "phase_d0_independent_verification.json"
    verification_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    manifest_rows = []
    for path in sorted(audit_dir.iterdir()):
        if not path.is_file() or path.name == "phase_d0_artifact_manifest.csv":
            continue
        manifest_rows.append(
            {
                "artifact": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (audit_dir / "phase_d0_artifact_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact", "size_bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
