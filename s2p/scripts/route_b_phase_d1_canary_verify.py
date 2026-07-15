#!/usr/bin/env python
"""Verify the paired U200/U1000 Phase-D1 implementation canary."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    arms = []
    rows = []
    for unique_h in (200, 1000):
        root = args.canary_root / f"SS0_IS0_U{unique_h}"
        contract = json.loads((root / "run_contract.json").read_text())
        complete = json.loads((root / "run_complete.json").read_text())
        closures = {
            role: json.loads((root / f"{role}_snapshot_closure.json").read_text())
            for role in ("P_low", "P_high")
        }
        if contract["canary"] is not True or complete["status"] != "PASS_CANARY":
            raise RuntimeError(f"U{unique_h} canary did not complete")
        for role, closure in closures.items():
            payload = Path(closure["immutable_path"])
            checks = {
                "path_exists": payload.is_file(),
                "hash_pass": sha256_file(payload) == closure["immutable_sha256"],
                "read_only": (payload.stat().st_mode & 0o222) == 0,
                "strict_reload_pass": closure["strict_reload_pass"] is True,
                "optimizer_state_exact_pass": closure["optimizer_state_exact_pass"] is True,
                "scheduler_state_exact_pass": closure["scheduler_state_exact_pass"] is True,
                "feature_exact_pass": (
                    closure["feature_reload_max_abs_diff"] == 0.0
                    and closure["feature_repeat_max_abs_diff"] == 0.0
                ),
            }
            rows.append(
                {
                    "unique_data_h": unique_h,
                    "snapshot_role": role,
                    "global_update": closure["global_update"],
                    "lr_after_update": closure["lr_after_update"],
                    "immutable_sha256": closure["immutable_sha256"],
                    **checks,
                    "all_pass": all(checks.values()),
                }
            )
        arms.append((contract, complete, closures))

    same_initial = arms[0][0]["initial_state_sha256"] == arms[1][0]["initial_state_sha256"]
    same_stream = arms[0][0]["stream_seed"] == arms[1][0]["stream_seed"]
    same_lr = all(
        arms[0][2][role]["lr_after_update"] == arms[1][2][role]["lr_after_update"]
        for role in ("P_low", "P_high")
    )
    updates = all(
        arm[2]["P_low"]["global_update"] == 2 and arm[2]["P_high"]["global_update"] == 4
        for arm in arms
    )
    all_pass = all(row["all_pass"] for row in rows) and same_initial and same_stream and same_lr and updates
    verdict = {
        "phase": "D1_implementation_canary",
        "status": "PASS" if all_pass else "FAIL",
        "u_arms": [200, 1000],
        "same_initial_state_across_u_arms": same_initial,
        "same_stream_contract_across_u_arms": same_stream,
        "same_lr_at_matched_updates": same_lr,
        "fixed_updates_2_and_4": updates,
        "snapshot_closures_all_pass": all(row["all_pass"] for row in rows),
        "trainer_called": True,
        "downstream_called": False,
        "target_labels_used": False,
        "launch_eight_trajectories_recommended": all_pass,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "phase_d1_canary_snapshot_verification.csv", rows)
    (args.out_dir / "phase_d1_canary_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
