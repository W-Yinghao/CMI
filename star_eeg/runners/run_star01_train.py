#!/usr/bin/env python
"""Approval-locked persistent launcher for one STAR_01A array cell."""

import argparse
import json
from pathlib import Path

from star_eeg.training.real_star_runner import RealStarConfig, run_real_star


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=[
        "H200_SSL_CONT", "H200_STAR_TRUE", "H200_STAR_SHUFFLED"
    ])
    parser.add_argument("--model-seed", type=int, required=True, choices=[0, 1])
    parser.add_argument("--approval-manifest", required=True)
    parser.add_argument("--immutable-manifest", required=True)
    parser.add_argument("--immutable-go-nogo", required=True)
    parser.add_argument("--anchor-manifest", required=True)
    parser.add_argument("--shuffled-manifest", required=True)
    parser.add_argument("--runtime-output-dir", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--faced-lmdb", default="/projects/EEG-foundation-model/FACED_data/processed")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    result = run_real_star(
        config=RealStarConfig(
            variant=args.variant,
            model_seed=args.model_seed,
            optimizer_steps=3750,
        ),
        repo_root=repo_root,
        immutable_manifest_path=Path(args.immutable_manifest),
        immutable_go_nogo_path=Path(args.immutable_go_nogo),
        anchor_manifest_path=Path(args.anchor_manifest),
        shuffled_manifest_path=Path(args.shuffled_manifest),
        faced_lmdb_path=Path(args.faced_lmdb),
        contract_dir=repo_root / "results/s2p_route_b_33ch_contract",
        runtime_output_dir=Path(args.runtime_output_dir),
        device_name=args.device,
        attempt_id=args.attempt_id,
        launch_approval_path=Path(args.approval_manifest),
    )
    completion = result["completion"]
    if completion.get("status") != "COMPLETE" or not all(
        completion.get("checks", {}).values()
    ):
        raise RuntimeError("formal STAR cell returned without a valid completion gate")
    print(json.dumps({
        "cell": result["cell"],
        "attempt_id": result["attempt_id"],
        "optimizer_steps": result["optimizer_steps"],
        "telemetry_rows": result["telemetry_rows"],
        "telemetry_sha256": result["telemetry_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "completion_hash": completion["completion_hash"],
        "target_metrics_computed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
