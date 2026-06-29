"""Confirmatory one-fold CLI (V100). Materializes a full-budget pilot manifest for one held-out target,
runs the seeds through the closed loop, and prints ONE canonical-JSON report to stdout (MNE / MOABB /
training chatter -> stderr). Exit 0 only when every seed deep-verifies and no target id was fit.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 ... python -m oaci.confirmatory.demo \
        --protocol oaci/protocol/confirmatory_v2.yaml --datalake-root "$OACI_DATALAKE_ROOT" \
        --output-root "$OUT/artifacts" --manifest-out "$OUT/pilot_manifest.yaml" --repo-root "$REPO" \
        --target-subject 1 --model-seeds 0,1,2

This is PIPELINE VALIDATION, not confirmatory efficacy evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import sys

from ..artifacts.canonical_json import canonical_json_bytes
from ..runtime.cuda import configure_cuda_determinism
from .onefold import DATASET, run_confirmatory_onefold
from .report import build_onefold_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.demo")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--datalake-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--target-subject", type=int, default=1)
    ap.add_argument("--model-seeds", default="0,1,2")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.model_seeds.split(","))
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            device, _runtime = configure_cuda_determinism()
            result = run_confirmatory_onefold(
                args.protocol, datalake_root=args.datalake_root, repo_root=args.repo_root,
                output_root=args.output_root, manifest_out=args.manifest_out, dataset_name=args.dataset,
                target_subject=args.target_subject, model_seeds=seeds, device=device)
            report = build_onefold_report(result)
        real_stdout.buffer.write(canonical_json_bytes(report))
        return 0 if report["all_seeds_deep_verified"] and report["all_target_fit_ids_empty"] else 1
    except Exception as e:  # noqa: BLE001
        print(f"confirmatory one-fold failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
