#!/bin/bash
# Project A Step 13 — ONE-cell GPU smoke: run the instrumented runner on one BNCI2014_001 cell and
# verify raw_results.json carries a non-empty label-free r1_diagnostics block AND a per-trial oracle
# prediction payload. Guards the full diagnostics array. Fast (epochs 2). NO --time.
#SBATCH --job-name=projA_step13_smoke
#SBATCH --partition=A100,H100,V100,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step13_smoke_%j.out
#SBATCH --error=logs/project_A_step13_smoke_%j.err

set -euo pipefail
mkdir -p logs
OUT=notes/project_A_observability/results/step13_smoke

conda run -n icml python -m h2cmi.run_real_audited_grid \
  --dataset BNCI2014_001 --subjects all --target-subjects all --seeds 0 \
  --epochs 2 --fast --align-factor subject --device cuda \
  --root-outdir "$OUT" --num-shards 9 --shard-index 0

OUT="$OUT" conda run -n icml python -c '
import json, glob, os
root = os.environ["OUT"]
oks = [d for d in glob.glob(os.path.join(root, "dataset=*"))
       if json.load(open(os.path.join(d, "run_manifest.json")))["status"] == "ok"]
assert oks, "no ok cell"
raw = json.load(open(os.path.join(oks[0], "raw_results.json")))
r1 = raw.get("r1_diagnostics") or {}
pt = raw.get("per_trial_oracle_predictions") or {}
present = [k for k, v in r1.items() if v is not None]
print("r1_diagnostics keys present:", len(present), "of", len(r1))
print("  sample:", {k: r1[k] for k in list(present)[:4]})
print("per_trial has y_true/identity_pred/adapt_pred:",
      all(k in pt for k in ("y_true", "identity_pred", "adapt_pred")), "n_trials:", len(pt.get("y_true", [])))
assert len(present) >= 8, "r1_diagnostics too sparse"
assert all(k in pt for k in ("y_true", "identity_pred", "adapt_pred")) and pt["y_true"], "per_trial missing"
# offline_tta must NOT still carry the popped blocks
assert "r1_prediction_diagnostics" not in raw.get("offline_tta", {})
assert "per_trial_oracle_predictions" not in raw.get("offline_tta", {})
print("STEP-13 GPU SMOKE PASSED (instrumented r1_diagnostics + per_trial oracle payload present, cuda)")
'
