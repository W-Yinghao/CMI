#!/bin/bash
# Project A Step 10 — ONE-cell GPU smoke on the binary dataset BNCI2014_004: confirms the data
# cache is present, the CUDA path works for a 2-class task, and --subjects all/--target-subjects
# all resolve + write grid_manifest (n_classes=2). Runs one shard fast (epochs 2) and FAILS LOUDLY
# unless a cell is ok and n_classes==2. Guards the full multi-dataset array. NO --time.
#SBATCH --job-name=projA_step10_smoke
#SBATCH --partition=A100,H100,V100,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_step10_smoke_%j.out
#SBATCH --error=logs/project_A_step10_smoke_%j.err

set -euo pipefail
mkdir -p logs
OUT=notes/project_A_observability/results/step10_smoke_bnci2014_004

conda run -n icml python -m h2cmi.run_real_audited_grid \
  --dataset BNCI2014_004 --subjects all --target-subjects all --seeds 0 \
  --epochs 2 --fast --align-factor subject --device cuda \
  --root-outdir "$OUT" --num-shards 9 --shard-index 0

# self-check via -c (NOT a heredoc: `conda run` does not forward stdin to `python -` reliably)
OUT="$OUT" conda run -n icml python -c '
import json, glob, os
root = os.environ["OUT"]
gm = json.load(open(os.path.join(root, "grid_manifest.json")))
oks = [d for d in glob.glob(os.path.join(root, "dataset=*"))
       if json.load(open(os.path.join(d, "run_manifest.json")))["status"] == "ok"]
print("grid_manifest n_classes=%s target_subjects=%s expected_cells=%d ok_cells=%d"
      % (gm["n_classes"], gm["target_subjects"], len(gm["expected_cells"]), len(oks)))
assert gm["n_classes"] == 2, "expected binary n_classes=2, got %s" % gm["n_classes"]
assert len(oks) >= 1, "no ok cell produced (cache missing or cuda path broken)"
print("STEP-10 GPU SMOKE PASSED (binary cuda path + all-resolution + grid_manifest clean)")
'
