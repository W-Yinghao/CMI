#!/bin/bash
# Project A — ONE-cell GPU smoke test: exercise the full CUDA path (train_h2 -> run_three_settings
# -> embed -> crossfit leakage -> finalize) fast (epochs 2, --fast) and FAIL LOUDLY unless the run
# manifest reports status=ok. Guards the 27-cell grid against device-placement bugs. NO --time.
#SBATCH --job-name=projA_gpu_smoke
#SBATCH --partition=A100,H100,V100,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/project_A_gpu_smoke_%j.out
#SBATCH --error=logs/project_A_gpu_smoke_%j.err

set -euo pipefail
mkdir -p logs
OUT=notes/project_A_observability/results/gpu_smoke

conda run -n icml python -m h2cmi.run_real_audited \
  --dataset BNCI2014_001 --subjects 1 2 3 --target-subject 1 --seed 0 \
  --epochs 2 --fast --align-factor subject --device cuda --outdir "$OUT"

STATUS=$(conda run -n icml python -c "import json;print(json.load(open('$OUT/run_manifest.json'))['status'])")
echo "smoke run status=$STATUS"
if [ "$STATUS" != "ok" ]; then
  echo "SMOKE FAIL: expected status=ok, got '$STATUS' (see run_manifest.json skip_reason)"
  conda run -n icml python -c "import json;print(json.load(open('$OUT/run_manifest.json')).get('skip_reason'))"
  exit 1
fi
echo "GPU SMOKE PASSED (full cuda path is clean)"
