#!/usr/bin/env bash
#SBATCH --job-name=oaci-c11-src-onefold
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -u
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE KMP_DETERMINISTIC_REDUCTION=true
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
: "${OACI_DATALAKE_ROOT:=/projects/EEG-foundation-model/datalake/raw}"
SC=/projects/EEG-foundation-model/yinghao/oaci-c11-scratch; mkdir -p "$SC"
echo "[c11-src-onefold] node=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null) target=${OACI_TARGET:-1} seed=${OACI_SEED:-0} temp=${OACI_TEMP:-0.1}"
$PY -m oaci.confirmatory.src_onefold --target "${OACI_TARGET:-1}" --seed "${OACI_SEED:-0}" \
    --datalake-root "$OACI_DATALAKE_ROOT" --manifest-out "$SC/src_pilot_manifest.yaml" \
    --bootstrap-mode "${OACI_BMODE:-full}" --smooth-temperature "${OACI_TEMP:-0.1}" \
    --out-md oaci/reports/C11_SRC_ONEFOLD_PILOT.md --out-json oaci/reports/C11_SRC_ONEFOLD_PILOT.json
echo "=== rc=$? ==="
