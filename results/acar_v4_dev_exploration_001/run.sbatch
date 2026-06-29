#!/bin/bash
#SBATCH --job-name=acar_v4_dev_expl_001
#SBATCH --partition=CPU
#SBATCH --nodelist=nodecpu05
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/infres/yinwang/acar_v4_dev_exploration_001/slurm_%j.out
#SBATCH --error=/home/infres/yinwang/acar_v4_dev_exploration_001/slurm_%j.err
set -o pipefail
export ACAR_FEAT_DUMP=/home/infres/yinwang/CMI_AAAI/archive/lpc-cmi-failed/results/feat_dump_v4
export OMP_NUM_THREADS=1
PYBIN=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
WT=/home/infres/yinwang/CMI_AAAI_acar
cd "$WT" || { echo "WORKTREE_MISSING"; exit 97; }
echo "START $(date -Iseconds) host=$(hostname)"
echo "node_platform=$($PYBIN -c 'import platform;print(platform.platform())')"
echo "HEAD=$(git rev-parse HEAD) status=[$(git status --porcelain=v1 --untracked-files=all)]"
SECONDS=0
PYTHONPATH="$WT" $PYBIN -u -m acar.v4.real_adapter --output /home/infres/yinwang/acar_v4_dev_exploration_001/dev_out 2>&1 | tee /home/infres/yinwang/acar_v4_dev_exploration_001/console.log
rc=${PIPESTATUS[0]}
echo "$rc" > /home/infres/yinwang/acar_v4_dev_exploration_001/exit_code.txt
echo "END $(date -Iseconds) elapsed_s=$SECONDS CLI_EXIT_CODE=$rc"
exit $rc
