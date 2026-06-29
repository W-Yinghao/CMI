#!/bin/bash
#SBATCH --job-name=acar_v3_dev_002
#SBATCH --partition=CPU
#SBATCH --nodelist=nodecpu05
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/infres/yinwang/acar_v3_dev_run_002/slurm_%j.out
#SBATCH --error=/home/infres/yinwang/acar_v3_dev_run_002/slurm_%j.err
set -o pipefail
export ACAR_FEAT_DUMP=/home/infres/yinwang/CMI_AAAI/archive/lpc-cmi-failed/results/feat_dump_v4
PYBIN=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
cd /home/infres/yinwang/ACAR_V3_LOCKED_RUN_817b04f || { echo "WORKTREE_MISSING"; exit 97; }
echo "START $(date -Iseconds) host=$(hostname)"
NODEPLAT=$($PYBIN -c "import platform;print(platform.platform())")
echo "node_platform=$NODEPLAT"
if [ "$NODEPLAT" != "Linux-6.12.0-211.7.3.el10_2.x86_64-x86_64-with-glibc2.39" ]; then echo "NODE_PLATFORM_MISMATCH (need Linux-6.12.0-211.7.3.el10_2.x86_64-x86_64-with-glibc2.39) -> abort, no DEV compute"; echo 98 > /home/infres/yinwang/acar_v3_dev_run_002/exit_code.txt; exit 98; fi
echo "HEAD=$(git rev-parse HEAD) tag=$(git rev-list -n1 acar-v3-dev-design-v1) status=[$(git status --porcelain=v1 --untracked-files=all)]"
SECONDS=0
$PYBIN -u -m acar.v3.run_dev_binding --input-manifest /home/infres/yinwang/acar_v3_dev_run_002/acar_v3_dev_inputs.json --output /home/infres/yinwang/acar_v3_dev_run_002/dev_out 2>&1 | tee /home/infres/yinwang/acar_v3_dev_run_002/console.log
rc=${PIPESTATUS[0]}
echo "$rc" > /home/infres/yinwang/acar_v3_dev_run_002/exit_code.txt
echo "END $(date -Iseconds) elapsed_s=$SECONDS CLI_EXIT_CODE=$rc"
exit $rc
