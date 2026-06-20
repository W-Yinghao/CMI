"""Write SLURM job-array files for the DualPC paper protocol.

The protocol generator prints reproducible commands. This helper turns the same
command list into:
  * a TSV task file for independent training/synthetic/selector jobs;
  * a GPU job-array wrapper that runs one task per array index;
  * a CPU post-processing script for readiness + paper-summary after the array.

It does not submit jobs.

Example:
  python scripts/dualpc_slurm_plan.py --profile paper --device cuda --seeds 0 1 2 \
    --out-dir results/dualpc_protocol_paper \
    --loso-datasets BNCI2014_001 MUMTAZ --scps-conditions PD SCZ
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualpc_protocol import build


def _label(comment: str) -> str:
    text = comment.lstrip("#").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "task"


def _is_post(comment: str) -> bool:
    clean = comment.lstrip("#").strip().lower()
    return (clean.startswith("readiness summary")
            or clean.startswith("paper aggregate")
            or clean.startswith("headline decision"))


def _write_task_file(path: Path, tasks: list[tuple[str, str]]):
    lines = []
    for i, (comment, command) in enumerate(tasks):
        lines.append(f"{i}\t{_label(comment)}\t{command}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_array_script(path: Path, task_file: Path, n_tasks: int, args):
    if n_tasks <= 0:
        raise ValueError("no array tasks to write")
    text = f"""#!/usr/bin/env bash
#SBATCH --job-name=dualpc-paper
#SBATCH --partition={args.partition}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --array=0-{n_tasks - 1}%{args.max_parallel}
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd {ROOT}
mkdir -p logs {args.out_dir} /tmp/matplotlib

export MNE_DATA=/projects/EEG-foundation-model/datalake/raw
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/datalake/raw
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib

TASK_FILE="${{TASK_FILE:-{task_file}}}"
line=$(awk -F '\\t' -v id="${{SLURM_ARRAY_TASK_ID}}" '$1 == id {{print $0}}' "$TASK_FILE")
if [[ -z "$line" ]]; then
  echo "No task for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}} in $TASK_FILE" >&2
  exit 2
fi
IFS=$'\\t' read -r task_id task_label task_cmd <<< "$line"
echo "task_id=$task_id label=$task_label"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "$task_cmd"
eval "$task_cmd"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_post_script(path: Path, post: list[tuple[str, str]], args):
    if not post:
        raise ValueError("no post commands to write")
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=dualpc-post",
        "#SBATCH --partition=CPU",
        f"#SBATCH --cpus-per-task={args.cpus}",
        "#SBATCH --mem=24G",
        "#SBATCH --output=logs/%x-%j.out",
        "#SBATCH --error=logs/%x-%j.err",
        "",
        "set -euo pipefail",
        f"cd {ROOT}",
        f"mkdir -p logs {args.out_dir} /tmp/matplotlib",
        "export PYTHONUNBUFFERED=1",
        "export MPLCONFIGDIR=/tmp/matplotlib",
        "",
    ]
    for comment, command in post:
        lines.extend([comment, command, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "paper"], default="paper")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default="results/dualpc_protocol_paper")
    ap.add_argument("--loso-datasets", nargs="+", default=["BNCI2014_001", "MUMTAZ"])
    ap.add_argument("--scps-conditions", nargs="+", default=["PD", "SCZ"])
    ap.add_argument("--task-file", default="scripts/dualpc_paper_tasks.tsv")
    ap.add_argument("--array-script", default="scripts/dualpc_paper_array.slurm")
    ap.add_argument("--post-script", default="scripts/dualpc_paper_post.slurm")
    ap.add_argument("--partition", default="V100")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", default="48G")
    ap.add_argument("--time", default="24:00:00")
    ap.add_argument("--max-parallel", type=int, default=2)
    args = ap.parse_args()

    commands = build(args)
    tasks = [(comment, command) for comment, command in commands if not _is_post(comment)]
    post = [(comment, command) for comment, command in commands if _is_post(comment)]

    task_file = Path(args.task_file)
    array_script = Path(args.array_script)
    post_script = Path(args.post_script)
    _write_task_file(task_file, tasks)
    _write_array_script(array_script, task_file, len(tasks), args)
    _write_post_script(post_script, post, args)

    print(f"wrote {len(tasks)} array tasks -> {task_file}")
    print(f"wrote array wrapper -> {array_script}")
    print(f"wrote post wrapper -> {post_script}")
    print("")
    print("Submit when GPU quota is available:")
    print(f"  jid=$(sbatch --parsable {array_script})")
    print(f"  sbatch --dependency=afterok:$jid {post_script}")


if __name__ == "__main__":
    main()
