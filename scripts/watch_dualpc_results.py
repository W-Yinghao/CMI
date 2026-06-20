#!/usr/bin/env python3
"""Per-task result watcher for the DualPC paper Slurm arrays.

The Slurm submission watcher advances jobs. This watcher follows individual array
elements and runs lightweight analysis as each task finishes and writes its JSON.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {message}\n")


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_array_ids(spec: str) -> set[int]:
    spec = spec.split("%", 1)[0]
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def active_task_ids(job_id: str, cwd: Path) -> set[int]:
    proc = run(["squeue", "-h", "-j", job_id, "-o", "%i"], cwd)
    ids: set[int] = set()
    if proc.returncode != 0:
        return ids
    for raw in proc.stdout.splitlines():
        raw = raw.strip()
        if not raw or "_" not in raw:
            continue
        task = raw.split("_", 1)[1]
        if task.startswith("[") and task.endswith("]"):
            ids.update(parse_array_ids(task[1:-1]))
        elif task.isdigit():
            ids.add(int(task))
    return ids


def read_tasks(path: Path) -> dict[int, dict]:
    tasks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task_id_s, label, command = line.split("\t", 2)
        task_id = int(task_id_s)
        out_json = ""
        parts = shlex.split(command)
        if "--out" in parts:
            out_i = parts.index("--out") + 1
            if out_i < len(parts):
                out_json = parts[out_i]
        tasks[task_id] = {
            "task_id": task_id,
            "label": label,
            "command": command,
            "out_json": out_json,
        }
    return tasks


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": now(), "updated_at": now(), "tasks": {}}


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def group_map(slurm_state: Path) -> tuple[dict[int, tuple[str, str]], list[str]]:
    obj = json.loads(slurm_state.read_text(encoding="utf-8"))
    mapping: dict[int, tuple[str, str]] = {}
    job_ids: list[str] = []
    for group in obj.get("groups", []):
        job_id = str(group.get("job_id", ""))
        name = str(group.get("name", ""))
        if not job_id:
            continue
        job_ids.append(job_id)
        for task_id in parse_array_ids(str(group.get("array", ""))):
            mapping[task_id] = (name, job_id)
    return mapping, job_ids


def log_paths(log_dir: Path, group_name: str, job_id: str, task_id: int) -> tuple[Path, Path]:
    stem = f"dualpc-paper-{group_name}-{job_id}_{task_id}"
    return log_dir / f"{stem}.out", log_dir / f"{stem}.err"


def valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def error_tail(path: Path, n: int = 40) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = "\n".join(lines[-n:])
    patterns = ("Traceback", "RuntimeError", "ValueError", "CUDA out of memory", "OutOfMemoryError")
    return tail if any(p in tail for p in patterns) else ""


def analyze_json(task: dict, args: argparse.Namespace) -> dict:
    out_path = args.workdir / task["out_json"]
    analysis_dir = args.workdir / args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    task_id = int(task["task_id"])
    readiness = analysis_dir / f"task_{task_id:02d}_{task['label']}_readiness.json"
    paper = analysis_dir / f"task_{task_id:02d}_{task['label']}_paper_summary.json"
    procs = {
        "readiness": run([args.python, "analysis/dualpc_readiness.py", str(out_path), "--out-json", str(readiness)], args.workdir),
        "paper_summary": run([args.python, "analysis/dualpc_paper_summary.py", str(out_path), "--out-json", str(paper)], args.workdir),
    }
    result = {
        "readiness": str(readiness.relative_to(args.workdir)),
        "paper_summary": str(paper.relative_to(args.workdir)),
        "analysis_ok": all(p.returncode == 0 for p in procs.values()),
    }
    if not result["analysis_ok"]:
        result["analysis_error"] = {
            name: (proc.stderr or proc.stdout)[-2000:]
            for name, proc in procs.items()
            if proc.returncode != 0
        }
    return result


def analyze_partial(args: argparse.Namespace) -> None:
    partial_dir = args.workdir / args.partial_dir
    paths = sorted(str(p) for p in partial_dir.glob("*.json")
                   if p.name not in {"dualpc_readiness_partial.json", "dualpc_paper_summary_partial.json"})
    if not paths:
        return
    run([args.python, "analysis/dualpc_readiness.py", *paths,
         "--out-json", str(args.partial_dir / "dualpc_readiness_partial.json")], args.workdir)
    run([args.python, "analysis/dualpc_paper_summary.py", *paths,
         "--out-json", str(args.partial_dir / "dualpc_paper_summary_partial.json")], args.workdir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--task-file", type=Path, default=Path("scripts/dualpc_paper_tasks.tsv"))
    parser.add_argument("--slurm-state", type=Path, default=Path("results/dualpc_protocol_paper/slurm_state.json"))
    parser.add_argument("--state-json", type=Path, default=Path("results/dualpc_protocol_paper/result_watch_state.json"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/dualpc_protocol_paper/per_task_analysis"))
    parser.add_argument("--partial-dir", type=Path, default=Path("results/dualpc_protocol_paper"))
    parser.add_argument("--log", type=Path, default=Path("logs/dualpc_result_watcher.log"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--python", default="/home/infres/yinwang/anaconda3/envs/icml/bin/python")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=120.0)
    args = parser.parse_args()

    args.workdir = args.workdir.resolve()
    args.task_file = (args.workdir / args.task_file).resolve()
    args.slurm_state = (args.workdir / args.slurm_state).resolve()
    args.state_json = (args.workdir / args.state_json).resolve()
    args.log = (args.workdir / args.log).resolve()
    args.log_dir = (args.workdir / args.log_dir).resolve()

    deadline = time.time() + args.max_hours * 3600
    log(args.log, f"result watcher started state={args.state_json}")
    while time.time() < deadline:
        tasks = read_tasks(args.task_file)
        state = load_state(args.state_json)
        if not args.slurm_state.exists():
            log(args.log, f"missing slurm state {args.slurm_state}")
            time.sleep(args.interval)
            continue
        task_to_group, job_ids = group_map(args.slurm_state)
        active = set()
        for job_id in job_ids:
            active.update(active_task_ids(job_id, args.workdir))

        changed = False
        for task_id, task in sorted(tasks.items()):
            task_state = state["tasks"].setdefault(str(task_id), {
                "label": task["label"],
                "out_json": task["out_json"],
                "status": "pending",
            })
            if task_id in active:
                if task_state.get("status") != "active":
                    task_state["status"] = "active"
                    task_state["updated_at"] = now()
                    changed = True
                continue
            if task_state.get("status") in {"analyzed", "complete_no_json", "failed"}:
                continue

            group_name, job_id = task_to_group.get(task_id, ("", ""))
            out_log, err_log = log_paths(args.log_dir, group_name, job_id, task_id) if job_id else (Path(), Path())
            err = error_tail(err_log)
            out_json = args.workdir / task["out_json"] if task["out_json"] else None
            if out_json and out_json.exists() and valid_json(out_json):
                task_state.update({
                    "status": "analyzing",
                    "completed_at": now(),
                    "out_log": str(out_log.relative_to(args.workdir)) if out_log.exists() else "",
                    "err_log": str(err_log.relative_to(args.workdir)) if err_log.exists() else "",
                })
                save_state(args.state_json, state)
                analysis = analyze_json(task, args)
                task_state.update({"status": "analyzed", "analyzed_at": now(), **analysis})
                log(args.log, f"analyzed task {task_id} {task['label']} -> {task['out_json']}")
                changed = True
                analyze_partial(args)
            elif err:
                task_state.update({
                    "status": "failed",
                    "failed_at": now(),
                    "error_tail": err,
                    "out_log": str(out_log.relative_to(args.workdir)) if out_log.exists() else "",
                    "err_log": str(err_log.relative_to(args.workdir)) if err_log.exists() else "",
                })
                log(args.log, f"failed task {task_id} {task['label']} see {err_log}")
                changed = True
            elif not task["out_json"] and (out_log.exists() or err_log.exists()):
                task_state.update({
                    "status": "complete_no_json",
                    "completed_at": now(),
                    "out_log": str(out_log.relative_to(args.workdir)) if out_log.exists() else "",
                    "err_log": str(err_log.relative_to(args.workdir)) if err_log.exists() else "",
                })
                log(args.log, f"completed no-json task {task_id} {task['label']}")
                changed = True

        if changed:
            save_state(args.state_json, state)
        counts = {}
        for item in state["tasks"].values():
            counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
        log(args.log, "result statuses " + json.dumps(counts, sort_keys=True))
        if len(state["tasks"]) == len(tasks) and all(v.get("status") in {"analyzed", "complete_no_json"} for v in state["tasks"].values()):
            state["done"] = True
            save_state(args.state_json, state)
            log(args.log, "result watcher completed")
            return 0
        if any(v.get("status") == "failed" for v in state["tasks"].values()):
            save_state(args.state_json, state)
            log(args.log, "result watcher found failed task")
            return 2
        time.sleep(args.interval)

    log(args.log, "result watcher deadline reached")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
