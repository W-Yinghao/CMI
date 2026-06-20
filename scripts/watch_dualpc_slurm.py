#!/usr/bin/env python3
"""Watch and advance the DualPC paper Slurm submission.

This watcher is intentionally conservative:
  * it never requests a Slurm time limit;
  * it retries later when submit limits block the next group;
  * it submits the CPU post job only after all GPU groups have job ids.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


OK_STATES = {"COMPLETED"}
BAD_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


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


def parse_job_id(stdout: str) -> str:
    first = stdout.strip().splitlines()[0].strip()
    return first.split(";")[0]


def load_state(path: Path, groups: list[dict[str, str]]) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "created_at": now(),
        "updated_at": now(),
        "groups": groups,
        "post_job_id": "",
        "post_submitted_at": "",
        "done": False,
    }


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_active(job_id: str, cwd: Path) -> bool:
    proc = run(["squeue", "-h", "-j", job_id], cwd)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def sacct_states(job_id: str, cwd: Path) -> set[str]:
    proc = run(["sacct", "-n", "-P", "-j", job_id, "--format=State"], cwd)
    states: set[str] = set()
    if proc.returncode != 0:
        return states
    for line in proc.stdout.splitlines():
        state = line.strip().split("|")[0].split()[0]
        if state:
            states.add(state)
    return states


def job_status(job_id: str, cwd: Path) -> str:
    if not job_id:
        return "UNSUBMITTED"
    if is_active(job_id, cwd):
        return "ACTIVE"
    states = sacct_states(job_id, cwd)
    if not states:
        return "UNKNOWN"
    if states & BAD_STATES:
        return "FAILED"
    if states and states <= OK_STATES:
        return "COMPLETED"
    return ",".join(sorted(states))


def submit_group(group: dict, args: argparse.Namespace) -> tuple[bool, str]:
    proc = run(
        [
            "sbatch",
            "--parsable",
            f"--job-name=dualpc-paper-{group['name']}",
            f"--partition={args.gpu_partition}",
            f"--array={group['array']}",
            args.array_script,
        ],
        args.workdir,
    )
    if proc.returncode == 0:
        return True, parse_job_id(proc.stdout)
    return False, (proc.stderr or proc.stdout).strip()


def submit_post(state: dict, args: argparse.Namespace) -> tuple[bool, str]:
    job_ids = [g["job_id"] for g in state["groups"] if g.get("job_id")]
    dep = "afterok:" + ":".join(job_ids)
    proc = run(
        [
            "sbatch",
            "--parsable",
            f"--dependency={dep}",
            "--job-name=dualpc-post",
            args.post_script,
        ],
        args.workdir,
    )
    if proc.returncode == 0:
        return True, parse_job_id(proc.stdout)
    return False, (proc.stderr or proc.stdout).strip()


def parse_group(text: str) -> dict[str, str]:
    name, array = text.split("=", 1)
    return {"name": name, "array": array, "job_id": "", "submitted_at": ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--array-script", default="scripts/dualpc_paper_array.slurm")
    parser.add_argument("--post-script", default="scripts/dualpc_paper_post.slurm")
    parser.add_argument("--gpu-partition", default="A40")
    parser.add_argument("--group", action="append", required=True, help="name=array, e.g. g2=14-19%2")
    parser.add_argument("--initial", action="append", default=[], help="name=jobid for already submitted groups")
    parser.add_argument("--state-json", type=Path, default=Path("results/dualpc_protocol_paper/slurm_state.json"))
    parser.add_argument("--log", type=Path, default=Path("logs/dualpc_slurm_watcher.log"))
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=120.0)
    args = parser.parse_args()

    args.workdir = args.workdir.resolve()
    args.state_json = (args.workdir / args.state_json).resolve()
    args.log = (args.workdir / args.log).resolve()

    groups = [parse_group(g) for g in args.group]
    initial = dict(item.split("=", 1) for item in args.initial)
    for group in groups:
        if group["name"] in initial:
            group["job_id"] = initial[group["name"]]
            group["submitted_at"] = now()

    state = load_state(args.state_json, groups)
    known = {g["name"]: g for g in state["groups"]}
    for group in groups:
        if group["name"] not in known:
            state["groups"].append(group)
        elif group.get("job_id") and not known[group["name"]].get("job_id"):
            known[group["name"]].update(group)
    save_state(args.state_json, state)

    deadline = time.time() + args.max_hours * 3600
    log(args.log, f"watcher started state={args.state_json}")
    while time.time() < deadline:
        state = load_state(args.state_json, groups)

        for group in state["groups"]:
            if group.get("job_id"):
                continue
            ok, value = submit_group(group, args)
            if ok:
                group["job_id"] = value
                group["submitted_at"] = now()
                save_state(args.state_json, state)
                log(args.log, f"submitted {group['name']} array={group['array']} job_id={value}")
            else:
                log(args.log, f"submit deferred {group['name']} array={group['array']}: {value}")
                break

        statuses = {g["name"]: job_status(g.get("job_id", ""), args.workdir) for g in state["groups"]}
        log(args.log, "gpu statuses " + json.dumps(statuses, sort_keys=True))
        if any(status == "FAILED" for status in statuses.values()):
            save_state(args.state_json, state)
            log(args.log, "at least one GPU group failed; not submitting post job")
            return 2

        all_submitted = all(g.get("job_id") for g in state["groups"])
        if all_submitted and not state.get("post_job_id"):
            ok, value = submit_post(state, args)
            if ok:
                state["post_job_id"] = value
                state["post_submitted_at"] = now()
                save_state(args.state_json, state)
                log(args.log, f"submitted post job_id={value}")
            else:
                log(args.log, f"post submit deferred: {value}")

        post_job_id = state.get("post_job_id", "")
        if post_job_id:
            post_status = job_status(post_job_id, args.workdir)
            log(args.log, f"post status {post_job_id}={post_status}")
            if post_status == "COMPLETED":
                state["done"] = True
                save_state(args.state_json, state)
                log(args.log, "watcher completed")
                return 0
            if post_status == "FAILED":
                save_state(args.state_json, state)
                log(args.log, "post job failed")
                return 3

        save_state(args.state_json, state)
        time.sleep(args.interval)

    log(args.log, "watcher deadline reached")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
