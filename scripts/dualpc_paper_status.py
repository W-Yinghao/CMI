"""Preflight/status report for the DualPC paper SLURM package.

This is intentionally read-only: it does not submit jobs and does not create or
modify result files. Use it before submission to catch plan drift, and after
jobs finish to see which JSON artifacts are missing or malformed.

Example:
  python scripts/dualpc_paper_status.py \
    --task-file scripts/dualpc_paper_tasks.tsv \
    --array-script scripts/dualpc_paper_array.slurm \
    --post-script scripts/dualpc_paper_post.slurm
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


def _flag_value(command: str, flag: str):
    parts = shlex.split(command)
    for i, part in enumerate(parts[:-1]):
        if part == flag:
            return parts[i + 1]
    return None


def _flag_values(command: str, flag: str):
    parts = shlex.split(command)
    out = []
    for i, part in enumerate(parts):
        if part != flag:
            continue
        j = i + 1
        while j < len(parts) and not parts[j].startswith("--"):
            out.append(parts[j])
            j += 1
        break
    return out


def _family(name: str):
    return str(name).split(":", 1)[0]


def _read_tasks(path: Path) -> list[dict]:
    tasks = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            tasks.append({"id": None, "label": f"line_{lineno}", "command": line,
                          "parse_error": "expected 3 tab-separated columns"})
            continue
        idx, label, command = parts
        try:
            idx_val = int(idx)
        except ValueError:
            idx_val = None
        tasks.append({"id": idx_val, "label": label, "command": command})
    return tasks


def _json_status(path: Path):
    if not path.exists():
        return "PENDING", "missing"
    try:
        obj = json.load(open(path))
    except Exception as exc:
        return "FAIL", f"parse error: {exc}"
    if isinstance(obj, dict) and (isinstance(obj.get("summary"), dict)
                                  or isinstance(obj.get("selection_records"), list)
                                  or isinstance(obj.get("counts"), dict)):
        return "DONE", "json parse ok"
    return "WARN", "json lacks summary/selection_records/counts"


def _array_range(text: str):
    m = re.search(r"^#SBATCH\s+--array=(\d+)-(\d+)(?:%(\d+))?", text, re.MULTILINE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def _post_outputs(text: str):
    outputs = []
    for line in text.splitlines():
        if ("dualpc_readiness.py" in line or "dualpc_paper_summary.py" in line
                or "dualpc_decision_gate.py" in line):
            out = _flag_value(line, "--out-json")
            if out:
                if "dualpc_readiness.py" in line:
                    label = "readiness"
                elif "dualpc_paper_summary.py" in line:
                    label = "paper_summary"
                else:
                    label = "decision"
                outputs.append({"label": label, "path": out, "command": line})
    return outputs


def _decision_command(text: str):
    for line in text.splitlines():
        if "dualpc_decision_gate.py" in line:
            return line
    return ""


def _decision_requirements(command: str):
    if not command:
        return {}
    return {
        "min_comparison_tasks": int(_flag_value(command, "--min-comparison-tasks") or 0),
        "min_selector_tasks": int(_flag_value(command, "--min-selector-tasks") or 0),
        "required_baselines": _flag_values(command, "--required-baselines"),
        "required_synthetic_groups": _flag_values(command, "--required-synthetic-groups"),
    }


def _runner_task_key(command: str):
    if "cmi.run_loso" in command:
        dataset = _flag_value(command, "--dataset")
        return f"loso:{dataset}" if dataset else ""
    if "cmi.run_scps_crossdataset" in command:
        condition = _flag_value(command, "--condition")
        return f"scps:{condition}" if condition else ""
    return ""


def _selector_task_key(command: str):
    if "cmi.run_lambda_select" not in command:
        return ""
    dataset = _flag_value(command, "--dataset")
    return f"loso:{dataset}" if dataset else ""


def _package_policy_checks(tasks: list[dict], decision_req: dict):
    checks = []
    if not decision_req:
        return checks

    synthetic_groups = set()
    runner_tasks: dict[str, set[str]] = {}
    selector_tasks = set()
    for task in tasks:
        if task.get("parse_error"):
            continue
        command = task.get("command", "")
        if "synthetic/dualpc_validation.py" in command:
            synthetic_groups.update(_flag_values(command, "--dgps"))
        runner_key = _runner_task_key(command)
        if runner_key:
            runner_tasks.setdefault(runner_key, set()).update(
                _family(c) for c in _flag_values(command, "--configs")
            )
        selector_key = _selector_task_key(command)
        if selector_key:
            selector_tasks.add(selector_key)

    required_groups = set(decision_req.get("required_synthetic_groups", []))
    missing_groups = sorted(required_groups - synthetic_groups)
    if missing_groups:
        checks.append({"status": "FAIL", "check": "policy_synthetic_groups",
                       "note": "missing " + ",".join(missing_groups)})
    else:
        checks.append({"status": "PASS", "check": "policy_synthetic_groups",
                       "note": ",".join(sorted(synthetic_groups))})

    min_comparison = int(decision_req.get("min_comparison_tasks", 0))
    if len(runner_tasks) < min_comparison:
        checks.append({"status": "FAIL", "check": "policy_comparison_task_count",
                       "note": f"{len(runner_tasks)} < {min_comparison}: {sorted(runner_tasks)}"})
    else:
        checks.append({"status": "PASS", "check": "policy_comparison_task_count",
                       "note": f"{len(runner_tasks)} tasks: {sorted(runner_tasks)}"})

    required_baselines = set(decision_req.get("required_baselines", []))
    missing_baselines = []
    for task, families in sorted(runner_tasks.items()):
        missing = sorted(required_baselines - families)
        if missing:
            missing_baselines.append(f"{task}:{','.join(missing)}")
    if missing_baselines:
        checks.append({"status": "FAIL", "check": "policy_runner_baselines",
                       "note": ";".join(missing_baselines[:5])})
    else:
        checks.append({"status": "PASS", "check": "policy_runner_baselines",
                       "note": ",".join(sorted(required_baselines))})

    min_selector = int(decision_req.get("min_selector_tasks", 0))
    if len(selector_tasks) < min_selector:
        checks.append({"status": "FAIL", "check": "policy_selector_task_count",
                       "note": f"{len(selector_tasks)} < {min_selector}: {sorted(selector_tasks)}"})
    else:
        checks.append({"status": "PASS", "check": "policy_selector_task_count",
                       "note": f"{len(selector_tasks)} tasks: {sorted(selector_tasks)}"})
    return checks


def collect_status(task_file: str, array_script: str, post_script: str):
    task_path = Path(task_file)
    array_path = Path(array_script)
    post_path = Path(post_script)
    checks = []
    task_rows = []
    post_rows = []

    if not task_path.exists():
        checks.append({"status": "FAIL", "check": "task_file_exists", "note": str(task_path)})
        tasks = []
    else:
        checks.append({"status": "PASS", "check": "task_file_exists", "note": str(task_path)})
        tasks = _read_tasks(task_path)

    ids = [t["id"] for t in tasks if t.get("id") is not None]
    expected = list(range(len(tasks)))
    if ids == expected and not any(t.get("parse_error") for t in tasks):
        checks.append({"status": "PASS", "check": "task_ids_contiguous", "note": f"0..{len(tasks) - 1}"})
    else:
        checks.append({"status": "FAIL", "check": "task_ids_contiguous",
                       "note": f"ids={ids}; expected={expected}"})

    output_paths = []
    for task in tasks:
        if task.get("parse_error"):
            task_rows.append({"status": "FAIL", **task, "output": "", "note": task["parse_error"]})
            continue
        out = _flag_value(task["command"], "--out")
        if out:
            output_paths.append(out)
            status, note = _json_status(Path(out))
            task_rows.append({"status": status, **task, "output": out, "note": note})
        else:
            task_rows.append({"status": "READY", **task, "output": "", "note": "no JSON output expected"})

    duplicate_outputs = sorted({p for p in output_paths if output_paths.count(p) > 1})
    if duplicate_outputs:
        checks.append({"status": "FAIL", "check": "unique_task_outputs",
                       "note": ",".join(duplicate_outputs)})
    else:
        checks.append({"status": "PASS", "check": "unique_task_outputs",
                       "note": f"{len(output_paths)} output JSONs"})

    if not array_path.exists():
        checks.append({"status": "FAIL", "check": "array_script_exists", "note": str(array_path)})
    else:
        arr_text = array_path.read_text()
        checks.append({"status": "PASS", "check": "array_script_exists", "note": str(array_path)})
        arr = _array_range(arr_text)
        if arr and arr[0] == 0 and arr[1] == max(0, len(tasks) - 1):
            checks.append({"status": "PASS", "check": "array_range_matches_tasks",
                           "note": f"{arr[0]}-{arr[1]}%{arr[2] or ''}"})
        else:
            checks.append({"status": "FAIL", "check": "array_range_matches_tasks",
                           "note": f"array={arr}; tasks={len(tasks)}"})
        if str(task_path) in arr_text:
            checks.append({"status": "PASS", "check": "array_references_task_file", "note": str(task_path)})
        else:
            checks.append({"status": "WARN", "check": "array_references_task_file",
                           "note": f"{task_path} not found literally"})

    if not post_path.exists():
        checks.append({"status": "FAIL", "check": "post_script_exists", "note": str(post_path)})
    else:
        post_text = post_path.read_text()
        checks.append({"status": "PASS", "check": "post_script_exists", "note": str(post_path)})
        outs = _post_outputs(post_text)
        decision_req = _decision_requirements(_decision_command(post_text))
        labels = {o["label"] for o in outs}
        if {"readiness", "paper_summary", "decision"} <= labels:
            checks.append({"status": "PASS", "check": "post_outputs_declared",
                           "note": ",".join(sorted(labels))})
        else:
            checks.append({"status": "FAIL", "check": "post_outputs_declared",
                           "note": ",".join(sorted(labels)) or "none"})
        for out in outs:
            status, note = _json_status(Path(out["path"]))
            post_rows.append({"status": status, "label": out["label"], "output": out["path"],
                              "note": note})
        checks.extend(_package_policy_checks(tasks, decision_req))

    return {"checks": checks, "tasks": task_rows, "post": post_rows}


def _print_table(title: str, cols: list[str], rows: list[dict]):
    print(f"\n# {title}")
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row.get(c, "")) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", default="scripts/dualpc_paper_tasks.tsv")
    ap.add_argument("--array-script", default="scripts/dualpc_paper_array.slurm")
    ap.add_argument("--post-script", default="scripts/dualpc_paper_post.slurm")
    ap.add_argument("--only-problems", action="store_true")
    args = ap.parse_args()

    status = collect_status(args.task_file, args.array_script, args.post_script)
    checks = status["checks"]
    tasks = status["tasks"]
    post = status["post"]
    if args.only_problems:
        checks = [r for r in checks if r["status"] != "PASS"]
        tasks = [r for r in tasks if r["status"] not in {"DONE", "READY", "PENDING"}]
        post = [r for r in post if r["status"] not in {"DONE", "PENDING"}]
    _print_table("plan_checks", ["status", "check", "note"], checks)
    _print_table("task_status", ["status", "id", "label", "output", "note"], tasks)
    _print_table("post_status", ["status", "label", "output", "note"], post)
    rows = checks + tasks + post
    counts = {k: sum(1 for r in rows if r.get("status") == k)
              for k in ("PASS", "READY", "DONE", "PENDING", "WARN", "FAIL")}
    print("\n# counts: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    if counts["FAIL"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
