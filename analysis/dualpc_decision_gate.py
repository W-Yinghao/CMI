"""Decide whether DualPC is ready to be promoted as the paper headline method.

This script consumes the post-run artifacts produced by the paper protocol:
``dualpc_readiness.json`` and ``dualpc_paper_summary.json``. It is intentionally
read-only and conservative: missing formal comparisons are PENDING, hard failures
are NOT_READY, and warnings keep the method in NEEDS_REVIEW rather than headline.

The gate is not a statistical test. It is the mechanical audit that the evidence
needed for the paper claim is present and points in the right direction:
accuracy parity plus no raised GLS conditional, GLS P(z), or JS P(Y|Z) probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_WARN_NOTES = {"baseline"}


def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None, {"status": "PENDING", "check": f"{p.name}_exists", "note": f"missing: {path}"}
    try:
        return json.load(open(p)), {"status": "PASS", "check": f"{p.name}_exists", "note": path}
    except Exception as exc:
        return None, {"status": "FAIL", "check": f"{p.name}_parse", "note": str(exc)}


def _worse(a: str, b: str) -> str:
    order = {"PASS": 0, "PENDING": 1, "WARN": 2, "FAIL": 3}
    return a if order[a] >= order[b] else b


def _decision_from_status(status: str) -> str:
    return {
        "PASS": "HEADLINE_READY",
        "PENDING": "PENDING",
        "WARN": "NEEDS_REVIEW",
        "FAIL": "NOT_READY",
    }[status]


def _family(name: Any) -> str:
    return str(name).split(":", 1)[0]


def _readiness_checks(obj: dict[str, Any], required_synthetic_groups: list[str]) -> list[dict[str, str]]:
    checks = []
    rows = obj.get("rows", []) if isinstance(obj, dict) else []
    counts = obj.get("counts", {}) if isinstance(obj, dict) else {}
    fail_n = int(counts.get("FAIL", 0) or 0)
    if fail_n:
        checks.append({"status": "FAIL", "check": "readiness_failures", "note": str(fail_n)})
    else:
        checks.append({"status": "PASS", "check": "readiness_failures", "note": "0"})

    nonbaseline_warns = []
    for row in rows:
        if row.get("status") != "WARN":
            continue
        note = str(row.get("note", "")).strip().lower()
        if note not in BASELINE_WARN_NOTES:
            nonbaseline_warns.append(row)
    if nonbaseline_warns:
        examples = ",".join(f"{r.get('suite')}:{r.get('method')}" for r in nonbaseline_warns[:5])
        checks.append({"status": "WARN", "check": "readiness_nonbaseline_warns",
                       "note": f"{len(nonbaseline_warns)} ({examples})"})
    else:
        checks.append({"status": "PASS", "check": "readiness_nonbaseline_warns", "note": "0"})

    if required_synthetic_groups:
        passed = {
            str(row.get("group"))
            for row in rows
            if row.get("suite") == "synthetic"
            and _family(row.get("method")) == "dualpc"
            and row.get("status") == "PASS"
        }
        missing = sorted(set(required_synthetic_groups) - passed)
        if missing:
            checks.append({"status": "PENDING", "check": "synthetic_group_coverage",
                           "note": "missing " + ",".join(missing)})
        else:
            checks.append({"status": "PASS", "check": "synthetic_group_coverage",
                           "note": ",".join(required_synthetic_groups)})
    return checks


def _paper_checks(obj: dict[str, Any], min_comparison_tasks: int, min_selector_tasks: int,
                  required_baselines: list[str], require_selector: bool) -> list[dict[str, str]]:
    checks = []
    comparisons = obj.get("comparison_summary", []) if isinstance(obj, dict) else []
    selectors = obj.get("selector_summary", []) if isinstance(obj, dict) else []
    tasks = sorted({str(r.get("task")) for r in comparisons if r.get("task")})
    if len(tasks) < min_comparison_tasks:
        checks.append({"status": "PENDING", "check": "comparison_task_count",
                       "note": f"{len(tasks)} < {min_comparison_tasks}: {tasks}"})
    else:
        checks.append({"status": "PASS", "check": "comparison_task_count",
                       "note": f"{len(tasks)} tasks: {tasks}"})

    missing_baselines = []
    required = set(required_baselines)
    for task in tasks:
        seen = {_family(r.get("baseline")) for r in comparisons if str(r.get("task")) == task}
        missing = sorted(required - seen)
        if missing:
            missing_baselines.append(f"{task}:{','.join(missing)}")
    if missing_baselines:
        checks.append({"status": "PENDING", "check": "comparison_baseline_coverage",
                       "note": ";".join(missing_baselines[:5])})
    else:
        checks.append({"status": "PASS", "check": "comparison_baseline_coverage",
                       "note": ",".join(required_baselines) if required_baselines else "not required"})

    fail_cmp = [r for r in comparisons if r.get("status") == "FAIL"]
    warn_cmp = [r for r in comparisons if r.get("status") == "WARN"]
    if fail_cmp:
        examples = ",".join(f"{r.get('task')}:{r.get('baseline')}" for r in fail_cmp[:5])
        checks.append({"status": "FAIL", "check": "comparison_failures",
                       "note": f"{len(fail_cmp)} ({examples})"})
    elif warn_cmp:
        examples = ",".join(f"{r.get('task')}:{r.get('baseline')}" for r in warn_cmp[:5])
        checks.append({"status": "WARN", "check": "comparison_warnings",
                       "note": f"{len(warn_cmp)} ({examples})"})
    else:
        checks.append({"status": "PASS", "check": "comparison_status", "note": f"{len(comparisons)} rows"})

    selector_tasks = sorted({str(r.get("task")) for r in selectors if r.get("task")})
    if require_selector and len(selector_tasks) < min_selector_tasks:
        checks.append({"status": "PENDING", "check": "selector_task_count",
                       "note": f"{len(selector_tasks)} < {min_selector_tasks}: {selector_tasks}"})
    elif require_selector:
        checks.append({"status": "PASS", "check": "selector_task_count",
                       "note": f"{len(selector_tasks)} tasks: {selector_tasks}"})

    if require_selector and not selectors:
        checks.append({"status": "PENDING", "check": "selector_summary_present", "note": "missing"})
    elif selectors:
        fail_sel = [r for r in selectors if r.get("status") == "FAIL"]
        warn_sel = [r for r in selectors if r.get("status") == "WARN"]
        incomplete = [r for r in selectors
                      if float(r.get("final_record_frac") or 0.0) < 1.0
                      or float(r.get("final_probe_valid_frac") or 0.0) < 1.0]
        if fail_sel:
            checks.append({"status": "FAIL", "check": "selector_failures", "note": str(len(fail_sel))})
        elif warn_sel or incomplete:
            checks.append({"status": "WARN", "check": "selector_warnings",
                           "note": f"warn={len(warn_sel)} incomplete={len(incomplete)}"})
        else:
            checks.append({"status": "PASS", "check": "selector_status", "note": f"{len(selectors)} rows"})
    else:
        checks.append({"status": "PASS", "check": "selector_summary_present", "note": "not required"})
    return checks


def evaluate(readiness_path: str, paper_summary_path: str, min_comparison_tasks: int = 1,
             min_selector_tasks: int = 0, required_baselines: list[str] | None = None,
             required_synthetic_groups: list[str] | None = None,
             require_selector: bool = True) -> dict[str, Any]:
    if required_baselines is None:
        required_baselines = ["erm", "lpc_prior"]
    if required_synthetic_groups is None:
        required_synthetic_groups = []
    checks = []
    readiness, readiness_check = _load_json(readiness_path)
    paper, paper_check = _load_json(paper_summary_path)
    checks.extend([readiness_check, paper_check])

    if readiness is not None:
        checks.extend(_readiness_checks(readiness, required_synthetic_groups))
    if paper is not None:
        checks.extend(_paper_checks(paper, min_comparison_tasks, min_selector_tasks,
                                    required_baselines, require_selector))

    status = "PASS"
    for check in checks:
        status = _worse(status, check["status"])
    decision = _decision_from_status(status)
    if decision == "HEADLINE_READY":
        note = "DualPC passes mechanical headline gates"
    elif decision == "NEEDS_REVIEW":
        note = "DualPC evidence is complete enough to inspect but has warnings"
    elif decision == "PENDING":
        note = "formal paper-profile evidence is incomplete"
    else:
        note = "DualPC should not be promoted as headline under current evidence"
    return {"decision": decision, "status": status, "note": note, "checks": checks}


def _print(out: dict[str, Any]):
    print("decision\tstatus\tnote")
    print(f"{out['decision']}\t{out['status']}\t{out['note']}")
    print("\n# checks")
    print("status\tcheck\tnote")
    for row in out["checks"]:
        print(f"{row['status']}\t{row['check']}\t{row['note']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readiness", default="results/dualpc_protocol_paper/dualpc_readiness.json")
    ap.add_argument("--paper-summary", default="results/dualpc_protocol_paper/dualpc_paper_summary.json")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--min-comparison-tasks", type=int, default=1)
    ap.add_argument("--min-selector-tasks", type=int, default=0)
    ap.add_argument("--required-baselines", nargs="+", default=["erm", "lpc_prior"],
                    help="Baseline method families required for every comparison task")
    ap.add_argument("--required-synthetic-groups", nargs="+", default=[],
                    help="Synthetic DGP groups that must have a PASS main DualPC readiness row")
    ap.add_argument("--allow-missing-selector", action="store_true")
    args = ap.parse_args()

    out = evaluate(args.readiness, args.paper_summary,
                   min_comparison_tasks=args.min_comparison_tasks,
                   min_selector_tasks=args.min_selector_tasks,
                   required_baselines=args.required_baselines,
                   required_synthetic_groups=args.required_synthetic_groups,
                   require_selector=not args.allow_missing_selector)
    _print(out)
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
