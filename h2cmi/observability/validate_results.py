"""Project A — validate audited run directories + emit the tracked summary digest.

For each run under `--root`, checks the claim-boundary invariants of an audited eval report
(08 §6/§7), then writes a committable summary JSON/MD (via `result_index`). Exits non-zero if
any run violates an invariant, so it can gate CI / a reviewer's acceptance.

  python -m h2cmi.observability.validate_results --root <run root> \
      --out-json <summary.json> --out-md <summary.md>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .result_index import KNOWN_ESTIMANDS, _load_json, build_summary, write_summary_md

_TARGET_METRIC_ESTIMANDS = {"balanced_accuracy", "target_gain", "target_risk"}
_RUN_DIR_RE = re.compile(r"dataset=(?P<dataset>.+)_target=(?P<target>-?\d+)_seed=(?P<seed>-?\d+)")


def validate_run(run_dir) -> Tuple[bool, List[str]]:
    """Return (ok, issues) for one audited run directory."""
    run_dir = Path(run_dir)
    issues: List[str] = []
    manifest = _load_json(run_dir / "run_manifest.json")
    report = _load_json(run_dir / "observability_report.json")
    if manifest is None:
        return False, ["run_manifest.json missing/invalid"]

    # legal SKIP vocabulary: only status=='skipped' (with a reason) short-circuits; any other
    # non-'ok' status is itself an error (closes the "set status to smuggle a report unaudited" hole)
    status = manifest.get("status")
    if status == "skipped":
        return (bool(manifest.get("skip_reason")),
                [] if manifest.get("skip_reason") else ["skipped run missing skip_reason"])
    if status != "ok":
        return False, [f"invalid manifest status {status!r} (expected 'ok' or 'skipped')"]

    # (1) report is legal JSON
    if report is None:
        return False, ["observability_report.json missing/invalid"]
    claims = report.get("claims", [])

    # (2) no forbidden-claim violations
    if report.get("forbidden_claims_violated"):
        issues.append(f"forbidden_claims_violated non-empty: {report['forbidden_claims_violated']}")

    for c in claims:
        est, name, reg = c.get("estimand"), c.get("name"), c.get("regime")
        # (0) every claim must carry a KNOWN estimand — blocks the relabel bypass
        if est not in KNOWN_ESTIMANDS:
            issues.append(f"{name}: unknown/relabelled estimand {est!r}")
        # (0b) a rejected claim can NEVER be identifiable, regardless of estimand kind
        if not c.get("allowed") and c.get("identifiable_estimand") is not None:
            issues.append(f"{name}: rejected claim marked identifiable ({c.get('identifiable_estimand')})")
        # (3) R0/R1 target metrics must be oracle/eval-only, not identifiable
        if est in _TARGET_METRIC_ESTIMANDS and reg in ("R0", "R1") and c.get("allowed"):
            if c.get("identifiable_estimand") is not None:
                issues.append(f"{name}: R0/R1 target metric marked identifiable "
                              f"({c['identifiable_estimand']})")
            if not c.get("reportable_metric"):
                issues.append(f"{name}: allowed target metric not reportable")
            if not c.get("oracle_fields_used_for_validation_only"):
                issues.append(f"{name}: target metric missing oracle/eval-only marking")
            # (6) metric payload present but is evidence only (does not license identification)
            if c.get("metric_payload") is None:
                issues.append(f"{name}: reportable target metric has no metric_payload")
        # (4) offline-TTA prior: undeclared C1∧C2∧C3 -> must be rejected; a rejected prior may
        #     only appear as conclusion=False (reported-not-identified), never as a conclusion.
        if est == "target_prior":
            missing = set(("C1", "C2", "C3")) - set(c.get("contracts_invoked", []))
            if missing and c.get("allowed"):
                issues.append(f"{name}: target prior allowed without C1∧C2∧C3 (missing {sorted(missing)})")
            if not c.get("allowed") and c.get("conclusion") is True:
                issues.append(f"{name}: rejected target prior asserted as a conclusion (overclaim)")
        # (5) leakage stays a diagnostic
        if est == "leakage" and not c.get("is_diagnostic"):
            issues.append(f"{name}: leakage claim not marked diagnostic")

    # (7) manifest <-> path consistency: EXACT token match on a canonical dir name, with the
    #     binding fields REQUIRED for an ok run (closes the substring-prefix + opt-in holes).
    m = _RUN_DIR_RE.fullmatch(run_dir.name)
    if m is None:
        issues.append(f"run dir '{run_dir.name}' not in canonical "
                      f"dataset=..._target=<int>_seed=<int> form")
    else:
        for field, group in (("dataset", "dataset"), ("target_subject", "target"), ("seed", "seed")):
            mv: Optional[object] = manifest.get(field)
            if mv is None:
                issues.append(f"manifest missing binding field '{field}' for an ok run")
            elif str(mv) != m.group(group):
                issues.append(f"manifest {field}={mv} != dir token '{m.group(group)}'")

    return (len(issues) == 0), issues


def validate_all(root) -> Dict[str, dict]:
    root = Path(root)
    out: Dict[str, dict] = {}
    for mp in sorted(root.glob("*/run_manifest.json")):
        ok, issues = validate_run(mp.parent)
        out[mp.parent.name] = {"valid": ok, "issues": issues}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate audited runs + emit summary digest")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--project", default="Project A")
    ap.add_argument("--step", default="Step 8")
    ap.add_argument("--dataset", default="BNCI2014_001")
    args = ap.parse_args(argv)

    validations = validate_all(args.root)
    summary = build_summary(args.root, project=args.project, step=args.step, dataset=args.dataset)
    summary["validation"] = {"all_valid": all(v["valid"] for v in validations.values()) if
                             validations else False, "per_run": validations}

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_summary_md(summary, args.out_md)

    a, v = summary["aggregate"], summary["validation"]
    print(f"runs={a['n_runs']} ok={a['n_ok']} skipped={a['n_skipped']} "
          f"all_valid={v['all_valid']} violations_empty={a['all_forbidden_violations_empty']} "
          f"target_metrics_oracle_only={a['all_target_metrics_oracle_only']}")
    for name, res in validations.items():
        if not res["valid"]:
            print(f"  INVALID {name}: {res['issues']}")
    return 0 if v["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
