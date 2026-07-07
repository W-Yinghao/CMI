"""Fork 1 Tier-1 --- target-information smoke driver (IMPLEMENTATION STAGE; experiments HARD-LOCKED).

This driver can ONLY:
  * `--dry-run`  : build the task plan + split schema and write a dry-run report (NO target labels read),
  * (bare) / `--execute` : HALT with EXPERIMENTS_DISABLED (runs_allowed=false in the driver config).

The real per-task eraser/benefit computation (which would read target-CALIBRATION labels) is intentionally
NOT wired in: it lives behind the execution lock and requires a separate PM go. See
notes/TARGET_INFO_TIER1_SMOKE_DRIVER_DESIGN.md.

  python -m tos_cmi.eeg.run_target_info_tier1_smoke --dry-run
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import yaml

if not __debug__:
    raise RuntimeError("run_target_info_tier1_smoke: leak/execution gates require assertions enabled; refuse "
                       "to run under -O / PYTHONOPTIMIZE.")

from tos_cmi.eeg.target_info_splits import (make_calibration_audit_splits, target_leak_structural_check,
                                            TARGET_LEAK_TOKEN)

CFG = "tos_cmi/eeg/configs/target_info_tier1_smoke_driver_fixed.yaml"
OUT = "tos_cmi/results/target_info/tier1_driver_dryrun"
HALT_MSG = "EXPERIMENTS_DISABLED: implementation-only stage; requires separate PM go."
DESIGN_LOCK_HASH = "3ad4ef312e325fa6"


def load_cfg(path=CFG):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("design_lock_hash") != DESIGN_LOCK_HASH:       # unconditional (survives -O); config-integrity lock
        raise ValueError("driver config design_lock_hash %r != frozen %r"
                         % (cfg.get("design_lock_hash"), DESIGN_LOCK_HASH))
    return cfg


def _folds(spec):
    return list(range(1, 6)) if spec == "first5" else list(range(1, int(spec) + 1))


def build_plan(cfg):
    """Enumerate plan rows at (dataset, backbone, world, fold, budget, intervention) granularity. No data read;
    k_grid / world_alpha_grid / repeats_R are inner-loop multipliers carried as metadata, not plan rows."""
    sc = cfg["tier1_scope"]
    folds = _folds(sc["folds"])
    budgets = list(cfg["budgets"].keys())
    rows = []
    for d in sc["datasets"]:
        for bb in sc["backbones"]:
            for w in cfg["worlds"]:
                for f in folds:
                    for bud in budgets:
                        for iv in cfg["interventions"]:
                            row = {"dataset": d, "backbone": bb, "world": w, "fold": f,
                                   "budget": bud, "intervention": iv}
                            if "k_grid" in cfg["budgets"][bud]:
                                row["k_grid"] = cfg["budgets"][bud]["k_grid"]
                                row["repeats_R"] = sc["repeats_R"]
                            rows.append(row)
    return rows


def build_schema(cfg):
    """Static split policy + label-access capability matrix. Declares WHICH components may read WHICH labels."""
    sc = cfg["tier1_scope"]
    return {
        "split_policy": {
            "stratified_by_class": True,
            "repeats_R": sc["repeats_R"],
            "calibration_used_by": ["B2/B3 gate benefit", "B4 oracle selector (diagnostic)"],
            "audit_used_by": ["final evaluation ONLY"],
            "k_unavailable_policy": "mark UNAVAILABLE; never reuse audit labels",
        },
        "label_access": {
            "source_labels": ["eraser_fit", "head_fit", "source_safety", "source_benefit_lcb"],
            "target_calibration_labels": ["B2_benefit_lcb", "B3_benefit_lcb"],
            "target_audit_labels": ["final_evaluation_only"],
            "B4_oracle_labels": ["diagnostic_selector_only"],
        },
        "hard_gates": [TARGET_LEAK_TOKEN, "EXPERIMENTS_DISABLED"],
        "inner_loops": {
            "world_alpha_grid": cfg.get("world_alpha_grid"),
            "k_grid": cfg["budgets"]["B2_k_labels_per_class"]["k_grid"],
            "repeats_R": sc["repeats_R"],
        },
        "b1_accept_forbidden": True,
        "b4_diagnostic_only": True,
    }


def _demo_structural_check(cfg):
    """Prove the leak gate is LIVE (not a comment) on synthetic disjoint splits --- reads no real EEG."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])                     # dummy target-subject labels
    splits = make_calibration_audit_splits(y, R=cfg["tier1_scope"]["repeats_R"], seed=0)
    return target_leak_structural_check(splits, cfg["budgets"])


def dry_run(cfg):
    os.makedirs(OUT, exist_ok=True)
    token = _demo_structural_check(cfg)                        # HALTS via AssertionError if leak invariants break
    plan, schema = build_plan(cfg), build_schema(cfg)
    json.dump({"n_plan_rows": len(plan), "plan": plan}, open("%s/target_info_tier1_plan.json" % OUT, "w"), indent=1)
    json.dump(schema, open("%s/target_info_tier1_schema.json" % OUT, "w"), indent=1)
    from tos_cmi.eeg.report_target_info_tier1 import write_report
    rpt = write_report(cfg, plan, schema, token, OUT)
    print("dry-run: %d plan rows ; structural gate %s ; report %s" % (len(plan), token, rpt))
    print("TARGET_INFO_TIER1_DRYRUN_DONE")
    return len(plan)


def run_cli(argv, cfg):
    """Return (exit_code, message). Deterministic + testable (no sys.exit here)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run and a.execute:                               # reject contradictory intent (R6)
        return 2, "CONFLICTING_FLAGS: --dry-run and --execute are mutually exclusive."
    if a.dry_run:
        n = dry_run(cfg)
        return 0, "TARGET_INFO_TIER1_DRYRUN_DONE (%d rows)" % n
    # bare invocation OR --execute: experiments are hard-locked at this stage
    if not cfg.get("runs_allowed", False):
        return 1, HALT_MSG
    # (unreachable while runs_allowed=false) real execution would require the separate PM go + wiring
    return 2, "runs_allowed=true but execution path is intentionally unimplemented at this stage"


def main():
    cfg = load_cfg()
    code, msg = run_cli(sys.argv[1:], cfg)
    print(msg)
    sys.exit(code)


if __name__ == "__main__":
    main()
