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
import hashlib
import numpy as np
import yaml

if not __debug__:
    raise RuntimeError("run_target_info_tier1_smoke: leak/execution gates require assertions enabled; refuse "
                       "to run under -O / PYTHONOPTIMIZE.")

from tos_cmi.eeg.target_info_splits import (make_calibration_audit_splits, select_k_per_class,
                                            target_leak_structural_check, budget_action, b1_triage_action,
                                            DecisionContext, SourceContext, CalibrationContext,
                                            UnlabeledTargetContext, AuditView, _family, TARGET_LEAK_TOKEN)

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


# ============================ full executable task expansion ============================
def _task_seed(fold, k, split_id, alpha):
    key = "f%s_k%s_s%s_a%s" % (fold, k, split_id, alpha)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _budget_units(fam, spec, R):
    """Inner-loop expansion per budget FAMILY. B2 = k x split; B3 = one SEQUENCE task per split (sweeps k
    internally with early stop); B0/B1/B4 = one decision (no k/split)."""
    if fam == "B2":
        return [{"k": k, "split_id": s} for k in spec["k_grid"] for s in range(1, R + 1)]
    if fam == "B3":
        return [{"split_id": s, "b3_sequence": True} for s in range(1, R + 1)]
    return [{}]                                                # B0 / B1 / B4


def expand_tasks(cfg):
    """Expand plan rows into executable task units. Every unit carries the fields the executor needs; the
    calibration/audit index hashes are DEFERRED_RUN (computed from real target trials only inside the locked
    executor -- dry-run reads no data). Deterministic + data-free."""
    sc = cfg["tier1_scope"]
    folds = _folds(sc["folds"])
    R = sc["repeats_R"]
    alphas = cfg["world_alpha_grid"]
    rows = []
    for d in sc["datasets"]:
        for bb in sc["backbones"]:
            for w in cfg["worlds"]:
                for a in alphas:
                    for f in folds:
                        for iv in cfg["interventions"]:
                            for bud in cfg["budgets"]:
                                for u in _budget_units(_family(bud), cfg["budgets"][bud], R):
                                    rows.append({
                                        "dataset": d, "backbone": bb, "world": w, "alpha": a, "fold": f,
                                        "target_subject": "fold%d_heldout" % f, "intervention": iv,
                                        "budget": bud, **u,
                                        "calibration_idx_hash": "DEFERRED_RUN", "audit_idx_hash": "DEFERRED_RUN",
                                        "random_seed": _task_seed(f, u.get("k", "-"), u.get("split_id", "-"), a)})
    return rows


# ============================ pure estimators (testable on dummy arrays) ============================
def _lazy_estimators():
    """Heavy imports are LAZY so dry-run / import stay free of sklearn+eraser modules (red-team invariant)."""
    from tos_cmi.eeg.source_ood_benefit_gate import _bacc, _boot_bound
    return _bacc, _boot_bound


def calibration_delta_bacc(eraser, src: SourceContext, cal: CalibrationContext):
    """ΔbAcc_cal = bAcc_cal(erased) - bAcc_cal(identity). Head fit on SOURCE, evaluated on target CALIBRATION
    labels ONLY. One scalar per (subject, split). Reads NO audit labels."""
    _bacc, _ = _lazy_estimators()
    return (_bacc(eraser(src.Zs), src.ys, eraser(cal.Zt_cal), cal.yt_cal)
            - _bacc(src.Zs, src.ys, cal.Zt_cal, cal.yt_cal))


def unlabeled_mismatch(src: SourceContext, unl: UnlabeledTargetContext):
    """B1 TRIAGE statistic: standardized mean shift between source and (unlabeled) target features. Decision-only
    magnitude; feeds b1_triage_action which can never accept."""
    mu_s, mu_t = src.Zs.mean(0), unl.Zt.mean(0)
    sd = src.Zs.std(0) + 1e-8
    return float(np.mean(np.abs(mu_s - mu_t) / sd))


def audit_scalar(eraser, src: SourceContext, audit: AuditView, Zt_audit):
    """The HONEST held-out number: ΔbAcc on the AUDIT split. The ONLY function that reads audit labels; it is
    never called by a gate and never feeds a DecisionContext."""
    _bacc, _ = _lazy_estimators()
    return (_bacc(eraser(src.Zs), src.ys, eraser(Zt_audit), audit.audit_y)
            - _bacc(src.Zs, src.ys, Zt_audit, audit.audit_y))


def compute_decision_row(budget, source_safety_pass, source_benefit_lcb,
                         cal_deltas, cal_random_deltas, cal_clusters, unlabeled_mismatch_score,
                         thr=0.01, boot_seed=0):
    """Build the DecisionContext (source + calibration ONLY) and return a decision row. Structurally has NO
    audit input, so a decision is invariant to any audit-label permutation (see the leak test)."""
    fam = _family(budget)
    kw = dict(budget=budget, source_safety_pass=source_safety_pass, benefit_thr=thr)
    if fam == "B0":
        kw["source_benefit_lcb"] = source_benefit_lcb
    elif fam == "B1":
        kw["unlabeled_triage"] = b1_triage_action(unlabeled_mismatch_score)
    elif fam in ("B2", "B3"):
        _, _boot = _lazy_estimators()
        lcb = _boot(cal_deltas, cal_clusters, "lower", rng=np.random.default_rng(boot_seed))
        rlcb = (_boot(cal_random_deltas, cal_clusters, "lower", rng=np.random.default_rng(boot_seed + 1))
                if cal_random_deltas is not None else float("-inf"))
        kw["cal_benefit_lcb"] = lcb
        kw["beats_random"] = bool(lcb == lcb and lcb > rlcb)   # gain not reproduced by same-k random
    ctx = DecisionContext(**kw)
    action = budget_action(ctx)
    specific = None
    if action == "accept":
        specific = "accepted_specific" if kw.get("beats_random", True) else "accepted_non_specific"
    return {"budget": budget, "action": action, "source_safety_pass": bool(source_safety_pass),
            "cal_benefit_lcb": kw.get("cal_benefit_lcb"), "beats_random": kw.get("beats_random"),
            "source_benefit_lcb": kw.get("source_benefit_lcb"), "unlabeled_triage": kw.get("unlabeled_triage"),
            "specificity": specific}                            # NOTE: no audit field anywhere


def b3_sequential_decision(k_grid, deltas_by_k, random_by_k, cal_clusters, thr=0.01, boot_seed=0):
    """Sequential calibration: reveal k in ASCENDING order; stop as soon as accept or not-beneficial is
    certified. Returns (action, k_used, ks_read). ks_read proves no FUTURE (larger-k) labels were read before
    stopping. NOT active learning -- fixed ascending reveal, no uncertainty sampling."""
    _, _boot = _lazy_estimators()
    ks_read = []
    for k in sorted(k_grid):
        ks_read.append(k)
        d = deltas_by_k.get(k)
        if d is None:                                          # k UNAVAILABLE for this subject -> ask for more
            continue
        r = random_by_k.get(k)
        lcb = _boot(d, cal_clusters, "lower", rng=np.random.default_rng(boot_seed))
        ucb = _boot(d, cal_clusters, "upper", rng=np.random.default_rng(boot_seed + 2))
        rlcb = (_boot(r, cal_clusters, "lower", rng=np.random.default_rng(boot_seed + 1))
                if r is not None else float("-inf"))
        if lcb == lcb and lcb > thr:                           # certified beneficial -> accept (flag specificity)
            spec = "accepted_specific" if (lcb > rlcb) else "accepted_non_specific"
            return {"action": "accept", "k_used": k, "ks_read": list(ks_read), "specificity": spec}
        if ucb == ucb and ucb <= thr:                         # certified NOT beneficial -> stop, abstain
            return {"action": "abstain", "k_used": k, "ks_read": list(ks_read), "specificity": None}
        # else: inconclusive -> request more labels (advance to next k)
    return {"action": "abstain", "k_used": sorted(k_grid)[-1] if k_grid else None,
            "ks_read": list(ks_read), "specificity": None}     # budget exhausted without certification


# ============================ locked real-EEG executor (never reached at this stage) ============================
def execute_real(cfg):
    """Would load real EEG dumps, inject the worlds, fit erasers on SOURCE, split target into calibration/audit,
    compute decisions via the pure functions above, and score audit separately. UNREACHABLE while
    runs_allowed=false (run_cli halts first); guarded here too (defense in depth). Delegates all leak-sensitive
    logic to the tested pure functions -- no bespoke label handling."""
    if not cfg.get("runs_allowed", False) or not cfg.get("experiments_allowed", False):
        return 1, HALT_MSG                                     # belt-and-suspenders: never run while locked
    # (intentionally not wired to real dumps at this stage; requires the separate PM go to enable + fill in)
    raise RuntimeError("execute_real reached with runs_allowed=true but real-dump wiring is deferred to PM go")


def build_schema(cfg, plan=None, expanded=None):
    """Static split policy + label-access capability matrix + code-path access map + expansion counts + the
    expected output schema of a real run. Declares WHICH code paths may read WHICH labels."""
    sc = cfg["tier1_scope"]
    from collections import Counter
    exp_by_budget = dict(Counter(_family(t["budget"]) for t in expanded)) if expanded is not None else None
    return {
        "split_policy": {
            "stratified_by_class": True,
            "repeats_R": sc["repeats_R"],
            "calibration_used_by": ["B2/B3 gate benefit"],
            "audit_used_by": ["final evaluation ONLY"],
            "k_unavailable_policy": "mark UNAVAILABLE; never reuse audit labels; do not silently shrink k",
            "b3_granularity": "one SEQUENCE task per split_id (reveals k=1..16 with early stop)",
            "b4_granularity": "one diagnostic task per (subject); uses all target labels; no cal/audit split",
        },
        "expansion": {
            "plan_rows": len(plan) if plan is not None else None,
            "expanded_tasks": len(expanded) if expanded is not None else None,
            "expanded_by_budget_family": exp_by_budget,
            "rule": "B0/B1/B4 = 1 unit; B2 = k_grid x R; B3 = R sequence tasks; all x world_alpha_grid",
        },
        "label_access": {
            "source_labels": ["eraser_fit", "head_fit", "source_safety", "source_benefit_lcb"],
            "target_calibration_labels": ["B2/B3 calibration_delta_bacc (cal split only)"],
            "target_audit_labels": ["audit_scalar (final evaluation ONLY)"],
            "B4_oracle_labels": ["diagnostic_selector_only"],
        },
        "code_path_access": {
            "compute_decision_row": ["source", "target_calibration (B2/B3)", "unlabeled_target (B1)"],
            "compute_decision_row_forbidden": ["target_audit_labels"],
            "audit_scalar": ["target_audit_labels_ONLY"],
            "calibration_delta_bacc": ["source", "target_calibration"],
        },
        "expected_run_output_schema": {
            "decision_rows": ["dataset", "backbone", "world", "alpha", "intervention", "budget", "k?",
                              "action", "cal_benefit_lcb", "beats_random", "specificity"],
            "audit_reported_separately": ["audit_delta_bacc", "audit_delta_nll"],
            "accounting": ["true_accept_rate", "false_accept_rate", "abstain_rate", "reject_rate",
                           "request_labels_rate", "label_budget_used", "oracle_gap",
                           "accepted_specific", "accepted_non_specific"],
            "b4_excluded_from": ["true_accept_rate", "false_accept_rate"],
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
    plan, expanded = build_plan(cfg), expand_tasks(cfg)
    schema = build_schema(cfg, plan, expanded)
    json.dump({"n_plan_rows": len(plan), "plan": plan}, open("%s/target_info_tier1_plan.json" % OUT, "w"), indent=1)
    json.dump({"n_expanded_tasks": len(expanded), "expanded_tasks": expanded},
              open("%s/target_info_tier1_expanded_tasks.json" % OUT, "w"), indent=1)
    json.dump(schema, open("%s/target_info_tier1_schema.json" % OUT, "w"), indent=1)
    from tos_cmi.eeg.report_target_info_tier1 import write_report
    rpt = write_report(cfg, plan, schema, token, OUT, expanded=expanded)
    print("dry-run: %d plan rows ; %d expanded tasks ; structural gate %s ; report %s"
          % (len(plan), len(expanded), token, rpt))
    print("TARGET_INFO_TIER1_DRYRUN_DONE")
    return len(plan), len(expanded)


def run_cli(argv, cfg):
    """Return (exit_code, message). Deterministic + testable (no sys.exit here)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run and a.execute:                               # reject contradictory intent (R6)
        return 2, "CONFLICTING_FLAGS: --dry-run and --execute are mutually exclusive."
    if a.dry_run:
        n_plan, n_exp = dry_run(cfg)
        return 0, "TARGET_INFO_TIER1_DRYRUN_DONE (%d plan rows, %d expanded tasks)" % (n_plan, n_exp)
    # bare invocation OR --execute: experiments are HARD-LOCKED at this stage.
    if not cfg.get("runs_allowed", False):
        return 1, HALT_MSG
    # (unreachable while runs_allowed=false) the wired executor is gated again inside execute_real.
    return execute_real(cfg)


def main():
    cfg = load_cfg()
    code, msg = run_cli(sys.argv[1:], cfg)
    print(msg)
    sys.exit(code)


if __name__ == "__main__":
    main()
