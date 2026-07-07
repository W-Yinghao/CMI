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
                                            UnlabeledTargetContext, AuditView, LabelAccessGuard,
                                            hash_array, hash_obj, _family, TARGET_LEAK_TOKEN)

CFG = "tos_cmi/eeg/configs/target_info_tier1_smoke_driver_fixed.yaml"
MANIFEST = "tos_cmi/eeg/configs/target_info_tier1_run_manifest.yaml"
OUT = "tos_cmi/results/target_info/tier1_driver_dryrun"
PREFLIGHT_OUT = "tos_cmi/results/target_info/tier1_preflight"
HALT_MSG = "EXPERIMENTS_DISABLED: implementation-only stage; requires separate PM go."
MANIFEST_HALT_MSG = "EXPERIMENTS_DISABLED: run manifest is preflight_only; requires separate PM go."
PREFLIGHT_HALT_MSG = "PREFLIGHT_DISABLED: run manifest preflight_allowed=false; requires separate PM go."
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
    labels ONLY. One scalar per (subject, split). Reads NO audit labels -- refuses an AuditView (P0-2 taint)."""
    if isinstance(cal, AuditView):
        raise TypeError("calibration_delta_bacc received an AuditView; cal deltas must come from CalibrationContext")
    if not isinstance(cal, CalibrationContext):
        raise TypeError("calibration_delta_bacc requires a CalibrationContext, got %s" % type(cal).__name__)
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
                         thr=0.01, boot_seed=0, calibration_idx_hash=None, calibration_label_hash=None):
    """Build the DecisionContext (source + calibration ONLY) and return a decision row. Structurally has NO
    audit input, so a decision is invariant to any audit-label permutation. Records calibration provenance +
    delta_source=calibration_only + decision_input_hash; audit hashes are joined LATER via finalize_decision_row
    (audit phase), so they never influence the decision."""
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
    decision = {"budget": budget, "action": action, "source_safety_pass": bool(source_safety_pass),
                "cal_benefit_lcb": kw.get("cal_benefit_lcb"), "beats_random": kw.get("beats_random"),
                "source_benefit_lcb": kw.get("source_benefit_lcb"),
                "unlabeled_triage": kw.get("unlabeled_triage"), "specificity": specific}
    decision["decision_input_hash"] = hash_obj(decision)       # depends only on source+calibration-derived fields
    decision["calibration_idx_hash"] = calibration_idx_hash
    decision["calibration_label_hash"] = calibration_label_hash
    decision["delta_source"] = "calibration_only"
    # audit_idx_hash / audit_label_hash are added ONLY in finalize_decision_row (audit phase); absent here.
    return decision


def finalize_decision_row(decision_row, audit_idx_hash, audit_label_hash, audit_delta_bacc=None,
                          audit_delta_nll=None):
    """AUDIT phase: attach audit provenance (and the honest held-out metrics) AFTER the decision is frozen. The
    decision-determining fields are untouched; audit hashes/metrics live in a separate sub-record so permuting
    audit labels changes only these, never the decision (see the audit-permutation test)."""
    row = dict(decision_row)
    row["audit_idx_hash"] = audit_idx_hash
    row["audit_label_hash"] = audit_label_hash
    row["audit_metric"] = {"audit_delta_bacc": audit_delta_bacc, "audit_delta_nll": audit_delta_nll}
    return row


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


# ============================ run authorization (P0-1: tamper-proof, not a YAML boolean) ============================
def _git_commit():
    """Best-effort current git HEAD (provenance only; never a hard blocker so tests stay deterministic)."""
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(CFG)) or ".", text=True).strip()
    except Exception:
        return None


def scope_hash(cfg):
    """Hash of the immutable experiment SCOPE (datasets/backbones/worlds/budgets/interventions/thresholds/alpha)."""
    scope = {"tier1_scope": cfg["tier1_scope"], "worlds": cfg["worlds"], "interventions": cfg["interventions"],
             "budgets": cfg["budgets"], "thresholds": cfg.get("thresholds"),
             "world_alpha_grid": cfg.get("world_alpha_grid")}
    return hash_obj(scope)


def load_manifest(path=MANIFEST):
    with open(path) as fh:
        return yaml.safe_load(fh)


def authorize_execution(cfg, manifest, enable_token, driver_cfg_path=CFG):
    """Return (authorized, reasons). Execution requires an APPROVED run manifest + a valid enable token + matching
    scope/driver hashes -- NOT the driver-config booleans. A plain YAML flip of runs_allowed/experiments_allowed in
    the DRIVER config cannot authorize a run, because this keys on the MANIFEST (P0-1)."""
    reasons = []
    if manifest.get("run_status") != "approved":
        reasons.append("manifest.run_status=%r != 'approved'" % manifest.get("run_status"))
    if not manifest.get("runs_allowed", False):
        reasons.append("manifest.runs_allowed is false")
    if not manifest.get("experiments_allowed", False):
        reasons.append("manifest.experiments_allowed is false")
    if manifest.get("enable_token_required", True):
        appr = manifest.get("approved_enable_token_sha256")
        got = hashlib.sha256(enable_token.encode()).hexdigest() if enable_token else None
        if not appr or got != appr:
            reasons.append("enable token missing/invalid")
    if manifest.get("approved_scope_hash") != scope_hash(cfg):
        reasons.append("scope_hash mismatch")
    try:
        cur_driver = hashlib.sha256(open(driver_cfg_path, "rb").read()).hexdigest()[:16]
    except Exception:
        cur_driver = None
    if manifest.get("approved_driver_hash") != cur_driver:
        reasons.append("driver_config_hash mismatch")
    appr_git = manifest.get("approved_git_commit")
    if appr_git not in (None, "ANY") and appr_git != _git_commit():   # git pin is optional (soft)
        reasons.append("git_commit mismatch")
    return (len(reasons) == 0, reasons)


# ============================ real-split PREFLIGHT (P0-3: split/schema only, NEVER metrics) ============================
def _preflight_from_labels(y, k_grid, R, seed):
    """Pure preflight over a target subject's labels: stratified cal/audit splits, per-class counts, k-availability,
    index/label hashes, disjointness. Produces NO gate action, NO ΔbAcc, NO performance metric. Testable on dummy
    arrays."""
    y = np.asarray(y).astype(int)
    splits = make_calibration_audit_splits(y, R=R, seed=seed)
    target_leak_structural_check(splits, {})                   # disjointness gate (empty budget_specs = idx-only)
    rows, unavailable = [], []
    for sid, (cal, aud) in enumerate(splits, 1):
        cal_h, aud_h = hash_array(cal), hash_array(aud)
        for k in k_grid:
            sel, status, eff = select_k_per_class(cal, y, k, seed)
            if status == "UNAVAILABLE":
                unavailable.append({"split_id": sid, "k": k, "eff_n_per_class": eff})
            else:                                              # selected k come ONLY from calibration (disjoint aud)
                if not set(sel.tolist()).isdisjoint(aud.tolist()):
                    raise AssertionError("preflight split %d k %d: selected k overlaps audit" % (sid, k))
            rows.append({"split_id": sid, "k": int(k), "status": status, "eff_n_per_class": eff,
                         "calibration_idx_hash": cal_h, "audit_idx_hash": aud_h,
                         "calibration_label_hash": hash_array(y[cal]), "audit_label_hash": hash_array(y[aud])})
    return {"n_splits": len(splits), "k_grid": list(k_grid), "rows": rows, "unavailable_k": unavailable}


FROZEN_ROOT = "tos_cmi/results/tos_cmi_eeg_frozen"


def preflight_real_splits(cfg, manifest, results_root=FROZEN_ROOT):
    """CLI preflight: load real frozen-dump TARGET LABELS ONLY, build stratified calibration/audit splits,
    per-class counts, k-availability, index/label hashes, disjointness. Writes split-only tables + report. Fits NO
    eraser, computes NO ΔbAcc / NLL / gate action / accept-reject / any performance metric. Splits are computed on
    the real target task labels and are WORLD-INDEPENDENT (both worlds share the same target labels; worlds differ
    only in injected FEATURES, which preflight never reads), so they are reported once per (dataset, fold)."""
    import glob
    import re
    import csv
    # hard guards -- preflight may NEVER run an experiment, even by manifest tampering
    if manifest.get("runs_allowed") or manifest.get("experiments_allowed"):
        return 1, "PREFLIGHT_REFUSED: manifest runs_allowed/experiments_allowed must be false"
    if manifest.get("allowed_command") != "preflight_real_splits_only":
        return 1, "PREFLIGHT_REFUSED: manifest.allowed_command != preflight_real_splits_only"
    if manifest.get("approved_scope_hash") not in (None, "ANY") and manifest["approved_scope_hash"] != scope_hash(cfg):
        return 1, "PREFLIGHT_REFUSED: scope_hash mismatch"
    drv = hashlib.sha256(open(CFG, "rb").read()).hexdigest()[:16]
    if manifest.get("approved_driver_hash") not in (None, "ANY") and manifest["approved_driver_hash"] != drv:
        return 1, "PREFLIGHT_REFUSED: driver_hash mismatch"

    sc = cfg["tier1_scope"]
    folds, R, seed = _folds(sc["folds"]), sc["repeats_R"], sc["seeds"][0]
    k_grid = cfg["budgets"]["B2_k_labels_per_class"]["k_grid"]
    rows, split_hash_rows, unavail_rows = [], [], []
    overlap_total, n_dumps = 0, 0
    for ds in sc["datasets"]:
        for bb in sc["backbones"]:
            dd = "%s/%s_%s_LOSO" % (results_root, ds, bb)
            ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                        key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))[:len(folds)]
            for fold, p in zip(folds, ps):
                sub = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
                yt = np.load(p, allow_pickle=True)["y_target"].astype(int)   # TARGET LABELS ONLY (no features)
                n_dumps += 1
                splits = make_calibration_audit_splits(yt, R=R, seed=seed)
                target_leak_structural_check(splits, {})                    # disjointness gate (raises on overlap)
                classes = sorted(set(yt.tolist()))
                for sid, (cal, aud) in enumerate(splits, 1):
                    overlap = len(set(cal.tolist()) & set(aud.tolist())); overlap_total += overlap
                    cal_ih, aud_ih = hash_array(cal), hash_array(aud)
                    cal_lh, aud_lh = hash_array(yt[cal]), hash_array(yt[aud])
                    split_hash_rows.append({"dataset": ds, "backbone": bb, "seed": seed, "fold": fold,
                                            "target_subject": sub, "split_id": sid, "n_calibration": int(len(cal)),
                                            "n_audit": int(len(aud)), "calibration_idx_hash": cal_ih,
                                            "audit_idx_hash": aud_ih, "calibration_label_hash": cal_lh,
                                            "audit_label_hash": aud_lh, "disjoint": overlap == 0})
                    for k in k_grid:
                        for c in classes:
                            n_tot = int((yt == c).sum()); n_cal = int((yt[cal] == c).sum())
                            n_aud = int((yt[aud] == c).sum()); avail = n_cal >= k
                            rows.append({"dataset": ds, "backbone": bb, "seed": seed, "fold": fold,
                                         "target_subject": sub, "split_id": sid, "k": int(k), "class": int(c),
                                         "n_target_total": n_tot, "n_calibration": n_cal, "n_audit": n_aud,
                                         "k_available": bool(avail),
                                         "unavailable_reason": "" if avail else "insufficient_calibration_class_%d" % c,
                                         "calibration_idx_hash": cal_ih, "audit_idx_hash": aud_ih,
                                         "calibration_label_hash": cal_lh, "audit_label_hash": aud_lh})
                        mincal = int(min((yt[cal] == c).sum() for c in classes))
                        if mincal < k:
                            unavail_rows.append({"dataset": ds, "fold": fold, "split_id": sid, "k": int(k),
                                                 "min_calibration_per_class": mincal})
    os.makedirs(PREFLIGHT_OUT, exist_ok=True)
    with open("%s/split_hashes.csv" % PREFLIGHT_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(split_hash_rows[0].keys())); w.writeheader(); w.writerows(split_hash_rows)
    with open("%s/unavailable_k_table.csv" % PREFLIGHT_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "fold", "split_id", "k", "min_calibration_per_class"])
        w.writeheader(); w.writerows(unavail_rows)
    manifest_out = {"purpose": "split_hash_unavailable_k_only", "no_metrics_emitted": True,
                    "datasets": sc["datasets"], "backbones": sc["backbones"], "seed": seed, "folds": folds,
                    "n_dumps": n_dumps, "R": R, "k_grid": k_grid, "n_split_rows": len(split_hash_rows),
                    "n_schema_rows": len(rows), "n_unavailable_k": len(unavail_rows),
                    "calibration_audit_overlap_total": overlap_total, "rows": rows}
    json.dump(manifest_out, open("%s/real_split_preflight_manifest.json" % PREFLIGHT_OUT, "w"), indent=1)
    from tos_cmi.eeg.report_target_info_tier1 import write_preflight_report
    rpt = write_preflight_report(manifest_out, split_hash_rows, unavail_rows, PREFLIGHT_OUT)
    print("preflight: %d dumps, %d splits, overlap=%d, unavailable-k=%d ; report %s"
          % (n_dumps, len(split_hash_rows), overlap_total, len(unavail_rows), rpt))
    print("TARGET_INFO_TIER1_PREFLIGHT_DONE")
    return 0, "TARGET_INFO_TIER1_PREFLIGHT_DONE (%d dumps, %d splits, overlap=%d)" % (
        n_dumps, len(split_hash_rows), overlap_total)


# ============================ locked real-EEG executor (never reached at this stage) ============================
def execute_real(cfg, manifest=None):
    """Would load real EEG dumps, inject the worlds, fit erasers on SOURCE, split target into calibration/audit,
    compute decisions via the pure functions above, and score audit separately. Reached only after
    authorize_execution passes (never at this stage); guarded here too (defense in depth). Delegates all
    leak-sensitive logic to the tested pure functions -- no bespoke label handling."""
    if not (manifest or {}).get("runs_allowed", False) or not (manifest or {}).get("experiments_allowed", False):
        return 1, MANIFEST_HALT_MSG                            # belt-and-suspenders: never run while locked
    # (intentionally not wired to real dumps at this stage; requires the separate PM go to enable + fill in)
    raise RuntimeError("execute_real reached with an approved manifest but real-dump wiring is deferred to PM go")


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
    ap.add_argument("--preflight-real-splits", action="store_true")
    ap.add_argument("--run-manifest", default=MANIFEST)
    ap.add_argument("--enable-token", default=None)
    a = ap.parse_args(argv)
    if sum([a.dry_run, a.execute, a.preflight_real_splits]) > 1:   # exactly one mode
        return 2, "CONFLICTING_FLAGS: choose one of --dry-run / --execute / --preflight-real-splits."
    if a.dry_run:
        n_plan, n_exp = dry_run(cfg)
        return 0, "TARGET_INFO_TIER1_DRYRUN_DONE (%d plan rows, %d expanded tasks)" % (n_plan, n_exp)
    if a.preflight_real_splits:
        manifest = load_manifest(a.run_manifest)
        if not manifest.get("preflight_allowed", False):          # gated: running preflight = separate PM go
            return 1, PREFLIGHT_HALT_MSG
        return preflight_real_splits(cfg, manifest)
    # bare invocation OR --execute: experiments require an APPROVED, token-backed manifest (P0-1/P0-4).
    manifest = load_manifest(a.run_manifest)
    ok, reasons = authorize_execution(cfg, manifest, a.enable_token)
    if not ok:
        return 1, MANIFEST_HALT_MSG                               # [blocked by: %s] omitted from the fixed message
    return execute_real(cfg, manifest)                            # only if fully authorized (never at this stage)


def main():
    cfg = load_cfg()
    code, msg = run_cli(sys.argv[1:], cfg)
    print(msg)
    sys.exit(code)


if __name__ == "__main__":
    main()
