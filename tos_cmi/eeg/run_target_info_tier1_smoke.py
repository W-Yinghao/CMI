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
                                            subject_seeded_splits, nested_k_subsets, SPLIT_RNG_SCHEME,
                                            effective_n_per_class,
                                            target_leak_structural_check, budget_action, b1_triage_action,
                                            DecisionContext, SourceContext, CalibrationContext,
                                            UnlabeledTargetContext, AuditView, LabelAccessGuard,
                                            hash_array, hash_obj, _family, TARGET_LEAK_TOKEN)

CFG = "tos_cmi/eeg/configs/target_info_tier1_smoke_driver_fixed.yaml"
MANIFEST = "tos_cmi/eeg/configs/target_info_tier1_run_manifest.yaml"
PROVIDER_VAL_MANIFEST = "tos_cmi/eeg/configs/target_info_tier1_provider_validation_manifest_TEMPLATE.yaml"
OUT = "tos_cmi/results/target_info/tier1_driver_dryrun"
PREFLIGHT_OUT = "tos_cmi/results/target_info/tier1_preflight_subject_seeded"
PROVIDER_VAL_OUT = "tos_cmi/results/target_info/tier1_provider_validation"
HALT_MSG = "EXPERIMENTS_DISABLED: implementation-only stage; requires separate PM go."
MANIFEST_HALT_MSG = "EXPERIMENTS_DISABLED: run manifest is preflight_only; requires separate PM go."
PREFLIGHT_HALT_MSG = "PREFLIGHT_DISABLED: run manifest preflight_allowed=false; requires separate PM go."
PROVIDER_VAL_HALT_MSG = "PROVIDER_VALIDATION_DISABLED: manifest provider_validation_allowed=false; requires separate PM go."
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
                         thr=0.01, boot_seed=0, calibration_idx_hash=None, calibration_label_hash=None,
                         cal_benefit_lcb=None, beats_random=None):
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
        if cal_benefit_lcb is not None or beats_random is not None:   # hardened bounded-LCB path (direct)
            kw["cal_benefit_lcb"] = cal_benefit_lcb
            kw["beats_random"] = bool(beats_random)
        else:                                                  # legacy bootstrap path (unit tests)
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


def b3_sequential_decision_bounded(k_grid, lcb_by_k, rlcb_by_k, thr):
    """Hardened B3: reveal k ascending; accept ONLY when the conservative bounded calibration LCB > thr AND the
    same-k random control does not also clear it. NaN LCB (underpowered, e.g. k=1) -> request more labels (advance).
    There is no bounded UCB to 'certify not-beneficial', so B3 either accepts or abstains -- it never accepts at an
    underpowered k. Returns action / k_used / ks_read / specificity / lcb_used."""
    ks_read = []
    for k in sorted(k_grid):
        ks_read.append(k)
        lcb = lcb_by_k.get(k)
        if lcb is None or lcb != lcb:                          # underpowered -> request more labels
            continue
        rlcb = rlcb_by_k.get(k)
        specific = (rlcb is None or rlcb != rlcb or lcb > rlcb)
        if lcb > thr and specific:
            return {"action": "accept", "k_used": k, "ks_read": list(ks_read),
                    "specificity": "accepted_specific" if specific else "accepted_non_specific", "lcb_used": lcb}
    return {"action": "abstain", "k_used": sorted(k_grid)[-1] if k_grid else None, "ks_read": list(ks_read),
            "specificity": None, "lcb_used": None}


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
    """Hash of the immutable experiment SCOPE (datasets/backbones/worlds/budgets/interventions/thresholds/alpha/
    split_rng)."""
    scope = {"tier1_scope": cfg["tier1_scope"], "worlds": cfg["worlds"], "interventions": cfg["interventions"],
             "budgets": cfg["budgets"], "thresholds": cfg.get("thresholds"),
             "world_alpha_grid": cfg.get("world_alpha_grid"), "split_rng": cfg.get("split_rng"),
             "target_calibration_lcb": cfg.get("target_calibration_lcb")}
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
    srng = cfg.get("split_rng", {})
    if srng.get("scheme") != SPLIT_RNG_SCHEME:
        return 1, "PREFLIGHT_REFUSED: split_rng.scheme != %s" % SPLIT_RNG_SCHEME
    gseed = srng["global_split_seed"]; cfrac = srng.get("calib_fraction", 0.5)
    rows, split_hash_rows, unavail_rows = [], [], []
    overlap_total, n_dumps, nested_ok_total, nested_checked = 0, 0, 0, 0
    diversity = []
    for ds in sc["datasets"]:
        for bb in sc["backbones"]:
            dd = "%s/%s_%s_LOSO" % (results_root, ds, bb)
            ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                        key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))[:len(folds)]
            for fold, p in zip(folds, ps):
                sub = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
                yt = np.load(p, allow_pickle=True)["y_target"].astype(int)   # TARGET LABELS ONLY (no features)
                n_dumps += 1
                seed_key = {"dataset": ds, "backbone": bb, "model_seed": seed, "fold": fold,
                            "target_subject": sub, "global_split_seed": gseed}
                splits = subject_seeded_splits(yt, seed_key, R=R, calib_fraction=cfrac)
                target_leak_structural_check(splits, {})                    # disjointness gate (raises on overlap)
                classes = sorted(set(yt.tolist()))
                cal_hashes = set()
                for sid, (cal, aud) in enumerate(splits, 1):
                    overlap = len(set(cal.tolist()) & set(aud.tolist())); overlap_total += overlap
                    cal_ih, aud_ih = hash_array(cal), hash_array(aud); cal_hashes.add(cal_ih)
                    cal_lh, aud_lh = hash_array(yt[cal]), hash_array(yt[aud])
                    pool_h = hash_obj({"cal_idx_hash": cal_ih, "n": int(len(cal))})
                    ksub, _order = nested_k_subsets(cal, yt, k_grid, seed_key, sid)
                    # nested check: k=1 ⊂ k=2 ⊂ ... (prefix subsets of a fixed order) and all ⊂ calibration pool
                    prev = None
                    for k in sorted(k_grid):
                        sel = ksub[k]
                        if sel is not None:
                            nested_checked += 1
                            in_pool = set(sel.tolist()).issubset(cal.tolist())
                            nests = (prev is None) or set(prev.tolist()).issubset(sel.tolist())
                            if in_pool and nests:
                                nested_ok_total += 1
                            prev = sel
                    split_hash_rows.append({"dataset": ds, "backbone": bb, "seed": seed, "fold": fold,
                                            "target_subject": sub, "split_id": sid, "n_calibration": int(len(cal)),
                                            "n_audit": int(len(aud)), "calibration_idx_hash": cal_ih,
                                            "audit_idx_hash": aud_ih, "calibration_label_hash": cal_lh,
                                            "audit_label_hash": aud_lh, "calibration_pool_hash": pool_h,
                                            "disjoint": overlap == 0})
                    for k in k_grid:
                        sel = ksub[k]
                        ksub_h = hash_array(sel) if sel is not None else "UNAVAILABLE"
                        for c in classes:
                            n_tot = int((yt == c).sum()); n_cal = int((yt[cal] == c).sum())
                            n_aud = int((yt[aud] == c).sum()); avail = sel is not None and n_cal >= k
                            rows.append({"dataset": ds, "backbone": bb, "seed": seed, "fold": fold,
                                         "target_subject": sub, "split_id": sid, "k": int(k), "class": int(c),
                                         "n_target_total": n_tot, "n_calibration": n_cal, "n_audit": n_aud,
                                         "k_available": bool(avail),
                                         "unavailable_reason": "" if avail else "insufficient_calibration_class_%d" % c,
                                         "calibration_idx_hash": cal_ih, "audit_idx_hash": aud_ih,
                                         "calibration_label_hash": cal_lh, "audit_label_hash": aud_lh,
                                         "k_subset_hash": ksub_h, "calibration_pool_hash": pool_h})
                        mincal = int(min((yt[cal] == c).sum() for c in classes))
                        if ksub[k] is None or mincal < k:
                            unavail_rows.append({"dataset": ds, "fold": fold, "split_id": sid, "k": int(k),
                                                 "min_calibration_per_class": mincal})
                diversity.append({"dataset": ds, "fold": fold, "target_subject": sub,
                                  "distinct_calibration_splits": len(cal_hashes), "R": R})
    os.makedirs(PREFLIGHT_OUT, exist_ok=True)
    with open("%s/split_hashes.csv" % PREFLIGHT_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(split_hash_rows[0].keys())); w.writeheader(); w.writerows(split_hash_rows)
    with open("%s/unavailable_k_table.csv" % PREFLIGHT_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "fold", "split_id", "k", "min_calibration_per_class"])
        w.writeheader(); w.writerows(unavail_rows)
    manifest_out = {"purpose": "split_hash_unavailable_k_only", "no_metrics_emitted": True,
                    "split_rng_scheme": SPLIT_RNG_SCHEME, "global_split_seed": gseed, "calib_fraction": cfrac,
                    "datasets": sc["datasets"], "backbones": sc["backbones"], "seed": seed, "folds": folds,
                    "n_dumps": n_dumps, "R": R, "k_grid": k_grid, "n_split_rows": len(split_hash_rows),
                    "n_schema_rows": len(rows), "n_unavailable_k": len(unavail_rows),
                    "calibration_audit_overlap_total": overlap_total,
                    "nested_k_checks_passed": "%d/%d" % (nested_ok_total, nested_checked),
                    "per_subject_split_diversity": diversity, "rows": rows}
    json.dump(manifest_out, open("%s/real_split_preflight_manifest.json" % PREFLIGHT_OUT, "w"), indent=1)
    from tos_cmi.eeg.report_target_info_tier1 import write_preflight_report
    rpt = write_preflight_report(manifest_out, split_hash_rows, unavail_rows, PREFLIGHT_OUT)
    print("preflight: %d dumps, %d splits, overlap=%d, unavailable-k=%d ; report %s"
          % (n_dumps, len(split_hash_rows), overlap_total, len(unavail_rows), rpt))
    print("TARGET_INFO_TIER1_PREFLIGHT_DONE")
    return 0, "TARGET_INFO_TIER1_PREFLIGHT_DONE (%d dumps, %d splits, overlap=%d)" % (
        n_dumps, len(split_hash_rows), overlap_total)


# ============================ per-task executor core (tested on dummy arrays; array-only, world-agnostic) =========
def _boot_delta_samples(eraser, src: SourceContext, cal: CalibrationContext, n_boot, seed):
    """Bootstrap ΔbAcc_cal samples over the CALIBRATION trials (labels from CalibrationContext only). Refuses an
    AuditView. Returns a list of ΔbAcc(erased)-ΔbAcc(full) values (degenerate single-class resamples dropped)."""
    if isinstance(cal, AuditView):
        raise TypeError("_boot_delta_samples received an AuditView; calibration deltas must come from calibration")
    _bacc, _ = _lazy_estimators()
    rng = np.random.default_rng(seed)
    n = len(cal.yt_cal)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yc = cal.yt_cal[idx]
        if len(set(yc.tolist())) < 2:
            continue
        Zc = cal.Zt_cal[idx]
        out.append(float(_bacc(eraser(src.Zs), src.ys, eraser(Zc), yc) - _bacc(src.Zs, src.ys, Zc, yc)))
    return out


def stratified_bounded_lcb(h_full, h_eras, eraser, cal: CalibrationContext, confidence=0.95, n_candidate=1,
                           drange=2.0):
    """Conservative FINITE-SAMPLE lower bound on the balanced ΔbAcc_cal (NOT a bootstrap, NOT a point estimate).
    Per class, the paired per-trial difference d_i = 1{erased correct} - 1{full correct} in {-1,0,1}; a
    Maurer-Pontil empirical-Bernstein lower bound on E[d] (range `drange`=2 for [-1,1]) with Bonferroni delta over
    (classes x candidate deployable interventions); the balanced LCB is the mean of the per-class LCBs. An
    underpowered class (n<2) -> NaN (=> the gate abstains). Reads calibration labels ONLY."""
    import math
    if isinstance(cal, AuditView):
        raise TypeError("stratified_bounded_lcb received an AuditView; calibration LCB must come from calibration")
    yc = np.asarray(cal.yt_cal).astype(int)
    pf = h_full.predict(cal.Zt_cal)
    pe = h_eras.predict(eraser(cal.Zt_cal))
    classes = sorted(set(yc.tolist()))
    delta = (1.0 - confidence) / max(1, len(classes) * max(1, n_candidate))   # Bonferroni
    per = []
    for c in classes:
        m = (yc == c)
        n = int(m.sum())
        if n < 2:                                              # underpowered -> abstain (never point estimate)
            return float("nan")
        d = (pe[m] == yc[m]).astype(float) - (pf[m] == yc[m]).astype(float)
        mean = float(d.mean()); v = float(d.var(ddof=1))
        eb = math.sqrt(2 * v * math.log(2 / delta) / n) + 7 * drange * math.log(2 / delta) / (3 * (n - 1))
        per.append(mean - eb)
    return float(np.mean(per))


def _delta_boot_fitfree(h_full, h_eras, eraser, cal: CalibrationContext, n_boot, seed):
    """ΔbAcc_cal bootstrap samples using PRE-FIT source heads (the source head is identical across cal resamples,
    so fit once then bootstrap the calibration eval only). Numerically identical to fitting per bootstrap, but
    ~n_boot x faster. Reads calibration labels ONLY; refuses an AuditView."""
    if isinstance(cal, AuditView):
        raise TypeError("_delta_boot_fitfree received an AuditView; calibration deltas must come from calibration")
    from sklearn.metrics import balanced_accuracy_score
    yc = np.asarray(cal.yt_cal).astype(int)
    pf = h_full.predict(cal.Zt_cal)                            # predictions fixed given the fixed source head
    pe = h_eras.predict(eraser(cal.Zt_cal))
    rng = np.random.default_rng(seed)
    n = len(yc)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(yc[idx].tolist())) < 2:
            continue
        out.append(float(balanced_accuracy_score(yc[idx], pe[idx]) - balanced_accuracy_score(yc[idx], pf[idx])))
    return out


def source_task_drop_ucb(eraser, src: SourceContext, safety_eps, n_boot=100, seed=0, min_clusters=3, q=0.95):
    """Source-only task-safety as a CLUSTER-BOOTSTRAP UPPER CONFIDENCE BOUND (NOT a point estimate). Cluster unit =
    source subject (`src.subj`); if subjects are unavailable, a documented trial-block fallback is used. With too
    few clusters the safety is UNDERPOWERED -> ABSTAIN (never a point-estimate accept). Returns a dict with
    task_drop_mean, task_drop_ucb, cluster_unit, n_clusters, ci_method, safety_status, reason, pass."""
    _bacc, _ = _lazy_estimators()
    from tos_cmi.eeg.run_v2_certificate import _stratified_split
    if src.subj is not None:
        cluster_unit, groups = "source_subject", np.asarray(src.subj)
    else:                                                       # documented fallback (NOT subject-level)
        cluster_unit, groups = "trial_block", (np.arange(len(src.ys)) % max(min_clusters, 4))
    clusters = sorted(set(groups.tolist()))
    n_clusters = len(clusters)
    base = {"cluster_unit": cluster_unit, "n_clusters": n_clusters,
            "ci_method": "cluster_bootstrap_q%.2f" % q}
    if n_clusters < min_clusters:                               # underpowered -> refuse, never fall back to point est
        return {**base, "task_drop_mean": float("nan"), "task_drop_ucb": float("nan"),
                "safety_status": "ABSTAIN", "reason": "underpowered_safety_ucb", "pass": False}
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=n_clusters, replace=True)
        idx = np.concatenate([np.where(groups == c)[0] for c in pick])
        Zb, yb = src.Zs[idx], src.ys[idx]
        if len(set(yb.tolist())) < 2:
            continue
        A, B = _stratified_split(len(yb), seed)
        drops.append(float(_bacc(Zb[A], yb[A], Zb[B], yb[B]) - _bacc(eraser(Zb[A]), yb[A], eraser(Zb[B]), yb[B])))
    if not drops:
        return {**base, "task_drop_mean": float("nan"), "task_drop_ucb": float("nan"),
                "safety_status": "ABSTAIN", "reason": "underpowered_safety_ucb", "pass": False}
    mean, ucb = float(np.mean(drops)), float(np.quantile(drops, q))
    return {**base, "task_drop_mean": mean, "task_drop_ucb": ucb, "safety_status": "OK", "reason": "",
            "pass": bool(ucb == ucb and ucb <= safety_eps)}     # ACCEPT-gate keys on the UCB, never the mean


def _source_benefit_samples(eraser, src: SourceContext, n_boot, seed):
    """B0 source-only benefit: bootstrap ΔbAcc(erased-full) on a held-out SOURCE split (no target labels).
    (`_stratified_split` returns boolean masks -> select rows FIRST, then bootstrap over the selected count.)"""
    _bacc, _ = _lazy_estimators()
    from tos_cmi.eeg.run_v2_certificate import _stratified_split
    A, B = _stratified_split(len(src.ys), seed)
    ZA, yA = src.Zs[A], src.ys[A]
    ZB, yB = src.Zs[B], src.ys[B]
    rng = np.random.default_rng(seed + 5)
    out = []
    nb = len(yB)
    for _ in range(n_boot):
        j = rng.integers(0, nb, nb)
        Zb, yb = ZB[j], yB[j]
        if len(set(yb.tolist())) < 2:
            continue
        out.append(float(_bacc(eraser(ZA), yA, eraser(Zb), yb) - _bacc(ZA, yA, Zb, yb)))
    return out


def execute_task(task, Zs, ys, z_src, Zt, yt, seed_key, eraser, eraser_random, cfg, n_boot=200, source_subj=None,
                 precomputed_safety=None):
    """One (dataset,world,fold,alpha,intervention,budget) task -> (decision_rows, audit_rows). Decision phase reads
    SOURCE + CALIBRATION only via guarded contexts (audit labels are structurally absent from compute_decision_row);
    audit phase runs AFTER the decision is frozen. Subject-seeded splits + nested k-subsets are replayed here.
    Source safety is a cluster-bootstrap UCB (never a point estimate). Array-only + world-agnostic (worlds are
    injected by execute_real), so this is unit-tested on dummy arrays."""
    fam = _family(task["budget"])
    thr = cfg["thresholds"]["benefit_lcb"]; safety_eps = cfg["thresholds"]["safety_eps"]
    R = cfg["tier1_scope"]["repeats_R"]; k_grid = cfg["budgets"]["B2_k_labels_per_class"]["k_grid"]
    cfrac = cfg["split_rng"]["calib_fraction"]
    lcb_cfg = cfg.get("target_calibration_lcb", {})
    conf = lcb_cfg.get("confidence", 0.95)
    n_candidate = max(1, len(cfg.get("interventions", [1])))   # familywise correction over candidate interventions
    src = SourceContext(np.asarray(Zs), np.asarray(ys).astype(int), np.asarray(z_src),
                        int(len(set(ys.tolist()))), subj=source_subj)
    yt = np.asarray(yt).astype(int); Zt = np.asarray(Zt)
    splits = subject_seeded_splits(yt, seed_key, R=R, calib_fraction=cfrac)
    target_leak_structural_check(splits, {})                   # per-split disjointness (raises -> caller HALTs run)
    _ = LabelAccessGuard("decision")                           # decision-phase guard (audit labels forbidden here)
    saf = precomputed_safety if precomputed_safety is not None else source_task_drop_ucb(eraser, src, safety_eps)
    src_pass = saf["pass"]
    safety_status = "pass" if src_pass else ("abstain_underpowered" if saf["safety_status"] == "ABSTAIN"
                                             else "reject")
    meta = {kk: task.get(kk) for kk in ("dataset", "backbone", "world", "alpha", "fold", "target_subject",
                                        "intervention", "budget")}
    dec_rows, aud_rows = [], []
    heads = None                                               # fit source heads ONCE per task (B2/B3); reuse across splits/k
    if fam in ("B2", "B3"):
        from sklearn.linear_model import LogisticRegression
        heads = {"full": LogisticRegression(max_iter=200, C=1.0).fit(src.Zs, src.ys),
                 "eras": LogisticRegression(max_iter=200, C=1.0).fit(eraser(src.Zs), src.ys),
                 "rand": LogisticRegression(max_iter=200, C=1.0).fit(eraser_random(src.Zs), src.ys)}

    def _nn(x):
        return None if (x is None or x != x) else x           # NaN/None -> None (clean rows; deterministic equality)

    def _emit(row, sid, cal_idx, aud_idx, k):
        """Attach split meta to the (audit-free) decision row and, in a SEPARATE audit-phase record, the held-out
        audit provenance + metric. decision_rows never carry an audit metric."""
        dec = {**meta, "split_id": sid, "k": k, "source_safety_status": safety_status,
               "domain_gain": None, "same_k_random_calibration": row.get("beats_random"),
               "calibration_benefit_lcb": _nn(row.get("cal_benefit_lcb")), "decision_action": row["action"],
               "decision_input_hash": row["decision_input_hash"], "calibration_idx_hash": row["calibration_idx_hash"],
               "calibration_label_hash": row["calibration_label_hash"],
               "audit_idx_hash_hash_only": hash_obj(hash_array(aud_idx)), "specificity": row.get("specificity")}
        dec_rows.append(dec)
        gate_a = LabelAccessGuard("audit")                     # audit labels readable ONLY now (post-decision)
        av = AuditView(np.asarray(aud_idx), gate_a.audit_labels(AuditView(np.asarray(aud_idx), yt[aud_idx])))
        # B4 oracle: compute the diagnostic audit ΔbAcc even though it is not a deployable accept (fixes the v0 bug)
        adb = (audit_scalar(eraser, src, av, Zt[aud_idx]) if (row["action"] == "accept" or fam == "B4") else None)
        adr = (audit_scalar(eraser_random, src, av, Zt[aud_idx]) if row["action"] == "accept" else None)
        aud_rows.append({**meta, "split_id": sid, "k": k, "decision_input_hash": row["decision_input_hash"],
                         "audit_idx_hash": hash_array(aud_idx), "audit_label_hash": hash_array(yt[aud_idx]),
                         "audit_delta_bacc": adb, "audit_delta_bacc_random": adr, "audit_delta_nll": None,
                         "specificity_flag": row.get("specificity")})

    for sid, (cal, aud) in enumerate(splits, 1):
        if fam == "B0":
            samp = _source_benefit_samples(eraser, src, n_boot, seed=sid)
            row = compute_decision_row(task["budget"], src_pass, None, None, None, None, None, thr=thr,
                                       calibration_idx_hash=hash_array(cal), calibration_label_hash="B0_source_only")
            # B0 accepts on source benefit LCB (no target labels): recompute action via source samples
            _, _boot = _lazy_estimators()
            lcb = _boot(samp, list(range(len(samp))), "lower", rng=np.random.default_rng(0)) if samp else float("nan")
            row = compute_decision_row(task["budget"], src_pass, lcb, None, None, None, None, thr=thr,
                                       calibration_idx_hash="B0_no_target", calibration_label_hash="B0_no_target")
            _emit(row, sid, cal, aud, None)
            break                                              # B0 is source-only: one decision, no per-split loop
        if fam == "B1":
            ms = unlabeled_mismatch(src, UnlabeledTargetContext(Zt))
            row = compute_decision_row(task["budget"], src_pass, None, None, None, None, ms, thr=thr,
                                       calibration_idx_hash="B1_unlabeled", calibration_label_hash="B1_unlabeled")
            _emit(row, sid, cal, aud, None)
            break                                              # B1 triage: one decision
        if fam == "B4":
            row = compute_decision_row(task["budget"], src_pass, None, None, None, None, None, thr=thr,
                                       calibration_idx_hash="B4_oracle", calibration_label_hash="B4_oracle")
            _emit(row, sid, cal, aud, None)                    # DIAGNOSTIC; excluded from deployable accounting
            break
        # B2 / B3: nested k-subset calibration
        ksub, _order = nested_k_subsets(cal, yt, k_grid, seed_key, sid)
        if fam == "B2":
            for k in k_grid:
                sel = ksub[k]
                if sel is None:                                # UNAVAILABLE: mark, never reuse audit
                    dec_rows.append({**meta, "split_id": sid, "k": int(k), "source_safety_status": safety_status, "decision_action": "unavailable_k",
                                     "calibration_benefit_lcb": None, "same_k_random_calibration": None,
                                     "domain_gain": None, "decision_input_hash": None,
                                     "calibration_idx_hash": hash_array(cal), "calibration_label_hash": None,
                                     "audit_idx_hash_hash_only": hash_obj(hash_array(aud)), "specificity": None})
                    continue
                cal_ctx = CalibrationContext(Zt[sel], yt[sel], effective_n_per_class(sel, yt))
                lcb = stratified_bounded_lcb(heads["full"], heads["eras"], eraser, cal_ctx, conf, n_candidate)
                rlcb = stratified_bounded_lcb(heads["full"], heads["rand"], eraser_random, cal_ctx, conf, n_candidate)
                beats = bool(lcb == lcb and (rlcb != rlcb or lcb > rlcb))   # random doesn't also clear a bounded LCB
                row = compute_decision_row(task["budget"], src_pass, None, None, None, None, None, thr=thr,
                                           calibration_idx_hash=hash_array(sel), calibration_label_hash=hash_array(yt[sel]),
                                           cal_benefit_lcb=lcb, beats_random=beats)
                _emit(row, sid, cal, aud, int(k))
        else:                                                  # B3 sequential over the nested k-grid (bounded LCB)
            lcb_by_k, rlcb_by_k = {}, {}
            for k in k_grid:
                sel = ksub[k]
                if sel is None:
                    lcb_by_k[k] = float("nan"); rlcb_by_k[k] = float("nan"); continue
                cal_ctx = CalibrationContext(Zt[sel], yt[sel], effective_n_per_class(sel, yt))
                lcb_by_k[k] = stratified_bounded_lcb(heads["full"], heads["eras"], eraser, cal_ctx, conf, n_candidate)
                rlcb_by_k[k] = stratified_bounded_lcb(heads["full"], heads["rand"], eraser_random, cal_ctx, conf, n_candidate)
            seq = b3_sequential_decision_bounded(k_grid, lcb_by_k, rlcb_by_k, thr)
            action = seq["action"] if src_pass else "reject"
            dih = hash_obj({**seq, "src": src_pass})
            dsel = {**meta, "split_id": sid, "k": seq["k_used"], "source_safety_status": safety_status,
                    "domain_gain": None, "same_k_random_calibration": seq["specificity"] == "accepted_specific",
                    "calibration_benefit_lcb": seq.get("lcb_used"), "decision_action": action,
                    "decision_input_hash": dih, "calibration_idx_hash": hash_array(cal),
                    "calibration_label_hash": hash_array(yt[cal]),
                    "audit_idx_hash_hash_only": hash_obj(hash_array(aud)), "specificity": seq["specificity"],
                    "b3_ks_read": seq["ks_read"]}
            dec_rows.append(dsel)
            gate_a = LabelAccessGuard("audit")
            av = AuditView(np.asarray(aud), gate_a.audit_labels(AuditView(np.asarray(aud), yt[aud])))
            adb = audit_scalar(eraser, src, av, Zt[aud]) if action == "accept" else None
            adr = audit_scalar(eraser_random, src, av, Zt[aud]) if action == "accept" else None
            aud_rows.append({**meta, "split_id": sid, "k": seq["k_used"], "decision_input_hash": dih,
                             "audit_idx_hash": hash_array(aud), "audit_label_hash": hash_array(yt[aud]),
                             "audit_delta_bacc": adb, "audit_delta_bacc_random": adr, "audit_delta_nll": None,
                             "specificity_flag": seq["specificity"]})
    return dec_rows, aud_rows


# ============================ two-phase output writer (tested) ============================
RUN_OUT = "tos_cmi/results/target_info/tier1_run"


def _write_csv(path, rows):
    import csv
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def write_two_phase_outputs(dec_rows, aud_rows, out=RUN_OUT):
    """Write decision_rows.csv (source/calibration-derivable ONLY), audit_rows.csv (post-decision audit metrics),
    joined_rows.csv (post-hoc join). Raises if a decision row carries an audit metric/label (only the hashed
    audit_idx_hash_hash_only provenance is permitted in decision rows)."""
    os.makedirs(out, exist_ok=True)
    for r in dec_rows:
        bad = [c for c in r if ("audit_delta" in c or "audit_label" in c or "audit_metric" in c)]
        if bad:
            raise AssertionError("decision row leaked audit field(s): %s" % bad)
    _write_csv("%s/decision_rows.csv" % out, dec_rows)
    _write_csv("%s/audit_rows.csv" % out, aud_rows)
    aud_by = {a["decision_input_hash"]: a for a in aud_rows if a.get("decision_input_hash")}
    joined = [{**d, **{k: v for k, v in aud_by.get(d.get("decision_input_hash"), {}).items() if k not in d}}
              for d in dec_rows]
    _write_csv("%s/joined_rows.csv" % out, joined)
    return out


def _assemble_run(cfg, provider, out=RUN_OUT, n_boot=200):
    """Consume a task provider (yields dicts with meta + arrays + seed_key + erasers), run execute_task per task,
    and write the two-phase outputs. World/label loading lives in the provider; the assembly + leak-safe wiring is
    here and is unit-tested with a synthetic provider."""
    dec_all, aud_all = [], []
    for it in provider:
        dec, aud = execute_task(it["meta"], it["Zs"], it["ys"], it["z_src"], it["Zt"], it["yt"],
                                it["seed_key"], it["eraser"], it["eraser_random"], cfg, n_boot=n_boot,
                                source_subj=it.get("source_subj"))
        dec_all += dec; aud_all += aud
    write_two_phase_outputs(dec_all, aud_all, out)
    return len(dec_all), len(aud_all)


# ============================ gated real-EEG executor (reachable ONLY via an approved manifest+token) =============
_WORLD_INJECT = {"v2_source_invisible_world_a": "v2", "source_rich_source_visible_world_a": "source_rich"}
_INTERV_MAP = {"leace": "leace_baseline"}                       # driver name -> FACTORIES key (others identical)


def _real_provider(cfg, results_root, seed):
    """Yield one item per (dataset,world,fold,alpha,intervention,budget): load real dump, inject the world, fit the
    eraser + a same-k random eraser on SOURCE, and hand arrays to execute_task. UNEXERCISED until the PM enables a
    run (this is the real-dump/world-injection surface the run-enable review must sign off)."""
    import glob
    import re
    from tos_cmi.eeg.semi_synthetic_real_latent import inject
    from tos_cmi.eeg.source_rich_worlds import inject_source_rich
    from tos_cmi.eeg.v2_worlds import FACTORIES
    from tos_cmi.eeg.run_v2_certificate import _nuisance_m
    from tos_cmi.eeg.erasure_baselines import _ids
    sc = cfg["tier1_scope"]; folds = _folds(sc["folds"]); gseed = cfg["split_rng"]["global_split_seed"]
    for ds in sc["datasets"]:
        for bb in sc["backbones"]:
            dd = "%s/%s_%s_LOSO" % (results_root, ds, bb)
            ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                        key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))[:len(folds)]
            for fold, p in zip(folds, ps):
                sub = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
                d = np.load(p, allow_pickle=True)
                Zs = d["Z_source"].astype(float); ys = d["y_source"].astype(int)
                Zt = d["Z_target"].astype(float); yt = d["y_target"].astype(int)
                subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
                m = _nuisance_m(Zs.shape[1], "fraction_of_z_dim", 0.20, 4)
                for world in cfg["worlds"]:
                    for alpha in cfg["world_alpha_grid"]:
                        if _WORLD_INJECT[world] == "source_rich":
                            inj = inject_source_rich(Zs, ys, subj, Zt, yt, alpha=alpha, m=m, seed=seed)
                        else:
                            inj = inject("A", Zs, ys, subj, Zt, yt, alpha=alpha, seed=seed, m=m)
                        Zs2, Zt2, z_src = inj["Zs2"], inj["Zt2"], inj["z_src"]
                        seed_key = {"dataset": ds, "backbone": bb, "model_seed": seed, "fold": fold,
                                    "target_subject": sub, "global_split_seed": gseed}
                        for interv in cfg["interventions"]:
                            F = FACTORIES[_INTERV_MAP.get(interv, interv)]
                            E = F(Zs2, ys, z_src, n_cls, seed)
                            Erand = FACTORIES["random_k"](Zs2, ys, z_src, n_cls, seed)
                            for budget in cfg["budgets"]:
                                yield {"meta": {"dataset": ds, "backbone": bb, "world": world, "alpha": alpha,
                                                "fold": fold, "target_subject": sub, "intervention": interv,
                                                "budget": budget},
                                       "Zs": Zs2, "ys": ys, "z_src": z_src, "Zt": Zt2, "yt": yt,
                                       "seed_key": seed_key, "eraser": E, "eraser_random": Erand,
                                       "source_subj": subj}


# ============================ parallel Tier-1 smoke runner + aggregation ============================
SMOKE_OUT = "tos_cmi/results/target_info/tier1_smoke"


def _smoke_worker(ds, bb, world, fold, alpha, interv, path, cfg, n_boot):
    """One (ds,world,fold,alpha,interv) worker: load dump, inject world, fit erasers, compute source-safety UCB
    ONCE, then run all budgets. Returns (decision_rows, audit_rows, error_or_None)."""
    import re
    try:
        from tos_cmi.eeg.semi_synthetic_real_latent import inject
        from tos_cmi.eeg.source_rich_worlds import inject_source_rich
        from tos_cmi.eeg.v2_worlds import FACTORIES
        from tos_cmi.eeg.run_v2_certificate import _nuisance_m
        from tos_cmi.eeg.erasure_baselines import _ids
        sub = int(re.search(r"sub(\d+)_", path.split("/")[-1]).group(1))
        d = np.load(path, allow_pickle=True)
        Zs = d["Z_source"].astype(float); ys = d["y_source"].astype(int)
        Zt = d["Z_target"].astype(float); yt = d["y_target"].astype(int)
        subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
        m = _nuisance_m(Zs.shape[1], "fraction_of_z_dim", 0.20, 4)
        if _WORLD_INJECT[world] == "source_rich":
            inj = inject_source_rich(Zs, ys, subj, Zt, yt, alpha=alpha, m=m, seed=0)
        else:
            inj = inject("A", Zs, ys, subj, Zt, yt, alpha=alpha, seed=0, m=m)
        Zs2, Zt2, z_src = inj["Zs2"], inj["Zt2"], inj["z_src"]
        E = FACTORIES[_INTERV_MAP.get(interv, interv)](Zs2, ys, z_src, n_cls, 0)
        Erand = FACTORIES["random_k"](Zs2, ys, z_src, n_cls, 0)
        src = SourceContext(Zs2, ys, z_src, n_cls, subj=subj)
        saf = source_task_drop_ucb(E, src, cfg["thresholds"]["safety_eps"], n_boot=n_boot)   # once per interv
        seed_key = {"dataset": ds, "backbone": bb, "model_seed": 0, "fold": fold, "target_subject": sub,
                    "global_split_seed": cfg["split_rng"]["global_split_seed"]}
        dec, aud = [], []
        for budget in cfg["budgets"]:
            meta = {"dataset": ds, "backbone": bb, "world": world, "alpha": alpha, "fold": fold,
                    "target_subject": sub, "intervention": interv, "budget": budget}
            dcs, acs = execute_task(meta, Zs2, ys, z_src, Zt2, yt, seed_key, E, Erand, cfg, n_boot=n_boot,
                                    source_subj=subj, precomputed_safety=saf)
            dec += dcs; aud += acs
        return dec, aud, None
    except Exception as e:
        return [], [], "%s/%s/%s/f%s/a%s/%s: %s" % (ds, bb, world, fold, alpha, interv, repr(e)[:200])


def aggregate_smoke(dec_all, aud_all, cfg):
    from collections import Counter
    thr = cfg["thresholds"]["benefit_lcb"]
    aud_by = {a["decision_input_hash"]: a for a in aud_all if a.get("decision_input_hash")}

    def audit_of(r):
        a = aud_by.get(r.get("decision_input_hash"))
        return a.get("audit_delta_bacc") if a else None

    def audit_rand_of(r):
        a = aud_by.get(r.get("decision_input_hash"))
        return a.get("audit_delta_bacc_random") if a else None
    per_budget = {}
    for r in dec_all:
        per_budget.setdefault(r["budget"], Counter())[r["decision_action"]] += 1
    per_budget = {b: dict(c) for b, c in per_budget.items()}
    # B2 k-curve per world
    k_curve = {}
    for r in dec_all:
        if _family(r["budget"]) != "B2" or r.get("decision_action") == "unavailable_k":
            continue
        s = k_curve.setdefault("%s|k%s" % (r["world"], r["k"]),
                               {"world": r["world"], "k": r["k"], "n": 0, "accept": 0, "true_accept": 0,
                                "false_accept": 0, "harmful_accept": 0, "abstain": 0, "specific_calibration": 0,
                                "specific_audit": 0, "non_specific": 0, "audit_sum": 0.0, "audit_n": 0})
        s["n"] += 1
        act = r["decision_action"]
        if act == "abstain":
            s["abstain"] += 1
        if act == "accept":
            s["accept"] += 1
            adb, adr = audit_of(r), audit_rand_of(r)
            if adb is not None and adb == adb:
                s["audit_sum"] += adb; s["audit_n"] += 1
                if adb > thr:
                    s["true_accept"] += 1
                if adb <= 0:
                    s["false_accept"] += 1
                if adb < -0.01:
                    s["harmful_accept"] += 1
                if adr is not None and adr == adr and adb > adr:
                    s["specific_audit"] += 1
            if r.get("specificity") == "accepted_specific":
                s["specific_calibration"] += 1
            elif r.get("specificity") == "accepted_non_specific":
                s["non_specific"] += 1
    for s in k_curve.values():
        s["accept_rate"] = s["accept"] / s["n"] if s["n"] else 0.0
        s["mean_audit_dbacc_accepted"] = (s["audit_sum"] / s["audit_n"]) if s["audit_n"] else None
    # B3 label budget + accepts/false
    b3 = [r for r in dec_all if _family(r["budget"]) == "B3"]
    b3_acc = [r for r in b3 if r["decision_action"] == "accept"]
    b3_budget = [r.get("k") for r in b3_acc if r.get("k") is not None]
    b3_false = len([r for r in b3_acc if (audit_of(r) is not None and audit_of(r) == audit_of(r) and audit_of(r) <= 0)])
    b3_k1 = len([r for r in b3_acc if r.get("k") == 1])
    b3_actions = dict(Counter(r["decision_action"] for r in b3))
    # B4 oracle audit (diagnostic gap) per world -- now COMPUTED (v0 bug fixed)
    b4_by_world = {}
    for r in dec_all:
        if _family(r["budget"]) != "B4":
            continue
        v = audit_of(r)
        if v is not None and v == v:
            b4_by_world.setdefault(r["world"], []).append(v)
    b4_oracle = {w: {"mean_audit_dbacc": float(np.mean(vs)), "max_audit_dbacc": float(np.max(vs)), "n": len(vs)}
                 for w, vs in b4_by_world.items()}
    # deployable false-accept rate (B2+B3 only; B4 excluded)
    dep_accepts = [r for r in dec_all if _family(r["budget"]) in ("B2", "B3") and r["decision_action"] == "accept"]
    dep_false = [r for r in dep_accepts if (audit_of(r) is not None and audit_of(r) == audit_of(r) and audit_of(r) <= 0)]
    dep_harmful = [r for r in dep_accepts if (audit_of(r) is not None and audit_of(r) == audit_of(r) and audit_of(r) < -0.01)]
    # STOP-condition checks
    stop = {
        "b1_accepts": len([r for r in dec_all if _family(r["budget"]) == "B1" and r["decision_action"] == "accept"]),
        "b4_deployable_accepts": len([r for r in dec_all if _family(r["budget"]) == "B4"
                                      and r["decision_action"] in ("accept", "deployable_accept")]),
        "unflagged_non_specific": len([r for r in dec_all if r.get("decision_action") == "accept"
                                       and r.get("same_k_random_calibration") is False
                                       and r.get("specificity") not in ("accepted_non_specific",)]),
        "point_estimate_safety": len([r for r in dec_all if str(r.get("source_safety_status", "")).startswith("point")]),
        "b3_k1_accepts": b3_k1,
    }
    return {"per_budget": per_budget, "b2_k_curve": sorted(k_curve.values(), key=lambda s: (s["world"], s["k"])),
            "b3_label_budget_mean": (float(np.mean(b3_budget)) if b3_budget else None), "b3_actions": b3_actions,
            "b3_accepts": len(b3_acc), "b3_false_accepts": b3_false, "b3_k1_accepts": b3_k1,
            "b4_oracle_by_world": b4_oracle, "n_deployable_accepts": len(dep_accepts),
            "n_deployable_false_accepts": len(dep_false), "n_deployable_harmful_accepts": len(dep_harmful),
            "deployable_false_accept_rate": (len(dep_false) / len(dep_accepts)) if dep_accepts else 0.0,
            "stop_conditions": stop, "n_decision_rows": len(dec_all), "n_audit_rows": len(aud_all)}


def _smoke_plot(summary, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        return "(matplotlib unavailable: %r)" % e
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    worlds = sorted(set(s["world"] for s in summary["b2_k_curve"]))
    for w in worlds:
        pts = sorted([s for s in summary["b2_k_curve"] if s["world"] == w], key=lambda s: s["k"])
        ks = [s["k"] for s in pts]
        ax[0].plot(ks, [s["accept_rate"] for s in pts], marker="o", label=w[:22])
        ax[1].plot(ks, [s["mean_audit_dbacc_accepted"] or 0 for s in pts], marker="s", label=w[:22])
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k labels/class"); ax[0].set_ylabel("B2 accept rate")
    ax[0].set_title("B2: accept rate vs k"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].axhline(0.01, ls="--", c="k", lw=0.6); ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("k labels/class"); ax[1].set_ylabel("mean AUDIT ΔbAcc (accepted)")
    ax[1].set_title("B2: accepted held-out ΔbAcc vs k"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    p = "%s/target_info_tier1_budget_curve.png" % out; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def run_smoke(cfg, results_root=FROZEN_ROOT, out=SMOKE_OUT, n_jobs=None, n_boot=100):
    """Parallel Tier-1 smoke over (ds,world,fold,alpha,interv) workers; each runs all budgets. Writes the smoke
    CSVs + summary + report + budget curve. Structural leak failures surface as worker errors (-> reported)."""
    import glob
    import re
    import json as _json
    from joblib import Parallel, delayed
    n_jobs = n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    sc = cfg["tier1_scope"]; folds = _folds(sc["folds"]); seed = sc["seeds"][0]
    tasks = []
    for ds in sc["datasets"]:
        for bb in sc["backbones"]:
            dd = "%s/%s_%s_LOSO" % (results_root, ds, bb)
            ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                        key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))[:len(folds)]
            for fold, p in zip(folds, ps):
                for world in cfg["worlds"]:
                    for alpha in cfg["world_alpha_grid"]:
                        for interv in cfg["interventions"]:
                            tasks.append((ds, bb, world, fold, alpha, interv, p))
    print("tier1 smoke: %d workers (n_jobs=%d, n_boot=%d)" % (len(tasks), n_jobs, n_boot), flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_smoke_worker)(ds, bb, world, fold, alpha, interv, p, cfg, n_boot)
        for (ds, bb, world, fold, alpha, interv, p) in tasks)
    dec_all, aud_all, fails = [], [], []
    for dec, aud, err in res:
        dec_all += dec; aud_all += aud
        if err:
            fails.append(err)
    os.makedirs(out, exist_ok=True)
    _write_csv("%s/target_info_tier1_smoke_decision_rows.csv" % out,
               [{k: v for k, v in r.items() if not ("audit_delta" in k or "audit_label" in k or "audit_metric" in k)}
                for r in dec_all])
    _write_csv("%s/target_info_tier1_smoke_audit_rows.csv" % out, aud_all)
    aud_by = {a["decision_input_hash"]: a for a in aud_all if a.get("decision_input_hash")}
    joined = [{**r, **{k: v for k, v in aud_by.get(r.get("decision_input_hash"), {}).items() if k not in r}}
              for r in dec_all]
    _write_csv("%s/target_info_tier1_smoke_joined_rows.csv" % out, joined)
    summary = aggregate_smoke(dec_all, aud_all, cfg)
    summary["n_workers"] = len(tasks); summary["n_failures"] = len(fails); summary["failures"] = fails[:20]
    summary["scope"] = {"datasets": sc["datasets"], "backbones": sc["backbones"], "folds": folds,
                        "worlds": cfg["worlds"], "budgets": list(cfg["budgets"]),
                        "k_grid": cfg["budgets"]["B2_k_labels_per_class"]["k_grid"], "R": sc["repeats_R"],
                        "world_alpha_grid": cfg.get("world_alpha_grid"), "n_boot": n_boot,
                        "split_rng_scheme": cfg["split_rng"]["scheme"]}
    _json.dump(summary, open("%s/target_info_tier1_smoke_summary.json" % out, "w"), indent=1)
    _json.dump({"scope": summary["scope"], "n_decision_rows": summary["n_decision_rows"],
                "n_failures": summary["n_failures"]}, open("%s/target_info_tier1_manifest.json" % out, "w"), indent=1)
    fig = _smoke_plot(summary, out)
    from tos_cmi.eeg.report_target_info_tier1 import write_smoke_report
    write_smoke_report(summary, fig, out)
    return len(dec_all), len(aud_all), fails, summary


def execute_real(cfg, manifest=None, results_root=FROZEN_ROOT, out=SMOKE_OUT):
    """Reached ONLY after authorize_execution passes (never while locked). Belt-and-suspenders manifest re-check,
    then run the parallel Tier-1 smoke and write outputs."""
    if not (manifest or {}).get("runs_allowed", False) or not (manifest or {}).get("experiments_allowed", False):
        return 1, MANIFEST_HALT_MSG                            # never run while locked
    if (manifest or {}).get("run_status") != "approved":
        return 1, MANIFEST_HALT_MSG
    nd, na, fails, summary = run_smoke(cfg, results_root, out)
    sc = summary["stop_conditions"]
    tripped = [k for k, v in sc.items() if v]
    return 0, "TARGET_INFO_TIER1_RUN_DONE (%d decision rows, %d audit rows, %d worker-fails, stop_tripped=%s)" % (
        nd, na, len(fails), tripped)


# ============================ provider-validation mode (gated; exercises _real_provider; metrics REDACTED) =========
_METRIC_FIELD_TOKENS = ("bacc", "delta", "nll", "benefit_lcb", "specific", "domain_gain", "action",
                        "accept", "reject", "abstain", "random", "audit_delta")


def _redact_validation_output(dec_rows, aud_rows, meta):
    """Schema-safe validation summary: row counts, SAFE (non-metric) field NAMES, redacted-metric-field COUNTS,
    contexts, hashes -- NEVER a metric VALUE (bAcc/ΔbAcc/NLL/accept-rate/...). Metric code may run internally; its
    numeric outputs are redacted from the committed summary."""
    def split_fields(rows):
        names = sorted({k for r in rows for k in r})
        safe = [n for n in names if not any(t in n.lower() for t in _METRIC_FIELD_TOKENS)]
        return safe, len([n for n in names if any(t in n.lower() for t in _METRIC_FIELD_TOKENS)])
    dsafe, dred = split_fields(dec_rows); asafe, ared = split_fields(aud_rows)
    idxh = sorted({r.get("calibration_idx_hash") for r in dec_rows if isinstance(r.get("calibration_idx_hash"), str)})
    return {**meta, "rows_completed": len(dec_rows), "audit_rows_completed": len(aud_rows),
            "decision_row_schema_present": bool(dec_rows), "audit_row_schema_present": bool(aud_rows),
            "decision_row_safe_fields": dsafe, "decision_row_redacted_metric_fields": dred,
            "audit_row_safe_fields": asafe, "audit_row_redacted_metric_fields": ared,
            "distinct_calibration_idx_hashes": len(idxh),
            "contexts_constructed": ["SourceContext", "CalibrationContext", "AuditView"],
            "metrics_computed_internally": True, "metrics_redacted": True}


def provider_validate_one_dump(cfg, manifest, results_root=FROZEN_ROOT, out=PROVIDER_VAL_OUT):
    """Exercise `_real_provider` on ONE real dump (Lee EEGNet, first fold, source-rich World A, split_id 0,
    B0/B2(k=4)/B4, identity/leace_baseline/random_k). Runs the FULL decision+audit path internally but REDACTS every
    metric VALUE from the committed output (this is a plumbing check, NOT a science result). Gated behind
    manifest.provider_validation_allowed; the real-dump load itself is UNEXERCISED until the PM enables it."""
    if manifest.get("runs_allowed") or manifest.get("experiments_allowed"):
        return 1, "PROVIDER_VALIDATION_REFUSED: manifest runs_allowed/experiments_allowed must be false"
    sc = manifest.get("scope", {})
    ds, bb = sc.get("dataset", "Lee2019_MI"), sc.get("backbone", "EEGNet")
    world = sc.get("world", "source_rich_world_a_source_visible")
    budgets = sc.get("budgets", ["B0_source_only", "B2_k_labels_per_class", "B4_oracle_selector"])
    intervs = sc.get("interventions", ["identity", "leace_baseline", "random_k"])
    kval = sc.get("k", 4)
    import glob
    import re
    from tos_cmi.eeg.source_rich_worlds import inject_source_rich
    from tos_cmi.eeg.v2_worlds import FACTORIES
    from tos_cmi.eeg.run_v2_certificate import _nuisance_m
    from tos_cmi.eeg.erasure_baselines import _ids
    dd = "%s/%s_%s_LOSO" % (results_root, ds, bb)
    ps = sorted(glob.glob("%s/sub*_erm_lam0_seed0.npz" % dd),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    if not ps:
        return 1, "PROVIDER_VALIDATION_REFUSED: no dump found for %s/%s" % (ds, bb)
    p = ps[0]; sub = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
    d = np.load(p, allow_pickle=True)
    Zs = d["Z_source"].astype(float); ys = d["y_source"].astype(int)
    Zt = d["Z_target"].astype(float); yt = d["y_target"].astype(int)
    subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
    m = _nuisance_m(Zs.shape[1], "fraction_of_z_dim", 0.20, 4)
    inj = inject_source_rich(Zs, ys, subj, Zt, yt, alpha=1.0, m=m, seed=0)
    Zs2, Zt2, z_src = inj["Zs2"], inj["Zt2"], inj["z_src"]
    seed_key = {"dataset": ds, "backbone": bb, "model_seed": 0, "fold": 1, "target_subject": sub,
                "global_split_seed": cfg["split_rng"]["global_split_seed"]}
    vcfg = dict(cfg)
    vcfg["tier1_scope"] = dict(cfg["tier1_scope"], repeats_R=1)      # split_id 0 only
    vcfg["budgets"] = dict(cfg["budgets"])
    vcfg["budgets"]["B2_k_labels_per_class"] = dict(cfg["budgets"]["B2_k_labels_per_class"], k_grid=[kval])
    dec_all, aud_all = [], []
    for interv in intervs:
        E = FACTORIES[interv](Zs2, ys, z_src, n_cls, 0); Erand = FACTORIES["random_k"](Zs2, ys, z_src, n_cls, 0)
        for budget in budgets:
            meta = {"dataset": ds, "backbone": bb, "world": world, "alpha": 1.0, "fold": 1,
                    "target_subject": sub, "intervention": interv, "budget": budget}
            dec, aud = execute_task(meta, Zs2, ys, z_src, Zt2, yt, seed_key, E, Erand, vcfg, n_boot=100,
                                    source_subj=subj)
            dec_all += dec; aud_all += aud
    summary = _redact_validation_output(dec_all, aud_all, {
        "scope": {"dataset": ds, "backbone": bb, "world": world, "fold": 1, "split_id": 0, "k": kval,
                  "budgets": budgets, "interventions": intervs, "n_dumps": 1},
        "source_shape": list(np.asarray(Zs2).shape), "target_shape": list(np.asarray(Zt2).shape)})
    os.makedirs(out, exist_ok=True)
    json.dump(summary, open("%s/provider_validation_manifest.json" % out, "w"), indent=1)
    json.dump({"decision_row_schema_present": summary["decision_row_schema_present"],
               "audit_row_schema_present": summary["audit_row_schema_present"], "metrics_redacted": True},
              open("%s/provider_validation_schema.json" % out, "w"), indent=1)
    from tos_cmi.eeg.report_target_info_tier1 import write_provider_validation_report
    rpt = write_provider_validation_report(summary, out)
    print("provider-validation: %d decision rows, %d audit rows (metrics REDACTED) ; report %s"
          % (summary["rows_completed"], summary["audit_rows_completed"], rpt))
    print("TARGET_INFO_TIER1_PROVIDER_VALIDATION_DONE")
    return 0, "TARGET_INFO_TIER1_PROVIDER_VALIDATION_DONE (%d rows, metrics redacted)" % summary["rows_completed"]


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
    ap.add_argument("--provider-validate-one-dump", action="store_true")
    ap.add_argument("--run-manifest", default=MANIFEST)
    ap.add_argument("--enable-token", default=None)
    a = ap.parse_args(argv)
    if sum([a.dry_run, a.execute, a.preflight_real_splits, a.provider_validate_one_dump]) > 1:  # exactly one mode
        return 2, "CONFLICTING_FLAGS: choose one of --dry-run / --execute / --preflight-real-splits / --provider-validate-one-dump."
    if a.dry_run:
        n_plan, n_exp = dry_run(cfg)
        return 0, "TARGET_INFO_TIER1_DRYRUN_DONE (%d plan rows, %d expanded tasks)" % (n_plan, n_exp)
    if a.preflight_real_splits:
        manifest = load_manifest(a.run_manifest)
        if not manifest.get("preflight_allowed", False):          # gated: running preflight = separate PM go
            return 1, PREFLIGHT_HALT_MSG
        return preflight_real_splits(cfg, manifest)
    if a.provider_validate_one_dump:
        pman = load_manifest(a.run_manifest)                      # provider-validation manifest via --run-manifest
        if not pman.get("provider_validation_allowed", False):    # gated: running provider-validation = separate PM go
            return 1, PROVIDER_VAL_HALT_MSG
        return provider_validate_one_dump(cfg, pman)
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
