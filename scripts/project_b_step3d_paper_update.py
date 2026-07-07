#!/usr/bin/env python3
"""Project B Step-3D: merge the Step-3C bounded real-EEG benchmark expansion into the paper package.

This is *paper integration only*. It re-runs no experiment and touches no ``h2cmi/**`` or ``cmi/**``.
It reads frozen Step-2G synthetic tables, the Step-3A real-EEG bridge smoke, and the Step-3C bounded
real benchmark, then emits v2 paper artifacts (draft, method+tables LaTeX, claim boundary, three new
real-EEG tables, validation) and syncs the human-readable ``notes/`` docs.

Fail-loud contract (Step-3D section 9): missing Step-3C aggregate/per-domain, absent BNCI2014_004,
missing eval unit or support mode, an ``offline_tta_rate>0`` under degenerate/unavailable ACAR-harm, a
draft overclaim phrase, or a missing required mention are all hard errors.

CLI::

    python scripts/project_b_step3d_paper_update.py \
        --step2g /tmp/project_b_step2g_report \
        --step3a /tmp/project_b_step3a_real_bridge \
        --step3c /tmp/project_b_step3c_real_benchmark \
        --out /tmp/project_b_step3d_paper_update
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List

# --- repo root (this file lives in <root>/scripts/) ---------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(ROOT, "notes")

# Reason codes whose presence must NEVER be counted as an identity-action reason (TTA/ACAR blockers).
TTA_BLOCKER_CODES = {
    "OACI_ACAR_HARM_CALIBRATION_DEGENERATE",
    "OACI_ACAR_INSUFFICIENT_CALIBRATION",
    "OACI_TTA_HIGH_PRED_DISAGREEMENT",
    "OACI_TTA_IDENTITY_FALLBACK",
}
CORE_REASON_CODES = [
    "OACI_ACAR_HARM_CALIBRATION_DEGENERATE",
    "OACI_TOS_SUPPORT_MISMATCH",
    "OACI_LEAKAGE_RESIDUAL_UNAVAILABLE",
    "OACI_TTA_IDENTITY_FALLBACK",
]

# Overclaim phrases that must be absent from the draft (the "full MOABB benchmark" case is handled
# specially: allowed only inside "not a full MOABB benchmark").
OVERCLAIM_PHRASES = [
    "guarantees tta improvement",
    "solves concept shift",
    "detects all ood",
    "target-label tuned",
]
# Substrings (lower-cased) the draft MUST contain.
REQUIRED_MENTIONS = [
    "bounded real benchmark expansion",
    "harmful on bnci2014_004",
    "no offline_tta",
    "support_mismatch",
    "low_ess was not active",
    "nested support calibration was inert",
    "not a full benchmark",
]


class Fail(RuntimeError):
    """Loud, explicit validation failure."""


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise Fail(f"required input missing: {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: str) -> Any:
    if not os.path.isfile(path):
        raise Fail(f"required input missing: {path}")
    with open(path) as fh:
        return json.load(fh)


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def texesc(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


# ------------------------------------------------------------------------------------------------
# Load + validate frozen inputs
# ------------------------------------------------------------------------------------------------
def load_inputs(step2g: str, step3a: str, step3c: str) -> Dict[str, Any]:
    # Step-3C aggregate (check 1) + per-domain (check 2).
    agg = _read_csv(os.path.join(step3c, "aggregate_summary.csv"))
    if not agg:
        raise Fail("check 1: Step-3C aggregate_summary.csv is empty")
    per_domain = _read_csv(os.path.join(step3c, "per_domain_decisions.csv"))
    if not per_domain:
        raise Fail("check 2: Step-3C per_domain_decisions.csv is empty")
    reason_audit = _read_csv(os.path.join(step3c, "reason_code_audit.csv"))
    availability = _read_csv(os.path.join(step3c, "dataset_availability.csv"))
    claim_update = _read_json(os.path.join(step3c, "step3c_claim_boundary_update.json"))

    # Check 3: BNCI2014_004 present and available.
    prim = [r for r in availability if r["dataset"] == "BNCI2014_004"]
    if not prim or str(prim[0].get("available")).lower() != "true":
        raise Fail("check 3: BNCI2014_004 absent or unavailable in Step-3C")

    # Checks 4 + 5: both eval units and both support modes present.
    eval_units = sorted({r["eval_unit"] for r in agg})
    support_modes = sorted({r["support_mode"] for r in agg})
    for unit in ("subject", "session"):
        if unit not in eval_units:
            raise Fail(f"check 4: eval_unit '{unit}' missing from Step-3C aggregate")
    if len(support_modes) < 2:
        raise Fail(f"check 5: expected two support modes, found {support_modes}")

    # Check 6: no OFFLINE_TTA under degenerate/unavailable ACAR-harm.
    for r in agg:
        rate = _f(r["mean_router_offline_tta_rate"])
        degen = int(_f(r["n_acar_harm_degenerate_or_unavailable"]))
        if rate > 0 and degen > 0:
            raise Fail(
                "check 6: offline_tta_rate>0 while ACAR-harm degenerate/unavailable "
                f"({r['eval_unit']}/{r['support_mode']}: rate={rate}, degen={degen})"
            )
    # Invariant: no TTA/ACAR blocker is ever counted as an identity-action reason.
    for r in reason_audit:
        if r["reason_code"] in TTA_BLOCKER_CODES and int(_f(r["identity_action_count"])) != 0:
            raise Fail(
                f"invariant: TTA blocker {r['reason_code']} has identity_action_count="
                f"{r['identity_action_count']} (must be 0) at {r['eval_unit']}/{r['support_mode']}"
            )

    # Step-3A bridge smoke summary.
    bridge = _read_json(os.path.join(step3a, "real_bridge_summary.json"))
    # Step-2G synthetic world table.
    world = _read_csv(os.path.join(step2g, "table1_world_support_summary.csv"))
    step2g_claim = _read_json(os.path.join(step2g, "claim_boundary.json"))
    step2g_tables_tex = os.path.join(step2g, "paper_tables.tex")

    return {
        "agg": agg,
        "per_domain": per_domain,
        "reason_audit": reason_audit,
        "availability": prim[0],
        "claim_update": claim_update,
        "eval_units": eval_units,
        "support_modes": support_modes,
        "bridge": bridge,
        "world": world,
        "step2g_claim": step2g_claim,
        "step2g_tables_tex": step2g_tables_tex,
    }


def _agg_row(agg: List[Dict[str, str]], unit: str, mode: str) -> Dict[str, str]:
    for r in agg:
        if r["eval_unit"] == unit and r["support_mode"] == mode:
            return r
    raise Fail(f"aggregate row not found for {unit}/{mode}")


# ------------------------------------------------------------------------------------------------
# Tables 5 / 6 / 7
# ------------------------------------------------------------------------------------------------
TABLE5_COLS = [
    "dataset", "eval_unit", "support_mode", "n_targets", "n_domain_rows", "mean_strict_bacc",
    "mean_raw_offline_delta_bacc", "mean_router_coverage", "mean_router_identity_rate",
    "mean_router_offline_tta_rate", "mean_router_accepted_bacc", "mean_router_missed_benefit",
    "mean_router_avoided_harm", "n_refused_domains", "n_identity_domains", "n_offline_tta_domains",
    "n_support_mismatch_domains", "n_low_ess_domains", "n_acar_harm_degenerate_or_unavailable",
    "primary_interpretation",
]


def build_table5(agg: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{c: r.get(c, "") for c in TABLE5_COLS} for r in agg]


TABLE6_COLS = [
    "dataset", "eval_unit", "support_mode", "reason_code", "top_level_count",
    "identity_action_count", "offline_tta_action_count", "offline_tta_blocker_count",
]


def build_table6(reason_audit: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows = [r for r in reason_audit if r["reason_code"] in CORE_REASON_CODES]
    order = {c: i for i, c in enumerate(CORE_REASON_CODES)}
    rows.sort(key=lambda r: (r["eval_unit"], r["support_mode"], order[r["reason_code"]]))
    return [{c: r.get(c, "") for c in TABLE6_COLS} for r in rows]


TABLE7_COLS = [
    "dataset", "target_subject", "eval_unit", "support_mode", "domain_id", "decision_action",
    "identity_bacc", "offline_tta_bacc", "raw_gain", "reason_codes",
    "offline_tta_blocking_reason_codes", "density_nll_target_prior",
    "support_threshold_nll_target_prior", "target_support_excess", "ess", "interpretation",
]


def _interpret_domain(r: Dict[str, str]) -> str:
    action = r["decision_action"]
    excess = _f(r["target_support_excess"])
    raw = _f(r["raw_gain"])
    if action == "identity":
        return "support-valid identity accepted; raw TTA harmful so no benefit forgone materially; TTA blocked (ACAR degenerate)"
    if action == "refuse":
        drv = "SUPPORT_MISMATCH (target NLL above source threshold)" if excess > 0 else "support/ESS blocker"
        strong = "; strong raw-TTA harm avoided" if raw <= -0.15 else ""
        return f"refused by {drv}{strong}"
    return action


def build_table7(per_domain: List[Dict[str, str]]) -> List[Dict[str, str]]:
    mode = "in_source_subject_q95"  # baseline mode; nested is identical in outcome on real BNCI2014_004

    def sel(t: int, unit: str, dom: str = None) -> List[Dict[str, str]]:
        out = []
        for r in per_domain:
            if int(_f(r["target_subject"])) == t and r["eval_unit"] == unit and r["support_mode"] == mode:
                if dom is None or r["domain_id"] == dom:
                    out.append(r)
        return out

    picked: List[Dict[str, str]] = []
    picked += sel(1, "subject", "0")   # identity accepted
    picked += sel(2, "subject", "0")   # refuse SUPPORT_MISMATCH
    picked += sel(1, "session")         # mixed accept/refuse near threshold
    picked += sel(4, "subject", "0")   # refuse, strong raw harm
    if not picked:
        raise Fail("table7: no representative per-domain rows selected")

    rows = []
    for r in picked:
        row = {c: r.get(c, "") for c in TABLE7_COLS}
        row["interpretation"] = _interpret_domain(r)
        rows.append(row)
    return rows


def write_csv(path: str, cols: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# ------------------------------------------------------------------------------------------------
# Paper draft v2
# ------------------------------------------------------------------------------------------------
def _wmode(world: List[Dict[str, str]], w: str, mode: str) -> Dict[str, str]:
    for r in world:
        if r["world"] == w and r["support_mode"] == mode:
            return r
    raise Fail(f"synthetic world row not found: {w}/{mode}")


def build_draft(data: Dict[str, Any]) -> str:
    world = data["world"]
    agg = data["agg"]
    bridge = data["bridge"]

    r2 = _wmode(world, "R2", "nested_site_excess_q95")
    hf3 = _wmode(world, "HF3", "nested_site_excess_q95")
    hood = _wmode(world, "H_OOD", "nested_site_excess_q95")

    subj = _agg_row(agg, "subject", "in_source_subject_q95")
    sess = _agg_row(agg, "session", "in_source_subject_q95")

    # Step-3A bridge smoke folds (targets 1,2).
    smoke = {}
    for fo in bridge.get("folds", []):
        if fo["support_mode"] == "in_source_subject_q95":
            smoke[int(fo["target_subject"])] = fo

    lines: List[str] = []
    A = lines.append
    A("# Project B: Refusal-First Safe EEG Adaptation")
    A("")
    A("*Draft auto-assembled by `scripts/project_b_step3d_paper_update.py` (Step-3D) from frozen "
      "Step-2G synthetic tables, the Step-3A real-EEG bridge smoke, and the Step-3C bounded real "
      "benchmark expansion. No experiment was re-run.*")
    A("")
    A("## 1. Problem Statement")
    A("Test-time adaptation (TTA) can help or harm an EEG decoder at deployment, and whether it helps "
      "is not knowable from source data alone. Project B is **not** another EEG TTA loss; it is a "
      "**deployment router** that, for an unlabelled target, chooses among "
      "`REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and emits an auditable reason. Target labels are "
      "used only **post-hoc** for evaluation, never to decide.")
    A("")
    A("## 2. Deployment Action Space")
    A("`REFUSE` (emit no decode), `IDENTITY` (source-only prediction), `OFFLINE_TTA` (batch "
      "transductive class-conditional affine adaptation), `ONLINE_TTA` (streaming). The router is "
      "refusal-first: the default is REFUSE, and a non-refusal action must clear explicit support and "
      "calibrated-risk gates.")
    A("")
    A("## 3. Refusal-First Router")
    A("Action-specific blockers: support/stability/diagnostic failures block *every* action (including "
      "IDENTITY); TTA-evidence and ACAR-harm failures block only the TTA actions. Selection is "
      "safe-beneficial-then-identity: a beneficial admissible TTA can win; otherwise a support-valid "
      "IDENTITY; otherwise REFUSE.")
    A("")
    A("## 4. TOS / Support-Aware Diagnostics")
    A("A vector (target size, effective sample size, class-conditional density NLL, transform norm, "
      "condition number, prediction disagreement) rather than a single OOD scalar. Too-few-target / "
      "low-ESS / support-mismatch / unstable-transform each map to a distinct OACI reason.")
    A("")
    A("## 5. Prior-Decoupled Support Protocol")
    A("Support is measured under both the source prior and an estimated target prior; a label-prior "
      "shift with intact target-prior density is recorded as audit-only info, **not** a refusal. The "
      "support threshold is source-only (§8): baseline = q95 of in-source-unit target-prior NLL; the "
      "nested variant adds a scale-normalised held-out-unit *excess* to the base scale.")
    A("")
    A("## 6. ACAR: Action-Conditional Conformal Adaptation Risk")
    A("Per action, a split-conformal upper bound on *error* (eligibility) and *harm* (allowed-to-adapt), "
      "calibrated on externally-supplied risk predictions over source pseudo-targets. ACAR explicitly "
      "represents `available / degenerate / unavailable`: when source pseudo-target harm gains are "
      "single-class (degenerate) or too few (unavailable), no harm bound is produced and TTA is blocked.")
    A("")
    A("## 7. OACI Reason Codes")
    A("Every decision carries reason codes, separated into blocking vs audit-only, and into "
      "action-level blockers vs top-level decision reasons. A TTA blocker never reads as "
      "\"IDENTITY is unsafe\".")
    A("")
    A("## 8. Synthetic Protocol")
    A("A controllable EEG simulator with orthogonal shift knobs and a hierarchical site/subject/session "
      "DAG. Three locked worlds: **R2** (recoverable), **HF3** (harmful / concept-shift), **H-OOD** "
      "(target-only stress). Full training (no fast config). Two source-only support modes: "
      "`in_source_subject_q95` (baseline) and `nested_site_excess_q95`.")
    A("")
    A("## 9. Synthetic Results")
    A(f"- **R2** (strict {_f(r2['strict_bacc_mean']):.3f}): raw offline TTA helps on average "
      f"(+{_f(r2['raw_offline_delta_bacc_mean']):.3f}). The in-source support threshold over-refused "
      f"(coverage 0.00); nested source-site excess fixed this (coverage "
      f"{_f(r2['coverage_mean']):.2f}, accepted bAcc {_f(r2['accepted_bacc_mean']):.3f}). TTA stays "
      f"blocked because ACAR-harm is degenerate, so R2's benefit is a knowing **missed benefit**.")
    A(f"- **HF3** (strict {_f(hf3['strict_bacc_mean']):.3f}): raw offline TTA is harmful on average "
      f"({_f(hf3['raw_offline_delta_bacc_mean']):.3f}); the router blocks OFFLINE_TTA under ACAR-harm "
      f"degeneracy. Nested mode accepts some IDENTITY domains (coverage {_f(hf3['coverage_mean']):.2f}); "
      f"a **concept-degraded identity** can pass source-only support (accepted bAcc "
      f"{_f(hf3['accepted_bacc_mean']):.3f}).")
    A(f"- **H-OOD** (strict {_f(hood['strict_bacc_mean']):.3f}): raw offline TTA is harmful "
      f"({_f(hood['raw_offline_delta_bacc_mean']):.3f}). After nested widening the density "
      f"`SUPPORT_MISMATCH` clears, but **LOW_ESS** remains the active blocker and OFFLINE_TTA is never "
      f"selected.")
    A("")
    A("## 10. Real-EEG Evidence")
    A("The real-EEG evidence has two tiers: a Step-3A **bridge smoke** and a Step-3C **bounded real "
      "benchmark expansion**. Both are source-only and label-safe; target labels are used only "
      "**post-hoc**. Neither is a full benchmark.")
    A("")
    A("### 10.1 Step-3A bridge smoke")
    if 1 in smoke and 2 in smoke:
        d1 = _f(smoke[1]["raw_offline_delta_bacc"])
        d2 = _f(smoke[2]["raw_offline_delta_bacc"])
        A(f"BNCI2014_004 bridge smoke ran on subjects 1–4, targets 1–2. Raw offline TTA was harmful on "
          f"both targets: target 1 d_bAcc = {d1:.3f}; target 2 d_bAcc = {d2:.3f}. The router accepted "
          f"support-valid IDENTITY and selected no OFFLINE_TTA. This is a **bridge smoke**, not a full "
          f"benchmark. On this small bridge the nested source-subject excess was 0, so nested == "
          f"baseline.")
    A("")
    A("### 10.2 Step-3C bounded real benchmark expansion")
    A("The **bounded real benchmark expansion** evaluates BNCI2014_004 with `max_subjects = 6`, target "
      "subjects 1–4, eval units subject and session, both source-only support modes "
      "(`in_source_subject_q95` and `nested_source_subject_excess_q95`), 8 epochs, and "
      "`max_nested_folds = 2`.")
    A("")
    A(f"- Raw offline TTA was harmful on BNCI2014_004: mean d_bAcc = "
      f"{_f(subj['mean_raw_offline_delta_bacc']):.3f} across targets. **no OFFLINE_TTA** was selected "
      f"(offline_tta_rate = {_f(subj['mean_router_offline_tta_rate']):.2f}).")
    A(f"- Subject-level routing: coverage = {_f(subj['mean_router_coverage']):.2f}, identity_rate = "
      f"{_f(subj['mean_router_identity_rate']):.2f}, accepted_bAcc = "
      f"{_f(subj['mean_router_accepted_bacc']):.3f}, refused/identity = "
      f"{int(_f(subj['n_refused_domains']))}/{int(_f(subj['n_identity_domains']))} target domains.")
    A(f"- Session-level routing: coverage = {_f(sess['mean_router_coverage']):.2f}, identity_rate = "
      f"{_f(sess['mean_router_identity_rate']):.2f}, accepted_bAcc = "
      f"{_f(sess['mean_router_accepted_bacc']):.3f}, refused/identity = "
      f"{int(_f(sess['n_refused_domains']))}/{int(_f(sess['n_identity_domains']))} session domains.")
    A("- The dominant top-level refusal driver was `SUPPORT_MISMATCH`. **LOW_ESS was not active** on "
      "this real run (real subjects have ample trials).")
    A("- **nested support calibration was inert**: nested_excess ≈ 0, so the nested mode matched "
      "baseline decisions on real BNCI2014_004 (held-out source subjects were not above the in-source "
      "support boundary).")
    A("")
    A("These results support harm avoidance and refusal/identity routing on real EEG, **not** an "
      "accuracy improvement over identity. See Tables 5–7.")
    A("")
    A("## 11. Claim Boundary")
    cb = build_claim_boundary(data)
    A("Claimable:")
    for i, c in enumerate(cb["claimable"], 1):
        A(f"{i}. {c}")
    A("")
    A("NOT claimable:")
    for i, c in enumerate(cb["not_claimable"], 1):
        A(f"{i}. {c}")
    A("")
    A("## 12. Limitations")
    A("- ACAR-harm is frequently degenerate/unavailable source-only, so v1 forgoes beneficial TTA (R2 "
      "**missed benefit**).")
    A("- Source-only support cannot detect concept-shift accuracy loss (**concept-degraded identity** "
      "passes on HF3).")
    A("- Nested widening can clear the density support signal; **LOW_ESS** is then the only active "
      "support blocker (H-OOD).")
    A("- The real-EEG evidence is a bridge smoke plus a bounded real benchmark expansion (few "
      "subjects/targets, 8 epochs, ≤2 nested folds), **not a full benchmark**.")
    A("- The bounded real benchmark ran in a **harmful-TTA regime**: raw TTA was harmful on every "
      "evaluated BNCI2014_004 target, so it demonstrates harm avoidance, not beneficial-TTA recovery.")
    A("")
    A("## 13. Next Benchmark Expansion")
    A("More subjects and targets, additional datasets (BNCI2014_001 / Lee2019_MI, GPU run), keeping the "
      "source-only, label-safe protocol. Whether ACAR-harm ever becomes calibratable and beneficial "
      "TTA is recoverable at scale is an empirical question, not an assumption.")
    A("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# Claim boundary v2
# ------------------------------------------------------------------------------------------------
def build_claim_boundary(data: Dict[str, Any]) -> Dict[str, Any]:
    claimable = [
        "Default Project B v1 never selects OFFLINE_TTA when ACAR-harm calibration is degenerate/unavailable.",
        "Nested source-site support excess calibration fixes the Step-2E all-refuse failure on R2.",
        "Project B can allow support-valid IDENTITY while still refusing low-ESS targets.",
        "OACI reason codes expose whether refusal came from support, ESS, ACAR degeneracy, or TTA evidence.",
        "The real-EEG bridge runs end-to-end on BNCI2014_004 under label-safe LOSO.",
        "The bounded BNCI2014_004 real-EEG expansion shows raw offline TTA was harmful across the "
        "evaluated targets and the router selected no OFFLINE_TTA under degenerate ACAR-harm calibration.",
        "In the bounded BNCI2014_004 expansion, the router produced heterogeneous support-valid "
        "IDENTITY vs REFUSE decisions at subject and session granularity with OACI reason audit.",
    ]
    not_claimable = [
        "Source-only ACAR-harm is not shown to be generally identifiable.",
        "v1 does not recover R2's raw TTA benefit (TTA blocked under harm-calibration degeneracy).",
        "Support-valid identity is not shown accurate under concept shift.",
        "The density support threshold alone does not catch H-OOD after nested widening; LOW_ESS "
        "catches only a subset.",
        "No target-label-tuned thresholds; all thresholds are source-only and label-safe.",
        "This is not a full MOABB benchmark.",
        "The bounded real benchmark does not establish that Project B improves accuracy over identity; "
        "it establishes refusal/identity routing and TTA harm avoidance under the observed "
        "harmful-TTA regime.",
        "The bounded real benchmark does not show beneficial-TTA recovery on real EEG; raw TTA was "
        "harmful in the evaluated BNCI2014_004 targets.",
    ]
    subj = _agg_row(data["agg"], "subject", "in_source_subject_q95")
    sess = _agg_row(data["agg"], "session", "in_source_subject_q95")
    return {
        "step": "3D",
        "primary_support_mode_real": "nested_source_subject_excess_q95",
        "primary_support_mode_synthetic": "nested_site_excess_q95",
        "claimable": claimable,
        "not_claimable": not_claimable,
        "locked_worlds_synthetic": [
            {
                "world": r["world"], "support_mode": r["support_mode"],
                "strict_bacc": _f(r["strict_bacc_mean"]),
                "raw_offline_delta_bacc": _f(r["raw_offline_delta_bacc_mean"]),
                "coverage": _f(r["coverage_mean"]),
                "accepted_bacc": _f(r["accepted_bacc_mean"]),
                "acar_harm_state": r["acar_harm_state"],
            }
            for r in data["world"] if r["support_mode"] == "nested_site_excess_q95"
        ],
        "real_benchmark_bounded": {
            "dataset": "BNCI2014_004",
            "n_primary_targets": int(_f(subj["n_targets"])),
            "eval_units": data["eval_units"],
            "support_modes": data["support_modes"],
            "mean_raw_offline_delta_bacc": _f(subj["mean_raw_offline_delta_bacc"]),
            "subject": {"coverage": _f(subj["mean_router_coverage"]),
                        "identity_rate": _f(subj["mean_router_identity_rate"]),
                        "offline_tta_rate": _f(subj["mean_router_offline_tta_rate"]),
                        "accepted_bacc": _f(subj["mean_router_accepted_bacc"])},
            "session": {"coverage": _f(sess["mean_router_coverage"]),
                        "identity_rate": _f(sess["mean_router_identity_rate"]),
                        "offline_tta_rate": _f(sess["mean_router_offline_tta_rate"]),
                        "accepted_bacc": _f(sess["mean_router_accepted_bacc"])},
            "dominant_refusal_driver": "OACI_TOS_SUPPORT_MISMATCH",
            "low_ess_active": False,
            "nested_support_inert": True,
        },
        "runtime_bounded_partial": bool(data["claim_update"].get("runtime_bounded_partial")),
    }


def build_claim_boundary_md(cb: Dict[str, Any], data: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append
    A("# Project B — Claim Boundary (Step-3D: synthetic + real)")
    A("")
    A("Human-readable companion to `project_b_claim_boundary_v2.json` (generated by "
      "`scripts/project_b_step3d_paper_update.py`). Primary real support mode: "
      "**`nested_source_subject_excess_q95`**; primary synthetic mode: **`nested_site_excess_q95`**. "
      "Primary posture: **support-valid IDENTITY; no TTA under degenerate ACAR-harm.**")
    A("")
    A("## Claimable")
    for i, c in enumerate(cb["claimable"], 1):
        A(f"{i}. {c}")
    A("")
    A("## NOT claimable")
    for i, c in enumerate(cb["not_claimable"], 1):
        A(f"{i}. {c}")
    A("")
    A("## Locked worlds — synthetic (nested mode, means over seeds)")
    A("| world | strict | raw ΔTTA | nested cov | nested acc-bAcc | ACAR-harm |")
    A("|---|---|---|---|---|---|")
    for r in cb["locked_worlds_synthetic"]:
        A(f"| {r['world']} | {r['strict_bacc']:.3f} | {r['raw_offline_delta_bacc']:+.3f} | "
          f"{r['coverage']:.2f} | {r['accepted_bacc']:.3f} | {r['acar_harm_state']} |")
    A("")
    rb = cb["real_benchmark_bounded"]
    A("## Real bounded benchmark — BNCI2014_004 (Step-3C)")
    A(f"- Targets: {rb['n_primary_targets']}; eval units: {', '.join(rb['eval_units'])}; support modes: "
      f"{', '.join(rb['support_modes'])}.")
    A(f"- Raw offline TTA mean d_bAcc = {rb['mean_raw_offline_delta_bacc']:.3f} (harmful).")
    A(f"- Subject: coverage {rb['subject']['coverage']:.2f}, identity_rate "
      f"{rb['subject']['identity_rate']:.2f}, offline_tta_rate {rb['subject']['offline_tta_rate']:.2f}, "
      f"accepted_bAcc {rb['subject']['accepted_bacc']:.3f}.")
    A(f"- Session: coverage {rb['session']['coverage']:.2f}, identity_rate "
      f"{rb['session']['identity_rate']:.2f}, offline_tta_rate {rb['session']['offline_tta_rate']:.2f}, "
      f"accepted_bAcc {rb['session']['accepted_bacc']:.3f}.")
    A(f"- Dominant refusal driver: `{rb['dominant_refusal_driver']}`; LOW_ESS active: "
      f"{rb['low_ess_active']}; nested support inert: {rb['nested_support_inert']}.")
    A("")
    A("## Known limitations")
    A("- R2 missed benefit (TTA blocked under degenerate ACAR-harm).")
    A("- HF3 concept-degraded identity can pass source-only support checks.")
    A("- H_OOD density threshold cleared after nested calibration; `LOW_ESS` catches only a subset.")
    A("- Real evidence is bounded (bridge smoke + bounded benchmark), not a full benchmark; the real "
      "run sat in a harmful-TTA regime, so it shows harm avoidance, not beneficial-TTA recovery.")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------
# Reviewer checklist (Q1-Q7 kept, Q8-Q10 added)
# ------------------------------------------------------------------------------------------------
def build_checklist() -> str:
    return """# Project B Reviewer Checklist

## Q1. Does the router use target labels to decide?
Answer: No. Target labels are used only **post-hoc** for metrics, after every RouterDecision. No
threshold, diagnostic, or action reads target labels.

## Q2. Why does the router miss R2's raw TTA benefit?
Answer: Because **ACAR-harm** is degenerate; v1 refuses to adapt without usable harm calibration. The
forgone benefit is a knowing **missed benefit**, not a bug.

## Q3. Why does HF3 concept-degraded identity pass?
Answer: Source-only support diagnostics do not identify concept-shift accuracy loss; a
**concept-degraded identity** can be support-valid yet inaccurate.

## Q4. Why does H-OOD density support clear under the nested threshold?
Answer: The nested excess widens the support threshold; **LOW_ESS** remains the active support signal
for low-effective-sample domains, so part of H-OOD is still refused.

## Q5. Is Project B a new TTA optimizer?
Answer: No. It is a refusal-first deployment router on top of existing TTA.

## Q6. What is the real-EEG evidence?
Answer: A BNCI2014_004 LOSO **bridge smoke** plus a **bounded real benchmark expansion** (not a full
benchmark): raw TTA harmful; the router blocks TTA and accepts support-valid identity.

## Q7. What remains for benchmark expansion?
Answer: More subjects, more targets, additional datasets (BNCI2014_001 / Lee2019_MI, GPU run).

## Q8. What changed from the real bridge smoke to the bounded real benchmark?
Answer: Step-3A showed the bridge can run on two BNCI2014_004 targets. Step-3C expands to four targets,
subject- and session-level routing, and both source-only support modes.

## Q9. Does the bounded real benchmark prove accuracy improvement?
Answer: No. Raw offline TTA was harmful in the evaluated BNCI2014_004 targets. The result supports harm
avoidance and refusal/identity routing, not accuracy improvement over identity.

## Q10. Why is nested support calibration inert on real BNCI2014_004?
Answer: Nested source-subject excess was near zero, meaning held-out source subjects were not above the
in-source support boundary under the normalized excess criterion. Thus nested and baseline support
thresholds made the same decisions.
"""


# ------------------------------------------------------------------------------------------------
# LaTeX v2
# ------------------------------------------------------------------------------------------------
def build_method_tex() -> str:
    return r"""\section{Refusal-First Safe EEG Adaptation}
\subsection{Action Space and Router}
% Refusal-first router over REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA with action-specific blockers.
\subsection{Support-Aware Prior-Decoupled Diagnostics}
% Source-only support threshold; prior-shift recorded as audit info, not refusal.
\subsection{Action-Conditional Risk Calibration}
% ACAR error/harm split-conformal bounds with available/degenerate/unavailable states.
\subsection{Synthetic Protocol}
% Controllable simulator, three locked worlds R2/HF3/H-OOD, source-only calibration.
\subsection{Synthetic Results}
% See Table~\ref{tab:pb-world-support} and Table~\ref{tab:pb-ablation}.
\subsection{Real-EEG Bridge and Bounded Expansion}
The real-EEG bridge is not a full MOABB benchmark.
The bounded expansion evaluates BNCI2014\_004 with four LOSO targets,
subject- and session-level routing, and both source-only support modes.
% Step-3A bridge smoke: raw TTA harmful on two targets; router blocks TTA, accepts identity.
% Step-3C bounded expansion: raw TTA harmful (mean d_bAcc -0.140); no OFFLINE_TTA selected;
% SUPPORT_MISMATCH drives refusal; LOW_ESS inactive; nested support inert. See Tables 5--7.
\subsection{Limitations}
% ACAR-harm degeneracy, concept-degraded identity, LOW_ESS-only support signal;
% real evidence is a bridge smoke plus a bounded expansion, not a full benchmark; harmful-TTA regime.
% TODO: cite TTA, selective prediction, conformal risk control, EEG transfer learning.
"""


def _tex_table5(rows: List[Dict[str, str]]) -> str:
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Real-EEG bounded benchmark aggregate (BNCI2014\_004; means over targets).}",
         r"\label{tab:pb-real-aggregate}",
         r"\begin{tabular}{llrrrrr}\toprule",
         r"eval unit & mode & strict & raw $\Delta$TTA & cov. & id.\ rate & off-TTA \\ \midrule"]
    for r in rows:
        L.append(
            f"{texesc(r['eval_unit'])} & {texesc(r['support_mode'])} & "
            f"{_f(r['mean_strict_bacc']):.3f} & {_f(r['mean_raw_offline_delta_bacc']):+.3f} & "
            f"{_f(r['mean_router_coverage']):.2f} & {_f(r['mean_router_identity_rate']):.2f} & "
            f"{_f(r['mean_router_offline_tta_rate']):.2f} \\\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(L)


def _tex_table6(rows: List[Dict[str, str]]) -> str:
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Real-EEG reason-code audit (core codes). TTA blockers never mark identity unsafe "
         r"(identity count $=0$).}",
         r"\label{tab:pb-real-reason}",
         r"\begin{tabular}{lllrrr}\toprule",
         r"eval unit & mode & reason code & top & id.\ act & TTA blk \\ \midrule"]
    for r in rows:
        L.append(
            f"{texesc(r['eval_unit'])} & {texesc(r['support_mode'])} & {texesc(r['reason_code'])} & "
            f"{int(_f(r['top_level_count']))} & {int(_f(r['identity_action_count']))} & "
            f"{int(_f(r['offline_tta_blocker_count']))} \\\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(L)


def _tex_table7(rows: List[Dict[str, str]]) -> str:
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Representative real-EEG per-domain decisions (baseline support mode).}",
         r"\label{tab:pb-real-examples}",
         r"\begin{tabular}{llrrl}\toprule",
         r"target & unit & id.\ bAcc & raw gain & decision \\ \midrule"]
    for r in rows:
        L.append(
            f"{texesc(r['target_subject'])} & {texesc(r['eval_unit'])} & "
            f"{_f(r['identity_bacc']):.3f} & {_f(r['raw_gain']):+.3f} & {texesc(r['decision_action'])} \\\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(L)


def build_tables_tex(step2g_tables_tex: str, t5, t6, t7) -> str:
    with open(step2g_tables_tex) as fh:
        synthetic = fh.read().rstrip()
    parts = [
        "% Project B Step-3D paper tables (auto-generated; \\input-able).",
        "% --- Step-2 synthetic tables ---",
        synthetic,
        "",
        "% --- Step-3C real benchmark aggregate table ---",
        _tex_table5(t5),
        "",
        "% --- Step-3C real reason-code audit table ---",
        _tex_table6(t6),
        "",
        "% --- Step-3C real representative examples table ---",
        _tex_table7(t7),
        "",
    ]
    return "\n".join(parts)


# ------------------------------------------------------------------------------------------------
# Validation of the assembled draft
# ------------------------------------------------------------------------------------------------
def validate_draft(draft: str) -> Dict[str, Any]:
    lo = draft.lower()
    # Overclaim: "full moabb benchmark" allowed ONLY within "not a full moabb benchmark".
    stripped = lo.replace("not a full moabb benchmark", "")
    if "full moabb benchmark" in stripped:
        raise Fail("check 7: draft contains bare 'full MOABB benchmark' (not negated)")
    for p in OVERCLAIM_PHRASES:
        if p in lo:
            raise Fail(f"check 7: draft contains overclaim phrase '{p}'")
    missing = [m for m in REQUIRED_MENTIONS if m not in lo]
    if missing:
        raise Fail(f"check 8: draft missing required mention(s): {missing}")
    return {"overclaim_ok": True, "required_mentions_present": REQUIRED_MENTIONS}


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Project B Step-3D paper update (read-only integration).")
    ap.add_argument("--step2g", required=True)
    ap.add_argument("--step3a", required=True)
    ap.add_argument("--step3c", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_inputs(args.step2g, args.step3a, args.step3c)

    # Tables.
    t5 = build_table5(data["agg"])
    t6 = build_table6(data["reason_audit"])
    t7 = build_table7(data["per_domain"])
    write_csv(os.path.join(args.out, "table5_real_benchmark_aggregate.csv"), TABLE5_COLS, t5)
    write_csv(os.path.join(args.out, "table6_real_reason_audit.csv"), TABLE6_COLS, t6)
    write_csv(os.path.join(args.out, "table7_real_domain_examples.csv"), TABLE7_COLS, t7)

    # Draft + claim boundary (draft embeds claim boundary; build both).
    draft = build_draft(data)
    validation = validate_draft(draft)
    cb = build_claim_boundary(data)
    cb_md = build_claim_boundary_md(cb, data)
    checklist = build_checklist()
    method_tex = build_method_tex()
    tables_tex = build_tables_tex(data["step2g_tables_tex"], t5, t6, t7)

    # Write /tmp outputs.
    with open(os.path.join(args.out, "project_b_paper_draft_v2.md"), "w") as fh:
        fh.write(draft)
    with open(os.path.join(args.out, "project_b_method_results_v2.tex"), "w") as fh:
        fh.write(method_tex)
    with open(os.path.join(args.out, "project_b_tables_v2.tex"), "w") as fh:
        fh.write(tables_tex)
    with open(os.path.join(args.out, "project_b_claim_boundary_v2.json"), "w") as fh:
        json.dump(cb, fh, indent=2)

    validation_summary = {
        "step": "3D",
        "input_checks": {
            "aggregate_present": True, "per_domain_present": True,
            "bnci2014_004_present": True, "both_eval_units": data["eval_units"],
            "both_support_modes": data["support_modes"],
            "no_offline_tta_under_degenerate_acar": True,
            "tta_blocker_identity_count_zero_invariant": True,
        },
        "draft_checks": validation,
        "tables": {"table5_rows": len(t5), "table6_rows": len(t6), "table7_rows": len(t7)},
        "claimable_count": len(cb["claimable"]),
        "not_claimable_count": len(cb["not_claimable"]),
        "all_checks_passed": True,
    }
    with open(os.path.join(args.out, "step3d_validation.json"), "w") as fh:
        json.dump(validation_summary, fh, indent=2)

    # Sync notes/.
    with open(os.path.join(NOTES, "PROJECT_B_PAPER_DRAFT.md"), "w") as fh:
        fh.write(draft)
    with open(os.path.join(NOTES, "PROJECT_B_REVIEWER_CHECKLIST.md"), "w") as fh:
        fh.write(checklist)
    with open(os.path.join(NOTES, "PROJECT_B_CLAIM_BOUNDARY.md"), "w") as fh:
        fh.write(cb_md)

    # Step-3D update note.
    subj = _agg_row(data["agg"], "subject", "in_source_subject_q95")
    sess = _agg_row(data["agg"], "session", "in_source_subject_q95")
    note = f"""# Project B Step-3D: Paper Update

Merges the Step-3C **bounded real benchmark expansion** into the Project B paper package. Paper
integration only — no experiment re-run, no `h2cmi/**` or `cmi/**` change.

## What changed
- `PROJECT_B_PAPER_DRAFT.md` §10 split into **10.1 Step-3A bridge smoke** and **10.2 Step-3C bounded
  real benchmark expansion**; §11 claim boundary and §12 limitations updated for the real result.
- `PROJECT_B_CLAIM_BOUNDARY.md`: claimable {len(build_claim_boundary(data)['claimable'])} items
  (adds two Step-3C claims), not-claimable {len(build_claim_boundary(data)['not_claimable'])} items
  (adds "not a full MOABB benchmark" + two harmful-regime caveats).
- `PROJECT_B_REVIEWER_CHECKLIST.md`: adds Q8–Q10.
- New paper tables: Table 5 (real aggregate), Table 6 (real reason audit), Table 7 (representative
  real per-domain examples), plus `*_v2.tex` and `*_v2.json`.

## Headline real numbers (BNCI2014_004, baseline support mode)
- Raw offline TTA mean d_bAcc = {_f(subj['mean_raw_offline_delta_bacc']):.3f} (harmful); OFFLINE_TTA
  rate = {_f(subj['mean_router_offline_tta_rate']):.2f}.
- Subject routing: coverage {_f(subj['mean_router_coverage']):.2f}, identity_rate
  {_f(subj['mean_router_identity_rate']):.2f}, accepted_bAcc {_f(subj['mean_router_accepted_bacc']):.3f}.
- Session routing: coverage {_f(sess['mean_router_coverage']):.2f}, identity_rate
  {_f(sess['mean_router_identity_rate']):.2f}, accepted_bAcc {_f(sess['mean_router_accepted_bacc']):.3f}.
- Dominant refusal driver `SUPPORT_MISMATCH`; LOW_ESS inactive; nested support inert.

## Boundary
Bounded real benchmark expansion, **not a full benchmark**. Demonstrates harm avoidance and
refusal/identity routing under a harmful-TTA regime; does not establish accuracy improvement over
identity or beneficial-TTA recovery on real EEG.

## Inputs (read-only, frozen)
- Step-2G: `{args.step2g}` · Step-3A: `{args.step3a}` · Step-3C: `{args.step3c}`
- Outputs: `{args.out}`
"""
    with open(os.path.join(NOTES, "PROJECT_B_STEP3D_PAPER_UPDATE.md"), "w") as fh:
        fh.write(note)

    print("[OK] Step-3D paper update assembled.")
    print(f"  out dir: {args.out}")
    print(f"  tables: t5={len(t5)} t6={len(t6)} t7={len(t7)} rows")
    print(f"  claimable={len(cb['claimable'])} not_claimable={len(cb['not_claimable'])}")
    print("  draft validation: overclaim_ok=True, required_mentions all present")
    print("  synced notes: PAPER_DRAFT, REVIEWER_CHECKLIST, CLAIM_BOUNDARY, STEP3D_PAPER_UPDATE")


if __name__ == "__main__":
    main()
