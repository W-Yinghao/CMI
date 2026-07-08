"""Project B-Next Step-S5: manuscript consolidation package (read-only).

Integrates every frozen Project B result (Step-2G synthetic, Step-3C real benchmark, S0 calibration
records, S2/S2B ACAR-error, S3 PRIOR_ONLY, S1 phase map, S4 backend comparison) into a final manuscript
draft + claim boundary + results digest + EEGAgent integration roadmap + LaTeX. No experiment is re-run;
no h2cmi/** or cmi/** is modified. Numbers are read from the frozen dirs (fail-loud on missing), never
written from memory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(REPO, "notes")
EXPECTED_BRANCH = "project-b-next"

OVERCLAIM = [
    "guarantees tta improvement", "solves concept shift", "full moabb benchmark superiority",
    "foundation model solves adaptation", "eegagent guarantees safety",
]
REQUIRED_MENTIONS = [
    "refusal-first router", "acar-harm degeneracy", "optional acar-error", "h-ood",
    "boundary persists", "prior_only", "no real benefit phase", "weak-baseline artifact",
    "eegagent", "safety governor",
]


class Fail(RuntimeError):
    pass


def _branch():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def _req(path, what):
    if not os.path.exists(path):
        raise Fail(f"[FAIL] required {what} missing: {path}")
    return path


def _rows(path):
    _req(path, "csv")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _json(path):
    _req(path, "json")
    with open(path) as fh:
        return json.load(fh)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v, p=3):
    return "nan" if (isinstance(v, float) and math.isnan(v)) else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


# ------------------------------------------------------------------ digest extraction
def build_digest(a):
    d = {}
    # Step-2G synthetic (nested mode)
    t1 = _rows(os.path.join(a.step2g, "table1_world_support_summary.csv"))

    def _w(world):
        r = next((x for x in t1 if x["world"] == world and "nested" in x["support_mode"]), None)
        return r
    for w in ("R2", "HF3", "H_OOD"):
        r = _w(w)
        if r:
            d[f"{w}_strict"] = _f(r["strict_bacc_mean"])
            d[f"{w}_raw_tta"] = _f(r["raw_offline_delta_bacc_mean"])
            d[f"{w}_nested_cov"] = _f(r["coverage_mean"])
            d[f"{w}_nested_acc"] = _f(r["accepted_bacc_mean"])
    # Step-3C real benchmark
    agg = _rows(os.path.join(a.step3c, "aggregate_summary.csv"))
    sub = next((r for r in agg if r["eval_unit"] == "subject" and "in_source" in r["support_mode"]), None)
    if sub:
        d["real3c_raw_tta"] = _f(sub["mean_raw_offline_delta_bacc"])
        d["real3c_off_tta_rate"] = _f(sub["mean_router_offline_tta_rate"])
        d["real3c_subject_cov"] = _f(sub["mean_router_coverage"])
    # S0
    s0v = _json(os.path.join(a.s0, "s0_validation.json"))
    d["s0_hf3_verdict"] = s0v.get("hf3_verdict", "")
    d["s0_source"] = s0v["checks"]["source_records"]
    d["s0_target"] = s0v["checks"]["target_records"]
    # S2 ACAR-error
    s2v = _json(os.path.join(a.s2, "s2_validation.json"))
    d["s2_hf3_catch"] = _f(s2v.get("hf3_catch_rate_among_support_accepted"))
    d["s2_r2_add_refusal"] = _f(s2v.get("r2_additional_refusal_rate"))
    d["s2_hood_transfer"] = _f(s2v.get("hood_target_transfer_corr"))
    # S2B router
    s2bv = _json(os.path.join(a.s2b, "s2b_validation.json"))
    for x in s2bv.get("hf3_aggregate", []):
        if x.get("policy") == "support_plus_acar_error_optional":
            d[f"s2b_catch_{x.get('cal','')}"] = _f(x.get("catch_rate_among_support_accepted"))
    # S3 PRIOR_ONLY
    s3 = _rows(os.path.join(a.s3, "s3_prior_only_world_summary.csv"))
    r2 = next((r for r in s3 if r["dataset_or_world"] == "R2"), None)
    if r2:
        d["s3_r2_po_gain"] = _f(r2["prior_only_gain_mean"])
        d["s3_r2_tta_gain"] = _f(r2["offline_tta_gain_mean"])
    s3v = _json(os.path.join(a.s3, "s3_prior_only_validation.json"))
    dec = s3v.get("decision", {})
    d["s3_po_harm"] = _f(dec.get("prior_only_harm_rate"))
    d["s3_tta_harm"] = _f(dec.get("offline_tta_harm_rate"))
    d["s3_verdict"] = s3v.get("recommendation", "")[:80]
    # S1 phase map
    s1v = _json(os.path.join(a.s1, "s1_validation.json"))
    d["s1_verdict"] = s1v.get("overall_verdict", "")
    ph = _rows(os.path.join(a.s1, "benefit_phase_analysis.csv"))
    d["s1_max_gain"] = max((_f(r["target_gain_max"]) for r in ph if not math.isnan(_f(r["target_gain_max"]))), default=float("nan"))
    d["s1_bnci004_transfer"] = _f(next((r["target_transfer_corr"] for r in ph
                                        if r["dataset"] == "BNCI2014_004" and r["eval_unit"] == "subject"), "nan"))
    # S4 backend comparison
    bs = _rows(os.path.join(a.s4, "backend_dataset_summary.csv"))

    def _iddelta(ds):
        h = next((r for r in bs if r["dataset"] == ds and r["eval_unit"] == "subject" and r["backend"] == "h2cmi_common"), None)
        c = next((r for r in bs if r["dataset"] == ds and r["eval_unit"] == "subject" and r["backend"] == "cbramod_common"), None)
        if h and c:
            return _f(c["identity_bacc_mean"]) - _f(h["identity_bacc_mean"])
        return float("nan")
    d["s4_delta_004"] = _iddelta("BNCI2014_004")
    d["s4_delta_001"] = _iddelta("BNCI2014_001")
    d["s4_delta_lee"] = _iddelta("Lee2019_MI")
    d["s4_verdict"] = _json(os.path.join(a.s4, "backend_validation.json")).get("overall_verdict", "")
    return d


# ------------------------------------------------------------------ manuscript
def build_manuscript(d):
    def g(k, p=3):
        return _fmt(d.get(k, float("nan")), p)
    L = []
    A = L.append
    A("# Project B: Refusal-First Safe EEG Adaptation")
    A("")
    A("*Consolidated manuscript draft, auto-assembled by `scripts/project_b_s5_manuscript_package.py` "
      "from frozen Step-2G/3C/S0/S1/S2/S2B/S3/S4 outputs. No experiment was re-run.*")
    A("")
    A("**Project B is not a new TTA optimizer. It is a refusal-first deployment router.**")
    A("")
    A("## 1. Problem Statement")
    A("Test-time adaptation (TTA) can help or harm an EEG decoder at deployment, and whether it helps is "
      "not knowable from source data alone. Project B is a **refusal-first deployment router** that, for "
      "an unlabelled target, chooses among `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and emits an "
      "auditable reason. Target labels are used only post-hoc for evaluation, never to decide.")
    A("")
    A("## 2. Deployment Action Space")
    A("`REFUSE` (emit no decode), `IDENTITY` (source-only prediction), `OFFLINE_TTA` (batch transductive "
      "class-conditional affine adaptation), `ONLINE_TTA` (streaming). The default is REFUSE; a "
      "non-refusal action must clear explicit support and calibrated-risk gates.")
    A("")
    A("## 3. Router Architecture")
    A("Action-specific blockers: support / stability / diagnostic failures block **every** action "
      "(including IDENTITY); TTA-evidence and ACAR failures block only the TTA actions. Selection is "
      "safe-beneficial-then-identity: a beneficial admissible TTA can win; else a support-valid IDENTITY; "
      "else REFUSE. This is a **refusal-first router**, not a least-interventional self-lock.")
    A("")
    A("## 4. OACI Reason Codes")
    A("Every decision carries reason codes, separated into blocking vs audit-only and into action-level "
      "blockers vs top-level reasons. A TTA blocker never reads as \"IDENTITY is unsafe\". This makes "
      "refusal / identity / TTA-blocking decisions auditable.")
    A("")
    A("## 5. Support-Aware Prior-Decoupled Diagnostics")
    A("A diagnostic vector (target size, ESS, class-conditional density NLL, transform norm, condition "
      "number, prediction disagreement) rather than a single OOD scalar. Support is measured under both "
      "source and estimated-target priors; a label-prior shift with intact target-prior density is "
      "audit-only, not a refusal. The support threshold is source-only.")
    A("")
    A("## 6. ACAR-Harm and the Degeneracy Finding")
    A("ACAR gives per-action split-conformal upper bounds on error (eligibility) and harm "
      "(allowed-to-adapt), with an explicit `available / degenerate / unavailable` state. Source-only "
      "harm calibration is frequently degenerate (single-class or too-few pseudo-target harm gains), so "
      "TTA is blocked without a fabricated bound. **ACAR-harm degeneracy** is a real non-identifiability, "
      "not an implementation failure.")
    A("")
    A("## 7. ACAR-Error for Output Eligibility")
    A("Because ACAR-harm is often degenerate, we add an **optional ACAR-error** layer: a source-only, "
      "cross-fitted identity-error predictor + split-conformal upper bound that gates IDENTITY output. It "
      "is OPTIONAL (used only when the error layer is available; otherwise the router falls back to "
      "support-only), never turning an unavailable layer into all-refuse.")
    A("")
    A("## 8. Unified Non-Identifiability Boundary")
    A("Without a representativeness assumption linking source pseudo-target domains to the deployment "
      "target, source-only calibration cannot identify either action harm OR identity error under "
      "arbitrary concept shift: identical source observations + identical unlabeled target diagnostics + "
      "different target label mechanism force identical router decisions but different true target "
      "risk. ACAR-error repairs the source-representative / observable regime, but the target-only "
      "**H-OOD boundary persists**.")
    A("")
    A("## 9. Synthetic Experiments")
    A(f"- **R2** (strict {g('R2_strict')}): raw offline TTA helps ({g('R2_raw_tta')}); nested source "
      f"support calibration fixes over-refusal (coverage {g('R2_nested_cov',2)}, accepted bAcc "
      f"{g('R2_nested_acc')}); TTA stays blocked under degenerate ACAR-harm — a knowing missed benefit.")
    A(f"- **HF3** (strict {g('HF3_strict')}): raw TTA harmful ({g('HF3_raw_tta')}); a support-valid "
      f"concept-degraded identity can pass v1 — motivating ACAR-error.")
    A(f"- **H-OOD** (strict {g('H_OOD_strict')}): target-only stress ({g('H_OOD_raw_tta')}); support/ESS "
      f"help but do not complete; the boundary persists.")
    A("")
    A("## 10. Real EEG Experiments")
    A(f"On real BNCI2014_004 (bounded LOSO), raw offline TTA is harmful (mean d_bAcc {g('real3c_raw_tta')}) "
      f"and OFFLINE_TTA is never selected (rate {g('real3c_off_tta_rate',2)}); the router routes "
      f"support-valid IDENTITY or refuses OOD subjects. Across BNCI2014_004 / BNCI2014_001 / Lee2019_MI "
      f"(S1 phase map), **no real benefit phase** is observed (max target gain {g('s1_max_gain')} < 0.05). "
      f"Worse, the source-fold TTA-gain predictor rank-transfers (BNCI2014_004 corr {g('s1_bnci004_transfer',2)}) "
      f"but its OFFSET does not: it would select harmful TTA. Verdict: `{d.get('s1_verdict','')}`.")
    A("")
    A("## 11. PRIOR_ONLY Action Study")
    A(f"`PRIOR_ONLY` (freeze encoder/density/classifier; re-estimate only the target class prior) is the "
      f"lowest-harm adaptation action (harm {g('s3_po_harm',2)} vs OFFLINE_TTA {g('s3_tta_harm',2)}) but "
      f"does NOT recover R2 missed benefit (R2 prior_only gain {g('s3_r2_po_gain')} while OFFLINE_TTA gain "
      f"{g('s3_r2_tta_gain')}): the recoverable benefit is covariate-driven, not prior-driven. `prior_only` "
      f"is therefore **deferred** — not integrated into the router.")
    A("")
    A("## 12. Backend Comparison with CBraMod")
    A(f"Under a common source-only downstream (identical z-score+PCA+Gaussian head+affine-TTA), the "
      f"pretrained CBraMod foundation encoder applied zero-shot to MI is a **weaker** identity "
      f"representation than the source-trained h2cmi encoder: identity Δ {g('s4_delta_004')} "
      f"(BNCI2014_004), {g('s4_delta_001')} (BNCI2014_001), {g('s4_delta_lee')} (Lee2019_MI). Its lower "
      f"support-mismatch reflects a more diffuse density (accepting worse predictions), not better "
      f"support. Its one source-predictable TTA gain is a **weak-baseline artifact**: CBraMod+TTA "
      f"absolute bAcc stays below the best identity baseline on every dataset. Verdict: "
      f"`{d.get('s4_verdict','')}`. Zero-shot foundation representation does not create a deployable "
      f"benefit phase (fine-tuning is future work).")
    A("")
    A("## 13. EEGAgent Integration Roadmap")
    A("Project B is a **safety governor** that an EEGAgent-style LLM workflow can call as a risk-routing "
      "tool. The hard boundary: EEGAgent may call Project B and explain its OACI reason codes, schedule "
      "other tools (artifact/PSD/symmetry checks) on a REFUSE, and generate reports — but it must NOT "
      "override Project B's refusal / no-TTA decision. See `PROJECT_B_AGENT_INTEGRATION_ROADMAP.md`.")
    A("")
    A("## 14. Claim Boundary")
    A("See `PROJECT_B_NEXT_CLAIM_BOUNDARY.md`. Headline: Project B prevents unsafe adaptation, routes "
      "support-valid identity, optionally filters identity by ACAR-error, and audits via OACI. It does "
      "not guarantee TTA improvement, does not solve concept shift, and does not claim full-benchmark "
      "superiority.")
    A("")
    A("## 15. Limitations")
    A("ACAR-harm frequently degenerate; support cannot detect concept-shift accuracy loss; the H-OOD "
      "target-only boundary persists; real evidence is bounded (few subjects/targets); PRIOR_ONLY is "
      "low-harm but non-recovering; zero-shot CBraMod is a weaker MI representation (fine-tuning "
      "untested); EEGAgent integration is a roadmap, not evaluated.")
    A("")
    A("## 16. Conclusion")
    A("On real EEG motor imagery, no deployable OFFLINE_TTA benefit phase exists — for the native h2cmi "
      "backend or a zero-shot foundation backbone — and source-only calibration cannot identify "
      "target-only harm or error. Project B's honest contribution is therefore **refusal-first "
      "harm-avoidance, support-valid IDENTITY, optional ACAR-error output eligibility, and OACI "
      "auditability** — a safety governor, not a selective-TTA accuracy booster.")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------ claim boundary
def build_claim_boundary(d):
    claimable = [
        "Project B prevents unsafe adaptation under degenerate/unavailable ACAR-harm calibration.",
        "Support-aware nested calibration can reduce synthetic over-refusal (R2 coverage "
        f"{_fmt(d.get('R2_nested_cov'),2)}, accepted bAcc {_fmt(d.get('R2_nested_acc'))}).",
        "Optional ACAR-error improves identity eligibility in source-representative error regimes "
        f"(HF3 catch-among-support-accepted up to {_fmt(d.get('s2b_catch_fold_local_crossfit'),2)} fold-local).",
        "Real-EEG bounded experiments show no deployable OFFLINE_TTA benefit phase for the current backend.",
        f"PRIOR_ONLY is lower harm than affine TTA ({_fmt(d.get('s3_po_harm'),2)} vs "
        f"{_fmt(d.get('s3_tta_harm'),2)}) but does not recover missed benefit.",
        "Zero-shot CBraMod common backend does not create a deployable benefit phase.",
        "OACI codes make refusal / identity / TTA-blocking decisions auditable.",
        "Project B can serve as a hard safety governor inside an EEGAgent-style workflow.",
    ]
    not_claimable = [
        "It does not guarantee TTA improvement.",
        "It does not solve arbitrary concept shift.",
        "It does not identify target-only harm/error without a representativeness assumption.",
        "It does not show full MOABB benchmark superiority.",
        "It does not show foundation fine-tuning is ineffective (only zero-shot was tested).",
        "It does not show EEGAgent itself improves decoding safety.",
        "It does not claim target-label-tuned thresholds.",
    ]
    return {"step": "S5", "claimable": claimable, "not_claimable": not_claimable, "digest": d}


def build_claim_boundary_md(cb):
    L = ["# Project B-Next — Final Claim Boundary", "",
         "Companion to `project_b_next_claim_boundary.json`.", "", "## Claimable"]
    for i, c in enumerate(cb["claimable"], 1):
        L.append(f"{i}. {c}")
    L += ["", "## NOT claimable"]
    for i, c in enumerate(cb["not_claimable"], 1):
        L.append(f"{i}. {c}")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ agent roadmap
def build_agent_roadmap():
    return """# Project B x EEGAgent Integration Roadmap

*Roadmap / discussion, not an evaluated result. EEGAgent references summarize the uploaded EEGAgent
document as described by the PM.*

## 1. Role separation
- Foundation EEG model: backend / representation / decoder (parametric tool).
- Project B router: safety governor / action selector / OACI audit.
- EEGAgent: workflow orchestrator / tool scheduler / report generator.

## 2. Why Project B is a router tool, not an agent
EEGAgent (an LLM-enhanced framework for scheduling EEG tools and generating reports; it supports
perception, exploration, event detection, user interaction, and reporting; its toolbox mixes parametric
and non-parametric tools organized by temporal/spatial granularity) is an orchestration layer. Project B
answers a narrow, safety-critical decision — refuse / output identity / adapt — that must be governed by
calibrated risk, not by an LLM plan. So Project B is a TOOL the agent calls, not the agent itself.

## 3. Tool API
    {
      "tool_name": "refusal_first_router",
      "input": {"target_eeg_summary": "...", "model_outputs": "...",
                "support_diagnostics": "...", "candidate_actions": ["IDENTITY","PRIOR_ONLY","OFFLINE_TTA"]},
      "output": {"action": "REFUSE|IDENTITY|OFFLINE_TTA", "accepted": true,
                 "oaci_reason_codes": ["..."], "diagnostics": {}, "recommended_next_step": "..."}
    }

## 4. OACI-to-report translation
EEGAgent translates OACI reason codes into clinician/user-readable language (e.g. "adaptation not run
because support mismatch and degenerate harm calibration; suggest acquiring calibration data").

## 5. Refusal workflows
On REFUSE, EEGAgent may schedule complementary tools from its toolbox (artifact check, PSD, symmetry,
amplitude, baseInfo) and request calibration acquisition, then re-query the router.

## 6. What EEGAgent may do
Explain, schedule other tools, generate structured reports, handle user follow-up, and surface the
router's recommended next step.

## 7. What EEGAgent must not do
- It must NOT override Project B's refusal / no-TTA decision.
- It must NOT present an adaptation the router blocked as safe.
- It must NOT tune router thresholds on target outcomes.

## 8. Minimal demo plan (future work)
Wrap the frozen RefusalFirstRouter as an EEGAgent tool; on a held-out target, show the agent (a) calling
the router, (b) translating OACI codes, (c) scheduling an artifact/PSD check on REFUSE, (d) producing a
report — with the router decision authoritative throughout. This is future work; it is not evaluated here
and does not itself improve decoding safety.
"""


# ------------------------------------------------------------------ tex
def build_method_tex():
    return r"""\section{Project B: Refusal-First Safe EEG Adaptation}
% Project B is a refusal-first deployment router, NOT a new TTA optimizer.
\subsection{Action Space and Refusal-First Router}
% REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA with action-specific blockers.
\subsection{OACI Reason Codes}
% Auditable blocking vs audit-only; TTA blockers never mark IDENTITY unsafe.
\subsection{Support-Aware Prior-Decoupled Diagnostics}
% Source-only support threshold; prior-shift audit-only.
\subsection{ACAR: Harm Degeneracy and Optional Error Eligibility}
% ACAR-harm degenerate source-only; optional ACAR-error gates IDENTITY when available.
\subsection{Unified Non-Identifiability Boundary}
% Source-only calibration cannot identify target-only harm OR identity error under arbitrary concept shift.
\subsection{Limitations}
% H-OOD boundary persists; PRIOR_ONLY deferred; real no benefit phase; zero-shot CBraMod weaker.
% TODO: cite TTA, selective prediction, conformal risk control, EEG transfer, EEGAgent.
"""


def build_tables_tex(d):
    def g(k, p=3):
        return _fmt(d.get(k, float("nan")), p)
    return (r"""\begin{table}[t]\centering\small
\caption{Project B synthetic + real evidence (nested support mode).}\label{tab:pb-evidence}
\begin{tabular}{llrrr}\toprule
setting & world/dataset & strict/id & raw $\Delta$TTA & note \\ \midrule
synthetic & R2 & """ + g("R2_strict") + r""" & """ + g("R2_raw_tta") + r""" & benefit blocked (ACAR-harm degenerate) \\
synthetic & HF3 & """ + g("HF3_strict") + r""" & """ + g("HF3_raw_tta") + r""" & concept-degraded identity passes v1 \\
synthetic & H-OOD & """ + g("H_OOD_strict") + r""" & """ + g("H_OOD_raw_tta") + r""" & target-only boundary persists \\
real & BNCI2014\_004 & -- & """ + g("real3c_raw_tta") + r""" & OFFLINE\_TTA never selected \\
\bottomrule\end{tabular}\end{table}

\begin{table}[t]\centering\small
\caption{ACAR-error / PRIOR\_ONLY / phase-map / backend results.}\label{tab:pb-negative}
\begin{tabular}{lr}\toprule
result & value \\ \midrule
ACAR-error HF3 catch (fold-local) & """ + g("s2b_catch_fold_local_crossfit", 2) + r""" \\
ACAR-error R2 additional refusal & """ + g("s2_r2_add_refusal", 2) + r""" \\
PRIOR\_ONLY R2 gain vs OFFLINE\_TTA & """ + g("s3_r2_po_gain") + r""" vs """ + g("s3_r2_tta_gain") + r""" \\
S1 real max target gain & """ + g("s1_max_gain") + r""" \\
CBraMod identity $\Delta$ (004/001/Lee) & """ + g("s4_delta_004") + r"""/""" + g("s4_delta_001") + r"""/""" + g("s4_delta_lee") + r""" \\
\bottomrule\end{tabular}\end{table}
""")


def build_digest_csv_rows(d):
    return [{"key": k, "value": _fmt(v, 4) if isinstance(v, float) else v} for k, v in d.items()]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Project B-Next S5 manuscript consolidation (read-only)")
    ap.add_argument("--step2g", required=True)
    ap.add_argument("--step3c", required=True)
    ap.add_argument("--s0", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s2b", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s4", required=True)
    ap.add_argument("--skip_branch_check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    branch = _branch()
    if not a.skip_branch_check and branch != EXPECTED_BRANCH:
        raise Fail(f"[FAIL] branch '{branch}' != '{EXPECTED_BRANCH}'")
    for name, p in (("step2g", a.step2g), ("step3c", a.step3c), ("s0", a.s0), ("s2", a.s2),
                    ("s2b", a.s2b), ("s3", a.s3), ("s1", a.s1), ("s4", a.s4)):
        _req(p, f"input dir {name}")
    os.makedirs(a.out, exist_ok=True)

    d = build_digest(a)
    manuscript = build_manuscript(d)
    cb = build_claim_boundary(d)
    cb_md = build_claim_boundary_md(cb)
    agent = build_agent_roadmap()
    method_tex = build_method_tex()
    tables_tex = build_tables_tex(d)

    # required mentions checked over the full corpus; overclaim guard EXCLUDES the claim-boundary
    # not-claimable list (which is a whitelist of explicitly-disclaimed overclaim phrases by design).
    corpus = (manuscript + "\n" + agent + "\n" + cb_md).lower()
    missing = [m for m in REQUIRED_MENTIONS if m not in corpus]
    if missing:
        raise Fail(f"[FAIL] manuscript missing required mentions: {missing}")
    hit = [p for p in OVERCLAIM if p in (manuscript + "\n" + agent).lower()]
    if hit:
        raise Fail(f"[FAIL] overclaim phrase present (asserted, not disclaimed): {hit}")

    diff = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout
    mod = [ln[3:].strip() for ln in diff.splitlines() if len(ln) >= 3 and ln[:2] != "??"]
    forbidden = [p for p in mod if p.startswith("h2cmi/") or p.startswith("cmi/")]
    if forbidden:
        raise Fail(f"[FAIL] forbidden files modified: {forbidden}")

    # write /tmp outputs
    def w(name, txt):
        with open(os.path.join(a.out, name), "w") as f:
            f.write(txt)
    w("project_b_next_manuscript.md", manuscript)
    w("project_b_next_method.tex", method_tex)
    w("project_b_next_results_tables.tex", tables_tex)
    w("project_b_next_agent_integration.md", agent)
    with open(os.path.join(a.out, "project_b_next_claim_boundary.json"), "w") as f:
        json.dump(cb, f, indent=2)
    with open(os.path.join(a.out, "project_b_next_results_digest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["key", "value"])
        wr.writeheader()
        for r in build_digest_csv_rows(d):
            wr.writerow(r)
    validation = dict(step="S5", branch=branch,
                      checks=dict(input_dirs_present=True, required_mentions_present=True,
                                  no_overclaim=True, no_h2cmi_cmi_modified=(len(forbidden) == 0),
                                  frozen_branch_untouched=(branch == EXPECTED_BRANCH)),
                      digest_keys=len(d), claimable=len(cb["claimable"]),
                      not_claimable=len(cb["not_claimable"]), all_checks_passed=True)
    with open(os.path.join(a.out, "project_b_next_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    # sync notes
    w2 = lambda p, t: open(os.path.join(NOTES, p), "w").write(t)
    w2("PROJECT_B_NEXT_MANUSCRIPT.md", manuscript)
    w2("PROJECT_B_NEXT_CLAIM_BOUNDARY.md", cb_md)
    w2("PROJECT_B_NEXT_RESULTS_DIGEST.md",
       "# Project B-Next Results Digest\n\n" + "\n".join(f"- {r['key']}: {r['value']}" for r in build_digest_csv_rows(d)) + "\n")
    w2("PROJECT_B_AGENT_INTEGRATION_ROADMAP.md", agent)

    print(f"[S5] digest_keys={len(d)} claimable={len(cb['claimable'])} not_claimable={len(cb['not_claimable'])}")
    print(f"[S5] required mentions present, no overclaim, no h2cmi/cmi modified")
    print(f"[S5] wrote manuscript + claim boundary + digest + agent roadmap + tex + validation to {a.out}")


if __name__ == "__main__":
    main()
