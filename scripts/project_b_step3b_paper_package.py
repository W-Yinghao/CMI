"""Project B Step-3B: assemble a paper-section draft + claim boundary from FROZEN evidence.

READ-ONLY over the Step-2G synthetic tables and the Step-3A real-EEG bridge outputs. Trains nothing,
reruns nothing, and modifies no h2cmi/** or cmi/** file. Emits a markdown draft, a reviewer checklist,
a real-bridge result note, LaTeX fragments, a merged claim-boundary, and a real-bridge table; also
syncs the three prose docs into notes/. Fails loudly on missing inputs, on any overclaim phrase in the
generated draft, or if a required limitation is not explicitly stated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

WORLDS = ["R2", "HF3", "H_OOD"]
OVERCLAIM_PHRASES = ["guarantees TTA improvement", "solves concept shift", "detects all OOD",
                     "full MOABB benchmark", "target-label tuned"]
REQUIRED_MENTIONS = ["post-hoc", "ACAR-harm", "degenerat", "missed benefit",
                     "concept-degraded identity", "LOW_ESS", "not a full benchmark"]

REPO_NOTES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- load frozen evidence
def load_evidence(step2g, step3a):
    need2g = ["table1_world_support_summary.csv", "table2_component_ablation.csv",
              "claim_boundary.json"]
    for fn in need2g:
        if not os.path.exists(os.path.join(step2g, fn)):
            sys.exit(f"[FAIL] missing Step-2G {fn}")
    fold_csv = os.path.join(step3a, "fold_summary.csv")
    avail = os.path.join(step3a, "availability_error.json")
    if not os.path.exists(fold_csv) and not os.path.exists(avail):
        sys.exit(f"[FAIL] missing Step-3A fold_summary.csv (and no availability_error.json) in {step3a}")

    t1 = {(r["world"], r["support_mode"]): r for r in _read_csv(os.path.join(step2g, "table1_world_support_summary.csv"))}
    have_worlds = {w for (w, _m) in t1}
    if not set(WORLDS) <= have_worlds:
        sys.exit(f"[FAIL] Step-2G missing worlds: {set(WORLDS) - have_worlds}")
    cb = json.load(open(os.path.join(step2g, "claim_boundary.json")))

    real = None
    real_status = "unavailable"
    if os.path.exists(fold_csv):
        folds = _read_csv(fold_csv)
        summ = json.load(open(os.path.join(step3a, "real_bridge_summary.json"))) \
            if os.path.exists(os.path.join(step3a, "real_bridge_summary.json")) else {}
        real = dict(folds=folds, summary=summ)
        real_status = summ.get("status", "ok" if folds else "unavailable")
        if real_status not in ("ok", "partial") and not os.path.exists(avail):
            sys.exit("[FAIL] real bridge status not ok/partial and no availability_error.json")
    return t1, cb, real, real_status


# --------------------------------------------------------------------------- draft builders
def _syn(t1, world):
    b = t1[(world, "in_source_subject_q95")]
    n = t1[(world, "nested_site_excess_q95")]
    return b, n


def build_draft(t1, cb, real, real_status):
    R2b, R2n = _syn(t1, "R2")
    HFb, HFn = _syn(t1, "HF3")
    OOb, OOn = _syn(t1, "H_OOD")

    real_lines = []
    if real and real["folds"]:
        for r in real["folds"]:
            real_lines.append(
                f"  - target {r['target_subject']} ({r['support_mode']}): strict bAcc "
                f"{_f(r['strict_bacc']):.3f}, raw offline TTA dbAcc {_f(r['raw_offline_delta_bacc']):+.3f}, "
                f"action(s) {r['action_counts']}, ACAR-harm {r['source_acar_harm_state']}.")
        sm = real["summary"]
        real_head = (f"On real **{sm.get('dataset','BNCI2014_004')}** (X shape {sm.get('X_shape')}, classes "
                     f"{sm.get('classes')}, subjects {sm.get('subjects')}, targets {sm.get('targets')}), the "
                     f"bridge ran end-to-end under label-safe LOSO. Target labels were used only **post-hoc**.")
    else:
        real_head = "The real-EEG dataset was unavailable; the bridge degraded cleanly (availability_error.json)."

    md = f"""# Project B: Refusal-First Safe EEG Adaptation

*Draft auto-assembled by `scripts/project_b_step3b_paper_package.py` from frozen Step-2G synthetic
tables and the Step-3A real-EEG bridge outputs. No experiment was re-run.*

## 1. Problem Statement
Test-time adaptation (TTA) can help or harm an EEG decoder at deployment, and whether it helps is not
knowable from source data alone. Project B is **not** another EEG TTA loss; it is a **deployment
router** that, for an unlabelled target, chooses among `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA`
and emits an auditable reason. Target labels are used only **post-hoc** for evaluation, never to decide.

## 2. Deployment Action Space
`REFUSE` (emit no decode), `IDENTITY` (source-only prediction), `OFFLINE_TTA` (batch transductive
class-conditional affine adaptation), `ONLINE_TTA` (streaming). The router is refusal-first: the
default is REFUSE, and a non-refusal action must clear explicit support and calibrated-risk gates.

## 3. Refusal-First Router
Action-specific blockers: support/stability/diagnostic failures block *every* action (including
IDENTITY); TTA-evidence and ACAR-harm failures block only the TTA actions. Selection is
safe-beneficial-then-identity: a beneficial admissible TTA can win; otherwise a support-valid IDENTITY;
otherwise REFUSE. This avoids a least-interventional self-lock while never adapting on unsafe grounds.

## 4. TOS / Support-Aware Diagnostics
A vector (target size, effective sample size, class-conditional density NLL, transform norm, condition
number, prediction disagreement) rather than a single OOD scalar. A too-few-target / low-ESS /
support-mismatch / unstable-transform condition each maps to a distinct OACI reason.

## 5. Prior-Decoupled Support Protocol
Support is measured under both the source prior and an estimated target prior; a label-prior shift with
intact target-prior density is recorded as audit-only info, **not** a refusal. The support threshold is
source-only (§8): baseline = q95 of in-source-unit target-prior NLL; the nested variant adds a
scale-normalised held-out-unit *excess* to the base scale.

## 6. ACAR: Action-Conditional Conformal Adaptation Risk
Per action, a split-conformal upper bound on *error* (eligibility) and *harm* (allowed-to-adapt),
calibrated on externally-supplied risk predictions over source pseudo-targets. Critically, ACAR
explicitly represents `available / degenerate / unavailable`: when the source pseudo-target harm gains
are single-class (degenerate) or too few (unavailable), no harm bound is produced and TTA is blocked.

## 7. OACI Reason Codes
Every decision carries reason codes, separated into blocking vs audit-only, and into action-level
blockers vs top-level decision reasons. A TTA blocker (e.g. `ACAR-harm` degeneracy, negative TTA
evidence) never reads as "IDENTITY is unsafe".

## 8. Synthetic Protocol
A controllable EEG simulator with orthogonal shift knobs and a hierarchical site/subject/session DAG.
Three locked worlds: **R2** (recoverable), **HF3** (harmful / concept-shift, source-calibratable
attempt), **H-OOD** (target-only stress). Full training (no fast config). Two source-only support
modes: `in_source_subject_q95` (baseline) and `nested_site_excess_q95`.

## 9. Synthetic Results
- **R2** (strict {_f(R2b['strict_bacc_mean']):.3f}): raw offline TTA helps on average
  ({_f(R2b['raw_offline_delta_bacc_mean']):+.3f}). The Step-2E in-source support threshold
  **over-refused** (coverage {_f(R2b['coverage_mean']):.2f}). Nested source-site excess calibration
  fixed this: coverage {_f(R2b['coverage_mean']):.2f} -> {_f(R2n['coverage_mean']):.2f}, accepted bAcc
  {_f(R2n['accepted_bacc_mean']):.3f}. TTA remained blocked because ACAR-harm is
  {R2n['acar_harm_state']}, so raw-TTA's benefit is a knowing **missed benefit**, not a policy bug.
- **HF3** (strict {_f(HFb['strict_bacc_mean']):.3f}): raw offline TTA is harmful on average
  ({_f(HFb['raw_offline_delta_bacc_mean']):+.3f}); the router blocks OFFLINE_TTA under ACAR-harm
  degeneracy. Nested mode accepts some IDENTITY domains (coverage {_f(HFn['coverage_mean']):.2f}); a
  **concept-degraded identity** can pass source-only support checks (accepted bAcc
  {_f(HFn['accepted_bacc_mean']):.3f}).
- **H-OOD** (strict {_f(OOb['strict_bacc_mean']):.3f}): raw offline TTA is harmful
  ({_f(OOb['raw_offline_delta_bacc_mean']):+.3f}). After nested widening the density
  `SUPPORT_MISMATCH` clears ({OOb['support_mismatch_domains_total']} -> {OOn['support_mismatch_domains_total']}
  domains), but **LOW_ESS** remains the active blocker for {OOn['low_ess_domains_total']} of the target
  domains, and OFFLINE_TTA is never selected.

## 10. Real-EEG Bridge Result
{real_head}
{os.linesep.join(real_lines)}
This is a **bridge smoke, not a full benchmark**: the router posture (block TTA when harm is
uncalibratable, accept support-valid IDENTITY) reproduces on real EEG. On this small bridge the nested
source-subject excess was 0, so nested == baseline (real subjects were less OOD than synthetic held-out
sites).

## 11. Claim Boundary
Claimable: (i) the router prevents TTA when source-only harm calibration is degenerate/unavailable;
(ii) nested source-held-out support excess calibration fixes the synthetic all-refuse over-refusal on
R2; (iii) it allows support-valid IDENTITY while refusing low-ESS targets; (iv) OACI reason codes
separate the refusal sources; (v) the real-EEG bridge runs end-to-end under label-safe LOSO.
NOT claimable: we do not claim source-only ACAR-harm is generally identifiable; we do not claim v1
recovers beneficial TTA under harm-calibration degeneracy; we do not claim support-valid IDENTITY is
accurate under concept shift; we do not claim density support alone catches H-OOD after nested
widening; we do not claim a complete MOABB benchmark from Step-3A; thresholds are source-only, never
tuned on target labels.

## 12. Limitations
- ACAR-harm is frequently degenerate/unavailable source-only, so v1 forgoes beneficial TTA (R2
  **missed benefit**).
- Source-only support cannot detect concept-shift accuracy loss (**concept-degraded identity** passes
  on HF3).
- Nested widening can clear the density support signal; **LOW_ESS** is then the only active support
  blocker (H-OOD).
- The real-EEG evidence is a bridge smoke (few subjects/targets, coarse eval unit), **not a full
  benchmark**.

## 13. Next Benchmark Expansion
More subjects and targets, session-level deployment routing, and additional datasets
(BNCI2014_001 / Lee2019_MI), keeping the source-only, label-safe protocol. Threshold and harm
calibration behavior at scale is an empirical question, not an assumption.
"""
    return md


def build_real_bridge_note(real, real_status):
    if not (real and real["folds"]):
        return "# Project B Step-3A Real-EEG Bridge Result\n\nData unavailable; bridge degraded cleanly.\n"
    sm = real["summary"]
    lines = ["# Project B Step-3A Real-EEG Bridge Result", "",
             f"Dataset **{sm.get('dataset')}**, X shape {sm.get('X_shape')}, classes {sm.get('classes')}, "
             f"subjects {sm.get('subjects')}, targets {sm.get('targets')}, eval_unit {sm.get('eval_unit')}, "
             f"status **{real_status}**. Target labels used only **post-hoc**; this is a bridge smoke, "
             f"**not a full benchmark**.", "",
             "| target | mode | strict | raw dTTA | coverage | action | ACAR-harm | avoided_harm |",
             "|---|---|---|---|---|---|---|---|"]
    for r in real["folds"]:
        lines.append(f"| {r['target_subject']} | {r['support_mode']} | {_f(r['strict_bacc']):.3f} | "
                     f"{_f(r['raw_offline_delta_bacc']):+.3f} | {_f(r['router_coverage']):.2f} | "
                     f"{r['action_counts']} | {r['source_acar_harm_state']} | {_f(r['router_avoided_harm']):.3f} |")
    lines += ["", "On both held-out targets raw offline TTA was harmful; the router blocked OFFLINE_TTA "
              "(ACAR-harm degenerate/unavailable) and accepted support-valid IDENTITY, avoiding the harm. "
              "Nested source-subject excess was 0, so nested == baseline on this smoke."]
    return os.linesep.join(lines) + os.linesep


def build_reviewer_checklist():
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
Answer: A BNCI2014_004 LOSO **bridge smoke** (not a full benchmark): raw TTA harmful on two targets;
the router blocks TTA and accepts support-valid identity.

## Q7. What remains for benchmark expansion?
Answer: More subjects, more targets, session-level routing, and additional datasets.
"""


def build_method_tex():
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
\subsection{Real-EEG Bridge}
% BNCI2014_004 LOSO bridge smoke: TTA harmful on held-out targets; router blocks TTA, accepts identity.
\subsection{Limitations}
% ACAR-harm degeneracy, concept-degraded identity, LOW_ESS-only support signal, bridge is not a full benchmark.
% TODO: cite TTA, selective prediction, conformal risk control, EEG transfer learning.
"""


def build_merged_claim_boundary(cb, real, real_status):
    merged = dict(cb)
    merged["real_bridge"] = dict(
        dataset=(real["summary"].get("dataset") if real else None),
        status=real_status,
        note="Bridge smoke on BNCI2014_004 LOSO; raw offline TTA harmful on held-out targets; router "
             "blocked TTA (ACAR-harm degenerate/unavailable) and accepted support-valid identity; "
             "not a full benchmark.")
    merged.setdefault("not_claimable", []).append(
        "We do not claim a complete MOABB benchmark from the Step-3A bridge smoke.")
    return merged


def build_real_table_csv(real):
    cols = ["target_subject", "support_mode", "strict_bacc", "raw_offline_delta_bacc",
            "router_coverage", "router_identity_rate", "router_offline_tta_rate",
            "router_accepted_bacc", "router_avoided_harm", "source_acar_harm_state",
            "support_threshold_nll_target_prior", "base_source_q95_nll_target_prior", "nested_excess_q95"]
    rows = real["folds"] if real else []
    return cols, [{c: r.get(c) for c in cols} for r in rows]


# --------------------------------------------------------------------------- validation + write
def validate_draft(draft: str):
    lo = draft.lower()
    hits = [p for p in OVERCLAIM_PHRASES if p.lower() in lo]
    if hits:
        sys.exit(f"[FAIL] draft contains overclaim phrase(s): {hits}")
    missing = [m for m in REQUIRED_MENTIONS if m.lower() not in lo]
    if missing:
        sys.exit(f"[FAIL] draft missing required mention(s): {missing}")
    print(f"[validate] no overclaim phrases; all {len(REQUIRED_MENTIONS)} required mentions present")


def main():
    ap = argparse.ArgumentParser(description="Project B Step-3B paper package")
    ap.add_argument("--step2g", required=True)
    ap.add_argument("--step3a", required=True)
    ap.add_argument("--out", default="/tmp/project_b_step3b_paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t1, cb, real, real_status = load_evidence(args.step2g, args.step3a)
    draft = build_draft(t1, cb, real, real_status)
    validate_draft(draft)

    real_note = build_real_bridge_note(real, real_status)
    checklist = build_reviewer_checklist()
    method_tex = build_method_tex()
    merged_cb = build_merged_claim_boundary(cb, real, real_status)
    rt_cols, rt_rows = build_real_table_csv(real)

    # copy Step-2G paper tables verbatim as the tables fragment
    tables_tex = ""
    src_tex = os.path.join(args.step2g, "paper_tables.tex")
    if os.path.exists(src_tex):
        tables_tex = open(src_tex).read()

    writes = {
        os.path.join(args.out, "project_b_paper_draft.md"): draft,
        os.path.join(args.out, "project_b_method_results.tex"): method_tex,
        os.path.join(args.out, "project_b_tables.tex"): tables_tex,
        os.path.join(args.out, "project_b_reviewer_checklist.md"): checklist,
        os.path.join(REPO_NOTES, "PROJECT_B_PAPER_DRAFT.md"): draft,
        os.path.join(REPO_NOTES, "PROJECT_B_STEP3A_REAL_BRIDGE_RESULT.md"): real_note,
        os.path.join(REPO_NOTES, "PROJECT_B_REVIEWER_CHECKLIST.md"): checklist,
    }
    for path, content in writes.items():
        with open(path, "w") as f:
            f.write(content)
    with open(os.path.join(args.out, "project_b_claim_boundary_merged.json"), "w") as f:
        json.dump(merged_cb, f, indent=2)
    with open(os.path.join(args.out, "project_b_real_bridge_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(rt_cols)
        for r in rt_rows:
            w.writerow([("" if r.get(c) is None else r.get(c)) for c in rt_cols])

    print(f"[step3b] wrote paper package to {args.out} and synced 3 docs to {REPO_NOTES}")
    print(f"[step3b] real bridge status = {real_status}; "
          f"claimable={len(merged_cb.get('claimable', []))} not_claimable={len(merged_cb.get('not_claimable', []))}")


if __name__ == "__main__":
    main()
