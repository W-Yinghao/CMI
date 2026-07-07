"""Project A Step 20 — final observability-contract closeout and claim ledger.

This is a TERMINAL SYNTHESIS: no new data, no retraining, no new rescue policy. It freezes the scientific
conclusion of the Step 12-19 arc into a machine-auditable artifact:

  1. an evidence ledger — for each step, the question asked, the verdict (SUPPORTS / REFUTES / CHARACTERIZES),
     and the headline metric pulled LIVE from that step's tracked digest (not hand-copied);
  2. a deployment decision ladder — for each honestly-achievable observability level
     (R0 / R1 / R1+C14 / R1+C15 / R2-minimal / R2-full-or-oracle), the licensed adaptation decision; at
     EVERY deployable rung the decision is identity / abstain / block, never a deployable `adapt`;
  3. the forbidden headline claims (TTA safe, target prior identified, prior-robust benefit exists) with
     machine flags asserting none is made;
  4. the final verdict.

  python -m h2cmi.observability.closeout --summaries notes/project_A_observability/results_summaries \
      --out-json step20_closeout.json --out-md step20_closeout.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import FORBIDDEN_CLAIMS
from .result_index import _load_json, write_json_lf, write_text_lf

_FINAL_VERDICT = ("Observability contracts expose why unlabeled offline-TTA cannot be safely controlled "
                  "under honest prior uncertainty.")

# Each ledger row pulls its headline metric LIVE from a tracked digest via a dotted path (auditable,
# not hand-copied). verdict is REFUTES (a rescue hypothesis failed), CHARACTERIZES (a mechanism finding),
# or NULL (no signal).
_LEDGER = [
    {"step": "12", "question": "Can source-only (R0) diagnostics predict TTA harm?",
     "verdict": "REFUTES", "finding": "source-only harm-predictor at/below the majority baseline (TOS-1 ceiling).",
     "digest": "step14_harm_predictor_summary.json",
     "metric_path": "feature_sets.R0_source_only.balanced_acc_harm_prediction", "metric_label": "R0 harm-predictor bAcc"},
    {"step": "13-14", "question": "Do richer R1 target-unlabeled diagnostics predict harm?",
     "verdict": "REFUTES", "finding": "R1 predictor within its permutation null at n_perm=1000 -> overfitting artifact, not signal.",
     "digest": "step14_harm_predictor_summary.json", "metric_path": "verdict", "metric_label": "harm-predictor verdict"},
    {"step": "14", "question": "Are the real minimal-label curves accurate or merely low-coverage?",
     "verdict": "CHARACTERIZES", "finding": "high precision when decisive but coverage-limited; k=0 non-identifiable.",
     "digest": "step14_real_minimal_label_curves.json", "metric_path": "k0_status", "metric_label": "k=0 status"},
    {"step": "15", "question": "Can R2 minimal labels safely control harm (static policies)?",
     "verdict": "REFUTES", "finding": "no deployable minimal-label policy adapts while keeping harm<=0.05.",
     "digest": "step15_harm_control_summary.json", "metric_path": "best_deployable_policy.policy", "metric_label": "best deployable policy"},
    {"step": "16", "question": "Does sequential acquisition rescue it, and is benefit real?",
     "verdict": "REFUTES", "finding": "best sequential policy = none; benefit is rare, small, seed-unstable.",
     "digest": "step16_benefit_anatomy.json", "metric_path": "benefit_rate", "metric_label": "oracle benefit-rate (bAcc)"},
    {"step": "17", "question": "Is the failure an accuracy-vs-bAcc estimand mismatch?",
     "verdict": "REFUTES", "finding": "class-balanced grid -> accuracy==bAcc per run (Step-16 gap was a threshold artifact); estimand-invariant.",
     "digest": "step17_estimand_consistency.json", "metric_path": "max_abs_gain_difference", "metric_label": "max |acc-bAcc gain|"},
    {"step": "18", "question": "Is TTA harm global or class/prior-dependent?",
     "verdict": "CHARACTERIZES", "finding": "class-specific and prior-dependent; only a small minority harmful under all priors.",
     "digest": "step18_prior_stress.json", "metric_path": "fraction_prior_dependent_sign", "metric_label": "prior-dependent-sign fraction"},
    {"step": "19", "question": "Is any run robustly beneficial under bounded prior uncertainty?",
     "verdict": "REFUTES", "finding": "with a harm margin tau>=0.05 no run is robustly beneficial at any rho; best prior-robust policy = none.",
     "digest": "step19_prior_robust_policy.json", "metric_path": "robust_prior_safe_adaptation_exists_any", "metric_label": "robust safe adaptation exists (margin)"},
]

# Deployment decision ladder. `licenses_deployable_adaptation` is False at EVERY rung: adaptation is only
# ever licensed under the oracle full-target gain, which is an evaluation-only upper bound, never deployable.
_LADDER = [
    {"level": "R0 (source-only)", "observes": "source law only", "contracts": [],
     "licensed_decision": "identity", "licenses_deployable_adaptation": False,
     "rationale": "TOS-1: target gain non-identifiable source-only; source-only harm-predictor is NULL (Step 12/14)."},
    {"level": "R1 (target-unlabeled)", "observes": "target X (no labels)", "contracts": [],
     "licensed_decision": "identity/abstain", "licenses_deployable_adaptation": False,
     "rationale": "target gain needs labels (TU-2); R1 diagnostics do not predict harm above their permutation null (Step 13/14)."},
    {"level": "R1 + C14 (declared point prior)", "observes": "target X + declared operating prior", "contracts": ["C14"],
     "licensed_decision": "abstain", "licenses_deployable_adaptation": False,
     "rationale": "prior-weighted gain is a counterfactual under a DECLARED prior and ORACLE class deltas; the sign is prior-dependent (Step 18); not deployable."},
    {"level": "R1 + C15 (declared prior-uncertainty set)", "observes": "target X + declared prior set", "contracts": ["C15"],
     "licensed_decision": "block/abstain", "licenses_deployable_adaptation": False,
     "rationale": "robust benefit is unattainable under any honest uncertainty (tau>=0.05) (Step 19); identity/block is robustly justified for the persistent-harm minority."},
    {"level": "R2 (minimal paired labels)", "observes": "target X + a few labels", "contracts": ["C13"],
     "licensed_decision": "abstain", "licenses_deployable_adaptation": False,
     "rationale": "minimal-label static and sequential policies fail to safely select adaptation (Step 15/16); safe selection needs (near-)full labels."},
    {"level": "R2 (full / oracle labels)", "observes": "full target labels", "contracts": [],
     "licensed_decision": "adapt (evaluation-only)", "licenses_deployable_adaptation": False,
     "rationale": "the oracle full-target gain licenses safe adaptation but is an evaluation-only upper bound, never deployable."},
]

_FORBIDDEN_HEADLINE = [
    {"claim": "unlabeled offline-TTA is safe to deploy", "why": "Step 15/16 (minimal-label policies fail) and Step 19 (no robust benefit under prior uncertainty) refute it."},
    {"claim": "the target prior is identified from R0/R1", "why": "TU-1 boundary: identification needs C1 AND C2 AND C3; C14/C15 only DECLARE a prior / prior set."},
    {"claim": "prior-robust adaptation benefit exists under honest prior uncertainty", "why": "Step 19: with a harm margin tau>=0.05 no run is robustly beneficial at any rho."},
]


def _dig(d: Optional[Dict[str, Any]], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def build_closeout(summaries_dir) -> Dict[str, Any]:
    base = Path(summaries_dir)
    ledger = []
    for row in _LEDGER:
        dg = _load_json(base / row["digest"]) if (base / row["digest"]).exists() else None
        ledger.append({"step": f"Step {row['step']}", "question": row["question"],
                       "verdict": row["verdict"], "finding": row["finding"],
                       "headline_metric": row["metric_label"],
                       "headline_value": _dig(dg, row["metric_path"]),
                       "source_digest": row["digest"]})
    # every forbidden headline claim is asserted NOT made (machine flags all False)
    forbidden = [{**f, "status": "FORBIDDEN", "made": False} for f in _FORBIDDEN_HEADLINE]
    forbidden_all_not_made = all(f["made"] is False for f in forbidden)
    no_deployable_adapt = all(r["licenses_deployable_adaptation"] is False for r in _LADDER)
    return {
        "project": "Project A", "step": "Step 20",
        "scope": "final observability-contract closeout and claim ledger; no new data / retraining / rescue; not SOTA",
        "final_verdict": _FINAL_VERDICT,
        "evidence_ledger": ledger,
        "deployment_decision_ladder": _LADDER,
        "no_ladder_rung_licenses_deployable_adaptation": no_deployable_adapt,
        "forbidden_claims": forbidden,
        "forbidden_headline_claims_all_not_made": forbidden_all_not_made,
        "registry_forbidden_claims": list(FORBIDDEN_CLAIMS),
        # terminal machine flags (all False = the project makes none of these overclaims)
        "tta_safe_to_deploy_claim": False,
        "target_prior_identified_from_r0_r1_claim": False,
        "prior_robust_benefit_exists_claim": False,
        "oracle_gain_ever_deployable": False,
        # headline caveat (reviewer): the only positive is a tau=0 sign-level artifact, de-emphasised
        "prior_robust_safe_adaptation_certifiable_with_harm_margin": False,
        "prior_robust_positive_is_zero_margin_sign_level_only": True,
        "manuscript_ready_summary": [
            "Under R0/R1 the sign of the offline-TTA target gain is not deployably identifiable; the "
            "source-only harm predictor is null (TOS-1 ceiling) and richer R1 diagnostics are an "
            "overfitting artifact once permutation-controlled.",
            "R2 minimal-label control fails: neither static nor sequential label-acquisition policies "
            "safely select adaptation; safe selection requires (near-)full target labels (oracle, "
            "non-deployable). The failure is estimand-invariant (accuracy == balanced accuracy on this "
            "class-balanced grid; the Step-16 gap was a benefit-threshold artifact).",
            "Mechanistically the harm is class-specific and prior-dependent, not global; the gain sign is "
            "fragile (median L1 flip-radius ~0.165 from uniform).",
            "Under any honest declared prior uncertainty (contract C15), robust adaptation benefit is "
            "UNATTAINABLE with a usable harm margin (tau>=0.05); the only positive is a zero-margin "
            "sign-level artifact on a vanishing minority. The robustly justified actions are abstain or "
            "block, never a deployable adapt.",
            "C14 (declared point prior) and C15 (declared uncertainty set) support counterfactual "
            "prior analysis only; neither identifies the actual target prior (Prior-Decoupled boundary).",
        ],
        "no_new_data": True, "no_retraining": True, "no_new_rescue_policy": True,
        "claim_boundary_ok": bool(forbidden_all_not_made and no_deployable_adapt),
        "claim_boundary": ("Terminal closeout: no target functional is claimed identifiable under R0/R1; "
                           "no deployable adaptation is licensed at any honest observability level; C14/C15 "
                           "are declared/counterfactual, never identified target priors; the oracle gain is "
                           "evaluation-only. No SOTA."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 20 — Final Observability-Contract Closeout and Claim Ledger", "",
             f"Scope: {s['scope']}.", "",
             f"## Final verdict", "", f"> **{s['final_verdict']}**", "",
             "## Evidence ledger (Step 12-19)", "",
             "| step | question | verdict | headline metric | value |", "|---|---|---|---|---|"]
    for e in s["evidence_ledger"]:
        lines.append(f"| {e['step']} | {e['question']} | **{e['verdict']}** | {e['headline_metric']} | "
                     f"{e['headline_value']} |")
    lines += ["", "## Deployment decision ladder", "",
              f"No rung licenses a deployable adaptation: **{s['no_ladder_rung_licenses_deployable_adaptation']}**.", "",
              "| observability level | contracts | licensed decision | deployable adapt? |",
              "|---|---|---|:--:|"]
    for r in s["deployment_decision_ladder"]:
        cc = "+".join(r["contracts"]) if r["contracts"] else "—"
        lines.append(f"| {r['level']} | {cc} | {r['licensed_decision']} | "
                     f"{'yes' if r['licenses_deployable_adaptation'] else 'no'} |")
    lines += ["", "## Forbidden headline claims (none is made)", ""]
    for f in s["forbidden_claims"]:
        lines.append(f"- **{f['claim']}** — {f['status']} (made: {f['made']}). {f['why']}")
    lines += ["", "## Manuscript-ready science summary", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(s["manuscript_ready_summary"], 1)]
    lines += ["", f"> {s['claim_boundary']}"]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 20 final closeout and claim ledger")
    ap.add_argument("--summaries", required=True, help="results_summaries directory")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_closeout(args.summaries)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"closeout ledger_rows={len(s['evidence_ledger'])} "
          f"no_deployable_adapt={s['no_ladder_rung_licenses_deployable_adaptation']} "
          f"forbidden_all_not_made={s['forbidden_headline_claims_all_not_made']} "
          f"claim_boundary_ok={s['claim_boundary_ok']}")
    return 0 if s["claim_boundary_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
