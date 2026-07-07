"""Project A Step 12 — science dashboard.

Combines the four Step-12 artifacts (harm-attribution table, retrospective harm predictor,
minimal-paired phase transition, multi-dataset digest) into a single reviewer-readable dashboard of
"what we learned" and "what remains unknown". It asserts nothing beyond those artifacts and makes no
SOTA claim; the oracle gain stays an evaluation label throughout.

  python -m h2cmi.observability.science_dashboard --harm-table ... --harm-predictor ... \
      --phase-transition ... --multidataset ... --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .result_index import _load_json, write_json_lf, write_text_lf


def build_dashboard(harm_table, harm_pred, phase, multi) -> Dict[str, Any]:
    fs = (harm_pred or {}).get("feature_sets", {})
    r0 = fs.get("R0_source_only", {})
    r1 = fs.get("R1_target_unlabeled", {})
    oracle_never = (harm_pred or {}).get("oracle_never_a_feature")

    # integrity conjunction: the whole science phase kept the claim boundary
    claim_boundary_ok = (
        oracle_never is True
        and "oracle_denylist" in (harm_table or {})
        and (phase or {}).get("k0_status") == "not_identified_R1"
        and (multi or {}).get("all_target_metrics_identifiable_null") is not False)

    beats = (harm_pred or {}).get("any_predictor_beats_majority_baseline")
    metrics = {
        "n_real_runs": (harm_table or {}).get("n_runs"),
        "real_harm_rate": (harm_table or {}).get("harm_rate"),
        "R0_harm_predictor_bacc": r0.get("balanced_acc_harm_prediction"),
        "R1_harm_predictor_bacc": r1.get("balanced_acc_harm_prediction"),
        "R0_to_R1_delta": (harm_pred or {}).get("r1_minus_r0_balanced_acc_delta"),
        "majority_baseline_bacc": (harm_pred or {}).get("majority_baseline_balanced_acc"),
        "harm_predictor_verdict": (harm_pred or {}).get("verdict"),
        "harm_predictor_beats_baseline": beats,
        "harm_predictor_minority_n": (harm_pred or {}).get("n_minority_class"),
        "minimal_paired_k0_status": (phase or {}).get("k0_status"),
        "minimal_paired_best_k": (phase or {}).get("best_k_overall"),
        "minimal_paired_phase_transition_observed": (phase or {}).get("phase_transition_observed"),
        "minimal_paired_k_per_shift": (phase or {}).get("phase_transition_k_per_shift"),
        "claim_boundary_ok": claim_boundary_ok,
        "oracle_gain_used_only_as_evaluation_label": oracle_never is True,
    }

    # honest characterization: below/at 0.5 baseline => NO signal (do not call it "a predictor")
    pred_line = (
        f"R0/R1 diagnostics do NOT retrospectively predict TTA harm above the 0.5 majority baseline "
        f"(R0 bAcc {metrics['R0_harm_predictor_bacc']}, R1 bAcc {metrics['R1_harm_predictor_bacc']}; "
        f"underpowered, minority n={metrics['harm_predictor_minority_n']}) — consistent with the "
        f"TOS-1 source-only ceiling; NULL result, not identifiability."
        if not beats else
        f"R0/R1 diagnostics give a RETROSPECTIVE harm predictor above baseline (R0 bAcc "
        f"{metrics['R0_harm_predictor_bacc']}, R1 bAcc {metrics['R1_harm_predictor_bacc']}, delta "
        f"{metrics['R0_to_R1_delta']}) — empirical retrospective, NOT identifiability.")
    trans = metrics["minimal_paired_phase_transition_observed"]
    trans_line = (
        f"Minimal paired information: harm-sign estimability is a phase transition in k "
        f"(observed={trans}, per-shift k {metrics['minimal_paired_k_per_shift']}); small true gains "
        f"need more labels, tiny gains stay unresolved — a labeled slice under an iid sampling contract.")
    learned = [
        f"Offline TTA harms most audited cells (real harm-rate {metrics['real_harm_rate']}).",
        pred_line,
        "R1 target-unlabeled diagnostics do NOT make target gain identifiable (TOS-1/TU-2 stand).",
        trans_line,
        "Exact counterexamples remain the proof layer; the real-EEG grids illustrate, they do not prove.",
    ]
    unknown = [
        "Whether these patterns hold on clinical / non-motor-imagery EEG.",
        "Whether stronger TTA baselines reduce the harm rate.",
        "Whether label-free target support/marginal diagnostics can be made reliable.",
        "Whether minimal-paired anchors can be collected cheaply in realistic BCI workflows.",
    ]
    return {
        "project": "Project A", "step": "Step 12",
        "scope": "scientific exploration (harm attribution + minimal-information phase transition); not SOTA",
        "metrics": metrics,
        "what_we_learned": learned,
        "what_remains_unknown": unknown,
        "claim_boundary": ("Oracle target gain is an evaluation label throughout; R0/R1 harm "
                           "prediction is retrospective, not target-gain identifiability; k>0 slices "
                           "are labeled slices under an iid sampling contract. No SOTA claim."),
    }


def write_md(d: Dict[str, Any], path) -> str:
    m = d["metrics"]
    lines = ["# Step 12 — Science Dashboard", "",
             f"Scope: {d['scope']}.", "",
             "## Key metrics", "",
             f"- real runs: **{m['n_real_runs']}** · real harm-rate: **{m['real_harm_rate']}**",
             f"- harm-predictor balanced-acc — R0: **{m['R0_harm_predictor_bacc']}** · R1: "
             f"**{m['R1_harm_predictor_bacc']}** · R1−R0 delta: **{m['R0_to_R1_delta']}** "
             f"(majority baseline **{m['majority_baseline_bacc']}**)",
             f"- minimal-paired: k0 **{m['minimal_paired_k0_status']}** · phase transition "
             f"**{m['minimal_paired_phase_transition_observed']}** · best k **{m['minimal_paired_best_k']}**",
             f"- claim boundary ok: **{m['claim_boundary_ok']}** · oracle gain evaluation-only: "
             f"**{m['oracle_gain_used_only_as_evaluation_label']}**", "",
             "## What we learned", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_we_learned"], 1)]
    lines += ["", "## What remains unknown", ""]
    lines += [f"{i}. {x}" for i, x in enumerate(d["what_remains_unknown"], 1)]
    lines += ["", "> " + d["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 12 science dashboard")
    ap.add_argument("--harm-table", required=True)
    ap.add_argument("--harm-predictor", required=True)
    ap.add_argument("--phase-transition", required=True)
    ap.add_argument("--multidataset", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    d = build_dashboard(_load_json(Path(args.harm_table)), _load_json(Path(args.harm_predictor)),
                        _load_json(Path(args.phase_transition)), _load_json(Path(args.multidataset)))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, d)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(d, args.out_md)
    m = d["metrics"]
    print(f"science_dashboard n_real_runs={m['n_real_runs']} real_harm_rate={m['real_harm_rate']} "
          f"R0_bAcc={m['R0_harm_predictor_bacc']} R1_bAcc={m['R1_harm_predictor_bacc']} "
          f"delta={m['R0_to_R1_delta']} phase_transition={m['minimal_paired_phase_transition_observed']} "
          f"best_k={m['minimal_paired_best_k']} claim_boundary_ok={m['claim_boundary_ok']}")
    return 0 if d["metrics"]["claim_boundary_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
