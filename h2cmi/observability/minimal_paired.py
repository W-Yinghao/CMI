"""Project A Step 12 — minimal-paired information phase-transition simulator.

A CONTROLLED simulator (not real EEG) that asks: as we add k labeled target trials, how does the
empirical estimability of offline-TTA harm/gain change? It is a phase-transition study of the R1→R2
boundary, NOT an identifiability proof.

Boundary discipline (hard):
  * k = 0 is the R1 boundary: the target gain is NOT identified (TU-2); no labeled slice exists, so
    the harm sign cannot be called (chance) and the risk CI is undefined.
  * k > 0 is the R2 boundary: an iid k-label slice estimates target risk *under an explicit iid
    sampling contract* with a finite-sample CI. It is NOT "full target risk identified" without that
    contract; the labeled-slice caveat is attached to every k>0 record.

  python -m h2cmi.observability.minimal_paired --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .result_index import write_json_lf, write_text_lf

# each shift fixes the TRUE identity/adapted accuracy; gain = acc_adapt - acc_identity (>0 = TTA helps)
SHIFTS: Dict[str, Dict[str, float]] = {
    "prior_shift_only": {"acc_identity": 0.55, "acc_adapt": 0.62},   # TTA (prior correction) helps
    "concept_shift":    {"acc_identity": 0.55, "acc_adapt": 0.45},   # TTA cannot fix concept -> harms
    "support_failure":  {"acc_identity": 0.58, "acc_adapt": 0.42},   # invalid transport -> harms hard
    "montage_transport_shift": {"acc_identity": 0.54, "acc_adapt": 0.57},  # mild transport help
}
# extends past 64 so the transition is LOCATED for realistic small gains (0.03-0.16) rather than
# reported as a bare "not observed" — the harm sign needs k > ~z^2 var / gain^2 labels to resolve.
_K_DEFAULT = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_Z = 1.96
_SAMPLING_CONTRACT = ("k>0: an iid k-label slice estimates target risk under an iid sampling "
                      "contract with a finite-sample CI; NOT full target risk without that contract.")


def simulate(shift: str, k: int, n_repeats: int, seed: int, z: float = _Z) -> Dict[str, Any]:
    import numpy as np
    p0 = SHIFTS[shift]["acc_identity"]
    pa = SHIFTS[shift]["acc_adapt"]
    true_gain = round(pa - p0, 4)
    if k == 0:
        return {"shift_type": shift, "k": 0, "true_gain": true_gain, "harm_sign_accuracy": 0.5,
                "decisive_rate": 0.0, "risk_ci_width": None, "abstention_rate_needed": 1.0,
                "identified_status": "not_identified_R1",
                "claim_boundary": "k=0: target gain non-identifiable under R1 (TU-2); no labeled slice."}
    rng = np.random.default_rng(seed)
    correct = abstain = 0
    widths: List[float] = []
    for _ in range(n_repeats):
        c0 = (rng.random(k) < p0).mean()
        ca = (rng.random(k) < pa).mean()
        gh = ca - c0
        se = ((c0 * (1 - c0) + ca * (1 - ca)) / k) ** 0.5
        widths.append(2 * z * se)
        lo, hi = gh - z * se, gh + z * se
        if lo <= 0 <= hi:
            abstain += 1                                       # CI straddles 0 -> cannot call sign
        elif (gh < 0) == (true_gain < 0):
            correct += 1                                       # decisive AND correct harm sign
    return {"shift_type": shift, "k": k, "true_gain": true_gain,
            "harm_sign_accuracy": round(correct / n_repeats, 4),
            "decisive_rate": round((n_repeats - abstain) / n_repeats, 4),
            "risk_ci_width": round(sum(widths) / len(widths), 4),
            "abstention_rate_needed": round(abstain / n_repeats, 4),
            "identified_status": "labeled_slice_under_iid_sampling_contract",
            "claim_boundary": _SAMPLING_CONTRACT}


def run(ks=None, shifts=None, n_repeats: int = 50, seed: int = 0) -> Dict[str, Any]:
    ks = ks or _K_DEFAULT
    shifts = shifts or list(SHIFTS)
    records: List[Dict[str, Any]] = []
    transition: Dict[str, Any] = {}
    for si, shift in enumerate(shifts):
        curve = [simulate(shift, k, n_repeats, seed + 1000 * si + k) for k in ks]
        records += curve
        # smallest k>0 whose harm-sign accuracy first reaches 0.9 (the phase-transition point)
        hit = next((r["k"] for r in curve if r["k"] > 0 and (r["harm_sign_accuracy"] or 0) >= 0.9), None)
        transition[shift] = hit
    observed = any(v is not None for v in transition.values())
    return {
        "project": "Project A", "step": "Step 12",
        "scope": "minimal-paired R1->R2 phase transition (simulator); not SOTA",
        "n_repeats": n_repeats, "seed": seed, "ks": ks, "shifts": shifts,
        "phase_transition_k_per_shift": transition,
        "phase_transition_observed": observed,
        "best_k_overall": min([v for v in transition.values() if v is not None], default=None),
        "k0_status": "not_identified_R1",
        "records": records,
        "claim_boundary": ("k=0 is the R1 non-identifiability boundary; k>0 is an R2 labeled slice "
                           "under an iid sampling contract, not full-target-risk identification."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 12 — minimal-paired phase transition (simulator)", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- repeats: **{s['n_repeats']}** · k grid: **{s['ks']}**",
             f"- phase transition observed (harm-sign acc ≥ 0.9): **{s['phase_transition_observed']}** · "
             f"best k overall: **{s['best_k_overall']}**",
             f"- k=0 status: **{s['k0_status']}**", "",
             "| shift | k | true_gain | harm_sign_acc | abstention | risk_ci_width | status |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for r in s["records"]:
        lines.append(
            f"| {r['shift_type']} | {r['k']} | {r['true_gain']} | {r['harm_sign_accuracy']} | "
            f"{r['abstention_rate_needed']} | {r['risk_ci_width']} | {r['identified_status']} |")
    lines += ["", "> " + _SAMPLING_CONTRACT]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 12 minimal-paired phase transition")
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = run(n_repeats=args.n_repeats, seed=args.seed)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"minimal_paired phase_transition_observed={s['phase_transition_observed']} "
          f"best_k={s['best_k_overall']} per_shift={s['phase_transition_k_per_shift']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
