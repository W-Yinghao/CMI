"""Project A Step 19 — prior-uncertainty robustness frontier (declared set; contract C15).

Step 18 showed most offline-TTA gain signs flip somewhere over the FULL simplex. That is the extreme
uncertainty set. Step 19 asks how FAR the operating prior must move from the benchmark-uniform prior to
flip the gain sign, and which runs are robustly harmful / beneficial / ambiguous over a bounded prior
set. For a class-delta vector `d` and reference prior `u`:

    gain(π) = <π, d>
    U_ρ = { π ∈ simplex : ||π − u||_1 ≤ ρ }
    robust_lower(ρ) = min_{π ∈ U_ρ} <π, d>,   robust_upper(ρ) = max_{π ∈ U_ρ} <π, d>

Because the objective is LINEAR and ||π − u||_1 = 2·(mass moved), the extrema are found EXACTLY by
greedy mass transfer: for the minimum, move up to ρ/2 mass from the highest-delta classes (capacity
u_c each) to the lowest-delta classes (capacity 1 − u_c each), while the transfer still lowers the
objective. This is verified in tests against the binary closed form and a brute-force simplex grid.

Boundary: the class deltas are oracle/evaluation-only; the prior-uncertainty set is a DECLARED external
operating assumption (contract C15). This does NOT identify the actual target prior under R0/R1.

  python -m h2cmi.observability.prior_uncertainty --prior-stress step18_prior_stress.json \
      --rhos 0.0 0.05 0.10 0.20 0.30 0.50 1.0 2.0 --taus 0.0 0.01 0.02 0.05 \
      --out-json step19_prior_uncertainty_frontier.json --out-md step19_prior_uncertainty_frontier.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .result_index import _load_json, write_json_lf, write_text_lf

_EPS = 1e-12
_C15 = "C15"


def _uniform(K: int) -> List[float]:
    return [1.0 / K] * K


def _segments(delta: List[float], u: List[float], minimize: bool) -> List[Tuple[float, float]]:
    """Greedy transfer segments (rate = Δobjective per unit mass, capacity) in optimal order.

    For minimize: add mass to lowest-delta classes (cap 1−u_c), remove from highest-delta (cap u_c),
    while rate = d_add − d_remove < 0. For maximize: mirror. Exact because the objective is linear.
    """
    K = len(delta)
    order = sorted(range(K), key=lambda i: delta[i])           # ascending delta
    adds = order[:] if minimize else order[::-1]               # where mass goes
    rems = order[::-1] if minimize else order[:]               # where mass comes from
    ai = ri = 0
    add_rem = 1.0 - u[adds[0]]
    rem_rem = u[rems[0]]
    segs: List[Tuple[float, float]] = []
    while ai < K and ri < K:
        a, r = adds[ai], rems[ri]
        if a == r:                                             # pointers met in the middle -> done
            break
        rate = delta[a] - delta[r]
        if (minimize and rate >= -_EPS) or ((not minimize) and rate <= _EPS):
            break                                              # no further beneficial transfer
        cap = min(add_rem, rem_rem)
        if cap > _EPS:
            segs.append((rate, cap))
        add_rem -= cap
        rem_rem -= cap
        if add_rem <= _EPS:
            ai += 1
            add_rem = (1.0 - u[adds[ai]]) if ai < K else 0.0
        if rem_rem <= _EPS:
            ri += 1
            rem_rem = u[rems[ri]] if ri < K else 0.0
    return segs


def robust_gain_bounds_l1(delta: List[float], rho: float,
                          center: Optional[List[float]] = None) -> Tuple[float, float]:
    """Exact (worst, best) prior-weighted gain over {π : ||π − center||_1 ≤ rho} ∩ simplex."""
    K = len(delta)
    u = center if center is not None else _uniform(K)
    base = sum(u[i] * delta[i] for i in range(K))
    budget = rho / 2.0

    def _apply(segs):
        shift = moved = 0.0
        for rate, cap in segs:
            t = min(cap, budget - moved)
            if t <= _EPS:
                break
            shift += rate * t
            moved += t
        return shift

    return base + _apply(_segments(delta, u, True)), base + _apply(_segments(delta, u, False))


def minimal_l1_radius_to_flip(delta: List[float], center: Optional[List[float]] = None,
                              tau: float = 0.0) -> Optional[float]:
    """Minimal ||π − center||_1 radius at which the gain sign crosses ±tau (default 0).

    Returns 0.0 if the uniform gain is already within [−tau, tau]; None if the sign cannot be flipped
    even over the whole simplex (ρ = 2).
    """
    K = len(delta)
    u = center if center is not None else _uniform(K)
    base = sum(u[i] * delta[i] for i in range(K))
    if base > tau:                                             # push DOWN toward tau
        minimize, target = True, tau
    elif base < -tau:                                          # push UP toward -tau
        minimize, target = False, -tau
    else:
        return 0.0                                            # already ambiguous / at the boundary
    cum = mass = 0.0
    for rate, cap in _segments(delta, u, minimize):
        if (minimize and base + cum + rate * cap <= target) or \
           ((not minimize) and base + cum + rate * cap >= target):
            t = (target - (base + cum)) / rate
            return 2.0 * (mass + t)
        cum += rate * cap
        mass += cap
    return None                                               # unflippable over the full simplex


def _sign_status(lower: float, upper: float, tau: float) -> str:
    if lower > tau:
        return "robust_benefit"
    if upper < -tau:
        return "robust_harm"
    return "ambiguous"


def _per_run(run, rhos, taus) -> Dict[str, Any]:
    delta = run["class_delta_vector"]
    K = len(delta)
    u = _uniform(K)
    uniform_gain = round(sum(u[i] * delta[i] for i in range(K)), 6)
    bounds = {}
    for rho in rhos:
        lo, hi = robust_gain_bounds_l1(delta, rho)
        bounds[str(rho)] = {"lower": round(lo, 6), "upper": round(hi, 6),
                            "sign_status": _sign_status(lo, hi, taus[0] if taus else 0.0)}
    flip = minimal_l1_radius_to_flip(delta)
    return {
        "dataset": run.get("dataset"), "target_subject": run.get("target_subject"),
        "seed": run.get("seed"), "n_classes": K,
        "uniform_gain": uniform_gain, "class_delta_vector": [round(d, 6) for d in delta],
        "minimal_l1_radius_to_flip_from_uniform": None if flip is None else round(flip, 6),
        "robust_bounds_by_rho": bounds,
        "claim_boundary": "counterfactual declared prior-uncertainty set; C15 required; not target-prior id",
    }


def build_summary(prior_stress, rhos, taus) -> Dict[str, Any]:
    import numpy as np
    src = [r for r in (prior_stress or {}).get("runs", []) if r.get("class_delta_vector")]
    runs = [_per_run(r, rhos, taus) for r in src]
    n = len(runs)
    tau0 = taus[0] if taus else 0.0

    def frac_status(rho, status):
        if not n:
            return None
        return round(sum(1 for r in runs if r["robust_bounds_by_rho"][str(rho)]["sign_status"] == status) / n, 4)

    flips = [r["minimal_l1_radius_to_flip_from_uniform"] for r in runs]
    finite = [f for f in flips if f is not None]

    def within(x):
        return round(sum(1 for f in finite if f <= x) / n, 4) if n else None

    return {
        "project": "Project A", "step": "Step 19",
        "scope": "prior-uncertainty robustness frontier (declared set, contract C15); not SOTA",
        "n_runs": n, "rhos": rhos, "taus": taus, "reference_prior": "uniform",
        "n_unflippable_over_simplex": sum(1 for f in flips if f is None),
        "median_flip_radius_from_uniform": round(float(np.median(finite)), 6) if finite else None,
        "q25_flip_radius": round(float(np.percentile(finite, 25)), 6) if finite else None,
        "q75_flip_radius": round(float(np.percentile(finite, 75)), 6) if finite else None,
        "fraction_flip_within_l1_0_10": within(0.10),
        "fraction_flip_within_l1_0_20": within(0.20),
        "fraction_flip_within_l1_0_50": within(0.50),
        "fraction_ambiguous_by_rho": {str(r): frac_status(r, "ambiguous") for r in rhos},
        "fraction_robust_harm_by_rho": {str(r): frac_status(r, "robust_harm") for r in rhos},
        "fraction_robust_benefit_by_rho": {str(r): frac_status(r, "robust_benefit") for r in rhos},
        "sign_status_tau": tau0,
        "prior_uncertainty_contract_required": _C15,
        "actual_target_prior_identified": False,
        "deployment_prior_identified_under_R1": False,
        "runs": runs,
        "claim_boundary_ok": True,
        "claim_boundary": ("robust gain bounds are over a DECLARED prior-uncertainty set (contract C15); "
                           "class deltas are oracle/evaluation-only; the actual target prior is NOT "
                           "identified (that needs TU-1). No SOTA."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 19 — prior-uncertainty robustness frontier", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · median L1 flip-radius from uniform "
             f"**{s['median_flip_radius_from_uniform']}** (q25 **{s['q25_flip_radius']}** / q75 "
             f"**{s['q75_flip_radius']}**) · unflippable over simplex **{s['n_unflippable_over_simplex']}**",
             f"- flip within L1 ≤0.10 **{s['fraction_flip_within_l1_0_10']}** · ≤0.20 "
             f"**{s['fraction_flip_within_l1_0_20']}** · ≤0.50 **{s['fraction_flip_within_l1_0_50']}**",
             f"- prior-uncertainty contract required **{s['prior_uncertainty_contract_required']}** · "
             f"actual target prior identified **{s['actual_target_prior_identified']}**", "",
             "| ρ | robust_harm | ambiguous | robust_benefit |", "|---:|---:|---:|---:|"]
    for r in s["rhos"]:
        lines.append(f"| {r} | {s['fraction_robust_harm_by_rho'][str(r)]} | "
                     f"{s['fraction_ambiguous_by_rho'][str(r)]} | {s['fraction_robust_benefit_by_rho'][str(r)]} |")
    lines += ["", "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 19 prior-uncertainty robustness frontier")
    ap.add_argument("--prior-stress", required=True)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(_load_json(Path(args.prior_stress)) or {}, args.rhos, args.taus)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"prior_uncertainty n_runs={s['n_runs']} median_flip={s['median_flip_radius_from_uniform']} "
          f"within_0.10={s['fraction_flip_within_l1_0_10']} within_0.20={s['fraction_flip_within_l1_0_20']} "
          f"robust_benefit@0.10={s['fraction_robust_benefit_by_rho'].get('0.1')} "
          f"unflippable={s['n_unflippable_over_simplex']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
