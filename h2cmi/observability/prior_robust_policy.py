"""Project A Step 19 — prior-robust adaptation policy (oracle/evaluation-only; contracts C14/C15).

Given the Step-19 robust gain bounds over declared L1 prior-uncertainty balls, this evaluates a
worst-case decision rule per (rho, tau):

    adapt   if robust_lower(rho) > tau        (robustly beneficial over the whole set U_rho)
    block   if robust_upper(rho) < -tau       (robustly harmful over U_rho -> keep identity)
    abstain otherwise                         (sign ambiguous under the declared uncertainty)

This is NOT a deployable selector: the class deltas are oracle target-label quantities and the prior
set is DECLARED (C15), so it never certifies a real deployment. It answers whether ANY run can be
certified robustly-beneficial under a bounded prior-uncertainty set, and how much identity/block is
robustly justified. Because `adapt` requires the worst-case gain over U_rho (which contains the uniform
prior) to exceed tau>=0, a robust `adapt` is never harmful under the benchmark uniform prior — reported
as a consistency check.

  python -m h2cmi.observability.prior_robust_policy \
      --prior-uncertainty step19_prior_uncertainty_frontier.json \
      --harm-thresholds 0.05 0.10 0.20 --rhos 0.05 0.10 0.20 0.50 \
      --out-json step19_prior_robust_policy.json --out-md step19_prior_robust_policy.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result_index import _load_json, write_json_lf, write_text_lf

_C14, _C15 = "C14", "C15"


def _rho_key(frontier_rhos, rho):
    """Match a requested rho to the frontier's str(rho) key (frontier keys come from float str())."""
    for fr in frontier_rhos:
        if abs(float(fr) - rho) < 1e-9:
            return str(fr)
    return None


def _cell(runs, rho_key, rho, tau) -> Dict[str, Any]:
    n = len(runs)
    adapt = block = abstain = adapt_uniform_harm = 0
    for r in runs:
        b = r["robust_bounds_by_rho"].get(rho_key)
        if b is None:
            abstain += 1
            continue
        if b["lower"] > tau:
            adapt += 1
            if r.get("uniform_gain", 0.0) < 0:
                adapt_uniform_harm += 1
        elif b["upper"] < -tau:
            block += 1
        else:
            abstain += 1
    return {
        "rho": rho, "tau": tau,
        "adaptation_coverage": round(adapt / n, 4) if n else None,
        "robust_harm_block_rate": round(block / n, 4) if n else None,
        "abstention_rate": round(abstain / n, 4) if n else None,
        "harm_rate_among_adapt_decisions_under_uniform": round(adapt_uniform_harm / adapt, 4) if adapt else None,
        "robust_prior_safe_adaptation_exists": adapt > 0,
        "claim_boundary": ("oracle class deltas + declared prior uncertainty (C15); not deployable "
                           "without R2 class-wise evidence"),
    }


def _select_best(cells) -> Optional[Dict[str, Any]]:
    elig = [c for c in cells if c["robust_prior_safe_adaptation_exists"]]
    if not elig:
        return None
    # most robust safe adaptation: max coverage, tie-break larger rho (wider set), then smaller tau
    return max(elig, key=lambda c: (c["adaptation_coverage"], c["rho"], -c["tau"]))


def build_summary(frontier, harm_thresholds, rhos) -> Dict[str, Any]:
    runs = (frontier or {}).get("runs", [])
    fr_rhos = (frontier or {}).get("rhos", [])
    taus = list(harm_thresholds)
    cells = []
    for rho in rhos:
        rk = _rho_key(fr_rhos, rho)
        if rk is None:                                         # requested rho not in the frontier
            continue
        for tau in taus:
            cells.append(_cell(runs, rk, rho, tau))
    best = _select_best(cells)
    any_safe = any(c["robust_prior_safe_adaptation_exists"] for c in cells)
    # any adapt decision that is harmful under the uniform prior would break the consistency invariant
    inconsistent = [c for c in cells if (c["harm_rate_among_adapt_decisions_under_uniform"] or 0) > 0]
    return {
        "project": "Project A", "step": "Step 19",
        "scope": "prior-robust adaptation policy over declared L1 prior sets (C15); not SOTA",
        "n_runs": len(runs), "rhos": rhos, "harm_thresholds": taus,
        "robust_prior_safe_adaptation_exists_any": any_safe,
        "best_prior_robust_policy": best,
        "robust_adapt_never_uniform_harmful": len(inconsistent) == 0,   # consistency invariant (must hold)
        "prior_uncertainty_contract_required": _C15,
        "point_prior_contract": _C14,
        "actual_target_prior_identified": False,
        "not_deployable_without_r2_class_evidence": True,
        "cells": cells,
        "claim_boundary_ok": len(inconsistent) == 0,
        "claim_boundary": ("worst-case decision rule over DECLARED L1 prior-uncertainty sets (C15); class "
                           "deltas are oracle/evaluation-only; not a deployable selector; does NOT identify "
                           "the actual target prior. No SOTA."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    b = s["best_prior_robust_policy"]
    best_str = ("none" if b is None else
                f"rho={b['rho']}, tau={b['tau']}, adapt-cov={b['adaptation_coverage']}")
    lines = ["# Step 19 — prior-robust adaptation policy", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- runs: **{s['n_runs']}** · robust-prior safe adaptation exists (any): "
             f"**{s['robust_prior_safe_adaptation_exists_any']}** · robust adapt never uniform-harmful: "
             f"**{s['robust_adapt_never_uniform_harmful']}**",
             f"- best prior-robust policy: **{best_str}**",
             f"- contract required **{s['prior_uncertainty_contract_required']}** · actual target prior "
             f"identified **{s['actual_target_prior_identified']}**", "",
             "| ρ | τ | adapt_cov | robust_harm_block | abstain | harm@adapt(uniform) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for c in s["cells"]:
        lines.append(f"| {c['rho']} | {c['tau']} | {c['adaptation_coverage']} | "
                     f"{c['robust_harm_block_rate']} | {c['abstention_rate']} | "
                     f"{c['harm_rate_among_adapt_decisions_under_uniform']} |")
    lines += ["", "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 19 prior-robust adaptation policy")
    ap.add_argument("--prior-uncertainty", required=True)
    ap.add_argument("--harm-thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.50])
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(_load_json(Path(args.prior_uncertainty)) or {}, args.harm_thresholds, args.rhos)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    b = s["best_prior_robust_policy"]
    print(f"prior_robust_policy n_runs={s['n_runs']} safe_adapt_any={s['robust_prior_safe_adaptation_exists_any']} "
          f"best={'none' if b is None else (b['rho'], b['tau'], b['adaptation_coverage'])} "
          f"robust_adapt_never_uniform_harmful={s['robust_adapt_never_uniform_harmful']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
