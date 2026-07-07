"""Project A Step 14 — power / sensitivity for retrospective harm prediction.

Quantifies how underpowered the harm-prediction setting is (few runs, tiny non-harmed minority,
high permutation-null p95) so the marginal R1 result is not over-read. It does NOT run models; it
reads the harm-attribution table + the harm-predictor summary.

Boundary: permutation-null significance is empirical retrospective evidence; it does NOT make target
gain identifiable under R1 (TOS-1/TU-2). Duplicating targets is NOT evidence.

  python -m h2cmi.observability.harm_power --harm-table ... --harm-predictor ... \
      --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .result_index import _load_json, write_json_lf, write_text_lf


def build_power(harm_table: Dict[str, Any], harm_pred: Dict[str, Any]) -> Dict[str, Any]:
    n = harm_pred.get("n_runs") or (harm_table or {}).get("n_runs")
    harmed = harm_pred.get("n_harmed")
    non = (n - harmed) if (isinstance(n, int) and isinstance(harmed, int)) else None
    minority = harm_pred.get("n_minority_class")
    frac = harm_pred.get("minority_fraction")
    r1 = (harm_pred.get("feature_sets", {}) or {}).get("R1_target_unlabeled", {})
    obs = r1.get("balanced_acc_harm_prediction")
    p95 = r1.get("perm_null_p95")
    margin = harm_pred.get("robust_margin", 0.03)
    mdb = round(p95 + margin, 4) if p95 is not None else None      # min detectable bAcc to be robust
    robust = bool(harm_pred.get("any_predictor_robust_signal"))
    underpowered = bool((frac is not None and frac < 0.2) or (isinstance(minority, int) and minority < 12))
    return {
        "project": "Project A", "step": "Step 14",
        "scope": "harm-prediction power/sensitivity; not SOTA",
        "n_runs": n, "n_harmed": harmed, "n_non_harmed": non,
        "minority_fraction": frac, "n_minority_class": minority, "n_groups": harm_pred.get("n_groups"),
        "observed_R1_bacc": obs, "perm_null_p95": p95, "required_margin_rule": margin,
        "minimum_detectable_bacc_approx": mdb, "robust_signal": robust,
        "underpowered": underpowered,
        "power_caveat": (
            f"Underpowered for stable harm-prediction claims: minority n={minority} (fraction {frac}), "
            f"high permutation-null p95={p95}. A balanced-acc below ~{mdb} cannot be distinguished from "
            "the overfitting-inflated null; the observed R1 signal is "
            f"{'ROBUST' if robust else 'NOT robust (marginal)'}."),
        "claim_boundary": ("Permutation-null significance is empirical retrospective evidence; it does "
                           "NOT make target gain identifiable under R1. Duplicating targets is NOT "
                           "evidence and is not performed."),
    }


def write_md(d: Dict[str, Any], path) -> str:
    lines = ["# Step 14 — harm-prediction power / sensitivity", "",
             f"Scope: {d['scope']}.", "",
             f"- runs: **{d['n_runs']}** · harmed **{d['n_harmed']}** · non-harmed **{d['n_non_harmed']}** · "
             f"minority fraction **{d['minority_fraction']}** · groups **{d['n_groups']}**",
             f"- observed R1 bAcc **{d['observed_R1_bacc']}** · permutation-null p95 **{d['perm_null_p95']}** · "
             f"min detectable bAcc ≈ **{d['minimum_detectable_bacc_approx']}**",
             f"- robust signal: **{d['robust_signal']}** · underpowered: **{d['underpowered']}**", "",
             f"> {d['power_caveat']}", "", f"> {d['claim_boundary']}"]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 14 harm-prediction power/sensitivity")
    ap.add_argument("--harm-table", required=True)
    ap.add_argument("--harm-predictor", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    d = build_power(_load_json(Path(args.harm_table)) or {}, _load_json(Path(args.harm_predictor)) or {})
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, d)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(d, args.out_md)
    print(f"harm_power n_runs={d['n_runs']} minority_frac={d['minority_fraction']} "
          f"underpowered={d['underpowered']} robust={d['robust_signal']} "
          f"min_detectable_bacc={d['minimum_detectable_bacc_approx']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
