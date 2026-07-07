"""Project A Step 17 — per-estimand harm-control frontier (accuracy gain vs balanced-accuracy gain).

Consumes the Step-17 estimand_consistency cells and builds ONE frontier per
(estimand, sampling) group, kept strictly separate:

  accuracy_gain:iid · accuracy_gain:class_balanced ·
  balanced_accuracy_gain:iid · balanced_accuracy_gain:class_balanced

There is deliberately NO single "best policy" across estimands: accuracy gain and balanced-accuracy
gain are different target functionals, so their frontiers are never merged or compared for a winner.
The balanced_accuracy_gain:class_balanced group requires calibration contract C13.

  python -m h2cmi.observability.estimand_frontier --consistency step17_estimand_consistency.json \
      --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result_index import _load_json, write_json_lf, write_text_lf

_HARM_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]
_MIN_COVERAGES = [0.01, 0.05, 0.10]
_ESTIMANDS = ["accuracy_gain", "balanced_accuracy_gain"]
_SAMPLINGS = ["iid", "class_balanced"]
_C13 = "C13"


def _group_key(estimand, sampling):
    return f"{estimand}:{sampling}"


def _requires_contract(estimand, sampling):
    return _C13 if (estimand == "balanced_accuracy_gain" and sampling == "class_balanced") else None


def _labels(c):
    k = c.get("k")
    if k in (None, "full"):
        return 1e9 if k is None else 1e8            # "full" is a large but finite label budget
    return int(k)


def _best_under(cells, harm_thr, min_cov) -> Optional[Dict[str, Any]]:
    elig = [c for c in cells if c.get("harm_rate_among_adapt_decisions") is not None
            and c["harm_rate_among_adapt_decisions"] <= harm_thr
            and (c.get("adaptation_coverage") or 0) >= min_cov]
    if not elig:
        return None
    b = max(elig, key=lambda c: (c["adaptation_coverage"], -_labels(c),
                                 -(c.get("missed_benefit_rate") if c.get("missed_benefit_rate") is not None else 1.0)))
    return {"policy": b["policy"], "k": b.get("k"), "tau": b.get("tau"),
            "adaptation_coverage": b["adaptation_coverage"],
            "harm": b["harm_rate_among_adapt_decisions"], "missed_benefit": b.get("missed_benefit_rate")}


def build_summary(consistency: Dict[str, Any]) -> Dict[str, Any]:
    all_cells = consistency.get("cells", [])
    groups = {}
    for e in _ESTIMANDS:
        for s in _SAMPLINGS:
            gk = _group_key(e, s)
            cells = [c for c in all_cells if c.get("estimand") == e and c.get("sampling") == s]
            best_by_thr = {f"best_under_harm_{str(h).replace('.', '_')}": _best_under(cells, h, 0.05)
                           for h in _HARM_THRESHOLDS}
            frontier = []
            for h in _HARM_THRESHOLDS:
                for mc in _MIN_COVERAGES:
                    b = _best_under(cells, h, mc)
                    frontier.append({"harm_threshold": h, "min_coverage": mc,
                                     "best_policy": None if b is None else b["policy"],
                                     "best_k": None if b is None else b["k"],
                                     "best_adaptation_coverage": None if b is None else b["adaptation_coverage"]})
            groups[gk] = {
                "estimand": e, "sampling": s, "requires_contract": _requires_contract(e, s),
                "n_cells": len(cells), **best_by_thr,
                "any_policy_meets_harm_0_05": best_by_thr["best_under_harm_0_05"] is not None,
                "any_policy_meets_harm_0_1": best_by_thr["best_under_harm_0_1"] is not None,
                "any_policy_meets_harm_0_2": best_by_thr["best_under_harm_0_2"] is not None,
                "frontier_table": frontier,
            }
    return {
        "project": "Project A", "step": "Step 17",
        "scope": "per-estimand harm-control frontier (accuracy vs balanced-accuracy); not SOTA",
        "estimands": _ESTIMANDS, "samplings": _SAMPLINGS,
        "group_keys": [_group_key(e, s) for e in _ESTIMANDS for s in _SAMPLINGS],
        "harm_thresholds": _HARM_THRESHOLDS, "min_coverages": _MIN_COVERAGES,
        "accuracy_policy_controls_bacc": False,
        "no_overall_best_across_estimands": True,      # HARD: frontiers are never merged/compared
        "overall_best_policy": None,                   # explicitly none — see no_overall_best_across_estimands
        "class_balanced_bacc_requires_contract": _C13,
        "groups": groups,
        "claim_boundary": ("Accuracy-gain and balanced-accuracy-gain frontiers are kept SEPARATE; there is "
                           "no overall best policy across estimands (different target functionals). The "
                           "balanced_accuracy_gain:class_balanced frontier requires contract C13. Frontiers "
                           "are R2 label-budget views, not R1 target-gain identifiability."),
        "claim_boundary_ok": True,
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 17 — per-estimand harm-control frontier", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- no overall best across estimands: **{s['no_overall_best_across_estimands']}** · "
             f"accuracy policy controls bAcc: **{s['accuracy_policy_controls_bacc']}** · "
             f"class-balanced bAcc requires **{s['class_balanced_bacc_requires_contract']}**", "",
             "| group | requires_contract | meets 0.05 | meets 0.10 | meets 0.20 |",
             "|---|---|---|---|---|"]
    for gk in s["group_keys"]:
        g = s["groups"][gk]
        lines.append(f"| {gk} | {g['requires_contract']} | {g['any_policy_meets_harm_0_05']} | "
                     f"{g['any_policy_meets_harm_0_1']} | {g['any_policy_meets_harm_0_2']} |")
    lines += ["", "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 17 per-estimand harm-control frontier")
    ap.add_argument("--consistency", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(_load_json(Path(args.consistency)) or {})
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    g = s["groups"]
    print("estimand_frontier groups=" + ",".join(s["group_keys"]) +
          f" acc_iid_meets_0.10={g['accuracy_gain:iid']['any_policy_meets_harm_0_1']} "
          f"bacc_cb_meets_0.10={g['balanced_accuracy_gain:class_balanced']['any_policy_meets_harm_0_1']} "
          f"no_overall_best={s['no_overall_best_across_estimands']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
