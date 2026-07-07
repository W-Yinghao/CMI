"""Project A Step 16 — harm-control policy frontier.

Combines the Step-15 static one-shot policy cells and the Step-16 sequential policy cells into a
harm/coverage/label Pareto view, and reports whether ANY deployable policy meets each harm threshold
(0.05 / 0.10 / 0.20 / 0.50) at each minimum adaptation-coverage. This answers whether harm<=0.05 was
simply too strict. The oracle policy is excluded from the deployable frontier.

  python -m h2cmi.observability.policy_frontier --static step15_harm_control_summary.json \
      --sequential step16_sequential_harm_control.json --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .result_index import _load_json, write_json_lf, write_text_lf

_HARM_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]
_MIN_COVERAGES = [0.01, 0.05, 0.10]
_LABEL_STATIC = {"plugin_sign", "ci_adapt_only_abstain", "ci_adapt_only_identity", "ci_three_way"}


def _collect(static, sequential) -> List[Dict[str, Any]]:
    cells = []
    for c in (static or {}).get("cells", []):
        if c.get("policy") not in _LABEL_STATIC:               # skip always_* and oracle
            continue
        cells.append({"source": "static", "policy": c["policy"], "k": c.get("k"), "tau": c.get("tau"),
                      "labels": c.get("k"), "adaptation_coverage": c.get("adaptation_coverage"),
                      "harm": c.get("harm_rate_among_adapt_decisions"),
                      "missed_benefit": c.get("missed_benefit_rate")})
    for c in (sequential or {}).get("cells", []):              # sequential cells are all deployable
        cells.append({"source": "sequential", "policy": c["policy"], "budget": c.get("budget"),
                      "tau": c.get("tau"), "labels": c.get("mean_labels_used"),
                      "adaptation_coverage": c.get("adaptation_coverage"),
                      "harm": c.get("harm_rate_among_adapt_decisions"),
                      "missed_benefit": c.get("missed_benefit_rate")})
    return cells


def _best_under(cells, harm_thr, min_cov) -> Optional[Dict[str, Any]]:
    elig = [c for c in cells if c["harm"] is not None and c["harm"] <= harm_thr
            and (c["adaptation_coverage"] or 0) >= min_cov]
    if not elig:
        return None
    b = max(elig, key=lambda c: (c["adaptation_coverage"], -(c["labels"] if c["labels"] is not None else 1e9),
                                 -(c["missed_benefit"] if c["missed_benefit"] is not None else 1.0)))
    return {"source": b["source"], "policy": b["policy"], "tau": b["tau"], "labels": b["labels"],
            "adaptation_coverage": b["adaptation_coverage"], "harm": b["harm"],
            "missed_benefit": b["missed_benefit"],
            **({"k": b.get("k")} if b["source"] == "static" else {"budget": b.get("budget")})}


def build_summary(static, sequential) -> Dict[str, Any]:
    cells = _collect(static, sequential)
    n_static = sum(1 for c in cells if c["source"] == "static")
    n_seq = sum(1 for c in cells if c["source"] == "sequential")
    best_by_thr = {f"best_under_harm_{str(h).replace('.', '_')}": _best_under(cells, h, 0.05)
                   for h in _HARM_THRESHOLDS}
    frontier = []
    for h in _HARM_THRESHOLDS:
        for mc in _MIN_COVERAGES:
            b = _best_under(cells, h, mc)
            frontier.append({"harm_threshold": h, "min_coverage": mc,
                             "best_policy": None if b is None else b["policy"],
                             "best_source": None if b is None else b["source"],
                             "best_adaptation_coverage": None if b is None else b["adaptation_coverage"],
                             "best_labels": None if b is None else b["labels"]})
    return {
        "project": "Project A", "step": "Step 16", "scope": "harm-control policy frontier (R2); not SOTA",
        "frontier_axes": ["harm_rate_among_adapt_decisions", "adaptation_coverage", "labels_used",
                          "missed_benefit_rate"],
        "harm_thresholds": _HARM_THRESHOLDS, "min_coverages": _MIN_COVERAGES,
        "n_static_cells": n_static, "n_sequential_cells": n_seq,
        "oracle_excluded": True,
        **best_by_thr,
        "any_policy_meets_harm_0_05": best_by_thr["best_under_harm_0_05"] is not None,
        "any_policy_meets_harm_0_1": best_by_thr["best_under_harm_0_1"] is not None,
        "any_policy_meets_harm_0_2": best_by_thr["best_under_harm_0_2"] is not None,
        "frontier_table": frontier,
        "claim_boundary": ("policy frontier under R2 label budgets / iid sampling contract; NOT R1 "
                           "target-gain identifiability; oracle excluded from the deployable frontier."),
    }


def write_md(s: Dict[str, Any], path) -> str:
    lines = ["# Step 16 — harm-control policy frontier", "",
             f"Scope: {s['scope']}. {s['claim_boundary']}", "",
             f"- static cells: **{s['n_static_cells']}** · sequential cells: **{s['n_sequential_cells']}** · "
             f"oracle excluded: **{s['oracle_excluded']}**",
             f"- any policy meets harm<=0.05: **{s['any_policy_meets_harm_0_05']}** · <=0.10: "
             f"**{s['any_policy_meets_harm_0_1']}** · <=0.20: **{s['any_policy_meets_harm_0_2']}**", "",
             "| harm_threshold | min_coverage | best_policy | source | best_adapt_cov | best_labels |",
             "|---:|---:|---|---|---:|---:|"]
    for f in s["frontier_table"]:
        lines.append(f"| {f['harm_threshold']} | {f['min_coverage']} | {f['best_policy']} | "
                     f"{f['best_source']} | {f['best_adaptation_coverage']} | {f['best_labels']} |")
    lines += ["", "> " + s["claim_boundary"]]
    text = "\n".join(lines) + "\n"
    write_text_lf(path, text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A Step 16 harm-control policy frontier")
    ap.add_argument("--static", required=True)
    ap.add_argument("--sequential", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    s = build_summary(_load_json(Path(args.static)) or {}, _load_json(Path(args.sequential)) or {})
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_lf(args.out_json, s)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_md(s, args.out_md)
    print(f"policy_frontier static={s['n_static_cells']} seq={s['n_sequential_cells']} "
          f"meets_0.05={s['any_policy_meets_harm_0_05']} meets_0.10={s['any_policy_meets_harm_0_1']} "
          f"meets_0.20={s['any_policy_meets_harm_0_2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
