#!/usr/bin/env python
"""CMI-Trace main objective table (P0.6): LaTeX table from the REAL objective_effect_summary.csv, with
fold-cluster 95% CIs (NOT seed SD). Rows = objective methods; columns = target bAcc, graph/node encoder-CMI,
per-domain risk var, IRMv1 diagnostic, exact-head R_rel(k=2) + random control. Fails loudly if the
aggregation CSV is absent (no fabricated numbers).

  python paper/cmi_trace/tables/build_objective_table.py \
      --summary results/cmi_trace_p0p1/objective_comparison/objective_effect_summary.csv \
      --out paper/cmi_trace/tables/objective_table.tex
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

METHOD_ORDER = ["erm", "coral", "label_coral", "irm", "vrex", "cond_dann", "cigl_graph_node", "cigl_nested"]
METRICS = ["target_bacc", "graph_kl", "node_kl", "per_domain_risk_variance", "irmv1_diagnostic",
           "R_rel_k2", "R_rel_k2_random_control"]
PRETTY = {"erm": "ERM", "coral": "CORAL", "label_coral": "C-CORAL", "irm": "IRMv1", "vrex": "V-REx",
          "cond_dann": "cond-DANN", "cigl_graph_node": "Enc-CMI (0.010)", "cigl_nested": "Enc-CMI (nested)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/cmi_trace_p0p1/objective_comparison/objective_effect_summary.csv")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--out", default="paper/cmi_trace/tables/objective_table.tex")
    a = ap.parse_args()
    if not Path(a.summary).exists():
        sys.exit(f"[table] MISSING {a.summary} — run P0.1 jobs + aggregate first. No numbers fabricated.")
    cell = defaultdict(dict)
    with open(a.summary) as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] != a.dataset:
                continue
            cell[r["method"]][r["metric"]] = (r["raw_mean"], r["cluster_ci_lo"], r["cluster_ci_hi"], r["n_folds"])

    def fmt(m, met):
        if met not in cell.get(m, {}):
            return "--"
        mean, lo, hi, _ = cell[m][met]
        return f"{float(mean):.3f} [{float(lo):.3f}, {float(hi):.3f}]"

    lines = [r"% AUTO-GENERATED from objective_effect_summary.csv (fold-cluster 95% CI). Do not hand-edit.",
             r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
             "Method & " + " & ".join(m.replace("_", "\\_") for m in METRICS) + r" \\", r"\midrule"]
    for m in METHOD_ORDER:
        if m not in cell:
            continue
        lines.append(PRETTY.get(m, m) + " & " + " & ".join(fmt(m, met) for met in METRICS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"[table] wrote {a.out} ({a.dataset}; {len(cell)} methods) — cluster CI, not seed SD")


if __name__ == "__main__":
    main()
