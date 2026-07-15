#!/usr/bin/env python
"""CMI-Trace Figure 3 (P0.6): three panels with fold-cluster 95% CIs, generated from the REAL aggregated
objective-comparison tables. Panel A = relative graph/node encoder-CMI reduction vs ERM; Panel B = paired
target ΔbAcc vs ERM; Panel C = exact-head ΔR_rel(k=2) vs ERM. Reads only the immutable aggregation CSVs;
fails loudly (no fabricated bars) if they are absent.

  python paper/cmi_trace/figures/make_fig3_cmi_reliance_bacc.py \
      --paired results/cmi_trace_p0p1/objective_comparison/paired_deltas.csv \
      --out paper/cmi_trace/figures/fig3_cmi_reliance_bacc.png
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path


def _load(path):
    if not Path(path).exists():
        sys.exit(f"[fig3] MISSING {path} — run the P0.1 jobs + scripts/aggregate_cmi_trace_objective.py first. "
                 f"No figure is fabricated.")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="results/cmi_trace_p0p1/objective_comparison/paired_deltas.csv")
    ap.add_argument("--out", default="paper/cmi_trace/figures/fig3_cmi_reliance_bacc.png")
    a = ap.parse_args()
    rows = _load(a.paired)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("graph_kl", "A: relative encoder-CMI Δ vs ERM (graph)"),
              ("target_bacc", "B: paired target ΔbAcc vs ERM"),
              ("R_rel_k2", "C: exact-head ΔR_rel(k=2) vs ERM")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, title) in zip(axes, panels):
        mrows = [r for r in rows if r["metric"] == metric]
        # one bar cluster per dataset×method
        labels = [f"{r['dataset'][:7]}·{r['method']}" for r in mrows]
        means = [float(r["delta_mean"]) for r in mrows]
        los = [float(r["cluster_ci_lo"]) for r in mrows]
        his = [float(r["cluster_ci_hi"]) for r in mrows]
        yerr = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
        x = range(len(means))
        ax.bar(x, means, yerr=yerr, capsize=3, color="#4C78A8")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print(f"[fig3] wrote {a.out} from {a.paired} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
