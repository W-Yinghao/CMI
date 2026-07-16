#!/usr/bin/env python
"""Relaxation Ladder Figure 2 — effect forest from REAL aggregated results. Two panels per stratum:
LW-LEACE vs full (delta bAcc) and specific gain (LW-LEACE vs same-rank random), grouped by protocol level +
dataset/backbone, with vertical reference lines at 0 and +0.01. Fails loudly if the aggregation CSVs are
absent (no fabricated bars).

  python paper/cmi_trace/figures/make_ladder_forest.py \
      --summary results/cmi_trace_relaxation_ladder/protocol_ladder_summary.csv \
      --paired  results/cmi_trace_relaxation_ladder/paired_deltas.csv \
      --out paper/cmi_trace/figures/ladder_forest.png
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

LV = {"L1_STRICT_SOURCE_FRESH_HEAD": "L1 strict/fresh", "L2_TARGET_X_UNLABELED_FRESH_HEAD": "L2 target-X/fresh",
      "L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD": "L3 oracle/fresh"}


def _load(p):
    if not Path(p).exists():
        sys.exit(f"[forest] MISSING {p} — run the ladder + aggregator first. No figure fabricated.")
    return list(csv.DictReader(open(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/cmi_trace_relaxation_ladder/protocol_ladder_summary.csv")
    ap.add_argument("--paired", default="results/cmi_trace_relaxation_ladder/paired_deltas.csv")
    ap.add_argument("--out", default="paper/cmi_trace/figures/ladder_forest.png")
    a = ap.parse_args()
    summ = [r for r in _load(a.summary) if r["eraser"] == "lw_leace_full" and r["level"] in LV]
    paired = [r for r in _load(a.paired) if r["level"] in LV]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _key(r):
        return f"{r['dataset'][:7]}·{r['training_method'][:8]}·{LV[r['level']]}"
    labels = [_key(r) for r in summ]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3, 0.32 * len(summ) + 1)), sharey=True)
    for ax, (rows, title, xlab) in zip(axes, [
            (summ, "LW-LEACE vs full", "ΔbAcc (erasure − full)"),
            (paired, "specific gain: LW-LEACE vs same-rank random", "Δ(LEACE) − Δ(random)")]):
        ys = range(len(rows))
        for i, r in enumerate(rows):
            lo, hi, m = float(r["ci_lo"]), float(r["ci_hi"]), float(r["mean"])
            col = "#2E7D32" if lo > 0 else ("#B71C1C" if hi < 0 else "#F9A825")
            ax.plot([lo, hi], [i, i], color=col, lw=2); ax.plot(m, i, "o", color=col)
        ax.axvline(0, color="k", lw=0.8); ax.axvline(0.01, color="gray", lw=0.9, ls="--")
        ax.set_title(title, fontsize=9); ax.set_xlabel(xlab, fontsize=8)
    axes[0].set_yticks(list(range(len(summ)))); axes[0].set_yticklabels(labels, fontsize=6)
    fig.suptitle("Relaxation ladder: subject-axis erasure effect (green=CI>0, red=CI<0, amber=straddles)", fontsize=10)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print(f"[forest] wrote {a.out} ({len(summ)} rows)")


if __name__ == "__main__":
    main()
