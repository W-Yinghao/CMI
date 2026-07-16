#!/usr/bin/env python
"""Relaxation Ladder Figure 3 — regime map from REAL diagnostics. Each point is a (dataset, method, subject)
cell: x = source task-direction consistency, y = task-subject subspace overlap. Points are colored by whether
L1 subject-axis erasure was beneficial / neutral / harmful (sign of the per-fold LW-LEACE delta bAcc) and
marked by source-only gate accept/refuse. Fails loudly if the diagnostics CSVs are absent.

  python paper/cmi_trace/figures/make_regime_map.py --out paper/cmi_trace/figures/regime_map.png
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = "results/cmi_trace_relaxation_ladder"


def _load(p):
    if not Path(p).exists():
        sys.exit(f"[regime] MISSING {p} — run the diagnostics + ladder first. No figure fabricated.")
    return list(csv.DictReader(open(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default=f"{ROOT}/direction_consistency.csv")
    ap.add_argument("--overlap", default=f"{ROOT}/subspace_overlap.csv")
    ap.add_argument("--raw", default=f"{ROOT}/dgcnn_ladder_raw_rows.jsonl")
    ap.add_argument("--out", default="paper/cmi_trace/figures/regime_map.png")
    a = ap.parse_args()
    # consistency per (dataset, method, subject): use binary pair or multiclass macro row
    cons = {}
    for r in _load(a.direction):
        if r["kind"] in ("binary", "multiclass_macro"):
            cons[(r["dataset"], r["training_method"], r["heldout_subject"])] = float(r["consistency"])
    ov = {(r["dataset"], r["training_method"], r["heldout_subject"]): float(r["normalized_overlap"])
          for r in _load(a.overlap)}
    # per-cell L1 lw_leace delta sign (beneficial/neutral/harmful), averaged over seeds
    if not Path(a.raw).exists():
        sys.exit(f"[regime] MISSING {a.raw}")
    d1 = defaultdict(list)
    for line in open(a.raw):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("fit_regime") == "L1_STRICT_SOURCE_FRESH_HEAD" and r.get("eraser") == "lw_leace_full":
            d1[(r["dataset"], r["training_method"], r["heldout_subject"])].append(float(r["delta_bacc"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for key in cons:
        if key not in ov or key not in d1:
            continue
        dm = float(np.mean(d1[key]))
        col = "#2E7D32" if dm > 0.01 else ("#B71C1C" if dm < -0.01 else "#F9A825")
        mk = "o" if key[0] == "BNCI2014_001" else "^"
        ax.scatter(cons[key], ov[key], c=col, marker=mk, s=45, edgecolor="k", linewidth=0.4, alpha=0.85)
    ax.set_xlabel("source task-direction consistency (cross-subject cosine)")
    ax.set_ylabel("task–subject subspace overlap (normalized)")
    ax.set_title("Regime map — L1 subject-axis erasure effect per subject\n"
                 "green=beneficial (>+0.01), amber=neutral, red=harmful (<-0.01); o=BNCI2014, ^=BNCI2015", fontsize=9)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=160)
    print(f"[regime] wrote {a.out}")


if __name__ == "__main__":
    main()
