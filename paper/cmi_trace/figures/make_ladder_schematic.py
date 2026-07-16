#!/usr/bin/env python
"""Relaxation Ladder Figure 1 — protocol-ladder schematic showing L0-L3 and exactly which TARGET information
each level is allowed (from the frozen LEVEL_META, not fabricated). Annotates each level's verdict when a
verdict.json is available. Schematic only (no result numbers).

  python paper/cmi_trace/figures/make_ladder_schematic.py --out paper/cmi_trace/figures/ladder_schematic.png
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import LEVELS, LEVEL_META

ROW = [("L0_STRICT_SOURCE_ORIGINAL_HEAD", "L0  strict / original head", "reliance anchor (deployable DG)"),
       ("L1_STRICT_SOURCE_FRESH_HEAD", "L1  strict / fresh head", "deployable source-only DG"),
       ("L2_TARGET_X_UNLABELED_FRESH_HEAD", "L2  target-X unlabeled / fresh", "transductive calibration"),
       ("L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD", "L3  oracle-global / fresh", "cohort-conditioned upper bound (NOT DG)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", default="results/cmi_trace_relaxation_ladder/verdict.json")
    ap.add_argument("--out", default="paper/cmi_trace/figures/ladder_schematic.png")
    a = ap.parse_args()
    verdicts = {}
    if Path(a.verdict).exists():
        v = json.load(open(a.verdict))
        verdicts = {k: d["verdict"] for k, d in v.get("verdicts", {}).items()}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off")
    ax.text(0.5, 0.97, "CMI-Trace protocol relaxation ladder — allowed target information per level",
            ha="center", fontsize=11, weight="bold", transform=ax.transAxes)
    cols = ["uses_target_x", "uses_target_subject_group", "uses_target_y", "is_source_only_dg",
            "is_transductive", "is_oracle_diagnostic"]
    short = ["tgt X", "tgt group", "tgt Y", "src-only DG", "transductive", "oracle"]
    y = 0.82
    ax.text(0.02, y + 0.06, "level", fontsize=8, weight="bold", transform=ax.transAxes)
    for j, s in enumerate(short):
        ax.text(0.42 + j * 0.095, y + 0.06, s, fontsize=7, weight="bold", ha="center", transform=ax.transAxes)
    for lv, label, note in ROW:
        m = LEVEL_META[lv]
        ax.text(0.02, y, label, fontsize=8, transform=ax.transAxes)
        ax.text(0.02, y - 0.028, note, fontsize=6.5, color="#555", transform=ax.transAxes)
        for j, c in enumerate(cols):
            val = m[c]
            mark = "✓" if val else "·"
            col = ("#2E7D32" if (c in ("is_source_only_dg",) and val) else
                   "#B71C1C" if (c in ("uses_target_x", "uses_target_y", "is_oracle_diagnostic") and val) else "#333")
            ax.text(0.42 + j * 0.095, y, mark, fontsize=10, ha="center", color=col, transform=ax.transAxes)
        y -= 0.19
    ax.text(0.02, 0.02, "target Y NEVER enters eraser/head/selection at any level (scoring only). "
            "L0/L1 use no target X; L2 uses target X + group; L3 fits geometry on the whole cohort (oracle).",
            fontsize=6.5, color="#333", transform=ax.transAxes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print(f"[schematic] wrote {a.out}")


if __name__ == "__main__":
    main()
