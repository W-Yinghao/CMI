#!/usr/bin/env python
"""FSR Step 2B — RQ4 branch-locality: DESCRIPTIVE only (no probe, no GPU, no inference).

Reads FBCSP_F0_AGGREGATE.csv from git branch `project/fbcsp-lgg-spatial-cmi-fusion` via `git show`.
Emits the branch-load table + a missing-metric report. RQ4 has ZERO quantitative rows because the
per-branch leakage (L1) and per-branch functional reliance (L5) metrics do not exist on disk.

Outputs (results/fsr_phase2b/):
    rq4_branch_load_descriptive.csv
    rq4_branch_missing_metric_report.md

    python scripts/fsr/build_rq4_branch_report.py
"""
from __future__ import annotations
import csv, io, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "fsr_phase2b"
REF = "project/fbcsp-lgg-spatial-cmi-fusion"
AGG = "results/fbcsp_lgg_f0_full_s012/FBCSP_F0_AGGREGATE.csv"
BRANCHES = ["graph", "temporal", "spatial"]


def git_text(path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{REF}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def load_status(drop, gate):
    if drop >= 0.05:
        return "load_bearing"
    if drop <= 0.0:
        return "neutral_or_slightly_harmful"
    return "weak"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(io.StringIO(git_text(AGG))))
    out_rows = []
    for r in rows:
        ds = r["dataset"]
        mean = float(r["mean_bacc"])
        for b in BRANCHES:
            zero = float(r[f"zero_{b}"])
            gate = float(r[f"gate_{b}"])
            drop = round(mean - zero, 4)
            out_rows.append(dict(
                dataset=ds, branch=b, mean_bacc=mean, zero_branch_bacc=zero,
                branch_ablation_drop=drop, gate_weight=gate,
                load_bearing_status=load_status(drop, gate),
                has_branch_leakage_probe="NO", has_branch_R3_reliance="NO",
                rq4_quantitative_status="BLOCKED_MISSING_METRIC"))
    _wcsv(OUT / "rq4_branch_load_descriptive.csv", out_rows)

    # missing-metric report
    spatial = [r for r in out_rows if r["branch"] == "spatial"]
    md = [
        "# FSR RQ4 — branch-locality (DESCRIPTIVE; blocked, not failed)",
        "",
        "Source: `FBCSP_F0_AGGREGATE.csv` on `project/fbcsp-lgg-spatial-cmi-fusion` @ 39c245a "
        "(backbone `FBCSPLGGGraph`, branches graph / temporal / spatial; **no separate node branch** — "
        "`permute_nodes` is a null, not a branch). Read via `git show`.",
        "",
        "## Branch load (from ablation + gate weights)",
        "",
        "| dataset | branch | ablation_drop (mean-zero) | gate_weight | status |",
        "|---|---|---|---|---|",
    ]
    for r in out_rows:
        md.append(f"| {r['dataset']} | {r['branch']} | {r['branch_ablation_drop']:+.4f} | "
                  f"{r['gate_weight']:.4f} | {r['load_bearing_status']} |")
    md += [
        "",
        "## What RQ4 CAN say (descriptive)",
        "- The **spatial branch is load-bearing**: " +
        "; ".join(f"{r['dataset']} zero_spatial drop {r['branch_ablation_drop']:+.4f}, gate {r['gate_weight']:.3f}"
                  for r in spatial) + ".",
        "- The **graph/temporal branches are neutral-to-slightly-harmful** on 4-class 2a and starved after fusion.",
        "- **P6 spatial-CMI is a scaffold / not promoted**, not a confirmed spatial-CMI result.",
        "",
        "## What RQ4 CANNOT say (blocked)",
        "- \"spatial leakage is harmful\" — no per-branch leakage probe exists.",
        "- \"graph leakage is benign\" — same.",
        "- \"per-branch CMI predicts reliance\" — no per-branch functional-reliance (L5) measurement exists.",
        "",
        "## Missing metrics (both HIGH, `needs_small_frozen_probe`)",
        "| missing_metric | needed_for | status | resolution |",
        "|---|---|---|---|",
        "| per-branch leakage probe on `spatial_z`/`graph_z`/`node_z` | RQ4 predictor (L1 per branch) | "
        "absent on disk (0 frozen embeddings, 0 per-branch probe) | small_frozen_probe (Phase 3/4, PM-gated) |",
        "| per-branch functional reliance (L5) | RQ4 endpoint | absent | small_frozen_probe (couples to the dump) |",
        "",
        "**RQ4 quantitative status: `BLOCKED_MISSING_METRIC` for every branch.** No probe is run in Step 2B; "
        "this is a blocked (not failed) RQ. Producing the two metrics requires re-inference to freeze "
        "`last_spatial_z`/`graph_z`/`node_z` per fold plus a trained per-branch domain probe and a per-branch "
        "R3-style removal replay — deferred to a PM-approved Phase-3/4 frozen-probe run.",
    ]
    (OUT / "rq4_branch_missing_metric_report.md").write_text("\n".join(md) + "\n")

    print(f"RQ4 branch_load rows: {len(out_rows)}")
    for r in spatial:
        print(f"  {r['dataset']} spatial: drop {r['branch_ablation_drop']:+.4f} gate {r['gate_weight']:.3f} -> {r['load_bearing_status']}")
    print("  RQ4 quantitative status: BLOCKED_MISSING_METRIC (per-branch leakage + reliance absent)")


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
