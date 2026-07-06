#!/usr/bin/env python
"""FSR Phase 3A/4 — generate manuscript figures + tables from FROZEN result CSVs (CPU-only).

No re-inference, no new data. Reads results/fsr_phase2*/ and writes paper/fsr/figures/*.pdf +
paper/fsr/tables/*.tex. Colorblind-safe (Okabe-Ito). Every figure encodes provenance/claim tier.
Figure 3 follows the Step-2C logic: main visual = benefit_claimable=0/40; secondary = sensitivity
strip; the negative correlation is NOT presented as a finding.

    python scripts/fsr/plot_phase3_figures.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[2]
P2, P2B, P2C = REPO / "results/fsr_phase2", REPO / "results/fsr_phase2b", REPO / "results/fsr_phase2c"
FIG = REPO / "paper/fsr/figures"
TAB = REPO / "paper/fsr/tables"
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
plt.rcParams.update({"font.size": 9, "axes.splines_top" if False else "axes.grid": False,
                     "savefig.bbox": "tight", "figure.dpi": 150})


def load(fp):
    with open(fp, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------- Figure 1: the ladder ----------------
def fig1_ladder():
    levels = [("L1", "Detectability", "can D be decoded from Z|Y?"),
              ("L2", "Reducibility", "does a method reduce measured leakage?"),
              ("L3", "Erasability", "can an eraser remove the subject signal?"),
              ("L4", "Task coupling", "is the subspace aligned with the task head?"),
              ("L5", "Functional reliance", "does removing it change the output?"),
              ("L6", "Target consequence", "harmful / benign / task-useful?")]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axis("off")
    y = 5.4
    for i, (lid, name, q) in enumerate(levels):
        col = OK["blue"] if i < 4 else OK["green"]
        ax.add_patch(FancyBboxPatch((0.3, y), 5.9, 0.72, boxstyle="round,pad=0.03",
                                    fc=col, ec="black", alpha=0.14, lw=1))
        ax.text(0.55, y + 0.36, lid, fontsize=11, fontweight="bold", va="center", color=col)
        ax.text(1.25, y + 0.46, name, fontsize=10, fontweight="bold", va="center")
        ax.text(1.25, y + 0.16, q, fontsize=8, va="center", color="#333333", style="italic")
        y -= 0.9
    ax.text(3.25, 6.35, "predictor side {L1-L4}", fontsize=8.5, ha="center", color=OK["blue"])
    ax.text(3.25, -0.15, "endpoint side {L5, L6}", fontsize=8.5, ha="center", color=OK["green"])
    ax.text(6.45, 5.0, "RQ1/RQ3", rotation=90, fontsize=8, ha="center", va="center", color=OK["orange"])
    ax.text(6.45, 2.3, "RQ2", rotation=90, fontsize=8, ha="center", va="center", color=OK["orange"])
    ax.set_xlim(0, 7)
    ax.set_ylim(-0.4, 6.6)
    ax.set_title("The functional-shortcut-reliance ladder (no L1$\\rightarrow$L5/L6 leap)", fontsize=10)
    fig.savefig(FIG / "fig1_ladder.pdf")
    plt.close(fig)


# ---------------- Figure 2: alignment vs leakage -> reliance ----------------
def fig2_alignment_vs_leakage():
    u = load(P2B / "rq3_unit_table.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    dcol = {"BNCI2014_001": OK["blue"], "BNCI2015_001": OK["orange"]}
    for ds, c in dcol.items():
        xs = [f(r["align_k2"]) for r in u if r["dataset"] == ds]
        ys = [f(r["R3_task_drop"]) for r in u if r["dataset"] == ds]
        a.scatter(xs, ys, s=14, c=c, alpha=0.7, label=ds.replace("BNCI2014_001", "2a").replace("BNCI2015_001", "2015"))
    a.set_xlabel("align_k2 (task-head alignment, L4)")
    a.set_ylabel("R3 task_drop (reliance, L5)")
    a.set_title("A. align $\\rightarrow$ reliance\n$\\rho$=+0.34 [+0.17,+0.50], n=126 (RECOMPUTED)", fontsize=8.5)
    a.legend(fontsize=7, frameon=False)
    # panel B: graph_kl seed0
    xs = [f(r["graph_kl"]) for r in u if r["graph_kl"] not in ("", None)]
    ys = [f(r["R3_task_drop"]) for r in u if r["graph_kl"] not in ("", None)]
    b.scatter(xs, ys, s=16, c=OK["red"], alpha=0.7)
    b.set_xlabel("graph_kl (leakage magnitude, L1)")
    b.set_ylabel("R3 task_drop (reliance, L5)")
    b.set_title("B. leakage $\\rightarrow$ reliance (WRONG sign)\n$\\rho$=$-$0.42 seed0 n=42 (SIGN_ONLY);"
                " pooled $-$0.34 FROZEN", fontsize=8.5)
    fig.text(0.5, -0.03, "signed Spearman difference (align $-$ graph_kl) = +0.816 [+0.219, +1.333], excludes 0",
             ha="center", fontsize=8, color=OK["green"])
    fig.tight_layout()
    fig.savefig(FIG / "fig2_alignment_vs_leakage.pdf")
    plt.close(fig)


# ---------------- Figure 3: erasure -> NO certified benefit (Step-2C logic) ----------------
def fig3_erasure_no_benefit():
    cells = load(P2B / "rq2_erasure_vs_target.csv")
    fam = load(P2C / "rq2_sensitivity_by_family.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    # panel A: E vs target_bAcc, marker by task-safety
    for r in cells:
        x, y = f(r["E_subject_removed"]), f(r["T_target_bAcc"])
        if x is None or y is None:
            continue
        if r["task_collapse"] == "YES":
            a.scatter(x, y, s=26, marker="x", c=OK["red"])
        elif r["binary_harm"] == "YES":
            a.scatter(x, y, s=24, marker="s", facecolors="none", edgecolors=OK["red"])
        else:
            a.scatter(x, y, s=18, c=OK["blue"], alpha=0.7)
    a.axhline(0, color=OK["grey"], lw=0.8, ls="--")
    a.set_xlabel("E: subject removed (L3)")
    a.set_ylabel("$\\Delta$ target bAcc (L6)")
    a.set_title("A. erasure $\\rightarrow$ target: benefit_claimable = 0/40\n"
                "(x = task-collapse, open sq = binary-harm)", fontsize=8.5)
    a.plot([], [], "x", c=OK["red"], label="task-collapse")
    a.plot([], [], "s", markerfacecolor="none", markeredgecolor=OK["red"], label="binary-harm", ls="")
    a.plot([], [], "o", c=OK["blue"], label="other", ls="")
    a.legend(fontsize=6.5, frameon=False, loc="lower left")
    # panel B: sensitivity strip
    labels = {"1_all_cells": "all", "2_clean_cells_no_collapse_no_harm": "clean",
              "3_excl_random_k": "–random_k", "4_excl_INLP": "–INLP",
              "5_excl_INLP_and_random_k": "–INLP,–rand_k", "6_LEACE_RLACE_only": "LEACE/RLACE",
              "7_within_dataset_backbone_rank_resid": "within ds×bb"}
    order = list(labels)
    ys = list(range(len(order)))[::-1]
    for yi, key in zip(ys, order):
        row = next(r for r in fam if r["subset"] == key)
        rho, lo, hi = f(row["rho"]), f(row["ci_lo"]), f(row["ci_hi"])
        col = OK["green"] if rho > 0 else OK["red"]
        b.plot([lo, hi], [yi, yi], color=col, lw=1.6)
        b.scatter([rho], [yi], s=22, color=col, zorder=3)
    b.axvline(0, color=OK["grey"], lw=0.8, ls="--")
    b.set_yticks(ys)
    b.set_yticklabels([labels[k] for k in order], fontsize=7)
    b.set_xlabel("$\\rho$(E, $\\Delta$target bAcc)")
    b.set_title("B. sensitivity: negative $\\rho$ NOT robust\n(LEACE/RLACE flips +0.54) $\\Rightarrow$ not a finding",
                fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_erasure_no_benefit.pdf")
    plt.close(fig)


# ---------------- Figure 4: branch load + missing instrument ----------------
def fig4_branch_blocked():
    rows = load(P2B / "rq4_branch_load_descriptive.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    branches = ["graph", "temporal", "spatial"]
    ds_list = sorted({r["dataset"] for r in rows})
    w = 0.36
    for j, ds in enumerate(ds_list):
        drops = [f(next(r for r in rows if r["dataset"] == ds and r["branch"] == br)["branch_ablation_drop"]) for br in branches]
        gates = [f(next(r for r in rows if r["dataset"] == ds and r["branch"] == br)["gate_weight"]) for br in branches]
        xpos = [k + (j - 0.5) * w for k in range(len(branches))]
        a.bar(xpos, drops, width=w, color=[OK["green"] if br == "spatial" else OK["grey"] for br in branches],
              label=ds.replace("BNCI2014_001", "2a").replace("BNCI2015_001", "2015"), edgecolor="black", lw=0.4)
        b.bar(xpos, gates, width=w, color=[OK["green"] if br == "spatial" else OK["grey"] for br in branches],
              edgecolor="black", lw=0.4)
    for ax, ttl, yl in ((a, "A. branch ablation drop (L4)", "mean $-$ zero_branch"),
                        (b, "B. fusion gate weight", "gate weight")):
        ax.set_xticks(range(len(branches)))
        ax.set_xticklabels(branches, fontsize=8)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
    a.legend(fontsize=7, frameon=False)
    fig.text(0.5, -0.04, "spatial branch load-bearing; per-branch LEAKAGE (L1) and RELIANCE (L5) NOT measured "
             "→ RQ4 BLOCKED (no checkpoint exists)", ha="center", fontsize=8, color=OK["red"])
    fig.tight_layout()
    fig.savefig(FIG / "fig4_branch_blocked.pdf")
    plt.close(fig)


# ---------------- Tables ----------------
def _tex_escape(s):
    return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("->", "$\\to$")


def table1_claim_ledger():
    rows = load(P2B / "claim_readiness_table.csv")
    lines = [r"\begin{tabular}{llp{7.6cm}}", r"\toprule",
             r"ID & Status & Claim \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['claim_id']} & {_tex_escape(r['status'])} & {_tex_escape(r['claim_text'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table1_claim_ledger.tex").write_text("\n".join(lines) + "\n")


def table2_route_firewall():
    rows = load(P2 / "analysis_inclusion_table.csv")
    keep = [r for r in rows if r["include_rq1"] == "YES" or r["include_rq2"] == "YES" or r["include_rq3"] == "YES"]
    others = [r for r in rows if r not in keep]
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Route & Predictor & Endpoint & Inclusion \\", r"\midrule"]
    for r in keep:
        lines.append(f"{_tex_escape(r['route'])} & {r['predictor_levels']} & {r['endpoint_levels']} & INCLUDED \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{4}}{{l}}{{{len(others)} other routes: SUPPORT/BOUNDARY/PROTOCOL/BACKGROUND\\_ONLY "
                 r"(no target-label fit enters any RQ)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table2_route_firewall.tex").write_text("\n".join(lines) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    fig1_ladder()
    fig2_alignment_vs_leakage()
    fig3_erasure_no_benefit()
    fig4_branch_blocked()
    table1_claim_ledger()
    table2_route_firewall()
    print("wrote:", *(p.name for p in sorted(FIG.glob("*.pdf"))), *(p.name for p in sorted(TAB.glob("*.tex"))))


if __name__ == "__main__":
    main()
