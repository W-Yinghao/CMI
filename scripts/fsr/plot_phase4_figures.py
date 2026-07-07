#!/usr/bin/env python
"""FSR Phase 6A — generate manuscript figures 4 (natural branch-local audit) + 5 (repair scope) from FROZEN
verdicts. No re-inference. Numbers are pinned from the frozen result JSON/CSV (provenance in comments).
Colorblind-safe (Okabe-Ito). Writes to ../CMI_AAAI_fsr/paper/fsr/figures/.

    <icml python> scripts/fsr/plot_phase4_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("/home/infres/yinwang/CMI_AAAI_fsr/paper/fsr/figures")
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
plt.rcParams.update({"font.size": 9, "axes.grid": False, "savefig.bbox": "tight", "figure.dpi": 150})

# ---- Fig 4: natural branch-local audit (Phase 4B; results/fsr_rq4_refit/branch_*.csv) ----
# L1 subject decodability (probe_bacc, pooled): spatial .865, temporal .485, graph .361
# L4 ablation drop: spatial +.085, graph -.019, temporal -.017
# L6 spatial subject-subspace erase task_drop (erasing HURTS): +0.015 (2a) / +0.050* (2015)
branches = ["spatial", "temporal", "graph"]
L1 = [0.865, 0.485, 0.361]
L4 = [0.085, -0.017, -0.019]
chance = 0.106
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
x = np.arange(3); w = 0.38
a.bar(x - w / 2, L1, w, color=OK["blue"], label="L1 subject decode")
a.bar(x + w / 2, L4, w, color=OK["orange"], label="L4 ablation load")
a.axhline(chance, ls=":", c=OK["grey"], lw=1); a.text(2.35, chance + 0.02, "chance", fontsize=7, c=OK["grey"])
a.axhline(0, c="k", lw=0.6)
a.set_xticks(x); a.set_xticklabels(branches); a.set_ylabel("bAcc / drop")
a.set_title("A  Spatial: most leaky + most load-bearing", fontsize=8.5)
a.legend(fontsize=7, frameon=False, loc="upper right")
a.spines[["top", "right"]].set_visible(False)
# Panel B: L6 task_drop after spatial subject-subspace erasure (positive = erasing HURTS)
td = [0.015, 0.050]; ds = ["2a", "2015"]
b.bar([0, 1], td, 0.5, color=[OK["red"], OK["red"]])
b.axhline(0, c="k", lw=0.8)
b.text(1, 0.052, "*", fontsize=13, ha="center")
b.annotate("erasing HELPS", (1.5, -0.008), fontsize=7, c=OK["green"], ha="right")
b.annotate("erasing HURTS", (1.5, 0.058), fontsize=7, c=OK["red"], ha="right")
b.set_xticks([0, 1]); b.set_xticklabels(ds); b.set_ylabel("L6 task drop (orig $-$ erased)")
b.set_ylim(-0.02, 0.072)
b.set_title("B  Erasing spatial subject subspace HURTS target", fontsize=8.5)
b.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig4_branch_natural.pdf"); plt.close(fig)
print("wrote fig4_branch_natural.pdf")

# ---- Fig 5: repair scope (Phase 4F first-moment / 4G second-moment) ----
# 4F: E4 netted 0.655, E3 random 0.102, ERASE artifact; abs E4-E3 = +0.033 bAcc; 73% mechanical identity
# 4G: E4b netted 0.18, E3 random ~0.13, ORACLE-E4b 0.136; E4b-E3 +0.005 (sub-DELTA), oracle-E3 +0.004
fig2, (c, d) = plt.subplots(1, 2, figsize=(7.0, 2.8))
# Panel C: first-moment (4F) netted recovery
labels_c = ["E4\n(mean-align)", "E3\n(random)", "ERASE"]
vals_c = [0.655, 0.102, 0.875]
cols_c = [OK["green"], OK["grey"], OK["red"]]
c.bar([0, 1, 2], vals_c, 0.6, color=cols_c)
c.axhline(0, c="k", lw=0.6)
c.set_xticks([0, 1, 2]); c.set_xticklabels(labels_c, fontsize=7.5); c.set_ylabel("netted recovery")
c.set_title("C  1st-moment offset: E4 scoped repair", fontsize=8.5)
c.annotate("abs $+0.033$ bAcc\nover random\n(73% mech. identity;\nERASE is task-destructive)",
           (0.5, 0.60), fontsize=6.6, ha="left", va="top")
c.set_ylim(-0.1, 1.0); c.spines[["top", "right"]].set_visible(False)
# Panel D: second-moment (4G) netted recovery + specificity
labels_d = ["E4b\n(cov-shrink)", "E3\n(random)", "ORACLE\n(true dir)"]
vals_d = [0.181, 0.13, 0.136]
cols_d = [OK["blue"], OK["grey"], OK["purple"]]
d.bar([0, 1, 2], vals_d, 0.6, color=cols_d)
d.axhline(0, c="k", lw=0.6)
d.set_xticks([0, 1, 2]); d.set_xticklabels(labels_d, fontsize=7.5)
d.set_title("D  2nd-moment stochastic: none", fontsize=8.5)
d.annotate("E4b $-$ E3 $= +0.005$\n(CI $\\ni 0$, $<$ DELTA);\noracle also sub-threshold",
           (0.5, 0.30), fontsize=6.6, ha="left", va="top")
d.set_ylim(-0.05, 0.45); d.spines[["top", "right"]].set_visible(False)
fig2.tight_layout(); fig2.savefig(FIG / "fig5_repair_scope.pdf"); plt.close(fig2)
print("wrote fig5_repair_scope.pdf")
