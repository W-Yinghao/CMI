#!/usr/bin/env python3
"""Fig 3 — frozen-confirmatory evidence: Route A (Z-only) NEGATIVE vs Route B3 (paired) POSITIVE.

Both columns are FROZEN CONFIRMATORY results (disjoint tags / manifests / seed streams), NOT
development estimates. Numbers are loaded and recomputed FROM the committed artifacts and asserted
against the pinned expected values, so the figure is provably artifact-locked.

  Route A : csc/results/confirmatory.json          (tag csc-confirmatory-v1 / dee8958, base seed 900000)
  Route B3: csc/results/b3_confirmatory_result.json (tag csc-b3-confirmatory-v1 / 0595f64, base seed 3000000)

Route A DEVELOPMENT numbers (power 0.83 / CP-LB 0.56 ; forbidden 0/12) are from the P1.5 development map
(not in the confirmatory JSON); pinned as constants with provenance here.

Output: csc/paper/figures/fig3_routeA_negative_B3_positive.{png,pdf}
Run:  PYTHONPATH=. conda run -n icml python csc/tools/make_fig3_evidence.py
"""
import json, os, collections
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def cp_up(k, n, a=0.05): return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
def cp_lo(k, n, a=0.05): return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
ALPHA = 0.05

# ---- Route A (from confirmatory.json) -------------------------------------------------
A = json.load(open(os.path.join(ROOT, "csc/results/confirmatory.json")))["per_point"]["P_baseline"]
a_forbidden, a_nvalid = int(A["forbidden"]), int(A["n_valid"])
a_forbid_rate, a_forbid_cpup = a_forbidden / a_nvalid, float(A["forbidden_cp_upper"])
a_power, a_power_cplo = float(A["power_conditional"]), float(A["power_conditional_cp_lower"])
assert a_forbidden == 1 and a_nvalid == 65, (a_forbidden, a_nvalid)
assert abs(a_forbid_cpup - 0.0709) < 1e-3 and abs(a_power - 0.4308) < 1e-3
A_DEV_POWER, A_DEV_POWER_CPLO, A_DEV_FORBID = 0.83, 0.56, 0.0   # P1.5 dev map (pinned, see docstring)

# ---- Route B3 (recompute from per_cluster) --------------------------------------------
B = json.load(open(os.path.join(ROOT, "csc/results/b3_confirmatory_result.json")))["per_cluster"]
ctrl = collections.defaultdict(lambda: [0, 0]); prim = collections.defaultdict(lambda: [0, 0])
guard = collections.defaultdict(lambda: [0, 0])
for r in B:
    if r["phase"] == "control":
        ctrl[(r["kind"], r["budget"])][0] += int(r["confirmed"]); ctrl[(r["kind"], r["budget"])][1] += 1
        if r["kind"] in ("missing_pair", "unequal_epochs_extreme"):
            guard[r["kind"]][0] += int(r["confirmed"]); guard[r["kind"]][1] += 1
    elif r["phase"] == "primary":
        prim[(r["kind"], r["budget"])][0] += int(r["confirmed"]); prim[(r["kind"], r["budget"])][1] += 1
# worst control cell by CP-upper
worst = max(ctrl.items(), key=lambda kv: cp_up(kv[1][0], kv[1][1]))
(wk, wb), (wc, wn) = worst
b_ctrl_rate, b_ctrl_cpup = wc / wn, cp_up(wc, wn)
# primary power: reported per-gating-cell (kind x budget, n=192 pooled over 4 strong scenarios).
# C4 headline CP-lower is the per-cell bound (n=192 -> 0.9845), NOT the 768-pooled bound.
GATING = {"paired_concept", "paired_concept_plus_cov"}
cells = [(k, b, v) for (k, b), v in prim.items() if k in GATING]
assert len(cells) == 4 and all(v[1] == 192 and v[0] == v[1] for _, _, v in cells), cells
percell_n = 192
b_power = min(v[0] / v[1] for _, _, v in cells)               # 1.000
b_power_cplo = min(cp_lo(v[0], v[1]) for _, _, v in cells)     # cp_lo(192,192) = 0.9845
g_missing, g_unequal = guard["missing_pair"], guard["unequal_epochs_extreme"]
assert wk == "clean" and wb == 30 and wc == 3 and wn == 288, worst
assert abs(b_ctrl_cpup - 0.0267) < 1e-3 and b_power == 1.0 and abs(b_power_cplo - 0.9845) < 1e-3
assert g_missing[0] == 0 and g_unequal[0] == 0

# ---- plot -----------------------------------------------------------------------------
GRAY, RED, GREEN = "#9e9e9e", "#c0392b", "#27ae60"
fig, ax = plt.subplots(2, 2, figsize=(9.2, 5.4))
for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=9)

# column headers
ax[0, 0].set_title("Route A — $Z$-only certificate", fontsize=11, weight="bold", pad=16)
ax[0, 1].set_title("Route B3 — paired, minimal-label", fontsize=11, weight="bold", pad=16)

# Row 0: false-certification rate (xlim 0..0.09), alpha line
for a in ax[0]:
    a.axvline(ALPHA, ls="--", lw=1.2, color="black"); a.set_xlim(0, 0.09)
    a.text(ALPHA, 1.55, r"$\alpha=0.05$", fontsize=8, ha="center")
ax[0, 0].barh([1], [a_forbid_rate], xerr=[[0], [a_forbid_cpup - a_forbid_rate]], color=RED,
              capsize=4, height=0.5, label="confirmatory")
ax[0, 0].barh([0], [A_DEV_FORBID], color=GRAY, height=0.5, label="development")
ax[0, 0].set_yticks([0, 1]); ax[0, 0].set_yticklabels(["dev\n0/12", "confirm.\n1/65"])
ax[0, 0].text(a_forbid_cpup + 0.002, 1, f"CP-up {a_forbid_cpup:.3f} > α  →  FAIL", va="center",
              fontsize=8.5, color=RED, weight="bold")
ax[0, 1].barh([0.5], [b_ctrl_rate], xerr=[[0], [b_ctrl_cpup - b_ctrl_rate]], color=GREEN,
              capsize=4, height=0.5)
ax[0, 1].set_yticks([0.5]); ax[0, 1].set_yticklabels(["worst\ncontrol\n3/288"])
ax[0, 1].text(b_ctrl_cpup + 0.002, 0.5, f"CP-up {b_ctrl_cpup:.3f} < α  →  PASS", va="center",
              fontsize=8.5, color=GREEN, weight="bold")
ax[0, 0].set_xlabel("false-certification rate", fontsize=9)
ax[0, 1].set_xlabel("false-confirmation rate", fontsize=9)

# Row 1: power (xlim 0..1.05), 0.50 reference line
for a in ax[1]:
    a.axvline(0.50, ls="--", lw=1.2, color="black"); a.set_xlim(0, 1.05)
    a.text(0.50, 1.55, "bar 0.50", fontsize=8, ha="center")
ax[1, 0].barh([1], [a_power], xerr=[[a_power - a_power_cplo], [0]], color=RED, capsize=4, height=0.5)
ax[1, 0].barh([0], [A_DEV_POWER], xerr=[[A_DEV_POWER - A_DEV_POWER_CPLO], [0]], color=GRAY,
              capsize=4, height=0.5)
ax[1, 0].set_yticks([0, 1]); ax[1, 0].set_yticklabels(["dev\n0.83", "confirm.\n0.43"])
ax[1, 0].text(a_power + 0.03, 1, f"CP-lo {a_power_cplo:.3f} < bar  →  FAIL",
              va="center", ha="left", fontsize=8.5, color=RED, weight="bold")
ax[1, 1].barh([0.5], [b_power], xerr=[[b_power - b_power_cplo], [0]], color=GREEN, capsize=4, height=0.5)
ax[1, 1].set_yticks([0.5]); ax[1, 1].set_yticklabels(["primary\n192/192"])
ax[1, 1].text(0.5, 0.5, f"{b_power:.3f}  (CP-lo {b_power_cplo:.3f})  →  PASS",
              va="center", ha="center", fontsize=8.5, color="white", weight="bold")
ax[1, 0].set_xlabel("visible-concept power", fontsize=9)
ax[1, 1].set_xlabel("primary concept-shift power", fontsize=9)
for a in ax.flat:
    a.set_ylim(-0.6, 1.7)

fig.suptitle("Frozen unseen-cluster confirmatory evidence (disjoint tags, manifests, seed streams)",
             fontsize=10.5, y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
out = os.path.join(ROOT, "csc/paper/figures/fig3_routeA_negative_B3_positive")
fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf")
print("wrote", out + ".png / .pdf")
print(f"A: forbid 1/65 cp_up {a_forbid_cpup:.4f}; power {a_power:.4f} cp_lo {a_power_cplo:.4f}")
print(f"B3: worst ctrl {wk}|m{wb} {wc}/{wn} cp_up {b_ctrl_cpup:.4f}; power {b_power:.3f} "
      f"(per-cell {percell_n}/{percell_n}) cp_lo {b_power_cplo:.4f}; "
      f"guards missing {g_missing[0]}/{g_missing[1]} unequal {g_unequal[0]}/{g_unequal[1]}")
