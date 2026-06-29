"""
Generate the three CSC manuscript figures (Direction A: identifiability/abstention boundary +
negative confirmatory result). Reads the committed artifacts; writes PNGs to notes/figures/.

  Fig 1  shift taxonomy + unidentifiability/abstention boundary           (schematic)
  Fig 2  certificate pipeline + fail-closed gates (3-state output)         (schematic)
  Fig 3  development operating map vs frozen confirmatory failure          (DATA-driven)

Run:  python -m csc.tools.make_paper_figures      (from the frozen csc checkout)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
FIGDIR = os.path.join(HERE, "notes", "figures")
DEV = os.path.join(HERE, "csc", "results", "envelope_p15_dev_c12.json")
CONF = os.path.join(HERE, "csc", "results", "confirmatory.json")


def _baseline_cell(devjson):
    for c in devjson["full"]["grid"]:
        if c["cell"] == "baseline":
            return c
    raise SystemExit("baseline cell not found in dev artifact")


# ----------------------------------------------------------------------------- Fig 3 (data)
def fig3():
    dev = _baseline_cell(json.load(open(DEV)))
    conf = json.load(open(CONF))["per_point"]["P_baseline"]
    fig, (axp, axf) = plt.subplots(1, 2, figsize=(9.2, 4.2))

    # --- power panel ---
    pw_dev, pw_dev_lb = dev["visible_concept_power"], dev["visible_concept_power_cp_lower"]
    pw_cf, pw_cf_lb = conf["power_conditional"], conf["power_conditional_cp_lower"]
    xs = [0, 1]
    axp.bar(xs, [pw_dev, pw_cf], width=0.55, color=["#5b8def", "#e0654b"])
    # CP-lower as a downward marker/error
    axp.errorbar(xs, [pw_dev, pw_cf], yerr=[[pw_dev - pw_dev_lb, pw_cf - pw_cf_lb], [0, 0]],
                 fmt="none", ecolor="black", capsize=5, lw=1.4)
    axp.axhline(0.50, ls="--", color="green", lw=1.3)
    axp.text(1.46, 0.505, "power bar 0.50", color="green", va="bottom", ha="right", fontsize=9)
    for x, p, lb in [(0, pw_dev, pw_dev_lb), (1, pw_cf, pw_cf_lb)]:
        axp.text(x, p + 0.02, f"{p:.2f}\n(CP-LB {lb:.2f})", ha="center", va="bottom", fontsize=9)
    axp.set_xticks(xs); axp.set_xticklabels(["DEVELOPMENT\n(12 clusters)", "CONFIRMATORY\n(unseen, N=65)"])
    axp.set_ylim(0, 1.0); axp.set_ylabel("visible-concept power")
    axp.set_title("(a) power: dev 0.83 → confirmatory 0.43 (FAIL)")

    # --- false-certification panel: forbidden CP-upper vs alpha ---
    fb_dev_ub = dev["any_forbidden_full_suite_cp_upper"]   # 0/12 -> 0.221
    fb_cf_ub = conf["forbidden_cp_upper"]                  # 1/65 -> 0.071
    axf.bar(xs, [fb_dev_ub, fb_cf_ub], width=0.55, color=["#5b8def", "#e0654b"])
    axf.axhline(0.05, ls="--", color="red", lw=1.3)
    axf.text(1.46, 0.056, "α = 0.05", color="red", va="bottom", ha="right", fontsize=9)
    axf.text(0, fb_dev_ub + 0.005, f"0/12\nCP-UB {fb_dev_ub:.3f}", ha="center", va="bottom", fontsize=9)
    axf.text(1, fb_cf_ub + 0.005, f"{conf['forbidden']}/{conf['n_valid']}\nCP-UB {fb_cf_ub:.3f}",
             ha="center", va="bottom", fontsize=9)
    axf.set_xticks(xs); axf.set_xticklabels(["DEVELOPMENT\n(12 clusters)", "CONFIRMATORY\n(unseen, N=65)"])
    axf.set_ylim(0, 0.26); axf.set_ylabel("forbidden-rate 95% CP upper bound")
    axf.set_title("(b) false-cert control: confirmatory CP-UB 0.071 > α (FAIL)")

    fig.suptitle("Fig 3. Development operating point vs frozen confirmatory failure (P_baseline, "
                 "csc-confirmatory-v1/dee8958)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "fig3_dev_vs_confirmatory.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


# ----------------------------------------------------------------------------- Fig 1 (schematic)
def _box(ax, x, y, w, h, text, fc, fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def fig1():
    fig, ax = plt.subplots(figsize=(9.2, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.7, "Fig 1. Shift taxonomy & the identifiability / abstention boundary",
            ha="center", fontsize=11)
    _box(ax, 0.3, 3.3, 3.0, 1.4, "Support-visible\nCOVARIATE shift\nP(Z) moves, P(Y|Z) fixed", "#cfe3ff")
    _box(ax, 3.6, 3.3, 3.0, 1.4, "BOUNDARY shift +\nsupport signature\n(class-conditional in R)", "#d7f0d0")
    _box(ax, 6.9, 3.3, 2.8, 1.4, "PURE CONDITIONAL\nshift\nP(Y|Z) moves, P(Z) fixed", "#f7d6cf")
    # identifiable vs not
    ax.plot([6.75, 6.75], [0.6, 4.9], ls="--", color="#888", lw=1.2)
    ax.text(3.4, 0.95, "IDENTIFIABLE from unlabeled Z\n(+ source anchor / support signature)\n"
                       "→ COVARIATE_COMPATIBLE or CONCEPT_SUSPECT", ha="center", fontsize=9,
            color="#1a5a1a")
    ax.text(8.3, 1.05, "UNIDENTIFIABLE from Z alone\n(Prop.: ∃ many Q(Y|Z) | same Q_Z)\n"
                       "→ MUST ABSTAIN", ha="center", fontsize=9, color="#8a2a1a")
    out = os.path.join(FIGDIR, "fig1_taxonomy_boundary.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ----------------------------------------------------------------------------- Fig 2 (schematic)
def fig2():
    fig, ax = plt.subplots(figsize=(9.6, 4.4)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    ax.text(6, 5.7, "Fig 2. Certificate pipeline & fail-closed gates (source-anchored, 3-state)",
            ha="center", fontsize=11)
    _box(ax, 0.2, 3.2, 2.3, 1.5, "Source (labeled)\n+ unlabeled target Z\nbuild shift ATLAS\n"
                                 "(cov / concept / label)", "#e8e8e8", 8.5)
    _box(ax, 2.9, 3.2, 2.5, 1.5, "GATES (fail-closed):\nsupport graph\nresidual-T (decoder)\n"
                                 "geometric max-stat\nattribution eigengap/stability", "#fff2cc", 8)
    _box(ax, 5.9, 3.2, 2.3, 1.5, "source_status\nVALID? else\nABSTAIN (UNIDENTIFIABLE)", "#ffe0e0", 8.5)
    _box(ax, 8.7, 4.35, 3.0, 0.95, "COVARIATE_COMPATIBLE\n(cov-stable, no concept)", "#cfe3ff", 8.5)
    _box(ax, 8.7, 3.2, 3.0, 0.95, "CONCEPT_SUSPECT\n(source-anchored evidence)", "#d7f0d0", 8.5)
    _box(ax, 8.7, 2.05, 3.0, 0.95, "UNIDENTIFIABLE\n(abstain)", "#f0f0f0", 8.5)
    for (x0, x1) in [(2.5, 2.9), (5.4, 5.9), (8.2, 8.7)]:
        ax.add_patch(FancyArrowPatch((x0, 3.95), (x1, 3.95), arrowstyle="-|>", mutation_scale=12, lw=1.2))
    ax.text(6, 1.3, "Pre-registered confirmatory endpoints (frozen): forbidden 0/N (CP-UB ≤ α) "
                    "AND power CP-LB ≥ bar, on UNSEEN clusters", ha="center", fontsize=8.5, color="#444")
    out = os.path.join(FIGDIR, "fig2_pipeline_gates.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    for f in (fig1, fig2, fig3):
        print("wrote", f())
    print("FIGURES_DONE")
