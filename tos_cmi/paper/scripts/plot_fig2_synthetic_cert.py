"""Fig 2 -- Synthetic certification line: geometry alone is not safety; certified refusal is necessary.
All panels from SAVED artifacts (no recompute). Panels:
  A. Certification frontier: the gate fires on real leakage (oracle ~1) but the plug-in is conservative.
  B. A weak critic UNSAFE-ACCEPTS a conditionally-unsafe deletion (probe UCB << Bayes gap).
  C. Estimator gap (Bayes - probe UCB) shrinks with n; the unsafe-accept only occurs at small n.
  D. The certified gate (plug-in + power floor) NEVER unsafe-accepts but is conservative (abstains/rejects).
Note: the score-Fisher-vs-mean-scatter detection on covariance-only leakage (Phase 1.2) is qualitative
(PHASE131_CERTIFICATION.md); not plotted here to avoid recompute.
Run: python -m tos_cmi.paper.scripts.plot_fig2_synthetic_cert
"""
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from . import plot_style as ps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "tos_cmi", "results")


def main():
    fig, ax = plt.subplots(1, 4, figsize=(ps.PANEL_W * 4, 3.0))

    # --- A. certification frontier: plug-in vs oracle detection at the boundary (real leakage) ---
    cells = sorted((json.load(open(p)) for p in glob.glob(os.path.join(R, "frontier_cells", "*.json"))),
                   key=lambda c: c["delta_Y"])
    if not cells:
        cells = sorted(json.load(open(os.path.join(R, "frontier.json")))["cells"], key=lambda c: c["delta_Y"])
    dY = [c["delta_Y"] for c in cells]
    ax[0].plot(dY, [c["boundary"]["pi_oracle"] for c in cells], "-o", color=ps.COLORS["tgt"], label="oracle")
    ax[0].plot(dY, [c["boundary"]["pi"] for c in cells], "-s", color=ps.COLORS["RZ"], label="plug-in")
    ax[0].plot(dY, [c["null"]["pi"] for c in cells], "-^", color=ps.COLORS["chance"], label="null (no leak)")
    ax[0].set_xlabel("true leakage delta_Y"); ax[0].set_ylabel("detection rate pi")
    ax[0].set_ylim(-0.03, 1.03); ax[0].set_title("Frontier: gate fires on\nreal leakage (plug-in conservative)")
    ax[0].legend(loc="center right"); ps.panel_label(ax[0], "A")

    # --- B + C. estimator diagnostics (weak probe vs Bayes oracle) ---
    e = json.load(open(os.path.join(R, "estimator_diag.json")))
    # B: scatter probe_ucb vs bayes; the accepted (unsafe) point is the geometry/weak-critic failure
    safe = [r for r in e if not r["unsafe_accept"]]; uns = [r for r in e if r["unsafe_accept"]]
    ax[1].scatter([r["probe_ucb"] for r in safe], [r["bayes"] for r in safe], s=28, color=ps.COLORS["RZ"], label="rejected (safe)")
    ax[1].scatter([r["probe_ucb"] for r in uns], [r["bayes"] for r in uns], s=70, color=ps.COLORS["subj"], marker="X", label="UNSAFE-ACCEPT")
    lim = max(max(r["bayes"] for r in e), max(r["probe_ucb"] for r in e)) * 1.1
    ax[1].plot([0, lim], [0, lim], ls=":", color="#999", lw=1)
    ax[1].set_xlabel("probe task-risk UCB"); ax[1].set_ylabel("Bayes-oracle task delta")
    ax[1].set_title("Weak critic unsafe-accepts\n(probe UCB << true Bayes gap)"); ax[1].legend(loc="lower right")
    ps.panel_label(ax[1], "B")

    # C: estimator gap (bayes - probe_ucb) vs n, by k
    ns = sorted(set(r["n"] for r in e))
    for k, col in [(1, ps.COLORS["RZ"]), (2, ps.COLORS["mlp"])]:
        g = [np.mean([r["bayes"] - r["probe_ucb"] for r in e if r["n"] == n and r["k"] == k]) for n in ns]
        ax[2].plot(ns, g, "-o", color=col, label="k=%d" % k)
    ax[2].set_xscale("log"); ax[2].set_xlabel("calibration n (log)"); ax[2].set_ylabel("Bayes - probe UCB")
    ax[2].axhline(0, ls=":", color="#999", lw=1)
    ax[2].set_title("Estimator gap shrinks with n\n(unsafe-accept only at small n)"); ax[2].legend()
    ps.panel_label(ax[2], "C")

    # --- D. certified gate decision counts (plug-in + power floor): zero unsafe-accepts, conservative ---
    pf = json.load(open(os.path.join(R, "phase_diagram_powerfloor.json")))
    counts = pf["summary"]["counts"]; order = ["SAFE_REJECT", "UNSAFE_REJECT", "BAYES_AMBIGUOUS"]
    labs = [o for o in order if o in counts] + [o for o in counts if o not in order]
    vals = [counts[o] for o in labs]
    colors = {"SAFE_REJECT": ps.COLORS["RZ"], "UNSAFE_REJECT": ps.COLORS["chance"], "BAYES_AMBIGUOUS": ps.COLORS["mlp"]}
    ax[3].bar(range(len(labs)), vals, color=[colors.get(l, "#777") for l in labs], alpha=0.9)
    ax[3].set_xticks(range(len(labs))); ax[3].set_xticklabels([l.replace("_", "\n") for l in labs], fontsize=7.5)
    ax[3].set_ylabel("# cells")
    ax[3].set_title("Certified gate: %d unsafe-accepts\n(conservative -> refusal)" % pf["summary"]["n_unsafe_accept"])
    ps.panel_label(ax[3], "D")

    fig.suptitle("Synthetic certification: geometry alone is not safety; certified refusal is necessary "
                 "(R=%s, beta=%s)" % (json.load(open(os.path.join(R, "frontier.json")))["meta"].get("R"),
                                      json.load(open(os.path.join(R, "frontier.json")))["meta"].get("beta")), y=1.02)
    ps.save(fig, "fig2_synthetic_cert")
    print("frontier cells=%d ; estimator rows=%d (unsafe=%d) ; powerfloor counts=%s n_unsafe=%d"
          % (len(cells), len(e), sum(r["unsafe_accept"] for r in e), counts, pf["summary"]["n_unsafe_accept"]))
    print("FIG2_DONE")


if __name__ == "__main__":
    main()
