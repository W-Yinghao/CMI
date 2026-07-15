"""Fig 6 -- seed-averaged removability vs latent dimension for TSMNet (SPD/LogEig) and EEGNet (conv).
Reads the multi-seed Track-C json (factorial_multiseed.json) written by
tos_cmi.eeg.factorial_multiseed_analysis, and plots each removability signal with FOLD-CLUSTER 95% CI error
bars over 3 seeds x 9 LOSO folds. The two arches coincide at low d_z but diverge at high d_z (residual
architecture effect); the console also prints the matched-dim contrast CIs and
the OLS coefficient CIs so the figure and the verdict stay in sync.
  python -m tos_cmi.paper.scripts.plot_fig6_factorial
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from . import plot_style as ps

FACT = os.path.join(ps.ROOT, "tos_cmi", "results", "tos_cmi_eeg_frozen", "factorial")
ACOL = {"TSMNet": "#1f77b4", "EEGNet": "#d62728"}
AMARK = {"TSMNet": "o", "EEGNet": "s"}


def _series(rows, bb, key):
    r = sorted([x for x in rows if x["backbone"] == bb], key=lambda x: x["z_dim"])
    x = [c["z_dim"] for c in r]
    m = np.array([c["%s_mean" % key] for c in r])
    lo = m - np.array([c["%s_lo" % key] for c in r])
    hi = np.array([c["%s_hi" % key] for c in r]) - m
    return x, m, np.vstack([lo, hi])


def main():
    j = json.load(open("%s/factorial_multiseed.json" % FACT))
    rows = list(j["cells"].values())
    chance = rows[0]["chance"]
    nseed = max(c["nseed"] for c in rows)
    fig, ax = plt.subplots(1, 4, figsize=(ps.PANEL_W * 4, 3.0))
    panels = [("full", "subject decode (MLP)", "Subject leakage grows\nwith latent capacity", None),
              ("res", "MLP residual after LEACE", "LEACE residual rises with $d_z$;\ncoincide at low $d_z$, diverge at high", None),
              ("task_full", "task decode (linear)", "Task preserved across\ncapacity (full vs LEACE)", "task_leace"),
              ("tos", "MLP residual after deletion", "Rank-limited deletion (TOS)\ndegrades with capacity", "rand")]
    for i, (key, ylab, title, key2) in enumerate(panels):
        a = ax[i]
        for bb in ["TSMNet", "EEGNet"]:
            x, m, err = _series(rows, bb, key)
            a.errorbar(x, m, yerr=err, fmt="-" + AMARK[bb], color=ACOL[bb], capsize=2, lw=1.3,
                       ms=4, label=bb)
            if key2:
                x2, m2, err2 = _series(rows, bb, key2)
                a.errorbar(x2, m2, yerr=err2, fmt="--" + AMARK[bb], color=ACOL[bb], alpha=0.55,
                           capsize=2, lw=1.1, ms=3,
                           label="%s (%s)" % (bb, "LEACE" if "task" in key2 else "rand-$k$"))
        a.set_xscale("log"); a.set_xlabel("latent dimension $d_z$"); a.set_ylabel(ylab)
        a.set_title(title, fontsize=9)
        if "subj" in ylab or "residual" in ylab:
            a.axhline(chance, ls=":", lw=1, color="#888"); a.set_ylim(0, 1.02)
        a.legend(fontsize=6.5, loc="best")
        ps.panel_label(a, "ABCD"[i])
    fig.suptitle("Seed-averaged removability vs latent dimension $d_z$ (%d seeds, fold-cluster 95%% CI): "
                 "low-$d_z$ cells coincide; high-$d_z$ SPD retains more nonlinear residual than conv (BCI-IV-2a)"
                 % nseed, y=1.02)
    ps.save(fig, "fig6_capacity_factorial")

    # console: keep the figure and the verdict in sync
    print("LEACE MLP residual at each cell (mean [fold-cluster 95%% CI]):")
    for r in sorted(rows, key=lambda r: (r["z_dim"], r["backbone"])):
        print("  %-7s z=%-3d residual=%.3f [%.3f, %.3f]"
              % (r["backbone"], r["z_dim"], r["res_mean"], r["res_lo"], r["res_hi"]))
    print("\nmatched-dim contrast (TSMNet - EEGNet residual):")
    for c in j.get("matched_dim_contrast", []):
        tag = " [UNDERPOWERED n<6 -> excluded]" if c.get("underpowered") else ""
        print("  z=%-3d vs z=%-3d : %+.3f [%+.3f, %+.3f] (%d folds)%s"
              % (c["tz"], c["ez"], c["diff"], c["lo"], c["hi"], c["ncommon"], tag))
    print("\nOLS residual ~ log(d_z) + arch + log(d_z):arch (fold-cluster 95%% CI):")
    for nm, v in j.get("ols", {}).items():
        excl = "***" if (v["lo"] > 0 or v["hi"] < 0) else "   "
        print("  %-16s %+.4f [%+.4f, %+.4f] %s" % (nm, v["coef"], v["lo"], v["hi"], excl))
    print("FIG6_DONE")


if __name__ == "__main__":
    main()
