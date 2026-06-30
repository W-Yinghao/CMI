"""Fig 6 -- removability is capacity-mediated, not architecture-type-mediated. Plots removability signals
vs latent dimension for TSMNet (SPD/LogEig) and EEGNet (conv); the near-coincidence of the two curves at
matched dimension resolves the architecture-vs-dimension confound. Reads the Track-C factorial json.
  python -m tos_cmi.paper.scripts.plot_fig6_factorial [seed]
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from . import plot_style as ps

FACT = os.path.join(ps.ROOT, "tos_cmi", "results", "tos_cmi_eeg_frozen", "factorial")
ACOL = {"TSMNet": "#1f77b4", "EEGNet": "#d62728"}
AMARK = {"TSMNet": "o", "EEGNet": "s"}


def _series(rows, bb, key):
    r = sorted([x for x in rows if x["backbone"] == bb], key=lambda x: x["z_dim"])
    return [x["z_dim"] for x in r], [x[key] for x in r]


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "0"
    rows = json.load(open("%s/factorial_removability_seed%s.json" % (FACT, seed)))
    chance = rows[0]["chance_subj"]
    fig, ax = plt.subplots(1, 4, figsize=(ps.PANEL_W * 4, 3.0))
    panels = [("subj_full_mlp", "subject decode (MLP)", "Subject leakage grows\nwith latent capacity", None),
              ("subj_LEACE_mlp", "MLP residual after LEACE", "LEACE residual is\ncapacity-mediated", None),
              ("task_full_lin", "task decode (linear)", "Task preserved across\ncapacity (full vs LEACE)", "task_LEACE_lin"),
              ("subj_TOS_mlp", "MLP residual after deletion", "Rank-limited deletion (TOS)\ndegrades with capacity", "subj_rand_mlp")]
    for i, (key, ylab, title, key2) in enumerate(panels):
        a = ax[i]
        for bb in ["TSMNet", "EEGNet"]:
            x, yv = _series(rows, bb, key)
            a.plot(x, yv, "-" + AMARK[bb], color=ACOL[bb], label=bb)
            if key2:   # secondary series (dashed): after-LEACE task, or random-k residual
                x2, y2 = _series(rows, bb, key2)
                a.plot(x2, y2, "--" + AMARK[bb], color=ACOL[bb], alpha=0.55,
                       label="%s (%s)" % (bb, "LEACE" if "task" in key2 else "rand-$k$"))
        a.set_xscale("log"); a.set_xlabel("latent dimension $d_z$"); a.set_ylabel(ylab)
        a.set_title(title, fontsize=9)
        if "subj" in key or "residual" in ylab:
            a.axhline(chance, ls=":", lw=1, color="#888"); a.set_ylim(0, 1.02)
        a.legend(fontsize=6.5, loc="best")
        ps.panel_label(a, "ABCD"[i])
    fig.suptitle("Removability vs latent dimension: TSMNet (SPD) and EEGNet (conv) nearly coincide at matched "
                 "$d_z$ -> capacity-mediated, not architecture-type (BCI-IV-2a; seed %s)" % seed, y=1.02)
    ps.save(fig, "fig6_capacity_factorial")
    # console: matched-dim overlap evidence
    print("LEACE residual at matched dim (capacity-mediated if TSMNet~EEGNet):")
    for r in sorted(rows, key=lambda r: (r["z_dim"], r["backbone"])):
        print("  %-7s z=%-3d residual=%.3f" % (r["backbone"], r["z_dim"], r["subj_LEACE_mlp"]))
    print("FIG6_DONE")


if __name__ == "__main__":
    main()
