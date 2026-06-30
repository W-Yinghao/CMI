"""Shared plot STYLE only -- no data logic (per write-up plan). Fonts, sizes, panel labels, save."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIGDIR = os.path.join(ROOT, "tos_cmi", "paper", "figures")

# consistent baseline colors across Fig 4 / Fig 5 (so panels are comparable)
COLORS = {
    "Z": "#444444",        # full latent (ERM)
    "RZ": "#1f77b4",       # complement after V_D deletion (informed)
    "PNZ": "#9467bd",      # the deleted V_D subspace itself
    "Rrand": "#d62728",    # random-k removal (control)
    "linear": "#1f77b4",
    "mlp": "#ff7f0e",
    "subj": "#d62728",
    "tgt": "#2ca02c",
    "chance": "#888888",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 120, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.4,
})

PANEL_W = 3.2   # inches per panel column


def panel_label(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top", ha="left")


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIGDIR, "%s.%s" % (name, ext))
        fig.savefig(out, bbox_inches="tight")   # include suptitle/labels, no clipping
    print("wrote %s.{pdf,png}" % os.path.join(FIGDIR, name))
