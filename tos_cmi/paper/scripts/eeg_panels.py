"""Shared EEG ablation/LPC panel drawers (figure logic, not style). Used by Fig 4 (TSMNet) and Fig 5
(EEGNet) so their subject-decode / task / selectivity panels share identical axes for direct comparison."""
from __future__ import annotations
import numpy as np
from . import plot_style as ps

REPS = ["Z", "RZ", "PNZ", "Rrand"]
REP_LABEL = {"Z": "full Z", "RZ": "RZ\n(del V_D)", "PNZ": "V_D\nonly", "Rrand": "random-k"}


def _bars(ax, abl, base_m, reps):
    """grouped bars: reps on x, linear vs mlp paired."""
    m = abl["metrics"]; x = np.arange(len(reps)); w = 0.38
    for i, fam in enumerate(["linear", "mlp"]):
        vals = [m.get("%s_%s_%s" % (base_m, r, fam), {}).get("mean", np.nan) for r in reps]
        sds = [m.get("%s_%s_%s" % (base_m, r, fam), {}).get("sd", 0.0) for r in reps]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=sds, capsize=2, label=fam,
               color=ps.COLORS["linear" if fam == "linear" else "mlp"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([REP_LABEL[r] for r in reps])
    ax.set_ylim(0, 1.02)


def draw_subject_decode(ax, abl):
    _bars(ax, abl, "domain", REPS)
    ax.axhline(abl["domain_chance"], ls="--", lw=1, color=ps.COLORS["chance"], label="chance")
    ax.set_ylabel("subject decode acc"); ax.set_title("Subject leakage vs deletion")
    ax.legend(loc="upper right", ncol=1)


def draw_task(ax, abl):
    _bars(ax, abl, "task", ["Z", "RZ"])
    ax.axhline(abl["label_chance"], ls="--", lw=1, color=ps.COLORS["chance"], label="chance")
    ax.set_ylabel("task decode acc"); ax.set_title("Task preserved after deletion")
    ax.legend(loc="lower right")


def draw_selectivity(ax, abl):
    """leakage removed by informed V_D vs same-k random: (Z - RZ) and (Z - Rrand)."""
    m = abl["metrics"]; x = np.arange(2); w = 0.38
    for i, fam in enumerate(["linear", "mlp"]):
        Z = lambda r: m.get("domain_%s_%s" % (r, fam), {}).get("mean", np.nan)
        vals = [Z("Z") - Z("RZ"), Z("Z") - Z("Rrand")]
        ax.bar(x + (i - 0.5) * w, vals, w, label=fam,
               color=ps.COLORS["linear" if fam == "linear" else "mlp"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["informed\n(V_D)", "random-k"])
    ax.set_ylabel("subject decode removed"); ax.set_ylim(0, 0.75)
    ax.set_title("Removal is selective?"); ax.legend(loc="upper right")


def draw_nDcand(ax, abl):
    v = abl["nDcand"]["vals"]; zd = abl["z_dim"]
    ax.hist(v, bins=np.arange(0.5, max(8, v.max() + 1.5)), color=ps.COLORS["PNZ"], alpha=0.85)
    ax.set_xlabel("nDcand (deleted dims)"); ax.set_ylabel("folds x seeds")
    ax.set_title("Candidate rank (z_dim=%d)" % zd)
    ax.text(0.97, 0.92, "mean %.1f / %d\n= %.0f%% of latent" % (v.mean(), zd, 100 * v.mean() / zd),
            transform=ax.transAxes, ha="right", va="top", fontsize=8)


def draw_lpc_tradeoff(ax, sweep):
    """global LPC: subject leakage (left) and target acc (right) vs lambda -> removable but no DG gain."""
    lams = sweep["lams"]; pl = sweep["per_lam"]
    subj = [pl[l]["subj"]["median"] for l in lams]; tgt = [pl[l]["tgt"]["median"] for l in lams]
    xs = np.arange(len(lams))
    ax.plot(xs, subj, "-o", color=ps.COLORS["subj"], label="subject decode")
    ax.axhline(sweep["chance_subj"], ls="--", lw=1, color=ps.COLORS["subj"], alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([("%g" % l) for l in lams]); ax.set_xlabel("LPC lambda")
    ax.set_ylabel("subject decode", color=ps.COLORS["subj"]); ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelcolor=ps.COLORS["subj"])
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(xs, tgt, "-s", color=ps.COLORS["tgt"], label="target bAcc")
    ax2.set_ylabel("target bAcc", color=ps.COLORS["tgt"]); ax2.tick_params(axis="y", labelcolor=ps.COLORS["tgt"])
    ax2.set_ylim(0.2, 0.55)
    ax.set_title("Global LPC: leakage falls, target flat")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7)
