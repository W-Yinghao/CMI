"""Fig 3 -- Global LPC removes subject leakage only by collapsing the representation (TSMNet/2a).
Top: collapse-mechanism curves (task CE, feature norm, lambda*penalty) vs epoch, median per lambda.
Bottom: the keystone -- raw LPC at lambda=1 drops task AND leakage together (collapse to origin), while
warm-ramp / scale-invariant prevent the collapse, restoring task but leaving subject leakage at ERM.
Run: python -m tos_cmi.paper.scripts.plot_fig3_lpc_collapse
"""
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from . import plot_style as ps
from ._data import LPC

LAMS = [0.0, 0.3, 1.0, 3.0]
LAMC = {0.0: "#888888", 0.3: "#1f77b4", 1.0: "#ff7f0e", 3.0: "#d62728"}


def _median_curves(metric):
    """median trajectory per lambda over folds x seeds, from TSMNet raw curve jsons."""
    out = {}
    for lam in LAMS:
        arrs = []
        for p in sorted(glob.glob(os.path.join(LPC, "TSMNet", "sub*_lam%g_seed*.json" % lam))):
            c = json.load(open(p)).get("curves") or []
            if not c:
                continue
            if metric == "penalty":
                arrs.append([e["train_lambda"] * e["train_LPC_raw"] for e in c])
            else:
                arrs.append([e[metric] for e in c])
        if arrs:
            L = min(len(a) for a in arrs)
            out[lam] = np.median(np.array([a[:L] for a in arrs]), axis=0)
    return out


def main():
    vc = json.load(open(os.path.join(LPC, "TSMNet", "variant_compare.json")))
    fig = plt.figure(figsize=(ps.PANEL_W * 3, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.32)

    specs = [("train_task_CE", "task cross-entropy", "task CE -> ln(4) = chance"),
             ("feat_norm_mean", "feature norm  ||Z||", "feature norm -> 0  (collapse to origin)"),
             ("penalty", "lambda x LPC penalty", "penalty -> 0  (trivially satisfied by Z->0)")]
    for j, (metric, ylab, title) in enumerate(specs):
        ax = fig.add_subplot(gs[0, j])
        cur = _median_curves(metric)
        for lam in LAMS:
            if lam in cur:
                ax.plot(cur[lam], color=LAMC[lam], label="lambda=%g" % lam)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=9)
        if metric == "train_task_CE":
            ax.axhline(np.log(4), ls="--", lw=1, color="#444", alpha=0.7)
            ax.text(0.97, np.log(4), " ln4", transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=7)
            ax.legend(loc="center right", fontsize=7)
        ps.panel_label(ax, "ABC"[j])

    # bottom: variant keystone bar
    axb = fig.add_subplot(gs[1, :])
    cfgs = [("ERM\n(lambda=0)", "erm"), ("raw LPC\nlambda=1", "raw_lpc|1.0"),
            ("warm-ramp\nlambda=1", "lpc_warm_ramp|1.0"), ("scale-inv\nlambda=1", "lpc_scale_invariant|1.0")]
    erm = vc["erm_ref"]; tab = vc["table"]
    task, subj, collapsed = [], [], []
    for _, key in cfgs:
        if key == "erm":
            task.append(erm["src"]); subj.append(erm["subj_dec"]); collapsed.append(False)
        else:
            v = tab[key]; task.append(v["src"])
            sd = v.get("subj_dec")
            isc = (v["feat_norm"] < 0.1)
            collapsed.append(isc)
            subj.append(erm["chance_subj"] if isc or sd is None or sd != sd else sd)
    x = np.arange(len(cfgs)); w = 0.38
    axb.bar(x - w / 2, task, w, label="source task bAcc", color=ps.COLORS["tgt"], alpha=0.9)
    bsubj = axb.bar(x + w / 2, subj, w, label="subject decode", color=ps.COLORS["subj"], alpha=0.9)
    axb.axhline(erm["chance_subj"], ls="--", lw=1, color=ps.COLORS["subj"], alpha=0.5)
    axb.text(0.005, erm["chance_subj"], " subj chance", transform=axb.get_yaxis_transform(), va="bottom", fontsize=7, color=ps.COLORS["subj"])
    axb.axhline(0.25, ls=":", lw=1, color=ps.COLORS["tgt"], alpha=0.5)
    for i, isc in enumerate(collapsed):
        if isc:
            axb.text(x[i] + w / 2, subj[i] + 0.03, "Z->0\n(collapsed)", ha="center", va="bottom", fontsize=7, color=ps.COLORS["subj"])
    axb.set_xticks(x); axb.set_xticklabels([c[0] for c in cfgs])
    axb.set_ylabel("accuracy"); axb.set_ylim(0, 1.08)
    axb.set_title("Raw LPC removes leakage only via collapse (task+leakage both die); preventing the "
                  "collapse restores task but leakage returns to ERM", fontsize=9)
    axb.legend(loc="upper center", ncol=2); ps.panel_label(axb, "D")

    fig.suptitle("Global LPC removes subject leakage only by collapsing the representation (TSMNet/2a)", y=1.0, fontsize=11)
    ps.save(fig, "fig3_lpc_collapse_mechanism")
    print("bottom bar: task=%s subj=%s collapsed=%s" % ([round(t, 2) for t in task], [round(s, 2) for s in subj], collapsed))
    print("FIG3_DONE")


if __name__ == "__main__":
    main()
