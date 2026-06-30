"""Fig 5 -- EEGNet contrast: low-rank deletion removes much more leakage (linear ~fully, MLP partial)
at ~0 task cost, yet global LPC leakage removal gives NO target-accuracy gain (removable but useless).
Panels A-C share axes with Fig 4 for direct comparison. Panel D = LPC leakage-vs-target trade-off.
Run: python -m tos_cmi.paper.scripts.plot_fig5_eegnet_contrast
"""
import matplotlib.pyplot as plt
from . import plot_style as ps
from . import eeg_panels as P
from ._data import load_ablation, load_lpc_sweep


def main():
    abl = load_ablation("EEGNet")
    sweep = load_lpc_sweep("EEGNet")
    fig, ax = plt.subplots(1, 4, figsize=(ps.PANEL_W * 4, 3.0))
    P.draw_subject_decode(ax[0], abl)
    P.draw_task(ax[1], abl)
    P.draw_selectivity(ax[2], abl)
    P.draw_lpc_tradeoff(ax[3], sweep)
    for a, L in zip(ax, "ABCD"):
        ps.panel_label(a, L)
    fig.suptitle("EEGNet (conv, z_dim=%d): low-rank deletion removes leakage (A-C) but global LPC removal "
                 "gives no target gain (D) -- removable but useless (n=%d folds x seeds)"
                 % (abl["z_dim"], abl["n"]), y=1.02)
    ps.save(fig, "fig5_eegnet_contrast")
    m = abl["metrics"]
    for fam in ["linear", "mlp"]:
        print("EEGNet %s: subj Z=%.3f RZ=%.3f Rrand=%.3f | task Z=%.3f RZ=%.3f"
              % (fam, m["domain_Z_%s" % fam]["mean"], m["domain_RZ_%s" % fam]["mean"],
                 m["domain_Rrand_%s" % fam]["mean"], m["task_Z_%s" % fam]["mean"], m["task_RZ_%s" % fam]["mean"]))
    pl = sweep["per_lam"]
    print("EEGNet LPC: " + " | ".join("lam%g subj=%.2f tgt=%.2f" %
          (l, pl[l]["subj"]["median"], pl[l]["tgt"]["median"]) for l in sweep["lams"]))
    print("FIG5_DONE")


if __name__ == "__main__":
    main()
