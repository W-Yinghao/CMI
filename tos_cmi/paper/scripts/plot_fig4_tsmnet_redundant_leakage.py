"""Fig 4 -- TSMNet subject leakage is high-dimensional / redundant: low-rank deletion only dents it.
Message: measurement positive (V_D localized), control negative (redundant), gate abstention correct.
Run: python -m tos_cmi.paper.scripts.plot_fig4_tsmnet_redundant_leakage
"""
import matplotlib.pyplot as plt
from . import plot_style as ps
from . import eeg_panels as P
from ._data import load_ablation


def main():
    abl = load_ablation("TSMNet")
    fig, ax = plt.subplots(1, 4, figsize=(ps.PANEL_W * 4, 3.0))
    P.draw_subject_decode(ax[0], abl)
    P.draw_task(ax[1], abl)
    P.draw_selectivity(ax[2], abl)
    P.draw_nDcand(ax[3], abl)
    for a, L in zip(ax, "ABCD"):
        ps.panel_label(a, L)
    fig.suptitle("TSMNet (LogEig/SPD, z_dim=%d): leakage redundant -> low-rank deletion insufficient "
                 "(n=%d folds x seeds, %d seeds)" % (abl["z_dim"], abl["n"], abl["n_seeds"]), y=1.02)
    ps.save(fig, "fig4_tsmnet_redundant_leakage")
    # console trace (numbers must match claim_evidence_table C4)
    m = abl["metrics"]
    for fam in ["linear", "mlp"]:
        print("TSMNet %s: subj Z=%.3f RZ=%.3f Rrand=%.3f | task Z=%.3f RZ=%.3f"
              % (fam, m["domain_Z_%s" % fam]["mean"], m["domain_RZ_%s" % fam]["mean"],
                 m["domain_Rrand_%s" % fam]["mean"], m["task_Z_%s" % fam]["mean"], m["task_RZ_%s" % fam]["mean"]))
    print("FIG4_DONE")


if __name__ == "__main__":
    main()
