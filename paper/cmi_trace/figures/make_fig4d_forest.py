#!/usr/bin/env python
"""CMI-Trace Figure 4D (P0.6): forest plot of source-fitted target-deployment ΔbAcc over all valid
dataset–backbone cells, with vertical reference lines at 0 and +0.01 (the practical threshold). Reads the
TOS deployment paired CSV (immutable); fails loudly if absent. Each row is annotated with its three-state
deployment CI verdict (deployment_ci.deployment_ci_state).

  python paper/cmi_trace/figures/make_fig4d_forest.py \
      --paired tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy/erasure_target_deploy_paired.csv \
      --out paper/cmi_trace/figures/fig4d_forest.png
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _load(path):
    if not Path(path).exists():
        sys.exit(f"[fig4d] MISSING {path} — run the TOS deployment bridge first. No figure is fabricated.")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired",
                    default="tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy/erasure_target_deploy_paired.csv")
    ap.add_argument("--out", default="paper/cmi_trace/figures/fig4d_forest.png")
    a = ap.parse_args()
    from tos_cmi.eeg.deployment_ci import deployment_ci_state
    rows = _load(a.paired)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels, means, los, his, colors = [], [], [], [], []
    cmap = {"confirmed_practical_benefit": "#2E7D32", "practical_gain_ruled_out": "#B71C1C",
            "inconclusive": "#F9A825"}
    for r in rows:
        lo, hi = float(r["dtgt_bacc_lo"]), float(r["dtgt_bacc_hi"])
        state = deployment_ci_state(lo, hi)
        labels.append(f"{r.get('backbone','?')}·{r.get('method','?')}")
        means.append(float(r["dtgt_bacc"])); los.append(lo); his.append(hi); colors.append(cmap[state])
    y = range(len(means))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(means) + 1)))
    for i, (m, lo, hi, c) in enumerate(zip(means, los, his, colors)):
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot(m, i, "o", color=c)
    ax.axvline(0, color="k", lw=0.8, ls="-")
    ax.axvline(0.01, color="gray", lw=0.9, ls="--", label="+0.01 practical threshold")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("target ΔbAcc (erasure − full), fold-cluster 95% CI")
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print(f"[fig4d] wrote {a.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
