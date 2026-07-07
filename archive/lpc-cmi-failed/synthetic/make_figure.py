"""Figure 2: target balanced accuracy and conditional domain leakage vs lambda,
per training objective, from synthetic/results.json."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "synthetic/results.json"))
sweep = res["sweep"]
lams = sorted([float(k) for k in sweep], key=float)
methods = list(sweep[str(lams[0])].keys())
labels = {"erm": "ERM", "marginal": "marginal I(Z;D)", "chain": "chain I(Z;D,Y)",
          "lpc_uniform": "LPC uniform", "lpc_prior": "LPC-CMI (ours)"}
styles = {"erm": "k--", "marginal": "C1-o", "chain": "C3-s",
          "lpc_uniform": "C0-^", "lpc_prior": "C2-D"}

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for m in methods:
    tgt = [sweep[str(l)][m]["target_bacc"][0] * 100 for l in lams]
    tsd = [sweep[str(l)][m]["target_bacc"][1] * 100 for l in lams]
    leak = [sweep[str(l)][m]["leakage_kl"][0] for l in lams]
    lsep = [sweep[str(l)][m]["label_sep"][0] * 100 for l in lams]
    x = range(len(lams))
    ax[0].errorbar(x, tgt, yerr=tsd, fmt=styles[m], capsize=2, label=labels.get(m, m), ms=5)
    ax[1].plot(x, leak, styles[m], label=labels.get(m, m), ms=5)
    ax[2].plot(x, lsep, styles[m], label=labels.get(m, m), ms=5)
for a, ttl, yl in zip(ax, ["Target balanced acc.", "Conditional domain leakage", "Label separability"],
                      ["target bal. acc. (%)", "KL(q‖π_y)  (↓ better)", "linear-probe acc. (%)"]):
    a.set_xticks(range(len(lams))); a.set_xticklabels([str(l) for l in lams])
    a.set_xlabel("λ (regularization strength)"); a.set_title(ttl); a.set_ylabel(yl); a.grid(alpha=.3)
ax[0].legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig("synthetic/figure2_sanity.png", dpi=150)
print("saved -> synthetic/figure2_sanity.png")
