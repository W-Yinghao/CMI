"""Consolidate all results/*.json into a master table + leakage-vs-robustness scatter (H1).

Handles both result schemas (pre/post metric fix). For SCPS (ADFTD) uses subject-level
balanced accuracy as the headline; otherwise per-target balanced accuracy.
"""
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCPS = {"ADFTD", "ADFTD_bin"}


def acc_of(v):
    return v.get("per_target_balanced_acc_mean", v.get("balanced_acc_mean"))


def worst_of(v):
    return v.get("worst_target_balanced_acc", v.get("worst_subject"))


rows = []
for f in sorted(glob.glob("results/*.json")):
    if "INVALID" in f:
        continue
    o = json.load(open(f))
    cfg = o.get("config", {})
    ds = cfg.get("dataset", os.path.basename(f).split("_")[0])
    bb = cfg.get("backbone", "EEGNet")
    tag = os.path.basename(f).replace(".json", "")
    for name, v in o["summary"].items():
        head = v.get("subject_balanced_acc") if ds in SCPS else acc_of(v)
        rows.append(dict(file=tag, dataset=ds, backbone=bb, config=name,
                         method=name.split(":")[0],
                         acc=head, mean_bacc=acc_of(v), worst=worst_of(v),
                         leak=v.get("leakage_kl"), label_sep=v.get("label_sep"),
                         scps=ds in SCPS))
df = pd.DataFrame(rows)
df.to_csv("analysis/master_table.csv", index=False)
print(f"=== {len(df)} (file,config) rows across {df.file.nunique()} result files ===")

# Per-method summary: how often does each method have the lowest leakage / beat ERM?
print("\n=== mean leakage by method (lower=better) ===")
print(df.groupby("method")["leak"].mean().sort_values().round(3).to_string())

# H1: across all (file) groups, does lower leakage track higher worst-target acc? (MCPS only)
mcps = df[(~df.scps) & df.worst.notna() & df.leak.notna()].copy()
if len(mcps) > 5:
    r = np.corrcoef(mcps.leak, mcps.worst)[0, 1]
    print(f"\n=== H1 (MCPS): corr(leakage, worst-target bacc) = {r:+.3f}  (n={len(mcps)}) ===")

# Figure: leakage vs mean balanced acc, colored by method
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
methods = ["erm", "lpc_prior", "cdann", "marginal", "lpc_uniform", "coral", "mmd", "irm", "vrex", "groupdro", "dann", "iib", "supcon", "lpc_supcon", "chain"]
cmap = {m: plt.cm.tab20(i / 20) for i, m in enumerate(methods)}
for a, (yl, yc) in zip(ax, [("mean balanced acc (%)", "mean_bacc"), ("worst-target bacc (%)", "worst")]):
    sub = df[df.leak.notna() & df[yc].notna() & (~df.scps)]
    for m in sub.method.unique():
        s = sub[sub.method == m]
        a.scatter(s.leak, s[yc] * 100, s=28, color=cmap.get(m, "gray"),
                  label=m, alpha=0.8, edgecolors="k", linewidths=0.3)
    a.set_xlabel("conditional domain leakage  KL(q‖π_y)  (↓)")
    a.set_ylabel(yl); a.set_xscale("symlog", linthresh=0.05); a.grid(alpha=.3)
ax[0].legend(fontsize=6, ncol=2, loc="lower right")
ax[0].set_title("Leakage vs mean accuracy (MCPS)")
ax[1].set_title("Leakage vs worst-target accuracy (H1)")
plt.tight_layout(); plt.savefig("analysis/leakage_vs_acc.png", dpi=150)
print("\nsaved -> analysis/master_table.csv, analysis/leakage_vs_acc.png")
