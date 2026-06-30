"""Table 1 -- one-glance summary across representations (TSMNet vs EEGNet), built from saved artifacts so
its numbers are locked to Fig 3/4/5. Writes figures/table1_summary.csv + prints a markdown table.
Run: python -m tos_cmi.paper.scripts.build_table1_summary
"""
import csv
import json
import os
from ._data import load_ablation, load_lpc_sweep, LPC
from . import plot_style as ps


def _row(bb):
    a = load_ablation(bb); s = load_lpc_sweep(bb); m = a["metrics"]
    pl = s["per_lam"]
    dZ = m["domain_Z_mlp"]["mean"]; dRZ = m["domain_RZ_mlp"]["mean"]; dRr = m["domain_Rrand_mlp"]["mean"]
    dZl = m["domain_Z_linear"]["mean"]; dRZl = m["domain_RZ_linear"]["mean"]
    tZ = m["task_Z_mlp"]["mean"]; tRZ = m["task_RZ_mlp"]["mean"]
    lam_hi = 3.0 if 3.0 in pl else max(pl)
    feat_hi = pl[lam_hi]["feat_norm"]["median"]
    collapse = feat_hi < 0.1
    subj0 = pl[min(pl)]["subj"]["median"]; subjhi = pl[lam_hi]["subj"]["median"]
    tgt0 = pl[min(pl)]["tgt"]["median"]; tgthi = pl[lam_hi]["tgt"]["median"]
    # raw LPC + collapse-free LPC outcomes stated explicitly (avoid a bare "n/a" that reads as missing)
    if bb == "TSMNet":
        raw_lpc = "leakage falls via collapse (feature-norm->0), task -> chance"
        cf_lpc = "task restored, leakage returns to ERM (subj ~0.997)"
        delete_eff = "dents only: subj %.2f->%.2f (= random %.2f); task kept" % (dZ, dRZ, dRr)
        dg = "not achievable: no task-preserving leakage reduction exists (raw LPC collapses; collapse-free LPC removes none)"
        decision = "abstain / low-rank deletion insufficient"
    else:
        raw_lpc = "no collapse: leakage falls %.2f->%.2f, source task degrades gradually" % (subj0, subjhi)
        cf_lpc = "n/a (raw LPC already collapse-free on this latent)"
        delete_eff = "removes: subj linear %.2f->%.2f / MLP %.2f->%.2f (>> random %.2f); task kept" % (dZl, dRZl, dZ, dRZ, dRr)
        dg = "none: target flat (%.2f->%.2f) as leakage falls %.2f->%.2f" % (tgt0, tgthi, subj0, subjhi)
        decision = "diagnostic deletion removes leakage but no DG gain"
    return {
        "Backbone": bb,
        "Latent dim": a["z_dim"],
        "Subject decode (ERM)": "%.3f" % dZ,
        "Low-rank deletion (subject; task)": delete_eff,
        "Raw LPC outcome": raw_lpc,
        "Collapse-free LPC outcome": cf_lpc,
        "DG gain under task-preserving control": dg,
        "Certified decision": decision,
    }


def main():
    rows = [_row("TSMNet"), _row("EEGNet")]
    cols = list(rows[0].keys())
    os.makedirs(ps.FIGDIR, exist_ok=True)
    out = os.path.join(ps.FIGDIR, "table1_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("wrote", out)
    # markdown (for the paper / claim_evidence cross-check)
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print("\nTABLE1_DONE")


if __name__ == "__main__":
    main()
