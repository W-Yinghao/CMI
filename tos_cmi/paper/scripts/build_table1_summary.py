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
    # collapse-free LPC removes leakage? TSMNet: read variant_compare keystone; EEGNet: raw LPC is collapse-free
    if bb == "TSMNet":
        vc = json.load(open(os.path.join(LPC, "TSMNet", "variant_compare.json")))
        cf_removes = "no (subj~ERM in all collapse-free cells)" if not vc["leakage_reduced_any_collapse_free"] else "yes"
        decision = "abstain / low-rank deletion insufficient"
        collapse_s = "yes (feature-norm -> 0)"
        delete_eff = "dents only (RZ %.2f vs random %.2f)" % (dRZ, dRr)
    else:
        cf_removes = "yes (subj %.2f -> %.2f, no collapse)" % (subj0, subjhi)
        decision = "diagnostic deletion removes leakage but no DG gain"
        collapse_s = "no (gradual; feat_norm stays > 0)"
        delete_eff = "linear %.2f->%.2f, MLP %.2f->%.2f (>> random %.2f)" % (dZl, dRZl, dZ, dRZ, dRr)
    return {
        "Backbone": bb,
        "Latent dim": a["z_dim"],
        "Subject decode (ERM, MLP)": "%.3f" % dZ,
        "Low-rank deletion effect": delete_eff,
        "Task cost of deletion": "~0 (%.2f->%.2f)" % (tZ, tRZ),
        "Global LPC collapse?": collapse_s,
        "Collapse-free LPC removes leakage?": cf_removes,
        "Target DG gain from removal?": "n/a (collapses)" if collapse else "no (tgt %.2f->%.2f)" % (tgt0, tgthi),
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
