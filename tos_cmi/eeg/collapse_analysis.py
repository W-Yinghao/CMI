"""Phase 2.1 -- classify the global-LPC collapse mechanism from the per-epoch curves, and plot.
Per run (fold x seed x lambda) reads tos_cmi/eeg curve json and derives:
  collapse_epoch, final_source_bAcc, final_target_bAcc, final_eff_rank, max_grad_norm,
  median_grad_LPC_over_task, median_grad_cosine, collapse_type.
collapse_type heuristic (chance=1/n_cls=0.25):
  no_collapse           : final source bAcc >= 0.33 (well above chance)
  optimization_instability: collapsed AND (a grad spike: max/median grad_total > 4, OR a NaN/inf,
                            OR a SHARP source-bAcc drop > 0.15 between consecutive logged epochs)
  smooth_compression    : collapsed AND no spike/sharp-drop (monotone eff_rank / bAcc decline)
Bimodality across folds/seeds at fixed lambda (some runs collapse, some not) is itself an
optimization signature and is reported as the per-lambda collapse fraction.
Plots (faceted by lambda, thin line per fold/seed): task_CE, lambda*LPC penalty, encoder grad norm,
effective rank -- all vs epoch. Outputs summary.json + collapse_curves_*.png.
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np

BASE = "tos_cmi/results/tos_cmi_eeg_frozen/lpc_collapse_curves"
CHANCE = 0.25
LAMS = [0.0, 0.3, 1.0, 3.0]


def _classify(rec):
    c = rec["curves"]
    if not c:
        return {"collapse_type": "no_curves"}
    ep = [e["epoch"] for e in c]
    bacc = np.array([e["eval_source_bAcc"] for e in c])
    eff = np.array([e["eff_rank"] for e in c])
    gnorm = np.array([e["grad_total_encoder_norm"] for e in c])
    nonfin = any(e.get("grad_nonfinite") for e in c)
    cos = [e["cos_task_LPC"] for e in c if "cos_task_LPC" in e]
    ratio = [e["grad_LPC_encoder_norm"] / (e["grad_task_encoder_norm"] + 1e-9)
             for e in c if "grad_LPC_encoder_norm" in e]
    final_b = float(bacc[-1]); final_eff = float(eff[-1])
    collapsed = final_b < 0.33
    # collapse epoch = first sustained drop below 0.33
    coll_ep = None
    for i in range(len(bacc)):
        if bacc[i] < 0.33 and (i == len(bacc) - 1 or bacc[i + 1] < 0.33):
            coll_ep = ep[i]; break
    spike = float(gnorm.max() / (np.median(gnorm) + 1e-9))
    sharp = float(np.max(-np.diff(bacc))) if len(bacc) > 1 else 0.0   # biggest 1-step bAcc drop
    if not collapsed:
        ctype = "no_collapse"
    elif nonfin or spike > 4.0 or sharp > 0.15:
        ctype = "optimization_instability"
    else:
        ctype = "smooth_compression"
    return {"collapse_type": ctype, "collapsed": bool(collapsed), "collapse_epoch": coll_ep,
            "final_source_bAcc": final_b, "final_target_bAcc": float(rec.get("final_target_bAcc", float("nan"))),
            "final_eff_rank": final_eff, "max_grad_norm": float(gnorm.max()),
            "grad_spike_ratio": spike, "sharpest_bAcc_drop": sharp, "grad_nonfinite": bool(nonfin),
            "median_grad_LPC_over_task": (float(np.median(ratio)) if ratio else None),
            "median_grad_cosine": (float(np.median(cos)) if cos else None)}


def _plots(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(matplotlib unavailable: %r -- skipping plots)" % e); return
    fields = [("train_task_CE", "task CE"), ("lpc_weighted", "lambda*LPC penalty"),
              ("grad_total_encoder_norm", "encoder grad norm"), ("eff_rank", "effective rank")]
    fig, axes = plt.subplots(len(fields), len(LAMS), figsize=(4 * len(LAMS), 3 * len(fields)),
                             squeeze=False)
    for fi, (key, ylab) in enumerate(fields):
        for li, lam in enumerate(LAMS):
            ax = axes[fi][li]
            for r in runs:
                if abs(r["lam"] - lam) > 1e-9 or not r["curves"]:
                    continue
                c = r["curves"]; x = [e["epoch"] for e in c]
                if key == "lpc_weighted":
                    yv = [e["train_lambda"] * e["train_LPC_raw"] for e in c]
                else:
                    yv = [e[key] for e in c]
                ax.plot(x, yv, lw=0.7, alpha=0.7)
            if fi == 0:
                ax.set_title("lambda=%g" % lam)
            if li == 0:
                ax.set_ylabel(ylab)
            if fi == len(fields) - 1:
                ax.set_xlabel("epoch")
    fig.suptitle("Phase 2.1 LPC collapse curves (BNCI2014_001/TSMNet; thin line = fold x seed)")
    fig.tight_layout()
    out = "%s/collapse_curves.png" % BASE
    fig.savefig(out, dpi=110); print("wrote", out)


def main():
    runs = []
    for p in sorted(glob.glob("%s/*.json" % BASE)):
        if os.path.basename(p) == "summary.json":
            continue
        r = json.load(open(p)); r["_cls"] = _classify(r); runs.append(r)
    print("=== Phase 2.1 per-run collapse classification ===")
    for r in sorted(runs, key=lambda r: (r["lam"], r["target_subject"], r["seed"])):
        cl = r["_cls"]
        print("lam=%-4g sub%d s%d | final src=%.3f tgt=%.3f effrk=%.0f | spike=%.1f sharp=%.2f "
              "cos=%s ratio=%s | %s%s"
              % (r["lam"], r["target_subject"], r["seed"], cl["final_source_bAcc"],
                 cl.get("final_target_bAcc", float("nan")), cl["final_eff_rank"],
                 cl["grad_spike_ratio"], cl["sharpest_bAcc_drop"],
                 None if cl["median_grad_cosine"] is None else round(cl["median_grad_cosine"], 2),
                 None if cl["median_grad_LPC_over_task"] is None else round(cl["median_grad_LPC_over_task"], 1),
                 cl["collapse_type"], "" if cl.get("collapse_epoch") is None else " @ep%s" % cl["collapse_epoch"]))
    # per-lambda bimodality (collapse fraction) + dominant type
    from collections import Counter
    print("\n=== per-lambda: collapse fraction (bimodality) + types ===")
    agg = {}
    for lam in LAMS:
        rs = [r for r in runs if abs(r["lam"] - lam) < 1e-9]
        if not rs:
            continue
        nc = sum(r["_cls"].get("collapsed") for r in rs)
        types = Counter(r["_cls"]["collapse_type"] for r in rs)
        agg["%g" % lam] = {"n": len(rs), "collapsed": nc, "collapse_fraction": nc / len(rs),
                           "types": dict(types),
                           "median_cos": float(np.median([r["_cls"]["median_grad_cosine"] for r in rs
                                                          if r["_cls"]["median_grad_cosine"] is not None] or [np.nan]))}
        print("  lam=%-4g  collapsed %d/%d (%.0f%%)  types=%s  median cos(task,LPC)=%.2f"
              % (lam, nc, len(rs), 100 * nc / len(rs), dict(types), agg["%g" % lam]["median_cos"]))
    summary = {"runs": [{"lam": r["lam"], "target_subject": r["target_subject"], "seed": r["seed"],
                         **r["_cls"]} for r in runs], "by_lambda": agg}
    json.dump(summary, open("%s/summary.json" % BASE, "w"), indent=1)
    print("wrote %s/summary.json" % BASE)
    _plots(runs)
    print("COLLAPSE_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
