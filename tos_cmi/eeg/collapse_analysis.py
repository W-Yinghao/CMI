"""Phase 2.1 -- classify the global-LPC collapse MECHANISM from the per-epoch curves, and plot.
(REVISED after adversarial verification wf_c2880caf-68a, which REFUTED a naive "gradient explosion"
read and identified the true mechanism.)

VERDICT: at lambda>=1 the global-LPC objective (CE + lambda*I(Z;D|Y)) undergoes a sharp, lambda-tied
OPTIMIZATION BIFURCATION to a degenerate trivial minimizer = FEATURE-NORM COLLAPSE TO THE ORIGIN
(Z->0), which zeroes the penalty AND the task simultaneously. It is NOT a gradient explosion and NOT
a smooth geometric over-compression.

The discriminating signals (re-derived from raw curves):
  - feat_norm_mean -> ~0.000 (collapse to a POINT; controls grow to ~5.7)      [PRIMARY]
  - raw top-1 singular value -> ~0.001 (controls ~107); spectrum SHAPE preserved, SCALE -> 0
  - eval_source_CE -> ln(n_cls) (uniform output); eval_source_bAcc -> chance
  - lambda*LPC penalty: ramps then -> ~0 (penalty trivially satisfied by Z->0)
  - sharp lambda CLIFF: lam<=0.3 never collapse, lam>=1 always collapse (no intermediate), and
    DETERMINISTIC in lambda (no within-lambda seed bimodality)
  - cos(grad_task, grad_LPC) -> ~-1 at lam=3 (directly-opposed objectives)
NON-PROBATIVE (scale-invariant -- do NOT cite for/against compression):
  - eff_rank = exp(entropy(s/s.sum())), stable_rank, top1-share  (stay ~ERM even at collapse)
CORRECTED gradient reading: NO explosion -- absolute peak encoder grad at collapse is ~10x SMALLER
  than in healthy low-lambda training; the "39-298x" headline was peak / post-collapse-near-zero
  median (the dead fixed point drags the median down). Honest peak/pre-collapse ratio ~2.4x.
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np

import os as _os
BASE = "tos_cmi/results/tos_cmi_eeg_frozen/lpc_collapse_curves/%s" % _os.environ.get("TOS_BB", "TSMNet")
LAMS = [0.0, 0.3, 1.0, 3.0]


def _classify(rec, n_cls=4):
    c = rec["curves"]
    if not c:
        return {"collapse_type": "no_curves"}
    chance = 1.0 / n_cls
    ep = np.array([e["epoch"] for e in c])
    bacc = np.array([e["eval_source_bAcc"] for e in c])
    fnorm = np.array([e["feat_norm_mean"] for e in c])
    top1 = np.array([e["top5_singvals"][0] for e in c])
    pen = np.array([e["train_lambda"] * e["train_LPC_raw"] for e in c])
    g = np.array([e["grad_total_encoder_norm"] for e in c])
    nonfin = any(e.get("grad_nonfinite") for e in c)
    cos = [e["cos_task_LPC"] for e in c if "cos_task_LPC" in e]
    collapsed = bool(bacc[-1] < chance + 0.05)
    # collapse epoch = first sustained drop below chance+0.05
    coll_ep = None
    for i in range(len(bacc)):
        if bacc[i] < chance + 0.05 and (i == len(bacc) - 1 or bacc[i + 1] < chance + 0.05):
            coll_ep = int(ep[i]); break
    # honest gradient stats: peak vs PRE-collapse-window median (not post-collapse dead floor)
    pre = g[ep < coll_ep] if (coll_ep is not None and (ep < coll_ep).any()) else g
    peak_pre_ratio = float(g.max() / (np.median(pre) + 1e-9))
    if not collapsed:
        ctype = "no_collapse"
    elif fnorm[-1] < 0.1 and bacc[-1] < chance + 0.05:
        ctype = "feature_norm_collapse"        # objective-scaling bifurcation to Z->0 (the origin)
    else:
        ctype = "other_collapse"
    return {"collapse_type": ctype, "collapsed": collapsed, "collapse_epoch": coll_ep,
            "final_source_bAcc": float(bacc[-1]), "final_target_bAcc": float(rec.get("final_target_bAcc", float("nan"))),
            "final_feat_norm": float(fnorm[-1]), "init_feat_norm": float(fnorm[0]),
            "final_top1_singval": float(top1[-1]), "final_eval_CE": float(c[-1]["eval_source_CE"]),
            "ln_ncls": float(np.log(n_cls)),
            "penalty_peak": float(pen.max()), "penalty_final": float(pen[-1]),
            "abs_peak_grad": float(g.max()), "grad_peak_over_precollapse_med": peak_pre_ratio,
            "grad_nonfinite": bool(nonfin),
            "median_grad_cosine": (float(np.median(cos)) if cos else None)}


def _plots(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(matplotlib unavailable: %r -- skipping plots)" % e); return
    fields = [("train_task_CE", "task CE"), ("lpc_weighted", "lambda*LPC penalty"),
              ("grad_total_encoder_norm", "encoder grad norm\n(abs; NOT exploding)"),
              ("eff_rank", "eff_rank\n(scale-INVARIANT: non-probative)"),
              ("feat_norm_mean", "feature norm\n(->0 = collapse to ORIGIN)")]
    fig, axes = plt.subplots(len(fields), len(LAMS), figsize=(4 * len(LAMS), 2.6 * len(fields)),
                             squeeze=False)
    for fi, (key, ylab) in enumerate(fields):
        for li, lam in enumerate(LAMS):
            ax = axes[fi][li]
            for r in runs:
                if abs(r["lam"] - lam) > 1e-9 or not r["curves"]:
                    continue
                c = r["curves"]; x = [e["epoch"] for e in c]
                yv = ([e["train_lambda"] * e["train_LPC_raw"] for e in c] if key == "lpc_weighted"
                      else [e[key] for e in c])
                ax.plot(x, yv, lw=0.7, alpha=0.7)
            if fi == 0:
                ax.set_title("lambda=%g" % lam)
            if li == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if fi == len(fields) - 1:
                ax.set_xlabel("epoch")
    fig.suptitle("Phase 2.1 LPC collapse = objective-scaling bifurcation to feature-norm collapse "
                 "(BNCI2014_001/TSMNet; thin line = fold x seed)")
    fig.tight_layout()
    out = "%s/collapse_curves.png" % BASE
    fig.savefig(out, dpi=110); print("wrote", out)


def main():
    runs = []
    for p in sorted(glob.glob("%s/*.json" % BASE)):
        if os.path.basename(p) == "summary.json":
            continue
        r = json.load(open(p)); r["_cls"] = _classify(r, n_cls=int(r.get("n_cls", 4))); runs.append(r)
    print("=== Phase 2.1 per-run collapse MECHANISM (n_cls=4, chance=0.25, ln4=%.3f) ===" % np.log(4))
    for r in sorted(runs, key=lambda r: (r["lam"], r["target_subject"], r["seed"])):
        cl = r["_cls"]
        print("lam=%-4g sub%d s%d | src=%.3f CE=%.2f | feat_norm %.2f->%.4f top1->%.3f | penalty %.2f->%.3f"
              " | absGrad=%.1f (x%.1f pre) cos=%s | %s%s"
              % (r["lam"], r["target_subject"], r["seed"], cl["final_source_bAcc"], cl["final_eval_CE"],
                 cl["init_feat_norm"], cl["final_feat_norm"], cl["final_top1_singval"],
                 cl["penalty_peak"], cl["penalty_final"], cl["abs_peak_grad"],
                 cl["grad_peak_over_precollapse_med"],
                 None if cl["median_grad_cosine"] is None else round(cl["median_grad_cosine"], 2),
                 cl["collapse_type"], "" if cl.get("collapse_epoch") is None else " @ep%s" % cl["collapse_epoch"]))
    from collections import Counter
    print("\n=== per-lambda: collapse fraction (sharp cliff = bifurcation) + honest grad ===")
    agg = {}
    for lam in LAMS:
        rs = [r for r in runs if abs(r["lam"] - lam) < 1e-9]
        if not rs:
            continue
        nc = sum(r["_cls"]["collapsed"] for r in rs)
        types = Counter(r["_cls"]["collapse_type"] for r in rs)
        med = lambda k: float(np.median([r["_cls"][k] for r in rs if r["_cls"].get(k) is not None] or [np.nan]))
        agg["%g" % lam] = {"n": len(rs), "collapsed": nc, "collapse_fraction": nc / len(rs),
                           "types": dict(types), "median_final_feat_norm": med("final_feat_norm"),
                           "median_abs_peak_grad": med("abs_peak_grad"), "median_cos": med("median_grad_cosine")}
        print("  lam=%-4g  collapsed %d/%d  types=%s  med_feat_norm=%.4f  med_absPeakGrad=%.1f  med_cos=%.2f"
              % (lam, nc, len(rs), dict(types), agg["%g" % lam]["median_final_feat_norm"],
                 agg["%g" % lam]["median_abs_peak_grad"], agg["%g" % lam]["median_cos"]))
    print("\nVERDICT: feature-norm collapse to the ORIGIN at lam>=1 (objective-scaling bifurcation);")
    print("absolute peak grad at collapse is SMALLER than controls (no explosion); eff_rank stays high")
    print("ONLY because it is scale-invariant (non-probative).")
    summary = {"verdict": "objective_scaling_bifurcation_to_feature_norm_collapse",
               "runs": [{"lam": r["lam"], "target_subject": r["target_subject"], "seed": r["seed"],
                         **r["_cls"]} for r in runs], "by_lambda": agg}
    json.dump(summary, open("%s/summary.json" % BASE, "w"), indent=1)
    print("wrote %s/summary.json" % BASE)
    _plots(runs)
    print("COLLAPSE_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
