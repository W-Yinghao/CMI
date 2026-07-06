"""Fork 2 Phase 0 --- mechanistic meta-analysis (NO new training). Establishes the E0 (subject-environment)
baseline: how much does any SOURCE-ONLY signal predict the actual target ΔbAcc under leave-one-subject-out?
REAL EEG is the PRIMARY panel; V2 semi-synthetic is a SEPARATE mechanism panel (never mixed into the real-EEG
correlations). Target enters only as the audited outcome. Reads committed aggregates; submit via SLURM (CPU).

Inputs (real EEG): Track B source-only gate (results/method_deepen/trackB/trackB_summary_full.json) x
erasure_report.json (subject decode linear+MLP, per <ds>_<bb>_LOSO) x erasure_target_deploy summaries
(target ΔbAcc, subject decode after). V2 panel: results/method_deepen/v2_stage2/v2_stage2_summary.json.
Outputs -> results/source_rich/phase0_meta/{mechanistic_meta_rows.csv, summary.json, report.md, figs/*.png}
and notes/SOURCE_RICH_PHASE0_META.md.
  python -m tos_cmi.eeg.report_mechanistic_meta
"""
from __future__ import annotations
import csv
import glob
import json
import os
import re
import numpy as np

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
TRACKB = "tos_cmi/results/method_deepen/trackB"
V2 = "tos_cmi/results/method_deepen/v2_stage2/v2_stage2_summary.json"
OUT = "tos_cmi/results/source_rich/phase0_meta"
NOTE = "tos_cmi/notes/SOURCE_RICH_PHASE0_META.md"
METHODS = ["LEACE", "RLACE", "TOS_VD", "INLP", "random_k"]


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y)[0])
    except Exception:
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return _pearson(rx, ry)


def _cluster_boot_corr(x, y, clusters, B=2000):
    """Cluster (dataset) bootstrap 95% CI of Pearson r. Resample clusters with replacement."""
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    by = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    uniq = sorted(by); rng = np.random.default_rng(0); rs = []
    if len(uniq) < 2:
        return float("nan"), float("nan")
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[c] for c in pick])
        r = _pearson(x[idx], y[idx])
        if r == r:
            rs.append(r)
    if not rs:
        return float("nan"), float("nan")
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _load_trackB():
    for f in ["%s/trackB_summary_full.json" % TRACKB, "%s/trackB_summary.json" % TRACKB]:
        if os.path.exists(f):
            return json.load(open(f))["summary"]
    return {}


def _load_deploy(ds):
    for f in ["%s/erasure_target_deploy/%s/erasure_target_deploy_summary.json" % (RESULTS, ds),
              "%s/erasure_target_deploy/erasure_target_deploy_summary.json" % RESULTS]:
        if os.path.exists(f):
            return json.load(open(f))["summary"]
    return {}


def _load_dsubj(ds, bb):
    """Δsubject decode (linear + MLP) per method = subj_full - subj_method, from erasure_report.json."""
    f = "%s/%s_%s_LOSO/erasure_report.json" % (RESULTS, ds, bb)
    if not os.path.exists(f):
        return {}
    agg = json.load(open(f)).get("aggregate", {})
    out = {}
    for m in ["LEACE", "RLACE", "TOS_VD", "INLP"]:
        lf, mf = agg.get("subj_full_lin"), agg.get("subj_full_mlp")
        lm, mm = agg.get("subj_%s_lin" % m), agg.get("subj_%s_mlp" % m)
        out[m] = (None if lf is None or lm is None else lf - lm,
                  None if mf is None or mm is None else mf - mm)
    return out


def _real_rows():
    tb = _load_trackB(); rows = []
    for key, g in tb.items():
        ds, bb, m = g["dataset"], g["backbone"], g["method"]
        dep = _load_deploy(ds).get("%s|%s" % (bb, m))
        ds_dsubj = _load_dsubj(ds, bb).get(m, (None, None))
        if not dep:
            continue
        rows.append(dict(dataset=ds, backbone=bb, method=m, gate_action=g.get("action", "?"),
                         task_drop_ucb=g.get("task_drop_ucb", float("nan")),
                         benefit_lcb=g.get("benefit_lcb", float("nan")),
                         benefit_mean=g.get("benefit_mean", float("nan")),
                         domain_gain=g.get("domain_gain", float("nan")),
                         dsubj_lin=ds_dsubj[0] if ds_dsubj[0] is not None else float("nan"),
                         dsubj_mlp=ds_dsubj[1] if ds_dsubj[1] is not None else float("nan"),
                         subj_after=dep.get("subj_dec_after_mean", float("nan")),
                         target_dbacc=dep.get("dtgt_bacc", float("nan")),
                         target_lo=dep.get("dtgt_bacc_lo", float("nan")),
                         target_hi=dep.get("dtgt_bacc_hi", float("nan"))))
    return rows


def _fig(x, y, xlabel, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return "(matplotlib unavailable: %r)" % e
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=26, alpha=0.75)
    ax.axhline(0, ls="--", c="k", lw=0.6); ax.axvline(0, ls="--", c="k", lw=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel("actual target ΔbAcc")
    r = _pearson(x, y); ax.set_title("%s\nPearson r=%+.3f" % (xlabel, r))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main():
    os.makedirs("%s/figs" % OUT, exist_ok=True)
    rows = _real_rows()
    if not rows:
        open(NOTE, "w").write("# Phase 0 --- NO real-EEG cells joined (aggregates missing on this branch)\n")
        print("META_NO_DATA"); return
    with open("%s/mechanistic_meta_rows.csv" % OUT, "w", newline="") as fh:
        cols = ["dataset", "backbone", "method", "gate_action", "task_drop_ucb", "benefit_lcb", "benefit_mean",
                "domain_gain", "dsubj_lin", "dsubj_mlp", "subj_after", "target_dbacc", "target_lo", "target_hi"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    tgt = [r["target_dbacc"] for r in rows]; ds_cl = [r["dataset"] for r in rows]
    # 1-7,+task drop = correlations of each source-only signal with actual target ΔbAcc
    sigs = [("1 Δsubject LINEAR decode vs target ΔbAcc", "dsubj_lin"),
            ("2 Δsubject MLP decode vs target ΔbAcc", "dsubj_mlp"),
            ("3 domain-gain vs target ΔbAcc", "domain_gain"),
            ("4 source task-drop UCB vs target ΔbAcc", "task_drop_ucb"),
            ("5 source-LOSO predicted ΔbAcc (mean) vs target ΔbAcc", "benefit_mean"),
            ("7 source-LOSO benefit LCB vs target ΔbAcc", "benefit_lcb")]
    corr = {}
    L = ["# Fork 2 Phase 0 --- mechanistic meta-analysis (REAL EEG primary; E0 subject environment)\n",
         "Joined %d real-EEG cells (Track B source-only gate x erasure_report x deployment). Target = audited "
         "outcome only. Datasets: %s ; backbones: %s.\n"
         % (len(rows), sorted(set(ds_cl)), sorted(set(r["backbone"] for r in rows))),
         "## Source-only signal vs actual target ΔbAcc (Pearson / Spearman / dataset-cluster 95%% CI)"]
    for name, kk in sigs:
        x = [r[kk] for r in rows]; p = _pearson(x, tgt); s = _spearman(x, tgt)
        lo, hi = _cluster_boot_corr(x, tgt, ds_cl)
        corr[kk] = {"pearson": p, "spearman": s, "ci": [lo, hi]}
        L.append("- %-52s : r %+.3f | rho %+.3f | 95%% CI [%+.3f,%+.3f]" % (name, p, s, lo, hi))
    # 6 sign agreement (source-LOSO predicted vs target)
    sa = [np.sign(r["benefit_mean"]) == np.sign(r["target_dbacc"]) for r in rows
          if r["benefit_mean"] == r["benefit_mean"] and r["target_dbacc"] == r["target_dbacc"]]
    sign_agree = float(np.mean(sa)) if sa else float("nan")
    L += ["", "## 6 source-LOSO sign agreement with target: %d/%d (%.0f%%)" % (sum(sa), len(sa), 100 * sign_agree)]
    # 8 random-k comparison
    rk = [r for r in rows if r["method"] == "random_k"]; pr = [r for r in rows if r["method"] != "random_k"]
    L += ["", "## 8 random-k control: mean target ΔbAcc %+.3f (vs principled %+.3f)"
          % (float(np.nanmean([r["target_dbacc"] for r in rk])) if rk else float("nan"),
             float(np.nanmean([r["target_dbacc"] for r in pr])))]
    # 9,10,11 breakdowns
    def _bd(keyfn, title):
        L2 = ["", "## %s" % title]
        for g in sorted(set(keyfn(r) for r in rows)):
            sub = [r for r in rows if keyfn(r) == g]
            L2.append("- %-16s n=%2d : source-LOSO-vs-target r %+.3f ; target ΔbAcc mean %+.3f [%+.3f,%+.3f]"
                      % (g, len(sub), _pearson([r["benefit_mean"] for r in sub], [r["target_dbacc"] for r in sub]),
                         float(np.nanmean([r["target_dbacc"] for r in sub])),
                         min(r["target_dbacc"] for r in sub), max(r["target_dbacc"] for r in sub)))
        return L2
    L += _bd(lambda r: r["dataset"], "9 per-dataset breakdown")
    L += _bd(lambda r: r["backbone"], "10 per-backbone breakdown")
    L += _bd(lambda r: r["method"], "11 per-method breakdown")
    # 12 gate-action vs actual target outcome
    L += ["", "## 12 gate-action vs actual target outcome"]
    for a in ["ACCEPT", "REJECT", "ABSTAIN"]:
        sub = [r for r in rows if r["gate_action"] == a]
        if not sub:
            L.append("- %-8s : (none)"); continue
        harmful = sum(1 for r in sub if r["target_hi"] < -0.01)
        benef = sum(1 for r in sub if r["target_lo"] > 0.01)
        L.append("- %-8s n=%2d : target ΔbAcc mean %+.3f ; harmful cells %d ; beneficial cells %d"
                 % (a, len(sub), float(np.nanmean([r["target_dbacc"] for r in sub])), harmful, benef))
    # figures
    f1 = _fig([r["dsubj_lin"] for r in rows], tgt, "Δsubject linear decode (erasure)", "%s/figs/subject_reduction_vs_target_delta.png" % OUT)
    f2 = _fig([r["benefit_mean"] for r in rows], tgt, "source-LOSO predicted ΔbAcc (E0 subject env)", "%s/figs/source_loso_vs_target_delta.png" % OUT)
    f3 = _fig([r["task_drop_ucb"] for r in rows], tgt, "source task-drop UCB", "%s/figs/task_drop_vs_target_delta.png" % OUT)
    # V2 semi-synthetic SEPARATE panel (never mixed)
    L += ["", "## V2 semi-synthetic panel (SEPARATE; mechanism control, NOT mixed with real EEG)"]
    if os.path.exists(V2):
        v = json.load(open(V2))["summary"]
        dep = [c for c in v.values() if c["deployable"]]
        rv = _pearson([c["benefit_lcb"] for c in dep], [c["dtgt_bacc"] for c in dep])
        L.append("- V2 stage2 deployable cells: source-LOSO benefit LCB vs target ΔbAcc Pearson r %+.3f (n=%d) "
                 "-- semi-synthetic; injected nuisance; kept apart from the real-EEG conclusion." % (rv, len(dep)))
    else:
        L.append("- (V2 stage2 summary not found)")
    # reading
    L += ["", "## Reading (E0 baseline)",
          "- If the source-LOSO<->target correlation (#5/#7) is ~0 / CI spans 0 under E0 subject environments, "
          "the E0 baseline does NOT make benefit source-visible -> E1-E5 must materially improve source-target "
          "alignment for Fork 2 to have a positive route.",
          "- If leakage reduction (#1/#2) is uncorrelated with target ΔbAcc, this supports 'leakage removal is "
          "not a benefit certificate' (do NOT overstate as 'leakage can never matter').",
          "- Any target-positive subgroup is a CANDIDATE only -> must be reproduced under the pre-registered "
          "dev/confirm split with random-k / task-drop / leakage audits before any claim."]
    json.dump({"n_cells": len(rows), "datasets": sorted(set(ds_cl)),
               "correlations": corr, "sign_agreement": sign_agree,
               "figs": [f1, f2, f3]}, open("%s/mechanistic_meta_summary.json" % OUT, "w"), indent=1)
    txt = "\n".join(L) + "\n"
    open("%s/mechanistic_meta_report.md" % OUT, "w").write(txt)
    open(NOTE, "w").write(txt)
    print(txt); print("META_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
