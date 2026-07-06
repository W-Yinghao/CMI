"""Fork 2 Phase 0 --- mechanistic meta-analysis (NO new training). Joins the committed real-EEG aggregates
(Track B source-only signals + frozen-erasure target-deployment outcomes) and asks whether ANY source-only
signal predicts the target outcome under the CURRENT subject-based environments. This scopes Fork 2: if the
source-LOSO<->target correlation is already flat, source environments must be redefined (E1-E5); if some
dataset/backbone has weak signal, analyze those first. Target enters only as the audited outcome here.

Reads:
  results/method_deepen/trackB/trackB_summary{_full}.json     (task_drop_ucb, benefit_lcb, domain_gain per ds|bb|method)
  results/tos_cmi_eeg_frozen/erasure_target_deploy[/<ds>]/erasure_target_deploy_summary.json  (target dbAcc, subj_dec_after)
Writes notes/SOURCE_RICH_PREFLIGHT.md. Submit via SLURM (CPU). Does NOT run until PM go.
  python -m tos_cmi.eeg.report_mechanistic_meta
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
TRACKB = "tos_cmi/results/method_deepen/trackB"
OUTNOTE = "tos_cmi/notes/SOURCE_RICH_PREFLIGHT.md"


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan"), float("nan"), len(x)
    try:
        from scipy.stats import pearsonr, spearmanr
        return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0]), len(x)
    except Exception:
        return float(np.corrcoef(x, y)[0, 1]), float("nan"), len(x)


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


def main():
    tb = _load_trackB()
    rows = []
    for key, g in tb.items():
        ds, bb, m = g["dataset"], g["backbone"], g["method"]
        dep = _load_deploy(ds).get("%s|%s" % (bb, m))
        if not dep:
            continue
        rows.append(dict(ds=ds, bb=bb, method=m,
                         task_drop=g.get("task_drop_ucb", float("nan")),
                         benefit=g.get("benefit_lcb", float("nan")),
                         domain_gain=g.get("domain_gain", float("nan")),
                         subj_after=dep.get("subj_dec_after_mean", float("nan")),
                         tgt=dep.get("dtgt_bacc", float("nan")), tgt_lo=dep.get("dtgt_bacc_lo", float("nan"))))
    L = ["# Fork 2 Phase 0 --- mechanistic meta-analysis (real EEG; source-only signal vs target outcome)\n",
         "Joined %d Track-B x deployment cells. Target = audited outcome only.\n" % len(rows)]
    if not rows:
        L.append("**NO joined cells found** (Track B / deploy summaries missing on this branch); run Track B "
                 "full + erasure_target_deploy first, or point paths at the committed aggregates.")
        open(OUTNOTE, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("META_NO_DATA"); return
    tgt = [r["tgt"] for r in rows]
    tests = [("1 domain-gain (Δsubject decode) vs target ΔbAcc", [r["domain_gain"] for r in rows]),
             ("2 (subject decode AFTER erasure) vs target ΔbAcc", [r["subj_after"] for r in rows]),
             ("3 source task-drop UCB vs target ΔbAcc", [r["task_drop"] for r in rows]),
             ("4 source-LOSO benefit LCB vs target ΔbAcc (E0 subject env)", [r["benefit"] for r in rows])]
    L += ["## Correlations across cells (Pearson / Spearman / n)"]
    for name, x in tests:
        p, s, n = _corr(x, tgt)
        L.append("- %-56s : Pearson %+.3f | Spearman %+.3f | n=%d" % (name, p, s, n))
    # 5 per dataset/backbone
    L += ["", "## Per dataset x backbone (source-LOSO benefit vs target ΔbAcc)"]
    for ds in sorted(set(r["ds"] for r in rows)):
        for bb in sorted(set(r["bb"] for r in rows if r["ds"] == ds)):
            sub = [r for r in rows if r["ds"] == ds and r["bb"] == bb]
            p, s, n = _corr([r["benefit"] for r in sub], [r["tgt"] for r in sub])
            L.append("- %-11s %-7s : Pearson %+.3f (n=%d) ; target ΔbAcc range [%+.3f,%+.3f]"
                     % (ds, bb, p, n, min(r["tgt"] for r in sub), max(r["tgt"] for r in sub)))
    # 6 sign agreement (source-LOSO benefit vs target)
    sa = [np.sign(r["benefit"]) == np.sign(r["tgt"]) for r in rows if r["benefit"] == r["benefit"] and r["tgt"] == r["tgt"]]
    L += ["", "## Sign agreement source-LOSO-benefit vs target ΔbAcc: %d/%d (%.0f%%)"
          % (sum(sa), len(sa), 100 * np.mean(sa) if sa else float("nan"))]
    # 7 random-k comparison
    rk = [r for r in rows if r["method"] == "random_k"]
    if rk:
        L += ["", "## random-k cells (control): target ΔbAcc mean %+.3f (vs principled mean %+.3f)"
              % (float(np.nanmean([r["tgt"] for r in rk])),
                 float(np.nanmean([r["tgt"] for r in rows if r["method"] != "random_k"])))]
    L += ["", "## Reading",
          "- If test #4 (source-LOSO benefit vs target) is ~0 under E0 subject environments, Fork 2's premise is "
          "that a DIFFERENT environment definition (E1-E5) is needed to make the shift source-visible.",
          "- Datasets/backbones with the least-flat source-target signal are prioritized for Phase 3."]
    open(OUTNOTE, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("META_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
