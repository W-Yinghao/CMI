"""V2 analysis (review V2_FROZEN §7): SEPARATE A and B verdicts -- never one headline mean.

A (out-of-support audit), unit = target subject nested in ordered source->target pair:
  unsupported-route adaptation count = k/N (metadata_action != identity), exact prediction-
  equivalence with identity, binomial upper 95% CI on the adaptation rate; always_pooled /
  always_canonical_CC / EA harm + mean/worst-quartile delta.
B (supported utility), unit = subject (dataset stratified):
  mean paired delta(metadata_only - identity) > 0, target-subject harm rate <= 0.20, worst-quartile
  delta, dataset-stratified effects, coverage (~1). current_joint is diagnostic-only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

METHODS = ["identity", "always_pooled", "always_canonical_CC", "metadata_only",
           "euclidean_alignment", "current_joint"]


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _wq(deltas):
    if not deltas:
        return float("nan")
    d = np.sort(np.asarray(deltas)); k = max(1, len(d) // 4)
    return float(d[:k].mean())


def _clopper_pearson_upper(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return float("nan")
    return float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0


def _boot_ci(vals, n_boot=10000, seed=0):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _cluster_boot_ci(cluster_vals, n_boot=10000, seed=0):
    """Subject-CLUSTERED bootstrap: cluster_vals = {cluster_key: [values]}. Resample CLUSTERS with
    replacement (so repeated sessions of one subject move together), pool their values, take the mean.
    Required because BNCI2014_004 contributes 3 cross-session units per subject (non-independent)."""
    keys = list(cluster_vals)
    if len(keys) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arrs = {k: np.asarray(v, float) for k, v in cluster_vals.items()}
    bs = []
    for _ in range(n_boot):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        pooled = np.concatenate([arrs[k] for k in pick])
        bs.append(pooled.mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def method_stats(rows, method):
    d = [r["delta"] for r in rows if r["method"] == method]
    harm = [r["harm"] for r in rows if r["method"] == method]
    return dict(n=len(d), mean_delta=float(np.mean(d)) if d else float("nan"),
                harm_rate=float(np.mean(harm)) if harm else float("nan"), worst_quartile=_wq(d))


def analyze_A(rows, label="A"):
    rows = [r for r in rows if r["mode"] == label]
    out = {"mode": label, "n_rows": len(rows)}
    md = [r for r in rows if r["method"] == "metadata_only"]
    n = len(md)
    adapt = [r for r in md if r["metadata_action"] != "identity"]
    equiv = [r for r in md if r.get("pred_equiv_identity")]
    out["metadata"] = dict(N_units=n, unsupported_route_adaptations=len(adapt),
                           adaptation_rate=len(adapt) / max(1, n),
                           adaptation_rate_upper95=_clopper_pearson_upper(len(adapt), n),
                           exact_identity_equivalence=len(equiv) / max(1, n),
                           geometry_all_unsupported=all(r["geometry"] == "UNSUPPORTED" for r in md))
    out["methods"] = {m: method_stats(rows, m) for m in METHODS}
    # per-pair breakdown (pair is the top unit level)
    pairs = sorted(set(r["pair"] for r in rows))
    out["per_pair"] = {}
    for p in pairs:
        pr = [r for r in rows if r["pair"] == p]
        out["per_pair"][p] = {m: method_stats(pr, m) for m in ("metadata_only", "always_pooled",
                                                               "always_canonical_CC", "euclidean_alignment")}
    return out


def analyze_B(rows):
    rows = [r for r in rows if r["mode"] == "B"]
    out = {"mode": "B", "n_rows": len(rows)}
    out["methods"] = {}
    for m in METHODS:
        st = method_stats(rows, m)
        clus = defaultdict(list)
        for r in rows:
            if r["method"] == m:
                clus[(r["source"], r["subject"])].append(r["delta"])   # cluster by dataset+subject
        lo, hi = _cluster_boot_ci(clus)
        st["delta_ci95_subjclustered"] = [lo, hi]
        st["n_subject_clusters"] = len(clus)
        out["methods"][m] = st
    # metadata_only coverage (fraction that actually adapt -- DIAG -> pooled so ~1)
    md = [r for r in rows if r["method"] == "metadata_only"]
    out["metadata_coverage"] = float(np.mean([r["metadata_action"] != "identity" for r in md])) if md else float("nan")
    out["metadata_actions"] = sorted(set(r["metadata_action"] for r in md))
    # dataset-stratified metadata_only and the always-pooled twin
    out["per_dataset"] = {}
    for ds in sorted(set(r["source"] for r in rows)):
        dr = [r for r in rows if r["source"] == ds]
        out["per_dataset"][ds] = {m: method_stats(dr, m) for m in ("metadata_only", "always_pooled",
                                                                   "euclidean_alignment", "current_joint")}
    # B verdict (subject-clustered CI)
    mo = out["methods"]["metadata_only"]
    ci = mo["delta_ci95_subjclustered"]
    out["verdict"] = dict(
        mean_paired_delta_positive=bool(mo["mean_delta"] > 0),
        delta_ci_excludes_0=bool(ci[0] > 0),
        harm_rate_within_0_20=bool(mo["harm_rate"] <= 0.20),
        coverage_near_1=bool(out["metadata_coverage"] >= 0.95))
    return out


def _fmt(m):
    return (f"n={m['n']:3d} mean_d={m['mean_delta']:+.3f} harm={m['harm_rate']:.2f} "
            f"wq={m['worst_quartile']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, nargs="+")
    ap.add_argument("--out", default="results/h2cmi/v2.report.json")
    args = ap.parse_args()
    rows = []
    for p in args.inp:
        rows += _rows(p)
    rep = {"A": analyze_A(rows, "A"), "B": analyze_B(rows)}
    if any(r["mode"] == "A_severe" for r in rows):
        rep["A_severe"] = analyze_A(rows, "A_severe")
    json.dump(rep, open(args.out, "w"), indent=2)

    A = rep["A"]; mdA = A["metadata"]
    print("=== A: out-of-support abstention audit (unit = target subject in pair) ===")
    print(f"  metadata: N={mdA['N_units']}  unsupported-route adaptations={mdA['unsupported_route_adaptations']}/"
          f"{mdA['N_units']}  rate={mdA['adaptation_rate']:.3f} (upper95={mdA['adaptation_rate_upper95']:.3f})  "
          f"exact-identity-equiv={mdA['exact_identity_equivalence']:.3f}  all_UNSUPPORTED={mdA['geometry_all_unsupported']}")
    for m in ("metadata_only", "always_pooled", "always_canonical_CC", "euclidean_alignment"):
        print(f"    {m:22s} {_fmt(A['methods'][m])}")
    print("=== B: supported-regime utility (unit = subject, dataset-stratified) ===")
    Bd = rep["B"]
    for m in METHODS:
        mm = Bd["methods"][m]
        ci = mm["delta_ci95_subjclustered"]
        print(f"    {m:22s} {_fmt(mm)} clustCI95=[{ci[0]:+.3f},{ci[1]:+.3f}] (n_clust={mm['n_subject_clusters']})")
    print(f"  metadata coverage={Bd['metadata_coverage']:.2f} actions={Bd['metadata_actions']}")
    for ds, dd in Bd["per_dataset"].items():
        print(f"    [{ds}] metadata_only {_fmt(dd['metadata_only'])}")
    print(f"  B verdict: {Bd['verdict']}")
    if "A_severe" in rep:
        s = rep["A_severe"]["metadata"]
        print(f"=== A-severe (descriptive): adaptations={s['unsupported_route_adaptations']}/{s['N_units']} "
              f"metadata_only {_fmt(rep['A_severe']['methods']['metadata_only'])} ===")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
