"""CMI-Trace P0.2 — unified objective->effect audit + fold/subject-cluster inference.

Artifact-driven (like R3): consumes the per-fold `.audit.npz` sidecar written by the objective-comparison
runner + the metrics JSON (leakage). NEVER retrains. For every trained model, on FROZEN source
representations, this computes the same set of columns so ERM / CORAL / C-CORAL / IRMv1 / V-REx /
encoder-CMI / conditional-adversarial receive an identical objective->effect audit:

  1-2. target/source balanced accuracy               (from metrics JSON / audit npz replay)
  3.   marginal graph moment gap                      (moment_gaps)
  4.   class-conditional graph moment gap             (moment_gaps)
  5-6. graph/node encoder-CMI posterior-KL + null     (from metrics JSON leakage block)
  7.   per-domain risk variance                       (per_domain_risk_variance)
  8.   IRMv1 diagnostic penalty (post hoc)            (irmv1_diagnostic)
  9.   feature norm / top singular value / eff. rank  (feature_geometry)
  10.  exact-head reliance R_rel at primary k=2       (cmi.eval.leakage_removal.evaluate_reliance)
  11.  same-rank random-subspace reliance control     (evaluate_reliance conditioning=random_subspace)

Main uncertainty = PAIRED fold/subject-cluster bootstrap (seeds within a fold travel together). Seed SD is
descriptive only. Pure numpy on load; torch not required.
"""
from __future__ import annotations
import numpy as np

from cmi.eval.audit_npz import load_audit_npz, replay_head, head_replay_ok
from cmi.eval.leakage_removal import evaluate_reliance


# --------------------------------------------------------------------- moment gaps (diagnostics)
def _cov(x):
    xm = x - x.mean(0, keepdims=True)
    return (xm.T @ xm) / max(1, x.shape[0] - 1)


def _pair_moment(a, b):
    md = float(np.mean((a.mean(0) - b.mean(0)) ** 2))
    cd = float(np.mean((_cov(a) - _cov(b)) ** 2))
    return md + cd


def marginal_moment_gap(z, d, n_dom=None, min_n=4):
    """Mean over unordered source-domain pairs of the CORAL discrepancy (squared mean gap + squared cov gap)
    of the marginal representation. 0 iff every domain shares mean+cov; positive otherwise."""
    z = np.asarray(z, float); d = np.asarray(d)
    doms = [i for i in np.unique(d) if (d == i).sum() >= min_n]
    pens = [_pair_moment(z[d == doms[a]], z[d == doms[b]])
            for a in range(len(doms)) for b in range(a + 1, len(doms))]
    return float(np.mean(pens)) if pens else 0.0


def class_conditional_moment_gap(z, y, d, min_n=4):
    """Mean over classes (present in >=2 source domains with adequate support) of the WITHIN-class marginal
    moment gap. Ignores a pure label-prior shift; detects a within-class domain moment gap. Also returns the
    support diagnostics (skipped under-supported cells)."""
    z = np.asarray(z, float); y = np.asarray(y); d = np.asarray(d)
    per_class, skipped, support = [], [], {}
    for c in np.unique(y):
        mc = y == c
        doms = [i for i in np.unique(d[mc]) if (mc & (d == i)).sum() >= min_n]
        for i in np.unique(d[mc]):
            n = int((mc & (d == i)).sum())
            if n < min_n:
                skipped.append((int(c), int(i), n))
        support[int(c)] = [int(x) for x in doms]
        if len(doms) < 2:
            continue
        pens = [_pair_moment(z[mc & (d == doms[a])], z[mc & (d == doms[b])])
                for a in range(len(doms)) for b in range(a + 1, len(doms))]
        per_class.append(float(np.mean(pens)))
    gap = float(np.mean(per_class)) if per_class else 0.0
    return gap, {"support": support, "skipped_cells": skipped, "n_qualifying_classes": len(per_class)}


# --------------------------------------------------------------------- risk / IRM diagnostics
def _softmax(logits):
    m = logits - logits.max(1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(1, keepdims=True)


def _ce(logits, y):
    p = _softmax(np.asarray(logits, float))
    n = len(y)
    return float(-np.mean(np.log(np.clip(p[np.arange(n), np.asarray(y)], 1e-12, 1.0))))


def per_domain_risk_variance(logits, y, d):
    """Variance across source domains of the per-domain cross-entropy risk (the V-REx quantity, post hoc)."""
    logits = np.asarray(logits, float); y = np.asarray(y); d = np.asarray(d)
    risks = [_ce(logits[d == i], y[d == i]) for i in np.unique(d) if (d == i).sum() > 0]
    return float(np.var(risks)) if len(risks) >= 2 else 0.0


def irmv1_diagnostic(logits, y, d):
    """IRMv1 penalty evaluated post hoc: mean over source domains of (d/ds CE(s*logits,y)|_{s=1})^2.
    Closed form g_i = <softmax(logits_i), logits_i> - logits_{i,y_i}; penalty_domain = (mean_i g_i)^2."""
    logits = np.asarray(logits, float); y = np.asarray(y); d = np.asarray(d)
    p = _softmax(logits)
    n = len(y)
    g = (p * logits).sum(1) - logits[np.arange(n), y]        # per-sample grad of CE wrt dummy scale at s=1
    pens = [float(np.mean(g[d == i]) ** 2) for i in np.unique(d) if (d == i).sum() > 0]
    return float(np.mean(pens)) if pens else 0.0


# --------------------------------------------------------------------- geometry
def feature_geometry(z):
    """Feature norm (mean row L2), top singular value, and effective (spectral) rank of the CENTERED matrix.
    Effective rank = exp(H) with H the Shannon entropy of the normalized singular-value spectrum
    (Roy & Vetterli 2007)."""
    z = np.asarray(z, float)
    row_norm = float(np.mean(np.linalg.norm(z, axis=1)))
    zc = z - z.mean(0, keepdims=True)
    s = np.linalg.svd(zc, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return {"feature_norm": row_norm, "top_singular_value": 0.0, "effective_rank": 0.0}
    p = s / s.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    return {"feature_norm": row_norm, "top_singular_value": float(s[0]), "effective_rank": eff_rank}


# --------------------------------------------------------------------- one full objective->effect row
def objective_effect_row(npz_path, leakage=None, primary_k=2, seed=0, representation="graph_z"):
    """Assemble one objective->effect row from a `.audit.npz` + optional leakage dict (from the metrics JSON).
    Source = rows whose domain != the held-out target domain (the npz stores target as a distinct domain id).
    Returns a flat dict (row); reliance uses the verified head-replay when available."""
    data = load_audit_npz(npz_path)
    y = np.asarray(data["y"]); d = np.asarray(data["d"])
    z = np.asarray(data[representation], float)
    logits = np.asarray(data["model_logits"], float)
    # target domain = the id present ONLY in target_indices (distinct id appended eval-only); else max id
    if "target_indices" in data and len(np.asarray(data["target_indices"])):
        tgt_dom = int(np.unique(d[np.asarray(data["target_indices"]).ravel()])[0])
    else:
        tgt_dom = int(d.max())
    src = d != tgt_dom
    zs, ys, ds, ls = z[src], y[src], d[src], logits[src]
    cc_gap, cc_support = class_conditional_moment_gap(zs, ys, ds)
    geom = feature_geometry(zs)
    row = {
        "dataset": data.get("dataset", ""), "method": data.get("method", ""),
        "fold": int(np.asarray(data.get("fold", -1))), "seed": int(np.asarray(data.get("seed", seed))),
        "target_subject": data.get("target_subject", ""),
        "marginal_moment_gap": marginal_moment_gap(zs, ds),
        "class_conditional_moment_gap": cc_gap,
        "cc_support_skipped_cells": len(cc_support["skipped_cells"]),
        "per_domain_risk_variance": per_domain_risk_variance(ls, ys, ds),
        "irmv1_diagnostic": irmv1_diagnostic(ls, ys, ds),
        "feature_norm": geom["feature_norm"], "top_singular_value": geom["top_singular_value"],
        "effective_rank": geom["effective_rank"],
        "head_replay_available": bool(head_replay_ok(data)),
    }
    # reliance at primary k=2 (label_conditional) + same-rank random-subspace control
    rel = evaluate_reliance(data, tgt_dom, k=primary_k, conditioning="label_conditional",
                            seed=seed, representation=representation)
    rnd = evaluate_reliance(data, tgt_dom, k=primary_k, conditioning="random_subspace",
                            seed=seed, representation=representation)
    row["R_rel_k2"] = rel["task_drop"]
    row["R_rel_k2_random_control"] = rnd["task_drop"]
    row["reliance_firewall_passed"] = bool(rel["firewall_passed"])
    row["reliance_removal_mode"] = rel["removal_mode"]
    row["source_task_bacc"] = rel["source_task_bacc_before"]
    row["target_task_bacc"] = rel["target_task_bacc_before"]
    if leakage is not None:
        for k in ("graph_kl", "graph_null", "graph_perm_p", "node_kl", "node_null", "node_perm_p"):
            if k in leakage:
                row[k] = leakage[k]
    return row


# --------------------------------------------------------------------- cluster inference (P0.2)
def cluster_bootstrap_ci(cluster_values, n_boot=10000, ci=0.95, seed=12345):
    """Paired fold/subject-cluster bootstrap CI of the MEAN over clusters. `cluster_values` = one scalar per
    outer fold/subject (seeds already averaged within the cluster). Resamples clusters with replacement.
    Returns (mean, lo, hi, n_clusters)."""
    v = np.asarray([x for x in cluster_values if x is not None and np.isfinite(x)], float)
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return (float(v[0]), float(v[0]), float(v[0]), 1)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, n, size=(n_boot, n))].mean(1)
    a = (1 - ci) / 2
    return (float(v.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)), n)


def _cluster_key(row):
    return (row["dataset"], int(row["fold"]))


def cluster_means(rows, metric):
    """Average `metric` within each (dataset, fold) cluster (seeds travel together)."""
    by = {}
    for r in rows:
        if metric in r and r[metric] is not None and np.isfinite(r[metric]):
            by.setdefault(_cluster_key(r), []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in by.items()}


def summarize_metric(rows, metric, n_boot=10000, ci=0.95):
    """Raw mean + cluster-bootstrap CI + descriptive seed SD for a single metric over a set of rows."""
    cm = cluster_means(rows, metric)
    mean, lo, hi, n_clusters = cluster_bootstrap_ci(list(cm.values()), n_boot=n_boot, ci=ci)
    vals = [float(r[metric]) for r in rows if metric in r and r[metric] is not None and np.isfinite(r[metric])]
    return {"metric": metric, "raw_mean": float(np.mean(vals)) if vals else float("nan"),
            "cluster_ci_lo": lo, "cluster_ci_hi": hi, "n_clusters": n_clusters,
            "n_folds": n_clusters, "n_rows": len(vals),
            "seed_sd_descriptive": float(np.std(vals)) if vals else float("nan")}


def paired_delta_vs_baseline(rows, metric, baseline_method="erm", n_boot=10000, ci=0.95):
    """Paired per-(dataset,fold) delta of `metric` vs a baseline method, with cluster-bootstrap CI. Seeds are
    averaged within each cluster before differencing so the pairing is at the fold/subject level."""
    base = {m: v for m, v in _by_method_cluster(rows, baseline_method, metric).items()}
    out = {}
    for method in sorted({r["method"] for r in rows}):
        if method == baseline_method:
            continue
        mm = _by_method_cluster(rows, method, metric)
        deltas = [mm[k] - base[k] for k in mm if k in base]
        mean, lo, hi, n = cluster_bootstrap_ci(deltas, n_boot=n_boot, ci=ci)
        out[method] = {"metric": metric, "vs": baseline_method, "delta_mean": mean,
                       "cluster_ci_lo": lo, "cluster_ci_hi": hi, "n_clusters": n}
    return out


def _by_method_cluster(rows, method, metric):
    by = {}
    for r in rows:
        if r["method"] == method and metric in r and r[metric] is not None and np.isfinite(r[metric]):
            by.setdefault(_cluster_key(r), []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in by.items()}
