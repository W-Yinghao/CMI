"""CIGL R1 — evidence hardening: exact permutation p-values, Benjamini-Hochberg FDR, and a hierarchical
(dataset -> fold -> seed) cluster bootstrap. Pure numpy; used to make "leakage exists and CIGL reduces it"
statistically defensible across the multi-fold / multi-seed audit (the phase-3A audit currently uses n_perm
20-50, 3 seeds, no FDR). Builds ON the existing within-label permutation null + probe in
cmi/eval/graph_leakage.py — this module only does the statistics on the collected observed/null values.
"""
from __future__ import annotations
import numpy as np


def exact_permutation_pvalue(observed, null_samples, tail="greater"):
    """Exact permutation p-value (Phipson & Smyth 2010): p = (1 + #{null tail-beyond observed}) / (B + 1).
    Never returns 0 (the observed statistic is itself one realization under H0). tail: 'greater' (leakage =
    higher KL, the default), 'less', or 'two_sided' (on |.|)."""
    null = np.asarray(null_samples, dtype=float)
    B = null.size
    if B == 0:
        raise ValueError("null_samples is empty")
    if tail == "greater":
        beyond = int(np.sum(null >= observed))
    elif tail == "less":
        beyond = int(np.sum(null <= observed))
    elif tail == "two_sided":
        beyond = int(np.sum(np.abs(null) >= abs(observed)))
    else:
        raise ValueError(f"tail must be greater|less|two_sided, got {tail}")
    return (1 + beyond) / (B + 1)


def benjamini_hochberg(pvalues, alpha=0.05):
    """Benjamini-Hochberg step-up FDR control. Returns rejected mask (in input order), BH-adjusted p-values,
    the critical p, and the count rejected. Controls FDR at `alpha` across the family of (fold/seed) tests."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return {"rejected": np.zeros(0, bool), "adjusted_p": np.zeros(0), "n_rejected": 0,
                "critical_p": 0.0, "alpha": alpha}
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if below.any():
        kmax = int(np.max(np.nonzero(below)[0]))       # largest rank meeting the BH threshold
        critical_p = float(ranked[kmax])
        rejected_ranked = ranked <= critical_p
    else:
        critical_p = 0.0
        rejected_ranked = np.zeros(m, bool)
    rejected = np.zeros(m, bool)
    rejected[order] = rejected_ranked
    # BH-adjusted p-values (monotone step-up)
    adj_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(adj_ranked, 0.0, 1.0)
    return {"rejected": rejected, "adjusted_p": adjusted, "n_rejected": int(rejected.sum()),
            "critical_p": critical_p, "alpha": alpha}


def hierarchical_bootstrap(records, value_key="value", levels=("dataset", "fold", "seed"),
                           statistic=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Cluster (hierarchical) bootstrap CI that respects the dataset -> fold -> seed nesting: resample the
    top-level clusters with replacement, then within each, resample the next level, ... down to the leaf
    values. Accounts for the non-independence of seeds within a fold and folds within a dataset (a flat
    bootstrap would understate the CI). `records` = list of dicts with the level keys + value_key.
    Returns point estimate + [alpha/2, 1-alpha/2] percentile interval."""
    records = list(records)
    if not records:
        raise ValueError("records is empty")
    rng = np.random.default_rng(seed)
    levels = tuple(levels)

    def _resample(recs, lvl):
        if lvl >= len(levels):
            vals = np.asarray([r[value_key] for r in recs], dtype=float)
            idx = rng.integers(0, vals.size, vals.size)
            return vals[idx].tolist()
        groups = {}
        for r in recs:
            groups.setdefault(r[levels[lvl]], []).append(r)
        keys = list(groups)
        chosen = rng.integers(0, len(keys), len(keys))
        out = []
        for c in chosen:
            out.extend(_resample(groups[keys[c]], lvl + 1))
        return out

    boots = np.empty(n_boot)
    for b in range(n_boot):
        leaves = _resample(records, 0)
        boots[b] = statistic(leaves)
    point = float(statistic([r[value_key] for r in records]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot, "levels": list(levels),
            "n_records": len(records)}


def harden_leakage_table(rows, alpha=0.05, boot_levels=("dataset", "fold", "seed"), n_boot=2000, seed=0):
    """Given per-(dataset,fold,seed) audit rows each carrying an `observed` KL and its permutation `null`
    (list), plus a `value` = the reduction (or KL) to bootstrap, produce the hardened summary: exact per-row
    p, BH-FDR across the family, fraction rejected, and a hierarchical-bootstrap CI on `value`.
    Each row: {dataset, fold, seed, observed, null:[...], value}."""
    rows = list(rows)
    pvals = [exact_permutation_pvalue(r["observed"], r["null"]) for r in rows]
    bh = benjamini_hochberg(pvals, alpha=alpha)
    ci = hierarchical_bootstrap(rows, value_key="value", levels=boot_levels, n_boot=n_boot, seed=seed) \
        if all("value" in r for r in rows) else None
    return {"n_tests": len(rows), "exact_p": pvals, "bh": bh,
            "frac_cleared_fdr": (bh["n_rejected"] / len(rows)) if rows else 0.0,
            "bootstrap_ci": ci}
