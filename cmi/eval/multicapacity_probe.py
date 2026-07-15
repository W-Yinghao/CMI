"""CMI-Trace P1.3 — predeclared multi-capacity posterior-probe family for the conditional-subject-leakage
ruler, with a VALID familywise-max null.

Capacity family (predeclared): linear / mlp_small (the current PRIMARY comparable quantity) / mlp_large.
Optional non-neural sensitivity diagnostics (kNN/HSIC) are left to callers.

Statistical rule (the whole reason this module exists):
  * Each capacity is reported SEPARATELY (kl, excess over its own retrained null, perm p, convergence).
  * The PRIMARY quantity stays `mlp_small` (fixed) so cross-condition comparisons are apples-to-apples.
  * If a "maximum detectable leakage over capacities" is reported, the null MUST repeat the SAME
    max-over-capacities operation on EVERY permutation — otherwise the familywise p-value is invalid
    (anti-conservative). Here every capacity's retrained within-label permutation null uses the SAME seed,
    so permutation index j applies the SAME permuted D across capacities; the familywise null at index j is
    the max over capacities under that shared permutation. The familywise maximum is an UNDERFITTING GUARD,
    never the headline.

Trial/node rows never cross the probe split: the probe only ever indexes the caller-provided (ptrain, peval)
which are trial-disjoint (see cmi.eval.conditional_subject_leakage.three_way_support_split).
"""
from __future__ import annotations
import numpy as np

from cmi.eval.graph_leakage import (fit_conditional_domain_probe, _permutation_null, _perm_summary)

# predeclared capacity family; mlp_small == the historical/primary probe (one 64-wide hidden layer)
CAPACITY_FAMILY = {
    "linear":    {"probe_hidden": [],        "dropout": 0.0},
    "mlp_small": {"probe_hidden": [64],      "dropout": 0.0},   # PRIMARY comparable quantity
    "mlp_large": {"probe_hidden": [128, 128], "dropout": 0.1},
}
PRIMARY_CAPACITY = "mlp_small"


def multicapacity_cmi(features, y, d, n_cls, n_dom, ptrain_idx, peval_idx, *,
                      capacities=None, n_perm=50, seed=0, device="cpu", epochs=100):
    """Fit the whole capacity family on FLAT features with a FIXED (ptrain, peval) split; report each
    capacity separately + a VALID familywise-max leakage with its own max-over-capacities null.

    Returns {per_capacity: {name: {...}}, primary_capacity, primary_kl, familywise_max_kl,
             familywise_max_perm_p, familywise_note}. The familywise null is the ELEMENTWISE MAX over
             capacities of the per-capacity retrained nulls (same permutation seed across capacities)."""
    caps = capacities or CAPACITY_FAMILY
    y = np.asarray(y).astype(np.int64); d = np.asarray(d).astype(np.int64)
    ptrain_idx = np.asarray(ptrain_idx, np.int64); peval_idx = np.asarray(peval_idx, np.int64)

    per_cap, null_by_cap = {}, {}
    for name, arch in caps.items():
        def fit(d_arr, arch=arch):
            return fit_conditional_domain_probe(features, y, d_arr, n_cls, n_dom,
                                                train_idx=ptrain_idx, val_idx=peval_idx, epochs=epochs,
                                                seed=seed, device=device,
                                                probe_hidden=arch["probe_hidden"], dropout=arch["dropout"])
        obs = fit(d)
        # same seed across capacities -> permutation index k uses the SAME permuted D for every capacity
        nulls = _permutation_null(fit, y, d, n_perm, seed, permute_idx=ptrain_idx)
        ps = _perm_summary(obs["kl_mean"], nulls)
        null_by_cap[name] = np.asarray(nulls, float)
        per_cap[name] = {"kl": float(obs["kl_mean"]), "excess_over_null": float(ps["excess_over_null"]),
                         "null_mean": float(ps["permutation_mean"]), "perm_p": float(ps["permutation_p"]),
                         "train_loss": float(obs["train_loss"]), "domain_acc": float(obs["domain_acc"]),
                         "leakage_advantage": float(obs["leakage_advantage"]),
                         "arch": arch["probe_hidden"], "dropout": arch["dropout"]}

    names = list(caps)
    obs_max = max(per_cap[c]["kl"] for c in names)
    n_perm_eff = min(len(null_by_cap[c]) for c in names)
    null_mat = np.vstack([null_by_cap[c][:n_perm_eff] for c in names])    # [n_cap, n_perm] shared-perm aligned
    null_max = null_mat.max(axis=0)                                        # familywise null (P1.3 correctness)
    p_fw = (1.0 + float(np.sum(null_max >= obs_max))) / (1.0 + null_max.size) if null_max.size else float("nan")
    return {"per_capacity": per_cap,
            "primary_capacity": PRIMARY_CAPACITY,
            "primary_kl": per_cap[PRIMARY_CAPACITY]["kl"] if PRIMARY_CAPACITY in per_cap else None,
            "primary_perm_p": per_cap[PRIMARY_CAPACITY]["perm_p"] if PRIMARY_CAPACITY in per_cap else None,
            "familywise_max_kl": float(obs_max),
            "familywise_max_capacity": max(names, key=lambda c: per_cap[c]["kl"]),
            "familywise_max_perm_p": float(p_fw),
            "familywise_note": "max-over-capacities leakage; null repeats the SAME max on every permutation "
                               "(shared permutation seed across capacities) -> valid familywise p. "
                               "Underfitting guard only; the PRIMARY reported quantity is mlp_small.",
            "n_perm": int(n_perm_eff)}
