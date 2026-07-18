"""C86H outcome-free resource benchmark (synthetic arrays only — NO real EEG/label).

Measures the two production cost drivers at the locked scale and extrapolates to the full
53-target / 2,048-chain campaign, so feasibility (CPU time, peak RAM, selection-freeze
storage, max-T runtime) is confirmed BEFORE any authorized data access. If infeasible, the
decision is STOP_BEFORE_DATA_ACCESS — never a post-hoc reduction of the chain count.
"""
from __future__ import annotations

import json
import resource
import time

import numpy as np

from ..c86d.policies import acquisition_path
from ..c86d.selection_worker import freeze_budget
from . import contract as K, field_spec


def _peak_ram_gib() -> float:
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _one_target_arrays(pool_size: int, seed: int):
    """In-memory pool (probs) + queried contributions for one synthetic target."""
    contexts = field_spec.field_context_keys()
    target_pool, contrib_by_trial = {}, {}
    for ci, ctx in enumerate(contexts):
        labels, probs = field_spec._synth_context("BENCH", 1, ctx, pool_size, seed + ci)
        pc = field_spec._contribs(probs, labels)
        for j in range(pool_size):
            trial = f"t{j}"
            target_pool.setdefault(trial, {})[ctx] = probs[j]
            contrib_by_trial.setdefault(trial, (int(labels[j]), {}))[1][ctx] = {
                f: pc[f][j].tolist() for f in ("nll", "correct", "confidence", "conf_bin")}
    return target_pool, contrib_by_trial


def benchmark_selection(pool_size: int = 44, n_chains: int = 2048,
                        methods=("P0", "A1", "A2H"), seed: int = 0) -> dict:
    """Time n_chains x len(methods) full selections+freezes for one target (worker inner loop)."""
    target_pool, contrib_by_trial = _one_target_arrays(pool_size, seed)
    budgets = list(K.BUDGET_GRID)
    t0 = time.time()
    n_sel = 0
    for chain in range(n_chains):
        cseed = seed * 1_000_003 + chain
        for method in methods:
            order, q_seq = acquisition_path(target_pool, method, cseed)
            queried = {t: contrib_by_trial[t] for t in order}
            for b in budgets:
                freeze_budget(queried, order, q_seq, b)
            n_sel += 1
    dt = time.time() - t0
    return {"pool_size": pool_size, "n_chains": n_chains, "methods": list(methods),
            "selections": n_sel, "seconds": round(dt, 3),
            "seconds_per_selection": round(dt / max(n_sel, 1), 5)}


def benchmark_maxt(n_targets: int, draws: int = K.MAXT_DRAWS, seed: int = 0) -> dict:
    """Time one within-cohort max-T over the 8-hypothesis family at n_targets scale."""
    from . import analysis as AN
    rng = np.random.default_rng(seed)
    fam = {(m, b): rng.normal(0.05, 0.1, n_targets)
           for m in K.ACTIVE_METHODS for b in K.FINITE_BUDGETS}
    ds = ("Brandl2020_CANONICAL_ADULT_V1" if n_targets == 16
          else "OpenNeuro_ds007221_HYBRID_ADULT_V1")
    t0 = time.time()
    out = AN.maxt_familywise(fam, ds, draws=draws)
    dt = time.time() - t0
    return {"n_targets": n_targets, "sign_mode": out["sign_mode"], "n_signs": out["n_signs"],
            "seconds": round(dt, 3)}


def resource_benchmark(pool_size: int = 44, n_chains: int = K.ACTIVE_CHAINS,
                       methods=("P0", "A1", "A2H")) -> dict:
    """Full outcome-free feasibility benchmark; extrapolate one-target cost to 53 targets."""
    sel = benchmark_selection(pool_size=pool_size, n_chains=n_chains, methods=methods)
    maxt_brandl = benchmark_maxt(16)          # 2**16 exhaustive
    maxt_ds = benchmark_maxt(37)              # MC 65536
    per_target_s = sel["seconds"]
    total_selection_s = per_target_s * K.N_TARGETS
    # ~1 freeze JSON per (method,target,chain); measure a representative blob size
    rep = json.dumps({"budgets": [{"budget": "FULL", "status": "AVAILABLE",
                                   "query_sequence": [f"t{j}" for j in range(pool_size)]}]})
    freeze_files = len(methods) * K.N_TARGETS * n_chains
    est_freeze_storage_gib = freeze_files * len(rep) * 6 / (1024.0 ** 3)  # rough blob-size factor
    return {
        "synthetic": True, "opened_real_data": False,
        "selection": sel,
        "maxt": {"brandl": maxt_brandl, "ds007221": maxt_ds},
        "extrapolation": {
            "n_targets": K.N_TARGETS, "n_chains": n_chains,
            "total_selection_seconds": round(total_selection_s, 1),
            "total_selection_hours": round(total_selection_s / 3600.0, 3),
            "freeze_files": freeze_files,
            "selection_freeze_storage_gib_rough": round(est_freeze_storage_gib, 3),
            "maxt_per_cohort_seconds": {"brandl": maxt_brandl["seconds"],
                                        "ds007221": maxt_ds["seconds"]},
        },
        "peak_ram_gib": round(_peak_ram_gib(), 3),
        "registered_ram_envelope_gib": 128,
        "decision_rule": "if infeasible vs envelope -> STOP_BEFORE_DATA_ACCESS (never reduce chains)",
    }


if __name__ == "__main__":
    print(json.dumps(resource_benchmark(), indent=2))
