"""C86H outcome-free resource benchmark (synthetic arrays only — NO real EEG/label).

Measures the ACTUAL production path — batch H1a order generation + sealed H1b contribution
evaluation with real NPZ serialization, plus the within-cohort max-T — for one representative
target, then extrapolates to the full 53-target / 2,048-chain campaign. This is not an
in-memory placeholder: it runs the real ``run_h1a`` / ``run_h1b_sealed`` and measures the
serialized compact-freeze bytes on disk. If infeasible vs the registered envelope the decision
is STOP_BEFORE_DATA_ACCESS — never a post-hoc reduction of the chain count.
"""
from __future__ import annotations

import glob
import json
import os
import resource
import shutil
import tempfile
import time

import numpy as np

from . import contract as K, field_spec, batch_h1


def _peak_ram_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)  # KiB->GiB


def benchmark_batch_h1(n_trials: int = 600, n_chains: int = K.ACTIVE_CHAINS,
                       seed: int = 0) -> dict:
    """Run the real batch H1 (order gen + sealed contribution eval + serialization) for ONE
    synthetic target at production scale; measure wall time and on-disk freeze bytes."""
    root = tempfile.mkdtemp(prefix="c86h_bench_")
    try:
        fr = os.path.join(root, "field")
        field_spec.synthesize_field(
            fr, {"BENCH": {"dataset": "BENCH", "subjects": [1], "n_trials": n_trials}}, seed=seed)
        methods = list(K.METHOD_REGISTRY); chains = list(range(n_chains))
        odir = os.path.join(root, "orders"); h1 = os.path.join(root, "h1")
        t0 = time.time()
        pool_sizes = batch_h1.run_h1a(os.path.join(fr, "acquisition_unlabeled_pool"),
                                      odir, methods, chains)
        t1 = time.time()
        batch_h1.run_h1b_sealed(odir, os.path.join(fr, "acquisition_label_oracle"),
                                os.path.join(fr, "query_contribution_store"), h1, methods, chains)
        t2 = time.time()
        freeze_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(h1, "*.npz")))
        order_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(odir, "*.npz")))
        ps = next(iter(pool_sizes.values()))
        return {"n_trials": n_trials, "pool_size": ps, "n_chains": n_chains,
                "h1a_seconds": round(t1 - t0, 3), "h1b_seconds": round(t2 - t1, 3),
                "per_target_seconds": round(t2 - t0, 3),
                "freeze_bytes_per_target": int(freeze_bytes),
                "order_bytes_per_target": int(order_bytes)}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def benchmark_maxt(n_targets: int, draws: int = K.MAXT_DRAWS, seed: int = 0) -> dict:
    from . import analysis as AN
    rng = np.random.default_rng(seed)
    fam = {(m, b): rng.normal(0.05, 0.1, n_targets)
           for m in K.ACTIVE_METHODS for b in K.FINITE_BUDGETS}
    ds = ("Brandl2020_CANONICAL_ADULT_V1" if n_targets == 16
          else "OpenNeuro_ds007221_HYBRID_ADULT_V1")
    t0 = time.time()
    out = AN.maxt_familywise(fam, ds, draws=draws)
    return {"n_targets": n_targets, "sign_mode": out["sign_mode"], "n_signs": out["n_signs"],
            "seconds": round(time.time() - t0, 3)}


def resource_benchmark(n_trials: int = 600, n_chains: int = K.ACTIVE_CHAINS) -> dict:
    """Full outcome-free feasibility benchmark of the real batch path; extrapolate one-target
    cost to 53 targets."""
    h1 = benchmark_batch_h1(n_trials=n_trials, n_chains=n_chains)
    maxt_brandl = benchmark_maxt(16)          # exhaustive 2**16
    maxt_ds = benchmark_maxt(37)              # MC 65536
    per_target_s = h1["per_target_seconds"]
    total_s = per_target_s * K.N_TARGETS
    freeze_gib = h1["freeze_bytes_per_target"] * K.N_TARGETS / (1024.0 ** 3)
    return {
        "synthetic": True, "opened_real_data": False,
        "batch_h1_one_target": h1,
        "maxt": {"brandl": maxt_brandl, "ds007221": maxt_ds},
        "extrapolation": {
            "n_targets": K.N_TARGETS, "n_chains": n_chains,
            "compact_freeze_files": K.N_TARGETS * len(K.METHOD_REGISTRY),   # 53*3 = 159
            "total_h1_seconds": round(total_s, 1),
            "total_h1_hours_serial": round(total_s / 3600.0, 3),
            "total_h1_hours_53core_parallel": round(per_target_s / 3600.0, 3),
            "compact_freeze_storage_gib": round(freeze_gib, 3),
            "maxt_per_cohort_seconds": {"brandl": maxt_brandl["seconds"],
                                        "ds007221": maxt_ds["seconds"]},
        },
        "peak_ram_gib": round(_peak_ram_gib(), 3),
        "registered_ram_envelope_gib": 128,
        "decision_rule": "if infeasible vs envelope -> STOP_BEFORE_DATA_ACCESS (never reduce chains)",
    }


if __name__ == "__main__":
    print(json.dumps(resource_benchmark(), indent=2))
