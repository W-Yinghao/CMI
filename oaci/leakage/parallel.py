"""Determinism-safe process-parallel leakage bootstrap.

PURE EXECUTION ACCELERATION. The per-replicate leakage estimate is computed by independent worker
processes, but the replicate values AND their order are IDENTICAL to the sequential loop:

* the bootstrap is replayed from an EXPLICIT ``LeakageBootstrapPlan`` -- no in-worker RNG, no redraw,
  no fold/support change; each worker just replays pre-planned candidate draws;
* the accepted draws are split into CONTIGUOUS chunks, mapped in order, and concatenated in chunk order,
  so the reduced ``reps`` / ``replicate_capacities`` are in ``accepted_candidate_ids`` order exactly as
  the sequential loop produces them;
* every worker pins ALL native thread pools to 1 (``threadpool_limits(1)`` + the OMP/MKL/... env), so
  each sklearn ``LogisticRegression`` fit is single-threaded and therefore bit-reproducible -- matching
  the production sequential path (which runs under ``OMP_NUM_THREADS=1``);
* the pool is PERSISTENT (created once, reused across every scoring) so the worker import of the leakage
  stack (~3s) is paid once, not per candidate; the context is ``spawn`` so no initialized CUDA context is
  inherited, and the leakage stack is torch-free, so a worker never touches CUDA.

The parallel mode is recorded in the runtime/provenance report ONLY; it never enters the manifest,
``critic_config_hash``, the bootstrap plan or any scientific hash. A parallel run and a sequential run of
the same science produce the SAME ``artifact_scientific_hash``.
"""
from __future__ import annotations

import atexit
import os

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# ambient execution-only config (NOT hashed; injected by the runner/CLI, read by compute_leakage_score)
_LEAKAGE_PARALLEL = {"n_jobs": 1, "backend": "sequential"}

_POOL = None          # persistent spawn pool
_POOL_NJOBS = None
_ATEXIT = False


def set_leakage_parallel(n_jobs: int = 1, backend: str = "sequential") -> None:
    if backend not in ("sequential", "process"):
        raise ValueError(f"leakage parallel backend must be sequential|process; got {backend!r}")
    _LEAKAGE_PARALLEL["n_jobs"] = int(n_jobs)
    _LEAKAGE_PARALLEL["backend"] = str(backend)
    if backend != "process" or int(n_jobs) <= 1:
        _shutdown_pool()                                     # release workers when leaving parallel mode


def get_leakage_parallel() -> dict:
    return dict(_LEAKAGE_PARALLEL)


def leakage_parallel_active() -> bool:
    p = _LEAKAGE_PARALLEL
    return p["backend"] == "process" and int(p["n_jobs"]) > 1


def leakage_parallel_report() -> dict:
    p = get_leakage_parallel()
    return {"leakage_parallel_mode": ("process_bootstrap_replicate" if leakage_parallel_active() else "sequential"),
            "leakage_parallel_n_jobs": int(p["n_jobs"]), "worker_threads": 1}


# ---- worker side (spawn) ----
def _init_worker() -> None:
    for v in _THREAD_ENV:
        os.environ[v] = "1"                                  # belt-and-suspenders with threadpool_limits


def _estimate_chunk(args):
    """Replay a CONTIGUOUS chunk of pre-planned draws -> [(cid, ok, err, LQ_ov, capacity), ...] in chunk
    order. Single-threaded (every native pool pinned to 1) so each fit is bit-reproducible."""
    from threadpoolctl import threadpool_limits

    from .estimate import estimate_extractable_leakage
    from .ucb import _rebuild
    feat, group_rows, by_id, support_graph, fold_plan, cfg, chunk = args
    out = []
    with threadpool_limits(limits=1):
        for cid in chunk:
            resampled = [g for g, m in by_id[cid].group_multiplicities for _ in range(int(m))]
            feat_b = _rebuild(feat, group_rows, resampled)
            try:
                est = estimate_extractable_leakage(feat_b, support_graph, fold_plan, cfg)
            except ValueError as e:                          # an accepted draw must NOT fail; carry it back
                out.append((cid, False, str(e), None, None))
                continue
            out.append((cid, True, None, float(est["extractable_LQ_ov"]), est["selected_capacity"]))
    return out


def _get_pool(n_jobs):
    global _POOL, _POOL_NJOBS, _ATEXIT
    if _POOL is not None and _POOL_NJOBS == n_jobs:
        return _POOL
    _shutdown_pool()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")                            # never inherit an initialized CUDA context
    _POOL = ctx.Pool(processes=int(n_jobs), initializer=_init_worker)
    _POOL_NJOBS = int(n_jobs)
    if not _ATEXIT:
        atexit.register(_shutdown_pool)
        _ATEXIT = True
    return _POOL


def _shutdown_pool() -> None:
    global _POOL, _POOL_NJOBS
    if _POOL is not None:
        try:
            _POOL.close()
            _POOL.join()
        except Exception:  # noqa: BLE001
            pass
        _POOL = None
        _POOL_NJOBS = None


def _contiguous_chunks(ids, k):
    """Split ``ids`` into ``k`` contiguous near-equal chunks (order-preserving on concatenation)."""
    n = len(ids)
    k = max(1, min(int(k), n))
    base, rem = divmod(n, k)
    out, i = [], 0
    for j in range(k):
        size = base + (1 if j < rem else 0)
        out.append(ids[i:i + size])
        i += size
    return out


def parallel_replicate_estimates(accepted_candidate_ids, feat, group_rows, by_id, support_graph,
                                 fold_plan, cfg, n_jobs):
    """Process-parallel replay of the accepted bootstrap draws over a PERSISTENT spawn pool. Returns
    (reps, replicate_capacities) in EXACTLY ``accepted_candidate_ids`` order (contiguous chunks mapped +
    concatenated in order), identical to the sequential loop. Raises for the first (in order) failed
    candidate with the sequential error message."""
    ids = list(accepted_candidate_ids)
    if not ids:
        return [], []
    pool = _get_pool(int(n_jobs))
    chunks = _contiguous_chunks(ids, int(n_jobs))
    args = [(feat, group_rows, by_id, support_graph, fold_plan, cfg, ch) for ch in chunks]
    chunk_results = pool.map(_estimate_chunk, args)          # order = chunks order
    flat = [r for ch in chunk_results for r in ch]           # contiguous chunks -> accepted order
    return _reduce_results(flat)


def _permutation_chunk(args):
    """Replay a CONTIGUOUS chunk of pre-planned permutation bit-rows -> [Δ, ...] in chunk order. The K1
    paired-permutation delta is computed by the SAME function the sequential path uses (lazy import breaks
    the decision<->leakage module cycle); single-threaded so each probe fit is bit-reproducible."""
    from threadpoolctl import threadpool_limits

    from ..decision.k1_permutation import k1_delta_for_bit_row
    feat_erm, feat_oaci, stratum_index, support_graph, fold_plan, cfg, chunk = args
    out = []
    with threadpool_limits(limits=1):
        for br in chunk:
            out.append(k1_delta_for_bit_row(feat_erm, feat_oaci, stratum_index, support_graph, fold_plan,
                                            cfg, br))
    return out


def parallel_paired_permutation_deltas(bit_rows, feat_erm, feat_oaci, stratum_index, support_graph,
                                       fold_plan, cfg, n_jobs):
    """Process-parallel replay of the permutation bit-rows over the PERSISTENT spawn pool. Returns the null
    Δ list in EXACTLY ``bit_rows`` order (contiguous chunks mapped + concatenated in order) — bit-identical
    to the sequential loop. Pure acceleration; enters no scientific hash."""
    rows = list(bit_rows)
    if not rows:
        return []
    pool = _get_pool(int(n_jobs))
    chunks = _contiguous_chunks(rows, int(n_jobs))
    args = [(feat_erm, feat_oaci, stratum_index, support_graph, fold_plan, cfg, ch) for ch in chunks]
    chunk_results = pool.map(_permutation_chunk, args)           # order = chunks order
    return [d for ch in chunk_results for d in ch]               # contiguous chunks -> bit_rows order


def _reduce_results(results):
    """Reduce per-candidate worker results (already in ``accepted_candidate_ids`` order) into
    (reps, replicate_capacities); raise for the FIRST (in order) failed candidate with the sequential
    error message. Pure + spawn-free so the failure path is unit-testable."""
    reps, caps = [], []
    for cid, ok, err, lq, cap in results:
        if not ok:
            raise ValueError(f"accepted bootstrap candidate {cid} failed during scoring: {err}")
        reps.append(lq)
        caps.append(cap)
    return reps, caps
