"""Determinism-safe process-parallel leakage bootstrap: the parallel replay is BIT-IDENTICAL to the
sequential loop, single-threaded + CUDA-free workers, and never enters any scientific hash.

The sequential reference is computed under ``threadpool_limits(1)`` so it matches the (single-threaded)
workers regardless of the ambient OMP thread count (the CI node runs OMP_NUM_THREADS>1; production runs
OMP_NUM_THREADS=1, where the sequential path is already single-threaded).

Standalone (``python -m oaci.tests.test_leakage_parallel``) and pytest-compatible.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from oaci.leakage.cache import critic_config_hash
from oaci.leakage.critic import CriticConfig
from oaci.leakage.crossfit import make_fold_plan
from oaci.leakage.parallel import (_contiguous_chunks, _reduce_results, get_leakage_parallel,
                                   leakage_parallel_report, set_leakage_parallel, _init_worker, _THREAD_ENV)
from oaci.leakage.plan import make_leakage_bootstrap_plan
from oaci.leakage.synthetic import make_nonlinear
from oaci.leakage.ucb import _design_from_feat, bootstrap_ucb

_CFG = CriticConfig(capacities=(0, 32))


def _fixture(seed=1, per_cell=40, B=24):
    feat, sg = make_nonlinear(seed=seed, per_cell=per_cell)
    fold = make_fold_plan(feat, sg, n_folds=2, seed=0)
    design = _design_from_feat(feat, sg)
    plan = make_leakage_bootstrap_plan(design, sg, fold, alpha=0.1, requested_replicates=B, seed=0,
                                       max_candidate_multiplier=5, max_invalid_draw_rate=1.0)
    return feat, sg, fold, plan


def _seq(feat, sg, fold, plan):
    with threadpool_limits(limits=1):                          # match the single-threaded workers
        return bootstrap_ucb(feat, sg, fold, _CFG, bootstrap_plan=plan, parallel_backend="sequential")


def _par(feat, sg, fold, plan, n=4):
    return bootstrap_ucb(feat, sg, fold, _CFG, bootstrap_plan=plan, parallel_backend="process",
                         parallel_n_jobs=n)


# ============ bit-exact equivalence ============
def test_parallel_bootstrap_matches_sequential_point_estimate():
    feat, sg, fold, plan = _fixture()
    s, p = _seq(feat, sg, fold, plan), _par(feat, sg, fold, plan)
    assert s["extractable_LQ_ov"] == p["extractable_LQ_ov"]
    assert s["L_abs"] == p["L_abs"] and s["L_cond"] == p["L_cond"]


def test_parallel_bootstrap_matches_sequential_ucl():
    feat, sg, fold, plan = _fixture()
    s, p = _seq(feat, sg, fold, plan), _par(feat, sg, fold, plan)
    assert np.array_equal(s["replicates"], p["replicates"])   # bit-identical replicate values + order
    assert s["bootstrap_ucl"] == p["bootstrap_ucl"]
    assert s["percentile_ucl"] == p["percentile_ucl"]
    assert s["n_bootstrap"] == p["n_bootstrap"]


def test_parallel_bootstrap_matches_replicate_capacity_sequence():
    feat, sg, fold, plan = _fixture()
    s, p = _seq(feat, sg, fold, plan), _par(feat, sg, fold, plan)
    assert list(s["replicate_capacities"]) == list(p["replicate_capacities"])


def test_parallel_matches_across_worker_counts():
    feat, sg, fold, plan = _fixture(B=20)
    base = _seq(feat, sg, fold, plan)
    for n in (1, 2, 3, 8):                                     # result independent of the worker count
        r = _par(feat, sg, fold, plan, n=n)
        assert np.array_equal(base["replicates"], r["replicates"]) and base["bootstrap_ucl"] == r["bootstrap_ucl"]


# ============ explicit-plan replay (no redraw) ============
def test_parallel_bootstrap_uses_explicit_plan_without_redraw():
    feat, sg, fold, plan = _fixture()
    s, p = _seq(feat, sg, fold, plan), _par(feat, sg, fold, plan)
    assert s["bootstrap_plan_hash"] == p["bootstrap_plan_hash"] == plan.plan_hash   # same plan, no redraw
    assert s["n_bootstrap"] == len(plan.accepted_candidate_ids)


# ============ failure path (order-preserving, candidate-id) ============
def test_parallel_bootstrap_preserves_candidate_id_failure_path():
    ok = [(7, True, None, 0.1, 0), (9, True, None, 0.2, 32)]
    assert _reduce_results(ok) == ([0.1, 0.2], [0, 32])
    bad = [(7, True, None, 0.1, 0), (9, False, "zero OOF mass", None, None), (11, False, "x", None, None)]
    try:
        _reduce_results(bad)
    except ValueError as e:
        assert "candidate 9 failed during scoring" in str(e)  # FIRST failure in order, by candidate id
        return
    raise AssertionError("a failed candidate must raise")


# ============ worker isolation ============
def test_parallel_workers_are_single_threaded():
    assert leakage_parallel_report()["worker_threads"] == 1
    saved = {v: os.environ.get(v) for v in _THREAD_ENV}
    try:
        for v in _THREAD_ENV:
            os.environ.pop(v, None)
        _init_worker()                                         # the worker init pins every native pool to 1
        assert all(os.environ[v] == "1" for v in _THREAD_ENV)
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def test_parallel_workers_do_not_initialize_cuda():
    # the entire leakage worker stack must be torch-free (so a spawn worker never touches CUDA)
    code = ("import oaci.leakage.parallel, oaci.leakage.ucb, oaci.leakage.estimate, "
            "oaci.leakage.crossfit, oaci.leakage.critic; import sys; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m=='torch')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"leakage worker stack imported torch: {r.stderr}"


# ============ no science-hash leakage ============
def test_parallel_mode_does_not_change_critic_or_bootstrap_hashes():
    feat, sg, fold, plan = _fixture()
    h0 = critic_config_hash(_CFG)
    set_leakage_parallel(8, "process")
    try:
        assert critic_config_hash(_CFG) == h0                 # ambient mode does NOT enter critic_config_hash
        p = _par(feat, sg, fold, plan)
        assert p["bootstrap_plan_hash"] == plan.plan_hash      # nor the bootstrap plan identity
    finally:
        set_leakage_parallel(1, "sequential")


def test_contiguous_chunks_preserve_order():
    ids = list(range(10))
    for k in (1, 3, 4, 10, 16):
        chunks = _contiguous_chunks(ids, k)
        assert [x for ch in chunks for x in ch] == ids        # concat in chunk order == original order
        assert sum(len(c) for c in chunks) == len(ids)
        assert all(chunks)                                     # no empty chunk (k clamped to <= n)


def test_set_get_leakage_parallel_roundtrip():
    try:
        set_leakage_parallel(16, "process")
        assert get_leakage_parallel() == {"n_jobs": 16, "backend": "process"}
        assert leakage_parallel_report()["leakage_parallel_mode"] == "process_bootstrap_replicate"
    finally:
        set_leakage_parallel(1, "sequential")
    assert leakage_parallel_report()["leakage_parallel_mode"] == "sequential"


def test_parallel_fake_two_level_matches_sequential_end_to_end():
    """The full fake two-level closed loop: sequential vs process-parallel leakage must produce identical
    selected checkpoints, leakage hashes, fold_result_hash and BOTH artifact hashes. Both runs are wrapped
    at one thread so the comparison is valid regardless of the ambient OMP count (production runs OMP=1)."""
    import os as _os
    import tempfile

    import oaci.protocol
    from oaci.artifacts.writer import GitEvidence, git_evidence_hash
    from oaci.runner.fake_artifact import run_fake_two_level
    from oaci.runner.scientific_hash import leakage_result_hash
    man = _os.path.join(_os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")
    order = ("ERM", "OACI", "global_lpc", "uniform")
    c, t = "c" * 40, "t" * 40
    ge = GitEvidence(c, t, ("oaci",), (), True, git_evidence_hash(c, t, ("oaci",), (), True))
    with threadpool_limits(limits=1):
        set_leakage_parallel(1, "sequential")
        a = run_fake_two_level(man, tempfile.mkdtemp(), model_seed=0, method_order=order, repo_root="/x", git_evidence=ge)
        set_leakage_parallel(4, "process")
        try:
            b = run_fake_two_level(man, tempfile.mkdtemp(), model_seed=0, method_order=order, repo_root="/x", git_evidence=ge)
        finally:
            set_leakage_parallel(1, "sequential")
    assert a.fold_result.fold_result_hash == b.fold_result.fold_result_hash          # binds everything
    assert a.write_result.artifact_scientific_hash == b.write_result.artifact_scientific_hash
    assert a.write_result.artifact_pure_science_hash == b.write_result.artifact_pure_science_hash
    for (lvl, la), (_, lb) in zip(a.fold_result.level_items, b.fold_result.level_items):
        for (n, ma), (_, mb) in zip(la.method_items, lb.method_items):
            assert ma.selection.model_hash == mb.selection.model_hash                 # same selected checkpoint
            for leak_a, leak_b in ((ma.selection_leakage, mb.selection_leakage),
                                   (ma.audit_leakage, mb.audit_leakage)):
                assert (leak_a is None) == (leak_b is None)
                if leak_a is not None:
                    assert leakage_result_hash(leak_a) == leakage_result_hash(leak_b)


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import oaci.leakage.parallel  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} leakage-parallel tests")


if __name__ == "__main__":
    _run_all()
