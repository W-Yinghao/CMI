"""B2a CPU tests: deterministic-runtime config guards, RNG snapshots, the forked-rng CUDA index, and
the frozen probe solver/tol/fit_intercept + smoke budget. The full GPU happy-path is in the GPU smoke.

Standalone (``python -m oaci.tests.test_cuda_runtime``) and pytest-compatible.
"""
from __future__ import annotations

import dataclasses
import os

import torch

import oaci.protocol
from oaci.leakage.cache import critic_config_hash
from oaci.leakage.critic import CriticConfig
from oaci.protocol.manifest_v2 import load_v2
from oaci.runner.config import RunnerExecutionConfig
from oaci.runtime.cuda import configure_cuda_determinism, runtime_evidence_hash
from oaci.runtime.rng_state import assert_rng_unchanged, snapshot_rng_state
from oaci.train.rng import forked_rng

_SMOKE = os.path.join(os.path.dirname(oaci.protocol.__file__), "smoke_v1.yaml")


def _with_env(**env):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    return saved


# ============================ cuda config guards (CPU-reachable) ============================
def test_cuda_config_requires_cublas_workspace_env():
    saved = _with_env(CUBLAS_WORKSPACE_CONFIG=None, PYTHONHASHSEED="0")
    try:
        configure_cuda_determinism()
    except RuntimeError as e:
        assert "CUBLAS_WORKSPACE_CONFIG" in str(e)
    else:
        raise AssertionError("a missing CUBLAS_WORKSPACE_CONFIG must be rejected")
    finally:
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def test_cuda_config_requires_pythonhashseed():
    saved = _with_env(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED=None)
    try:
        configure_cuda_determinism()
    except RuntimeError as e:
        assert "PYTHONHASHSEED" in str(e)
    else:
        raise AssertionError("a missing PYTHONHASHSEED must be rejected")
    finally:
        os.environ.update({k: v for k, v in saved.items() if v is not None})
        os.environ.pop("PYTHONHASHSEED", None) if saved.get("PYTHONHASHSEED") is None else None


def test_cuda_config_requires_cuda_available_on_a_cpu_node():
    saved = _with_env(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
    try:
        if torch.cuda.is_available():
            return                                            # on a GPU node this path is the happy path
        try:
            configure_cuda_determinism()
        except RuntimeError as e:
            assert "CUDA is not available" in str(e)
        else:
            raise AssertionError("a CPU node must be rejected before configuring flags")
    finally:
        os.environ.update({k: (v if v is not None else "") for k, v in saved.items()})


def test_cuda_runtime_evidence_hash_binds_versions_device_and_flags():
    base = {"torch_version": "2.0", "device_name": "V100", "deterministic_warn_only": False, "x": 1}
    h = runtime_evidence_hash(base)
    for over in ({"torch_version": "2.1"}, {"device_name": "A100"}, {"deterministic_warn_only": True}):
        assert runtime_evidence_hash({**base, **over}) != h


def test_runtime_evidence_hash_is_canonical():
    from oaci.artifacts.canonical_json import canonical_json_hash
    payload = {"torch_version": "2.0", "device_capability": [7, 0], "cudnn_version": 8902}
    assert runtime_evidence_hash(payload) == canonical_json_hash(payload)
    assert runtime_evidence_hash(payload) == runtime_evidence_hash({"cudnn_version": 8902, "device_capability": [7, 0], "torch_version": "2.0"})


def test_native_threadpool_record_collection_and_single_thread_check():
    from oaci.runtime.cuda import NativeThreadPoolRecord, assert_single_threaded, collect_native_threadpools, thread_env_record
    recs = collect_native_threadpools()
    assert all(isinstance(r, NativeThreadPoolRecord) for r in recs)        # structure
    assert all(isinstance(e, tuple) and len(e) == 2 for e in thread_env_record())
    assert_single_threaded((NativeThreadPoolRecord("a", "a", "x", 1, "1"),))
    try:
        assert_single_threaded((NativeThreadPoolRecord("openmp", "openmp", "libgomp", 4, "1"),))
    except RuntimeError:
        pass
    else:
        raise AssertionError("a multi-threaded native pool must be rejected")


def test_runtime_flags_are_rechecked_and_drift_is_detected():
    import types
    from oaci.runtime.cuda import assert_cuda_runtime_unchanged
    live = types.SimpleNamespace(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads())
    assert_cuda_runtime_unchanged(live)                                    # matches the live flags
    live.cudnn_benchmark = not live.cudnn_benchmark
    try:
        assert_cuda_runtime_unchanged(live)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a drifted runtime flag must be detected")


# ============================ rng snapshots ============================
def test_rng_snapshot_is_stable_and_detects_change():
    a = snapshot_rng_state(torch.device("cpu"))
    assert a.snapshot_hash == snapshot_rng_state(torch.device("cpu")).snapshot_hash and a.torch_cuda_hash is None
    torch.randn(2)
    try:
        assert_rng_unchanged(a, snapshot_rng_state(torch.device("cpu")), "test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("an RNG change must be detected")


def test_forked_rng_uses_explicit_cuda_index_and_restores_cpu():
    before = torch.random.get_rng_state()
    with forked_rng(7, torch.device("cpu")):
        torch.randn(4)
    assert torch.equal(before, torch.random.get_rng_state())
    try:
        with forked_rng(7, torch.device("cuda")):           # un-indexed cuda is forbidden
            pass
    except ValueError:
        pass
    else:
        raise AssertionError("CUDA RNG isolation must require an indexed device")


# ============================ frozen probe + smoke budget ============================
def test_probe_solver_tol_and_intercept_are_manifest_frozen():
    ec = RunnerExecutionConfig.from_manifest(load_v2(_SMOKE))
    assert ec.critic.solver == "lbfgs" and ec.critic.tol == 1e-4 and ec.critic.fit_intercept is True


def test_critic_config_hash_binds_solver_tol_and_intercept():
    c = CriticConfig(capacities=(0, 8))
    base = critic_config_hash(c)
    for over in (dict(solver="saga"), dict(tol=1e-3), dict(fit_intercept=False)):
        assert critic_config_hash(dataclasses.replace(c, **over)) != base


def test_domain_probe_passes_all_frozen_solver_fields():
    import numpy as np
    from oaci.leakage.critic import DomainProbe
    cfg = CriticConfig(capacities=(0,), solver="liblinear", tol=1e-3, fit_intercept=False, max_iter=50)
    Z = np.random.default_rng(0).standard_normal((20, 3))
    p = DomainProbe(0, 2, cfg).fit(Z, np.array([0, 1] * 10))
    assert p._model.solver == "liblinear" and p._model.tol == 1e-3 and p._model.fit_intercept is False


def test_smoke_budget_is_minimal_and_probe_frozen():
    m = load_v2(_SMOKE)
    t, p = m.training, m.probe
    assert (t.stage1_epochs, t.stage2_epochs, t.stage1_steps_per_epoch, t.stage2_steps_per_epoch) == (2, 2, 2, 2)
    assert p.selection_bootstrap == 8 and p.audit_bootstrap == 8 and m.evaluation.paired_bootstrap == 8
    assert p.solver == "lbfgs" and p.fit_intercept is True


def test_confirmatory_budget_is_unchanged():
    p = load_v2(os.path.join(os.path.dirname(_SMOKE), "confirmatory_v2.yaml")).probe
    assert p.selection_bootstrap == 200 and p.audit_bootstrap == 2000   # confirmatory protocol untouched


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runtime.cuda  # noqa: F401
    import oaci.runner.bnci_artifact  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} cuda-runtime tests")


if __name__ == "__main__":
    _run_all()
