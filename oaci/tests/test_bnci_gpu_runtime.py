"""GPU-only runtime + model acceptance (B2b-ii). Run by slurm_gpu_smoke.sh on a V100; on a CPU node it
skips with a clear message (exit 0). Determinism must be configured BEFORE any CUDA tensor, so this
module configures it first and creates no CUDA tensor beforehand.

    python -m oaci.tests.test_bnci_gpu_runtime
"""
from __future__ import annotations

import sys

import torch

from oaci.models import build_model
from oaci.models.bn_audit import bn_buffer_hash
from oaci.runtime.cuda import (assert_cuda_runtime_unchanged, assert_single_threaded,
                               collect_native_threadpools, configure_cuda_determinism)


def _skip():
    if not torch.cuda.is_available():
        print("SKIP  no CUDA device (run on the GPU smoke job)", file=sys.stderr)
        return True
    return False


_DEVICE = None
_RUNTIME = None


def _device():
    global _DEVICE, _RUNTIME
    if _DEVICE is None:
        _DEVICE, _RUNTIME = configure_cuda_determinism()
    return _DEVICE


def test_cuda_determinism_is_configured_with_warn_only_false():
    if _skip():
        return
    _device()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def test_interop_threads_and_native_pools_are_single_threaded():
    if _skip():
        return
    _device()
    assert torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
    assert_single_threaded(collect_native_threadpools())


def test_unknown_driver_version_is_not_recorded():
    if _skip():
        return
    _device()
    assert _RUNTIME.driver_version != "unknown" and _RUNTIME.runtime_hash


def test_runtime_flags_recheck_passes_after_config():
    if _skip():
        return
    _device()
    assert_cuda_runtime_unchanged(_RUNTIME)


def _shallow():
    return build_model("shallow_convnet", in_chans=22, in_times=385, n_classes=4, temporal_filters=40,
                       temporal_kernel_samples=25, pool_kernel_samples=75, pool_stride_samples=15,
                       dropout=0.5, safe_log_eps=1e-6)


def test_shallow_real_shape_forward_on_cuda():
    if _skip():
        return
    dev = _device()
    m = _shallow().to(dev).eval()
    with torch.inference_mode():
        out = m(torch.zeros(5, 22, 385, device=dev))
    assert tuple(out.logits.shape) == (5, 4) and tuple(out.z.shape) == (5, 800)


def test_dummy_inference_and_eval_forward_preserve_bn_on_cuda():
    if _skip():
        return
    dev = _device()
    m = _shallow().to(dev)
    state = {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}
    assert int(state["bn.num_batches_tracked"]) == 0          # the dummy dim-inference forward did not update BN
    before = bn_buffer_hash(state)
    m.eval()
    with torch.inference_mode():
        m(torch.randn(8, 22, 385, device=dev))
    after = bn_buffer_hash({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()})
    assert after == before                                    # eval/inference forward never touches BN buffers


def test_nondeterministic_operation_fails_instead_of_warning():
    if _skip():
        return
    dev = _device()
    # The CONTRACT is error-mode determinism (warn_only=False) -- that is what makes any op torch
    # classifies as nondeterministic raise instead of silently warning. (Which specific ops are
    # classified is a torch-build detail: e.g. bincount has a deterministic CUDA path on this build.)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    probes = (
        ("kthvalue", lambda: torch.kthvalue(torch.randn(256, device=dev), 7)),
        ("median_dim", lambda: torch.median(torch.randn(16, 16, device=dev), dim=0)),
        ("bincount", lambda: torch.bincount(torch.randint(0, 9, (256,), device=dev))),
        ("put_no_accumulate", lambda: torch.zeros(16, device=dev).put_(
            torch.randint(0, 16, (32,), device=dev), torch.randn(32, device=dev), accumulate=False)),
    )
    raised = []
    for name, op in probes:
        try:
            op()
        except RuntimeError as e:
            if "deterministic" in str(e).lower():
                raised.append(name)
    print(f"  nondeterministic CUDA ops that raised under warn_only=False: {raised}", file=sys.stderr)
    assert raised, "with warn_only=False, at least one known-nondeterministic CUDA op must raise (none did)"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}", file=sys.stderr)
    print(f"PASS  {len(fns)} bnci-gpu-runtime tests", file=sys.stderr)


if __name__ == "__main__":
    _run_all()
