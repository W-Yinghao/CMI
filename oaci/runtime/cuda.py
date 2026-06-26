"""Deterministic CUDA runtime configuration (B2).

Configured BEFORE any CUDA tensor/model is created: the process-start env (CUBLAS_WORKSPACE_CONFIG,
PYTHONHASHSEED) is checked, the PyTorch determinism flags are set with warn_only=False (an
unsupported non-deterministic op FAILS, never warns), TF32 is off, and CPU thread counts are pinned.
Bit-exactness is claimed only within one job / node / GPU / driver+CUDA+cuDNN+PyTorch / scientific git
tree -- never across releases, platforms or GPU models.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..artifacts.canonical_json import canonical_json_hash

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_DYNAMIC")


@dataclass(frozen=True)
class NativeThreadPoolRecord:
    internal_api: str
    user_api: str
    prefix: str
    num_threads: int
    version: str | None


@dataclass(frozen=True)
class CudaRuntimeEvidence:
    cuda_initialized_before_config: bool
    cublas_workspace_config: str
    python_hash_seed: str
    cuda_visible_devices: str
    device_index: int
    device_name: str
    device_capability: tuple
    total_memory_bytes: int
    torch_version: str
    torch_cuda_version: str
    cudnn_version: int | None
    driver_version: str
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    float32_matmul_precision: str
    torch_num_threads: int
    torch_num_interop_threads: int
    native_threadpools: tuple
    thread_env: tuple
    runtime_hash: str


def _driver_version() -> str:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def collect_native_threadpools() -> tuple:
    """Every loaded BLAS/OpenMP pool (via threadpoolctl) -- the smoke requires all single-threaded."""
    import threadpoolctl
    out = []
    for i in sorted(threadpoolctl.threadpool_info(), key=lambda d: (d.get("internal_api", ""), d.get("prefix", ""))):
        out.append(NativeThreadPoolRecord(internal_api=i.get("internal_api", ""), user_api=i.get("user_api", ""),
                                          prefix=i.get("prefix", ""), num_threads=int(i.get("num_threads", -1)),
                                          version=i.get("version")))
    return tuple(out)


def assert_single_threaded(records) -> None:
    bad = [r for r in records if r.num_threads != 1]
    if bad:
        raise RuntimeError(f"native thread pools must be single-threaded; offenders: "
                           f"{[(r.prefix, r.num_threads) for r in bad]}")


def thread_env_record() -> tuple:
    return tuple((k, os.environ.get(k, "")) for k in _THREAD_ENV)


def runtime_evidence_hash(ev_fields: dict) -> str:
    return canonical_json_hash(ev_fields)


def configure_cuda_determinism(*, expected_workspace_config=":4096:8", expected_python_hash_seed="0",
                               expected_visible_device_count=1) -> tuple:
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA is already initialized; determinism must be configured before any CUDA tensor")
    ws = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    if ws != expected_workspace_config:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be {expected_workspace_config!r} (set before Python), got {ws!r}")
    phs = os.environ.get("PYTHONHASHSEED", "")
    if phs != expected_python_hash_seed:
        raise RuntimeError(f"PYTHONHASHSEED must be {expected_python_hash_seed!r} (set before Python), got {phs!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != int(expected_visible_device_count):
        raise RuntimeError(f"expected exactly {expected_visible_device_count} visible GPU, "
                           f"got {torch.cuda.device_count()}")

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)                 # strict: do NOT swallow a failure here
    if torch.get_num_threads() != 1:
        raise RuntimeError(f"torch num_threads must be 1, got {torch.get_num_threads()}")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError(f"torch interop threads must be 1, got {torch.get_num_interop_threads()}")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic algorithms did not enable")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("deterministic algorithms must NOT be warn-only")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    driver = _driver_version()
    if driver == "unknown":
        raise RuntimeError("driver_version is 'unknown'; nvidia-smi must succeed in the formal GPU smoke")
    pools = collect_native_threadpools()
    assert_single_threaded(pools)
    ev = _build_evidence(ws, phs, props, driver, pools)
    return device, ev


def _build_evidence(ws, phs, props, driver, pools) -> CudaRuntimeEvidence:
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    base = dict(
        cuda_initialized_before_config=False, cublas_workspace_config=ws, python_hash_seed=phs,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), device_index=0, device_name=props.name,
        device_capability=[props.major, props.minor], total_memory_bytes=int(props.total_memory),
        torch_version=torch.__version__, torch_cuda_version=str(torch.version.cuda), cudnn_version=cudnn,
        driver_version=driver, deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_deterministic=True, cudnn_benchmark=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
        float32_matmul_precision="highest", torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        native_threadpools=[[p.internal_api, p.user_api, p.prefix, p.num_threads, p.version] for p in pools],
        thread_env=[list(e) for e in thread_env_record()])
    rh = runtime_evidence_hash(base)
    return CudaRuntimeEvidence(
        cuda_initialized_before_config=False, cublas_workspace_config=ws, python_hash_seed=phs,
        cuda_visible_devices=base["cuda_visible_devices"], device_index=0, device_name=props.name,
        device_capability=(props.major, props.minor), total_memory_bytes=int(props.total_memory),
        torch_version=torch.__version__, torch_cuda_version=str(torch.version.cuda), cudnn_version=cudnn,
        driver_version=driver, deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_deterministic=True, cudnn_benchmark=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
        float32_matmul_precision="highest", torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(), native_threadpools=pools,
        thread_env=thread_env_record(), runtime_hash=rh)


def assert_cuda_runtime_unchanged(initial: CudaRuntimeEvidence) -> None:
    """Re-check the live determinism flags against the configured evidence (after each run)."""
    cur = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
               cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
               matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
               float32_matmul_precision=torch.get_float32_matmul_precision(),
               torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads())
    for k, v in cur.items():
        if getattr(initial, k) != v:
            raise RuntimeError(f"CUDA runtime flag {k} drifted: configured {getattr(initial, k)!r}, now {v!r}")
