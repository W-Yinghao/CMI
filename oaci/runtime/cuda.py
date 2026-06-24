"""Deterministic CUDA runtime configuration (B2).

Configured BEFORE any CUDA tensor/model is created: the process-start env (CUBLAS_WORKSPACE_CONFIG,
PYTHONHASHSEED) is checked, the PyTorch determinism flags are set with warn_only=False (an
unsupported non-deterministic op FAILS, never warns), TF32 is off, and CPU thread counts are pinned.
Bit-exactness is claimed only within one job / node / GPU / driver+CUDA+cuDNN+PyTorch / scientific git
tree -- never across releases, platforms or GPU models.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch

from .rng_state import _h  # noqa: F401


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
    runtime_hash: str


def _driver_version() -> str:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def runtime_evidence_hash(ev_fields: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(ev_fields):
        h.update(f"{k}={ev_fields[k]!r}".encode())
    return h.hexdigest()


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
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                                       # already set / parallel work started: leave as-is
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic algorithms did not enable")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("deterministic algorithms must NOT be warn-only")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    fields = dict(
        cuda_initialized_before_config=False, cublas_workspace_config=ws, python_hash_seed=phs,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), device_index=0,
        device_name=props.name, device_capability=(props.major, props.minor), total_memory_bytes=int(props.total_memory),
        torch_version=torch.__version__, torch_cuda_version=str(torch.version.cuda),
        cudnn_version=(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        driver_version=_driver_version(), deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_deterministic=True, cudnn_benchmark=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
        float32_matmul_precision="highest", torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads())
    ev = CudaRuntimeEvidence(**fields, runtime_hash=runtime_evidence_hash(fields))
    return device, ev
