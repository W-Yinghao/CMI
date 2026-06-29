"""ACAR v4 — regen runtime env-lock SCHEMA + validator (Option B; code-only, NO torch capture).

NON-BINDING. Defines the runtime lock the all-DEV substrate regeneration MUST pin (torch/braindecode/CUDA + device +
determinism + threads), as a PURE schema + fail-closed validator + canonical hasher. It does NOT capture a real runtime
(no torch import): a SCHEMA-ONLY template carries status "SCHEMA_ONLY_NOT_CAPTURED" with placeholder versions; only a lock
that was genuinely captured on the training node may carry status "CAPTURED_AND_VERIFIED" (validator enforces that captured
locks have non-empty real version/device fields — a skeleton cannot impersonate a captured environment). run_regen_substrate
requires CAPTURED_AND_VERIFIED before the (still B1-gated) training. Device choice is PINNED here — not a post-hoc freedom.
"""
from __future__ import annotations
import hashlib
import json

SCHEMA_VERSION = "acar_v4_regen_env_lock/1"
STATUSES = ("SCHEMA_ONLY_NOT_CAPTURED", "CAPTURED_AND_VERIFIED")
DEVICE_KINDS = ("cuda", "cpu")

_REQUIRED = (
    "schema_version", "status",
    "python_version", "torch_version", "braindecode_version", "numpy_version", "scipy_version", "sklearn_version",
    "cuda_version", "cudnn_version", "device_kind", "device_name", "driver_version",
    "torch_deterministic_algorithms", "seed",
    "torch_intraop_threads", "torch_interop_threads", "omp_num_threads", "threadpool_backends",
    "pipeline_config_sha256", "protocol_commit",
)
# version/identity fields a CAPTURED_AND_VERIFIED lock must fill with real (non-empty) values
_CAPTURED_NONEMPTY = ("python_version", "torch_version", "braindecode_version", "numpy_version", "scipy_version",
                      "sklearn_version", "device_name")
# extra fields required to be non-empty when device_kind == "cuda"
_CUDA_NONEMPTY = ("cuda_version", "cudnn_version", "driver_version")


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in "0123456789abcdef" for c in s)


def expected_regen_env_fields():
    return _REQUIRED


def validate_regen_env_lock(lock):
    """PURE fail-closed schema validation (no torch import). Required fields exactly (no missing/extra), known status +
    device_kind, strict-int seed 0, deterministic flag True, positive-int thread counts, 64/40-hex hashes. A
    CAPTURED_AND_VERIFIED lock additionally MUST have non-empty real version/device fields (skeleton can't fake capture).
    Returns the lock."""
    if not isinstance(lock, dict):
        raise ValueError("env lock must be a JSON object")
    missing = [f for f in _REQUIRED if f not in lock]
    if missing:
        raise ValueError(f"env lock missing fields: {missing}")
    extra = [f for f in lock if f not in _REQUIRED]
    if extra:
        raise ValueError(f"env lock has unknown extra fields: {extra}")
    if lock["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if lock["status"] not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if lock["device_kind"] not in DEVICE_KINDS:
        raise ValueError(f"device_kind must be one of {DEVICE_KINDS}")
    if type(lock["seed"]) is not int or lock["seed"] != 0:
        raise ValueError("seed must be the int 0 (no bool/str/float)")
    if lock["torch_deterministic_algorithms"] is not True:
        raise ValueError("torch_deterministic_algorithms must be true")
    for tf in ("torch_intraop_threads", "torch_interop_threads", "omp_num_threads"):
        v = lock[tf]
        if type(v) is not int or v < 1:
            raise ValueError(f"{tf} must be a positive int")
    if not isinstance(lock["threadpool_backends"], (list, dict)):
        raise ValueError("threadpool_backends must be a list or dict")
    if not _is_hex(lock["pipeline_config_sha256"], 64):
        raise ValueError("pipeline_config_sha256 must be a 64-char lowercase sha-256")
    if not _is_hex(lock["protocol_commit"], 40):
        raise ValueError("protocol_commit must be a full 40-char lowercase git SHA-1")
    for sf in ("python_version", "torch_version", "braindecode_version", "numpy_version", "scipy_version",
               "sklearn_version", "cuda_version", "cudnn_version", "device_name", "driver_version"):
        if not isinstance(lock[sf], str):
            raise ValueError(f"{sf} must be a string")
    if lock["status"] == "CAPTURED_AND_VERIFIED":
        empty = [f for f in _CAPTURED_NONEMPTY if not lock[f]]
        if empty:
            raise ValueError(f"CAPTURED_AND_VERIFIED lock must fill real values for {empty}")
        if lock["device_kind"] == "cuda":
            empty_cuda = [f for f in _CUDA_NONEMPTY if not lock[f]]
            if empty_cuda:
                raise ValueError(f"CAPTURED cuda lock must fill {empty_cuda}")
    return lock


def hash_regen_env_lock(lock):
    """Canonical sha-256 over the validated lock (sorted keys, allow_nan=False, compact)."""
    validate_regen_env_lock(lock)
    return hashlib.sha256(json.dumps(lock, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def schema_only_template(*, protocol_commit, pipeline_config_sha256, device_kind="cpu"):
    """A SCHEMA-ONLY lock skeleton (status SCHEMA_ONLY_NOT_CAPTURED, placeholder versions) — passes the schema validator
    but is REJECTED by run_regen_substrate (which requires CAPTURED_AND_VERIFIED). The real capture step (torch import on
    the training node) is gated to B1; this only fixes the structure to review."""
    if device_kind not in DEVICE_KINDS:
        raise ValueError(f"device_kind must be one of {DEVICE_KINDS}")
    return {
        "schema_version": SCHEMA_VERSION, "status": "SCHEMA_ONLY_NOT_CAPTURED",
        "python_version": "", "torch_version": "", "braindecode_version": "", "numpy_version": "",
        "scipy_version": "", "sklearn_version": "", "cuda_version": "", "cudnn_version": "",
        "device_kind": device_kind, "device_name": "", "driver_version": "",
        "torch_deterministic_algorithms": True, "seed": 0,
        "torch_intraop_threads": 1, "torch_interop_threads": 1, "omp_num_threads": 1, "threadpool_backends": [],
        "pipeline_config_sha256": pipeline_config_sha256, "protocol_commit": protocol_commit,
    }
