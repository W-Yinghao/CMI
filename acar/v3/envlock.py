"""ACAR v3 ENVIRONMENT LOCK. Pins the exact runtime (library versions, determinism flags, scipy Wilcoxon convention,
numpy quantile method, schemas, frozen constants, the seven DEV cohorts, and the full HP) for the DEV-design lock.

`build_env_lock()` is the SINGLE source of the lock dict; `notes/ACAR_V3_ENV_LOCK.json` is its frozen serialization.
`verify_env_lock()` rebuilds the dict from the CURRENT process and asserts it byte-matches the stored lock — so the
binding DEV run fails closed on any library/flag drift. The lock's `env_lock_sha256` is referenced by the frozen
manifest.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform

from acar.config import DISEASE, B, MIN_BATCH, N_CLS, RHO
from .predictors import env_versions, HP, SCHEMA_VERSION
from .loader import LOADER_SCHEMA, PROB_SCHEMA
from .data import DATA_SCHEMA

ENV_LOCK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             "notes", "ACAR_V3_ENV_LOCK.json")


_TPL = None                                                     # held threadpoolctl controller (keeps the limit alive)


def apply_runtime():
    """Apply the REQUIRED deterministic single-thread runtime to the CURRENT process (idempotent). Forces torch
    deterministic algorithms, torch intra-op threads = 1, OMP_NUM_THREADS = 1, and a global threadpoolctl limit of 1
    (so OpenBLAS/OpenMP/MKL behind numpy/scipy/sklearn run single-threaded). Inter-op threads are set best-effort (only
    settable before parallel work; not part of the verified hash). Called by the lock generator AND `run_binding_dev`
    BEFORE verifying the lock, so the binding run actually executes under the locked settings."""
    global _TPL
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)                        # best-effort (fresh process only)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits
        if _TPL is None:
            _TPL = threadpool_limits(limits=1)                  # global, kept alive at module scope
    except Exception:                                           # pragma: no cover
        pass


def _runtime_state():
    """Hashed runtime block — only what is reliably FORCEABLE and verifiable in BOTH a cold (CLI) and a warm (test)
    process after apply_runtime(): torch deterministic + intra-op threads, OMP env, and each threadpool backend's
    internal_api / num_threads / version (num_threads == 1 under the global limit). Inter-op is excluded (unsettable in
    a warm process)."""
    import torch
    try:
        from threadpoolctl import threadpool_info
        tp = sorted(({"internal_api": str(i.get("internal_api")), "num_threads": int(i.get("num_threads")),
                      "version": str(i.get("version"))} for i in threadpool_info()),
                    key=lambda d: (d["internal_api"], d["version"]))
    except Exception:                                           # pragma: no cover
        tp = [{"internal_api": "UNAVAILABLE", "num_threads": -1, "version": ""}]
    return {"torch_deterministic": bool(torch.are_deterministic_algorithms_enabled()),
            "torch_num_threads": int(torch.get_num_threads()),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "threadpool": tp}


def build_env_lock() -> dict:
    """Deterministic lock payload reflecting the CURRENT process runtime (env_lock_sha256 last, over the rest)."""
    lock = {
        "env_versions": env_versions(),
        "platform": platform.platform(),
        "runtime": _runtime_state(),
        "scipy_wilcoxon": {"exact_max_n": 25, "n_perm": 20000, "seed": 0,
                           "method": "PermutationMethod", "continuity": "wilcox zero_method"},
        "numpy_quantile_method": "linear",
        "schemas": {"pred": SCHEMA_VERSION, "loader": LOADER_SCHEMA, "prob": PROB_SCHEMA, "data": DATA_SCHEMA},
        "frozen_constants": {"B": B, "MIN_BATCH": MIN_BATCH, "N_CLS": N_CLS, "RHO": RHO},
        "cohorts": DISEASE,
        "HP": {k: (list(v) if isinstance(v, tuple) else v) for k, v in HP.items()},
    }
    blob = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    lock["env_lock_sha256"] = hashlib.sha256(blob).hexdigest()
    return lock


def env_lock_sha256() -> str:
    return build_env_lock()["env_lock_sha256"]


def write_env_lock(path=ENV_LOCK_PATH) -> str:
    apply_runtime()                                              # record the lock UNDER the required runtime
    lock = build_env_lock()
    with open(path, "w") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write("\n")
    return lock["env_lock_sha256"]


def load_env_lock(path=ENV_LOCK_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def verify_env_lock(path=ENV_LOCK_PATH):
    """FAIL-CLOSED: the running process must reproduce the stored lock exactly (versions, runtime threads, schemas, HP,
    cohorts). Applies the required runtime first, so the threadpool limit is in effect when the state is read. Returns
    the verified env_lock_sha256."""
    apply_runtime()
    stored = load_env_lock(path)
    current = build_env_lock()
    if current != stored:
        diffs = [k for k in set(current) | set(stored) if current.get(k) != stored.get(k)]
        raise ValueError(f"environment-lock mismatch (drift in {sorted(diffs)}); refusing the binding DEV run")
    return stored["env_lock_sha256"]
