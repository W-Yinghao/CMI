"""Deterministic RNG contract for future C85T execution.

C85TL may exercise this module only with ``SHADOW_*`` scenario identifiers.
Registered S0-S10 streams require an execution-time authorization token.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import sys
from typing import Final, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError


REPO_ROOT: Final = Path(__file__).resolve().parents[2]
EXPECTED_PREFIX: Final = Path(
    "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact"
)
EXPECTED_PYTHON: Final = "3.13.7"
EXPECTED_NUMPY_RUNTIME: Final = "2.4.4"
EXPECTED_NUMPY_METADATA_FIRST_MATCH: Final = "2.3.3"
REGISTERED_SCENARIOS: Final = tuple(f"S{index}" for index in range(11))
SHADOW_SCENARIOS: Final = (
    "SHADOW_NORMAL_A",
    "SHADOW_RADEMACHER_A",
    "SHADOW_RADEMACHER_B",
)
SEED_NAMESPACE: Final = "C85_SYNTHETIC_V1"
REGISTERED_EXECUTION_TOKEN: Final = "C85T_LOCKED_EXECUTION_AUTHORIZATION_REPLAYED"

EXPECTED_NUMPY_FILES: Final = {
    "lib/python3.13/site-packages/numpy/__init__.py": (
        25946,
        "2e8da3e4385e79c4885b3f7324a8b957e6f01732b239e99e266a12c62a008b8d",
    ),
    "lib/python3.13/site-packages/numpy/version.py": (
        293,
        "0210aea664834faf54cdd051807f66d751d087e26489bea49f407f6f8e8790dc",
    ),
    "lib/python3.13/site-packages/numpy/random/__init__.py": (
        7480,
        "585ce7b73b5454d6a25c2a50967f2dc322fc1d214d4bb5c0589949b105e06ea9",
    ),
    "lib/python3.13/site-packages/numpy/random/_generator.cpython-313-x86_64-linux-gnu.so": (
        809408,
        "7bf16558f84b1ab33f74564b00b1dc22b7b5c98dbdf84d4080d098a55a0ac3e2",
    ),
    "lib/python3.13/site-packages/numpy/random/_pcg64.cpython-313-x86_64-linux-gnu.so": (
        141216,
        "2524bfa5f1382e4876a02b7bb2db1478e91928b656b7b10081a0ab75cb77277c",
    ),
    "lib/python3.13/site-packages/numpy/random/bit_generator.cpython-313-x86_64-linux-gnu.so": (
        217952,
        "d2e9e7c44224dd62fd08166ca0f8dea28f3aed2553216eab79335a2a60ca549d",
    ),
    "lib/python3.13/site-packages/numpy/random/_common.cpython-313-x86_64-linux-gnu.so": (
        257624,
        "2ccca5b02271b61cfe7501982719fd73fbb272d1153f8485de406a6805a0926d",
    ),
    "lib/python3.13/site-packages/numpy-2.3.3.dist-info/METADATA": (
        62117,
        "661ed2938131e161a5f723099c7236e709e705dec856d1ef37527d37d0ac4dc6",
    ),
    "lib/python3.13/site-packages/numpy-2.3.3.dist-info/RECORD": (
        108814,
        "3da1ea45333b7e59d8cb6cda47a242da92916050125db222881901b0bfcebae6",
    ),
    "lib/python3.13/site-packages/numpy-2.4.4.dist-info/METADATA": (
        6608,
        "8f9e04338ffa930868a2d157ce44f6e4cb7ac524b968fb13d04187a6ac575282",
    ),
    "lib/python3.13/site-packages/numpy-2.4.4.dist-info/RECORD": (
        110452,
        "61b4277a1cb01c55aad8dd9d7df01c8fbd8dec62650d2101902e651dfb37187f",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(
    scenario_id: str,
    replicate_id: int,
    *,
    execution_token: str | None = None,
) -> int:
    """Return the locked low-64 little-endian SHA-256 seed.

    Registered streams are deliberately inaccessible to readiness tests.
    """

    if not isinstance(replicate_id, int) or not 0 <= replicate_id <= 4095:
        raise DecisionContractError("replicate ID must be an integer in 0..4095")
    if scenario_id in REGISTERED_SCENARIOS:
        if execution_token != REGISTERED_EXECUTION_TOKEN:
            raise DecisionContractError(
                "registered S0-S10 RNG requires consumed C85T authorization"
            )
    elif scenario_id not in SHADOW_SCENARIOS:
        raise DecisionContractError("unknown C85T RNG scenario identifier")
    payload = f"{SEED_NAMESPACE}|{scenario_id}|{replicate_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def generator(
    scenario_id: str,
    replicate_id: int,
    *,
    execution_token: str | None = None,
) -> np.random.Generator:
    seed = deterministic_seed(
        scenario_id, replicate_id, execution_token=execution_token
    )
    return np.random.Generator(np.random.PCG64DXSM(seed))


def draw_standard_normal(
    scenario_id: str,
    replicate_id: int,
    action_count: int,
    *,
    execution_token: str | None = None,
) -> np.ndarray:
    if not isinstance(action_count, int) or action_count <= 0:
        raise DecisionContractError("action count must be positive")
    values = generator(
        scenario_id, replicate_id, execution_token=execution_token
    ).standard_normal(action_count, dtype=np.float64)
    return np.asarray(values, dtype="<f8")


def draw_s9_rademacher_prefixes(
    scenario_id: str,
    replicate_id: int,
    *,
    execution_token: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw 51 L values followed by 46 H values from one generator."""

    rng = generator(scenario_id, replicate_id, execution_token=execution_token)
    low = rng.integers(0, 2, size=51, dtype=np.uint8)
    high = rng.integers(0, 2, size=46, dtype=np.uint8)
    low = (2 * low.astype(np.int8) - 1).astype(np.int8, copy=False)
    high = (2 * high.astype(np.int8) - 1).astype(np.int8, copy=False)
    return low, high


def validate_environment(*, strict_prefix: bool = True) -> dict[str, object]:
    """Fail closed on runtime, metadata, or bound NumPy-byte drift."""

    observed_prefix = Path(sys.prefix).resolve()
    if strict_prefix and observed_prefix != EXPECTED_PREFIX.resolve():
        raise DecisionContractError(
            f"C85T environment prefix drift: {observed_prefix}"
        )
    observed_python = ".".join(map(str, sys.version_info[:3]))
    if observed_python != EXPECTED_PYTHON:
        raise DecisionContractError(f"C85T Python drift: {observed_python}")
    if np.__version__ != EXPECTED_NUMPY_RUNTIME:
        raise DecisionContractError(f"C85T NumPy runtime drift: {np.__version__}")
    first_match = metadata.version("numpy")
    if first_match != EXPECTED_NUMPY_METADATA_FIRST_MATCH:
        raise DecisionContractError(
            f"C85T NumPy metadata first-match drift: {first_match}"
        )

    rows: list[dict[str, object]] = []
    for relative, (expected_size, expected_sha) in EXPECTED_NUMPY_FILES.items():
        path = EXPECTED_PREFIX / relative
        if not path.is_file():
            raise DecisionContractError(f"bound NumPy file is absent: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if (size, digest) != (expected_size, expected_sha):
            raise DecisionContractError(f"bound NumPy file drift: {path}")
        rows.append(
            {
                "path": str(path),
                "size": size,
                "sha256": digest,
            }
        )
    return {
        "prefix": str(observed_prefix),
        "python": observed_python,
        "numpy_runtime": np.__version__,
        "numpy_metadata_first_match": first_match,
        "bit_generator": "PCG64DXSM",
        "bound_files": rows,
    }


def canonical_float64_bytes(values: Sequence[float] | np.ndarray) -> bytes:
    array = np.asarray(values, dtype="<f8")
    if not np.isfinite(array).all():
        raise DecisionContractError("nonfinite RNG output")
    return array.tobytes(order="C")


def canonical_array_sha256(values: Sequence[float] | np.ndarray) -> str:
    return hashlib.sha256(canonical_float64_bytes(values)).hexdigest()

