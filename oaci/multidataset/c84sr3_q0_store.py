"""C84SR3 Q0 storage adapter with availability-safe dataset budget grids."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import c84sr1_q0_store as historical
from .c84sr1_common import Q0_CHAINS
from .c84sr3_common import finite_budgets


SCHEMA_VERSION = "c84sr3_q0_context_shard_v2"
SHARD_INDEX_FIELDS = historical.SHARD_INDEX_FIELDS


def build_context_payload(**kwargs: Any) -> dict[str, np.ndarray]:
    return historical.build_context_payload(
        **kwargs, budget_provider=finite_budgets, schema_version=SCHEMA_VERSION,
    )


def validate_payload(
    payload: Mapping[str, np.ndarray], *, chains: int = Q0_CHAINS,
) -> dict[str, Any]:
    return historical.validate_payload(
        payload, chains=chains, budget_provider=finite_budgets,
        schema_version=SCHEMA_VERSION,
    )


def write_context_shard(
    path: str | Path, payload: Mapping[str, np.ndarray], *, chains: int = Q0_CHAINS,
) -> dict[str, Any]:
    return historical.write_context_shard(
        path, payload, chains=chains, budget_provider=finite_budgets,
        schema_version=SCHEMA_VERSION,
    )


def read_context_shard(
    path: str | Path, *, expected_sha256: str | None = None,
    chains: int = Q0_CHAINS,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return historical.read_context_shard(
        path, expected_sha256=expected_sha256, chains=chains,
        budget_provider=finite_budgets, schema_version=SCHEMA_VERSION,
    )


def synthetic_payload(
    identity: Mapping[str, Any], candidate_ids: Sequence[str], *,
    chains: int = Q0_CHAINS,
) -> dict[str, np.ndarray]:
    return historical.synthetic_payload(
        identity, candidate_ids, chains=chains, budget_provider=finite_budgets,
        schema_version=SCHEMA_VERSION,
    )
