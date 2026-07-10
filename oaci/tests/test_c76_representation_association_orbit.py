from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from oaci.conditioned_ceiling_coverage import c76_protocol


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_c76_parent_and_execution_boundary_are_locked():
    assert c76_protocol.PARENT_COMMIT.startswith("fb8a412")
    assert c76_protocol.NULL_REPLICATES == 499
    assert c76_protocol.ORBIT_REPLICATES == 4
    assert c76_protocol.KERNEL_FAMILIES == ("rbf", "laplacian")


def test_c76_registered_orbits_cover_global_and_checkpoint_GL_families():
    rows = c76_protocol.orbit_registry()
    assert [row["orbit"] for row in rows] == [f"O{index}" for index in range(8)]
    assert {row["scope"] for row in rows} == {"global", "checkpoint"}
    assert {row["family"] for row in rows} >= {"orthogonal", "diagonal", "nonorthogonal", "signed_permutation"}
    assert all(float(row["condition_bound"]) <= 3.0 for row in rows)


def test_c76_kernel_family_and_null_family_are_frozen():
    kernels = c76_protocol.kernel_registry()
    assert len(kernels) == 24
    assert {row["path"] for row in kernels} == {"strict_source", "target_unlabeled"}
    assert all(row["null_reselects_bandwidth"] == 1 for row in kernels)
    nulls = c76_protocol.null_registry()
    assert len(nulls) == 6
    assert all(row["required_for_candidate"] == 1 for row in nulls)


def test_c76_candidate_gate_requires_prediction_actionability_and_orbit_robustness():
    rows = c76_protocol.qualification_registry()
    assert len(rows) == 24
    for candidate in {row["candidate"] for row in rows}:
        gates = {row["gate"] for row in rows if row["candidate"] == candidate}
        assert {"orbit_robustness", "incremental_R2", "global_max_stat_p", "material_actionability", "target_label_leakage"} <= gates
        assert all(row["all_required"] == 1 for row in rows if row["candidate"] == candidate)
