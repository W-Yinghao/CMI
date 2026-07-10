from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from oaci.conditioned_ceiling_coverage import c75_modeling
from oaci.conditioned_ceiling_coverage import c76_orbit
from oaci.conditioned_ceiling_coverage import c76_protocol
from oaci.conditioned_ceiling_coverage import c76_representation_association_orbit
from oaci.conditioned_ceiling_coverage import c76_statistics
from oaci.conditioned_ceiling_coverage import synthetic_association_generator


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_c76_parent_and_execution_boundary_are_locked():
    assert c76_protocol.PARENT_COMMIT.startswith("fb8a412")
    assert c76_protocol.NULL_REPLICATES == 499
    assert c76_protocol.ORBIT_REPLICATES == 4
    assert c76_protocol.KERNEL_FAMILIES == ("rbf", "laplacian")


def test_c76_protocol_hash_and_no_forward_boundary_replay():
    protocol = json.loads(c76_protocol.PROTOCOL_PATH.read_text())
    assert _sha256(c76_protocol.PROTOCOL_PATH) == c76_protocol.PROTOCOL_SHA_PATH.read_text().strip()
    assert protocol["parent_C75_result_commit"].startswith("fb8a412")
    assert protocol["execution_boundary"]["forward_passes"] is False
    assert protocol["execution_boundary"]["training"] is False
    assert protocol["execution_boundary"]["T3_HO_z_Wz_access"] is False
    assert protocol["data_role"]["T2_units"] == 216
    assert protocol["data_role"]["T3_HO_units"] == 1052


def test_c76_locked_registry_hashes_replay():
    protocol = json.loads(c76_protocol.PROTOCOL_PATH.read_text())
    for item in protocol["locked_registry_tables"].values():
        path = Path(item["path"])
        assert _sha256(path) == item["sha256"]
        assert path.stat().st_size == item["size_bytes"]


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


def test_c76_target_candidate_excludes_function_invariant_Wz_tail():
    F4 = np.arange(7 * 35, dtype=float).reshape(7, 35)
    geometry, invariant = c76_representation_association_orbit._target_feature_partition(F4)
    assert geometry.shape == (7, 20)
    assert invariant.shape == (7, 15)
    assert np.array_equal(np.concatenate((geometry, invariant), axis=1), F4)
    assert not np.shares_memory(geometry, invariant)


def test_c76_target_partition_rejects_dimension_drift():
    with np.testing.assert_raises(RuntimeError):
        c76_representation_association_orbit._target_feature_partition(np.zeros((4, 34)))


def test_c76_candidate_gate_requires_prediction_actionability_and_orbit_robustness():
    rows = c76_protocol.qualification_registry()
    assert len(rows) == 24
    for candidate in {row["candidate"] for row in rows}:
        gates = {row["gate"] for row in rows if row["candidate"] == candidate}
        assert {"orbit_robustness", "incremental_R2", "global_max_stat_p", "material_actionability", "target_label_leakage"} <= gates
        assert all(row["all_required"] == 1 for row in rows if row["candidate"] == candidate)


def test_c76_all_registered_orbits_preserve_Wz_on_dummy_data():
    rng = np.random.default_rng(76)
    z = rng.normal(size=(16, c76_orbit.DIMENSION))
    W = rng.normal(size=(4, c76_orbit.DIMENSION))
    reference = z @ W.T
    for orbit, replicate in c76_orbit.orbit_variants():
        transform = c76_orbit.make_transform(orbit, replicate, "unit-a")
        observed = c76_orbit.apply_z(z, transform) @ c76_orbit.apply_W(W, transform).T
        assert np.max(np.abs(observed - reference)) < 1e-10
        assert transform.condition_number <= 3.0 + 1e-12


def test_c76_global_and_checkpoint_transform_scope_hashes_are_distinct():
    global_left = c76_orbit.make_transform("O5", 0, "unit-a")
    global_right = c76_orbit.make_transform("O5", 0, "unit-b")
    checkpoint_left = c76_orbit.make_transform("O6", 0, "unit-a")
    checkpoint_right = c76_orbit.make_transform("O6", 0, "unit-b")
    assert global_left.transform_hash == global_right.transform_hash
    assert checkpoint_left.transform_hash != checkpoint_right.transform_hash


def test_c76_blocked_permutations_preserve_registered_strata():
    targets = np.repeat(np.arange(3), 8)
    trajectory = np.tile(np.repeat(np.asarray(["a", "b"]), 4), 3)
    seed = np.tile(np.repeat(np.arange(2), 4), 3)
    level = np.tile(np.repeat(np.arange(2), 2), 6)
    order = np.tile(np.arange(8), 3)
    for scheme in ("N3_trajectory_preserving", "N4_candidate_within_target"):
        permutation = c76_statistics.blocked_permutation(
            scheme, targets, trajectory, seed, level, order,
            np.random.default_rng(761),
        )
        assert np.array_equal(targets, targets[permutation])
        if scheme == "N3_trajectory_preserving":
            assert np.array_equal(trajectory, trajectory[permutation])


def test_c76_registered_association_detects_nonlinear_within_target_signal():
    rng = np.random.default_rng(762)
    targets = np.repeat(np.arange(6), 24)
    x = rng.normal(size=(len(targets), 3))
    y = x[:, 0] ** 2 - 1.0 + 0.15 * rng.normal(size=len(targets))
    observed, _ = c76_statistics.crossfit_association(
        x, y, targets, kernel_family="rbf", bandwidth_factor=1.0,
        statistic="centered_hsic",
    )
    permuted = y.copy()
    for target in range(6):
        mask = targets == target
        permuted[mask] = rng.permutation(permuted[mask])
    null, _ = c76_statistics.crossfit_association(
        x, permuted, targets, kernel_family="rbf", bandwidth_factor=1.0,
        statistic="centered_hsic",
    )
    assert observed > null


def test_c76_rbf_alignment_preserves_c75_arithmetic_order_exactly():
    rng = np.random.default_rng(765)
    targets = np.repeat(np.arange(4), 12)
    features = rng.normal(size=(48, 7))
    outcome = rng.normal(size=48)
    c75_value, _ = c75_modeling.crossfit_kernel_alignment_statistic(
        features, outcome, targets, 1.0,
    )
    c76_value, _ = c76_statistics.crossfit_association(
        features, outcome, targets, kernel_family="rbf", bandwidth_factor=1.0,
        statistic="normalized_alignment",
    )
    assert c76_value == c75_value


def test_c76_kernel_ridge_crossfit_recovers_shared_nonlinear_relation():
    rng = np.random.default_rng(763)
    targets = np.repeat(np.arange(5), 16)
    x = rng.normal(size=(len(targets), 2))
    y = np.sin(x[:, 0]) + 0.05 * rng.normal(size=len(targets))
    result = c76_statistics.crossfit_krr(x, y, targets)
    centered = c76_statistics.center_within_groups(y[:, None], targets)[:, 0]
    assert 1.0 - np.sum((centered - result.prediction) ** 2) / np.sum(centered ** 2) > 0.5


def test_c76_synthetic_benchmark_separates_null_and_actionable_cases():
    rows, summary = synthetic_association_generator.run_benchmark(replicates=8, seed=764)
    by_case = {row["case"]: row for row in summary}
    assert len(rows) == 8 * 7
    assert by_case["S6_predictive_actionable"]["median_within_target_association"] > by_case["S0_no_association"]["median_within_target_association"]
    assert by_case["S6_predictive_actionable"]["median_incremental_R2"] > by_case["S0_no_association"]["median_incremental_R2"]
    assert by_case["S6_predictive_actionable"]["mean_top1_increment"] > by_case["S0_no_association"]["mean_top1_increment"]
    assert by_case["S4_factorization_invariant_endpoint"]["median_orbit_effect_retention"] == 1.0
    assert by_case["S5_association_no_extreme_action"]["median_within_target_association"] > by_case["S0_no_association"]["median_within_target_association"]
    assert abs(by_case["S5_association_no_extreme_action"]["mean_top1_increment"]) < 0.1
