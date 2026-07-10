from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from oaci.conditioned_ceiling_coverage import c75_protocol
from oaci.conditioned_ceiling_coverage import c75_data
from oaci.conditioned_ceiling_coverage import c75_modeling
from oaci.conditioned_ceiling_coverage import c75_projection
from oaci.conditioned_ceiling_coverage import synthetic_factorization_generator


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_c75_protocol_hash_and_parent_are_locked():
    protocol = json.loads(c75_protocol.PROTOCOL_PATH.read_text())
    assert _sha256(c75_protocol.PROTOCOL_PATH) == c75_protocol.PROTOCOL_SHA_PATH.read_text().strip()
    assert protocol["parent_C74_result_commit"].startswith("fe467b9")
    assert protocol["execution_boundary"]["forward_passes"] is False
    assert protocol["execution_boundary"]["T3_HO_z_Wz_access"] is False
    assert protocol["data_role"]["T2_units"] == 216
    assert protocol["data_role"]["T3_HO_units"] == 1052


def test_c75_registered_blocks_are_low_dimensional_and_availability_labeled():
    rows = list(csv.DictReader(open(c75_protocol.TABLE_DIR / "feature_block_registry.csv")))
    assert [row["block"] for row in rows] == ["F0", "F1", "F2", "F3", "F4", "F5"]
    assert [int(row["dimension"]) for row in rows] == [9, 25, 25, 18, 35, 15]
    assert {row["block"] for row in rows if row["qualification_candidate"] == "1"} == {"F2", "F4"}
    availability = {row["block"]: row for row in csv.DictReader(open(c75_protocol.TABLE_DIR / "feature_availability_ledger.csv"))}
    assert availability["F2"]["available_strict_DG"] == "1"
    assert availability["F4"]["target_unlabeled"] == "1"
    assert availability["F5"]["target_label_derived"] == "1"


def test_c75_factorization_equivalence_is_exact_on_dummy_data():
    rng = np.random.default_rng(75)
    z = rng.normal(size=(64, 16))
    W = rng.normal(size=(4, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    scales = np.linspace(0.5, 2.0, 16)
    A = q @ np.diag(scales) @ q.T
    transformed_z = z @ A.T
    transformed_W = W @ np.linalg.inv(A)
    assert np.max(np.abs(transformed_z @ transformed_W.T - z @ W.T)) < 1e-10
    assert not np.allclose(np.linalg.norm(transformed_z, axis=1), np.linalg.norm(z, axis=1))


def test_c75_Wz_is_exactly_redundant_with_logits_and_bias():
    rng = np.random.default_rng(750)
    Wz = rng.normal(size=(128, 4))
    bias = rng.normal(size=4)
    logits = Wz + bias
    reconstructed = logits - bias
    assert np.max(np.abs(reconstructed - Wz)) <= 4 * np.finfo(np.float64).eps
    design = np.column_stack((reconstructed, Wz))
    assert np.linalg.matrix_rank(design) == np.linalg.matrix_rank(Wz)


def test_c75_qualification_requires_every_locked_gate():
    rows = list(csv.DictReader(open(c75_protocol.TABLE_DIR / "t3_qualification_gates.csv")))
    assert len(rows) == 14
    assert {row["candidate"] for row in rows} == {"F2_strict_source", "F4_target_unlabeled"}
    assert all(row["all_required"] == "1" for row in rows)
    assert sum(row["gate"] == "target_label_leakage" for row in rows) == 2


def test_c75_endpoint_metrics_are_oriented_and_exact_for_perfect_predictions():
    labels = np.tile(np.arange(4), 8)
    logits = np.full((len(labels), 4), -8.0)
    logits[np.arange(len(labels)), labels] = 8.0
    metrics = c75_data.endpoint_metrics(logits, labels)
    assert metrics["bAcc"] == 1.0
    assert metrics["NLL"] < 1e-5
    assert metrics["ECE"] < 1e-5
    assert np.allclose(metrics["recall"], 1.0)


def test_c75_registered_z_and_W_feature_dimensions_are_fixed():
    rng = np.random.default_rng(751)
    z = rng.normal(size=(300, 16))
    moments, spectrum = c75_data.z_features(z, np.arange(256))
    W = rng.normal(size=(4, 16))
    bias = rng.normal(size=4)
    assert moments.shape == (4,)
    assert spectrum.shape == (6,)
    assert c75_data.W_features(W, bias).shape == (10,)
    assert c75_data.alignment_features(z, W).shape == (5,)


def test_c75_projection_summary_has_one_canonical_float64_reduction_path():
    rng = np.random.default_rng(759)
    Wz = rng.normal(size=(4096, 4)).astype(np.float32)
    logits_minus_b = Wz.astype(float)
    left = c75_data.projection_summary(logits_minus_b)
    right = c75_data.projection_summary(logits_minus_b)
    assert np.array_equal(left, right)


def test_c75_column_space_ridge_is_invariant_to_exact_duplicate_columns():
    rng = np.random.default_rng(752)
    train = rng.normal(size=(80, 7))
    test = rng.normal(size=(20, 7))
    response = rng.normal(size=80)
    duplicated_train = np.column_stack((train, train[:, 2:5]))
    duplicated_test = np.column_stack((test, test[:, 2:5]))
    reference, audit_reference = c75_modeling.ridge_fold_predict(
        train, response, test, alpha=1.0, column_space=True,
    )
    duplicated, audit_duplicated = c75_modeling.ridge_fold_predict(
        duplicated_train, response, duplicated_test, alpha=1.0, column_space=True,
    )
    assert audit_reference["rank"] == audit_duplicated["rank"]
    assert np.max(np.abs(reference - duplicated)) < 1e-10


def test_c75_blocked_permutations_never_cross_registered_groups():
    targets = np.repeat(np.arange(2), 12)
    trajectories = np.tile(np.repeat(np.asarray(["a", "b", "c"]), 4), 2)
    permutation = c75_modeling.blocked_permutation_indices(
        targets, trajectories, np.random.default_rng(753), within_trajectory=True,
    )
    assert np.array_equal(targets, targets[permutation])
    assert np.array_equal(trajectories, trajectories[permutation])
    target_only = c75_modeling.blocked_permutation_indices(
        targets, trajectories, np.random.default_rng(754), within_trajectory=False,
    )
    assert np.array_equal(targets, targets[target_only])


def test_c75_kernel_bandwidth_is_estimated_from_outer_training_targets():
    rng = np.random.default_rng(758)
    targets = np.repeat(np.arange(4), 8)
    features = rng.normal(size=(32, 3))
    residual = rng.normal(size=32)
    statistic, bandwidths = c75_modeling.crossfit_kernel_alignment_statistic(
        features, residual, targets, 1.0,
    )
    assert np.isfinite(statistic)
    assert len(bandwidths) == 4
    assert all(value > 0 for value in bandwidths)
    rescaled, rescaled_bandwidths = c75_modeling.crossfit_kernel_alignment_statistic(
        features * np.asarray([1.0, 10.0, 100.0]) + np.asarray([2.0, -4.0, 7.0]),
        residual, targets, 1.0,
    )
    assert abs(statistic - rescaled) < 1e-12
    assert np.allclose(bandwidths, rescaled_bandwidths, atol=1e-12)


def test_c75_projection_variance_estimand_accounts_to_one():
    rng = np.random.default_rng(755)
    payloads = {
        target: [{"Wz": rng.normal(size=(32, 4))} for _ in range(5)]
        for target in (1, 2, 3)
    }
    result = c75_projection.variance_audit(payloads, bootstrap_repeats=20)
    assert len(result["by_target_class"]) == 12
    assert all(abs(row["accounting_sum"] - 1.0) < 1e-12 for row in result["by_target_class"])
    assert all(row["causal_interpretation"] == 0 for row in result["by_target_class"])


def test_c75_synthetic_benchmark_separates_incremental_from_null_cases():
    rows, summary = synthetic_factorization_generator.construct_validity_benchmark(
        replicates=30, seed=756,
    )
    rates = {row["case"]: row["detection_rate"] for row in summary}
    assert len(rows) == 90
    assert rates["incremental_representation"] > rates["stable_endpoint_irrelevant"]
    assert rates["incremental_representation"] > rates["functionally_redundant"]


def test_c75_reparameterization_audit_preserves_function_not_coordinates():
    rows = synthetic_factorization_generator.factorization_reparameterization_audit(seed=757)
    assert all(row["function_invariant"] == 1 for row in rows)
    nonorthogonal = next(row for row in rows if row["transform"] == "nonorthogonal_condition_le_4")
    assert nonorthogonal["coordinate_geometry_invariant"] == 0
    assert nonorthogonal["Wz_max_abs_error"] < 1e-10
