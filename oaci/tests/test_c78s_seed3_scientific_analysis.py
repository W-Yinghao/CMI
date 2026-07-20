from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c78s_data as data
from oaci.conditioned_ceiling_coverage import c78s_modeling as modeling
from oaci.conditioned_ceiling_coverage import c78s_protocol as protocol
from oaci.conditioned_ceiling_coverage import c78s_seed3_scientific_analysis as analysis


def test_protocol_hash_is_authoritative():
    assert protocol.sha256_file(protocol.PROTOCOL_PATH) == protocol.PROTOCOL_SHA_PATH.read_text().strip()
    assert protocol.PROTOCOL_SHA_PATH.read_text().strip() == "df85699090a65d1e1766d754bcebd9eb5648cc13e4441d8074a3f4884487c7f8"


def test_primary_targets_exclude_canary():
    assert protocol.PRIMARY_TARGETS == (1, 2, 3, 5, 6, 7, 8, 9)
    assert 4 not in protocol.PRIMARY_TARGETS


def test_seed_and_field_counts_are_locked():
    assert protocol.SEED == 3
    assert protocol.PRIMARY_UNITS == 1296
    assert protocol.FULL_FIELD_UNITS == 1458


def test_registered_resampling_counts():
    assert protocol.NULL_REPLICATES == 499
    assert protocol.BOOTSTRAP_REPLICATES == 2000
    assert protocol.TRIAL_BOOTSTRAP_REPLICATES == 499


def test_prefix_grid_and_geometry_epsilon():
    assert protocol.PREFIX_SIZES == (5, 10, 20, 40)
    assert protocol.PRIMARY_GEOMETRY_EPSILON == 0.05


def test_feature_registry_exact_blocks_and_dimensions():
    registry = protocol.feature_registry()
    assert [row["block"] for row in registry] == [f"F{i}" for i in range(6)]
    assert [row["dimension"] for row in registry] == [9, 25, 25, 18, 35, 15]


def test_trial_id_is_never_registered_as_predictor():
    assert all(row["predictor_trial_id"] == 0 for row in protocol.feature_registry())


def test_only_F5_has_target_labels():
    labeled = [row["block"] for row in protocol.feature_registry() if row["target_labels"]]
    assert labeled == ["F5"]


def test_risk_registry_has_no_duplicate_keys():
    risks = protocol.risk_registry()
    assert len(risks) >= 24
    assert len({row["risk"] for row in risks}) == len(risks)


def test_primary_paths_are_predeclared():
    assert set(analysis.PRIMARY_PATHS) == {
        "strict_source_F2",
        "target_unlabeled_F4_geometry",
        "target_unlabeled_F4_full_mixed",
        "construction_F5_positive",
    }


def test_F4_primary_partition_is_twenty_plus_fifteen():
    arrays = {"F4": np.zeros((3, 35)), "unit_id": np.asarray(["a", "b", "c"]), "target_id": np.asarray([1, 2, 3])}
    with pytest.raises(RuntimeError):
        analysis._prepare_arrays(arrays)
    arrays = {
        "F4": np.zeros((1296, 35)),
        "unit_id": np.asarray([f"u{i}" for i in range(1296)]),
        "target_id": np.repeat(protocol.PRIMARY_TARGETS, 162),
    }
    prepared = analysis._prepare_arrays(arrays)
    assert prepared["F4_geometry"].shape == (1296, 20)
    assert prepared["F4_functional_projection"].shape == (1296, 15)


def test_center_within_groups_zeroes_group_means():
    values = np.asarray([[1.0], [3.0], [4.0], [10.0]])
    groups = np.asarray(["a", "a", "b", "b"])
    centered = modeling.center_within_groups(values, groups)
    assert np.allclose(centered[:2].mean(axis=0), 0)
    assert np.allclose(centered[2:].mean(axis=0), 0)


def test_centered_r2_is_one_for_exact_prediction():
    y = np.asarray([1.0, 2.0, 4.0, 7.0])
    groups = np.asarray(["a", "a", "b", "b"])
    prediction = modeling.center_within_groups(y[:, None], groups)[:, 0]
    assert modeling.centered_r2(y, prediction, groups) == pytest.approx(1.0)


def test_safe_spearman_handles_constant_vector():
    assert np.isnan(modeling.safe_spearman(np.ones(4), np.arange(4)))


def test_pairwise_accuracy_perfect_order():
    assert modeling.pairwise_accuracy(np.arange(5.0), np.arange(5.0)) == 1.0


def test_holm_adjust_matches_known_example():
    rows = [{"raw_p": value} for value in (0.01, 0.03, 0.04)]
    adjusted = modeling.holm_adjust(rows)
    assert [row["Holm_p"] for row in adjusted] == pytest.approx([0.03, 0.06, 0.06])


def test_exact_sign_flip_detects_all_positive_eight_target_effects():
    assert modeling.exact_sign_flip_p(np.ones(8)) < 0.05


def test_cell_actionability_has_random_context():
    utility = np.asarray([0.1, 0.9, 0.4, 0.8])
    good = np.asarray([0, 1, 0, 1])
    score = np.asarray([0.0, 1.0, 0.2, 0.8])
    rows = modeling.cell_actionability(
        utility, good, {"score": score},
        np.asarray(["a"] * 4), np.asarray([1] * 4), np.asarray([0] * 4),
    )
    assert rows[0]["random_top1"] == 0.25
    assert rows[0]["score_oracle_best_in_predicted_top1"] == 1
    assert rows[0]["score_regret"] == 0


def test_actionability_summary_keeps_prediction_and_regret_separate():
    utility = np.asarray([0.1, 0.9, 0.4, 0.8])
    good = np.asarray([0, 1, 0, 1])
    rows = modeling.cell_actionability(
        utility, good,
        {"prior": np.asarray([1.0, 0.0, 0.0, 0.0]), "full": utility},
        np.asarray(["a"] * 4), np.asarray([1] * 4), np.asarray([0] * 4),
    )
    summary = modeling.summarize_actionability(rows, "prior", "full")
    assert summary["standardized_regret_reduction"] > 0
    assert summary["delta_oracle_best_in_predicted_top1"] == 1


def _permutation_arrays():
    rows = []
    for target in (1, 2):
        for level in (0, 1):
            for regime, orders in (("ERM", (0,)), ("OACI", (1, 2)), ("SRC", (1, 2))):
                for order in orders:
                    rows.append((target, level, regime, order, f"t{target}|l{level}|{regime}"))
    return {
        "target_id": np.asarray([row[0] for row in rows]),
        "level": np.asarray([row[1] for row in rows]),
        "regime": np.asarray([row[2] for row in rows]),
        "candidate_order": np.asarray([row[3] for row in rows]),
        "trajectory_id": np.asarray([row[4] for row in rows]),
    }


@pytest.mark.parametrize("scheme", [
    "target_block_permutation",
    "checkpoint_block_permutation",
    "trajectory_preserving_permutation",
    "candidate_within_target_regime_permutation",
    "nested_bandwidth_null",
])
def test_blocked_permutations_are_bijections(scheme):
    arrays = _permutation_arrays()
    permutation = modeling.blocked_permutation(scheme, arrays, np.random.default_rng(9))
    assert sorted(permutation.tolist()) == list(range(len(permutation)))


def test_target_block_permutation_preserves_template():
    arrays = _permutation_arrays()
    permutation = modeling.blocked_permutation("target_block_permutation", arrays, np.random.default_rng(5))
    for index, source in enumerate(permutation):
        assert arrays["level"][index] == arrays["level"][source]
        assert arrays["regime"][index] == arrays["regime"][source]
        assert arrays["candidate_order"][index] == arrays["candidate_order"][source]


def test_trajectory_permutation_does_not_cross_trajectory():
    arrays = _permutation_arrays()
    permutation = modeling.blocked_permutation("trajectory_preserving_permutation", arrays, np.random.default_rng(2))
    assert np.array_equal(arrays["trajectory_id"], arrays["trajectory_id"][permutation])


def test_association_statistic_detects_matching_kernel():
    y = np.asarray([-1.0, -0.5, 0.5, 1.0])
    kernel = np.outer(y, y)
    assert modeling.association_statistic(kernel, y, "normalized_alignment") > 0.9


def test_centered_hsic_is_finite():
    y = np.asarray([-1.0, -0.5, 0.5, 1.0])
    kernel = np.exp(-((y[:, None] - y[None, :]) ** 2))
    assert np.isfinite(modeling.association_statistic(kernel, y, "centered_hsic"))


def test_crossfit_ridge_returns_every_row():
    targets = np.repeat(np.arange(4), 5)
    cells = targets.astype(str)
    X = np.column_stack((np.arange(20), np.arange(20) ** 2))
    y = X[:, 0] * 0.1
    result = modeling.crossfit_ridge(
        X, y, outer_groups=targets, inner_groups=targets, center_groups=cells,
    )
    assert result.prediction.shape == y.shape
    assert len(result.fold_rows) == 4


def test_crossfit_logistic_deviance_reports_improvement():
    targets = np.repeat(np.arange(4), 20)
    x = np.tile(np.linspace(-2, 2, 20), 4)
    y = (x > 0).astype(float)
    raw = np.zeros((len(x), 1))
    full = np.column_stack((np.zeros(len(x)), x, x ** 2))
    result = modeling.crossfit_logistic_deviance(raw, full, y, targets)
    assert result["incremental_deviance_reduction"] > 0


def test_rank_utility_is_bounded():
    metrics = np.asarray([
        [0.4, 1.2, 0.2],
        [0.8, 0.7, 0.1],
        [0.6, 0.9, 0.3],
    ])
    utility = analysis._rank_utility(metrics)
    assert np.all((utility >= 0) & (utility <= 1))


def test_batch_endpoint_perfect_predictions():
    labels = np.asarray([0, 1, 2, 3])
    probabilities = np.eye(4)[labels][None, :, :]
    prediction = labels[None, :]
    metrics = analysis._batch_endpoint(probabilities, prediction, np.arange(4), labels)
    assert metrics[0, 0] == 1.0
    assert metrics[0, 1] == 0.0
    assert metrics[0, 2] == 0.0


def test_outcome_names_are_five_registered_endpoints():
    assert data.OUTCOME_NAMES == (
        "continuous_joint_utility", "target_bAcc", "neg_target_NLL",
        "neg_target_ECE", "primary_joint_good",
    )


def test_main_analysis_has_no_torch_or_EEG_loader_import():
    tree = ast.parse(Path(analysis.__file__).read_text())
    modules = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    modules.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "torch" not in modules
    assert "moabb" not in modules
    assert "mne" not in modules


def test_primary_route_source_contains_oracle_rejection():
    source = Path(protocol.__file__).read_text()
    assert '"same_label_oracle_view" in serialized' in source
    assert '"/oracle/" in serialized' in source


def test_execution_lock_scope_explicitly_blocks_forbidden_work():
    source = Path(protocol.__file__).read_text()
    for key in ("training", "forward", "reinference", "GPU", "seed4", "C79", "BNCI2014_004", "manuscript"):
        assert f'"{key}": False' in source


def test_slurm_analysis_is_cpu_high_48_core():
    script = Path("oaci/slurm_c78s_analysis.sh").read_text()
    assert "#SBATCH --partition=cpu-high" in script
    assert "#SBATCH --cpus-per-task=48" in script
    assert "--gres=gpu" not in script


def test_slurm_analysis_uses_single_registered_entrypoint():
    script = Path("oaci/slurm_c78s_analysis.sh").read_text()
    assert "c78s_seed3_scientific_analysis run" in script
    assert "torchrun" not in script


def test_c79_protocol_path_is_distinct_and_not_execution_artifact():
    assert analysis.C79_PROTOCOL_PATH.name == "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.json"
    assert "execution" not in analysis.C79_PROTOCOL_PATH.name.lower()


def test_report_does_not_use_deployable_claim_language_as_verdict():
    source = Path(analysis.__file__).read_text()
    assert "does not claim universal" in source
    assert "C79 is protocol-ready only. It remains unauthorized." in source


def test_no_raw_cache_paths_are_declared_as_git_artifacts():
    implementation = set(protocol.IMPLEMENTATION_FILES)
    assert all("/projects/" not in path for path in implementation)
    assert all(not path.endswith(".npz") for path in implementation)


def test_authorization_is_direct_and_scope_bound():
    assert protocol.AUTHORIZATION_MODE == "direct_explicit_user_authorization"
    assert len(protocol.AUTHORIZATION_EVIDENCE_SHA256) == 64


def test_route_contract_has_exact_four_nonoracle_names_when_present():
    if not protocol.ROUTE_PATH.exists():
        pytest.skip("route is generated in the prospective lock step")
    route = json.loads(protocol.ROUTE_PATH.read_text())
    expected = {
        "instrumentation_manifest",
        "strict_source_input",
        "target_unlabeled_input",
        "target_construction_view",
        "target_evaluation_view",
    }
    assert all(set(views) == expected for views in route["views"].values())
    raw = protocol.ROUTE_PATH.read_text()
    assert "same_label_oracle_view" not in raw
    assert "/oracle/" not in raw
