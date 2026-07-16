from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oaci.multidataset import c84s_selection_freeze as freeze
from oaci.multidataset import c84s_evaluation as evaluation
from oaci.multidataset import c84s_selectors as selectors
from oaci.multidataset.c84s_common import C84SContractError


def score_rows():
    return [{
        "dataset": "D", "target_subject_id": "1", "panel": "A",
        "training_seed": 5, "level": 0, "method_id": "U5",
        "candidate_index": index, "candidate_id": f"c{index:02d}",
        "raw_score": float(index % 7),
    } for index in range(81)]


def q0_rows():
    return [{
        "dataset": "D", "target_subject_id": "1", "panel": "A",
        "training_seed": 5, "level": 0, "chain": 0, "chain_seed": 9,
        "budget": "1", "sample_trial_id_sha256": "a" * 64,
        "sample_size": 2, "selected_candidate_index": 0,
        "selected_candidate_id": "c00",
        "top5_candidate_indices": list(range(5)),
        "top5_candidate_ids": [f"c{i:02d}" for i in range(5)],
        "top10_candidate_indices": list(range(10)),
        "top10_candidate_ids": [f"c{i:02d}" for i in range(10)],
        "candidate_score_vector_sha256": "b" * 64,
        "construction_metrics_sha256": "c" * 64,
    }]


def fixed_rows():
    return [{
        "dataset": "D", "target_subject_id": "1", "panel": "A",
        "training_seed": 5, "level": 0, "method_id": "B1",
        "selected_candidate_index": 0, "selected_candidate_id": "c00",
    }]


def access_rows():
    return [{
        "stage": "selection", "method_id": "U5", "view": "target_unlabeled",
        "read_allowed": 1, "rows": 40, "labels": 0,
    }]


def test_selection_rejects_evaluation_descriptor() -> None:
    with pytest.raises(C84SContractError, match="reached selection"):
        freeze.validate_selection_inputs({"evaluation_label_view": "/secret"})


def test_all_81_scores_are_ranked_with_canonical_ties() -> None:
    rows = freeze.validate_zero_label_rows(score_rows())
    ranks = freeze.build_rank_rows(rows)
    by_index = {row["candidate_index"]: row["rank"] for row in ranks}
    assert by_index[6] == 1
    assert by_index[13] == 2
    assert by_index[0] < by_index[7]


def test_selection_freeze_round_trip(tmp_path: Path) -> None:
    root = freeze.freeze_selection(
        tmp_path / "freeze", input_descriptor={"same_label_oracle_accessed": False},
        zero_label_rows=score_rows(), q0_rows=q0_rows(),
        fixed_default_rows=fixed_rows(), access_rows=access_rows(),
    )
    manifest = freeze.replay_selection_freeze(root)
    assert manifest["evaluation_label_descriptor_received"] is False
    assert set(manifest["artifacts"]) >= {
        "zero_label_candidate_scores.csv", "q0_chain_selection.npz",
        "q0_sample_digest_registry.csv",
    }


def test_partial_freeze_never_publishes_final_root(tmp_path: Path) -> None:
    root = tmp_path / "freeze"
    with pytest.raises(C84SContractError, match="injected"):
        freeze.freeze_selection(
            root, input_descriptor={"same_label_oracle_accessed": False},
            zero_label_rows=score_rows(), q0_rows=q0_rows(),
            fixed_default_rows=fixed_rows(), access_rows=access_rows(),
            failure_injection_after="zero_label_candidate_scores.csv",
        )
    assert not root.exists()


def test_missing_candidate_score_fails_before_write(tmp_path: Path) -> None:
    with pytest.raises(C84SContractError, match="81 candidates"):
        freeze.freeze_selection(
            tmp_path / "freeze", input_descriptor={"same_label_oracle_accessed": False},
            zero_label_rows=score_rows()[:-1], q0_rows=q0_rows(),
            fixed_default_rows=fixed_rows(), access_rows=access_rows(),
        )
    assert not (tmp_path / "freeze").exists()


def test_binary_selector_family_executes_with_frozen_formulas() -> None:
    rng = np.random.default_rng(84)
    labels = np.tile([0, 1], 12)
    domains = np.repeat([10, 11], 12)
    source = rng.dirichlet(np.ones(2), size=(81, len(labels)))
    target_logits = rng.normal(size=(81, 30, 2))
    regimes = np.asarray(["ERM"] + ["OACI"] * 40 + ["SRC"] * 40)
    orders = np.asarray([0] + list(range(1, 41)) + list(range(1, 41)))
    scores, aline = selectors.score_context(
        source, labels, domains, target_logits, regimes, orders,
    )
    assert set(scores) == set(selectors.SELECTION_METHODS)
    assert all(value.shape == (81,) for value in scores.values())
    assert aline["pair_count"] == 3240
    assert selectors.descending_order(np.zeros(81))[0] == 0


def test_registered_zero_label_score_directions_on_binary_fixture() -> None:
    labels = np.tile([0, 1], 10)
    domains = np.repeat([1, 2], 10)
    source = np.eye(2)[labels] * 0.8 + 0.1
    source[[0, 5, 10, 15]] = source[[0, 5, 10, 15]][:, ::-1]
    high = np.eye(2)[labels] * 0.9 + 0.05
    low = np.full((len(labels), 2), 0.5)
    assert selectors.score_atc(source, labels, domains, high) > selectors.score_atc(source, labels, domains, low)
    assert selectors.score_nuclear_norm(high) > selectors.score_nuclear_norm(low)
    assert 0 <= selectors.score_cott(source, labels, domains, high, np.array([0.5, 0.5])) <= 1


def test_evaluation_utility_regret_and_random_oracle_controls() -> None:
    labels = np.tile([0, 1], 10)
    logits = np.zeros((81, len(labels), 2), dtype=float)
    quality = np.linspace(0.0, 5.0, 81)
    for candidate, scale in enumerate(quality):
        logits[candidate, np.arange(len(labels)), labels] = scale
    utility, metrics = evaluation.context_candidate_utility(logits, labels)
    assert utility.shape == (81,) and metrics.shape == (81, 3)
    assert evaluation.standardized_regret(utility, 80) == 0.0
    random = evaluation.evaluate_uniform_random(utility)
    oracle = evaluation.evaluate_oracle(utility, ["ERM"] + ["OACI"] * 40 + ["SRC"] * 40)
    assert random["top5"] == 5 / 81
    assert oracle["standardized_regret"] == 0.0
    assert evaluation.source_relative_regret_gain(0.8, 0.4) == 0.5


def test_estimation_MAE_uses_registered_measured_performance_not_utility() -> None:
    score = np.linspace(0.1, 0.9, 81)
    utility = np.linspace(0.9, 0.1, 81)
    measured_bacc = score + 0.02
    result = evaluation.measurement_metrics(
        score, utility, estimate_semantics=True,
        estimated_performance_target=measured_bacc,
    )
    assert result["accuracy_estimation_MAE"] == pytest.approx(0.02)
