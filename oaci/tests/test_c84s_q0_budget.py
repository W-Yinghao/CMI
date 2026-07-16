from __future__ import annotations

import numpy as np
import pytest

from oaci.multidataset import c84s_q0_budget as q0
from oaci.multidataset.c84s_common import C84SContractError


def construction_fixture(per_class: int = 40):
    labels = np.repeat([0, 1], per_class)
    trial_ids = np.asarray([f"trial-{index:03d}" for index in range(len(labels))])
    return trial_ids, labels


def test_stream_seed_is_stable_and_scoped() -> None:
    assert q0.stream_seed("Lee2019_MI", 19, 0) == q0.stream_seed("Lee2019_MI", 19, 0)
    assert q0.stream_seed("Lee2019_MI", 19, 0) != q0.stream_seed("Lee2019_MI", 19, 1)
    assert q0.stream_seed("Lee2019_MI", 19, 0) != q0.stream_seed("Cho2017", 19, 0)


def test_nested_samples_pair_budgets_and_cover_full() -> None:
    ids, labels = construction_fixture()
    samples = q0.nested_trial_samples(
        ids, labels, dataset="D", target_subject=1, chain=2,
    )
    assert [len(samples[value]) for value in q0.PRIMARY_BUDGETS] == [2, 4, 8, 16, 80]
    for left, right in zip(q0.PRIMARY_BUDGETS, q0.PRIMARY_BUDGETS[1:]):
        assert set(samples[left]) <= set(samples[right])


def test_extended_grid_is_supported_without_changing_primary() -> None:
    ids, labels = construction_fixture()
    grid = (1, 2, 4, 8, 16, 32, "FULL")
    samples = q0.nested_trial_samples(
        ids, labels, dataset="Lee2019_MI", target_subject=19, chain=0,
        budgets=grid,
    )
    assert len(samples[32]) == 64 and len(samples["FULL"]) == 80
    assert q0.PRIMARY_BUDGETS == (1, 2, 4, 8, "FULL")


def test_endpoint_metrics_are_binary_and_use_fifteen_bin_ece() -> None:
    labels = np.asarray([0, 0, 1, 1])
    logits = np.asarray([[5, 0], [4, 0], [0, 4], [0, 5]], dtype=float)
    metrics = q0.endpoint_metrics(logits, labels)
    assert metrics["bAcc"] == 1.0
    assert metrics["NLL"] > 0.0
    assert 0.0 <= metrics["ECE"] <= 1.0


def test_candidate_ties_choose_smallest_canonical_index() -> None:
    assert q0.descending_order(np.zeros(81))[0] == 0


def test_select_chain_freezes_scores_without_evaluation_input() -> None:
    ids, labels = construction_fixture(per_class=12)
    rng = np.random.default_rng(8)
    logits = rng.normal(size=(81, len(ids), 2))
    rows = q0.select_chain(
        logits, ids, labels, dataset="D", target_subject=1, chain=0,
    )
    assert [row["budget"] for row in rows] == ["1", "2", "4", "8", "FULL"]
    assert all(len(row["candidate_score_vector_sha256"]) == 64 for row in rows)


def test_infeasible_budget_fails_closed() -> None:
    ids, labels = construction_fixture(per_class=7)
    with pytest.raises(C84SContractError, match="infeasible"):
        q0.nested_trial_samples(ids, labels, dataset="D", target_subject=1, chain=0)

