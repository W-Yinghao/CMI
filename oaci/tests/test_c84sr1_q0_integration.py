from pathlib import Path

import numpy as np

from oaci.multidataset import c84s_q0_budget as legacy
from oaci.multidataset.c84sr1_q0_store import (
    build_context_payload, read_context_shard, synthetic_payload,
    validate_payload, write_context_shard,
)


def _fixture(chains: int = 3):
    rng = np.random.default_rng(84)
    trial_ids = np.asarray([f"trial-{index}" for index in range(128)])
    labels = np.asarray([0] * 64 + [1] * 64)
    logits = rng.normal(size=(81, 128, 2))
    rows = [
        {
            "dataset": "Lee2019_MI", "target_subject_id": "1",
            "target_trial_id": trial_ids[index],
            "canonical_class_label": int(labels[index]),
        }
        for index in range(128)
    ]
    identity = {
        "dataset": "Lee2019_MI", "target_subject_id": "1",
        "panel": "A", "training_seed": 5, "level": 0,
    }
    payload = build_context_payload(
        identity=identity, candidate_ids=[f"candidate-{index}" for index in range(81)],
        target_logits=logits, target_trial_ids=trial_ids,
        construction_rows=rows, chains=chains,
    )
    return payload, logits, labels, trial_ids


def test_vectorized_q0_matches_locked_per_chain_implementation():
    payload, logits, labels, trial_ids = _fixture()
    for chain in range(3):
        samples = legacy.nested_trial_samples(
            trial_ids, labels, dataset="Lee2019_MI", target_subject="1",
            chain=chain, budgets=(1, 2, 4, 8, 16, 32, "FULL"),
        )
        index = {value: position for position, value in enumerate(trial_ids.tolist())}
        for budget in (1, 2, 4, 8, 16, 32):
            selected = np.asarray([index[value] for value in samples[budget]])
            scores, _ = legacy.candidate_scores(logits[:, selected], labels[selected])
            expected = legacy.descending_order(scores)
            mask = (
                (payload["finite_chain"] == chain)
                & (payload["finite_budget_code"] == budget)
            )
            assert np.array_equal(payload["finite_candidate_order"][mask][0], expected)


def test_q0_shard_exact_round_trip(tmp_path: Path):
    payload = synthetic_payload(
        {"dataset": "PhysionetMI", "target_subject_id": "2", "panel": "B", "training_seed": 6, "level": 1},
        [f"candidate-{index}" for index in range(81)], chains=4,
    )
    identity = write_context_shard(tmp_path / "context.npz", payload, chains=4)
    replay, observed = read_context_shard(
        tmp_path / "context.npz", expected_sha256=identity["sha256"], chains=4,
    )
    assert observed["total_records"] == 17
    assert all(np.array_equal(replay[key], value) for key, value in payload.items())


def test_q0_incomplete_chain_fails():
    payload = synthetic_payload(
        {"dataset": "PhysionetMI", "target_subject_id": "2", "panel": "B", "training_seed": 6, "level": 1},
        [f"candidate-{index}" for index in range(81)], chains=4,
    )
    payload["finite_chain"][0] = 3
    try:
        validate_payload(payload, chains=4)
    except RuntimeError as error:
        assert "chain coverage" in str(error)
    else:
        raise AssertionError("incomplete Q0 chain coverage was accepted")
