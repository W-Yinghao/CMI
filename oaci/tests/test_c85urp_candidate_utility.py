from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oaci.multidataset.c84s_common import C84SContractError
from oaci.multidataset.c84sr1_context_enumerator import ContextDescriptor
from oaci.theory.c85u_persistence import (
    load_context_artifact,
    validate_context_payload,
    write_context_artifact,
)
from oaci.theory.c85u_result_manifest import (
    publish_utility_field,
    validate_utility_manifest,
)
from oaci.theory.c85u_utility_builder import (
    align_evaluation_rows,
    compute_context_utility_payload,
)

from .c85urp_test_support import shadow_context, shadow_payload, shadow_rows


def test_shadow_metric_utility_and_persistence_replay(tmp_path: Path) -> None:
    payload = shadow_payload()
    replay = validate_context_payload(payload)
    assert replay["candidate_rows"] == 81
    assert np.max(np.abs(
        payload["composite_utility"]
        - np.mean(np.column_stack((
            payload["bAcc_midrank_percentile"],
            payload["negative_NLL_midrank_percentile"],
            payload["negative_ECE_midrank_percentile"],
        )), axis=1)
    )) <= 1e-12
    path = tmp_path / "context.npz"
    written = write_context_artifact(path, payload)
    loaded, persisted = load_context_artifact(path, expected_sha256=written["artifact_sha256"])
    assert persisted["utility_vector_sha256"] == replay["utility_vector_sha256"]
    assert np.array_equal(loaded["composite_utility"], payload["composite_utility"])


def test_midrank_ties_first_argmax_and_zero_spread_regret() -> None:
    payload = shadow_payload(zero_spread=True)
    assert np.all(payload["composite_utility"] == payload["composite_utility"][0])
    assert np.all(payload["utility_rank_midrank"] == 0.5)
    assert int(payload["best_candidate_index"]) == 0
    assert int(payload["exact_comaximizer_count"]) == 81
    assert np.array_equal(payload["canonical_utility_order_position"], np.arange(1, 82))
    assert np.all(payload["standardized_regret"] == 0.0)


def test_candidate_and_trial_identity_drift_fail() -> None:
    context = shadow_context()
    rows = shadow_rows()
    ids = np.asarray([row["target_trial_id"] for row in rows], dtype=str)
    with pytest.raises(C84SContractError, match="candidate identity/order"):
        compute_context_utility_payload(
            context=context,
            candidate_data={
                "candidate_ids": ["wrong"] * 81,
                "target_trial_ids": ids,
                "target_logits": np.zeros((81, len(ids), 2)),
            },
            evaluation_rows=rows,
            evaluation_label_view_manifest_sha256="d" * 64,
        )
    with pytest.raises(C84SContractError, match="outside frozen target artifact"):
        align_evaluation_rows(
            context, ids[:-1], rows,
        )


def test_tampered_metric_or_identity_fails() -> None:
    payload = shadow_payload()
    tampered = {key: np.array(value, copy=True) for key, value in payload.items()}
    tampered["balanced_accuracy"][0] += 0.01
    with pytest.raises(C84SContractError):
        validate_context_payload(tampered)
    tampered = {key: np.array(value, copy=True) for key, value in payload.items()}
    tampered["candidate_id"][0] = "different"
    with pytest.raises(C84SContractError):
        validate_context_payload(tampered)


def test_atomic_shadow_field_and_partial_failure(tmp_path: Path) -> None:
    final = tmp_path / "complete"
    result = publish_utility_field(
        payloads=[shadow_payload()], final_root=final,
        input_identity={"fixture": "SHADOW"}, expected_contexts=1,
        expected_candidate_rows=81,
    )
    assert result["contexts"] == 1
    assert validate_utility_manifest(
        final, expected_contexts=1, expected_candidate_rows=81,
    )["status"] == "PASS"

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        publish_utility_field(
            payloads=[shadow_payload()], final_root=failed,
            input_identity={"fixture": "SHADOW"}, expected_contexts=1,
            expected_candidate_rows=81, failure_after_context=0,
        )
    assert not failed.exists()
    assert list(tmp_path.glob(".failed.staging-*"))


def test_registered_arithmetic_is_exact() -> None:
    assert 944 * 81 == 76_464
