from __future__ import annotations

import numpy as np
import pytest

from oaci.multidataset.c84s_common import C84SContractError
from oaci.multidataset.c84sr1_common import FIXED_METHODS, SCORE_METHODS
from oaci.multidataset.c84sr3_q0_store import synthetic_payload
from oaci.theory.c85u_historical_decision_replay import (
    compare_context_endpoints,
    replay_context_endpoints,
)

from oaci.tests.c85urp_test_support import shadow_context, shadow_payload


def _fixture(chains: int = 4):
    context = shadow_context()
    payload = shadow_payload()
    score_orders = {
        method: np.roll(np.arange(81, dtype=np.int16), offset)
        for offset, method in enumerate(SCORE_METHODS)
    }
    fixed = {method: index for index, method in enumerate(FIXED_METHODS)}
    q0 = synthetic_payload(
        context.identity(), [row.unit_id for row in context.candidates], chains=chains,
    )
    return payload, score_orders, fixed, q0


def test_deterministic_and_frozen_q0_action_replay() -> None:
    payload, score_orders, fixed, q0 = _fixture()
    endpoints = replay_context_endpoints(
        payload=payload, score_orders=score_orders,
        fixed_selected_indices=fixed, q0_payload=q0, q0_chains=4,
    )
    assert len(endpoints) == 20
    assert endpoints["Q0_B1"]["selected_regime"] == "STOCHASTIC_Q0"
    assert endpoints["Q0_FULL"]["selected_regime"] in {"ERM", "OACI", "SRC"}
    assert compare_context_endpoints(endpoints, endpoints) == {
        "selected_utility": 0.0,
        "standardized_regret": 0.0,
        "top1": 0.0,
        "top5": 0.0,
        "top10": 0.0,
    }


def test_missing_q0_chain_fails() -> None:
    payload, score_orders, fixed, q0 = _fixture()
    q0 = {key: np.array(value, copy=True) for key, value in q0.items()}
    q0["finite_budget_code"] = q0["finite_budget_code"][:-1]
    q0["finite_candidate_order"] = q0["finite_candidate_order"][:-1]
    with pytest.raises(C84SContractError, match="chain coverage"):
        replay_context_endpoints(
            payload=payload, score_orders=score_orders,
            fixed_selected_indices=fixed, q0_payload=q0, q0_chains=4,
        )


def test_historical_endpoint_mismatch_fails() -> None:
    payload, score_orders, fixed, q0 = _fixture()
    endpoints = replay_context_endpoints(
        payload=payload, score_orders=score_orders,
        fixed_selected_indices=fixed, q0_payload=q0, q0_chains=4,
    )
    historical = {
        method: dict(values) for method, values in endpoints.items()
    }
    historical["S1"]["selected_utility"] += 1e-4
    with pytest.raises(C84SContractError, match="endpoint replay mismatch"):
        compare_context_endpoints(endpoints, historical)


def test_score_order_and_fixed_method_coverage_fail_closed() -> None:
    payload, score_orders, fixed, q0 = _fixture()
    score_orders.pop("S1")
    with pytest.raises(C84SContractError, match="score-method set"):
        replay_context_endpoints(
            payload=payload, score_orders=score_orders,
            fixed_selected_indices=fixed, q0_payload=q0, q0_chains=4,
        )
