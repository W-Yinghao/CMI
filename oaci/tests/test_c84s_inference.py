from __future__ import annotations

import numpy as np
import pytest

from oaci.multidataset import c84s_inference as inference
from oaci.multidataset.c84s_common import C84SContractError


def test_eight_context_aggregation_is_equal_weighted() -> None:
    rows = [
        {"panel": panel, "training_seed": seed, "level": level,
         "value": float(index + 1)}
        for index, (panel, seed, level) in enumerate(
            (p, s, l) for p in ("A", "B") for s in (5, 6) for l in (0, 1)
        )
    ]
    result = inference.aggregate_context_rows(rows, value_field="value")
    assert result["primary"] == 4.5
    assert set(result["panel_seed"]) == {"A5", "A6", "B5", "B6"}


def test_missing_or_duplicate_context_fails() -> None:
    rows = [{"panel": "A", "training_seed": 5, "level": 0, "value": 1.0}] * 8
    with pytest.raises(C84SContractError, match="coverage"):
        inference.aggregate_context_rows(rows, value_field="value")


def test_rademacher_maxT_is_deterministic_and_familywise() -> None:
    effects = np.full((22, 6), 0.20)
    first = inference.rademacher_maxT(
        effects, dataset="Lee2019_MI", family="fixture", null_margin=0.05,
        draws=4096,
    )
    second = inference.rademacher_maxT(
        effects, dataset="Lee2019_MI", family="fixture", null_margin=0.05,
        draws=4096,
    )
    np.testing.assert_array_equal(first["pvalue"], second["pvalue"])
    assert np.all(np.asarray(first["pvalue"]) <= 0.05)


def test_q1_requires_every_registered_component() -> None:
    effects = np.full(22, 0.10)
    assert inference.q1_pass(effects, 0.01, [0.1, 0.1, 0.1, -0.01])
    assert not inference.q1_pass(effects, 0.10, [0.1] * 4)
    bad = effects.copy(); bad[0] = -0.11
    assert not inference.q1_pass(bad, 0.01, [0.1] * 4)


def test_q2_requires_upper_bound_and_target_floor() -> None:
    excess = np.full(22, 0.01)
    assert inference.q2_pass(excess, 0.01, 0.04, [0.01] * 4)
    assert not inference.q2_pass(excess, 0.01, 0.06, [0.01] * 4)
    bad = excess.copy(); bad[0] = 0.21
    assert not inference.q2_pass(bad, 0.01, 0.04, [0.01] * 4)


def test_dataset_decisions_keep_panel_stability_separate_from_3of4_pass() -> None:
    q1 = {method: np.full(22, 0.20) for method in inference.PRIMARY_METHODS}
    q2 = {method: np.full(22, 0.01) for method in inference.PRIMARY_METHODS}
    q1_cells = {method: [0.1, 0.1, 0.1, -0.01] for method in inference.PRIMARY_METHODS}
    q2_cells = {method: [0.01] * 4 for method in inference.PRIMARY_METHODS}
    result = inference.dataset_q1_q2(
        dataset="Lee2019_MI", q1_effects=q1, q2_excess=q2,
        q1_panel_seed=q1_cells, q2_panel_seed=q2_cells, draws=4096,
    )
    assert result["U5"]["Q1_pass"] is True
    assert result["U5"]["panel_seed_Q1_all_directional"] is False


def test_loto_preservation_uses_same_method() -> None:
    panels = [["U13"]] * 16 + [["U5"]] * 6
    result = inference.loto_preservation(
        dataset="Lee2019_MI", full_supporting_methods=["U13"],
        omitted_panel_method_sets=panels,
    )
    assert result["pass"] is False
    panels[16] = ["U13"]
    assert inference.loto_preservation(
        dataset="Lee2019_MI", full_supporting_methods=["U13"],
        omitted_panel_method_sets=panels,
    )["pass"] is True


def test_loto_stable_C_requires_no_hidden_Q1_method() -> None:
    panels = [[] for _ in range(20)]
    result = inference.loto_preservation(
        dataset="Cho2017", full_supporting_methods=[],
        omitted_panel_method_sets=panels, full_category="C",
    )
    assert result["pass"] is True
    panels[:6] = [["U5"] for _ in range(6)]
    assert inference.loto_preservation(
        dataset="Cho2017", full_supporting_methods=[],
        omitted_panel_method_sets=panels, full_category="C",
    )["pass"] is False


def test_budget_closure_preserves_nonmonotone_failure() -> None:
    effects = {
        1: np.full(20, 0.2), 2: np.full(20, 0.2),
        4: np.zeros(20), 8: np.full(20, 0.2), "FULL": np.full(20, 0.2),
    }
    cells = {budget: [0.2] * 4 for budget in effects}
    result = inference.qualify_budget_curve(
        effects, cells, dataset="Cho2017", draws=4096,
    )
    assert result["direct"][0] is True
    assert result["closure"][0] is False
    assert result["Bstar"] == 8
