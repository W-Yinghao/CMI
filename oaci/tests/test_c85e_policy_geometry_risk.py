"""Shadow-only C85E policy-use, geometry, risk, and theorem guards."""
from __future__ import annotations

import math

import numpy as np
import pytest

from oaci.theory.c85e_action_geometry import (
    EPSILON_GRID, TAU_GRID, context_geometry, raw_utility_gaps,
    soft_action_weights,
)
from oaci.theory.c85e_policy_use import (
    C85EPolicyUseError, summarize_deterministic_policy_use,
    summarize_stochastic_q0_context,
)
from oaci.theory.c85e_rank_topk_regret import join_measurement_and_geometry
from oaci.theory.c85e_robust_risk import (
    CVAR_ALPHA_GRID, empirical_upper_cvar, target_equal_regrets,
)
from oaci.theory.c85e_theorem_bridge import (
    THEOREM_STATUSES, theorem_applicability_matrix,
)
from oaci.theory.c85e_execute import (
    C85EInputBundle, FrozenQ0Actions, UtilityContext, build_analysis_tables,
)
from oaci.theory.c85e_policy_use import DETERMINISTIC_METHODS
from oaci.theory.c85e_result_manifest import REGISTERED_TABLES


def _identity(index: int = 0) -> dict[str, object]:
    return {
        "dataset": "Shadow", "target_subject_id": "T1", "panel": "A",
        "training_seed": 5, "level": index,
    }


def _action_row(action: int, regret: float, *, level: int = 0) -> dict[str, object]:
    return {
        **_identity(level), "selected_candidate_index": action,
        "selected_regime": "ERM" if action == 0 else "OACI",
        "standardized_regret": regret,
        # Candidate strings are attached metadata, not the cross-context key.
        "candidate_id": f"zoo-specific-{action}",
    }


def test_raw_gap_and_standardized_regret_are_not_interchanged() -> None:
    utility = np.linspace(0.2, 0.7, 81, dtype=np.float64)
    gaps = raw_utility_gaps(utility)
    standardized = gaps / (utility.max() - utility.min())
    assert gaps[79] == pytest.approx(0.00625)
    assert standardized[79] == pytest.approx(0.0125)
    geometry = context_geometry(utility)
    assert geometry["summary"]["best_second_raw_utility_gap"] == pytest.approx(gaps[79])
    assert geometry["summary"]["geometry_scale"] == "RAW_COMPOSITE_UTILITY_GAP"


def test_locked_geometry_grids_and_stable_soft_weights() -> None:
    assert EPSILON_GRID == (0.005, 0.01, 0.02, 0.05)
    assert TAU_GRID == (0.005, 0.01, 0.02, 0.05, 0.10)
    utility = np.full(81, -1_000.0)
    utility[0] = 1_000.0
    weights = soft_action_weights(utility, 0.005)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(1.0)
    assert context_geometry(np.ones(81))["summary"]["exact_comaximizer_count"] == 81


def test_exact_policy_collapse_is_distinct_from_near_collapse() -> None:
    reference = [_action_row(0, 0.2)]
    exact = summarize_deterministic_policy_use(
        [_action_row(0, 0.2)], reference, method_id="U11", reference_id="B1",
        scope={"scope": "shadow"},
    )
    near = summarize_deterministic_policy_use(
        [_action_row(1, 0.21)], reference, method_id="U11", reference_id="B1",
        scope={"scope": "shadow"},
    )
    assert exact["exact_collapse"] is True
    assert exact["T3_exactly_applicable"] is True
    assert exact["divergent_context_regret_difference"] is None
    assert near["action_divergence_rate"] == 1.0
    assert near["exact_collapse"] is False
    assert near["T3_exactly_applicable"] is False
    assert near["divergent_context_regret_difference"] == pytest.approx(0.01)


def test_canonical_action_index_not_candidate_string_controls_equivalence() -> None:
    left = _action_row(4, 0.3)
    right = _action_row(4, 0.3)
    left["candidate_id"] = "zoo-A-unit"
    right["candidate_id"] = "zoo-B-unit"
    row = summarize_deterministic_policy_use(
        [left], [right], method_id="U13", reference_id="B1",
        scope={"scope": "shadow"},
    )
    assert row["exact_collapse"] is True
    with pytest.raises(C85EPolicyUseError, match="0..80"):
        summarize_deterministic_policy_use(
            [_action_row(81, 0.3)], [right], method_id="U13", reference_id="B1",
            scope={"scope": "shadow"},
        )


def test_finite_q0_remains_a_stochastic_action_distribution() -> None:
    actions = np.tile(np.array([0, 1], dtype=np.uint8), 1_024)
    row = summarize_stochastic_q0_context(
        selected_actions=actions,
        selected_regimes=["ERM" if value == 0 else "OACI" for value in actions],
        reference_action=0, identity=_identity(),
    )
    assert row["chains"] == 2_048
    assert row["probability_action_differs_from_reference"] == 0.5
    assert row["action_distribution"] == {"0": 0.5, "1": 0.5}
    assert row["action_entropy"] == pytest.approx(math.log(2.0))
    assert row["stochastic_policy_preserved"] is True
    assert row["chains_are_scientific_sample"] is False


def test_target_equal_aggregation_and_fractional_empirical_cvar() -> None:
    rows = []
    for target, regret in (("small", 0.1), ("large", 0.9)):
        for panel in ("A", "B"):
            for seed in (5, 6):
                for level in (0, 1):
                    rows.append({
                        "dataset": "Shadow", "target_subject_id": target,
                        "method_id": "U13", "panel": panel,
                        "training_seed": seed, "level": level,
                        "standardized_regret": regret,
                    })
    targets = target_equal_regrets(rows)
    assert len(targets) == 2
    assert sorted(row["standardized_target_regret"] for row in targets) == [0.1, 0.9]
    assert all(row["context_repeats"] == 8 for row in targets)
    assert CVAR_ALPHA_GRID == (0.50, 0.75, 0.90)
    assert empirical_upper_cvar([0.0, 1.0, 2.0], 0.50) == pytest.approx(5.0 / 3.0)
    assert empirical_upper_cvar([0.0, 1.0, 2.0], 0.75) == pytest.approx(2.0)


def test_measurement_applicability_preserves_nulls() -> None:
    geometry = [{
        **_identity(), "best_second_raw_utility_gap": 0.01,
        "best_fifth_raw_utility_gap": 0.02, "best_tenth_raw_utility_gap": 0.03,
        "utility_range": 0.5,
    }]
    method = [{
        **_identity(), "method_id": "B1", "rank_measurement_applicable": 0,
        "Spearman": None, "Kendall": None, "pairwise_ordering_accuracy": None,
        "top1": 0.0, "top5": 1.0, "top10": 1.0,
        "standardized_regret": 0.1, "selected_utility": 0.8,
    }]
    row = join_measurement_and_geometry(method, geometry)[0]
    assert row["Spearman"] is None
    assert row["Kendall"] is None
    assert row["pairwise_ordering_accuracy"] is None


def test_theorem_applicability_is_fail_closed_and_statuses_are_immutable() -> None:
    no_collapse = theorem_applicability_matrix([])
    labels = {row["theorem_id"]: row["applicability"] for row in no_collapse}
    assert labels["T1"] == "ASSUMPTIONS_NOT_IDENTIFIED"
    assert labels["T3"] == "NOT_APPLICABLE"
    assert labels["T4"] == "ASSUMPTIONS_NOT_IDENTIFIED"
    assert labels["T5"] == "OPEN_THEOREM"
    assert labels["T7"] == "ASSUMPTIONS_NOT_IDENTIFIED"
    exact = theorem_applicability_matrix([{
        "exact_collapse": True, "scope": "dataset_x_level", "dataset": "Shadow",
        "level": 0, "method_id": "U11", "reference_id": "B1",
    }])
    t3 = [row for row in exact if row["theorem_id"] == "T3"]
    assert len(t3) == 1 and t3[0]["applicability"] == "EXACTLY_APPLICABLE"
    assert THEOREM_STATUSES == {
        "T1": "PROVED", "T2": "COUNTEREXAMPLE", "T3": "PROVED",
        "T4": "PROVED", "T5": "OPEN", "T6": "COUNTEREXAMPLE", "T7": "PROVED",
    }
    assert all(row["theorem_status_changed"] is False for row in exact)


def test_full_shadow_analysis_materializes_every_registered_table() -> None:
    contexts: dict[str, UtilityContext] = {}
    actions: dict[str, dict[str, int]] = {}
    q0: list[FrozenQ0Actions] = []
    method_rows: list[dict[str, object]] = []
    candidate_ids = tuple(f"shadow-{index:02d}" for index in range(81))
    regimes = tuple("ERM" if index == 0 else "OACI" for index in range(81))
    methods = (*DETERMINISTIC_METHODS, "Q0_B1")
    for dataset_index, dataset in enumerate(("Lee2019_MI", "Cho2017", "PhysionetMI")):
        for panel in ("A", "B"):
            for seed in (5, 6):
                for level in (0, 1):
                    context_id = f"{dataset}-{panel}-{seed}-{level}"
                    utility = np.linspace(0.0, 1.0, 81) + dataset_index * 1e-5
                    contexts[context_id] = UtilityContext(
                        context_id=context_id, dataset=dataset, target_subject_id="T1",
                        panel=panel, training_seed=seed, level=level, utility=utility,
                        candidate_ids=candidate_ids, regimes=regimes,
                    )
                    actions[context_id] = {
                        method: (index % 3) for index, method in enumerate(DETERMINISTIC_METHODS)
                    }
                    q0.append(FrozenQ0Actions(
                        context_id=context_id, method_id="Q0_B1",
                        selected_actions=np.tile(np.array([0, 1], dtype=np.uint8), 1_024),
                    ))
                    for index, method in enumerate(methods):
                        method_rows.append({
                            "dataset": dataset, "target_subject_id": "T1", "panel": panel,
                            "training_seed": seed, "level": level, "method_id": method,
                            "standardized_regret": (index + level) / 100.0,
                            "selected_utility": 1.0 - (index + level) / 100.0,
                            "source_relative_regret_gain": 0.0,
                            "top1": float(index == 0), "top5": 1.0, "top10": 1.0,
                            "coverage": 1.0, "rank_measurement_applicable": 1,
                            "performance_estimate_applicable": 0,
                            "Spearman": 0.1, "Kendall": 0.05,
                            "pairwise_ordering_accuracy": 0.55,
                            "accuracy_estimation_MAE": None,
                        })
    frontier = tuple({
        "dataset": dataset, "budget": "1", "mean_effect": "0.1",
        "maxT_pvalue": "0.5", "direct_qualification": "0",
        "closure_qualification": "0", "Bstar": "NONE",
        "level0_Bstar": "NONE", "level1_Bstar": "NONE",
    } for dataset in ("Lee2019_MI", "Cho2017", "PhysionetMI"))
    bundle = C85EInputBundle(
        contexts=contexts, method_rows=tuple(method_rows), deterministic_actions=actions,
        q0_actions=tuple(q0), compact_tables={
            "label_budget_frontier.csv": frontier,
            "level_specific_Q1_Q2.csv": (), "target_level_method_effects.csv": (),
            "panel_seed_stability.csv": (),
        }, input_replay_sha256="0" * 64,
    )
    tables = build_analysis_tables(bundle)
    assert set(tables) == set(REGISTERED_TABLES)
    assert all(tables[name] for name in REGISTERED_TABLES)
    assert all(
        row["result_tag"] == "POST_C84S_EXPLORATORY"
        for rows in tables.values() for row in rows
    )
    support_types = {row["object_type"] for row in tables["support_level_policy_use_profile.csv"]}
    assert support_types == {"REALIZED_POLICY_USE", "FROZEN_LABEL_FRONTIER_COMPONENT"}
