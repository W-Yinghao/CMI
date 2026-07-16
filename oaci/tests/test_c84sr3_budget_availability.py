import numpy as np

from oaci.multidataset.c84s_label_views import align_and_split_labels
from oaci.multidataset.c84sr1_synthetic import (
    synthetic_contexts, synthetic_label_rows,
)
from oaci.multidataset.c84sr3_common import (
    METHOD_CONTEXT_ROWS, Q0_RECORDS, expected_method_context_rows,
    expected_methods, finite_budgets,
)
from oaci.multidataset.c84sr3_q0_store import synthetic_payload, validate_payload
from oaci.multidataset.c84sr3_stage_b_selection import construction_budget_availability


def test_repaired_budget_and_method_arithmetic_is_exact():
    assert finite_budgets("Lee2019_MI") == (1, 2, 4, 8, 16)
    assert finite_budgets("Cho2017") == (1, 2, 4, 8, 16, 32)
    assert finite_budgets("PhysionetMI") == (1, 2, 4, 8)
    assert "Q0_B32" not in expected_methods("Lee2019_MI")
    assert "Q0_B32" in expected_methods("Cho2017")
    assert expected_method_context_rows() == METHOD_CONTEXT_ROWS == 18_432
    assert Q0_RECORDS == 8_750_000


def test_construction_only_preflight_marks_only_lee_b32_unavailable():
    registry, labels, _ = synthetic_label_rows()
    construction, _, _ = align_and_split_labels(registry, labels)
    rows = construction_budget_availability(
        construction, synthetic_contexts(), synthetic=True,
    )
    assert len(rows) == 19
    lookup = {(row["dataset"], row["budget"]): row for row in rows}
    lee = lookup[("Lee2019_MI", "32")]
    assert lee == {
        "dataset": "Lee2019_MI", "budget": "32", "budget_role": "SECONDARY",
        "targets": 22, "feasible_targets": 0, "infeasible_targets": 22,
        "min_labels_per_class": 25, "max_labels_per_class": 25,
        "operative": 0, "disposition": "INPUT_UNAVAILABLE_ALL_TARGETS",
    }
    assert lookup[("Lee2019_MI", "16")]["operative"] == 1
    assert lookup[("Cho2017", "32")]["feasible_targets"] == 20
    assert lookup[("PhysionetMI", "8")]["feasible_targets"] == 76


def test_v2_q0_shard_omits_lee_b32_without_changing_primary_budgets():
    identity = {
        "dataset": "Lee2019_MI", "target_subject_id": "1", "panel": "A",
        "training_seed": 5, "level": 0,
    }
    payload = synthetic_payload(
        identity, [f"candidate_{index:02d}" for index in range(81)], chains=8,
    )
    replay = validate_payload(payload, chains=8)
    assert replay["finite_records"] == 5 * 8
    assert set(np.asarray(payload["finite_budget_code"]).tolist()) == {1, 2, 4, 8, 16}

