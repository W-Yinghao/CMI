import numpy as np

from oaci.multidataset.c84sr1_common import expected_methods
from oaci.multidataset.c84sr1_method_context_materialization import (
    METHOD_CONTEXT_FIELDS_V2, materialize_context,
)
from oaci.multidataset.c84sr1_q0_store import synthetic_payload


def _regimes():
    return ["ERM"] + ["OACI"] * 40 + ["SRC"] * 40


def test_materialization_integrates_q0_and_preserves_null_applicability():
    identity = {
        "dataset": "Lee2019_MI", "target_subject_id": "1",
        "panel": "A", "training_seed": 5, "level": 0,
    }
    candidate_ids = [f"candidate-{index}" for index in range(81)]
    q0 = synthetic_payload(identity, candidate_ids, chains=4)
    utility = np.linspace(0.0, 1.0, 81)
    metrics = np.column_stack((utility, 1 - utility, 1 - utility))
    scores = {method: np.linspace(0.0, 1.0, 81) for method in ("S1", "U5", "U7", "U11", "U13", "U14", "U15")}
    fixed = {"B1": 0, "B2": 40, "B3": 80, "B4O": 20, "B4S": 60}
    rows, regime_rows, diagnostics = materialize_context(
        identity=identity, candidate_ids=candidate_ids, regimes=_regimes(),
        utility=utility, evaluation_metrics=metrics, score_vectors=scores,
        fixed_selected_indices=fixed, q0_payload=q0, q0_chains=4,
    )
    assert [row["method_id"] for row in rows] == list(expected_methods("Lee2019_MI"))
    assert len(rows) == 21 and len(regime_rows) == 18 and len(diagnostics) == 6
    assert all(tuple(row) == METHOD_CONTEXT_FIELDS_V2 for row in rows)
    b0 = next(row for row in rows if row["method_id"] == "B0")
    q1 = next(row for row in rows if row["method_id"] == "Q0_B1")
    s1 = next(row for row in rows if row["method_id"] == "S1")
    assert b0["rank_measurement_applicable"] == 0 and b0["Spearman"] is None
    assert q1["rank_measurement_applicable"] == 1 and q1["accuracy_estimation_MAE"] is None
    assert s1["performance_estimate_applicable"] == 1 and s1["accuracy_estimation_MAE"] is not None
    assert q1["coverage"] == 1.0 and q1["selected_regime"] == "STOCHASTIC_Q0"


def test_no_context_catastrophic_field_in_v2_schema():
    assert "catastrophic_failure" not in METHOD_CONTEXT_FIELDS_V2
    assert "rank_measurement_applicable" in METHOD_CONTEXT_FIELDS_V2
    assert "performance_estimate_applicable" in METHOD_CONTEXT_FIELDS_V2
