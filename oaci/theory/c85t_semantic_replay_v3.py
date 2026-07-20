"""Semantic replay of a staged or frozen C85T V3 execution bundle."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85t_execution_context_v3 import (
    ValidatedC85TExecutionContext,
    sha256_file,
    validate_registered_execution_context,
)
from .c85t_monte_carlo import _summarize_s9_arrays_v2, summarize_near_replicates_v2
from .c85t_proofs import PROOF_FILENAMES
from .c85t_registered_v3 import (
    execute_registered_exact_scenarios_v3,
    replay_registered_s9_artifacts_v3,
)
from .c85t_result_manifest import read_deterministic_npz


RESULT_SCHEMA_V3 = "c85t_synthetic_validation_and_proof_candidates_result_v3"
SUCCESS_GATE_V3 = (
    "C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_"
    "C85V_REVIEW_REQUIRED"
)
SEMANTIC_RECEIPT_SCHEMA_V3 = "c85t_result_semantic_replay_receipt_v3"
THEOREM_IDS = tuple(f"T{index}" for index in range(1, 8))
SCENARIO_IDS = tuple(f"S{index}" for index in range(11))
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_STATEMENT_SHA = re.compile(r"Statement SHA-256: `([0-9a-f]{64})`")
_ALLOWED_DISPOSITIONS = {
    "PROPOSED_PROOF",
    "PROPOSED_COUNTEREXAMPLE",
    "INCOMPLETE_OPEN",
    "PROPOSED_INVALIDATION",
}
_CHECK_CLASS = "PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY"
_EXACT_FIELDS = {
    "S0": {"optimal_constant_risk"},
    "S1": {
        "coarse_registered_risk",
        "rich_unrestricted_risk",
        "rich_registered_risk",
    },
    "S2": {"action_divergence", "registered_risk", "reference_risk"},
    "S3": {"selected_action", "regret", "spearman"},
    "S4": {"top4_localization", "selected_regret"},
    "S5": {
        "candidate_open_lower",
        "candidate_open_upper",
        "endpoint_policy",
        "status",
    },
    "S6": {
        "gaps",
        "epsilon_near_optimal_set",
        "near_optimal_count",
        "hill_2_effective_size",
        "entropy_effective_size",
        "t7_primary_union_bound",
        "historical_looser_diagnostic",
    },
    "S7": {
        "gaps",
        "epsilon_near_optimal_set",
        "near_optimal_count",
        "hill_2_effective_size",
        "entropy_effective_size",
        "t7_primary_union_bound",
        "historical_looser_diagnostic",
    },
    "S8": {
        "identified_set_infinity_diameter",
        "optimal_randomized_action_distribution",
        "minimax_regret",
        "extreme_point_constraint_slacks",
        "active_constraints",
        "pure_action_minimax_regret",
        "randomization_gain",
    },
    "S9": {
        "population_means",
        "passive_allocation",
        "neyman_allocation",
        "passive_analytic_variance",
        "neyman_analytic_variance",
    },
    "S10": {
        "coarse_policy",
        "coarse_risk",
        "historical_rich_risk",
        "rich_unrestricted_risk",
        "v2_rich_risk",
        "rich_gap",
        "reversal",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DecisionContractError(f"C85T V3 CSV is absent: {path.name}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_exact_scenario_schema(exact: Mapping[str, Any]) -> None:
    if set(exact) != set(SCENARIO_IDS) or len(exact) != len(SCENARIO_IDS):
        raise DecisionContractError("C85T V3 exact scenario keys are not exactly S0..S10")
    for scenario_id, fields in _EXACT_FIELDS.items():
        row = exact[scenario_id]
        if not isinstance(row, dict) or set(row) != fields:
            raise DecisionContractError(f"C85T V3 exact schema drifted: {scenario_id}")
    expected_s10 = {
        "coarse_policy": [1, 1],
        "coarse_risk": "11/40",
        "historical_rich_risk": "11/40",
        "rich_unrestricted_risk": "0",
        "v2_rich_risk": "3/5",
        "rich_gap": "3/5",
        "reversal": "13/40",
    }
    if exact["S10"] != expected_s10:
        raise DecisionContractError("C85T V3 S10 exact values drifted")
    s8 = exact["S8"]
    rational_fields = (
        "identified_set_infinity_diameter",
        "minimax_regret",
        "pure_action_minimax_regret",
        "randomization_gain",
    )
    if any(not isinstance(s8[field], str) for field in rational_fields):
        raise DecisionContractError("C85T V3 S8 rational certificate drifted")
    for field in (
        "optimal_randomized_action_distribution",
        "extreme_point_constraint_slacks",
        "active_constraints",
    ):
        if not isinstance(s8[field], list) or not s8[field]:
            raise DecisionContractError(f"C85T V3 S8 certificate field drifted: {field}")


def _validate_near_arrays(
    scenario_id: str,
    arrays: Mapping[str, np.ndarray],
    *,
    utilities: np.ndarray,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "replicate_id": (np.dtype("<u2"), (4096,)),
        "selected_action": (np.dtype("<u2"), (4096,)),
        "top1": (np.dtype("uint8"), (4096,)),
        "outside_A_epsilon": (np.dtype("uint8"), (4096,)),
        "selection_regret": (np.dtype("<f8"), (4096,)),
    }
    if set(arrays) != set(required):
        raise DecisionContractError(f"{scenario_id} V3 replicate fields drifted")
    for name, (dtype, shape) in required.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
            raise DecisionContractError(f"{scenario_id} V3 replicate field drifted: {name}")
    if not np.array_equal(arrays["replicate_id"], np.arange(4096, dtype="<u2")):
        raise DecisionContractError(f"{scenario_id} V3 replicate IDs drifted")
    selected = arrays["selected_action"].astype(np.int64)
    if np.any(selected < 0) or np.any(selected >= utilities.size):
        raise DecisionContractError(f"{scenario_id} selected action is out of range")
    if not np.isin(arrays["top1"], (0, 1)).all() or not np.isin(
        arrays["outside_A_epsilon"], (0, 1)
    ).all():
        raise DecisionContractError(f"{scenario_id} V3 indicators are not binary")
    optimum = int(np.argmax(utilities))
    expected_top1 = (selected == optimum).astype(np.uint8)
    near = np.asarray(geometry["epsilon_near_optimal_set"], dtype=np.int64)
    expected_outside = (~np.isin(selected, near)).astype(np.uint8)
    expected_regret = np.asarray(
        utilities[optimum] - utilities[selected], dtype="<f8"
    )
    if not np.array_equal(arrays["top1"], expected_top1):
        raise DecisionContractError(f"{scenario_id} top1 semantics drifted")
    if not np.array_equal(arrays["outside_A_epsilon"], expected_outside):
        raise DecisionContractError(f"{scenario_id} outside-set semantics drifted")
    if not np.array_equal(arrays["selection_regret"], expected_regret):
        raise DecisionContractError(f"{scenario_id} regret semantics drifted")
    return summarize_near_replicates_v2(scenario_id, arrays, geometry)


def _validate_s9_arrays(
    arrays: Mapping[str, np.ndarray], population: np.ndarray
) -> dict[str, Any]:
    required: set[str] = set()
    for design in ("passive", "neyman"):
        required.update(
            f"{design}_{field}"
            for field in (
                "replicate_id",
                "selected_action",
                "correct_best",
                "top2_coverage",
                "selection_regret",
                "D_hat",
            )
        )
    required.update(
        f"paired_passive_minus_neyman_{field}"
        for field in ("selection_regret", "correct_best", "top2_coverage", "D_hat")
    )
    if set(arrays) != required:
        raise DecisionContractError("C85T V3 S9 replicate fields drifted")
    true_best = int(np.argmin(population))
    for design in ("passive", "neyman"):
        selected = np.asarray(arrays[f"{design}_selected_action"])
        if selected.dtype != np.dtype("uint8") or selected.shape != (4096,):
            raise DecisionContractError(f"C85T V3 S9 action field drifted: {design}")
        if np.any(selected > 3):
            raise DecisionContractError(f"C85T V3 S9 selected action is out of range: {design}")
        for binary in ("correct_best", "top2_coverage"):
            value = np.asarray(arrays[f"{design}_{binary}"])
            if value.dtype != np.dtype("uint8") or not np.isin(value, (0, 1)).all():
                raise DecisionContractError(f"C85T V3 S9 indicator drifted: {design}/{binary}")
        expected_correct = (selected == true_best).astype(np.uint8)
        if not np.array_equal(arrays[f"{design}_correct_best"], expected_correct):
            raise DecisionContractError(f"C85T V3 S9 correct-best semantics drifted: {design}")
        expected_regret = np.asarray(
            population[selected.astype(np.int64)] - population[true_best], dtype="<f8"
        )
        observed_regret = np.asarray(arrays[f"{design}_selection_regret"])
        if (
            observed_regret.dtype != np.dtype("<f8")
            or not np.isfinite(observed_regret).all()
            or np.any(observed_regret < 0)
            or not np.array_equal(observed_regret, expected_regret)
        ):
            raise DecisionContractError(f"C85T V3 S9 regret semantics drifted: {design}")
    for endpoint in ("selection_regret", "correct_best", "top2_coverage", "D_hat"):
        expected = np.asarray(
            arrays[f"passive_{endpoint}"].astype("<f8")
            - arrays[f"neyman_{endpoint}"].astype("<f8"),
            dtype="<f8",
        )
        observed = np.asarray(arrays[f"paired_passive_minus_neyman_{endpoint}"])
        if observed.dtype != np.dtype("<f8") or not np.array_equal(observed, expected):
            raise DecisionContractError(f"C85T V3 S9 paired endpoint drifted: {endpoint}")
    return _summarize_s9_arrays_v2(arrays, population)


def _validate_raw_digest_rows(rows: Sequence[Mapping[str, str]]) -> None:
    required = {
        "replicate_id",
        "L_sha256",
        "H_sha256",
        "combined_sha256",
        "dtype",
        "L_count",
        "H_count",
    }
    if len(rows) != 4096:
        raise DecisionContractError("C85T V3 S9 raw-digest row count drifted")
    for replicate, row in enumerate(rows):
        if set(row) != required or int(row["replicate_id"]) != replicate:
            raise DecisionContractError("C85T V3 S9 raw-digest IDs drifted")
        if row["dtype"] != "<i8" or row["L_count"] != "51" or row["H_count"] != "46":
            raise DecisionContractError("C85T V3 S9 raw-digest dtype/count drifted")
        for key in ("L_sha256", "H_sha256", "combined_sha256"):
            if not _HEX_256.fullmatch(row[key]):
                raise DecisionContractError(f"C85T V3 S9 digest format drifted: {key}")


def _validate_proof_candidates(
    root: Path, statements: Mapping[str, str]
) -> list[dict[str, str]]:
    rows = _read_csv(root / "proof_candidate_dispositions.csv")
    if len(rows) != 7 or [row.get("theorem_id") for row in rows] != list(THEOREM_IDS):
        raise DecisionContractError("C85T V3 proof disposition coverage drifted")
    for row in rows:
        theorem_id = row["theorem_id"]
        if (
            row.get("historical_status") != "OPEN"
            or row.get("formal_status") != "OPEN"
            or row.get("candidate_disposition") not in _ALLOWED_DISPOSITIONS
            or row.get("check_class") != _CHECK_CLASS
        ):
            raise DecisionContractError(f"C85T V3 proof disposition drifted: {theorem_id}")
        proof_path = root / "c85t_proof_candidates" / PROOF_FILENAMES[theorem_id]
        if not proof_path.is_file() or sha256_file(proof_path) != row.get(
            "proof_candidate_sha256"
        ):
            raise DecisionContractError(f"C85T V3 proof hash drifted: {theorem_id}")
        text = proof_path.read_text()
        match = _STATEMENT_SHA.search(text)
        expected_statement_sha = hashlib.sha256(
            statements[theorem_id].encode("utf-8")
        ).hexdigest()
        if not match or match.group(1) != expected_statement_sha:
            raise DecisionContractError(f"C85T V3 statement SHA drifted: {theorem_id}")
        if "not an independent proof review" not in text or "`OPEN`" not in text:
            raise DecisionContractError(f"C85T V3 proof governance drifted: {theorem_id}")
    return rows


def validate_result_semantics_v3(
    root: Path,
    *,
    context: ValidatedC85TExecutionContext,
    contract: Mapping[str, Any],
    statements: Mapping[str, str],
    shadow_expected_exact: Mapping[str, Any] | None = None,
    shadow_expected_s9_arrays: Mapping[str, np.ndarray] | None = None,
    shadow_expected_s9_digest_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive every V3 count and identity from persisted artifacts."""

    validate_registered_execution_context(context)
    root = root.resolve()
    exact_path = root / "exact_scenario_results.json"
    summary_path = root / "monte_carlo_summary.json"
    result_path = root / "C85T_RESULT.json"
    if not exact_path.is_file() or not summary_path.is_file() or not result_path.is_file():
        raise DecisionContractError("C85T V3 core result artifact is absent")
    exact = json.loads(exact_path.read_text())
    _assert_exact_scenario_schema(exact)
    scope = context.lock.get("execution_scope", "REGISTERED_C85T")
    if scope == "REGISTERED_C85T":
        if any(
            value is not None
            for value in (
                shadow_expected_exact,
                shadow_expected_s9_arrays,
                shadow_expected_s9_digest_rows,
            )
        ):
            raise DecisionContractError("shadow expectations cannot enter registered replay")
        expected_exact = execute_registered_exact_scenarios_v3(contract, context=context)
    elif scope == "SHADOW_READINESS_ONLY":
        if shadow_expected_exact is None:
            raise DecisionContractError("shadow semantic replay requires shadow exact fixture")
        expected_exact = dict(shadow_expected_exact)
    else:
        raise DecisionContractError("unknown C85T V3 execution scope")
    if exact != expected_exact:
        raise DecisionContractError("C85T V3 exact scenario replay drifted")

    scenarios = {row["id"]: row for row in contract["scenarios"]}
    if tuple(scenarios) != SCENARIO_IDS:
        raise DecisionContractError("C85T V3 contract scenario order drifted")
    summaries = json.loads(summary_path.read_text())
    for scenario_id in ("S6", "S7"):
        arrays = read_deterministic_npz(root / f"{scenario_id}_replicates.npz")
        replay = _validate_near_arrays(
            scenario_id,
            arrays,
            utilities=np.asarray(scenarios[scenario_id]["utilities"][0], dtype="<f8"),
            geometry=exact[scenario_id],
        )
        if summaries.get(scenario_id) != replay:
            raise DecisionContractError(f"{scenario_id} V3 aggregate replay drifted")

    s9_arrays = read_deterministic_npz(root / "S9_replicates.npz")
    population = np.asarray(
        [float(value) for value in summaries["S9_population_mean_losses"]], dtype="<f8"
    )
    s9_replay = _validate_s9_arrays(s9_arrays, population)
    for key in ("analytic_variance", "universal_active_superiority_claim"):
        s9_replay[key] = summaries["S9"][key]
    if summaries.get("S9") != s9_replay:
        raise DecisionContractError("C85T V3 S9 aggregate replay drifted")
    digest_rows = _read_csv(root / "S9_raw_draw_digest_registry.csv")
    _validate_raw_digest_rows(digest_rows)
    if scope == "REGISTERED_C85T":
        expected_summary, expected_arrays, expected_digest_rows = (
            replay_registered_s9_artifacts_v3(contract, context=context)
        )
        if any(
            not np.array_equal(s9_arrays[name], expected_arrays[name])
            for name in sorted(expected_arrays)
        ) or set(s9_arrays) != set(expected_arrays):
            raise DecisionContractError("C85T V3 S9 deterministic array replay drifted")
        expected_rows = [{key: str(value) for key, value in row.items()} for row in expected_digest_rows]
        if digest_rows != expected_rows:
            raise DecisionContractError("C85T V3 S9 deterministic digest replay drifted")
        if expected_summary != summaries["S9"]:
            raise DecisionContractError("C85T V3 S9 deterministic summary replay drifted")
    else:
        if shadow_expected_s9_arrays is None or shadow_expected_s9_digest_rows is None:
            raise DecisionContractError("shadow S9 semantic replay fixture is incomplete")
        if set(s9_arrays) != set(shadow_expected_s9_arrays) or any(
            not np.array_equal(s9_arrays[name], shadow_expected_s9_arrays[name])
            for name in s9_arrays
        ):
            raise DecisionContractError("shadow S9 persisted array replay drifted")
        expected_rows = [
            {key: str(value) for key, value in row.items()}
            for row in shadow_expected_s9_digest_rows
        ]
        if digest_rows != expected_rows:
            raise DecisionContractError("shadow S9 digest replay drifted")

    proof_rows = _validate_proof_candidates(root, statements)
    copied_receipt = root / "authorization_consumed.json"
    if (
        not copied_receipt.is_file()
        or sha256_file(copied_receipt) != context.external_consumption_receipt_sha256
        or copied_receipt.read_bytes()
        != context.external_consumption_receipt_path.read_bytes()
    ):
        raise DecisionContractError("C85T V3 copied authorization receipt drifted")
    result = json.loads(result_path.read_text())
    required_identity = {
        "schema_version": RESULT_SCHEMA_V3,
        "final_gate": SUCCESS_GATE_V3,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_id": context.authorization_id,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "HEAD": context.head,
        "scenario_count": len(exact),
        "S6_S7_logical_replicate_rows": 8192,
        "S9_logical_replicate_design_rows": 8192,
        "S9_raw_draw_digest_rows": len(digest_rows),
        "proof_candidate_count": len(proof_rows),
        "formal_theorem_statuses": {theorem_id: "OPEN" for theorem_id in THEOREM_IDS},
        "real_project_data_access": 0,
        "active_acquisition": 0,
        "C85V_authorized": False,
        "C85E_authorized": False,
        "manuscript_modified": False,
    }
    for key, expected in required_identity.items():
        if result.get(key) != expected:
            raise DecisionContractError(f"C85T V3 result identity drifted: {key}")
    return {
        "schema_version": SEMANTIC_RECEIPT_SCHEMA_V3,
        "scenario_results": len(exact),
        "scenario_keys": list(SCENARIO_IDS),
        "S6_logical_replicate_rows": 4096,
        "S7_logical_replicate_rows": 4096,
        "S6_S7_logical_replicate_rows": 8192,
        "S9_logical_replicate_design_rows": 8192,
        "S9_raw_draw_digest_rows": len(digest_rows),
        "proof_candidates": len(proof_rows),
        "formal_theorem_status_OPEN": 7,
        "rng_digest_replay_rows": len(digest_rows),
        "rng_replay_scope": scope,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "execution_lock_sha256": context.execution_lock_sha256,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "protected_counters_zero": True,
        "status": "SEMANTIC_REPLAY_PASS",
    }


__all__ = (
    "RESULT_SCHEMA_V3",
    "SEMANTIC_RECEIPT_SCHEMA_V3",
    "SUCCESS_GATE_V3",
    "validate_result_semantics_v3",
)
