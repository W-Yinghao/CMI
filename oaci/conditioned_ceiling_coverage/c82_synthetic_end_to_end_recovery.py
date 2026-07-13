"""End-to-end synthetic calibration for the C82 recovery entrypoint."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from . import c82_post_c81_baseline_recovery as recovery


BASE_SEED = 820082
SCENARIOS = {
    "same_method_A": recovery.GATE_A,
    "same_method_B": recovery.GATE_B,
    "stable_C": recovery.GATE_C,
    "different_A_methods": recovery.GATE_D,
    "different_B_methods": recovery.GATE_D,
    "blocker_E": recovery.GATE_E,
}


def _rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{BASE_SEED}|{label}".encode("ascii")).digest()[:8], "little")
    return np.random.default_rng(seed)


def _regret(scenario: str, seed: int, method: str, target: int, level: int) -> float:
    base = {
        "S1": 0.65,
        "B0": 0.58,
        "B1": 0.62,
        "B2": 0.60,
        "B3": 0.61,
        "B4O": 0.59,
        "B4S": 0.60,
        "B5": 0.0,
    }.get(method, 0.64)
    if scenario == "same_method_A" and method == "U7":
        base = 0.30
    elif scenario == "same_method_B" and method == "U5":
        base = 0.48
    elif scenario == "different_A_methods":
        if (seed == 3 and method == "U7") or (seed == 4 and method == "U14"):
            base = 0.30
    elif scenario == "different_B_methods":
        if (seed == 3 and method == "U5") or (seed == 4 and method == "U14"):
            base = 0.48
    variation = ((target * 3 + level + seed) % 5 - 2) * 0.001
    return float(np.clip(base + variation, 0.0, 1.0))


def _regime(method: str) -> str:
    if method == "B0":
        return "RANDOM"
    if method == "B5":
        return "ORACLE"
    if method == "B1":
        return "ERM"
    if method in {"B2", "B4O", "U1", "U3", "U5", "U7", "U11", "U13", "U15"}:
        return "OACI"
    return "SRC"


def synthetic_fixture(
    scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown C82 synthetic scenario: {scenario}")
    rows: list[dict[str, Any]] = []
    q0_rows: list[dict[str, Any]] = []
    measurement: list[dict[str, Any]] = []
    for seed in recovery.SEEDS:
        for target in recovery.PRIMARY_TARGETS:
            for level in recovery.LEVELS:
                utility = np.linspace(0.08, 0.92, recovery.CANDIDATES)
                utility = np.clip(utility + ((target + seed + level) % 3 - 1) * 0.002, 0.0, 1.0)
                for method_index, method in enumerate(recovery.CONTEXT_METHODS):
                    regret = _regret(scenario, seed, method, target, level)
                    top1 = float(max(0.0, 0.10 - regret * 0.08))
                    top5 = float(max(top1, 0.42 - regret * 0.12))
                    top10 = float(max(top5, 0.66 - regret * 0.10))
                    row = {
                        "seed": seed,
                        "target": target,
                        "level": level,
                        "method_id": method,
                        "standardized_regret": regret,
                        "selected_utility": 1.0 - 0.65 * regret,
                        "top1": top1,
                        "top5": top5,
                        "top10": top10,
                        "coverage_top1": min(1.0, top1 + 0.10),
                        "coverage_top5": min(1.0, top5 + 0.15),
                        "coverage_top10": min(1.0, top10 + 0.18),
                        "selected_regime": _regime(method),
                        "evaluation_label_access_after_selection_freeze": True,
                        "same_label_oracle_accessed": False,
                        "target4_primary": False,
                    }
                    # Deliberately vary insertion order; canonical validation must ignore it.
                    if (method_index + target + level) % 2:
                        row = {key: row[key] for key in reversed(tuple(row))}
                    rows.append(row)
                    if method in recovery.RANK_METHODS:
                        rng = _rng(f"{scenario}|{seed}|{target}|{level}|{method}")
                        strength = 0.90 if method in {"U7", "U15", "S1"} else 0.45
                        score = strength * utility + (1.0 - strength) * rng.normal(0.5, 0.18, len(utility))
                        if method in recovery.PERFORMANCE_ESTIMATE_METHODS:
                            score = np.clip(score, 0.0, 1.0)
                        measurement.append({
                            "seed": seed, "target": target, "level": level, "method_id": method,
                            "scores": score, "utility": utility,
                        })
                budget_regret = {
                    "1": 0.35, "2": 0.30, "4": 0.26, "8": 0.22,
                    "16": 0.19, "32": 0.16, "FULL": 0.13,
                }
                for budget in recovery.Q0_BUDGETS:
                    regret = budget_regret[budget] + ((target + seed + level) % 5 - 2) * 0.001
                    q0_rows.append({
                        "seed": seed, "target": target, "level": level, "budget": budget,
                        "standardized_regret": regret,
                        "top1": 0.04 + 0.005 * recovery.Q0_BUDGETS.index(budget),
                        "top5": 0.18 + 0.02 * recovery.Q0_BUDGETS.index(budget),
                        "top10": 0.31 + 0.025 * recovery.Q0_BUDGETS.index(budget),
                        "coverage_top1": 0.14 + 0.02 * recovery.Q0_BUDGETS.index(budget),
                        "coverage_top5": 0.45 + 0.03 * recovery.Q0_BUDGETS.index(budget),
                        "coverage_top10": 0.65 + 0.025 * recovery.Q0_BUDGETS.index(budget),
                    })
    identity = {
        "mode": "synthetic_fixture",
        "manifest_self_sha256": recovery.canonical_sha256({"scenario": scenario, "kind": "manifest"}),
        "payload_sha256": recovery.canonical_sha256({"scenario": scenario, "kind": "payload"}),
        "contexts": 32,
        "methods": 19,
        "recomputed": False,
    }
    return rows, q0_rows, measurement, identity


def _write_report_csv(path: Path, fields: list[str], rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty C82P synthetic report: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_result(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "C82_POST_C81_BASELINE_RECOVERY.json").read_text())


def generate() -> dict[str, Any]:
    recovery.load_protocol()
    taxonomy_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    maxT_rows: list[dict[str, Any]] = []
    loto_rows: list[dict[str, Any]] = []
    atomic_rows: list[dict[str, Any]] = []
    same_method_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="c82p-e2e-") as temporary:
        root = Path(temporary)
        for scenario, expected in SCENARIOS.items():
            rows, q0, measurement, identity = synthetic_fixture(scenario)
            directory = root / scenario
            run = recovery.run_recovery(
                method_context_rows=rows,
                q0_rows=q0,
                measurement_records=measurement,
                selection_identity=identity,
                final_directory=directory,
                synthetic=True,
                blocker=scenario == "blocker_E",
            )
            result = _load_result(directory)
            observed = result["primary_taxonomy"]
            passed = observed == expected
            taxonomy_rows.append({
                "scenario": scenario, "expected_gate": expected, "observed_gate": observed,
                "passed": int(passed), "real_field_used": 0,
            })
            manifest_rows.append({
                "scenario": scenario,
                "artifact_manifest_sha256": run["artifact_manifest_sha256"],
                "result_sha256": run["result_sha256"],
                "table_count": run["table_count"],
                "method_context_rows": run["method_context_rows"],
                "all_required_artifacts_present": int(
                    run["table_count"] == len(recovery.TABLE_FIELDS)
                    and (directory / "objective_dependence_Q3.csv").is_file()
                    and (directory / "information_class_summary_Q5.csv").is_file()
                    and (directory / "accuracy_on_the_line_diagnostic.csv").is_file()
                ),
                "passed": int(
                    run["table_count"] == len(recovery.TABLE_FIELDS)
                    and run["method_context_rows"] == 672
                ),
                "real_field_used": 0,
            })
            with (directory / "seed_method_Q1_Q2.csv").open(newline="") as handle:
                evidence = list(csv.DictReader(handle))
            maxT_rows.append({
                "scenario": scenario,
                "Q1_rows": len(evidence),
                "Q2_rows": len(evidence),
                "all_pvalues_in_unit_interval": int(all(
                    0.0 <= float(row["Q1_maxT_p"]) <= 1.0
                    and 0.0 <= float(row["Q2_maxT_p"]) <= 1.0
                    for row in evidence
                )),
                "same_target_sign_vectors": 256,
                "passed": int(len(evidence) == 12),
                "real_field_used": 0,
            })
            with (directory / "leave_one_target_method_stability.csv").open(newline="") as handle:
                panels = list(csv.DictReader(handle))
            loto_rows.append({
                "scenario": scenario,
                "panels": len(panels),
                "same_method_column_present": int(all("supporting_same_methods" in row for row in panels)),
                "preserved": sum(int(row["panel_preserved"]) for row in panels),
                "passed": int(len(panels) == 16),
                "real_field_used": 0,
            })
            if scenario in {"same_method_A", "same_method_B", "different_A_methods", "different_B_methods", "stable_C"}:
                same_method_rows.append({
                    "test_id": scenario,
                    "seed3_qualifying_methods": "|".join(result["A_seed3"] or result["B_seed3"]) or "NONE",
                    "seed4_qualifying_methods": "|".join(result["A_seed4"] or result["B_seed4"]) or "NONE",
                    "common_methods": "|".join(result["A_intersection"] or result["B_intersection"]) or "NONE",
                    "expected_gate": expected,
                    "status": "PASS" if passed else "FAIL",
                })

        rows, q0, measurement, identity = synthetic_fixture("same_method_A")
        reordered = recovery.validate_method_context_rows(rows)
        atomic_rows.append({
            "test_id": "different_dictionary_insertion_order",
            "expected": "SUCCESS", "observed": "SUCCESS" if len(reordered) == 672 else "FAIL",
            "final_directory_visible": 0, "passed": int(len(reordered) == 672),
        })
        missing = dict(rows[0])
        missing.pop("top10")
        try:
            recovery.validate_method_context_rows([missing, *rows[1:]])
            missing_observed = "UNEXPECTED_SUCCESS"
        except recovery.C82ValidationError:
            missing_observed = "REJECTED_BEFORE_WRITE"
        atomic_rows.append({
            "test_id": "missing_field", "expected": "REJECTED_BEFORE_WRITE",
            "observed": missing_observed, "final_directory_visible": 0,
            "passed": int(missing_observed == "REJECTED_BEFORE_WRITE"),
        })
        unknown = dict(rows[0])
        unknown["unregistered"] = 1
        try:
            recovery.validate_method_context_rows([unknown, *rows[1:]])
            unknown_observed = "UNEXPECTED_SUCCESS"
        except recovery.C82ValidationError:
            unknown_observed = "REJECTED_BEFORE_WRITE"
        atomic_rows.append({
            "test_id": "unknown_field", "expected": "REJECTED_BEFORE_WRITE",
            "observed": unknown_observed, "final_directory_visible": 0,
            "passed": int(unknown_observed == "REJECTED_BEFORE_WRITE"),
        })
        partial = root / "partial_failure"
        try:
            recovery.run_recovery(
                method_context_rows=rows, q0_rows=q0, measurement_records=measurement,
                selection_identity=identity, final_directory=partial, synthetic=True,
                inject_partial_write_failure=True,
            )
            partial_observed = "UNEXPECTED_SUCCESS"
        except recovery.C82ValidationError:
            partial_observed = "FAILED_NO_FINAL_DIRECTORY"
        atomic_rows.append({
            "test_id": "partial_table_write", "expected": "FAILED_NO_FINAL_DIRECTORY",
            "observed": partial_observed, "final_directory_visible": int(partial.exists()),
            "passed": int(partial_observed == "FAILED_NO_FINAL_DIRECTORY" and not partial.exists()),
        })
        post_eval = root / "post_eval_failure"
        try:
            recovery.run_recovery(
                method_context_rows=rows, q0_rows=q0, measurement_records=measurement,
                selection_identity=identity, final_directory=post_eval, synthetic=True,
                inject_post_evaluation_failure=True,
            )
            post_observed = "UNEXPECTED_SUCCESS"
            consumed = 0
        except recovery.C82PostEvaluationFailure as exc:
            post_observed = exc.final_gate
            consumed = int(exc.authorization_consumed)
        atomic_rows.append({
            "test_id": "post_evaluation_exception", "expected": recovery.GATE_E,
            "observed": post_observed, "final_directory_visible": int(post_eval.exists()),
            "passed": int(post_observed == recovery.GATE_E and consumed == 1 and not post_eval.exists()),
        })

    recovery.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _write_report_csv(
        recovery.TABLE_DIR / "synthetic_taxonomy_calibration.csv",
        ["scenario", "expected_gate", "observed_gate", "passed", "real_field_used"], taxonomy_rows,
    )
    _write_report_csv(
        recovery.TABLE_DIR / "synthetic_end_to_end_result_manifest.csv",
        [
            "scenario", "artifact_manifest_sha256", "result_sha256", "table_count",
            "method_context_rows", "all_required_artifacts_present", "passed", "real_field_used",
        ], manifest_rows,
    )
    _write_report_csv(
        recovery.TABLE_DIR / "synthetic_maxT_noninferiority.csv",
        [
            "scenario", "Q1_rows", "Q2_rows", "all_pvalues_in_unit_interval",
            "same_target_sign_vectors", "passed", "real_field_used",
        ], maxT_rows,
    )
    _write_report_csv(
        recovery.TABLE_DIR / "synthetic_LOTO_calibration.csv",
        ["scenario", "panels", "same_method_column_present", "preserved", "passed", "real_field_used"],
        loto_rows,
    )
    _write_report_csv(
        recovery.TABLE_DIR / "atomic_result_freeze_test.csv",
        ["test_id", "expected", "observed", "final_directory_visible", "passed"], atomic_rows,
    )
    _write_report_csv(
        recovery.TABLE_DIR / "same_method_identity_test.csv",
        [
            "test_id", "seed3_qualifying_methods", "seed4_qualifying_methods", "common_methods",
            "expected_gate", "status",
        ], same_method_rows,
    )
    all_passed = all(row["passed"] for rows in (taxonomy_rows, manifest_rows, maxT_rows, loto_rows, atomic_rows) for row in rows)
    all_passed = all_passed and all(row["status"] == "PASS" for row in same_method_rows)
    if not all_passed:
        raise RuntimeError("C82 synthetic end-to-end recovery calibration failed")
    return {
        "taxonomy_scenarios": len(taxonomy_rows),
        "atomic_tests": len(atomic_rows),
        "same_method_tests": len(same_method_rows),
        "all_taxonomy_branches": sorted({row["observed_gate"].split("_")[0] for row in taxonomy_rows}),
        "all_passed": True,
        "real_field_used": False,
        "selection_payload_opened": False,
        "evaluation_view_opened": False,
    }


def main() -> int:
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
