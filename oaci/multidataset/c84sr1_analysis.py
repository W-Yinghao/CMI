"""C84SR1 nullable-measurement analysis and atomic V2 result publication."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import c84s_analysis as historical
from .c84s_common import (
    atomic_publish_directory, canonical_sha256, read_json, require,
    sha256_file, write_csv, write_json,
)
from .c84sr1_common import DATASET_TARGET_COUNTS, expected_methods
from .c84sr1_method_context_materialization import (
    METHOD_CONTEXT_FIELDS_V2, Q0_MC_FIELDS, Q0_REGIME_FIELDS,
)


MEASUREMENT_FIELDS_V2 = (
    "dataset", "method_id", "rank_measurement_applicable",
    "performance_estimate_applicable", "mean_Spearman", "mean_Kendall",
    "mean_pairwise_ordering_accuracy", "accuracy_estimation_MAE", "Q1_pass",
    "measurement_substitutes_for_regret",
)
REGIME_FIELDS_V2 = (
    "dataset", "method_id", "selected_regime", "weighted_contexts", "fraction",
)
RESULT_TABLE_FIELDS_V2 = {
    **historical.RESULT_TABLE_FIELDS,
    "method_context_decisions.csv": METHOD_CONTEXT_FIELDS_V2,
    "measurement_vs_decision.csv": MEASUREMENT_FIELDS_V2,
    "selected_regime_distribution.csv": REGIME_FIELDS_V2,
    "q0_selected_regime_distribution.csv": Q0_REGIME_FIELDS,
    "q0_monte_carlo_diagnostics.csv": Q0_MC_FIELDS,
}


def validate_method_context_rows_v2(
    rows: Sequence[Mapping[str, Any]],
    *, method_provider: Callable[[str], tuple[str, ...]] = expected_methods,
    expected_row_count: int = 18608,
) -> list[dict[str, Any]]:
    expected_fields = set(METHOD_CONTEXT_FIELDS_V2)
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    groups: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    targets: dict[str, set[str]] = defaultdict(set)
    for raw in rows:
        require(set(raw) == expected_fields, "C84SR1 method-context field-set drift")
        row = {field: raw[field] for field in METHOD_CONTEXT_FIELDS_V2}
        row["dataset"] = str(row["dataset"])
        row["target_subject_id"] = str(row["target_subject_id"])
        row["panel"] = str(row["panel"])
        row["training_seed"] = int(row["training_seed"])
        row["level"] = int(row["level"])
        row["method_id"] = str(row["method_id"])
        row["rank_measurement_applicable"] = int(row["rank_measurement_applicable"])
        row["performance_estimate_applicable"] = int(row["performance_estimate_applicable"])
        require(row["method_id"] in method_provider(row["dataset"]), "method identity drift")
        for field in (
            "standardized_regret", "selected_utility", "source_relative_regret_gain",
            "top1", "top5", "top10", "coverage",
        ):
            row[field] = float(row[field])
            require(np.isfinite(row[field]), f"nonfinite decision field: {field}")
        require(0 <= row["standardized_regret"] <= 1 + 1e-12, "regret outside [0,1]")
        require(0 <= row["selected_utility"] <= 1 + 1e-12, "utility outside [0,1]")
        require(0 <= row["top1"] <= row["top5"] <= row["top10"] <= 1,
                "top-k ordering/value drift")
        require(0 <= row["coverage"] <= 1, "coverage outside [0,1]")
        require(row["rank_measurement_applicable"] in (0, 1) and
                row["performance_estimate_applicable"] in (0, 1),
                "measurement applicability flag drift")
        for field in ("Spearman", "Kendall", "pairwise_ordering_accuracy"):
            value = row[field]
            if row["rank_measurement_applicable"]:
                require(value is not None and value != "", f"applicable rank metric absent: {field}")
                row[field] = float(value)
                require(np.isfinite(row[field]), f"nonfinite rank metric: {field}")
            else:
                require(value in (None, ""), f"inapplicable rank metric is non-null: {field}")
                row[field] = None
        mae = row["accuracy_estimation_MAE"]
        if row["performance_estimate_applicable"]:
            require(mae is not None and mae != "", "applicable performance MAE absent")
            row["accuracy_estimation_MAE"] = float(mae)
            require(np.isfinite(row["accuracy_estimation_MAE"]), "nonfinite performance MAE")
        else:
            require(mae in (None, ""), "inapplicable performance MAE is non-null")
            row["accuracy_estimation_MAE"] = None
        identity = tuple(row[field] for field in METHOD_CONTEXT_FIELDS_V2[:6])
        require(identity not in seen, "duplicate C84SR1 method-context row")
        seen.add(identity)
        context = tuple(row[field] for field in METHOD_CONTEXT_FIELDS_V2[:5])
        groups[context].add(row["method_id"])
        targets[row["dataset"]].add(row["target_subject_id"])
        output.append(row)
    require({dataset: len(values) for dataset, values in targets.items()} == DATASET_TARGET_COUNTS,
            "method-context target coverage drift")
    for context, methods in groups.items():
        require(methods == set(method_provider(str(context[0]))), f"context method coverage drift: {context}")
    require(len(groups) == 944 and len(output) == expected_row_count,
            "method-context exact arithmetic drift")
    lookup = {tuple(row[field] for field in METHOD_CONTEXT_FIELDS_V2[:6]): row for row in output}
    for context, methods in groups.items():
        source = lookup[context + ("S1",)]["standardized_regret"]
        for method in methods:
            row = lookup[context + (method,)]
            expected_gain = 0.0 if source <= 1e-15 else (source - row["standardized_regret"]) / source
            require(abs(row["source_relative_regret_gain"] - expected_gain) <= 1e-10,
                    f"source-relative gain drift: {context}/{method}")
    return sorted(output, key=lambda row: tuple(row[field] for field in METHOD_CONTEXT_FIELDS_V2[:6]))


def _historical_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Non-persisted projection; decision inference never uses measurement fields."""
    output = []
    for row in rows:
        output.append({
            **{field: row[field] for field in METHOD_CONTEXT_FIELDS_V2[:14]},
            "catastrophic_failure": 0,
            "Spearman": 0.0 if row["Spearman"] is None else row["Spearman"],
            "Kendall": 0.0 if row["Kendall"] is None else row["Kendall"],
            "pairwise_ordering_accuracy": (
                0.0 if row["pairwise_ordering_accuracy"] is None else row["pairwise_ordering_accuracy"]
            ),
            "accuracy_estimation_MAE": row["accuracy_estimation_MAE"],
        })
    return output


def _measurement_table(
    rows: Sequence[Mapping[str, Any]], dataset_decisions: Sequence[Mapping[str, Any]],
    *, method_provider: Callable[[str], tuple[str, ...]] = expected_methods,
) -> list[dict[str, Any]]:
    q1 = {(row["dataset"], row["method_id"]): int(row["Q1_pass"]) for row in dataset_decisions}
    output = []
    for dataset in DATASET_TARGET_COUNTS:
        for method in method_provider(dataset):
            method_rows = [row for row in rows if row["dataset"] == dataset and row["method_id"] == method]
            rank_flag = int(method_rows[0]["rank_measurement_applicable"])
            performance_flag = int(method_rows[0]["performance_estimate_applicable"])
            require(all(int(row["rank_measurement_applicable"]) == rank_flag for row in method_rows),
                    "rank applicability varies within method/dataset")
            require(all(int(row["performance_estimate_applicable"]) == performance_flag for row in method_rows),
                    "performance applicability varies within method/dataset")
            maes = [row["accuracy_estimation_MAE"] for row in method_rows if row["accuracy_estimation_MAE"] is not None]
            output.append({
                "dataset": dataset, "method_id": method,
                "rank_measurement_applicable": rank_flag,
                "performance_estimate_applicable": performance_flag,
                "mean_Spearman": None if not rank_flag else float(np.mean([row["Spearman"] for row in method_rows])),
                "mean_Kendall": None if not rank_flag else float(np.mean([row["Kendall"] for row in method_rows])),
                "mean_pairwise_ordering_accuracy": None if not rank_flag else float(np.mean([
                    row["pairwise_ordering_accuracy"] for row in method_rows
                ])),
                "accuracy_estimation_MAE": None if not performance_flag else float(np.mean(maes)),
                "Q1_pass": q1.get((dataset, method), 0),
                "measurement_substitutes_for_regret": 0,
            })
    return output


def _regime_table(
    rows: Sequence[Mapping[str, Any]], q0_regime_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    weights: dict[tuple[str, str, str], float] = defaultdict(float)
    totals: dict[tuple[str, str], float] = defaultdict(float)
    finite_q0 = {"Q0_B1", "Q0_B2", "Q0_B4", "Q0_B8", "Q0_B16", "Q0_B32"}
    for row in rows:
        key = (str(row["dataset"]), str(row["method_id"]))
        if row["method_id"] not in finite_q0:
            regime = str(row["selected_regime"])
            weights[key + (regime,)] += 1.0
            totals[key] += 1.0
    for row in q0_regime_rows:
        method = f"Q0_B{row['budget']}"
        key = (str(row["dataset"]), method)
        weights[key + (str(row["regime"]),)] += float(row["fraction"])
        totals[key] += float(row["fraction"])
    output = []
    for dataset, method, regime in sorted(weights):
        weight = weights[(dataset, method, regime)]
        total = totals[(dataset, method)]
        require(total > 0, "selected-regime total is zero")
        output.append({
            "dataset": dataset, "method_id": method, "selected_regime": regime,
            "weighted_contexts": weight, "fraction": weight / total,
        })
    return output


def derive_tables_v2(
    rows: Sequence[Mapping[str, Any]],
    *,
    q0_regime_rows: Sequence[Mapping[str, Any]],
    q0_mc_rows: Sequence[Mapping[str, Any]],
    draws: int,
    blocker: bool,
    method_provider: Callable[[str], tuple[str, ...]] = expected_methods,
    expected_row_count: int = 18608,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    normalized = validate_method_context_rows_v2(
        rows, method_provider=method_provider, expected_row_count=expected_row_count,
    )
    projected = historical.validate_method_context_rows(
        _historical_projection(normalized),
        expected_method_provider=method_provider,
        expected_row_count=expected_row_count,
    )
    tables, result = historical._derive_tables(
        projected, draws=draws, blocker=blocker,
        expected_method_provider=method_provider,
    )
    tables["method_context_decisions.csv"] = normalized
    tables["measurement_vs_decision.csv"] = _measurement_table(
        normalized, tables["dataset_Q1_Q2.csv"], method_provider=method_provider,
    )
    tables["selected_regime_distribution.csv"] = _regime_table(normalized, q0_regime_rows)
    tables["q0_selected_regime_distribution.csv"] = [dict(row) for row in q0_regime_rows]
    tables["q0_monte_carlo_diagnostics.csv"] = [dict(row) for row in q0_mc_rows]
    require(set(tables) == set(RESULT_TABLE_FIELDS_V2), "C84SR1 result-table set drift")
    for name, fields in RESULT_TABLE_FIELDS_V2.items():
        require(tables[name], f"empty C84SR1 result table: {name}")
        require(all(tuple(row) == fields for row in tables[name]), f"C84SR1 table field order drift: {name}")
    return tables, result


def run_analysis_and_freeze_v2(
    method_context_rows: Sequence[Mapping[str, Any]],
    *,
    q0_regime_rows: Sequence[Mapping[str, Any]],
    q0_mc_rows: Sequence[Mapping[str, Any]],
    selection_freeze_identity: Mapping[str, Any],
    evaluation_view_identity: Mapping[str, Any],
    final_root: str | Path,
    draws: int = 65536,
    blocker: bool = False,
    synthetic: bool = False,
    failure_injection_after: str | None = None,
    method_provider: Callable[[str], tuple[str, ...]] = expected_methods,
    expected_row_count: int = 18608,
    result_schema: str = "c84sr1_result_v2",
    manifest_schema: str = "c84sr1_result_artifact_manifest_v2",
) -> dict[str, Any]:
    require(selection_freeze_identity.get("status") == "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
            "Stage C lacks immutable selection freeze")
    require(len(str(selection_freeze_identity.get("sha256", ""))) == 64, "selection-freeze SHA drift")
    require(evaluation_view_identity.get("kind") == "evaluation", "Stage C evaluation descriptor drift")
    require(len(str(evaluation_view_identity.get("manifest_sha256", ""))) == 64,
            "evaluation-view SHA drift")
    tables, result = derive_tables_v2(
        method_context_rows, q0_regime_rows=q0_regime_rows,
        q0_mc_rows=q0_mc_rows, draws=draws, blocker=blocker,
        method_provider=method_provider, expected_row_count=expected_row_count,
    )
    result.update({
        "schema_version": result_schema,
        "selection_freeze_sha256": str(selection_freeze_identity["sha256"]),
        "evaluation_view_manifest_sha256": str(evaluation_view_identity["manifest_sha256"]),
        "synthetic": bool(synthetic),
    })

    def writer(staging: Path) -> None:
        artifacts = []
        for name in RESULT_TABLE_FIELDS_V2:
            digest = write_csv(staging / name, tables[name])
            artifacts.append({"path": name, "rows": len(tables[name]), "sha256": digest})
            if failure_injection_after == name:
                raise RuntimeError("injected C84SR1 Stage-C result failure")
        manifest = {
            "schema_version": manifest_schema,
            "selection_freeze_sha256": str(selection_freeze_identity["sha256"]),
            "evaluation_view_manifest_sha256": str(evaluation_view_identity["manifest_sha256"]),
            "table_count": len(artifacts), "artifacts": artifacts,
            "all_tables_validated_before_publication": True,
        }
        manifest_sha = write_json(staging / "C84S_RESULT_ARTIFACT_MANIFEST.json", manifest)
        for identity in artifacts:
            require(sha256_file(staging / identity["path"]) == identity["sha256"],
                    f"result artifact SHA drift: {identity['path']}")
        if failure_injection_after == "C84S_RESULT_ARTIFACT_MANIFEST.json":
            raise RuntimeError("injected C84SR1 post-manifest failure")
        final = {
            **result, "artifact_manifest_sha256": manifest_sha,
            "artifact_manifest_table_count": len(artifacts),
            "result_identity_sha256": canonical_sha256({
                "result": result, "artifact_manifest_sha256": manifest_sha,
            }),
        }
        write_json(staging / "C84S_RESULT.json", final)

    published = atomic_publish_directory(final_root, writer)
    final = read_json(published / "C84S_RESULT.json")
    require(sha256_file(published / "C84S_RESULT_ARTIFACT_MANIFEST.json") ==
            final["artifact_manifest_sha256"], "published result identity drift")
    return final
