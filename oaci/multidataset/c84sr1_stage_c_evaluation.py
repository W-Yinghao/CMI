"""C84SR1 Stage-C immutable-selection held-evaluation materialization."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import c84s_evaluation as evaluation
from .c84s_common import read_csv, read_json, require, sha256_file, write_json
from .c84sr1_analysis import run_analysis_and_freeze_v2
from .c84sr1_common import (
    FIXED_METHODS, MAXT_DRAWS, Q0_CHAINS, SCORE_METHODS,
)
from .c84sr1_context_enumerator import ContextDescriptor, enumerate_contexts
from .c84sr1_field_reader import FrozenZooReader, zoo_then_target_key
from .c84sr1_method_context_materialization import materialize_context
from .c84sr1_q0_store import read_context_shard


def _verify_selection_freeze(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "C84S_SELECTION_FREEZE_MANIFEST_V2.json"
    handoff_path = root / "C84S_STAGE_B_HANDOFF.json"
    require(manifest_path.is_file() and handoff_path.is_file(), "Stage-B freeze handoff absent")
    manifest = read_json(manifest_path)
    handoff = read_json(handoff_path)
    manifest_sha = sha256_file(manifest_path)
    require(manifest["status"] == "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
            "Stage-B selection is not frozen")
    require(handoff["selection_freeze_manifest_sha256"] == manifest_sha,
            "Stage-B handoff manifest identity drift")
    require(handoff["evaluation_descriptor_received"] is False,
            "evaluation descriptor reached Stage B")
    require(manifest["evaluation_label_descriptor_received"] is False,
            "evaluation descriptor recorded in selection freeze")
    require(manifest["same_label_oracle_accessed"] is False,
            "oracle access recorded in selection freeze")
    for identity in manifest["artifacts"].values():
        path = root / identity["path"]
        require(path.is_file() and sha256_file(path) == identity["sha256"],
                f"Stage-B artifact identity drift: {identity['path']}")
    return manifest, {
        "status": manifest["status"], "sha256": manifest_sha,
        "root": str(root),
    }


def _load_evaluation_seal(path: str | Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    seal = read_json(path)
    receipt_sha = str(seal.pop("receipt_identity_sha256"))
    from .c84s_common import canonical_sha256
    require(receipt_sha == canonical_sha256(seal), "Stage-A evaluation seal identity drift")
    seal["receipt_identity_sha256"] = receipt_sha
    require(seal["stage"] == "Stage_A_evaluation_seal", "Stage-C evaluation seal stage drift")
    require(seal["released_to_Stage_B"] is False, "evaluation descriptor was released to Stage B")
    descriptor = dict(seal["evaluation_descriptor"])
    require(descriptor["kind"] == "evaluation", "Stage-C label-view kind drift")
    root = Path(descriptor["root"])
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file() and sha256_file(manifest_path) == descriptor["manifest_sha256"],
            "evaluation label-view manifest drift")
    manifest = read_json(manifest_path)
    require(manifest["kind"] == "evaluation" and manifest["candidate_artifacts"] == 0 and
            manifest["EEG_arrays"] == 0, "evaluation view contains forbidden payload")
    table_path = root / manifest["table"]["path"]
    require(sha256_file(table_path) == manifest["table"]["sha256"],
            "evaluation label table SHA drift")
    rows = read_csv(table_path)
    require(len(rows) == int(descriptor["row_count"]), "evaluation label row count drift")
    return descriptor, rows


def _context_prefix(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["dataset"]), str(row["target_subject_id"]), str(row["panel"]),
        int(row["training_seed"]), int(row["level"]),
    )


def _load_selection_tables(
    root: Path, contexts: Sequence[ContextDescriptor],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, int]],
    dict[str, dict[str, str]],
]:
    by_id = {row.context_id: row for row in contexts}
    score_vectors: dict[str, dict[str, np.ndarray]] = {
        context_id: {method: np.full(81, np.nan) for method in SCORE_METHODS}
        for context_id in by_id
    }
    score_path = root / "candidate_scores.csv"
    score_rows = 0
    with score_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            context_id = str(row["context_id"])
            require(context_id in by_id, "score row has unknown context")
            method = str(row["method_id"])
            require(method in SCORE_METHODS, "score row method drift")
            index = int(row["candidate_index"])
            require(0 <= index < 81, "score candidate index drift")
            require(str(row["candidate_id"]) == by_id[context_id].candidates[index].unit_id,
                    "score candidate identity drift")
            require(np.isnan(score_vectors[context_id][method][index]), "duplicate score row")
            score_vectors[context_id][method][index] = float(row["raw_score"])
            score_rows += 1
    require(score_rows == len(contexts) * len(SCORE_METHODS) * 81,
            "candidate-score exact coverage drift")
    require(all(np.all(np.isfinite(values)) for methods in score_vectors.values()
                for values in methods.values()), "candidate-score vector is incomplete")

    expected_ranks: dict[tuple[str, str], np.ndarray] = {}
    for context_id, methods in score_vectors.items():
        for method, values in methods.items():
            order = np.lexsort((np.arange(81), -values))
            ranks = np.empty(81, dtype=int)
            ranks[order] = np.arange(1, 82)
            expected_ranks[(context_id, method)] = ranks

    rank_rows = 0
    seen_ranks: set[tuple[str, str, int]] = set()
    with (root / "candidate_ranks.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context_id, method, index = str(row["context_id"]), str(row["method_id"]), int(row["candidate_index"])
            require(context_id in by_id and method in SCORE_METHODS and 0 <= index < 81,
                    "candidate-rank identity drift")
            key = (context_id, method, index)
            require(key not in seen_ranks, "duplicate rank row")
            seen_ranks.add(key)
            require(int(row["rank"]) == int(expected_ranks[(context_id, method)][index]),
                    "score/rank replay drift")
            require(str(row["candidate_id"]) == by_id[context_id].candidates[index].unit_id,
                    "rank candidate identity drift")
            rank_rows += 1
    require(rank_rows == score_rows, "candidate-rank exact coverage drift")

    fixed: dict[str, dict[str, int]] = {context_id: {} for context_id in by_id}
    fixed_rows = 0
    with (root / "fixed_default_selections.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context_id, method = str(row["context_id"]), str(row["method_id"])
            require(context_id in by_id and method in FIXED_METHODS, "fixed-selection identity drift")
            index = int(row["selected_candidate_index"])
            require(method not in fixed[context_id], "duplicate fixed selection")
            require(str(row["selected_candidate_id"]) == by_id[context_id].candidates[index].unit_id,
                    "fixed selected candidate drift")
            fixed[context_id][method] = index
            fixed_rows += 1
    require(fixed_rows == len(contexts) * len(FIXED_METHODS) and
            all(set(values) == set(FIXED_METHODS) for values in fixed.values()),
            "fixed-selection exact coverage drift")

    shards: dict[str, dict[str, str]] = {}
    with (root / "q0_selection_shard_index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context_id = str(row["context_id"])
            require(context_id in by_id and context_id not in shards, "Q0 shard-index context drift")
            shards[context_id] = dict(row)
    require(set(shards) == set(by_id), "Q0 shard-index exact context coverage drift")
    return score_vectors, fixed, shards


def _evaluation_labels_for_context(
    context: ContextDescriptor,
    target_trial_ids: np.ndarray,
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in evaluation_rows
            if str(row["dataset"]) == context.dataset and
            str(row["target_subject_id"]) == context.target_subject_id]
    require(rows, "evaluation labels absent for target context")
    label_by_id = {str(row["target_trial_id"]): int(row["canonical_class_label"]) for row in rows}
    require(len(label_by_id) == len(rows), "duplicate evaluation trial identity")
    index = {str(value): position for position, value in enumerate(target_trial_ids.tolist())}
    require(set(label_by_id).issubset(index), "evaluation trial is outside frozen target artifact")
    ordered_ids = [str(value) for value in target_trial_ids.tolist() if str(value) in label_by_id]
    require(set(ordered_ids) == set(label_by_id), "evaluation trial alignment drift")
    return (
        np.asarray([index[value] for value in ordered_ids], dtype=int),
        np.asarray([label_by_id[value] for value in ordered_ids], dtype=int),
    )


def run_stage_c(
    *,
    selection_root: str | Path,
    evaluation_seal_path: str | Path,
    final_root: str | Path,
    contexts: Sequence[ContextDescriptor] | None = None,
    context_loader: Callable[[ContextDescriptor], Mapping[str, Any]] | None = None,
    utility_provider: Callable[[ContextDescriptor, Mapping[str, Any], Sequence[Mapping[str, Any]]], tuple[np.ndarray, np.ndarray]] | None = None,
    q0_chains: int = Q0_CHAINS,
    maxT_draws: int = MAXT_DRAWS,
    synthetic: bool = False,
    blocker: bool = False,
    failure_injection_context: int | None = None,
) -> dict[str, Any]:
    selection_root = Path(selection_root)
    manifest, selection_identity = _verify_selection_freeze(selection_root)
    evaluation_descriptor, evaluation_rows = _load_evaluation_seal(evaluation_seal_path)
    require(int(manifest["Q0_chains"]) == int(q0_chains), "Stage-B/Stage-C Q0 chain count drift")
    contexts = sorted(
        list(enumerate_contexts() if contexts is None else contexts),
        key=zoo_then_target_key,
    )
    if not synthetic:
        require(len(contexts) == 944 and q0_chains == Q0_CHAINS and maxT_draws == MAXT_DRAWS,
                "real Stage-C scope reduction")
    scores, fixed, shards = _load_selection_tables(selection_root, contexts)
    loader = FrozenZooReader(include_source=False) if context_loader is None else context_loader
    method_rows: list[dict[str, Any]] = []
    q0_regime_rows: list[dict[str, Any]] = []
    q0_mc_rows: list[dict[str, Any]] = []
    for position, context in enumerate(contexts):
        if failure_injection_context is not None and position == failure_injection_context:
            raise RuntimeError("injected Stage-C materialization failure")
        data = dict(loader(context))
        candidate_ids = [row.unit_id for row in context.candidates]
        require(list(map(str, data["candidate_ids"])) == candidate_ids,
                "Stage-C candidate identity drift")
        if utility_provider is None:
            evaluation_index, labels = _evaluation_labels_for_context(
                context, np.asarray(data["target_trial_ids"], dtype=str), evaluation_rows,
            )
            utility, evaluation_metrics = evaluation.context_candidate_utility(
                np.asarray(data["target_logits"], dtype=float)[:, evaluation_index], labels,
            )
        else:
            utility, evaluation_metrics = utility_provider(context, data, evaluation_rows)
        shard_identity = shards[context.context_id]
        q0_payload, replay = read_context_shard(
            selection_root / shard_identity["path"],
            expected_sha256=shard_identity["sha256"], chains=q0_chains,
        )
        require(replay["context_id"] == context.context_id, "Q0 shard/context mismatch")
        rows, regimes, diagnostics = materialize_context(
            identity=context.identity(), candidate_ids=candidate_ids,
            regimes=[row.regime for row in context.candidates],
            utility=np.asarray(utility, dtype=float),
            evaluation_metrics=np.asarray(evaluation_metrics, dtype=float),
            score_vectors=scores[context.context_id],
            fixed_selected_indices=fixed[context.context_id],
            q0_payload=q0_payload, q0_chains=q0_chains,
        )
        method_rows.extend(rows)
        q0_regime_rows.extend(regimes)
        q0_mc_rows.extend(diagnostics)
    if not synthetic:
        require(len(method_rows) == 18608, "Stage-C method-context exact row count drift")
        require(len(q0_regime_rows) == 13344, "Stage-C Q0 regime exact row count drift")
        require(len(q0_mc_rows) == 4448, "Stage-C Q0 MC exact row count drift")
    return run_analysis_and_freeze_v2(
        method_rows, q0_regime_rows=q0_regime_rows, q0_mc_rows=q0_mc_rows,
        selection_freeze_identity=selection_identity,
        evaluation_view_identity=evaluation_descriptor,
        final_root=final_root, draws=maxT_draws, blocker=blocker,
        synthetic=synthetic,
    )


def run_stage_c_real(
    *, selection_root: str | Path, evaluation_seal_path: str | Path,
    final_root: str | Path,
) -> dict[str, Any]:
    attempt_path = Path(final_root).parent / f"{Path(final_root).name}.stage_c_attempt.json"
    require(not attempt_path.exists(), "Stage-C attempt exists")
    attempt = {
        "schema_version": "c84sr1_stage_attempt_v1", "stage": "Stage_C",
        "status": "STARTED", "started_at_unix_ns": time.time_ns(),
        "selection_mutation": 0, "construction_descriptor_received": 0,
        "evaluation_descriptor_received": 1,
    }
    write_json(attempt_path, attempt)
    try:
        result = run_stage_c(
            selection_root=selection_root, evaluation_seal_path=evaluation_seal_path,
            final_root=final_root,
        )
        attempt.update({"status": "COMPLETE", "finished_at_unix_ns": time.time_ns(), "result": result})
        write_json(attempt_path, attempt)
        return result
    except BaseException as error:
        attempt.update({
            "status": "FAILED", "finished_at_unix_ns": time.time_ns(),
            "error_type": type(error).__name__, "error": str(error),
            "partial_final_root": Path(final_root).exists(),
        })
        write_json(attempt_path, attempt)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--evaluation-seal", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = run_stage_c_real(
        selection_root=args.selection_root, evaluation_seal_path=args.evaluation_seal,
        final_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
