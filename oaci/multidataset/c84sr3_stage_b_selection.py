"""C84SR3 Stage-B selection freeze with availability-safe Q0 budgets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import c84s_selectors as selectors
from .c84s_common import (
    atomic_publish_directory, require, sha256_file, write_csv, write_json,
)
from .c84sr1_common import FIXED_METHODS, Q0_CHAINS, SCORE_METHODS
from .c84sr1_context_enumerator import ContextDescriptor, enumerate_contexts
from .c84sr1_field_reader import FrozenZooReader, zoo_then_target_key
from .c84sr1_stage_b_selection import (
    ACCESS_FIELDS, CONTEXT_FIELDS, CSVStream, FIXED_FIELDS, Q0_COVERAGE_FIELDS,
    Q0_REGIME_FIELDS, RANK_FIELDS, SAMPLE_FIELDS, SCORE_FIELDS,
    _context_prefix, _fixed_indices, _load_construction_handoff,
)
from .c84sr3_common import (
    DATASET_TARGET_COUNTS, EXPECTED_CONSTRUCTION_CLASS_RANGE,
    HISTORICAL_SECONDARY_BUDGETS, OPERATIVE_FINITE_BUDGETS,
    Q0_RECORDS, Q0_SAMPLE_DIGEST_ROWS, finite_budgets,
)
from .c84sr3_q0_store import (
    SHARD_INDEX_FIELDS, build_context_payload, write_context_shard,
)


AVAILABILITY_FIELDS = (
    "dataset", "budget", "budget_role", "targets", "feasible_targets",
    "infeasible_targets", "min_labels_per_class", "max_labels_per_class",
    "operative", "disposition",
)


def _q0_sample_plan_identity(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in (
        "finite_chain", "finite_chain_seed", "finite_budget_code",
        "finite_sample_size", "finite_sample_digest", "FULL_sample_size",
        "FULL_sample_digest",
    ):
        value = np.ascontiguousarray(np.asarray(payload[name]))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def construction_budget_availability(
    construction_rows: Sequence[Mapping[str, Any]],
    contexts: Sequence[ContextDescriptor], *, synthetic: bool,
) -> list[dict[str, Any]]:
    target_sets: dict[str, set[str]] = defaultdict(set)
    for context in contexts:
        target_sets[context.dataset].add(context.target_subject_id)
    require(
        {dataset: len(targets) for dataset, targets in target_sets.items()}
        == DATASET_TARGET_COUNTS,
        "construction availability target coverage drift",
    )

    counts: Counter[tuple[str, str, int]] = Counter()
    seen_trials: set[tuple[str, str, str]] = set()
    for row in construction_rows:
        dataset = str(row["dataset"])
        target = str(row["target_subject_id"])
        label = int(row["canonical_class_label"])
        require(dataset in target_sets and target in target_sets[dataset],
                "construction row is outside registered targets")
        require(label in (0, 1), "construction class identity drift")
        trial_key = (dataset, target, str(row["target_trial_id"]))
        require(trial_key not in seen_trials, "duplicate construction trial identity")
        seen_trials.add(trial_key)
        counts[(dataset, target, label)] += 1

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_TARGET_COUNTS:
        targets = sorted(target_sets[dataset])
        class_counts = [counts[(dataset, target, label)] for target in targets for label in (0, 1)]
        require(all(value >= 8 for value in class_counts),
                f"primary Q0 construction minimum failed: {dataset}")
        if not synthetic:
            expected_min, expected_max = EXPECTED_CONSTRUCTION_CLASS_RANGE[dataset]
            require(min(class_counts) == expected_min and max(class_counts) == expected_max,
                    f"construction availability range drift: {dataset}")

        planned = (1, 2, 4, 8, *HISTORICAL_SECONDARY_BUDGETS[dataset])
        for budget in (*planned, "FULL"):
            if budget == "FULL":
                feasible = len(targets)
                role = "PRIMARY"
            else:
                feasible = sum(
                    min(counts[(dataset, target, 0)], counts[(dataset, target, 1)]) >= int(budget)
                    for target in targets
                )
                role = "PRIMARY" if int(budget) <= 8 else "SECONDARY"
            operative = budget == "FULL" or budget in OPERATIVE_FINITE_BUDGETS[dataset]
            disposition = (
                "FEASIBLE" if feasible == len(targets) else
                "INPUT_UNAVAILABLE_ALL_TARGETS" if feasible == 0 else
                "PARTIALLY_AVAILABLE_FORBIDDEN"
            )
            rows.append({
                "dataset": dataset, "budget": str(budget), "budget_role": role,
                "targets": len(targets), "feasible_targets": feasible,
                "infeasible_targets": len(targets) - feasible,
                "min_labels_per_class": min(class_counts),
                "max_labels_per_class": max(class_counts),
                "operative": int(operative), "disposition": disposition,
            })
            if operative:
                require(feasible == len(targets),
                        f"operative Q0 budget is not universally feasible: {dataset}/B{budget}")
            elif dataset == "Lee2019_MI" and budget == 32:
                require(feasible == 0, "Lee B32 availability contract drift")
            else:
                require(False, f"unexpected unavailable Q0 budget: {dataset}/B{budget}")
    require(len(rows) == 19, "Q0 availability registry row arithmetic drift")
    return rows


def run_stage_b(
    *, stage_a_handoff_path: str | Path, final_root: str | Path,
    contexts: Sequence[ContextDescriptor] | None = None,
    context_loader: Callable[[ContextDescriptor], Mapping[str, Any]] | None = None,
    score_provider: Callable[[ContextDescriptor, Mapping[str, Any]], Mapping[str, np.ndarray]] | None = None,
    q0_builder: Callable[..., Mapping[str, np.ndarray]] = build_context_payload,
    chains: int = Q0_CHAINS, synthetic: bool = False,
    failure_injection_context: int | None = None,
) -> dict[str, Any]:
    handoff, construction_rows = _load_construction_handoff(stage_a_handoff_path)
    contexts = sorted(
        list(enumerate_contexts() if contexts is None else contexts),
        key=zoo_then_target_key,
    )
    if not synthetic:
        require(chains == Q0_CHAINS and len(contexts) == 944,
                "real C84SR3 Stage-B scope reduction")
    availability = construction_budget_availability(
        construction_rows, contexts, synthetic=synthetic,
    )
    loader = FrozenZooReader(include_source=True) if context_loader is None else context_loader
    final_root = Path(final_root)
    require(not final_root.exists(), "C84SR3 Stage-B final root exists")
    active_streams: dict[str, CSVStream] = {}

    def close_active_streams() -> None:
        for stream in active_streams.values():
            stream.abort()

    def writer(staging: Path) -> None:
        availability_sha = write_csv(staging / "q0_budget_availability.csv", availability)
        streams = {
            "candidate_scores.csv": CSVStream(staging / "candidate_scores.csv", SCORE_FIELDS),
            "candidate_ranks.csv": CSVStream(staging / "candidate_ranks.csv", RANK_FIELDS),
            "fixed_default_selections.csv": CSVStream(staging / "fixed_default_selections.csv", FIXED_FIELDS),
            "q0_selection_shard_index.csv": CSVStream(staging / "q0_selection_shard_index.csv", SHARD_INDEX_FIELDS),
            "q0_sample_digest_registry.csv": CSVStream(staging / "q0_sample_digest_registry.csv", SAMPLE_FIELDS),
            "q0_selected_regime_distribution.csv": CSVStream(staging / "q0_selected_regime_distribution.csv", Q0_REGIME_FIELDS),
            "q0_selection_coverage_diagnostics.csv": CSVStream(staging / "q0_selection_coverage_diagnostics.csv", Q0_COVERAGE_FIELDS),
            "selection_input_access_ledger.csv": CSVStream(staging / "selection_input_access_ledger.csv", ACCESS_FIELDS),
        }
        active_streams.update(streams)
        seen_samples: set[tuple[str, str]] = set()
        sample_plan_identities: dict[tuple[str, str], str] = {}
        q0_records = 0
        context_rows = []
        for context_index, context in enumerate(contexts):
            if failure_injection_context is not None and context_index == failure_injection_context:
                raise RuntimeError("injected C84SR3 Stage-B context failure")
            data = dict(loader(context))
            candidate_ids = list(map(str, data["candidate_ids"]))
            regimes = list(map(str, data["regimes"]))
            trajectory_orders = [int(value) for value in data["trajectory_orders"]]
            require(candidate_ids == [row.unit_id for row in context.candidates],
                    "context loader candidate identity drift")
            if score_provider is None:
                all_scores, _ = selectors.score_context(
                    np.asarray(data["source_probabilities"]), np.asarray(data["source_labels"]),
                    np.asarray(data["source_domains"]), np.asarray(data["target_logits"]),
                    regimes, trajectory_orders,
                )
                score_vectors = {method: all_scores[method] for method in SCORE_METHODS}
            else:
                score_vectors = dict(score_provider(context, data))
            require(set(score_vectors) == set(SCORE_METHODS), "Stage-B score-method set drift")
            prefix = _context_prefix(context)
            for method in SCORE_METHODS:
                scores = np.asarray(score_vectors[method], dtype=float)
                require(scores.shape == (81,) and np.all(np.isfinite(scores)),
                        f"score vector drift: {method}")
                order = np.lexsort((np.arange(81), -scores))
                ranks = np.empty(81, dtype=int)
                ranks[order] = np.arange(1, 82)
                for candidate_index, candidate_id in enumerate(candidate_ids):
                    streams["candidate_scores.csv"].write({
                        **prefix, "method_id": method, "candidate_index": candidate_index,
                        "candidate_id": candidate_id, "raw_score": float(scores[candidate_index]),
                    })
                    streams["candidate_ranks.csv"].write({
                        **prefix, "method_id": method, "candidate_index": candidate_index,
                        "candidate_id": candidate_id, "rank": int(ranks[candidate_index]),
                    })
            for method, selected in _fixed_indices(regimes, trajectory_orders).items():
                streams["fixed_default_selections.csv"].write({
                    **prefix, "method_id": method, "selected_candidate_index": selected,
                    "selected_candidate_id": candidate_ids[selected],
                })
            payload = q0_builder(
                identity=context.identity(), candidate_ids=candidate_ids,
                target_logits=np.asarray(data["target_logits"]),
                target_trial_ids=np.asarray(data["target_trial_ids"], dtype=str),
                construction_rows=construction_rows, chains=chains,
            )
            relative_shard = Path("q0_shards") / f"{context.context_id}.npz"
            shard = write_context_shard(staging / relative_shard, payload, chains=chains)
            streams["q0_selection_shard_index.csv"].write({
                key: (str(relative_shard) if key == "path" else shard[key])
                for key in SHARD_INDEX_FIELDS
            })
            q0_records += int(shard["total_records"])
            codes = np.asarray(payload["finite_budget_code"], dtype=np.uint8)
            selected = np.asarray(payload["finite_selected_index"], dtype=np.uint8)
            finite_regimes = np.asarray(regimes, dtype=str)[selected]
            for budget in finite_budgets(context.dataset):
                mask = codes == budget
                streams["q0_selection_coverage_diagnostics.csv"].write({
                    **prefix, "budget": str(budget), "expected_records": chains,
                    "observed_records": int(np.sum(mask)), "coverage": float(np.sum(mask) / chains),
                })
                for regime in ("ERM", "OACI", "SRC"):
                    count = int(np.sum(finite_regimes[mask] == regime))
                    streams["q0_selected_regime_distribution.csv"].write({
                        **prefix, "budget": str(budget), "regime": regime,
                        "chain_count": count, "fraction": count / float(chains),
                    })
            streams["q0_selection_coverage_diagnostics.csv"].write({
                **prefix, "budget": "FULL", "expected_records": 1,
                "observed_records": 1, "coverage": 1.0,
            })
            full_selected = int(np.asarray(payload["FULL_selected_index"])[0])
            for regime in ("ERM", "OACI", "SRC"):
                count = int(regimes[full_selected] == regime)
                streams["q0_selected_regime_distribution.csv"].write({
                    **prefix, "budget": "FULL", "regime": regime,
                    "chain_count": count, "fraction": float(count),
                })
            sample_key = (context.dataset, context.target_subject_id)
            sample_plan_identity = _q0_sample_plan_identity(payload)
            if sample_key in sample_plan_identities:
                require(
                    sample_plan_identities[sample_key] == sample_plan_identity,
                    "Q0 paired sample-plan identity drift across repeated contexts",
                )
            else:
                sample_plan_identities[sample_key] = sample_plan_identity
            if sample_key not in seen_samples:
                seen_samples.add(sample_key)
                chains_array = np.asarray(payload["finite_chain"])
                seeds_array = np.asarray(payload["finite_chain_seed"])
                sizes = np.asarray(payload["finite_sample_size"])
                digests = np.asarray(payload["finite_sample_digest"])
                for index in range(len(chains_array)):
                    streams["q0_sample_digest_registry.csv"].write({
                        "dataset": context.dataset, "target_subject_id": context.target_subject_id,
                        "chain": int(chains_array[index]), "chain_seed": int(seeds_array[index]),
                        "budget": str(int(codes[index])),
                        "sample_trial_id_sha256": bytes(digests[index]).hex(),
                        "sample_size": int(sizes[index]),
                    })
                streams["q0_sample_digest_registry.csv"].write({
                    "dataset": context.dataset, "target_subject_id": context.target_subject_id,
                    "chain": "FULL", "chain_seed": 0, "budget": "FULL",
                    "sample_trial_id_sha256": bytes(np.asarray(payload["FULL_sample_digest"])[0]).hex(),
                    "sample_size": int(np.asarray(payload["FULL_sample_size"])[0]),
                })
            source_rows = len(np.asarray(data["source_labels"]))
            target_rows = len(np.asarray(data["target_trial_ids"]))
            for method in SCORE_METHODS:
                view = "strict_source_and_target_unlabeled" if method in {"U7", "U13", "U15"} else (
                    "strict_source" if method == "S1" else "target_unlabeled"
                )
                streams["selection_input_access_ledger.csv"].write({
                    **prefix, "method_id": method, "view": view, "read_allowed": 1,
                    "rows": source_rows + target_rows if "and" in view else source_rows if view == "strict_source" else target_rows,
                    "labels": source_rows if "source" in view else 0,
                })
            streams["selection_input_access_ledger.csv"].write({
                **prefix, "method_id": "Q0", "view": "target_construction_labels",
                "read_allowed": 1,
                "rows": sum(str(row["dataset"]) == context.dataset and
                            str(row["target_subject_id"]) == context.target_subject_id
                            for row in construction_rows),
                "labels": 1,
            })
            context_rows.append({**prefix, "candidate_count": 81})

        artifacts = {name: stream.close() for name, stream in streams.items()}
        artifacts["q0_budget_availability.csv"] = {
            "path": "q0_budget_availability.csv", "rows": len(availability),
            "sha256": availability_sha,
        }
        require(len(context_rows) == len(contexts), "Stage-B context count drift")
        if not synthetic:
            expected_rows = {
                "candidate_scores.csv": 535248,
                "candidate_ranks.csv": 535248,
                "fixed_default_selections.csv": 4720,
                "q0_selection_shard_index.csv": 944,
                "q0_sample_digest_registry.csv": Q0_SAMPLE_DIGEST_ROWS,
            }
            for name, expected in expected_rows.items():
                require(artifacts[name]["rows"] == expected,
                        f"C84SR3 Stage-B exact row count drift: {name}")
            require(q0_records == Q0_RECORDS, "C84SR3 Stage-B Q0 record count drift")
        context_sha = write_json(staging / "C84S_STAGE_B_CONTEXT_REGISTRY.json", {
            "schema_version": "c84sr3_stage_b_context_registry_v2",
            "contexts": context_rows, "context_count": len(context_rows),
        })
        artifacts["C84S_STAGE_B_CONTEXT_REGISTRY.json"] = {
            "path": "C84S_STAGE_B_CONTEXT_REGISTRY.json", "rows": len(context_rows),
            "sha256": context_sha,
        }
        manifest = {
            "schema_version": "c84sr3_selection_freeze_manifest_v3",
            "status": "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
            "stage_A_handoff_sha256": sha256_file(stage_a_handoff_path),
            "evaluation_label_descriptor_received": False,
            "same_label_oracle_accessed": False,
            "contexts": len(contexts), "score_methods": list(SCORE_METHODS),
            "fixed_methods": list(FIXED_METHODS), "Q0_chains": chains,
            "Q0_records": q0_records, "Q0_shards": len(contexts),
            "Lee_Q0_B32_status": "INPUT_UNAVAILABLE_NO_SELECTION_OR_RESULT_ROW",
            "artifacts": artifacts,
        }
        manifest_sha = write_json(staging / "C84S_SELECTION_FREEZE_MANIFEST_V3.json", manifest)
        (staging / "C84S_SELECTION_FREEZE_MANIFEST_V3.sha256").write_text(
            f"{manifest_sha}  C84S_SELECTION_FREEZE_MANIFEST_V3.json\n", encoding="ascii",
        )
        write_json(staging / "C84S_STAGE_B_HANDOFF_V2.json", {
            "schema_version": "c84sr3_stage_b_to_c_handoff_v2", "stage": "Stage_B_to_Stage_C",
            "selection_freeze_manifest_sha256": manifest_sha,
            "selection_freeze_status": manifest["status"],
            "evaluation_descriptor_received": False,
        })

    writer.cleanup_on_failure = close_active_streams  # type: ignore[attr-defined]
    published = atomic_publish_directory(final_root, writer)
    manifest_path = published / "C84S_SELECTION_FREEZE_MANIFEST_V3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "stage": "Stage_B", "root": str(published), "status": manifest["status"],
        "sha256": sha256_file(manifest_path), "contexts": manifest["contexts"],
        "Q0_records": manifest["Q0_records"],
    }


def run_stage_b_real(*, stage_a_handoff_path: str | Path, final_root: str | Path) -> dict[str, Any]:
    attempt_path = Path(final_root).parent / f"{Path(final_root).name}.stage_b_attempt.json"
    require(not attempt_path.exists(), "C84SR3 Stage-B attempt exists")
    attempt = {
        "schema_version": "c84sr3_stage_attempt_v2", "stage": "Stage_B",
        "status": "STARTED", "started_at_unix_ns": time.time_ns(),
        "evaluation_descriptor_received": False, "scientific_statistics": 0,
    }
    write_json(attempt_path, attempt)
    try:
        result = run_stage_b(
            stage_a_handoff_path=stage_a_handoff_path, final_root=final_root,
        )
        attempt.update({"status": "COMPLETE", "finished_at_unix_ns": time.time_ns(), "result": result})
        write_json(attempt_path, attempt)
        return result
    except BaseException as error:
        attempt.update({
            "status": "FAILED", "finished_at_unix_ns": time.time_ns(),
            "error_type": type(error).__name__, "error": str(error),
            "error_notes": list(getattr(error, "__notes__", ())),
            "partial_final_root": Path(final_root).exists(),
        })
        write_json(attempt_path, attempt)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--stage-a-handoff", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = run_stage_b_real(
        stage_a_handoff_path=args.stage_a_handoff, final_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
