"""C84SR1 Stage-B complete selection materialization and atomic freeze."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import c84s_selectors as selectors
from .c84s_common import (
    atomic_publish_directory, read_csv, read_json, require, sha256_file, write_json,
)
from .c84sr1_common import (
    FIXED_METHODS, Q0_CHAINS, SCORE_METHODS, reject_evaluation_tokens,
)
from .c84sr1_context_enumerator import ContextDescriptor, enumerate_contexts
from .c84sr1_field_reader import FrozenZooReader, zoo_then_target_key
from .c84sr1_q0_store import (
    SHARD_INDEX_FIELDS, build_context_payload, write_context_shard,
)


CONTEXT_FIELDS = (
    "context_id", "dataset", "target_subject_id", "panel", "training_seed", "level",
)
SCORE_FIELDS = CONTEXT_FIELDS + ("method_id", "candidate_index", "candidate_id", "raw_score")
RANK_FIELDS = CONTEXT_FIELDS + ("method_id", "candidate_index", "candidate_id", "rank")
FIXED_FIELDS = CONTEXT_FIELDS + ("method_id", "selected_candidate_index", "selected_candidate_id")
SAMPLE_FIELDS = (
    "dataset", "target_subject_id", "chain", "chain_seed", "budget",
    "sample_trial_id_sha256", "sample_size",
)
Q0_REGIME_FIELDS = CONTEXT_FIELDS + ("budget", "regime", "chain_count", "fraction")
Q0_COVERAGE_FIELDS = CONTEXT_FIELDS + ("budget", "expected_records", "observed_records", "coverage")
ACCESS_FIELDS = CONTEXT_FIELDS + ("method_id", "view", "read_allowed", "rows", "labels")


class CSVStream:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.path = path
        self.fields = tuple(fields)
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, lineterminator="\n")
        self.writer.writeheader()
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        require(tuple(row) == self.fields, f"streaming CSV field order drift: {self.path.name}")
        self.writer.writerow(row)
        self.rows += 1

    def close(self) -> dict[str, Any]:
        self.handle.flush()
        self.handle.close()
        return {"path": self.path.name, "rows": self.rows, "sha256": sha256_file(self.path)}


def _context_prefix(context: ContextDescriptor) -> dict[str, Any]:
    return {
        "context_id": context.context_id, "dataset": context.dataset,
        "target_subject_id": context.target_subject_id, "panel": context.panel,
        "training_seed": context.training_seed, "level": context.level,
    }


def _default_context_loader(context: ContextDescriptor) -> dict[str, Any]:
    raise RuntimeError("use one FrozenZooReader instance per Stage-B execution")


def _load_construction_handoff(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    handoff = read_json(path)
    reject_evaluation_tokens(handoff)
    require(handoff["stage"] == "Stage_A_to_Stage_B", "Stage-A-to-B handoff stage drift")
    descriptor = handoff["construction_descriptor"]
    require(descriptor["kind"] == "construction", "Stage B received non-construction view")
    root = Path(descriptor["root"])
    manifest_path = root / "manifest.json"
    require(sha256_file(manifest_path) == descriptor["manifest_sha256"], "construction manifest SHA drift")
    manifest = read_json(manifest_path)
    require(manifest["candidate_artifacts"] == manifest["EEG_arrays"] == 0,
            "construction view contains forbidden payload")
    labels_path = root / manifest["table"]["path"]
    require(sha256_file(labels_path) == manifest["table"]["sha256"], "construction label table SHA drift")
    rows = read_csv(labels_path)
    require(len(rows) == int(descriptor["row_count"]), "construction row count drift")
    return handoff, rows


def _fixed_indices(regimes: Sequence[str], orders: Sequence[int]) -> dict[str, int]:
    lookup = {(str(regime), int(order)): index for index, (regime, order) in enumerate(zip(regimes, orders))}
    result = {
        "B1": lookup[("ERM", 0)], "B2": lookup[("OACI", 40)],
        "B3": lookup[("SRC", 40)], "B4O": lookup[("OACI", 20)],
        "B4S": lookup[("SRC", 20)],
    }
    require(set(result) == set(FIXED_METHODS), "fixed default identity drift")
    return result


def run_stage_b(
    *,
    stage_a_handoff_path: str | Path,
    final_root: str | Path,
    contexts: Sequence[ContextDescriptor] | None = None,
    context_loader: Callable[[ContextDescriptor], Mapping[str, Any]] | None = None,
    score_provider: Callable[[ContextDescriptor, Mapping[str, Any]], Mapping[str, np.ndarray]] | None = None,
    q0_builder: Callable[..., Mapping[str, np.ndarray]] = build_context_payload,
    chains: int = Q0_CHAINS,
    synthetic: bool = False,
    failure_injection_context: int | None = None,
) -> dict[str, Any]:
    handoff, construction_rows = _load_construction_handoff(stage_a_handoff_path)
    contexts = sorted(
        list(enumerate_contexts() if contexts is None else contexts),
        key=zoo_then_target_key,
    )
    loader = FrozenZooReader(include_source=True) if context_loader is None else context_loader
    if not synthetic:
        require(chains == Q0_CHAINS and len(contexts) == 944, "real Stage-B scope reduction")
    final_root = Path(final_root)
    require(not final_root.exists(), "Stage-B final root exists")

    def writer(staging: Path) -> None:
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
        shard_root = staging / "q0_shards"
        seen_samples: set[tuple[str, str]] = set()
        q0_records = 0
        context_rows = []
        for context_index, context in enumerate(contexts):
            if failure_injection_context is not None and context_index == failure_injection_context:
                raise RuntimeError("injected Stage-B context failure")
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
                require(scores.shape == (81,) and np.all(np.isfinite(scores)), f"score vector drift: {method}")
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
            shard_row = {
                key: (str(relative_shard) if key == "path" else shard[key])
                for key in SHARD_INDEX_FIELDS
            }
            streams["q0_selection_shard_index.csv"].write(shard_row)
            q0_records += int(shard["total_records"])
            codes = np.asarray(payload["finite_budget_code"], dtype=np.uint8)
            selected = np.asarray(payload["finite_selected_index"], dtype=np.uint8)
            finite_regimes = np.asarray(regimes, dtype=str)[selected]
            for budget in sorted(set(codes.tolist())):
                mask = codes == budget
                expected = chains
                streams["q0_selection_coverage_diagnostics.csv"].write({
                    **prefix, "budget": str(int(budget)), "expected_records": expected,
                    "observed_records": int(np.sum(mask)), "coverage": float(np.mean(mask) * len(codes) / expected),
                })
                for regime in ("ERM", "OACI", "SRC"):
                    count = int(np.sum(finite_regimes[mask] == regime))
                    streams["q0_selected_regime_distribution.csv"].write({
                        **prefix, "budget": str(int(budget)), "regime": regime,
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
        require(len(context_rows) == len(contexts), "Stage-B context count drift")
        if not synthetic:
            expected_rows = {
                "candidate_scores.csv": 535248, "candidate_ranks.csv": 535248,
                "fixed_default_selections.csv": 4720,
                "q0_selection_shard_index.csv": 944,
                "q0_sample_digest_registry.csv": 1138806,
            }
            for name, expected in expected_rows.items():
                require(artifacts[name]["rows"] == expected, f"Stage-B exact row count drift: {name}")
            require(q0_records == 9110448, "Stage-B exact Q0 record count drift")
        context_sha = write_json(staging / "C84S_STAGE_B_CONTEXT_REGISTRY.json", {
            "schema_version": "c84sr1_stage_b_context_registry_v1",
            "contexts": context_rows, "context_count": len(context_rows),
        })
        artifacts["C84S_STAGE_B_CONTEXT_REGISTRY.json"] = {
            "path": "C84S_STAGE_B_CONTEXT_REGISTRY.json", "rows": len(context_rows),
            "sha256": context_sha,
        }
        manifest = {
            "schema_version": "c84sr1_selection_freeze_manifest_v2",
            "status": "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
            "stage_A_handoff_sha256": sha256_file(stage_a_handoff_path),
            "evaluation_label_descriptor_received": False,
            "same_label_oracle_accessed": False,
            "contexts": len(contexts), "score_methods": list(SCORE_METHODS),
            "fixed_methods": list(FIXED_METHODS), "Q0_chains": chains,
            "Q0_records": q0_records, "Q0_shards": len(contexts),
            "artifacts": artifacts,
        }
        manifest_sha = write_json(staging / "C84S_SELECTION_FREEZE_MANIFEST_V2.json", manifest)
        (staging / "C84S_SELECTION_FREEZE_MANIFEST_V2.sha256").write_text(
            f"{manifest_sha}  C84S_SELECTION_FREEZE_MANIFEST_V2.json\n", encoding="ascii",
        )
        write_json(staging / "C84S_STAGE_B_HANDOFF.json", {
            "schema_version": "c84sr1_stage_b_to_c_handoff_v1", "stage": "Stage_B_to_Stage_C",
            "selection_freeze_manifest_sha256": manifest_sha,
            "selection_freeze_status": manifest["status"],
            "evaluation_descriptor_received": False,
        })

    published = atomic_publish_directory(final_root, writer)
    manifest = read_json(published / "C84S_SELECTION_FREEZE_MANIFEST_V2.json")
    return {
        "stage": "Stage_B", "root": str(published),
        "status": manifest["status"],
        "sha256": sha256_file(published / "C84S_SELECTION_FREEZE_MANIFEST_V2.json"),
        "contexts": manifest["contexts"], "Q0_records": manifest["Q0_records"],
    }


def run_stage_b_real(*, stage_a_handoff_path: str | Path, final_root: str | Path) -> dict[str, Any]:
    attempt_path = Path(final_root).parent / f"{Path(final_root).name}.stage_b_attempt.json"
    require(not attempt_path.exists(), "Stage-B attempt exists")
    attempt = {
        "schema_version": "c84sr1_stage_attempt_v1", "stage": "Stage_B",
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
