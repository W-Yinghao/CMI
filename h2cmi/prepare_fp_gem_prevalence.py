"""Build the frozen P13 Lee2019_MI prevalence-stress manifest."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from h2cmi.data.real_eeg import load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.w1_repaired_split import indices_from_trial_ids, load_manifest_csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "h2cmi/results/fp_gem_prevalence"
P12_SPLIT = ROOT / "h2cmi/results/review_completion/w1_repaired_split_manifest.csv"
P12_RESULTS = ROOT / "h2cmi/results/fp_gem_main/fp_gem_results.csv"
P12_ARTIFACTS = ROOT / "h2cmi/results/fp_gem_main/fp_gem_job_artifact_manifest.csv"
P12_UNITS = ROOT / "h2cmi/results/fp_gem_main/fp_gem_units.csv"
P12_RUNNER = ROOT / "h2cmi/run_fp_gem.py"
P12_CONFIG = ROOT / "h2cmi/results/fp_gem_main/fp_gem_config.json"
P12_RAW_ROOT = Path("/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12")
DATASET = "Lee2019_MI"
SEEDS = (0, 1, 2)
Q_VALUES = ("0.1", "0.5", "0.9")
METHODS = (
    "source_only_tsmnet",
    "rct",
    "spdim_geodesic",
    "spdim_bias",
    "Joint-GEM",
    "FP-GEM",
)
EXPECTED_SHA256 = {
    "p12_results": "f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d",
    "p12_artifacts": "f19edc94251a4339e596bfb148ebef0ba23afdf18fa32f874cb4d3ba0500efc5",
    "p12_units": "3bb1250b3faf583ff79324326b0159b6a6dd9f8efd3a92ecc21231e31fb2c267",
    "p12_runner": "720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd",
    "p12_config": "d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165",
    "p12_split": "e9ebe6e9421bdcf10f8a952623285cec0842f5cb6b868e8147f13dde23e8a712",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=int)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(compact(value).encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def repeat_or_crop(values: list[str], count: int) -> list[str]:
    if not values or count < 0:
        raise ValueError("invalid deterministic class pool/count")
    return [values[index % len(values)] for index in range(count)]


def batch_ids(q: str, original: list[str], labels: dict[str, int]) -> tuple[list[str], list[int]]:
    if q == "0.5":
        return list(original), [25, 25]
    n_total = len(original)
    n_class0 = int(round(float(q) * n_total))
    counts = [n_class0, n_total - n_class0]
    pools = {
        cls: [trial_id for trial_id in original if labels[trial_id] == cls]
        for cls in (0, 1)
    }
    selected = repeat_or_crop(pools[0], counts[0]) + repeat_or_crop(pools[1], counts[1])
    return selected, counts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_paths = {
        "p12_results": P12_RESULTS,
        "p12_artifacts": P12_ARTIFACTS,
        "p12_units": P12_UNITS,
        "p12_runner": P12_RUNNER,
        "p12_config": P12_CONFIG,
        "p12_split": P12_SPLIT,
    }
    observed = {name: sha256_file(path) for name, path in frozen_paths.items()}
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"P12 frozen-input mismatch: {observed}")

    split_rows = [row for row in load_manifest_csv(P12_SPLIT) if row["dataset"] == DATASET]
    if len(split_rows) != 162:
        raise RuntimeError(f"expected 162 Lee split rows, found {len(split_rows)}")
    split_index = {
        (int(row["target_subject"]), int(row["source_seed"])): row
        for row in split_rows
    }
    p12_results = [row for row in read_csv(P12_RESULTS) if row["dataset"] == DATASET]
    result_index = {
        (int(row["target_subject"]), int(row["source_seed"]), row["method"]): row
        for row in p12_results
    }
    if len(result_index) != 972:
        raise RuntimeError(f"expected 972 Lee P12 rows, found {len(result_index)}")
    artifacts = [row for row in read_csv(P12_ARTIFACTS) if row["dataset"] == DATASET]
    artifact_index = {
        (int(row["target_subject"]), int(row["source_seed"])): row
        for row in artifacts
    }
    units = [row for row in read_csv(P12_UNITS) if row["dataset"] == DATASET]
    unit_index = {
        (int(row["target_subject"]), int(row["source_seed"])): row
        for row in units
    }
    if len(artifact_index) != 162 or len(unit_index) != 162:
        raise RuntimeError("P12 Lee unit/artifact coverage mismatch")

    ep = load_dataset(DATASET, MOABB_CLASS[DATASET]().subject_list)
    subjects = sorted(int(value) for value in np.unique(ep.subject))
    if subjects != list(range(1, 55)):
        raise RuntimeError(f"unexpected Lee subject IDs: {subjects}")

    csv_rows: list[dict[str, Any]] = []
    manifest_units: list[dict[str, Any]] = []
    repeated_occurrences = 0
    for target in subjects:
        base = split_index[(target, 0)]
        original = list(base["adapt_trial_ids"])
        evaluation = list(base["eval_trial_ids"])
        for seed in SEEDS[1:]:
            other = split_index[(target, seed)]
            if other["adapt_trial_ids"] != original or other["eval_trial_ids"] != evaluation:
                raise RuntimeError(f"seed-dependent target reservoir for subject {target}")
        adapt_indices = indices_from_trial_ids(original)
        labels = {trial_id: int(ep.y[index]) for trial_id, index in zip(original, adapt_indices)}
        if Counter(labels.values()) != {0: 25, 1: 25}:
            raise RuntimeError(f"unexpected P12 adaptation composition for subject {target}")
        reservoir_positions = {trial_id: position for position, trial_id in enumerate(original)}
        eval_hash = sha256_json(evaluation)

        batches = {}
        for q in Q_VALUES:
            selected, counts = batch_ids(q, original, labels)
            if len(selected) != 50 or counts not in ([5, 45], [25, 25], [45, 5]):
                raise RuntimeError(f"invalid stress batch for subject={target}, q={q}")
            if q == "0.5" and selected != original:
                raise RuntimeError("q=0.5 is not the exact original P12 adaptation order")
            occurrence_counter: Counter[str] = Counter()
            occurrence_rows = []
            for position, trial_id in enumerate(selected):
                repeat_index = occurrence_counter[trial_id]
                occurrence_counter[trial_id] += 1
                if repeat_index > 0:
                    repeated_occurrences += 1
                occurrence_rows.append({
                    "position": position,
                    "trial_id": trial_id,
                    "reservoir_position": reservoir_positions[trial_id],
                    "builder_class_label": labels[trial_id],
                    "repeat_index": repeat_index,
                })
            batch_payload = {
                "dataset": DATASET,
                "target_subject": target,
                "q": q,
                "n_adapt": len(selected),
                "class_counts_adapt": counts,
                "trial_ids": selected,
                "resampling_rule": "class_order_then_repeat_or_crop_modulo_pool",
            }
            batch_hash = sha256_json(batch_payload)
            batches[q] = {
                "adaptation_manifest_hash": batch_hash,
                "class_counts_adapt": counts,
                "n_adapt": len(selected),
                "trial_ids": selected,
            }
            for seed in SEEDS:
                checkpoint_hash = result_index[(target, seed, "source_only_tsmnet")][
                    "source_checkpoint_file_sha256"
                ]
                for occurrence in occurrence_rows:
                    csv_rows.append({
                        "dataset": DATASET,
                        "target_subject": target,
                        "source_seed": seed,
                        "q": q,
                        "adapt_position": occurrence["position"],
                        "trial_id": occurrence["trial_id"],
                        "epoch_index": int(occurrence["trial_id"].rsplit(":epoch=", 1)[1]),
                        "reservoir_position": occurrence["reservoir_position"],
                        "builder_class_label": occurrence["builder_class_label"],
                        "repeat_index": occurrence["repeat_index"],
                        "is_repeated_occurrence": occurrence["repeat_index"] > 0,
                        "is_original_p12_center": q == "0.5",
                        "adaptation_manifest_hash": batch_hash,
                        "checkpoint_hash": checkpoint_hash,
                        "p12_split_hash": split_index[(target, seed)]["split_hash"],
                        "eval_trial_ids_sha256": eval_hash,
                    })

        for seed in SEEDS:
            key = (target, seed)
            result_rows = [result_index[(target, seed, method)] for method in METHODS]
            checkpoint_hashes = {row["source_checkpoint_file_sha256"] for row in result_rows}
            source_hashes = {row["source_model_sha256"] for row in result_rows}
            if len(checkpoint_hashes) != 1 or len(source_hashes) != 1:
                raise RuntimeError(f"P12 six-method source mismatch for {key}")
            checkpoint_hash = checkpoint_hashes.pop()
            source_hash = source_hashes.pop()
            artifact = artifact_index[key]
            raw_path = Path(artifact["raw_path"])
            checkpoint_path = P12_RAW_ROOT / f"source_checkpoints/{DATASET}_target{target}_seed{seed}.pt"
            sidecar_path = checkpoint_path.with_suffix(".json")
            if not raw_path.exists() or sha256_file(raw_path) != artifact["raw_sha256"]:
                raise RuntimeError(f"P12 raw-unit checksum mismatch for {key}")
            if not checkpoint_path.exists() or sha256_file(checkpoint_path) != checkpoint_hash:
                raise RuntimeError(f"P12 checkpoint checksum mismatch for {key}")
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar["source_model_sha256_actual"] != source_hash:
                raise RuntimeError(f"P12 source-state sidecar mismatch for {key}")
            raw_payload = json.loads(raw_path.read_text())
            if raw_payload["source_checkpoint"]["source_checkpoint_file_sha256"] != checkpoint_hash:
                raise RuntimeError(f"P12 raw checkpoint reference mismatch for {key}")
            manifest_units.append({
                "dataset": DATASET,
                "target_subject": target,
                "source_seed": seed,
                "hardware_group": unit_index[key]["hardware_group"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "source_model_sha256": source_hash,
                "p12_raw_path": str(raw_path),
                "p12_raw_sha256": artifact["raw_sha256"],
                "p12_split_hash": split_index[key]["split_hash"],
                "adapt_reservoir_trial_ids": original,
                "eval_trial_ids": evaluation,
                "eval_trial_ids_sha256": eval_hash,
                "class_counts_eval": [25, 25],
                "batches": batches,
            })

    manifest_units.sort(key=lambda row: (row["target_subject"], row["source_seed"]))
    manifest_payload = {
        "schema": "fp_gem_fixed_reservoir_prevalence_v1",
        "dataset": DATASET,
        "subjects": subjects,
        "source_seeds": list(SEEDS),
        "q_values": list(Q_VALUES),
        "adaptation_batch_size": 50,
        "evaluation_batch_size": 50,
        "resampling_rule": "q=0.5 exact P12 order; endpoints class-order then deterministic modulo repeat/crop",
        "labels_used_by_builder_only": True,
        "runtime_manifest_contains_class_labels": False,
        "units": manifest_units,
    }
    manifest_payload["semantic_sha256"] = sha256_json(manifest_payload)
    json_path = OUT / "fp_gem_prevalence_manifest.json"
    write_json(json_path, manifest_payload)

    csv_path = OUT / "fp_gem_prevalence_manifest.csv"
    fieldnames = [
        "dataset", "target_subject", "source_seed", "q", "adapt_position", "trial_id",
        "epoch_index", "reservoir_position", "builder_class_label", "repeat_index",
        "is_repeated_occurrence", "is_original_p12_center", "adaptation_manifest_hash",
        "checkpoint_hash", "p12_split_hash", "eval_trial_ids_sha256",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    config = {
        "schema": "fp_gem_prevalence_config_v1",
        "dataset": DATASET,
        "subjects": subjects,
        "source_seeds": list(SEEDS),
        "q_values": [0.1, 0.5, 0.9],
        "methods": list(METHODS),
        "adaptation_batch_size": 50,
        "class_counts_by_q": {"0.1": [5, 45], "0.5": [25, 25], "0.9": [45, 5]},
        "resampling_rule": "q=0.5 exact P12 order; endpoints class-order then deterministic modulo repeat/crop",
        "q_hidden_from_methods": True,
        "target_labels_passed_to_methods": False,
        "target_performance_selection": False,
        "p12": {
            "commit": "3bba1d0bb4f803948e855b6fa707e64b13f2a99a",
            "results_path": str(P12_RESULTS.relative_to(ROOT)),
            "results_sha256": EXPECTED_SHA256["p12_results"],
            "artifact_manifest_path": str(P12_ARTIFACTS.relative_to(ROOT)),
            "artifact_manifest_sha256": EXPECTED_SHA256["p12_artifacts"],
            "runner_path": str(P12_RUNNER.relative_to(ROOT)),
            "runner_sha256": EXPECTED_SHA256["p12_runner"],
            "config_path": str(P12_CONFIG.relative_to(ROOT)),
            "config_sha256": EXPECTED_SHA256["p12_config"],
            "split_path": str(P12_SPLIT.relative_to(ROOT)),
            "split_sha256": EXPECTED_SHA256["p12_split"],
            "split_semantic_sha256": "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e",
            "raw_root": str(P12_RAW_ROOT),
        },
        "manifest": {
            "csv_path": str(csv_path.relative_to(ROOT)),
            "csv_sha256": sha256_file(csv_path),
            "json_path": str(json_path.relative_to(ROOT)),
            "json_sha256": sha256_file(json_path),
            "semantic_sha256": manifest_payload["semantic_sha256"],
        },
        "geometry_displacement": "sqrt(||b_q-b_0.5||_2^2 + ||a_q-a_0.5||_2^2)",
        "q05_prediction_recovery": {
            "reason": "P12 persisted hashes but not prediction vectors required for disagreement",
            "policy": "deterministic hash-gated replay; P12 rows remain the accepted q=0.5 result rows",
            "accept_only_if_all_six_prediction_and_logits_hashes_match_p12": True,
            "new_q05_result_claim": False,
        },
        "expected_counts": {
            "target_subjects": 54,
            "target_seed_units": 162,
            "manifest_occurrence_rows": 24300,
            "new_adaptation_rows": 1620,
            "reused_p12_center_rows": 972,
            "reused_source_only_endpoint_rows": 324,
            "final_result_rows": 2916,
            "per_subject_q_method_rows": 972,
            "sensitivity_subject_method_rows": 324,
            "geometry_rows": 972,
        },
        "statistics": {
            "seed_average_first": True,
            "biological_cluster": "target_subject",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260710,
            "interval": "percentile_95",
            "paired_methods_and_q_preserved": True,
            "primary_endpoint": "0.5*(abs(bacc_q01-bacc_q05)+abs(bacc_q09-bacc_q05))",
            "primary_comparison": "FP-GEM sensitivity minus Joint-GEM sensitivity",
            "primary_support_rule": "paired 95% CI entirely below zero",
            "equivalence_or_noninferiority_claim": False,
        },
        "execution": {
            "raw_root": "/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13",
            "runtime_python": "/home/infres/yinwang/anaconda3/envs/icml/bin/python",
            "max_gpu_tasks": 8,
            "monitoring": "squeue_only",
            "fresh_source_training_permitted": False,
        },
    }
    config_path = OUT / "fp_gem_prevalence_config.json"
    write_json(config_path, config)
    print(json.dumps({
        "status": "pass",
        "manifest_units": len(manifest_units),
        "manifest_occurrence_rows": len(csv_rows),
        "repeated_occurrences_per_seed_expanded": repeated_occurrences * len(SEEDS),
        "manifest_csv_sha256": sha256_file(csv_path),
        "manifest_json_sha256": sha256_file(json_path),
        "manifest_semantic_sha256": manifest_payload["semantic_sha256"],
        "config_sha256": sha256_file(config_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
