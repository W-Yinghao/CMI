"""Build the CPU-only P10 repaired-W1 cross-pipeline evidence freeze."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "h2cmi" / "results" / "review_completion"

H2_RESULTS = OUT / "w1_repaired_h2cmi_results.csv"
H2_SUMMARY = OUT / "w1_repaired_h2cmi_summary.json"
SPDIM_RESULTS = OUT / "spdim_w1_repaired_three_seed_results.csv"
SPDIM_SUMMARY = OUT / "spdim_w1_repaired_three_seed_summary.json"
SPDIM_METHOD_CI = OUT / "spdim_w1_repaired_three_seed_method_ci.csv"
SPDIM_CONTRAST_CI = OUT / "spdim_w1_repaired_three_seed_contrast_ci.csv"
MANIFEST_CSV = OUT / "w1_repaired_split_manifest.csv"
MANIFEST_JSON = OUT / "w1_repaired_split_manifest.json"
LEGACY_QUARANTINE = OUT / "w1_legacy_split_quarantine.json"
COMMAND_LOG = OUT / "COMMAND_LOG.md"

COMPARABILITY_MD = OUT / "w1_repaired_pipeline_comparability_audit.md"
COMPARABILITY_JSON = OUT / "w1_repaired_pipeline_comparability_audit.json"
CROSS_RESULTS_CSV = OUT / "w1_repaired_cross_pipeline_results.csv"
CROSS_RESULTS_MD = OUT / "w1_repaired_cross_pipeline_results.md"
CROSS_RESULTS_JSON = OUT / "w1_repaired_cross_pipeline_results.json"
HARM_CSV = OUT / "w1_repaired_cross_pipeline_harm.csv"
HARM_MD = OUT / "w1_repaired_cross_pipeline_harm.md"
FREEZE_MD = OUT / "FINAL_REPAIRED_W1_EVIDENCE_FREEZE.md"
FREEZE_JSON = OUT / "FINAL_REPAIRED_W1_EVIDENCE_FREEZE.json"
RED_TEAM_REVIEW = OUT / "w1_repaired_cross_pipeline_red_team_review.md"

H2_RESULT_COMMIT = "bc61ee11d21e023966fa9be637e960fdaf77a9c1"
SPDIM_RESULT_COMMIT = "8972de878a93e00a5b6cf6b8118bc32adc05eb48"
H2_RESULT_SHA256 = "6d5106a78dad9ce852c8e01ca292ef5b4a37bbeaaaac810a177dccb8b6b9089c"
SPDIM_RESULT_SHA256 = "95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3"
MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
SPLIT_FAMILY = "class_stratified_half"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260710
INTERVAL_LABEL = "posthoc_cross_pipeline_comparability_audit"

DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
SUBJECTS = {
    "BNCI2014_001": list(range(1, 10)),
    "Cho2017": list(range(1, 53)),
    "Lee2019_MI": list(range(1, 55)),
}
SEEDS = [0, 1, 2]

METHOD_SPECS = {
    "h2cmi": [
        {
            "method": "identity_uniform",
            "source_value": "identity_uniform",
            "label": "identity, uniform decision prior",
            "baseline": True,
        },
        {
            "method": "pooled",
            "source_value": "pooled_uniform",
            "label": "pooled",
            "baseline": False,
        },
        {
            "method": "frsc",
            "source_value": "fixed_reference_oneshot_uniform",
            "label": "FRSC",
            "baseline": False,
        },
        {
            "method": "fixed_prior_iterative_uniform",
            "source_value": "fixed_iterative_geometry_uniform",
            "label": "fixed-prior iterative, uniform decision prior",
            "baseline": False,
        },
        {
            "method": "joint_fit_geometry_uniform",
            "source_value": "joint_geometry_uniform",
            "label": "joint-fit geometry, uniform decision prior",
            "baseline": False,
        },
        {
            "method": "latent_im_diag",
            "source_value": "latent_im_diag_uniform",
            "label": "Latent-IM-Diag",
            "baseline": False,
        },
    ],
    "spdim": [
        {
            "method": "source_only_tsmnet",
            "source_value": "source_only_tsmnet",
            "label": "source-only TSMNet",
            "baseline": True,
        },
        {
            "method": "rct",
            "source_value": "rct",
            "label": "RCT",
            "baseline": False,
        },
        {
            "method": "spdim_geodesic",
            "source_value": "spdim_geodesic",
            "label": "SPDIM geodesic",
            "baseline": False,
        },
        {
            "method": "spdim_bias",
            "source_value": "spdim_bias",
            "label": "SPDIM bias",
            "baseline": False,
        },
    ],
}

PIPELINE_META = {
    "h2cmi": {
        "label": "H2CMI repaired W1",
        "baseline": "identity_uniform",
        "result_path": str(H2_RESULTS.relative_to(ROOT)),
        "result_sha256": H2_RESULT_SHA256,
        "result_commit": H2_RESULT_COMMIT,
    },
    "spdim": {
        "label": "Official SPDIM repaired W1",
        "baseline": "source_only_tsmnet",
        "result_path": str(SPDIM_RESULTS.relative_to(ROOT)),
        "result_sha256": SPDIM_RESULT_SHA256,
        "result_commit": SPDIM_RESULT_COMMIT,
    },
}

PIPELINE_PROPERTIES = {
    "h2cmi": {
        "backbone": "H2Encoder with HybridHead",
        "source_training_objective": (
            "H2 hybrid discriminative and class-conditional density objective; "
            "the accepted W1 source policy has CMI disabled"
        ),
        "baseline": "identity geometry with uniform decision prior",
        "feature_space": "frozen H2 latent z_c",
        "adaptation_action_family": (
            "class-conditional diagonal-affine latent transforms and decision-prior variants"
        ),
        "original_bootstrap": {
            "replicates": 5000,
            "seed": 20260709,
            "cluster": "dataset_x_target_subject",
        },
    },
    "spdim": {
        "backbone": "official TSMNet",
        "source_training_objective": "TSMNet source cross-entropy objective",
        "baseline": "unadapted source-only TSMNet",
        "feature_space": "TSMNet SPD-manifold representation",
        "adaptation_action_family": "RCT and official SPDIM geodesic or bias actions",
        "original_bootstrap": {
            "replicates": 10000,
            "seed": 20260710,
            "cluster": "dataset_x_target_subject",
        },
    },
}

SPDIM_CONTRASTS = {
    "rct_minus_source_only_tsmnet": ("rct", "source_only_tsmnet"),
    "spdim_geodesic_minus_source_only_tsmnet": ("spdim_geodesic", "source_only_tsmnet"),
    "spdim_bias_minus_source_only_tsmnet": ("spdim_bias", "source_only_tsmnet"),
    "spdim_geodesic_minus_rct": ("spdim_geodesic", "rct"),
    "spdim_bias_minus_rct": ("spdim_bias", "rct"),
}
HARM_THRESHOLDS = [0.0, -0.01, -0.02]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader), list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_origin_commit(path: Path) -> str:
    relative = str(path.relative_to(ROOT))
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def parse_list(value: str) -> list[Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, got {type(parsed).__name__}")
    return parsed


def trial_indices(values: list[str]) -> list[int]:
    marker = ":epoch="
    indices = []
    for value in values:
        if marker not in value:
            raise ValueError(f"trial id lacks epoch marker: {value}")
        indices.append(int(value.rsplit(marker, 1)[1]))
    return indices


def sha_indices(values: list[int]) -> str:
    text = ",".join(str(int(value)) for value in values)
    return hashlib.sha256(text.encode()).hexdigest()


def packet_hash(rows: list[dict[str, Any]], field: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["dataset"], item["target_subject"], item["source_seed"])):
        payload = {
            "dataset": row["dataset"],
            "target_subject": row["target_subject"],
            "source_seed": row["source_seed"],
            field: row[field],
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest() -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    rows, _ = read_csv(MANIFEST_CSV)
    normalized = []
    mapping = {}
    for row in rows:
        item = {
            "dataset": row["dataset"],
            "target_subject": int(row["target_subject"]),
            "source_seed": int(row["source_seed"]),
            "adapt_trial_ids": [str(value) for value in parse_list(row["adapt_trial_ids"])],
            "eval_trial_ids": [str(value) for value in parse_list(row["eval_trial_ids"])],
            "n_adapt": int(row["n_adapt"]),
            "n_eval": int(row["n_eval"]),
            "class_counts_adapt": [int(value) for value in parse_list(row["class_counts_adapt"])],
            "class_counts_eval": [int(value) for value in parse_list(row["class_counts_eval"])],
            "split_family": row["split_family"],
            "split_hash": row["split_hash"],
            "adapt_eval_disjoint": row["adapt_eval_disjoint"] == "True",
            "both_classes_adapt": row["both_classes_adapt"] == "True",
            "both_classes_eval": row["both_classes_eval"] == "True",
        }
        key = (item["dataset"], item["target_subject"], item["source_seed"])
        if key in mapping:
            raise ValueError(f"duplicate manifest key: {key}")
        if set(item["adapt_trial_ids"]) & set(item["eval_trial_ids"]):
            raise ValueError(f"manifest trial overlap: {key}")
        if item["split_family"] != SPLIT_FAMILY:
            raise ValueError(f"legacy or unexpected split in manifest: {key}")
        if not item["adapt_eval_disjoint"] or not item["both_classes_adapt"] or not item["both_classes_eval"]:
            raise ValueError(f"invalid repaired split flags: {key}")
        if len(item["adapt_trial_ids"]) != item["n_adapt"] or len(item["eval_trial_ids"]) != item["n_eval"]:
            raise ValueError(f"trial count mismatch: {key}")
        mapping[key] = item
        normalized.append(item)

    expected_keys = {
        (dataset, subject, seed)
        for dataset in DATASETS
        for subject in SUBJECTS[dataset]
        for seed in SEEDS
    }
    if set(mapping) != expected_keys or len(rows) != 345:
        raise ValueError("manifest does not contain the exact 115 x 3 repaired units")
    manifest_meta = json.loads(MANIFEST_JSON.read_text())
    if manifest_meta.get("manifest_hash") != MANIFEST_HASH or manifest_meta.get("n_manifest_rows") != 345:
        raise ValueError("manifest metadata hash/count mismatch")
    return normalized, mapping, manifest_meta


def expected_method_keys(pipeline: str) -> set[tuple[str, int, int, str]]:
    methods = [spec["method"] for spec in METHOD_SPECS[pipeline]]
    return {
        (dataset, subject, seed, method)
        for dataset in DATASETS
        for subject in SUBJECTS[dataset]
        for seed in SEEDS
        for method in methods
    }


def validate_common_row(row: dict[str, str], manifest_row: dict[str, Any], *, path: Path) -> None:
    if row["status"] != "ok" or row["failure_reason"]:
        raise ValueError(f"non-ok result in {path}: {row}")
    if row["split_family"] != SPLIT_FAMILY:
        raise ValueError(f"legacy split row entered {path}")
    if row["manifest_hash"] != MANIFEST_HASH or row["split_hash"] != manifest_row["split_hash"]:
        raise ValueError(f"manifest/split hash mismatch in {path}")
    if int(row["n_adapt"]) != manifest_row["n_adapt"] or int(row["n_eval"]) != manifest_row["n_eval"]:
        raise ValueError(f"adapt/eval count mismatch in {path}")
    if [int(value) for value in parse_list(row["class_counts_adapt"])] != manifest_row["class_counts_adapt"]:
        raise ValueError(f"adapt class count mismatch in {path}")
    if [int(value) for value in parse_list(row["class_counts_eval"])] != manifest_row["class_counts_eval"]:
        raise ValueError(f"eval class count mismatch in {path}")
    value = float(row["bacc"])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid bAcc in {path}: {value}")


def load_h2_rows(manifest: dict[tuple[str, int, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw, _ = read_csv(H2_RESULTS)
    source_to_spec = {spec["source_value"]: spec for spec in METHOD_SPECS["h2cmi"]}
    selected = []
    keys = set()
    units = set()
    for row in raw:
        if row["branch"] not in source_to_spec:
            continue
        spec = source_to_spec[row["branch"]]
        unit = (row["dataset"], int(row["target_subject"]), int(row["source_seed"]))
        validate_common_row(row, manifest[unit], path=H2_RESULTS)
        key = (*unit, spec["method"])
        if key in keys:
            raise ValueError(f"duplicate selected H2CMI key: {key}")
        keys.add(key)
        units.add(unit)
        selected.append({
            "pipeline": "h2cmi",
            "dataset": unit[0],
            "target_subject": unit[1],
            "source_seed": unit[2],
            "method": spec["method"],
            "bacc": float(row["bacc"]),
        })
    if keys != expected_method_keys("h2cmi") or len(selected) != 2070 or len(units) != 345:
        raise ValueError("H2CMI selected-method coverage mismatch")
    return selected, {
        "raw_rows": len(raw),
        "selected_rows": len(selected),
        "target_seed_units": len(units),
        "selected_methods": [spec["method"] for spec in METHOD_SPECS["h2cmi"]],
        "all_selected_rows_repaired_split": True,
    }


def load_spdim_rows(manifest: dict[tuple[str, int, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw, _ = read_csv(SPDIM_RESULTS)
    allowed = {spec["source_value"]: spec for spec in METHOD_SPECS["spdim"]}
    selected = []
    keys = set()
    units = set()
    adapt_hash_failures = 0
    eval_hash_failures = 0
    for row in raw:
        if row["method"] not in allowed:
            raise ValueError(f"unexpected SPDIM method: {row['method']}")
        spec = allowed[row["method"]]
        unit = (row["dataset"], int(row["target_subject"]), int(row["source_seed"]))
        manifest_row = manifest[unit]
        validate_common_row(row, manifest_row, path=SPDIM_RESULTS)
        if row["adapt_eval_disjoint"] != "True" or row["both_classes_adapt"] != "True" or row["both_classes_eval"] != "True":
            raise ValueError(f"SPDIM split flag failure: {unit}")
        if row["target_label_leakage_detected"] != "False" or row["method_selection_uses_target_performance"] != "False":
            raise ValueError(f"SPDIM leakage/selection flag failure: {unit}")
        if row["official_pretrained_weight_used"] != "False" or row["third_party_vendored"] != "False":
            raise ValueError(f"SPDIM provenance policy failure: {unit}")
        adapt_expected = sha_indices(trial_indices(manifest_row["adapt_trial_ids"]))
        eval_expected = sha_indices(trial_indices(manifest_row["eval_trial_ids"]))
        adapt_hash_failures += row["adapt_idx_sha256"] != adapt_expected
        eval_hash_failures += row["eval_idx_sha256"] != eval_expected
        key = (*unit, spec["method"])
        if key in keys:
            raise ValueError(f"duplicate SPDIM key: {key}")
        keys.add(key)
        units.add(unit)
        selected.append({
            "pipeline": "spdim",
            "dataset": unit[0],
            "target_subject": unit[1],
            "source_seed": unit[2],
            "method": spec["method"],
            "bacc": float(row["bacc"]),
        })
    if keys != expected_method_keys("spdim") or len(selected) != 1380 or len(units) != 345:
        raise ValueError("SPDIM selected-method coverage mismatch")
    if adapt_hash_failures or eval_hash_failures:
        raise ValueError("SPDIM trial-index hashes do not match the repaired manifest")
    return selected, {
        "raw_rows": len(raw),
        "selected_rows": len(selected),
        "target_seed_units": len(units),
        "selected_methods": [spec["method"] for spec in METHOD_SPECS["spdim"]],
        "adapt_index_hash_failures": adapt_hash_failures,
        "eval_index_hash_failures": eval_hash_failures,
        "all_selected_rows_repaired_split": True,
    }


def build_seed_averaged_units(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], float]:
    grouped: dict[tuple[str, str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (row["pipeline"], row["dataset"], row["target_subject"], row["method"])
        grouped[key].append((row["source_seed"], row["bacc"]))
    units = {}
    for key, values in grouped.items():
        if sorted(seed for seed, _ in values) != SEEDS:
            raise ValueError(f"missing or duplicate source seed for {key}")
        units[key] = float(np.mean([value for _, value in values]))
    expected = sum(len(specs) for specs in METHOD_SPECS.values()) * 115
    if len(units) != expected:
        raise ValueError(f"expected {expected} seed-averaged method units, got {len(units)}")
    return units


def method_label(pipeline: str, method: str) -> str:
    matches = [spec["label"] for spec in METHOD_SPECS[pipeline] if spec["method"] == method]
    if len(matches) != 1:
        raise ValueError(f"unknown method: {pipeline}/{method}")
    return matches[0]


def bootstrap_indices() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {
        dataset: rng.integers(
            0,
            len(SUBJECTS[dataset]),
            size=(BOOTSTRAP_REPLICATES, len(SUBJECTS[dataset])),
        )
        for dataset in DATASETS
    }


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def build_result_rows(
    units: dict[tuple[str, str, int, str], float],
    indices: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], np.ndarray]]:
    output = []
    distributions = {}
    for pipeline, specs in METHOD_SPECS.items():
        baseline = PIPELINE_META[pipeline]["baseline"]
        arrays = {
            method: {
                dataset: np.asarray(
                    [units[(pipeline, dataset, subject, method)] for subject in SUBJECTS[dataset]],
                    dtype=float,
                )
                for dataset in DATASETS
            }
            for method in [spec["method"] for spec in specs]
        }
        for spec in specs:
            method = spec["method"]
            method_dist = {
                dataset: arrays[method][dataset][indices[dataset]].mean(axis=1)
                for dataset in DATASETS
            }
            delta_values = {
                dataset: arrays[method][dataset] - arrays[baseline][dataset]
                for dataset in DATASETS
            }
            delta_dist = {
                dataset: delta_values[dataset][indices[dataset]].mean(axis=1)
                for dataset in DATASETS
            }
            for dataset in DATASETS:
                distributions[(pipeline, method, "per_dataset", dataset)] = method_dist[dataset]
                method_low, method_high = percentile_interval(method_dist[dataset])
                delta_low, delta_high = percentile_interval(delta_dist[dataset])
                output.append(result_row(
                    pipeline=pipeline,
                    method=method,
                    baseline=baseline,
                    scope="per_dataset",
                    dataset=dataset,
                    mean_bacc=float(arrays[method][dataset].mean()),
                    ci=(method_low, method_high),
                    delta=float(delta_values[dataset].mean()),
                    delta_ci=(delta_low, delta_high),
                    n_targets=len(SUBJECTS[dataset]),
                    n_datasets=1,
                ))

            subject_weighted_dist = sum(
                method_dist[dataset] * len(SUBJECTS[dataset]) for dataset in DATASETS
            ) / 115.0
            subject_weighted_delta_dist = sum(
                delta_dist[dataset] * len(SUBJECTS[dataset]) for dataset in DATASETS
            ) / 115.0
            dataset_macro_dist = sum(method_dist.values()) / len(DATASETS)
            dataset_macro_delta_dist = sum(delta_dist.values()) / len(DATASETS)
            observed_by_dataset = {
                dataset: float(arrays[method][dataset].mean()) for dataset in DATASETS
            }
            observed_delta_by_dataset = {
                dataset: float(delta_values[dataset].mean()) for dataset in DATASETS
            }
            observed_subject_weighted = sum(
                observed_by_dataset[dataset] * len(SUBJECTS[dataset]) for dataset in DATASETS
            ) / 115.0
            observed_subject_weighted_delta = sum(
                observed_delta_by_dataset[dataset] * len(SUBJECTS[dataset]) for dataset in DATASETS
            ) / 115.0
            observed_dataset_macro = float(np.mean(list(observed_by_dataset.values())))
            observed_dataset_macro_delta = float(np.mean(list(observed_delta_by_dataset.values())))
            for scope, method_distribution, delta_distribution, estimate, delta_estimate in (
                (
                    "subject_weighted",
                    subject_weighted_dist,
                    subject_weighted_delta_dist,
                    observed_subject_weighted,
                    observed_subject_weighted_delta,
                ),
                (
                    "dataset_macro",
                    dataset_macro_dist,
                    dataset_macro_delta_dist,
                    observed_dataset_macro,
                    observed_dataset_macro_delta,
                ),
            ):
                distributions[(pipeline, method, scope, "ALL")] = method_distribution
                output.append(result_row(
                    pipeline=pipeline,
                    method=method,
                    baseline=baseline,
                    scope=scope,
                    dataset="ALL",
                    mean_bacc=estimate,
                    ci=percentile_interval(method_distribution),
                    delta=delta_estimate,
                    delta_ci=percentile_interval(delta_distribution),
                    n_targets=115,
                    n_datasets=3,
                ))
    if len(output) != 50:
        raise ValueError(f"expected 50 standardized result rows, got {len(output)}")
    return output, distributions


def result_row(
    *,
    pipeline: str,
    method: str,
    baseline: str,
    scope: str,
    dataset: str,
    mean_bacc: float,
    ci: tuple[float, float],
    delta: float,
    delta_ci: tuple[float, float],
    n_targets: int,
    n_datasets: int,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "pipeline_label": PIPELINE_META[pipeline]["label"],
        "method": method,
        "method_label": method_label(pipeline, method),
        "baseline_method": baseline,
        "is_pipeline_baseline": method == baseline,
        "metric": "bacc",
        "scope": scope,
        "dataset": dataset,
        "mean_bacc": mean_bacc,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "delta_vs_pipeline_baseline": delta,
        "delta_ci_low": delta_ci[0],
        "delta_ci_high": delta_ci[1],
        "n_target_subjects": n_targets,
        "n_datasets": n_datasets,
        "n_source_seeds": 3,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_cluster": "dataset_x_target_subject",
        "dataset_stratified": True,
        "seeds_averaged_first": True,
        "interval_label": INTERVAL_LABEL,
        "status": "ok",
        "provenance_artifact": PIPELINE_META[pipeline]["result_path"],
        "result_sha256": PIPELINE_META[pipeline]["result_sha256"],
        "result_commit": PIPELINE_META[pipeline]["result_commit"],
    }


def find_result_row(
    rows: list[dict[str, Any]],
    *,
    pipeline: str,
    method: str,
    scope: str,
    dataset: str = "ALL",
) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row["pipeline"] == pipeline
        and row["method"] == method
        and row["scope"] == scope
        and row["dataset"] == dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one result row, got {len(matches)}")
    return matches[0]


def build_harm_rows(units: dict[tuple[str, str, int, str], float]) -> list[dict[str, Any]]:
    output = []
    for pipeline, specs in METHOD_SPECS.items():
        baseline = PIPELINE_META[pipeline]["baseline"]
        for spec in specs:
            method = spec["method"]
            if method == baseline:
                continue
            deltas = {
                dataset: [
                    units[(pipeline, dataset, subject, method)]
                    - units[(pipeline, dataset, subject, baseline)]
                    for subject in SUBJECTS[dataset]
                ]
                for dataset in DATASETS
            }
            for threshold in HARM_THRESHOLDS:
                per_dataset_counts = {
                    dataset: sum(value < threshold for value in deltas[dataset])
                    for dataset in DATASETS
                }
                per_dataset_rates = {
                    dataset: per_dataset_counts[dataset] / len(SUBJECTS[dataset])
                    for dataset in DATASETS
                }
                for dataset in DATASETS:
                    output.append(harm_row(
                        pipeline=pipeline,
                        method=method,
                        baseline=baseline,
                        threshold=threshold,
                        scope="per_dataset",
                        dataset=dataset,
                        harm_count=per_dataset_counts[dataset],
                        denominator=len(SUBJECTS[dataset]),
                        harm_rate=per_dataset_rates[dataset],
                        raw_harm_count=per_dataset_counts[dataset],
                        raw_denominator=len(SUBJECTS[dataset]),
                        count_definition="subject_count",
                    ))
                total_count = sum(per_dataset_counts.values())
                output.append(harm_row(
                    pipeline=pipeline,
                    method=method,
                    baseline=baseline,
                    threshold=threshold,
                    scope="subject_weighted",
                    dataset="ALL",
                    harm_count=total_count,
                    denominator=115,
                    harm_rate=total_count / 115.0,
                    raw_harm_count=total_count,
                    raw_denominator=115,
                    count_definition="subject_count",
                ))
                output.append(harm_row(
                    pipeline=pipeline,
                    method=method,
                    baseline=baseline,
                    threshold=threshold,
                    scope="dataset_macro",
                    dataset="ALL",
                    harm_count="NA",
                    denominator=3,
                    harm_rate=float(np.mean(list(per_dataset_rates.values()))),
                    raw_harm_count=total_count,
                    raw_denominator=115,
                    count_definition=(
                        "harm_count_not_defined_for_macro_rate; denominator is datasets; "
                        "raw subject counts are context only"
                    ),
                ))
    if len(output) != 120:
        raise ValueError(f"expected 120 harm rows, got {len(output)}")
    return output


def harm_row(
    *,
    pipeline: str,
    method: str,
    baseline: str,
    threshold: float,
    scope: str,
    dataset: str,
    harm_count: int | str,
    denominator: int,
    harm_rate: float,
    raw_harm_count: int,
    raw_denominator: int,
    count_definition: str,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "pipeline_label": PIPELINE_META[pipeline]["label"],
        "method": method,
        "method_label": method_label(pipeline, method),
        "baseline_method": baseline,
        "baseline_label": method_label(pipeline, baseline),
        "metric": "bacc",
        "threshold_operator": "<",
        "threshold": threshold,
        "scope": scope,
        "dataset": dataset,
        "harm_count": harm_count,
        "denominator": denominator,
        "harm_rate": harm_rate,
        "raw_subject_harm_count_context": raw_harm_count,
        "raw_subject_denominator_context": raw_denominator,
        "count_definition": count_definition,
        "seeds_averaged_first": True,
        "status": "ok",
        "provenance_artifact": PIPELINE_META[pipeline]["result_path"],
        "result_sha256": PIPELINE_META[pipeline]["result_sha256"],
    }


def compare_p9_accepted_statistics(
    result_rows: list[dict[str, Any]],
    distributions: dict[tuple[str, str, str, str], np.ndarray],
) -> dict[str, Any]:
    method_rows, _ = read_csv(SPDIM_METHOD_CI)
    contrast_rows, _ = read_csv(SPDIM_CONTRAST_CI)
    method_checks = {}
    for spec in METHOD_SPECS["spdim"]:
        method = spec["method"]
        accepted = next(
            row for row in method_rows
            if row["metric"] == "bacc"
            and row["method"] == method
            and row["estimand"] == "subject_weighted"
            and row["dataset"] == "ALL"
        )
        recomputed = find_result_row(
            result_rows,
            pipeline="spdim",
            method=method,
            scope="subject_weighted",
        )
        diffs = {
            "estimate": abs(float(accepted["estimate"]) - recomputed["mean_bacc"]),
            "ci_low": abs(float(accepted["ci_low"]) - recomputed["ci_low"]),
            "ci_high": abs(float(accepted["ci_high"]) - recomputed["ci_high"]),
        }
        method_checks[method] = {"max_abs_difference": max(diffs.values()), "fields": diffs}

    contrast_checks = {}
    for name, (method, baseline) in SPDIM_CONTRASTS.items():
        accepted = next(
            row for row in contrast_rows
            if row["metric"] == "bacc"
            and row["contrast"] == name
            and row["estimand"] == "subject_weighted"
            and row["dataset"] == "ALL"
        )
        method_row = find_result_row(
            result_rows,
            pipeline="spdim",
            method=method,
            scope="subject_weighted",
        )
        baseline_row = find_result_row(
            result_rows,
            pipeline="spdim",
            method=baseline,
            scope="subject_weighted",
        )
        paired_distribution = (
            distributions[("spdim", method, "subject_weighted", "ALL")]
            - distributions[("spdim", baseline, "subject_weighted", "ALL")]
        )
        paired_low, paired_high = percentile_interval(paired_distribution)
        recomputed = {
            "estimate": method_row["mean_bacc"] - baseline_row["mean_bacc"],
            "ci_low": paired_low,
            "ci_high": paired_high,
        }
        diffs = {
            "estimate": abs(float(accepted["estimate"]) - recomputed["estimate"]),
            "ci_low": abs(float(accepted["ci_low"]) - recomputed["ci_low"]),
            "ci_high": abs(float(accepted["ci_high"]) - recomputed["ci_high"]),
        }
        contrast_checks[name] = {"max_abs_difference": max(diffs.values()), "fields": diffs}
    max_difference = max(
        [item["max_abs_difference"] for item in method_checks.values()]
        + [item["max_abs_difference"] for item in contrast_checks.values()]
    )
    if max_difference > 1e-12:
        raise ValueError(f"P10 standardized SPDIM statistics differ from accepted P9: {max_difference}")
    return {
        "method_subject_weighted_checks": method_checks,
        "contrast_subject_weighted_checks": contrast_checks,
        "max_abs_difference": max_difference,
        "pass": True,
    }


def exact_spdim_headlines() -> tuple[dict[str, Any], dict[str, Any]]:
    method_rows, _ = read_csv(SPDIM_METHOD_CI)
    contrast_rows, _ = read_csv(SPDIM_CONTRAST_CI)
    methods = {}
    for spec in METHOD_SPECS["spdim"]:
        method = spec["method"]
        row = next(
            item for item in method_rows
            if item["metric"] == "bacc"
            and item["method"] == method
            and item["estimand"] == "subject_weighted"
            and item["dataset"] == "ALL"
        )
        methods[method] = {
            "estimate": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }
    contrasts = {}
    for name in SPDIM_CONTRASTS:
        scope_values = {}
        for estimand in ("subject_weighted", "dataset_macro"):
            row = next(
                item for item in contrast_rows
                if item["metric"] == "bacc"
                and item["contrast"] == name
                and item["estimand"] == estimand
                and item["dataset"] == "ALL"
            )
            scope_values[estimand] = {
                "estimate": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
        contrasts[name] = scope_values
    return methods, contrasts


def h2_decomposition_headlines() -> dict[str, float]:
    rows, _ = read_csv(H2_RESULTS)
    decomposition = [row for row in rows if row["branch"] == "__decomposition__"]
    if len(decomposition) != 345:
        raise ValueError("H2CMI decomposition row count mismatch")
    values = {
        "G": float(np.mean([float(row["G"]) for row in decomposition])),
        "P": float(np.mean([float(row["P"]) for row in decomposition])),
        "I_int": float(np.mean([float(row["I_int"]) for row in decomposition])),
        "full_joint_delta": float(np.mean([float(row["total_joint_delta"]) for row in decomposition])),
    }
    expected = {
        "G": 0.007827697262479871,
        "P": -0.009660225442834139,
        "I_int": 0.005238325281803545,
        "full_joint_delta": 0.003405797101449273,
    }
    if max(abs(values[key] - expected[key]) for key in expected) > 1e-12:
        raise ValueError("H2CMI accepted decomposition headlines do not recompute")
    return values


def result_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "targets": {
            dataset: sorted({
                row["target_subject"] for row in rows if row["dataset"] == dataset
            })
            for dataset in DATASETS
        },
        "source_seeds": sorted({row["source_seed"] for row in rows}),
        "metric": "bacc",
        "standardized_bootstrap_cluster": "dataset_x_target_subject",
    }


def comparability_payload(
    manifest_rows: list[dict[str, Any]],
    h2_validation: dict[str, Any],
    spdim_validation: dict[str, Any],
    h2_rows: list[dict[str, Any]],
    spdim_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    adapt_packet = packet_hash(manifest_rows, "adapt_trial_ids")
    eval_packet = packet_hash(manifest_rows, "eval_trial_ids")
    h2_coverage = result_coverage(h2_rows)
    spdim_coverage = result_coverage(spdim_rows)
    expected_targets = {dataset: SUBJECTS[dataset] for dataset in DATASETS}
    same_target_subjects = (
        h2_coverage["targets"] == spdim_coverage["targets"] == expected_targets
    )
    same_source_seeds = (
        h2_coverage["source_seeds"] == spdim_coverage["source_seeds"] == SEEDS
    )
    same_adaptation_trial_ids = bool(
        same_target_subjects
        and h2_validation["all_selected_rows_repaired_split"]
        and spdim_validation["all_selected_rows_repaired_split"]
        and spdim_validation["adapt_index_hash_failures"] == 0
    )
    same_evaluation_trial_ids = bool(
        same_target_subjects
        and h2_validation["all_selected_rows_repaired_split"]
        and spdim_validation["all_selected_rows_repaired_split"]
        and spdim_validation["eval_index_hash_failures"] == 0
    )
    same_metric = h2_coverage["metric"] == spdim_coverage["metric"] == "bacc"
    same_bootstrap_cluster = (
        h2_coverage["standardized_bootstrap_cluster"]
        == spdim_coverage["standardized_bootstrap_cluster"]
        == "dataset_x_target_subject"
    )
    same_backbone = (
        PIPELINE_PROPERTIES["h2cmi"]["backbone"]
        == PIPELINE_PROPERTIES["spdim"]["backbone"]
    )
    same_source_training_objective = (
        PIPELINE_PROPERTIES["h2cmi"]["source_training_objective"]
        == PIPELINE_PROPERTIES["spdim"]["source_training_objective"]
    )
    same_source_only_baseline = (
        PIPELINE_PROPERTIES["h2cmi"]["baseline"]
        == PIPELINE_PROPERTIES["spdim"]["baseline"]
    )
    same_feature_space = (
        PIPELINE_PROPERTIES["h2cmi"]["feature_space"]
        == PIPELINE_PROPERTIES["spdim"]["feature_space"]
    )
    same_adaptation_action_family = (
        PIPELINE_PROPERTIES["h2cmi"]["adaptation_action_family"]
        == PIPELINE_PROPERTIES["spdim"]["adaptation_action_family"]
    )
    adapter_only_head_to_head_valid = all([
        same_target_subjects,
        same_adaptation_trial_ids,
        same_evaluation_trial_ids,
        same_source_seeds,
        same_metric,
        same_bootstrap_cluster,
        same_backbone,
        same_source_training_objective,
        same_source_only_baseline,
        same_feature_space,
        same_adaptation_action_family,
    ])
    full_pipeline_same_split_comparison_valid = all([
        same_target_subjects,
        same_adaptation_trial_ids,
        same_evaluation_trial_ids,
        same_source_seeds,
        same_metric,
        same_bootstrap_cluster,
    ])
    same_original_bootstrap_configuration = (
        PIPELINE_PROPERTIES["h2cmi"]["original_bootstrap"]
        == PIPELINE_PROPERTIES["spdim"]["original_bootstrap"]
    )
    return {
        "status": "pass",
        "audit_label": "P10 repaired-W1 pipeline comparability audit",
        "h2cmi_result_commit": H2_RESULT_COMMIT,
        "spdim_result_commit": SPDIM_RESULT_COMMIT,
        "h2cmi_result_sha256": H2_RESULT_SHA256,
        "spdim_result_sha256": SPDIM_RESULT_SHA256,
        "repaired_manifest_hash": MANIFEST_HASH,
        "same_target_subjects": same_target_subjects,
        "same_adaptation_trial_ids": same_adaptation_trial_ids,
        "same_evaluation_trial_ids": same_evaluation_trial_ids,
        "same_source_seeds": same_source_seeds,
        "same_metric": same_metric,
        "same_bootstrap_cluster": same_bootstrap_cluster,
        "same_backbone": same_backbone,
        "same_source_training_objective": same_source_training_objective,
        "same_source_only_baseline": same_source_only_baseline,
        "same_feature_space": same_feature_space,
        "same_adaptation_action_family": same_adaptation_action_family,
        "adapter_only_head_to_head_valid": adapter_only_head_to_head_valid,
        "full_pipeline_same_split_comparison_valid": full_pipeline_same_split_comparison_valid,
        "same_original_bootstrap_configuration": same_original_bootstrap_configuration,
        "standardized_posthoc_interval_policy": {
            "label": INTERVAL_LABEL,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "dataset_x_target_subject",
            "dataset_stratified": True,
            "source_seeds_averaged_first": True,
        },
        "split_evidence": {
            "manifest_rows": 345,
            "target_subjects": 115,
            "source_seeds": SEEDS,
            "datasets": DATASETS,
            "adaptation_trial_id_packet_sha256": adapt_packet,
            "evaluation_trial_id_packet_sha256": eval_packet,
            "h2cmi_result_split_hashes_match_manifest": True,
            "spdim_result_split_and_index_hashes_match_manifest": True,
            "h2cmi_validation": h2_validation,
            "spdim_validation": spdim_validation,
        },
        "coverage": {"h2cmi": h2_coverage, "spdim": spdim_coverage},
        "pipeline_differences": PIPELINE_PROPERTIES,
        "permitted_interpretation": (
            "same-split full-pipeline comparison across independently trained pipelines"
        ),
        "prohibited_claims": [
            "H2CMI outperforms SPDIM as an adaptation algorithm",
            "SPDIM outperforms H2CMI as an adaptation algorithm",
            "absolute bAcc differences are caused solely by the adapter",
        ],
    }


def write_comparability_markdown(payload: dict[str, Any]) -> None:
    fields = [
        "same_target_subjects",
        "same_adaptation_trial_ids",
        "same_evaluation_trial_ids",
        "same_source_seeds",
        "same_metric",
        "same_bootstrap_cluster",
        "same_backbone",
        "same_source_training_objective",
        "same_source_only_baseline",
        "same_feature_space",
        "same_adaptation_action_family",
        "adapter_only_head_to_head_valid",
        "full_pipeline_same_split_comparison_valid",
    ]
    reasons = {
        "same_target_subjects": "Both packets cover the same 115 repaired-W1 targets.",
        "same_adaptation_trial_ids": "Both consume the frozen manifest; all result split hashes match it.",
        "same_evaluation_trial_ids": "Both consume the frozen manifest; SPDIM index hashes also recompute exactly.",
        "same_source_seeds": "Both use source seeds 0, 1, and 2.",
        "same_metric": "Both are summarized with balanced accuracy on the repaired evaluation blocks.",
        "same_bootstrap_cluster": "P10 standardizes both to dataset x target_subject clusters.",
        "same_backbone": "H2Encoder/HybridHead and official TSMNet are different models.",
        "same_source_training_objective": "The H2 hybrid objective differs from TSMNet cross-entropy training.",
        "same_source_only_baseline": "Identity-H2 and source-only TSMNet are different trained baselines.",
        "same_feature_space": "H2 latent z_c and the TSMNet SPD representation are different.",
        "same_adaptation_action_family": "The pipelines expose different adaptation operators.",
        "adapter_only_head_to_head_valid": "Backbone, source model, feature space, baseline, and actions are not controlled.",
        "full_pipeline_same_split_comparison_valid": "Targets, split, seeds, and metric are harmonized.",
    }
    lines = [
        "# Repaired-W1 H2CMI/SPDIM Pipeline Comparability Audit",
        "",
        "- status: `pass`",
        f"- H2CMI result commit: `{H2_RESULT_COMMIT}`",
        f"- SPDIM result commit: `{SPDIM_RESULT_COMMIT}`",
        f"- H2CMI result SHA-256: `{H2_RESULT_SHA256}`",
        f"- SPDIM result SHA-256: `{SPDIM_RESULT_SHA256}`",
        f"- repaired manifest hash: `{MANIFEST_HASH}`",
        f"- standardized interval label: `{INTERVAL_LABEL}`",
        "",
        "## Gate",
        "",
        "| field | value | evidence/interpretation |",
        "|---|---|---|",
    ]
    for field in fields:
        lines.append(f"| `{field}` | `{str(payload[field]).lower()}` | {reasons[field]} |")
    lines.extend([
        "",
        "## Valid Interpretation",
        "",
        "This is a same-split full-pipeline comparison. It is not a controlled adapter-only comparison.",
        "",
        "The original P7 and P9 interval configurations were not identical. P10 recomputes both under the explicitly post-hoc 10,000-replicate, seed-20260710, dataset-stratified subject-cluster policy.",
        "",
        "## Prohibited Claims",
        "",
        "- H2CMI outperforms SPDIM as an adaptation algorithm.",
        "- SPDIM outperforms H2CMI as an adaptation algorithm.",
        "- Any absolute bAcc difference is attributable solely to the adapter.",
    ])
    COMPARABILITY_MD.write_text("\n".join(lines) + "\n")


def format_interval(row: dict[str, Any], field: str = "mean_bacc") -> str:
    if field == "mean_bacc":
        return f"{row['mean_bacc']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
    return (
        f"{row['delta_vs_pipeline_baseline']:+.4f} "
        f"[{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}]"
    )


def write_results_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Standardized Repaired-W1 Cross-Pipeline Results",
        "",
        f"All intervals in this file are labeled `{INTERVAL_LABEL}`. They are post-hoc comparability intervals, not preregistered primary intervals.",
        "",
        "Seeds 0/1/2 are averaged within dataset x target_subject x method before aggregation. Bootstrap: 10,000 replicates, seed 20260710, dataset-stratified target-subject clusters.",
        "",
    ]
    for pipeline in ("h2cmi", "spdim"):
        lines.extend([
            f"## {PIPELINE_META[pipeline]['label']}",
            "",
            "| method | BNCI2014-001 | Cho2017 | Lee2019-MI | subject-weighted | delta vs own baseline | dataset-macro |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for spec in METHOD_SPECS[pipeline]:
            method = spec["method"]
            dataset_rows = {
                dataset: find_result_row(
                    rows,
                    pipeline=pipeline,
                    method=method,
                    scope="per_dataset",
                    dataset=dataset,
                )
                for dataset in DATASETS
            }
            sw = find_result_row(rows, pipeline=pipeline, method=method, scope="subject_weighted")
            dm = find_result_row(rows, pipeline=pipeline, method=method, scope="dataset_macro")
            lines.append(
                f"| {spec['label']} | {format_interval(dataset_rows['BNCI2014_001'])} | "
                f"{format_interval(dataset_rows['Cho2017'])} | {format_interval(dataset_rows['Lee2019_MI'])} | "
                f"{format_interval(sw)} | {format_interval(sw, 'delta')} | {format_interval(dm)} |"
            )
        lines.append("")
    lines.extend([
        "## Claim Boundary",
        "",
        "Absolute values may be compared only as a same-split full-pipeline diagnostic. Each delta is paired only against that pipeline's own baseline. The table does not isolate an adapter effect across pipelines.",
    ])
    CROSS_RESULTS_MD.write_text("\n".join(lines) + "\n")


def write_harm_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Repaired-W1 Cross-Pipeline Harm Audit",
        "",
        "Every delta is seed-averaged at the target-subject level and is relative to the method's own pipeline baseline.",
        "",
        "Harm rates must not be compared as if H2CMI identity and source-only TSMNet were the same source model.",
        "",
        "For dataset-macro rows, an integer harm count is not mathematically defined because the estimand is the arithmetic mean of three unequal-denominator dataset rates. The CSV reports `NA` for that count, denominator `3`, the macro rate, and raw subject counts only as context.",
        "",
        "| pipeline | method | threshold | scope | dataset | harm count | denominator | harm rate |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pipeline']} | {row['method_label']} | {row['threshold']:+.2f} | "
            f"{row['scope']} | {row['dataset']} | {row['harm_count']} | "
            f"{row['denominator']} | {row['harm_rate']:.4f} |"
        )
    HARM_MD.write_text("\n".join(lines) + "\n")


def write_freeze_markdown(
    h2_headlines: dict[str, float],
    spdim_methods: dict[str, Any],
    spdim_contrasts: dict[str, Any],
) -> None:
    lines = [
        "# Final Repaired-W1 Evidence Freeze",
        "",
        "## Frozen Status",
        "",
        "- Legacy W1 split: quarantined; no legacy row enters this freeze.",
        "- H2CMI repaired W1: confirmatory accepted packet.",
        "- SPDIM: official repaired-split three-source-seed same-split baseline.",
        "- Additional GPU work required: `false`.",
        f"- H2CMI source SHA-256: `{H2_RESULT_SHA256}`",
        f"- SPDIM source SHA-256: `{SPDIM_RESULT_SHA256}`",
        f"- repaired manifest hash: `{MANIFEST_HASH}`",
        "",
        "## H2CMI Repaired-Split Headlines",
        "",
        f"- G: `{h2_headlines['G']:+.7f}`",
        f"- P: `{h2_headlines['P']:+.7f}`",
        f"- I_int: `{h2_headlines['I_int']:+.7f}`",
        f"- full joint delta: `{h2_headlines['full_joint_delta']:+.7f}`",
        "",
        "## Official SPDIM Repaired-Split Headlines",
        "",
        "| method | subject-weighted bAcc [95% CI] |",
        "|---|---:|",
    ]
    for spec in METHOD_SPECS["spdim"]:
        row = spdim_methods[spec["method"]]
        lines.append(
            f"| {spec['label']} | {row['estimate']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines.extend([
        "",
        "| paired contrast | subject-weighted estimate [95% CI] |",
        "|---|---:|",
    ])
    for name in SPDIM_CONTRASTS:
        row = spdim_contrasts[name]["subject_weighted"]
        lines.append(
            f"| {name} | {row['estimate']:+.4f} "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "## Final Claim Gate",
        "",
        "RCT improves over source-only. Neither SPDIM geodesic nor SPDIM bias improves over RCT. Equivalence and noninferiority are not supported.",
        "",
        "H2CMI and SPDIM can be shown in a same-split full-pipeline table, but their absolute difference cannot be attributed solely to adaptation because backbone, source objective, baseline, feature space, and action family differ.",
    ])
    FREEZE_MD.write_text("\n".join(lines) + "\n")


def append_command_log() -> None:
    marker = "Per PM P10, completed the CPU-only repaired-W1 comparability audit and final evidence freeze"
    text = COMMAND_LOG.read_text()
    if marker in text:
        return
    entry = f"""
- {marker}. Source artifacts remained byte-identical: H2CMI
  `{H2_RESULT_SHA256}` and official SPDIM `{SPDIM_RESULT_SHA256}`. Both packets
  match repaired manifest `{MANIFEST_HASH}`. The standardized table contains
  50 pipeline-method-scope rows; the harmonized harm audit contains 120 rows.
  All newly recomputed intervals are labeled
  `{INTERVAL_LABEL}` and use 10,000 dataset-stratified target-subject cluster
  replicates with seed `{BOOTSTRAP_SEED}` after averaging source seeds. The
  audit permits only same-split full-pipeline comparison, not adapter-only
  attribution. No experiment, model run, split change, seed, or method was
  added.
"""
    COMMAND_LOG.write_text(text.rstrip() + "\n\n" + entry.lstrip())


def main() -> None:
    source_checks = {
        "h2cmi_result_sha256": sha256_file(H2_RESULTS),
        "spdim_result_sha256": sha256_file(SPDIM_RESULTS),
        "h2cmi_result_commit": git_origin_commit(H2_RESULTS),
        "spdim_result_commit": git_origin_commit(SPDIM_RESULTS),
    }
    if source_checks != {
        "h2cmi_result_sha256": H2_RESULT_SHA256,
        "spdim_result_sha256": SPDIM_RESULT_SHA256,
        "h2cmi_result_commit": H2_RESULT_COMMIT,
        "spdim_result_commit": SPDIM_RESULT_COMMIT,
    }:
        raise RuntimeError(f"source artifact gate failed: {source_checks}")

    h2_summary = json.loads(H2_SUMMARY.read_text())
    spdim_summary = json.loads(SPDIM_SUMMARY.read_text())
    legacy_quarantine = json.loads(LEGACY_QUARANTINE.read_text())
    if h2_summary.get("status") != "pass" or not h2_summary.get("validation", {}).get("validation_pass"):
        raise RuntimeError("accepted P7 summary is not pass")
    if spdim_summary.get("status") != "pass" or not spdim_summary.get("validation", {}).get("validation_pass"):
        raise RuntimeError("accepted P9 summary is not pass")
    if legacy_quarantine.get("status") != "pass" or legacy_quarantine.get("prohibited_future_use") != "confirmatory_mi_aggregate_or_spdim_baseline":
        raise RuntimeError("legacy split quarantine is not active")

    manifest_rows, manifest, manifest_meta = load_manifest()
    h2_rows, h2_validation = load_h2_rows(manifest)
    spdim_rows, spdim_validation = load_spdim_rows(manifest)
    units = build_seed_averaged_units(h2_rows + spdim_rows)
    indices = bootstrap_indices()
    result_rows, distributions = build_result_rows(units, indices)
    harm_rows = build_harm_rows(units)
    p9_recompute = compare_p9_accepted_statistics(result_rows, distributions)
    h2_headlines = h2_decomposition_headlines()
    spdim_methods, spdim_contrasts = exact_spdim_headlines()
    claim_checks = {
        "rct_improves_over_source_only": (
            spdim_contrasts["rct_minus_source_only_tsmnet"]["subject_weighted"]["ci_low"] > 0.0
        ),
        "spdim_geodesic_improves_over_rct": (
            spdim_contrasts["spdim_geodesic_minus_rct"]["subject_weighted"]["ci_low"] > 0.0
        ),
        "spdim_bias_improves_over_rct": (
            spdim_contrasts["spdim_bias_minus_rct"]["subject_weighted"]["ci_low"] > 0.0
        ),
    }
    if claim_checks != {
        "rct_improves_over_source_only": True,
        "spdim_geodesic_improves_over_rct": False,
        "spdim_bias_improves_over_rct": False,
    }:
        raise RuntimeError(f"accepted SPDIM claim gate changed: {claim_checks}")

    comparability = comparability_payload(
        manifest_rows,
        h2_validation,
        spdim_validation,
        h2_rows,
        spdim_rows,
    )
    write_comparability_markdown(comparability)
    comparability["artifacts"] = {
        "audit_md": {
            "path": str(COMPARABILITY_MD.relative_to(ROOT)),
            "sha256": sha256_file(COMPARABILITY_MD),
        }
    }
    write_json(COMPARABILITY_JSON, comparability)

    write_csv(CROSS_RESULTS_CSV, result_rows)
    write_results_markdown(result_rows)
    cross_results_payload = {
        "status": "pass",
        "label": "P10 standardized repaired-W1 cross-pipeline result table",
        "interval_label": INTERVAL_LABEL,
        "aggregation": {
            "source_seeds_averaged_first": True,
            "cluster": "dataset_x_target_subject",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "dataset_stratified": True,
            "paired_methods_within_subject": True,
        },
        "row_count": len(result_rows),
        "rows": result_rows,
        "p9_accepted_statistic_recompute": p9_recompute,
        "claim_boundary": {
            "adapter_only_comparison": False,
            "full_pipeline_same_split_comparison": True,
        },
        "artifacts": {
            "results_csv": {
                "path": str(CROSS_RESULTS_CSV.relative_to(ROOT)),
                "sha256": sha256_file(CROSS_RESULTS_CSV),
            },
            "results_md": {
                "path": str(CROSS_RESULTS_MD.relative_to(ROOT)),
                "sha256": sha256_file(CROSS_RESULTS_MD),
            },
        },
    }
    write_json(CROSS_RESULTS_JSON, cross_results_payload)

    write_csv(HARM_CSV, harm_rows)
    write_harm_markdown(harm_rows)

    write_freeze_markdown(h2_headlines, spdim_methods, spdim_contrasts)
    legacy_w1_split_quarantined = bool(
        legacy_quarantine.get("status") == "pass"
        and legacy_quarantine.get("prohibited_future_use")
        == "confirmatory_mi_aggregate_or_spdim_baseline"
        and legacy_quarantine.get("old_w1_confirmatory_status") is False
    )
    repaired_h2cmi_w1_confirmatory = bool(
        h2_summary.get("status") == "pass"
        and h2_summary.get("validation", {}).get("validation_pass") is True
        and h2_validation["target_seed_units"] == 345
    )
    spdim_source_seeds = sorted({row["source_seed"] for row in spdim_rows})
    official_spdim_complete = bool(
        spdim_summary.get("status") == "pass"
        and spdim_summary.get("validation", {}).get("validation_pass") is True
        and spdim_validation["selected_rows"] == 1380
        and spdim_source_seeds == SEEDS
    )
    official_spdim_result_is_seed0_only = spdim_source_seeds == [0]
    equivalence_analysis_present = False
    noninferiority_margin_preregistered = False
    additional_gpu_required = not (
        repaired_h2cmi_w1_confirmatory and official_spdim_complete
    )
    freeze_payload = {
        "status": "pass",
        "label": "P10 final repaired-W1 evidence freeze",
        "legacy_w1_split_quarantined": legacy_w1_split_quarantined,
        "repaired_h2cmi_w1_confirmatory": repaired_h2cmi_w1_confirmatory,
        "official_spdim_three_seed_same_split_baseline_complete": official_spdim_complete,
        "official_spdim_result_is_seed0_only": official_spdim_result_is_seed0_only,
        "rct_improves_over_source_only": claim_checks["rct_improves_over_source_only"],
        "spdim_geodesic_improves_over_rct": claim_checks["spdim_geodesic_improves_over_rct"],
        "spdim_bias_improves_over_rct": claim_checks["spdim_bias_improves_over_rct"],
        "equivalence_claim_supported": equivalence_analysis_present,
        "noninferiority_claim_supported": noninferiority_margin_preregistered,
        "adapter_only_h2cmi_vs_spdim_comparison_supported": comparability[
            "adapter_only_head_to_head_valid"
        ],
        "full_pipeline_same_split_comparison_supported": comparability[
            "full_pipeline_same_split_comparison_valid"
        ],
        "additional_gpu_required": additional_gpu_required,
        "verdict_derivation": {
            "legacy_quarantine_artifact_active": legacy_w1_split_quarantined,
            "h2cmi_summary_and_unit_gate_pass": repaired_h2cmi_w1_confirmatory,
            "spdim_summary_seed_and_row_gate_pass": official_spdim_complete,
            "spdim_source_seeds": spdim_source_seeds,
            "equivalence_analysis_present": equivalence_analysis_present,
            "noninferiority_margin_preregistered": noninferiority_margin_preregistered,
            "comparability_audit": {
                "adapter_only": comparability["adapter_only_head_to_head_valid"],
                "full_pipeline_same_split": comparability[
                    "full_pipeline_same_split_comparison_valid"
                ],
            },
        },
        "source_artifacts": {
            "h2cmi": {
                "result_commit": H2_RESULT_COMMIT,
                "result_path": str(H2_RESULTS.relative_to(ROOT)),
                "result_sha256": H2_RESULT_SHA256,
            },
            "spdim": {
                "result_commit": SPDIM_RESULT_COMMIT,
                "result_path": str(SPDIM_RESULTS.relative_to(ROOT)),
                "result_sha256": SPDIM_RESULT_SHA256,
            },
            "manifest": {
                "path": str(MANIFEST_CSV.relative_to(ROOT)),
                "manifest_hash": MANIFEST_HASH,
                "file_sha256": sha256_file(MANIFEST_CSV),
            },
        },
        "accepted_headline_numbers": {
            "h2cmi_repaired_split": h2_headlines,
            "official_spdim_repaired_split": {
                "subject_weighted_method_bacc": spdim_methods,
                "paired_contrasts": spdim_contrasts,
            },
        },
        "validation": {
            "p7_result_sha_matches": source_checks["h2cmi_result_sha256"] == H2_RESULT_SHA256,
            "p9_result_sha_matches": source_checks["spdim_result_sha256"] == SPDIM_RESULT_SHA256,
            "manifest_hash_matches": manifest_meta["manifest_hash"] == MANIFEST_HASH,
            "p7_target_seed_units": h2_validation["target_seed_units"],
            "p9_target_seed_method_rows": spdim_validation["selected_rows"],
            "p7_units_are_115_subjects_x_3_seeds": h2_validation["target_seed_units"] == 345,
            "p9_units_are_115_subjects_x_3_seeds_x_4_methods": spdim_validation["selected_rows"] == 1380,
            "legacy_split_rows_in_freeze": 0,
            "cross_pipeline_result_rows": len(result_rows),
            "harm_rows": len(harm_rows),
            "all_50_standardized_rows_generated_from_seed_averaged_units": len(result_rows) == 50,
            "all_120_harm_rows_generated_from_seed_averaged_units": len(harm_rows) == 120,
            "h2cmi_decomposition_headlines_recomputed": len(h2_headlines) == 4,
            "all_five_spdim_contrast_cis_recomputed": (
                p9_recompute["pass"]
                and len(p9_recompute["contrast_subject_weighted_checks"]) == 5
            ),
            "reported_aggregates_recompute": p9_recompute["pass"],
            "all_json_csv_source_files_parsed": True,
        },
        "artifacts": {
            "comparability_audit_md": {
                "path": str(COMPARABILITY_MD.relative_to(ROOT)),
                "sha256": sha256_file(COMPARABILITY_MD),
            },
            "comparability_audit_json": {
                "path": str(COMPARABILITY_JSON.relative_to(ROOT)),
                "sha256": sha256_file(COMPARABILITY_JSON),
            },
            "cross_pipeline_results_csv": {
                "path": str(CROSS_RESULTS_CSV.relative_to(ROOT)),
                "sha256": sha256_file(CROSS_RESULTS_CSV),
            },
            "cross_pipeline_results_md": {
                "path": str(CROSS_RESULTS_MD.relative_to(ROOT)),
                "sha256": sha256_file(CROSS_RESULTS_MD),
            },
            "cross_pipeline_results_json": {
                "path": str(CROSS_RESULTS_JSON.relative_to(ROOT)),
                "sha256": sha256_file(CROSS_RESULTS_JSON),
            },
            "harm_csv": {
                "path": str(HARM_CSV.relative_to(ROOT)),
                "sha256": sha256_file(HARM_CSV),
            },
            "harm_md": {
                "path": str(HARM_MD.relative_to(ROOT)),
                "sha256": sha256_file(HARM_MD),
            },
            "freeze_md": {
                "path": str(FREEZE_MD.relative_to(ROOT)),
                "sha256": sha256_file(FREEZE_MD),
            },
            "generator": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "independent_red_team_review": {
                "path": str(RED_TEAM_REVIEW.relative_to(ROOT)),
                "sha256": sha256_file(RED_TEAM_REVIEW),
            },
        },
    }
    write_json(FREEZE_JSON, freeze_payload)
    append_command_log()

    print(json.dumps({
        "status": "pass",
        "cross_pipeline_result_rows": len(result_rows),
        "harm_rows": len(harm_rows),
        "h2cmi_result_sha256": source_checks["h2cmi_result_sha256"],
        "spdim_result_sha256": source_checks["spdim_result_sha256"],
        "manifest_hash": MANIFEST_HASH,
        "p9_recompute_max_abs_difference": p9_recompute["max_abs_difference"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
