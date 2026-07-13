"""Independent completion and claim audit for the frozen P12 result packet."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12")
METHODS = (
    "source_only_tsmnet",
    "rct",
    "spdim_geodesic",
    "spdim_bias",
    "Joint-GEM",
    "FP-GEM",
)
BASELINES = METHODS[:-1]
DATASETS = ("BNCI2014_001", "Lee2019_MI")
METRICS = ("bacc", "acc")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260710
TOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float) -> bool:
    return abs(left - right) <= TOL


def independent_bootstrap(per_subject: list[dict[str, object]]):
    by_dataset: dict[str, dict[int, dict[str, dict[str, object]]]] = defaultdict(dict)
    for row in per_subject:
        by_dataset[str(row["dataset"])].setdefault(int(row["target_subject"]), {})[
            str(row["method"])
        ] = row

    arrays = {}
    for dataset in DATASETS:
        targets = sorted(by_dataset[dataset])
        arrays[dataset] = {
            metric: np.asarray(
                [
                    [float(by_dataset[dataset][target][method][metric]) for method in METHODS]
                    for target in targets
                ]
            )
            for metric in METRICS
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    method_rows = []
    contrast_rows = []
    for metric in METRICS:
        dataset_points = {dataset: arrays[dataset][metric].mean(axis=0) for dataset in DATASETS}
        counts = {dataset: len(arrays[dataset][metric]) for dataset in DATASETS}
        boot = {dataset: np.empty((BOOTSTRAP_REPLICATES, len(METHODS))) for dataset in DATASETS}
        for replicate in range(BOOTSTRAP_REPLICATES):
            for dataset in DATASETS:
                values = arrays[dataset][metric]
                indices = rng.integers(0, len(values), size=len(values))
                boot[dataset][replicate] = values[indices].mean(axis=0)

        for method_index, method in enumerate(METHODS):
            for dataset in DATASETS:
                values = boot[dataset][:, method_index]
                method_rows.append({
                    "metric": metric,
                    "estimand": "per_dataset",
                    "dataset": dataset,
                    "method": method,
                    "estimate": float(dataset_points[dataset][method_index]),
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                })
            subject_values = sum(
                counts[dataset] * boot[dataset][:, method_index] for dataset in DATASETS
            ) / sum(counts.values())
            subject_point = sum(
                counts[dataset] * dataset_points[dataset][method_index] for dataset in DATASETS
            ) / sum(counts.values())
            method_rows.append({
                "metric": metric,
                "estimand": "subject_weighted",
                "dataset": "ALL",
                "method": method,
                "estimate": float(subject_point),
                "ci_low": float(np.percentile(subject_values, 2.5)),
                "ci_high": float(np.percentile(subject_values, 97.5)),
            })
            macro_values = np.mean(
                [boot[dataset][:, method_index] for dataset in DATASETS], axis=0
            )
            macro_point = np.mean(
                [dataset_points[dataset][method_index] for dataset in DATASETS]
            )
            method_rows.append({
                "metric": metric,
                "estimand": "dataset_macro",
                "dataset": "ALL",
                "method": method,
                "estimate": float(macro_point),
                "ci_low": float(np.percentile(macro_values, 2.5)),
                "ci_high": float(np.percentile(macro_values, 97.5)),
            })

        fp_index = METHODS.index("FP-GEM")
        for estimand, dataset in (
            ("per_dataset", DATASETS[0]),
            ("per_dataset", DATASETS[1]),
            ("subject_weighted", "ALL"),
            ("dataset_macro", "ALL"),
        ):
            if estimand == "per_dataset":
                fp_values = boot[dataset][:, fp_index]
                fp_point = dataset_points[dataset][fp_index]
            elif estimand == "subject_weighted":
                fp_values = sum(
                    counts[item] * boot[item][:, fp_index] for item in DATASETS
                ) / sum(counts.values())
                fp_point = sum(
                    counts[item] * dataset_points[item][fp_index] for item in DATASETS
                ) / sum(counts.values())
            else:
                fp_values = np.mean([boot[item][:, fp_index] for item in DATASETS], axis=0)
                fp_point = np.mean([dataset_points[item][fp_index] for item in DATASETS])

            for baseline in BASELINES:
                baseline_index = METHODS.index(baseline)
                if estimand == "per_dataset":
                    baseline_values = boot[dataset][:, baseline_index]
                    baseline_point = dataset_points[dataset][baseline_index]
                elif estimand == "subject_weighted":
                    baseline_values = sum(
                        counts[item] * boot[item][:, baseline_index] for item in DATASETS
                    ) / sum(counts.values())
                    baseline_point = sum(
                        counts[item] * dataset_points[item][baseline_index] for item in DATASETS
                    ) / sum(counts.values())
                else:
                    baseline_values = np.mean(
                        [boot[item][:, baseline_index] for item in DATASETS], axis=0
                    )
                    baseline_point = np.mean(
                        [dataset_points[item][baseline_index] for item in DATASETS]
                    )
                differences = fp_values - baseline_values
                contrast_rows.append({
                    "metric": metric,
                    "comparison": f"FP-GEM minus {baseline}",
                    "estimand": estimand,
                    "dataset": dataset,
                    "estimate": float(fp_point - baseline_point),
                    "ci_low": float(np.percentile(differences, 2.5)),
                    "ci_high": float(np.percentile(differences, 97.5)),
                })
    return method_rows, contrast_rows


def indexed(rows: list[dict[str, object]], fields: tuple[str, ...]):
    return {tuple(str(row[field]) for field in fields): row for row in rows}


def main() -> int:
    result_path = HERE / "fp_gem_results.csv"
    result_rows = read_csv("fp_gem_results.csv")
    recorded_subject_rows = read_csv("fp_gem_per_subject.csv")
    recorded_contrasts = read_csv("fp_gem_contrast_ci.csv")
    artifact_rows = read_csv("fp_gem_job_artifact_manifest.csv")
    summary = json.loads((HERE / "fp_gem_summary.json").read_text())
    job_record = json.loads((HERE / "fp_gem_submission_record.json").read_text())

    checks: dict[str, bool] = {}
    keys = [
        (row["dataset"], row["target_subject"], row["source_seed"], row["method"])
        for row in result_rows
    ]
    unit_keys = {(row["dataset"], row["target_subject"], row["source_seed"]) for row in result_rows}
    checks["result_rows_1134"] = len(result_rows) == 1134
    checks["unique_result_keys"] = len(keys) == len(set(keys))
    checks["unit_count_189"] = len(unit_keys) == 189
    checks["source_seeds_exact"] = sorted({int(row["source_seed"]) for row in result_rows}) == [0, 1, 2]
    checks["method_set_exact"] = {row["method"] for row in result_rows} == set(METHODS)
    checks["dataset_subject_counts"] = Counter(
        dataset for dataset, _target, _seed in unit_keys
    ) == {"BNCI2014_001": 27, "Lee2019_MI": 162}
    checks["all_rows_ok"] = all(row["status"] == "ok" for row in result_rows)
    checks["hashes_complete"] = all(
        row["prediction_hash"] and row["logits_hash"] for row in result_rows
    )
    checks["no_target_label_leakage"] = all(
        row["target_label_leakage_detected"] == "False" for row in result_rows
    )
    checks["no_target_performance_selection"] = all(
        row["target_performance_selection_detected"] == "False" for row in result_rows
    )
    checks["backbone_and_classifier_frozen"] = all(
        row["backbone_frozen"] == "True" and row["classifier_frozen"] == "True"
        for row in result_rows
    )
    origin_counts = Counter(row["result_origin"] for row in result_rows)
    checks["result_origins_disclosed"] = origin_counts == {
        "P12_same_checkpoint_control": 756,
        "P12_new_method": 378,
    }
    source_counts = Counter(row["source_model_sha256"] for row in result_rows)
    checkpoint_counts = Counter(row["source_checkpoint_file_sha256"] for row in result_rows)
    checks["six_methods_share_each_source_state"] = (
        len(source_counts) == 189 and set(source_counts.values()) == {6}
    )
    checks["six_methods_share_each_checkpoint_file"] = (
        len(checkpoint_counts) == 189 and set(checkpoint_counts.values()) == {6}
    )
    checks["result_checksum_matches_summary"] = (
        sha256(result_path) == summary["artifacts"]["fp_gem_results.csv"]
    )

    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(row["dataset"], int(row["target_subject"]), row["method"])].append(row)
    recomputed_subject_rows = []
    seed_coverage = True
    for (dataset, target, method), rows in sorted(grouped.items()):
        seed_coverage &= sorted(int(row["source_seed"]) for row in rows) == [0, 1, 2]
        recomputed_subject_rows.append({
            "dataset": dataset,
            "target_subject": target,
            "method": method,
            "n_source_seeds": 3,
            "acc": float(np.mean([float(row["acc"]) for row in rows])),
            "bacc": float(np.mean([float(row["bacc"]) for row in rows])),
        })
    checks["seed_average_first"] = seed_coverage and len(recomputed_subject_rows) == 378
    recorded_subject_index = indexed(
        recorded_subject_rows, ("dataset", "target_subject", "method")
    )
    checks["per_subject_file_recomputes"] = all(
        close(row[metric], float(recorded_subject_index[
            (row["dataset"], str(row["target_subject"]), row["method"])
        ][metric]))
        for row in recomputed_subject_rows
        for metric in METRICS
    )

    method_rows, contrast_rows = independent_bootstrap(recomputed_subject_rows)
    summary_method_index = indexed(
        summary["method_summaries"], ("metric", "estimand", "dataset", "method")
    )
    checks["method_points_and_cis_recompute"] = all(
        all(close(row[field], float(summary_method_index[
            (row["metric"], row["estimand"], row["dataset"], row["method"])
        ][field])) for field in ("estimate", "ci_low", "ci_high"))
        for row in method_rows
    )
    recorded_contrast_index = indexed(
        recorded_contrasts, ("metric", "comparison", "estimand", "dataset")
    )
    checks["contrast_points_and_cis_recompute"] = all(
        all(close(row[field], float(recorded_contrast_index[
            (row["metric"], row["comparison"], row["estimand"], row["dataset"])
        ][field])) for field in ("estimate", "ci_low", "ci_high"))
        for row in contrast_rows
    )
    checks["bootstrap_policy_exact"] = (
        summary["aggregation"]["seed_average_first"]
        and summary["aggregation"]["cluster_unit"] == "dataset x target_subject"
        and summary["aggregation"]["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
        and summary["aggregation"]["bootstrap_seed"] == BOOTSTRAP_SEED
        and summary["aggregation"]["dataset_stratified"]
        and summary["aggregation"]["paired_methods_preserved"]
    )

    raw_payloads = [json.loads(path.read_text()) for path in sorted((RAW_ROOT / "units").glob("*.json"))]
    checks["raw_unit_count_189"] = len(raw_payloads) == 189
    checks["raw_split_and_leakage_gates"] = all(
        payload["status"] == "ok"
        and payload["adapt_eval_disjoint"]
        and payload["both_classes_adapt"]
        and payload["both_classes_eval"]
        and not payload["target_labels_passed_to_adaptation"]
        and not payload["target_performance_selection"]
        for payload in raw_payloads
    )
    checks["raw_exact_p9_config_retrain_disclosed"] = all(
        payload["source_checkpoint"]["source_reproduction_mode"]
        == "exact_p9_configuration_retrain"
        and not payload["source_checkpoint"]["p9_checkpoint_file_available"]
        for payload in raw_payloads
    )
    checks["raw_fp_prior_fixed"] = all(
        payload["geometry"]["fp_pi_fit"]
        == payload["geometry"]["source_empirical_prior"]
        for payload in raw_payloads
    )
    checks["raw_classifier_frozen"] = all(
        payload["rct"]["classifier_sha256_before_rct"]
        == payload["rct"]["classifier_sha256_after_rct"]
        for payload in raw_payloads
    )
    checks["raw_runner_config_manifest_frozen"] = all(
        payload["provenance"]["runner_sha256"]
        == "720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd"
        and payload["provenance"]["config_sha256"]
        == "d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165"
        and payload["manifest_hash"]
        == "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
        for payload in raw_payloads
    )
    checks["artifact_manifest_all_pass"] = (
        len(artifact_rows) == 189
        and all(
            row["status"] == "pass"
            and row["stdout_status"] == "exists_complete_clean_launch"
            and row["stderr_status"] in {"empty", "known_harmless_warnings_only"}
            for row in artifact_rows
        )
    )
    accepted_tasks = job_record["accepted_result_tasks"]
    checks["eight_result_tasks_complete"] = (
        len(accepted_tasks) == 8
        and sum(int(task["source_units"]) for task in accepted_tasks) == 189
        and sum(int(task["result_rows"]) for task in accepted_tasks) == 1134
        and all(task["stdout_status"] == "shard_complete" for task in accepted_tasks)
        and all(task["stderr_status"] == "empty" for task in accepted_tasks)
    )
    checks["excluded_attempts_zero_accepted_rows"] = all(
        int(item["accepted_rows"]) == 0 for item in job_record["excluded_launches"]
    )
    checks["final_squeue_absent"] = summary["final_squeue"]["all_absent"]

    contrast_index = indexed(
        contrast_rows, ("metric", "comparison", "estimand", "dataset")
    )
    def contrast(baseline: str, estimand: str = "subject_weighted"):
        return contrast_index[("bacc", f"FP-GEM minus {baseline}", estimand, "ALL")]

    source = contrast("source_only_tsmnet")
    rct = contrast("rct")
    geodesic = contrast("spdim_geodesic")
    bias = contrast("spdim_bias")
    joint = contrast("Joint-GEM")
    claim_gate = {
        "fp_gem_improves_over_source_only_supported": source["ci_low"] > 0,
        "fp_gem_improves_over_rct_supported": rct["ci_low"] > 0,
        "fp_gem_improves_over_spdim_geodesic_supported": geodesic["ci_low"] > 0,
        "fp_gem_improves_over_spdim_bias_supported": bias["ci_low"] > 0,
        "fp_gem_improves_over_joint_gem_subject_weighted_supported": joint["ci_low"] > 0,
        "equivalence_or_noninferiority_supported": False,
        "broad_benchmark_claim_supported": False,
        "direct_p9_checkpoint_reuse_claim_supported": False,
    }
    checks["claim_gate_expected"] = claim_gate == {
        "fp_gem_improves_over_source_only_supported": True,
        "fp_gem_improves_over_rct_supported": False,
        "fp_gem_improves_over_spdim_geodesic_supported": False,
        "fp_gem_improves_over_spdim_bias_supported": False,
        "fp_gem_improves_over_joint_gem_subject_weighted_supported": False,
        "equivalence_or_noninferiority_supported": False,
        "broad_benchmark_claim_supported": False,
        "direct_p9_checkpoint_reuse_claim_supported": False,
    }

    red_team_pass = all(checks.values())
    payload = {
        "status": "pass" if red_team_pass else "blocked",
        "red_team_pass": red_team_pass,
        "checks": checks,
        "claim_gate": claim_gate,
        "counts": {
            "result_rows": len(result_rows),
            "unit_count": len(unit_keys),
            "per_subject_rows": len(recomputed_subject_rows),
            "contrast_rows": len(contrast_rows),
            "artifact_manifest_rows": len(artifact_rows),
            "result_origin_counts": dict(origin_counts),
            "p9_reference_state_hash_match_count": sum(
                payload["source_checkpoint"]["p9_state_hash_matches_actual"]
                for payload in raw_payloads
            ),
        },
        "frozen_hashes": {
            "result_sha256": sha256(result_path),
            "runner_sha256": "720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd",
            "config_sha256": "d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165",
            "manifest_semantic_sha256": "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e",
        },
        "subject_weighted_bacc_contrasts": {
            baseline: contrast(baseline) for baseline in BASELINES
        },
        "adversarial_findings": [
            "P9 checkpoint files were unavailable; this packet uses exact-P9-configuration retrains, and all six methods share one persisted source checkpoint within each target-seed unit.",
            "The actual P12 source-state hashes do not match the P9 reference hashes, so direct P9 checkpoint or direct committed-row reuse must not be claimed.",
            "FP-GEM improves over source-only, but its subject-weighted contrast is negative against RCT and both SPDIM variants.",
            "FP-GEM minus Joint-GEM is small and its subject-weighted interval crosses zero; a general fixed-prior superiority claim is not supported.",
            "No equivalence, noninferiority, broad-benchmark, or target-tuned claim is supported.",
        ],
    }
    (HERE / "FP_GEM_FINAL_RED_TEAM.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# FP-GEM Final Red-Team Review",
        "",
        f"Status: `{'PASS' if red_team_pass else 'BLOCKED'}`.",
        "",
        "This review independently parsed the final CSV and raw unit payloads, rebuilt the seed-averaged subject table, reran the frozen 10,000-replicate paired cluster bootstrap, and compared every method estimate and contrast interval with the analyzer output.",
        "",
        "## Gate Results",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in checks.items())
    lines.extend([
        "",
        "## Adversarial Findings",
        "",
    ])
    lines.extend(f"- {finding}" for finding in payload["adversarial_findings"])
    lines.extend([
        "",
        "## Claim Gate",
        "",
    ])
    lines.extend(f"- {name}: `{value}`" for name, value in claim_gate.items())
    lines.extend([
        "",
        f"Final result SHA-256: `{payload['frozen_hashes']['result_sha256']}`.",
    ])
    (HERE / "FP_GEM_FINAL_RED_TEAM.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "checks": len(checks),
        "failed_checks": [name for name, value in checks.items() if not value],
    }, indent=2, sort_keys=True))
    return 0 if red_team_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
