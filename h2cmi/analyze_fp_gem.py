"""Merge and analyze the frozen P12 FP-GEM same-backbone experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from h2cmi import run_fp_gem as runner


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "source_only_tsmnet",
    "rct",
    "spdim_geodesic",
    "spdim_bias",
    "Joint-GEM",
    "FP-GEM",
)
CONTRAST_BASELINES = (
    "source_only_tsmnet",
    "rct",
    "spdim_geodesic",
    "spdim_bias",
    "Joint-GEM",
)
METRICS = ("bacc", "acc")
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260710


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def class_counts_text(value: Any) -> str:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return json.dumps([int(item) for item in parsed], separators=(",", ":"))


def accepted_stderr(path: Path) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    allowed = {
        "Trials demeaned and stacked with zero buffer to create continuous data -- edge effects present": "moabb_zero_buffer_edge_warning",
        "Matplotlib is building the font cache; this may take a moment.": "matplotlib_font_cache_notice",
    }
    unexpected = [line for line in lines if line not in allowed]
    forbidden = (
        "not compatible",
        "no kernel image",
        "cuda initialization",
        "unsupported gpu",
        "traceback",
        "runtimeerror",
    )
    failures = [line for line in lines if any(item in line.lower() for item in forbidden)]
    if not path.exists():
        status = "missing"
    elif not lines:
        status = "empty"
    elif not unexpected and not failures:
        status = "known_harmless_warnings_only"
    else:
        status = "real_or_unexpected_failure"
    return {
        "path": str(path),
        "status": status,
        "sha256": sha256_file(path) if path.exists() else "",
        "warning_patterns": sorted({allowed[line] for line in lines if line in allowed}),
        "unexpected_lines": unexpected[:20],
        "real_failure_lines": failures[:20],
    }


def accepted_stdout(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    provenance = payload["provenance"]
    required = (
        f"launch_commit={provenance['launch_commit']}",
        f"runner_sha256={provenance['runner_sha256']}",
        f"config_sha256={provenance['config_sha256']}",
        "repo_status_porcelain_begin\nrepo_status_porcelain_end",
        '"status": "ok"',
    )
    missing = [item for item in required if item not in text]
    return {
        "path": str(path),
        "status": "exists_complete_clean_launch" if path.exists() and not missing else "missing_or_incomplete",
        "sha256": sha256_file(path) if path.exists() else "",
        "missing_required_fragments": missing,
    }


def squeue_absence(job_ids: list[str]) -> dict[str, Any]:
    command = ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T|%P|%R"]
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"squeue validation failed: {proc.stderr.strip()}")
    return {
        "command": " ".join(command),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "all_absent": not proc.stdout.strip(),
    }


def load_raw_units(config, raw_root: Path, units: list[dict[str, str]]):
    payloads = []
    artifact_rows = []
    failures = []
    for unit in units:
        dataset = unit["dataset"]
        target = int(unit["target_subject"])
        seed = int(unit["source_seed"])
        path = raw_root / f"units/{dataset}_target{target}_seed{seed}.json"
        if not path.exists():
            failures.append(f"missing raw unit {path}")
            continue
        payload = json.loads(path.read_text())
        key = (dataset, target, seed)
        observed = (payload.get("dataset"), int(payload.get("target_subject", -1)), int(payload.get("source_seed", -1)))
        if payload.get("status") != "ok" or observed != key:
            failures.append(f"failed or mismatched unit {key}: {payload.get('failure_reason', payload.get('status'))}")
            continue
        results = payload.get("results", [])
        if {row["method"] for row in results} != set(runner.NEW_METHODS) or len(results) != 2:
            failures.append(f"new-method coverage mismatch {key}")
            continue
        checks = {
            "manifest_hash": payload["manifest_hash"] == runner.MANIFEST_HASH,
            "split_hash": payload["split_hash"] == unit["split_hash"],
            "adapt_eval_disjoint": payload["adapt_eval_disjoint"],
            "both_classes_adapt": payload["both_classes_adapt"],
            "both_classes_eval": payload["both_classes_eval"],
            "no_target_labels": not payload["target_labels_passed_to_adaptation"],
            "no_target_selection": not payload["target_performance_selection"],
            "source_state_expected": payload["source_checkpoint"]["source_model_sha256_expected"] == unit["expected_source_model_sha256"],
            "source_state_exact": payload["source_checkpoint"]["source_model_sha256_actual"] == unit["expected_source_model_sha256"],
            "source_checkpoint_file": sha256_file(payload["source_checkpoint"]["source_checkpoint_path"]) == payload["source_checkpoint"]["source_checkpoint_file_sha256"],
            "feature_dimension": payload["feature_hook"]["dimension"] == config["feature_space"]["dimension"],
            "feature_semantics": max(
                payload["feature_hook"]["source_semantic_max_abs_error"],
                payload["feature_hook"]["adapt_semantic_max_abs_error"],
                payload["feature_hook"]["eval_semantic_max_abs_error"],
            ) <= 1e-7,
            "classifier_frozen_rct": payload["rct"]["classifier_sha256_before_rct"] == payload["rct"]["classifier_sha256_after_rct"],
            "parameters_frozen_rct": payload["rct"]["parameters_sha256_before_rct"] == payload["rct"]["parameters_sha256_after_rct"],
            "fp_prior_fixed": payload["geometry"]["fp_pi_fit"] == payload["geometry"]["source_empirical_prior"],
            "clean_launch": payload["provenance"]["clean_worktree"] and payload["provenance"]["git_status_porcelain"] == "",
            "runner_hash": payload["provenance"]["runner_sha256"] == sha256_file(runner.__file__),
            "config_hash": payload["provenance"]["config_sha256"] == runner.FROZEN_CONFIG_SHA256,
            "no_pretrained": not payload["provenance"]["official_pretrained_weight_used"],
            "no_vendoring": not payload["provenance"]["third_party_vendored"],
        }
        if not all(checks.values()):
            failures.append(f"unit validation failed {key}: {[name for name, value in checks.items() if not value]}")
            continue
        array_job = payload["provenance"]["slurm_array_job_id"]
        array_task = payload["provenance"]["slurm_array_task_id"]
        stdout_path = raw_root / f"logs/{array_job}_{array_task}.out"
        stderr_path = raw_root / f"logs/{array_job}_{array_task}.err"
        stdout = accepted_stdout(stdout_path, payload)
        stderr = accepted_stderr(stderr_path)
        if stdout["status"] != "exists_complete_clean_launch" or stderr["status"] not in {"empty", "known_harmless_warnings_only"}:
            failures.append(f"stdout/stderr gate failed {key}: {stdout['status']} / {stderr['status']}")
            continue
        artifact_rows.append({
            "dataset": dataset,
            "target_subject": target,
            "source_seed": seed,
            "hardware_group": unit["hardware_group"],
            "slurm_array_job_id": array_job,
            "slurm_array_task_id": array_task,
            "raw_path": str(path),
            "raw_sha256": sha256_file(path),
            "stdout_path": str(stdout_path),
            "stdout_sha256": stdout["sha256"],
            "stdout_status": stdout["status"],
            "stderr_path": str(stderr_path),
            "stderr_sha256": stderr["sha256"],
            "stderr_status": stderr["status"],
            "source_model_sha256": unit["expected_source_model_sha256"],
            "source_checkpoint_file_sha256": payload["source_checkpoint"]["source_checkpoint_file_sha256"],
            "status": "pass",
        })
        payloads.append(payload)
    if failures:
        raise RuntimeError("raw FP-GEM completion gate failed:\n" + "\n".join(failures[:50]))
    if len(payloads) != 189:
        raise RuntimeError(f"expected 189 accepted unit payloads, found {len(payloads)}")
    return payloads, artifact_rows


def flatten_new_rows(payloads):
    rows = []
    source_files = {}
    for payload in payloads:
        key = (payload["dataset"], payload["target_subject"], payload["source_seed"])
        source_files[key] = payload["source_checkpoint"]["source_checkpoint_file_sha256"]
        for result in payload["results"]:
            rows.append({
                "dataset": result["dataset"],
                "target_subject": int(result["target_subject"]),
                "source_seed": int(result["source_seed"]),
                "split_family": result["split_family"],
                "method": result["method"],
                "n_adapt": int(result["n_adapt"]),
                "n_eval": int(result["n_eval"]),
                "class_counts_adapt": class_counts_text(result["class_counts_adapt"]),
                "class_counts_eval": class_counts_text(result["class_counts_eval"]),
                "acc": float(result["acc"]),
                "bacc": float(result["bacc"]),
                "macro_f1": float(result["macro_f1"]),
                "prediction_hash": result["prediction_hash"],
                "logits_hash": result["logits_hash"],
                "status": result["status"],
                "failure_reason": result["failure_reason"],
                "source_model_sha256": result["source_model_sha256"],
                "source_checkpoint_file_sha256": result["source_checkpoint_file_sha256"],
                "result_origin": "P12_new_method",
                "target_label_leakage_detected": False,
                "target_performance_selection_detected": False,
                "classifier_frozen": True,
                "backbone_frozen": True,
            })
    return rows, source_files


def load_reused_p9(config, source_files, units_by_key):
    path = ROOT / config["p9_pipeline"]["results_path"]
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], int(row["target_subject"]), int(row["source_seed"]))
            if key not in units_by_key:
                continue
            if row["method"] not in runner.P9_METHODS:
                continue
            unit = units_by_key[key]
            if row["source_model_sha256"] != unit["expected_source_model_sha256"]:
                raise RuntimeError(f"P9 source state mismatch at {key}")
            if row["split_hash"] != unit["split_hash"]:
                raise RuntimeError(f"P9 split hash mismatch at {key}")
            if int(row["n_adapt"]) != int(unit["n_adapt"]) or int(row["n_eval"]) != int(unit["n_eval"]):
                raise RuntimeError(f"P9 split count mismatch at {key}")
            rows.append({
                "dataset": row["dataset"],
                "target_subject": int(row["target_subject"]),
                "source_seed": int(row["source_seed"]),
                "split_family": row["split_family"],
                "method": row["method"],
                "n_adapt": int(row["n_adapt"]),
                "n_eval": int(row["n_eval"]),
                "class_counts_adapt": class_counts_text(row["class_counts_adapt"]),
                "class_counts_eval": class_counts_text(row["class_counts_eval"]),
                "acc": float(row["acc"]),
                "bacc": float(row["bacc"]),
                "macro_f1": float(row["macro_f1"]),
                "prediction_hash": row["prediction_hash"],
                "logits_hash": row["logits_hash"],
                "status": row["status"],
                "failure_reason": row["failure_reason"],
                "source_model_sha256": row["source_model_sha256"],
                "source_checkpoint_file_sha256": source_files[key],
                "result_origin": "P9_committed_reuse",
                "target_label_leakage_detected": row["target_label_leakage_detected"].lower() == "true",
                "target_performance_selection_detected": row["method_selection_uses_target_performance"].lower() == "true",
                "classifier_frozen": True,
                "backbone_frozen": True,
            })
    if len(rows) != 756:
        raise RuntimeError(f"expected 756 reused P9 rows, found {len(rows)}")
    return rows


def seed_average(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["target_subject"], row["method"])].append(row)
    out = []
    for (dataset, target, method), group in sorted(grouped.items()):
        seeds = sorted(row["source_seed"] for row in group)
        if seeds != [0, 1, 2]:
            raise RuntimeError(f"seed coverage mismatch {(dataset, target, method)}: {seeds}")
        out.append({
            "dataset": dataset,
            "target_subject": target,
            "method": method,
            "n_source_seeds": 3,
            "acc": float(np.mean([row["acc"] for row in group])),
            "bacc": float(np.mean([row["bacc"] for row in group])),
        })
    if len(out) != 378:
        raise RuntimeError(f"expected 378 seed-averaged rows, found {len(out)}")
    return out


def bootstrap(per_subject):
    by_dataset = defaultdict(dict)
    for row in per_subject:
        by_dataset[row["dataset"]].setdefault(row["target_subject"], {})[row["method"]] = row
    datasets = list(runner.SELECTED_DATASETS)
    for dataset in datasets:
        for target, methods in by_dataset[dataset].items():
            if set(methods) != set(METHODS):
                raise RuntimeError(f"paired method coverage mismatch {(dataset, target)}")
    points = {}
    bootstrap_values = defaultdict(list)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arrays = {}
    for dataset in datasets:
        targets = sorted(by_dataset[dataset])
        arrays[dataset] = {
            metric: np.asarray([[by_dataset[dataset][target][method][metric] for method in METHODS] for target in targets])
            for metric in METRICS
        }
    for metric in METRICS:
        dataset_means = {dataset: arrays[dataset][metric].mean(0) for dataset in datasets}
        counts = {dataset: arrays[dataset][metric].shape[0] for dataset in datasets}
        for mi, method in enumerate(METHODS):
            for dataset in datasets:
                points[(metric, "per_dataset", dataset, method)] = float(dataset_means[dataset][mi])
            points[(metric, "subject_weighted", "ALL", method)] = float(
                sum(counts[ds] * dataset_means[ds][mi] for ds in datasets) / sum(counts.values())
            )
            points[(metric, "dataset_macro", "ALL", method)] = float(
                np.mean([dataset_means[ds][mi] for ds in datasets])
            )
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = {}
            for dataset in datasets:
                values = arrays[dataset][metric]
                indices = rng.integers(0, len(values), size=len(values))
                sampled[dataset] = values[indices].mean(0)
            for mi, method in enumerate(METHODS):
                for dataset in datasets:
                    bootstrap_values[(metric, "per_dataset", dataset, method)].append(sampled[dataset][mi])
                bootstrap_values[(metric, "subject_weighted", "ALL", method)].append(
                    sum(counts[ds] * sampled[ds][mi] for ds in datasets) / sum(counts.values())
                )
                bootstrap_values[(metric, "dataset_macro", "ALL", method)].append(
                    np.mean([sampled[ds][mi] for ds in datasets])
                )
    method_rows = []
    for key, point in sorted(points.items()):
        values = np.asarray(bootstrap_values[key])
        metric, estimand, dataset, method = key
        method_rows.append({
            "metric": metric,
            "estimand": estimand,
            "dataset": dataset,
            "method": method,
            "estimate": point,
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "n_subjects": len(by_dataset[dataset]) if estimand == "per_dataset" else sum(len(by_dataset[ds]) for ds in datasets),
        })
    contrast_rows = []
    for metric in METRICS:
        for estimand, dataset in (("per_dataset", datasets[0]), ("per_dataset", datasets[1]), ("subject_weighted", "ALL"), ("dataset_macro", "ALL")):
            fp_key = (metric, estimand, dataset, "FP-GEM")
            for baseline in CONTRAST_BASELINES:
                base_key = (metric, estimand, dataset, baseline)
                values = np.asarray(bootstrap_values[fp_key]) - np.asarray(bootstrap_values[base_key])
                contrast_rows.append({
                    "metric": metric,
                    "comparison": f"FP-GEM minus {baseline}",
                    "estimand": estimand,
                    "dataset": dataset,
                    "n_subjects": len(by_dataset[dataset]) if estimand == "per_dataset" else 63,
                    "estimate": points[fp_key] - points[base_key],
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "cluster_unit": "dataset x target_subject",
                })
    return method_rows, contrast_rows


def format_ci(row) -> str:
    return f"{row['estimate']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def write_head_to_head(path: Path, method_rows, contrast_rows, result_sha):
    lookup = {(row["metric"], row["estimand"], row["dataset"], row["method"]): row for row in method_rows}
    contrasts = {(row["metric"], row["estimand"], row["dataset"], row["comparison"]): row for row in contrast_rows}
    lines = [
        "# FP-GEM Same-Backbone Head-to-Head",
        "",
        "Status: frozen P12 two-dataset, three-source-seed comparison. This is not a broad benchmark.",
        "",
        "## Mean Balanced Accuracy",
        "",
        "| method | BNCI2014_001 | Lee2019_MI | subject-weighted | dataset-macro |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for estimand, dataset in (("per_dataset", "BNCI2014_001"), ("per_dataset", "Lee2019_MI"), ("subject_weighted", "ALL"), ("dataset_macro", "ALL")):
            row = lookup[("bacc", estimand, dataset, method)]
            cells.append(f"{row['estimate']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Primary Paired Balanced-Accuracy Contrasts",
        "",
        "| comparison | BNCI2014_001 | Lee2019_MI | subject-weighted | dataset-macro |",
        "|---|---:|---:|---:|---:|",
    ])
    for baseline in CONTRAST_BASELINES:
        comparison = f"FP-GEM minus {baseline}"
        cells = []
        for estimand, dataset in (("per_dataset", "BNCI2014_001"), ("per_dataset", "Lee2019_MI"), ("subject_weighted", "ALL"), ("dataset_macro", "ALL")):
            cells.append(format_ci(contrasts[("bacc", estimand, dataset, comparison)]))
        lines.append(f"| {comparison} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Accuracy summaries and contrasts are included in the machine-readable packet. Source seeds are averaged within subject/method before every aggregate. Intervals are 10,000-replicate dataset-stratified paired cluster-bootstrap intervals with seed 20260710.",
        "",
        "No equivalence, noninferiority, broad-benchmark, target-selection, or third-dataset claim is permitted. Interpret estimates only under the precommitted grid in `FP_GEM_METHOD_FREEZE.md`.",
        "",
        f"Final result CSV SHA-256: `{result_sha}`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_execution_audit(path, validation, artifact_rows, job_record, queue):
    status_counts = Counter(row["stderr_status"] for row in artifact_rows)
    stdout_counts = Counter(row["stdout_status"] for row in artifact_rows)
    lines = [
        "# FP-GEM Execution Audit",
        "",
        f"- status: `{'PASS' if validation['validation_pass'] else 'BLOCKED'}`",
        f"- accepted target-seed tasks: `{validation['accepted_unit_count']}`",
        f"- new method rows: `{validation['new_row_count']}`",
        f"- reused P9 rows: `{validation['reused_p9_row_count']}`",
        f"- final rows: `{validation['final_row_count']}`",
        f"- final squeue absence: `{queue['all_absent']}`",
        f"- stdout statuses: `{dict(stdout_counts)}`",
        f"- stderr statuses: `{dict(status_counts)}`",
        "",
        "## Accepted Jobs",
        "",
        f"- smoke: `{job_record['smoke']}`",
    ]
    for array in job_record["arrays"]:
        lines.append(f"- array: `{array}`")
    lines.extend([
        "",
        "Completion uses only `squeue` absence plus stdout/stderr and artifact parse/count/checksum validation. No Slurm accounting command is used.",
        "",
        "## Scientific Gates",
        "",
    ])
    for key, value in validation.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12")
    ap.add_argument("--out-dir", default="h2cmi/results/fp_gem_main")
    ap.add_argument("--job-record", default="/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p12/submission_record.json")
    args = ap.parse_args()
    config = runner.load_config()
    runner.validate_frozen_inputs(config)
    raw_root = Path(args.raw_root)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    units = runner.load_units()
    job_record = json.loads(Path(args.job_record).read_text())
    job_ids = [str(job_record["smoke"]["job_id"])] + [str(item["job_id"]) for item in job_record["arrays"]]
    queue = squeue_absence(job_ids)
    if not queue["all_absent"]:
        raise RuntimeError(f"accepted P12 jobs remain in squeue: {queue['stdout']}")
    payloads, artifact_rows = load_raw_units(config, raw_root, units)
    new_rows, source_files = flatten_new_rows(payloads)
    units_by_key = {
        (row["dataset"], int(row["target_subject"]), int(row["source_seed"])): row
        for row in units
    }
    reused_rows = load_reused_p9(config, source_files, units_by_key)
    final_rows = sorted(
        reused_rows + new_rows,
        key=lambda row: (runner.SELECTED_DATASETS.index(row["dataset"]), row["target_subject"], row["source_seed"], METHODS.index(row["method"])),
    )
    keys = [(row["dataset"], row["target_subject"], row["source_seed"], row["method"]) for row in final_rows]
    validation = {
        "accepted_unit_count": len(payloads),
        "new_row_count": len(new_rows),
        "reused_p9_row_count": len(reused_rows),
        "final_row_count": len(final_rows),
        "expected_new_rows": 378,
        "expected_reused_p9_rows": 756,
        "expected_final_rows": 1134,
        "duplicate_keys": len(keys) - len(set(keys)),
        "all_rows_ok": all(row["status"] == "ok" for row in final_rows),
        "prediction_hashes_complete": all(row["prediction_hash"] for row in final_rows),
        "logits_hashes_complete": all(row["logits_hash"] for row in final_rows),
        "all_adapt_eval_disjoint": all(payload["adapt_eval_disjoint"] for payload in payloads),
        "all_eval_both_classes": all(payload["both_classes_eval"] for payload in payloads),
        "target_label_leakage_detected": any(payload["target_labels_passed_to_adaptation"] for payload in payloads),
        "target_performance_selection_detected": any(payload["target_performance_selection"] for payload in payloads),
        "all_source_states_match_p9": all(
            payload["source_checkpoint"]["source_model_sha256_actual"] == payload["source_checkpoint"]["source_model_sha256_expected"]
            for payload in payloads
        ),
        "all_classifiers_frozen": all(
            payload["rct"]["classifier_sha256_before_rct"] == payload["rct"]["classifier_sha256_after_rct"]
            for payload in payloads
        ),
        "manifest_hash": runner.MANIFEST_HASH,
        "source_seeds": sorted({row["source_seed"] for row in final_rows}),
        "squeue_absence": queue["all_absent"],
        "stdout_validation_pass": all(row["stdout_status"] == "exists_complete_clean_launch" for row in artifact_rows),
        "stderr_validation_pass": all(row["stderr_status"] in {"empty", "known_harmless_warnings_only"} for row in artifact_rows),
    }
    validation["target_label_leakage_detected"] = validation["target_label_leakage_detected"] or any(
        row["target_label_leakage_detected"] for row in final_rows
    )
    validation["target_performance_selection_detected"] = validation["target_performance_selection_detected"] or any(
        row["target_performance_selection_detected"] for row in final_rows
    )
    validation["validation_pass"] = (
        validation["accepted_unit_count"] == 189
        and validation["new_row_count"] == 378
        and validation["reused_p9_row_count"] == 756
        and validation["final_row_count"] == 1134
        and validation["duplicate_keys"] == 0
        and validation["all_rows_ok"]
        and validation["prediction_hashes_complete"]
        and validation["logits_hashes_complete"]
        and validation["all_adapt_eval_disjoint"]
        and validation["all_eval_both_classes"]
        and not validation["target_label_leakage_detected"]
        and not validation["target_performance_selection_detected"]
        and validation["all_source_states_match_p9"]
        and validation["all_classifiers_frozen"]
        and validation["source_seeds"] == [0, 1, 2]
        and validation["squeue_absence"]
        and validation["stdout_validation_pass"]
        and validation["stderr_validation_pass"]
    )
    if not validation["validation_pass"]:
        raise RuntimeError(f"P12 final validation failed: {validation}")
    result_fields = [
        "dataset", "target_subject", "source_seed", "split_family", "method",
        "n_adapt", "n_eval", "class_counts_adapt", "class_counts_eval",
        "acc", "bacc", "macro_f1", "prediction_hash", "logits_hash", "status",
        "failure_reason", "source_model_sha256", "source_checkpoint_file_sha256",
        "result_origin", "target_label_leakage_detected",
        "target_performance_selection_detected", "classifier_frozen", "backbone_frozen",
    ]
    results_path = out / "fp_gem_results.csv"
    write_csv(results_path, result_fields, final_rows)
    result_sha = sha256_file(results_path)
    per_subject = seed_average(final_rows)
    per_subject_path = out / "fp_gem_per_subject.csv"
    write_csv(per_subject_path, ["dataset", "target_subject", "method", "n_source_seeds", "acc", "bacc"], per_subject)
    method_rows, contrast_rows = bootstrap(per_subject)
    contrast_path = out / "fp_gem_contrast_ci.csv"
    write_csv(contrast_path, list(contrast_rows[0]), contrast_rows)
    manifest_path = out / "fp_gem_job_artifact_manifest.csv"
    write_csv(manifest_path, list(artifact_rows[0]), artifact_rows)
    head_path = out / "fp_gem_head_to_head.md"
    write_head_to_head(head_path, method_rows, contrast_rows, result_sha)
    audit_path = out / "fp_gem_execution_audit.md"
    write_execution_audit(audit_path, validation, artifact_rows, job_record, queue)
    summary = {
        "status": "pass",
        "label": "P12 Fixed-Prior Geometry EM same-backbone head-to-head",
        "config_sha256": runner.FROZEN_CONFIG_SHA256,
        "runner_sha256": sha256_file(runner.__file__),
        "analyzer_sha256": sha256_file(__file__),
        "manifest_hash": runner.MANIFEST_HASH,
        "p9_results_sha256": runner.P9_RESULTS_SHA256,
        "datasets": list(runner.SELECTED_DATASETS),
        "source_seeds": list(runner.SELECTED_SEEDS),
        "methods": list(METHODS),
        "aggregation": {
            "seed_average_first": True,
            "cluster_unit": "dataset x target_subject",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "dataset_stratified": True,
            "paired_methods_preserved": True,
        },
        "method_summaries": method_rows,
        "contrasts": contrast_rows,
        "validation": validation,
        "artifacts": {
            "fp_gem_results.csv": result_sha,
            "fp_gem_per_subject.csv": sha256_file(per_subject_path),
            "fp_gem_contrast_ci.csv": sha256_file(contrast_path),
            "fp_gem_head_to_head.md": sha256_file(head_path),
            "fp_gem_execution_audit.md": sha256_file(audit_path),
            "fp_gem_job_artifact_manifest.csv": sha256_file(manifest_path),
        },
        "job_record": job_record,
        "final_squeue": queue,
    }
    write_json(out / "fp_gem_summary.json", summary)
    print(json.dumps({
        "status": "pass",
        "result_rows": len(final_rows),
        "result_sha256": result_sha,
        "contrast_rows": len(contrast_rows),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
