"""Validate and analyze the frozen P13 fixed-reservoir prevalence stress."""
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

from h2cmi import run_fp_gem_prevalence as runner


ROOT = Path(__file__).resolve().parents[1]
METHODS = runner.METHODS
Q_VALUES = runner.Q_VALUES
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_710
COMPARATORS = ("Joint-GEM", "rct", "spdim_geodesic", "spdim_bias")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def compact_counts(value: Any) -> str:
    return json.dumps([int(item) for item in value], separators=(",", ":"))


def accepted_stderr(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.exists() else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    harmless_exact = {
        "Trials demeaned and stacked with zero buffer to create continuous data -- edge effects present",
        "Matplotlib is building the font cache; this may take a moment.",
    }
    unexpected = [line for line in lines if line not in harmless_exact]
    forbidden = (
        "traceback",
        "runtimeerror",
        "not compatible",
        "no kernel image",
        "cuda initialization",
        "unsupported gpu",
        "out of memory",
    )
    failures = [line for line in lines if any(token in line.lower() for token in forbidden)]
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
        "sha256": sha256_file(path) if path.exists() else "",
        "status": status,
        "unexpected_lines": unexpected[:20],
        "real_failure_lines": failures[:20],
    }


def accepted_stdout(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.exists() else ""
    provenance = payload["provenance"]
    required = (
        f"launch_commit={provenance['launch_commit']}",
        f"runner_sha256={provenance['runner_sha256']}",
        f"config_sha256={provenance['config_sha256']}",
        "repo_status_porcelain_begin\nrepo_status_porcelain_end",
        '"status": "ok"',
    )
    missing = [fragment for fragment in required if fragment not in text]
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
        "status": "exists_complete_clean_launch" if path.exists() and not missing else "missing_or_incomplete",
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


def load_raw_units(
    config: dict[str, Any],
    units: dict[tuple[int, int], dict[str, Any]],
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_runner_hash: str | None = None
    expected_launch_commit: str | None = None
    for target, seed in sorted(units):
        unit = units[(target, seed)]
        path = raw_root / f"units/{runner.DATASET}_target{target}_seed{seed}.json"
        if not path.exists():
            failures.append(f"missing raw unit: {path}")
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            failures.append(f"unparseable raw unit {path}: {exc}")
            continue
        key = (payload.get("dataset"), payload.get("target_subject"), payload.get("source_seed"))
        if payload.get("status") != "ok" or key != (runner.DATASET, target, seed):
            failures.append(f"failed/mismatched unit {(target, seed)}: {payload.get('failure_reason', key)}")
            continue
        rows = payload.get("results", [])
        geometry = payload.get("geometry", [])
        result_keys = {(float(row["q"]), row["method"]) for row in rows}
        geometry_keys = {(float(row["q"]), row["method"]) for row in geometry}
        expected_result_keys = {(float(q), method) for q in Q_VALUES for method in METHODS}
        expected_geometry_keys = {
            (float(q), method) for q in Q_VALUES for method in ("Joint-GEM", "FP-GEM")
        }
        provenance = payload.get("provenance", {})
        checks = {
            "result_coverage": len(rows) == 18 and result_keys == expected_result_keys,
            "geometry_coverage": len(geometry) == 6 and geometry_keys == expected_geometry_keys,
            "checkpoint_hash": payload["checkpoint"]["checkpoint_sha256"] == unit["checkpoint_sha256"],
            "source_state_hash": payload["checkpoint"]["source_model_sha256"] == unit["source_model_sha256"],
            "checkpoint_file_unchanged": sha256_file(unit["checkpoint_path"]) == unit["checkpoint_sha256"],
            "p12_raw_unchanged": sha256_file(unit["p12_raw_path"]) == unit["p12_raw_sha256"],
            "split_hash": payload["split"]["p12_split_hash"] == unit["p12_split_hash"],
            "adapt_reservoir": payload["split"]["adapt_reservoir_trial_ids"] == unit["adapt_reservoir_trial_ids"],
            "evaluation_reservoir": payload["split"]["eval_trial_ids"] == unit["eval_trial_ids"],
            "adapt_eval_disjoint": payload["split"]["adapt_eval_disjoint"],
            "balanced_eval": payload["split"]["class_counts_eval"] == [25, 25],
            "q05_center": payload["q05_center_reproduction"]["p12_center_reuse_approved"],
            "q05_all_hashes": payload["q05_center_reproduction"]["all_six_prediction_and_logits_hashes_match"],
            "q05_geometry": payload["q05_center_reproduction"]["both_gem_geometry_hashes_match"],
            "clean_launch": provenance.get("clean_worktree") is True,
            "no_fresh_training": provenance.get("fresh_source_training_performed") is False,
            "no_target_labels": provenance.get("target_labels_passed_to_adaptation") is False,
            "no_q_input": provenance.get("q_passed_to_method") is False,
            "no_target_selection": provenance.get("target_performance_selection") is False,
            "no_pretrained": provenance.get("official_pretrained_weight_used") is False,
            "no_vendoring": provenance.get("third_party_vendored") is False,
            "config_hash": provenance.get("config_sha256") == runner.P13_CONFIG_SHA256,
            "manifest_hash": provenance.get("manifest_sha256") == runner.P13_MANIFEST_SHA256,
            "hardware_group": provenance.get("hardware_group") == unit["hardware_group"],
        }
        if not all(checks.values()):
            failures.append(
                f"unit gate failed {(target, seed)}: {[name for name, passed in checks.items() if not passed]}"
            )
            continue
        if expected_runner_hash is None:
            expected_runner_hash = provenance["runner_sha256"]
            expected_launch_commit = provenance["launch_commit"]
        if provenance["runner_sha256"] != expected_runner_hash:
            failures.append(f"mixed runner hashes at {(target, seed)}")
            continue
        if provenance["launch_commit"] != expected_launch_commit:
            failures.append(f"mixed launch commits at {(target, seed)}")
            continue
        array_job = str(provenance["slurm_array_job_id"])
        array_task = str(provenance["slurm_array_task_id"])
        stdout_path = raw_root / f"logs/{array_job}_{array_task}.out"
        stderr_path = raw_root / f"logs/{array_job}_{array_task}.err"
        stdout = accepted_stdout(stdout_path, payload)
        stderr = accepted_stderr(stderr_path)
        if stdout["status"] != "exists_complete_clean_launch":
            failures.append(f"stdout gate failed {(target, seed)}: {stdout}")
            continue
        if stderr["status"] not in {"empty", "known_harmless_warnings_only"}:
            failures.append(f"stderr gate failed {(target, seed)}: {stderr}")
            continue
        artifacts.append({
            "dataset": runner.DATASET,
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
            "launch_commit": provenance["launch_commit"],
            "runner_sha256": provenance["runner_sha256"],
            "checkpoint_sha256": unit["checkpoint_sha256"],
            "status": "pass",
        })
        payloads.append(payload)
    if failures:
        raise RuntimeError("P13 raw-unit gate failed:\n" + "\n".join(failures[:100]))
    if len(payloads) != 162:
        raise RuntimeError(f"expected 162 accepted units, found {len(payloads)}")
    return payloads, artifacts


def flatten(payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int, float, str], list[int]]]:
    results: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    prediction_vectors: dict[tuple[int, int, float, str], list[int]] = {}
    for payload in payloads:
        for row in payload["results"]:
            key = (int(row["target_subject"]), int(row["source_seed"]), float(row["q"]), row["method"])
            if key in prediction_vectors:
                raise RuntimeError(f"duplicate prediction-vector key: {key}")
            prediction_vectors[key] = [int(value) for value in row["prediction_vector"]]
            results.append({
                "dataset": row["dataset"],
                "target_subject": int(row["target_subject"]),
                "source_seed": int(row["source_seed"]),
                "q": float(row["q"]),
                "method": row["method"],
                "checkpoint_hash": row["checkpoint_hash"],
                "adaptation_manifest_hash": row["adaptation_manifest_hash"],
                "n_adapt": int(row["n_adapt"]),
                "class_counts_adapt": compact_counts(row["class_counts_adapt"]),
                "n_eval": int(row["n_eval"]),
                "class_counts_eval": compact_counts(row["class_counts_eval"]),
                "acc": float(row["acc"]),
                "bacc": float(row["bacc"]),
                "macro_f1": float(row["macro_f1"]),
                "prediction_hash": row["prediction_hash"],
                "logits_hash": row["logits_hash"],
                "status": row["status"],
                "failure_reason": row["failure_reason"],
                "result_origin": row["result_origin"],
                "target_labels_passed_to_adaptation": row["target_labels_passed_to_adaptation"],
                "q_passed_to_method": row["q_passed_to_method"],
                "target_performance_selection": row["target_performance_selection"],
            })
        for row in payload["geometry"]:
            geometry.append({
                "dataset": row["dataset"],
                "target_subject": int(row["target_subject"]),
                "source_seed": int(row["source_seed"]),
                "q": float(row["q"]),
                "method": row["method"],
                "checkpoint_hash": row["checkpoint_hash"],
                "adaptation_manifest_hash": row["adaptation_manifest_hash"],
                "translation_vector_hash": row["translation_vector_hash"],
                "log_scale_vector_hash": row["log_scale_vector_hash"],
                "translation_norm": float(row["translation_norm"]),
                "log_scale_norm": float(row["log_scale_norm"]),
                "translation_displacement_from_q05": float(row["translation_displacement_from_q05"]),
                "log_scale_displacement_from_q05": float(row["log_scale_displacement_from_q05"]),
                "geometry_displacement_from_q05": float(row["geometry_displacement_from_q05"]),
                "fitted_prior_class0": float(row["fitted_prior"][0]),
                "fitted_prior_class1": float(row["fitted_prior"][1]),
                "source_prior_class0": float(row["source_prior"][0]),
                "source_prior_class1": float(row["source_prior"][1]),
                "tta_seed": int(row["tta_seed"]),
            })
    results.sort(key=lambda row: (row["target_subject"], row["source_seed"], row["q"], METHODS.index(row["method"])))
    geometry.sort(key=lambda row: (row["target_subject"], row["source_seed"], row["q"], row["method"]))
    return results, geometry, prediction_vectors


def seed_averaged_subjects(
    rows: list[dict[str, Any]],
    predictions: dict[tuple[int, int, float, str], list[int]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    indexed = {
        (row["target_subject"], row["source_seed"], row["q"], row["method"]): row
        for row in rows
    }
    subjects = sorted({row["target_subject"] for row in rows})
    bacc_cube = np.empty((len(subjects), len(METHODS), len(Q_VALUES)), dtype=np.float64)
    acc_cube = np.empty_like(bacc_cube)
    per_subject: list[dict[str, Any]] = []
    for si, subject in enumerate(subjects):
        for mi, method in enumerate(METHODS):
            bacc_by_q = []
            acc_by_q = []
            disagreement_by_q = []
            for qi, q_text in enumerate(Q_VALUES):
                q = float(q_text)
                group = [indexed[(subject, seed, q, method)] for seed in runner.SEEDS]
                bacc = float(np.mean([row["bacc"] for row in group]))
                acc = float(np.mean([row["acc"] for row in group]))
                bacc_cube[si, mi, qi] = bacc
                acc_cube[si, mi, qi] = acc
                bacc_by_q.append(bacc)
                acc_by_q.append(acc)
                seed_disagreement = []
                for seed in runner.SEEDS:
                    current = np.asarray(predictions[(subject, seed, q, method)], dtype=np.int64)
                    center = np.asarray(predictions[(subject, seed, 0.5, method)], dtype=np.int64)
                    seed_disagreement.append(float(np.mean(current != center)))
                disagreement_by_q.append(float(np.mean(seed_disagreement)))
            sensitivity = 0.5 * (abs(bacc_by_q[0] - bacc_by_q[1]) + abs(bacc_by_q[2] - bacc_by_q[1]))
            endpoint_mean = 0.5 * (bacc_by_q[0] + bacc_by_q[2])
            worst = min(bacc_by_q[0], bacc_by_q[2])
            endpoint_disagreement = 0.5 * (disagreement_by_q[0] + disagreement_by_q[2])
            per_subject.append({
                "dataset": runner.DATASET,
                "target_subject": subject,
                "method": method,
                "n_source_seeds": 3,
                "bacc_q01": bacc_by_q[0],
                "bacc_q05": bacc_by_q[1],
                "bacc_q09": bacc_by_q[2],
                "acc_q01": acc_by_q[0],
                "acc_q05": acc_by_q[1],
                "acc_q09": acc_by_q[2],
                "prevalence_sensitivity": sensitivity,
                "endpoint_mean_bacc": endpoint_mean,
                "worst_prevalence_bacc": worst,
                "prediction_disagreement_q01_vs_q05": disagreement_by_q[0],
                "prediction_disagreement_q09_vs_q05": disagreement_by_q[2],
                "endpoint_mean_prediction_disagreement": endpoint_disagreement,
            })
    if len(per_subject) != 324:
        raise RuntimeError(f"expected 324 subject-method rows, found {len(per_subject)}")
    return per_subject, bacc_cube, acc_cube


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_endpoints(
    per_subject: list[dict[str, Any]],
    bacc_cube: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    subjects = sorted({row["target_subject"] for row in per_subject})
    lookup = {(row["target_subject"], row["method"]): row for row in per_subject}
    sensitivity = np.asarray([
        [lookup[(subject, method)]["prevalence_sensitivity"] for method in METHODS]
        for subject in subjects
    ])
    endpoint_mean = np.asarray([
        [lookup[(subject, method)]["endpoint_mean_bacc"] for method in METHODS]
        for subject in subjects
    ])
    worst = np.asarray([
        [lookup[(subject, method)]["worst_prevalence_bacc"] for method in METHODS]
        for subject in subjects
    ])
    disagreement = np.asarray([
        [lookup[(subject, method)]["endpoint_mean_prediction_disagreement"] for method in METHODS]
        for subject in subjects
    ])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPLICATES, len(subjects)))

    def boot_mean(matrix: np.ndarray) -> np.ndarray:
        return matrix[indices].mean(axis=1)

    sensitivity_boot = boot_mean(sensitivity)
    endpoint_boot = boot_mean(endpoint_mean)
    worst_boot = boot_mean(worst)
    disagreement_boot = boot_mean(disagreement)
    q_boot = np.stack([boot_mean(bacc_cube[:, :, qi]) for qi in range(len(Q_VALUES))], axis=2)

    sensitivity_rows: list[dict[str, Any]] = []
    for mi, method in enumerate(METHODS):
        low, high = percentile_ci(sensitivity_boot[:, mi])
        sensitivity_rows.append({
            "row_type": "method_summary",
            "method": method,
            "comparison": "",
            "estimate": float(sensitivity[:, mi].mean()),
            "ci_low": low,
            "ci_high": high,
            "n_subjects": len(subjects),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster_unit": "target_subject",
            "support_rule_pass": "",
        })
    fp_index = METHODS.index("FP-GEM")
    primary_verdict: dict[str, Any] | None = None
    for comparator in COMPARATORS:
        comparator_index = METHODS.index(comparator)
        boot_delta = sensitivity_boot[:, fp_index] - sensitivity_boot[:, comparator_index]
        low, high = percentile_ci(boot_delta)
        comparison = f"FP-GEM minus {comparator}"
        row = {
            "row_type": "paired_contrast",
            "method": "FP-GEM",
            "comparison": comparison,
            "estimate": float((sensitivity[:, fp_index] - sensitivity[:, comparator_index]).mean()),
            "ci_low": low,
            "ci_high": high,
            "n_subjects": len(subjects),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster_unit": "target_subject",
            "support_rule_pass": high < 0.0,
        }
        sensitivity_rows.append(row)
        if comparator == "Joint-GEM":
            primary_verdict = row

    endpoint_rows: list[dict[str, Any]] = []
    endpoint_specs = [
        ("endpoint_mean_bacc", endpoint_mean, endpoint_boot),
        ("worst_prevalence_bacc", worst, worst_boot),
        ("endpoint_mean_prediction_disagreement", disagreement, disagreement_boot),
    ]
    for qi, q_text in enumerate(Q_VALUES):
        endpoint_specs.append((f"mean_bacc_q{q_text.replace('.', '')}", bacc_cube[:, :, qi], q_boot[:, :, qi]))
    for endpoint, point_matrix, boot_matrix in endpoint_specs:
        for mi, method in enumerate(METHODS):
            low, high = percentile_ci(boot_matrix[:, mi])
            endpoint_rows.append({
                "row_type": "method_summary",
                "endpoint": endpoint,
                "method": method,
                "comparison": "",
                "estimate": float(point_matrix[:, mi].mean()),
                "ci_low": low,
                "ci_high": high,
                "n_subjects": len(subjects),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "cluster_unit": "target_subject",
            })
        for comparator in COMPARATORS:
            comparator_index = METHODS.index(comparator)
            delta = boot_matrix[:, fp_index] - boot_matrix[:, comparator_index]
            low, high = percentile_ci(delta)
            endpoint_rows.append({
                "row_type": "paired_contrast",
                "endpoint": endpoint,
                "method": "FP-GEM",
                "comparison": f"FP-GEM minus {comparator}",
                "estimate": float((point_matrix[:, fp_index] - point_matrix[:, comparator_index]).mean()),
                "ci_low": low,
                "ci_high": high,
                "n_subjects": len(subjects),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "cluster_unit": "target_subject",
            })
    if primary_verdict is None:
        raise RuntimeError("primary sensitivity contrast missing")
    return sensitivity_rows, endpoint_rows, primary_verdict


def geometry_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in ("Joint-GEM", "FP-GEM"):
        for q_text in Q_VALUES:
            q = float(q_text)
            group = [row for row in rows if row["method"] == method and row["q"] == q]
            output.append({
                "method": method,
                "q": q,
                "n_subject_seed_units": len(group),
                "mean_geometry_displacement_from_q05": float(np.mean([row["geometry_displacement_from_q05"] for row in group])),
                "mean_translation_displacement_from_q05": float(np.mean([row["translation_displacement_from_q05"] for row in group])),
                "mean_log_scale_displacement_from_q05": float(np.mean([row["log_scale_displacement_from_q05"] for row in group])),
                "mean_fitted_prior_class0": float(np.mean([row["fitted_prior_class0"] for row in group])),
                "mean_source_prior_class0": float(np.mean([row["source_prior_class0"] for row in group])),
            })
    return output


def write_head_to_head(
    path: Path,
    sensitivity_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    result_sha: str,
) -> None:
    sensitivity = {row.get("comparison") or row["method"]: row for row in sensitivity_rows}
    endpoint = {
        (row["endpoint"], row.get("comparison") or row["method"]): row
        for row in endpoint_rows
    }
    geometry = {(row["method"], row["q"]): row for row in geometry_rows}

    def ci(row: dict[str, Any], signed: bool = False) -> str:
        spec = "+.4f" if signed else ".4f"
        return f"{row['estimate']:{spec}} [{row['ci_low']:{spec}}, {row['ci_high']:{spec}}]"

    lines = [
        "# FP-GEM Fixed-Reservoir Prevalence Stress",
        "",
        "Status: P13 final targeted Lee2019_MI prevalence intervention. This is not a natural-transfer benchmark or a hyperparameter search.",
        "",
        "## Prevalence Sensitivity",
        "",
        "Lower is better. Source seeds are averaged within each target subject before the paired subject bootstrap.",
        "",
        "| method | sensitivity |",
        "|---|---:|",
    ]
    for method in METHODS:
        lines.append(f"| {method} | {ci(sensitivity[method])} |")
    lines.extend([
        "",
        "## Frozen Comparisons",
        "",
        "| comparison | paired difference | 95% CI entirely below zero |",
        "|---|---:|---:|",
    ])
    for comparator in COMPARATORS:
        key = f"FP-GEM minus {comparator}"
        row = sensitivity[key]
        lines.append(f"| {key} | {ci(row, signed=True)} | {str(row['support_rule_pass']).lower()} |")
    lines.extend([
        "",
        "The primary FP-GEM design claim uses only `FP-GEM minus Joint-GEM` and is supported only when the full percentile 95% CI is below zero. The three external comparisons are reported separately and are not selected post hoc.",
        "",
        "## Secondary Endpoints",
        "",
        "| method | endpoint mean bAcc | worst-prevalence bAcc | endpoint disagreement |",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        lines.append(
            f"| {method} | {ci(endpoint[('endpoint_mean_bacc', method)])} | "
            f"{ci(endpoint[('worst_prevalence_bacc', method)])} | "
            f"{ci(endpoint[('endpoint_mean_prediction_disagreement', method)])} |"
        )
    lines.extend([
        "",
        "## Mean bAcc By Prevalence",
        "",
        "| method | q=0.1 | q=0.5 | q=0.9 |",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        lines.append(
            f"| {method} | {ci(endpoint[('mean_bacc_q01', method)])} | "
            f"{ci(endpoint[('mean_bacc_q05', method)])} | "
            f"{ci(endpoint[('mean_bacc_q09', method)])} |"
        )
    lines.extend([
        "",
        "## Geometry Diagnostics",
        "",
        "| method | q | mean geometry displacement from q=0.5 | mean fitted class-0 prior |",
        "|---|---:|---:|---:|",
    ])
    for method in ("Joint-GEM", "FP-GEM"):
        for q in (0.1, 0.5, 0.9):
            row = geometry[(method, q)]
            lines.append(
                f"| {method} | {q:.1f} | {row['mean_geometry_displacement_from_q05']:.4f} | "
                f"{row['mean_fitted_prior_class0']:.4f} |"
            )
    lines.extend([
        "",
        "The evaluation block is unchanged and balanced. Target labels were used only by the offline intervention builder; adaptation methods received ordered EEG/features without labels or q. q=0.5 is an exact hash-gated P12 replay/reuse center.",
        "",
        "No equivalence, noninferiority, broad-benchmark, or natural-transfer superiority claim is made.",
        "",
        f"Final result CSV SHA-256: `{result_sha}`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_execution_audit(
    path: Path,
    validation: dict[str, Any],
    artifacts: list[dict[str, Any]],
    job_record: dict[str, Any],
    queue: dict[str, Any],
) -> None:
    stdout_counts = Counter(row["stdout_status"] for row in artifacts)
    stderr_counts = Counter(row["stderr_status"] for row in artifacts)
    lines = [
        "# FP-GEM Prevalence-Stress Execution Audit",
        "",
        f"- status: `{'PASS' if validation['validation_pass'] else 'BLOCKED'}`",
        f"- accepted target-seed units: `{validation['accepted_unit_count']}`",
        f"- final rows: `{validation['final_result_rows']}`",
        f"- geometry rows: `{validation['geometry_rows']}`",
        f"- final squeue absence: `{queue['all_absent']}`",
        f"- stdout statuses: `{dict(stdout_counts)}`",
        f"- stderr statuses: `{dict(stderr_counts)}`",
        "",
        "## Accepted Jobs",
        "",
        f"- checkpoint gate: `{job_record['checkpoint_gate']}`",
    ]
    for array in job_record["arrays"]:
        lines.append(f"- array: `{array}`")
    lines.extend([
        "",
        "Completion uses job absence from `squeue` plus stdout, stderr, artifact parse/count, and checksum gates. No Slurm accounting command is used.",
        "",
        "## Validation",
        "",
    ])
    for key, value in validation.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13")
    parser.add_argument("--out-dir", default="h2cmi/results/fp_gem_prevalence")
    parser.add_argument("--job-record", default="/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13/submission_record.json")
    args = parser.parse_args()
    out = ROOT / args.out_dir
    raw_root = Path(args.raw_root)
    config = runner.load_config()
    frozen = runner.validate_frozen_inputs(config)
    units, _ = runner.load_manifest(config)
    job_record = json.loads(Path(args.job_record).read_text())
    job_ids = [str(job_record["checkpoint_gate"]["job_id"])] + [
        str(item["job_id"]) for item in job_record["arrays"]
    ]
    queue = squeue_absence(job_ids)
    if not queue["all_absent"]:
        raise RuntimeError(f"accepted P13 jobs remain in squeue: {queue['stdout']}")
    payloads, artifacts = load_raw_units(config, units, raw_root)
    results, geometry, prediction_vectors = flatten(payloads)
    keys = [(row["target_subject"], row["source_seed"], row["q"], row["method"]) for row in results]
    geometry_keys = [(row["target_subject"], row["source_seed"], row["q"], row["method"]) for row in geometry]
    q05_rows = [row for row in results if row["q"] == 0.5]
    source_endpoint_rows = [
        row for row in results if row["q"] != 0.5 and row["method"] == "source_only_tsmnet"
    ]
    new_rows = [row for row in results if row["result_origin"] == "P13_new_prevalence_adaptation"]
    source_hash_consistent = True
    for target in runner.SUBJECTS:
        for seed in runner.SEEDS:
            hashes = {
                row["prediction_hash"]
                for row in results
                if row["target_subject"] == target
                and row["source_seed"] == seed
                and row["method"] == "source_only_tsmnet"
            }
            source_hash_consistent &= len(hashes) == 1
    validation = {
        "accepted_unit_count": len(payloads),
        "final_result_rows": len(results),
        "geometry_rows": len(geometry),
        "duplicate_result_keys": len(keys) - len(set(keys)),
        "duplicate_geometry_keys": len(geometry_keys) - len(set(geometry_keys)),
        "all_rows_ok": all(row["status"] == "ok" for row in results),
        "prediction_hashes_complete": all(row["prediction_hash"] for row in results),
        "logits_hashes_complete": all(row["logits_hash"] for row in results),
        "checkpoint_hashes_complete": all(row["checkpoint_hash"] for row in results),
        "adaptation_manifest_hashes_complete": all(row["adaptation_manifest_hash"] for row in results),
        "fixed_batch_size": all(row["n_adapt"] == 50 for row in results),
        "balanced_evaluation": all(row["class_counts_eval"] == "[25,25]" for row in results),
        "q05_exact_p12_center_count": len(q05_rows),
        "q05_all_exact": all(payload["q05_center_reproduction"]["p12_center_reuse_approved"] for payload in payloads),
        "source_only_endpoint_reuse_count": len(source_endpoint_rows),
        "source_only_q_independent": source_hash_consistent,
        "new_adaptation_row_count": len(new_rows),
        "target_label_leakage_detected": any(row["target_labels_passed_to_adaptation"] for row in results),
        "q_passed_to_method_detected": any(row["q_passed_to_method"] for row in results),
        "target_performance_selection_detected": any(row["target_performance_selection"] for row in results),
        "fresh_source_training_detected": any(payload["provenance"]["fresh_source_training_performed"] for payload in payloads),
        "all_adapt_eval_disjoint": all(payload["split"]["adapt_eval_disjoint"] for payload in payloads),
        "all_checkpoint_hashes_exact_p12": all(
            payload["checkpoint"]["checkpoint_sha256"] == units[(payload["target_subject"], payload["source_seed"])]["checkpoint_sha256"]
            for payload in payloads
        ),
        "all_runner_hashes_identical": len({payload["provenance"]["runner_sha256"] for payload in payloads}) == 1,
        "all_launch_commits_identical": len({payload["provenance"]["launch_commit"] for payload in payloads}) == 1,
        "manifest_sha256": runner.P13_MANIFEST_SHA256,
        "manifest_semantic_sha256": runner.P13_MANIFEST_SEMANTIC_SHA256,
        "config_sha256": runner.P13_CONFIG_SHA256,
        "source_seeds": sorted({row["source_seed"] for row in results}),
        "q_values": sorted({row["q"] for row in results}),
        "squeue_absence": queue["all_absent"],
        "stdout_validation_pass": all(row["stdout_status"] == "exists_complete_clean_launch" for row in artifacts),
        "stderr_validation_pass": all(row["stderr_status"] in {"empty", "known_harmless_warnings_only"} for row in artifacts),
    }
    validation["validation_pass"] = (
        validation["accepted_unit_count"] == 162
        and validation["final_result_rows"] == 2916
        and validation["geometry_rows"] == 972
        and validation["duplicate_result_keys"] == 0
        and validation["duplicate_geometry_keys"] == 0
        and validation["all_rows_ok"]
        and validation["prediction_hashes_complete"]
        and validation["logits_hashes_complete"]
        and validation["checkpoint_hashes_complete"]
        and validation["adaptation_manifest_hashes_complete"]
        and validation["fixed_batch_size"]
        and validation["balanced_evaluation"]
        and validation["q05_exact_p12_center_count"] == 972
        and validation["q05_all_exact"]
        and validation["source_only_endpoint_reuse_count"] == 324
        and validation["source_only_q_independent"]
        and validation["new_adaptation_row_count"] == 1620
        and not validation["target_label_leakage_detected"]
        and not validation["q_passed_to_method_detected"]
        and not validation["target_performance_selection_detected"]
        and not validation["fresh_source_training_detected"]
        and validation["all_adapt_eval_disjoint"]
        and validation["all_checkpoint_hashes_exact_p12"]
        and validation["all_runner_hashes_identical"]
        and validation["all_launch_commits_identical"]
        and validation["source_seeds"] == [0, 1, 2]
        and validation["q_values"] == [0.1, 0.5, 0.9]
        and validation["squeue_absence"]
        and validation["stdout_validation_pass"]
        and validation["stderr_validation_pass"]
    )
    if not validation["validation_pass"]:
        raise RuntimeError(f"P13 validation failed: {validation}")

    result_fields = [
        "dataset", "target_subject", "source_seed", "q", "method", "checkpoint_hash",
        "adaptation_manifest_hash", "n_adapt", "class_counts_adapt", "n_eval",
        "class_counts_eval", "acc", "bacc", "macro_f1", "prediction_hash", "logits_hash",
        "status", "failure_reason", "result_origin", "target_labels_passed_to_adaptation",
        "q_passed_to_method", "target_performance_selection",
    ]
    result_path = out / "fp_gem_prevalence_results.csv"
    write_csv(result_path, result_fields, results)
    result_sha = sha256_file(result_path)
    per_subject, bacc_cube, _ = seed_averaged_subjects(results, prediction_vectors)
    per_subject_path = out / "fp_gem_prevalence_per_subject.csv"
    write_csv(per_subject_path, list(per_subject[0]), per_subject)
    sensitivity_rows, endpoint_rows, primary = bootstrap_endpoints(per_subject, bacc_cube)
    sensitivity_path = out / "fp_gem_prevalence_sensitivity_ci.csv"
    write_csv(sensitivity_path, list(sensitivity_rows[0]), sensitivity_rows)
    endpoint_path = out / "fp_gem_prevalence_endpoint_ci.csv"
    write_csv(endpoint_path, list(endpoint_rows[0]), endpoint_rows)
    geometry_path = out / "fp_gem_prevalence_geometry_diagnostic.csv"
    write_csv(geometry_path, list(geometry[0]), geometry)
    artifact_path = out / "fp_gem_prevalence_job_artifact_manifest.csv"
    write_csv(artifact_path, list(artifacts[0]), artifacts)
    geometry_means = geometry_summary(geometry)
    head_path = out / "fp_gem_prevalence_head_to_head.md"
    write_head_to_head(head_path, sensitivity_rows, endpoint_rows, geometry_means, result_sha)
    audit_path = out / "fp_gem_prevalence_execution_audit.md"
    write_execution_audit(audit_path, validation, artifacts, job_record, queue)
    summary = {
        "status": "pass",
        "label": "P13 targeted fixed-reservoir prevalence stress",
        "dataset": runner.DATASET,
        "subjects": list(runner.SUBJECTS),
        "source_seeds": list(runner.SEEDS),
        "q_values": [float(q) for q in Q_VALUES],
        "methods": list(METHODS),
        "frozen_inputs": frozen,
        "aggregation": {
            "seed_average_first": True,
            "biological_cluster": "target_subject",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_methods_and_q_preserved": True,
            "interval": "percentile_95",
        },
        "primary_endpoint": "0.5*(abs(bacc_q01-bacc_q05)+abs(bacc_q09-bacc_q05))",
        "primary_comparison": primary,
        "primary_method_claim_supported": bool(primary["support_rule_pass"]),
        "external_sensitivity_comparisons": [
            row for row in sensitivity_rows
            if row["row_type"] == "paired_contrast" and row["comparison"] != "FP-GEM minus Joint-GEM"
        ],
        "sensitivity_summaries": sensitivity_rows,
        "secondary_endpoint_summaries": endpoint_rows,
        "geometry_summaries": geometry_means,
        "validation": validation,
        "no_equivalence_or_noninferiority_claim": True,
        "artifacts": {
            "fp_gem_prevalence_results.csv": result_sha,
            "fp_gem_prevalence_per_subject.csv": sha256_file(per_subject_path),
            "fp_gem_prevalence_sensitivity_ci.csv": sha256_file(sensitivity_path),
            "fp_gem_prevalence_endpoint_ci.csv": sha256_file(endpoint_path),
            "fp_gem_prevalence_geometry_diagnostic.csv": sha256_file(geometry_path),
            "fp_gem_prevalence_job_artifact_manifest.csv": sha256_file(artifact_path),
            "fp_gem_prevalence_head_to_head.md": sha256_file(head_path),
            "fp_gem_prevalence_execution_audit.md": sha256_file(audit_path),
        },
        "job_record": job_record,
        "final_squeue": queue,
    }
    summary_path = out / "fp_gem_prevalence_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({
        "status": "pass",
        "result_rows": len(results),
        "result_sha256": result_sha,
        "primary_comparison": primary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
