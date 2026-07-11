"""Validate P9 shards and build the official three-source-seed SPDIM packet."""
from __future__ import annotations

import csv
import getpass
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "h2cmi" / "results" / "review_completion"
SHARD_DIR = OUT_DIR / "spdim_w1_repaired_seeds12_shards"
EXCLUDED_DIR = OUT_DIR / "spdim_w1_repaired_seeds12_excluded"
COMMAND_LOG = OUT_DIR / "COMMAND_LOG.md"

LAUNCH_COMMIT = "493f649911f354d4deaa307c80be3cb367149fce"
ARRAY_JOB_ID = "892389"
EXCLUDED_ARRAY_JOB_ID = "892385"
MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
MANIFEST_FILE_SHA256 = "e9ebe6e9421bdcf10f8a952623285cec0842f5cb6b868e8147f13dde23e8a712"
P8_RUNNER_SHA256 = "946b28b93f0ddbce395ade7c6a13d30b20f368fe7a1ae22fbefa01f291e82be8"
P9_CONTROLLER_SHA256 = "3be4f8edabe577b471cf899b17cd1249749092949071334a5c7566a805910207"
CONFIG_SHA256 = "6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b"
LAUNCHER_SHA256 = "cc474195bd510065ef19518c33744e19f87e32d5fd4909b17d294a113d43a1dd"
EXTERNAL_SHA = "1b0de0ccd4c48a4ff28f087b866a0b671b029c39"
SEED0_SHA256 = "118ec37f3a195d50c24abf24b4c61048cdbc0ffff7d9c0f0bf51c83f7f69229c"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260710

DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
METHODS = ["source_only_tsmnet", "rct", "spdim_geodesic", "spdim_bias"]
METRICS = ["acc", "bacc"]
EXPECTED_SUBJECTS = {
    "BNCI2014_001": set(range(1, 10)),
    "Cho2017": set(range(1, 53)),
    "Lee2019_MI": set(range(1, 55)),
}
EXPECTED_PER_SEED_DATASET_ROWS = {"BNCI2014_001": 36, "Cho2017": 208, "Lee2019_MI": 216}
EXPECTED_SEEDS12_DATASET_ROWS = {"BNCI2014_001": 72, "Cho2017": 416, "Lee2019_MI": 432}
EXPECTED_FINAL_DATASET_ROWS = {"BNCI2014_001": 108, "Cho2017": 624, "Lee2019_MI": 648}
CONTRASTS = {
    "rct_minus_source_only_tsmnet": ("rct", "source_only_tsmnet"),
    "spdim_geodesic_minus_source_only_tsmnet": ("spdim_geodesic", "source_only_tsmnet"),
    "spdim_bias_minus_source_only_tsmnet": ("spdim_bias", "source_only_tsmnet"),
    "spdim_geodesic_minus_rct": ("spdim_geodesic", "rct"),
    "spdim_bias_minus_rct": ("spdim_bias", "rct"),
}
HARM_THRESHOLDS = [0.0, -0.01, -0.02]

SHARDS = [
    {"id": "seed1_shard0", "task": 0, "seed": 1, "job_id": "892464", "partition": "V100", "device": "Tesla V100-PCIE-16GB", "capability": "7.0", "target_spec": "BNCI2014_001=1-9;Cho2017=1-20", "rows": 116, "dataset_rows": {"BNCI2014_001": 36, "Cho2017": 80}},
    {"id": "seed1_shard1", "task": 1, "seed": 1, "job_id": "892465", "partition": "V100", "device": "Tesla V100S-PCIE-32GB", "capability": "7.0", "target_spec": "Cho2017=21-49", "rows": 116, "dataset_rows": {"Cho2017": 116}},
    {"id": "seed1_shard2", "task": 2, "seed": 1, "job_id": "892466", "partition": "V100", "device": "Tesla V100S-PCIE-32GB", "capability": "7.0", "target_spec": "Cho2017=50-52;Lee2019_MI=1-26", "rows": 116, "dataset_rows": {"Cho2017": 12, "Lee2019_MI": 104}},
    {"id": "seed1_shard3", "task": 3, "seed": 1, "job_id": "892467", "partition": "V100", "device": "Tesla V100S-PCIE-32GB", "capability": "7.0", "target_spec": "Lee2019_MI=27-54", "rows": 112, "dataset_rows": {"Lee2019_MI": 112}},
    {"id": "seed2_shard0", "task": 4, "seed": 2, "job_id": "892842", "partition": "V100", "device": "Tesla V100S-PCIE-32GB", "capability": "7.0", "target_spec": "BNCI2014_001=1-9;Cho2017=1-20", "rows": 116, "dataset_rows": {"BNCI2014_001": 36, "Cho2017": 80}},
    {"id": "seed2_shard1", "task": 5, "seed": 2, "job_id": "892883", "partition": "A40", "device": "NVIDIA A40", "capability": "8.6", "target_spec": "Cho2017=21-49", "rows": 116, "dataset_rows": {"Cho2017": 116}},
    {"id": "seed2_shard2", "task": 6, "seed": 2, "job_id": "892957", "partition": "V100", "device": "Tesla V100S-PCIE-32GB", "capability": "7.0", "target_spec": "Cho2017=50-52;Lee2019_MI=1-26", "rows": 116, "dataset_rows": {"Cho2017": 12, "Lee2019_MI": 104}},
    {"id": "seed2_shard3", "task": 7, "seed": 2, "job_id": "892389", "partition": "A100", "device": "NVIDIA A100-SXM4-40GB", "capability": "8.0", "target_spec": "Lee2019_MI=27-54", "rows": 112, "dataset_rows": {"Lee2019_MI": 112}},
]

SEEDS12_RESULTS = OUT_DIR / "spdim_w1_repaired_seeds12_results.csv"
SEEDS12_AUDIT = OUT_DIR / "spdim_w1_repaired_seeds12_audit.md"
SEEDS12_SUMMARY = OUT_DIR / "spdim_w1_repaired_seeds12_summary.json"
THREE_RESULTS = OUT_DIR / "spdim_w1_repaired_three_seed_results.csv"
THREE_SUMMARY = OUT_DIR / "spdim_w1_repaired_three_seed_summary.json"
THREE_DIGEST = OUT_DIR / "spdim_w1_repaired_three_seed_result_digest.md"
RED_TEAM_REVIEW = OUT_DIR / "spdim_w1_repaired_three_seed_red_team_review.md"
METHOD_CI = OUT_DIR / "spdim_w1_repaired_three_seed_method_ci.csv"
CONTRAST_CI = OUT_DIR / "spdim_w1_repaired_three_seed_contrast_ci.csv"
HARM_CSV = OUT_DIR / "spdim_w1_repaired_three_seed_harm.csv"
STABILITY_CSV = OUT_DIR / "spdim_w1_repaired_three_seed_seed_stability.csv"
FAILURE_TRACE = OUT_DIR / "spdim_w1_repaired_three_seed_failure_trace.txt"
SEED0_RESULTS = OUT_DIR / "spdim_w1_repaired_seed0_results.csv"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader), list(reader.fieldnames)


def line_value(text: str, key: str) -> str | None:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


def final_squeue_gate() -> dict[str, Any]:
    user = getpass.getuser()
    command = ["squeue", "-h", "-u", user, "-o", "%i|%F|%K|%t|%j|%P"]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"squeue failed with code {proc.returncode}: {proc.stderr.strip()}")
    matching = []
    for line in proc.stdout.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, array_id, task_id, state, name, partition = parts
        if job_id == ARRAY_JOB_ID or array_id == ARRAY_JOB_ID or job_id.startswith(f"{ARRAY_JOB_ID}_"):
            matching.append({"job_id": job_id, "array_id": array_id, "task_id": task_id, "state": state, "name": name, "partition": partition})
    return {
        "command": " ".join(command),
        "user": user,
        "array_job_id": ARRAY_JOB_ID,
        "matching_queue_rows": matching,
        "final_squeue_absent": not matching,
        "stderr": proc.stderr.strip(),
    }


def accepted_stderr(path: Path) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    allowed = {
        "Trials demeaned and stacked with zero buffer to create continuous data -- edge effects present": "moabb_zero_buffer_edge_warning",
        "Matplotlib is building the font cache; this may take a moment.": "matplotlib_font_cache_notice",
    }
    unexpected = [line for line in lines if line not in allowed]
    forbidden_fragments = [
        "not compatible", "no kernel image", "cuda initialization", "unsupported gpu", "traceback", "runtimeerror",
    ]
    real_failure_lines = [line for line in lines if any(fragment in line.lower() for fragment in forbidden_fragments)]
    if not path.exists():
        status = "missing"
    elif not lines:
        status = "empty"
    elif not unexpected and not real_failure_lines:
        status = "known_harmless_warnings_only"
    else:
        status = "real_or_unexpected_failure"
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "lines": len(lines),
        "status": status,
        "warning_patterns": sorted({allowed[line] for line in lines if line in allowed}),
        "unexpected_lines": unexpected[:20],
        "real_failure_lines": real_failure_lines[:20],
        "sha256": sha256_file(path) if path.exists() else "",
    }


def accepted_stdout(path: Path, shard: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    required = {
        "job_id": shard["job_id"],
        "array_job_id": ARRAY_JOB_ID,
        "array_task_id": str(shard["task"]),
        "seed": str(shard["seed"]),
        "shard_id": shard["id"],
        "launch_commit": LAUNCH_COMMIT,
        "manifest_hash": MANIFEST_HASH,
        "target_spec": shard["target_spec"],
        "expected_rows": str(shard["rows"]),
        "controller_sha256": P9_CONTROLLER_SHA256,
        "runner_sha256": P8_RUNNER_SHA256,
        "config_sha256": CONFIG_SHA256,
        "manifest_csv_sha256": MANIFEST_FILE_SHA256,
        "external_sha": EXTERNAL_SHA,
        "runtime_sys_executable": "/home/infres/yinwang/anaconda3/envs/icml/bin/python",
        "runtime_sys_prefix": "/home/infres/yinwang/anaconda3/envs/icml",
        "runtime_python_version": "3.9.25",
        "runtime_pytorch_version": "2.8.0+cu128",
        "runtime_cuda_version": "12.8",
        "runtime_cuda_available": "True",
        "runtime_device": "cuda",
        "runtime_device_name": shard["device"],
        "runtime_moabb_version": "1.2.0",
        "runtime_mne_version": "1.8.0",
    }
    observed = {key: line_value(text, key) for key in required}
    mismatches = {key: {"expected": value, "observed": observed[key]} for key, value in required.items() if observed[key] != value}
    clean = "repo_status_porcelain_begin\nrepo_status_porcelain_end" in text
    forbidden = [fragment for fragment in ("not compatible", "no kernel image", "CUDA initialization", "Traceback") if fragment.lower() in text.lower()]
    status = "pass" if path.exists() and text and not mismatches and clean and not forbidden else "fail"
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "lines": len(text.splitlines()),
        "status": status,
        "clean_porcelain_block": clean,
        "header_mismatches": mismatches,
        "forbidden_runtime_patterns": forbidden,
        "runtime": {key.removeprefix("runtime_"): observed[key] for key in observed if key.startswith("runtime_")},
        "sha256": sha256_file(path) if path.exists() else "",
    }


def excluded_p100_evidence() -> dict[str, Any]:
    tasks = []
    for task in (0, 1):
        stdout = EXCLUDED_DIR / f"p100_892385_task{task}.stdout.txt"
        stderr = EXCLUDED_DIR / f"p100_892385_task{task}.stderr.txt"
        stdout_text = stdout.read_text()
        stderr_text = stderr.read_text()
        incompatible = "sm_60 is not compatible" in stderr_text and "supports CUDA capabilities sm_70" in stderr_text
        tasks.append({
            "array_task_id": task,
            "job_id": line_value(stdout_text, "job_id"),
            "shard_id": line_value(stdout_text, "shard_id"),
            "stdout": str(stdout.relative_to(ROOT)),
            "stdout_sha256": sha256_file(stdout),
            "stderr": str(stderr.relative_to(ROOT)),
            "stderr_sha256": sha256_file(stderr),
            "stderr_status": "unsupported_gpu_architecture_real_failure" if incompatible else "missing_required_failure_evidence",
            "unsupported_sm60_confirmed": incompatible,
            "training_output_detected": "epoch=" in stdout_text,
            "slurm_cancellation_recorded": "CANCELLED" in stderr_text,
        })

    retained_summaries = []
    for path in sorted(SHARD_DIR.glob("*_summary.json")):
        payload = json.loads(path.read_text())
        retained_summaries.append({
            "path": str(path.relative_to(ROOT)),
            "job_id": str(payload.get("launch_provenance", {}).get("slurm_job_id", "")),
            "row_count": int(payload.get("validation", {}).get("row_count", 0) or 0),
        })
    p100_job_ids = {item["job_id"] for item in tasks if item["job_id"]}
    p100_attributed = [item for item in retained_summaries if item["job_id"] in p100_job_ids]
    accepted_result_rows = sum(item["row_count"] for item in p100_attributed)
    result_like = sorted(str(path.relative_to(ROOT)) for path in EXCLUDED_DIR.glob("*results*"))
    return {
        "array_job_id": EXCLUDED_ARRAY_JOB_ID,
        "partition": "P100",
        "status": "zero_result_excluded_launch",
        "reason": "Tesla P100 sm_60 is unsupported by the frozen PyTorch build, which supports sm_70 and newer.",
        "started_job_ids": sorted(p100_job_ids),
        "accepted_result_rows": accepted_result_rows,
        "result_like_files_in_excluded_evidence": result_like,
        "retained_result_summaries": retained_summaries,
        "p100_attributed_result_summaries": p100_attributed,
        "tasks_with_logs": tasks,
        "tasks_without_logs": [2, 3, 4, 5, 6, 7],
        "gate_pass": (
            not result_like
            and accepted_result_rows == 0
            and not p100_attributed
            and len(p100_job_ids) == len(tasks)
            and all(task["unsupported_sm60_confirmed"] for task in tasks)
            and all(task["slurm_cancellation_recorded"] for task in tasks)
            and all(not task["training_output_detected"] for task in tasks)
        ),
    }


def load_shards() -> tuple[list[dict[str, str]], list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    metadata = []
    errors = []
    fieldnames: list[str] | None = None
    for shard in SHARDS:
        base = shard["id"]
        result_path = SHARD_DIR / f"{base}_results.csv"
        summary_path = SHARD_DIR / f"{base}_summary.json"
        stdout_path = SHARD_DIR / f"{base}.stdout.txt"
        stderr_path = SHARD_DIR / f"{base}.stderr.txt"
        shard_rows, fields = read_csv(result_path)
        if fieldnames is None:
            fieldnames = fields
        elif fields != fieldnames:
            errors.append(f"{base}: CSV header differs")
        summary = json.loads(summary_path.read_text())
        launch = summary.get("launch_provenance", {})
        validation = summary.get("validation", {})
        controller = summary.get("p9_controller", {})
        runtime = summary.get("runtime_environment", {})
        stdout = accepted_stdout(stdout_path, shard)
        stderr = accepted_stderr(stderr_path)
        result_sha = sha256_file(result_path)
        checks = {
            "row_count": len(shard_rows) == shard["rows"],
            "dataset_rows": dict(Counter(row["dataset"] for row in shard_rows)) == shard["dataset_rows"],
            "summary_pass": summary.get("status") == "pass",
            "summary_validation": validation.get("validation_pass") is True,
            "summary_rows": validation.get("row_count") == shard["rows"],
            "summary_checksum": summary.get("result_csv_sha256") == result_sha,
            "summary_seed": summary.get("source_seed") == shard["seed"],
            "summary_shard": summary.get("shard_id") == base,
            "summary_target_spec": summary.get("target_spec") == shard["target_spec"],
            "launch_commit": launch.get("launch_commit") == LAUNCH_COMMIT,
            "clean_worktree": launch.get("clean_worktree") is True and launch.get("git_status_porcelain") == "",
            "dirty_disallowed": launch.get("runner_dirty_allowed") is False,
            "job_id": launch.get("slurm_job_id") == shard["job_id"],
            "runner_checksum": launch.get("runner_file_sha256") == P8_RUNNER_SHA256,
            "config_checksum": launch.get("config_sha256") == CONFIG_SHA256,
            "manifest_hash": launch.get("manifest_hash") == MANIFEST_HASH,
            "manifest_file_checksum": launch.get("manifest_file_sha256") == MANIFEST_FILE_SHA256,
            "external_commit": launch.get("external_spdim_commit") == EXTERNAL_SHA,
            "controller_checksum": controller.get("sha256") == P9_CONTROLLER_SHA256,
            "controller_seed": controller.get("executed_seed") == shard["seed"],
            "runtime_device": runtime.get("device_name") == shard["device"],
            "runtime_icml": runtime.get("sys_executable") == "/home/infres/yinwang/anaconda3/envs/icml/bin/python",
            "stdout": stdout["status"] == "pass",
            "stderr": stderr["status"] in {"empty", "known_harmless_warnings_only"},
        }
        if not all(checks.values()):
            errors.append(f"{base}: failed checks {sorted(key for key, value in checks.items() if not value)}")
        for row in shard_rows:
            row["_source_shard"] = base
        rows.extend(shard_rows)
        metadata.append({
            "shard_id": base,
            "array_task_id": shard["task"],
            "job_id": shard["job_id"],
            "partition": shard["partition"],
            "source_seed": shard["seed"],
            "target_spec": shard["target_spec"],
            "expected_rows": shard["rows"],
            "row_count": len(shard_rows),
            "result_csv": str(result_path.relative_to(ROOT)),
            "result_csv_sha256": result_sha,
            "summary_json": str(summary_path.relative_to(ROOT)),
            "summary_json_sha256": sha256_file(summary_path),
            "stdout": stdout,
            "stderr": stderr,
            "runtime_environment": runtime,
            "compute_capability": shard["capability"],
            "compute_capability_source": "GPU model architecture mapping; the runtime controller did not emit capability directly.",
            "checks": checks,
            "shard_pass": all(checks.values()),
        })
    if fieldnames is None:
        raise RuntimeError("no shard CSVs")
    return rows, metadata, fieldnames, errors


def row_key(row: dict[str, str]) -> tuple[str, int, int, str]:
    return row["dataset"], int(row["target_subject"]), int(row["source_seed"]), row["method"]


def sorted_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ds_order = {value: index for index, value in enumerate(DATASETS)}
    method_order = {value: index for index, value in enumerate(METHODS)}
    return sorted(rows, key=lambda row: (ds_order[row["dataset"]], int(row["target_subject"]), int(row["source_seed"]), method_order[row["method"]]))


def validate_result_rows(rows: list[dict[str, str]], seeds: set[int]) -> dict[str, Any]:
    keys = [row_key(row) for row in rows]
    dataset_counts = Counter(row["dataset"] for row in rows)
    seed_counts = Counter(int(row["source_seed"]) for row in rows)
    seed_dataset = Counter((int(row["source_seed"]), row["dataset"]) for row in rows)
    method_counts = Counter(row["method"] for row in rows)
    expected_dataset = EXPECTED_SEEDS12_DATASET_ROWS if seeds == {1, 2} else EXPECTED_FINAL_DATASET_ROWS
    expected_seed_counts = {seed: 460 for seed in seeds}
    expected_method_rows = 230 if seeds == {1, 2} else 345
    subjects = {dataset: {int(row["target_subject"]) for row in rows if row["dataset"] == dataset} for dataset in DATASETS}
    eval_counts = [json.loads(row["class_counts_eval"]) for row in rows]
    adapt_counts = [json.loads(row["class_counts_adapt"]) for row in rows]
    checks = {
        "expected_row_count": len(rows) == len(seeds) * 460,
        "unique_keys": len(keys) == len(set(keys)),
        "dataset_rows": dict(dataset_counts) == expected_dataset,
        "seed_rows": dict(seed_counts) == expected_seed_counts,
        "seed_dataset_rows": all(seed_dataset[(seed, dataset)] == EXPECTED_PER_SEED_DATASET_ROWS[dataset] for seed in seeds for dataset in DATASETS),
        "method_rows": dict(method_counts) == {method: expected_method_rows for method in METHODS},
        "status_ok": Counter(row["status"] for row in rows) == {"ok": len(rows)},
        "subject_coverage": all(subjects[dataset] == EXPECTED_SUBJECTS[dataset] for dataset in DATASETS),
        "prediction_hash_complete": all(row["prediction_hash"] for row in rows),
        "logits_hash_complete": all(row["logits_hash"] for row in rows),
        "split_hash_complete": all(row["split_hash"] for row in rows),
        "both_classes_adapt": all(len(counts) == 2 and min(counts) > 0 for counts in adapt_counts),
        "both_classes_eval": all(len(counts) == 2 and min(counts) > 0 for counts in eval_counts),
        "balanced_eval": all(counts[0] == counts[1] for counts in eval_counts),
        "adapt_eval_disjoint": all(row["adapt_eval_disjoint"] == "True" for row in rows),
        "acc_equals_bacc": all(abs(float(row["acc"]) - float(row["bacc"])) <= 1e-12 for row in rows),
        "no_failure_reason": all(not row["failure_reason"] for row in rows),
        "no_target_label_leakage": all(row["target_label_leakage_detected"] == "False" for row in rows),
        "no_target_performance_selection": all(row["method_selection_uses_target_performance"] == "False" for row in rows),
        "no_pretrained_weight": all(row["official_pretrained_weight_used"] == "False" for row in rows),
        "no_vendoring": all(row["third_party_vendored"] == "False" for row in rows),
        "manifest_hash": {row["manifest_hash"] for row in rows} == {MANIFEST_HASH},
        "external_commit": {row["official_sha"] for row in rows} == {EXTERNAL_SHA},
        "source_model_hash_complete": all(row["source_model_sha256"] for row in rows),
        "trial_index_hashes_complete": all(row["source_idx_sha256"] and row["adapt_idx_sha256"] and row["eval_idx_sha256"] for row in rows),
    }
    return {
        "validation_pass": all(checks.values()),
        "checks": checks,
        "row_count": len(rows),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "dataset_rows": dict(sorted(dataset_counts.items())),
        "seed_rows": {str(key): value for key, value in sorted(seed_counts.items())},
        "seed_dataset_rows": {f"seed{seed}:{dataset}": seed_dataset[(seed, dataset)] for seed in sorted(seeds) for dataset in DATASETS},
        "method_rows": dict(sorted(method_counts.items())),
        "single_class_adapt_rows": sum(min(counts) <= 0 for counts in adapt_counts),
        "single_class_eval_rows": sum(min(counts) <= 0 for counts in eval_counts),
        "adapt_eval_disjoint_failures": sum(row["adapt_eval_disjoint"] != "True" for row in rows),
        "prediction_hash_missing_rows": sum(not row["prediction_hash"] for row in rows),
        "logits_hash_missing_rows": sum(not row["logits_hash"] for row in rows),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def seed_averaged_units(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, float]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["target_subject"]), row["method"])].append(row)
    units = {}
    for key, group in grouped.items():
        seeds = sorted(int(row["source_seed"]) for row in group)
        if seeds != [0, 1, 2]:
            raise ValueError(f"{key}: expected seeds 0/1/2, got {seeds}")
        units[key] = {metric: mean([float(row[metric]) for row in group]) for metric in ("acc", "bacc", "macro_f1")}
    if len(units) != 115 * len(METHODS):
        raise ValueError(f"expected 460 seed-averaged units, got {len(units)}")
    return units


def bootstrap_tables(units: dict[tuple[str, int, str], dict[str, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = {dataset: sorted(EXPECTED_SUBJECTS[dataset]) for dataset in DATASETS}
    arrays: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for dataset in DATASETS:
        for metric in METRICS:
            for method in METHODS:
                arrays[dataset][metric][method] = np.asarray([units[(dataset, subject, method)][metric] for subject in subjects[dataset]], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = {dataset: rng.integers(0, len(subjects[dataset]), size=(BOOTSTRAP_REPLICATES, len(subjects[dataset]))) for dataset in DATASETS}

    def interval(distribution: np.ndarray) -> tuple[float, float]:
        low, high = np.quantile(distribution, [0.025, 0.975])
        return float(low), float(high)

    method_rows = []
    method_boot: dict[tuple[str, str, str], np.ndarray] = {}
    for metric in METRICS:
        for method in METHODS:
            per_dataset_dist = {}
            for dataset in DATASETS:
                values = arrays[dataset][metric][method]
                distribution = values[indices[dataset]].mean(axis=1)
                per_dataset_dist[dataset] = distribution
                method_boot[(metric, method, dataset)] = distribution
                low, high = interval(distribution)
                method_rows.append({
                    "metric": metric, "method": method, "estimand": "per_dataset", "dataset": dataset,
                    "n_clusters": len(values), "estimate": float(values.mean()), "ci_low": low, "ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                    "cluster_unit": "dataset_x_target_subject", "dataset_stratified": True, "seeds_averaged_first": True,
                })
            weighted = sum(per_dataset_dist[dataset] * len(subjects[dataset]) for dataset in DATASETS) / 115.0
            macro = sum(per_dataset_dist.values()) / len(DATASETS)
            observed_dataset = {dataset: arrays[dataset][metric][method].mean() for dataset in DATASETS}
            observed_weighted = sum(observed_dataset[dataset] * len(subjects[dataset]) for dataset in DATASETS) / 115.0
            observed_macro = sum(observed_dataset.values()) / len(DATASETS)
            for estimand, distribution, estimate in (
                ("subject_weighted", weighted, observed_weighted),
                ("dataset_macro", macro, observed_macro),
            ):
                low, high = interval(distribution)
                method_rows.append({
                    "metric": metric, "method": method, "estimand": estimand, "dataset": "ALL",
                    "n_clusters": 115, "estimate": float(estimate), "ci_low": low, "ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                    "cluster_unit": "dataset_x_target_subject", "dataset_stratified": True, "seeds_averaged_first": True,
                })

    contrast_rows = []
    for metric in METRICS:
        for contrast, (method, baseline) in CONTRASTS.items():
            per_dataset_dist = {}
            observed_dataset = {}
            for dataset in DATASETS:
                delta = arrays[dataset][metric][method] - arrays[dataset][metric][baseline]
                distribution = delta[indices[dataset]].mean(axis=1)
                per_dataset_dist[dataset] = distribution
                observed_dataset[dataset] = delta.mean()
                low, high = interval(distribution)
                contrast_rows.append({
                    "metric": metric, "contrast": contrast, "method": method, "baseline": baseline,
                    "estimand": "per_dataset", "dataset": dataset, "n_clusters": len(delta),
                    "estimate": float(delta.mean()), "ci_low": low, "ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                    "cluster_unit": "dataset_x_target_subject", "dataset_stratified": True,
                    "paired_methods": True, "seeds_averaged_first": True,
                })
            weighted = sum(per_dataset_dist[dataset] * len(subjects[dataset]) for dataset in DATASETS) / 115.0
            macro = sum(per_dataset_dist.values()) / len(DATASETS)
            observed_weighted = sum(observed_dataset[dataset] * len(subjects[dataset]) for dataset in DATASETS) / 115.0
            observed_macro = sum(observed_dataset.values()) / len(DATASETS)
            for estimand, distribution, estimate in (
                ("subject_weighted", weighted, observed_weighted),
                ("dataset_macro", macro, observed_macro),
            ):
                low, high = interval(distribution)
                contrast_rows.append({
                    "metric": metric, "contrast": contrast, "method": method, "baseline": baseline,
                    "estimand": estimand, "dataset": "ALL", "n_clusters": 115,
                    "estimate": float(estimate), "ci_low": low, "ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                    "cluster_unit": "dataset_x_target_subject", "dataset_stratified": True,
                    "paired_methods": True, "seeds_averaged_first": True,
                })
    return method_rows, contrast_rows


def harm_rows(units: dict[tuple[str, int, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for contrast, (method, baseline) in CONTRASTS.items():
        by_dataset = {
            dataset: [units[(dataset, subject, method)]["bacc"] - units[(dataset, subject, baseline)]["bacc"] for subject in sorted(EXPECTED_SUBJECTS[dataset])]
            for dataset in DATASETS
        }
        for threshold in HARM_THRESHOLDS:
            for dataset in DATASETS:
                values = by_dataset[dataset]
                count = sum(value < threshold for value in values)
                rows.append({
                    "metric": "bacc", "contrast": contrast, "method": method, "baseline": baseline,
                    "scope": "per_dataset", "dataset": dataset, "threshold": threshold,
                    "n_clusters": len(values), "harm_count": count, "harm_rate": count / len(values),
                    "seeds_averaged_first": True,
                })
            all_values = [value for dataset in DATASETS for value in by_dataset[dataset]]
            count = sum(value < threshold for value in all_values)
            rows.append({
                "metric": "bacc", "contrast": contrast, "method": method, "baseline": baseline,
                "scope": "subject_weighted", "dataset": "ALL", "threshold": threshold,
                "n_clusters": len(all_values), "harm_count": count, "harm_rate": count / len(all_values),
                "seeds_averaged_first": True,
            })
    return rows


def stability_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seed_means: dict[tuple[str, str, str, int], float] = {}
    for dataset in DATASETS:
        for metric in METRICS:
            for method in METHODS:
                for seed in (0, 1, 2):
                    values = [float(row[metric]) for row in rows if row["dataset"] == dataset and row["method"] == method and int(row["source_seed"]) == seed]
                    seed_means[(dataset, metric, method, seed)] = mean(values)
    ranks = {}
    for dataset in DATASETS:
        for metric in METRICS:
            for seed in (0, 1, 2):
                values = {method: seed_means[(dataset, metric, method, seed)] for method in METHODS}
                for method in METHODS:
                    ranks[(dataset, metric, method, seed)] = 1 + sum(value > values[method] for value in values.values())
    output = []
    for dataset in DATASETS:
        for metric in METRICS:
            for method in METHODS:
                values = [seed_means[(dataset, metric, method, seed)] for seed in (0, 1, 2)]
                output.append({
                    "dataset": dataset, "metric": metric, "method": method,
                    "seed0_mean": values[0], "seed1_mean": values[1], "seed2_mean": values[2],
                    "range": max(values) - min(values), "population_std": statistics.pstdev(values),
                    "seed0_rank": ranks[(dataset, metric, method, 0)],
                    "seed1_rank": ranks[(dataset, metric, method, 1)],
                    "seed2_rank": ranks[(dataset, metric, method, 2)],
                    "rank_rule": "1 + count of strictly greater method means; ties share rank",
                })
    return output


def write_evidence_readmes(shards: list[dict[str, Any]], excluded: dict[str, Any]) -> None:
    lines = [
        "# P9 Accepted Seed 1/2 Shard Evidence", "",
        "This directory contains the eight accepted seed-by-shard result CSVs, summaries, and Slurm stdout/stderr files used by the final merge.", "",
        "- Result CSV and summary bytes are copied unchanged from the repository-external run cache.",
        "- Text logs have only trailing horizontal whitespace removed for Git whitespace validation.",
        "- Every committed input checksum is recorded below and in the seeds-1/2 summary JSON.", "",
        "| shard | job | partition | GPU | rows | result sha256 | summary sha256 | stdout sha256 | stderr sha256 |", "|---|---:|---|---|---:|---|---|---|---|",
    ]
    for shard in shards:
        lines.append(
            f"| {shard['shard_id']} | {shard['job_id']} | {shard['partition']} | {shard['runtime_environment']['device_name']} | {shard['row_count']} | "
            f"`{shard['result_csv_sha256']}` | `{shard['summary_json_sha256']}` | `{shard['stdout']['sha256']}` | `{shard['stderr']['sha256']}` |"
        )
    lines.extend(["", "## Internal Validation Review", "", "- All eight inputs pass provenance, log, row-count, split, and leakage gates.", "- The excluded P100 launch is stored separately and contributes no rows."])
    (SHARD_DIR / "README.md").write_text("\n".join(lines) + "\n")

    ex_lines = [
        "# Excluded P100 Launch 892385", "",
        "Array `892385` was canceled because the frozen PyTorch build does not support Tesla P100 compute capability sm_60. This is a real compatibility failure, not a harmless warning.", "",
        "- accepted result rows: `0`",
        "- result-like files in this evidence directory: `0`",
        "- tasks 0 and 1 produced launch/error logs; tasks 2-7 never started.", "",
        "| task | stdout sha256 | stderr sha256 | verdict |", "|---:|---|---|---|",
    ]
    for task in excluded["tasks_with_logs"]:
        ex_lines.append(f"| {task['array_task_id']} | `{task['stdout_sha256']}` | `{task['stderr_sha256']}` | {task['stderr_status']} |")
    ex_lines.extend(["", "## Internal Validation Review", "", "- No P100 row enters any accepted or final CSV.", "- The incompatibility warning remains classified as a real failure."])
    (EXCLUDED_DIR / "README.md").write_text("\n".join(ex_lines) + "\n")


def find_ci(rows: list[dict[str, Any]], *, metric: str, estimand: str, name_key: str, name: str, dataset: str = "ALL") -> dict[str, Any]:
    matches = [row for row in rows if row["metric"] == metric and row["estimand"] == estimand and row[name_key] == name and row["dataset"] == dataset]
    if len(matches) != 1:
        raise ValueError(f"CI lookup expected one row, got {len(matches)}")
    return matches[0]


def write_digest(method_rows: list[dict[str, Any]], contrast_rows: list[dict[str, Any]], harm: list[dict[str, Any]], stability: list[dict[str, Any]], final_sha: str) -> None:
    lines = [
        "# Official SPDIM W1 Repaired-Split Three-Seed Result Digest", "",
        "Label: Official SPDIM W1 repaired-split three-source-seed same-split baseline.", "",
        "- rows: `1380/1380`",
        f"- final_result_sha256: `{final_sha}`",
        f"- bootstrap: `{BOOTSTRAP_REPLICATES}` dataset-stratified subject-cluster replicates, seed `{BOOTSTRAP_SEED}`",
        "- seeds are averaged within dataset x target subject x method before aggregation.", "",
        "## Method bAcc", "",
        "| scope | dataset | source_only_tsmnet | rct | spdim_geodesic | spdim_bias |", "|---|---|---:|---:|---:|---:|",
    ]
    for estimand, dataset in [("per_dataset", value) for value in DATASETS] + [("subject_weighted", "ALL"), ("dataset_macro", "ALL")]:
        values = []
        for method in METHODS:
            row = find_ci(method_rows, metric="bacc", estimand=estimand, name_key="method", name=method, dataset=dataset)
            values.append(f"{row['estimate']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        lines.append(f"| {estimand} | {dataset} | " + " | ".join(values) + " |")
    lines.extend(["", "## Paired bAcc Contrasts", "", "| contrast | subject-weighted estimate [95% CI] | dataset-macro estimate [95% CI] |", "|---|---:|---:|"])
    for contrast in CONTRASTS:
        sw = find_ci(contrast_rows, metric="bacc", estimand="subject_weighted", name_key="contrast", name=contrast)
        dm = find_ci(contrast_rows, metric="bacc", estimand="dataset_macro", name_key="contrast", name=contrast)
        lines.append(f"| {contrast} | {sw['estimate']:+.4f} [{sw['ci_low']:+.4f}, {sw['ci_high']:+.4f}] | {dm['estimate']:+.4f} [{dm['ci_low']:+.4f}, {dm['ci_high']:+.4f}] |")
    lines.extend(["", "## Subject-Weighted Harm Rate at Delta < 0", "", "| contrast | count / 115 | rate |", "|---|---:|---:|"])
    for contrast in CONTRASTS:
        row = next(item for item in harm if item["contrast"] == contrast and item["scope"] == "subject_weighted" and item["threshold"] == 0.0)
        lines.append(f"| {contrast} | {row['harm_count']} / {row['n_clusters']} | {row['harm_rate']:.4f} |")
    lines.extend(["", "## Seed Stability (bAcc)", "", "| dataset | method | seed 0 | seed 1 | seed 2 | range | population SD | ranks 0/1/2 |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for row in stability:
        if row["metric"] != "bacc":
            continue
        lines.append(f"| {row['dataset']} | {row['method']} | {row['seed0_mean']:.4f} | {row['seed1_mean']:.4f} | {row['seed2_mean']:.4f} | {row['range']:.4f} | {row['population_std']:.4f} | {row['seed0_rank']}/{row['seed1_rank']}/{row['seed2_rank']} |")
    lines.extend([
        "", "## Interpretation", "",
        "SPDIM-specific claims are determined by the paired `spdim_geodesic_minus_rct` and `spdim_bias_minus_rct` rows above. No equivalence or noninferiority claim is made, and no post-hoc margin is introduced.", "",
        "## Internal Validation Review", "",
        "- Seed rows are not treated as independent biological units.",
        "- P8 seed-0 bytes remain unchanged and seed-0 rows are copied, not recomputed.",
        "- The P100 launch contributes zero rows; all accepted shards pass real-failure log screening.",
    ])
    THREE_DIGEST.write_text("\n".join(lines) + "\n")


def write_audit(seeds12: dict[str, Any], final_validation: dict[str, Any], excluded: dict[str, Any], seed0_sha: str, final_sha: str) -> None:
    lines = [
        "# SPDIM W1 Repaired Seeds 1/2 Execution and Three-Seed Merge Audit", "",
        f"- status: `{seeds12['status']}`",
        "- final label: Official SPDIM W1 repaired-split three-source-seed same-split baseline.",
        f"- accepted array: `{ARRAY_JOB_ID}`",
        f"- excluded zero-result array: `{EXCLUDED_ARRAY_JOB_ID}` on P100",
        f"- final_squeue_absent: `{seeds12['monitoring']['final_squeue_absent']}`", "",
        "## Submission and Partition History", "",
        "The committed launcher default was `#SBATCH --partition=H100,L40S`. The effective accepted submission was:", "",
        "```bash",
        "sbatch --partition=A40 h2cmi/results/review_completion/slurm/spdim_w1_repaired_seeds12_8task.slurm",
        "```", "",
        "The pending array was then expanded in place, without cancellation or resubmission:", "",
        "```bash",
        "scontrol update JobId=892389 Partition=H100,A100,L40S,A40",
        "scontrol update JobId=892389 Partition=H100,A100,L40S,A40,V100",
        "```", "",
        f"- launch commit: `{LAUNCH_COMMIT}`",
        "- working directory: `/home/infres/yinwang/CMI_AAAI_spdim_clean_a8b9368`",
        "- array and concurrency: `0-7%4`",
        "- resource-only override: runner, controller, config, manifest, methods, seeds, hyperparameters, and target specs were unchanged.", "",
        "## Accepted Result-Carrying Tasks", "",
        "| task | job | seed | shard | partition | GPU | capability | rows | stdout | stderr |", "|---:|---:|---:|---|---|---|---|---:|---|---|",
    ]
    for shard in seeds12["shards"]:
        lines.append(f"| {shard['array_task_id']} | {shard['job_id']} | {shard['source_seed']} | {shard['shard_id']} | {shard['partition']} | {shard['runtime_environment']['device_name']} | {shard['compute_capability']}* | {shard['row_count']} | {shard['stdout']['status']} | {shard['stderr']['status']} |")
    lines.extend([
        "", "*Compute capability is mapped from the recorded GPU model because the runtime controller did not emit `torch.cuda.get_device_capability`; compatibility is additionally gated by absence of unsupported-architecture, kernel-image, and CUDA-initialization failures.", "",
        "All tasks recorded the `icml` executable, Python 3.9.25, PyTorch 2.8.0+cu128, CUDA 12.8, MOABB 1.2.0, MNE 1.8.0, frozen checksums, seed, shard, and expected rows.", "",
        "## Excluded P100 Launch", "",
        f"Array `{EXCLUDED_ARRAY_JOB_ID}` was canceled after its task logs reported unsupported Tesla P100 sm_60. It produced `{excluded['accepted_result_rows']}` accepted rows. This is a real compatibility failure and is not classified as harmless.", "",
        "| task | stderr verdict | stdout sha256 | stderr sha256 |", "|---:|---|---|---|",
    ])
    for task in excluded["tasks_with_logs"]:
        lines.append(f"| {task['array_task_id']} | {task['stderr_status']} | `{task['stdout_sha256']}` | `{task['stderr_sha256']}` |")
    lines.extend([
        "", "## Row and Merge Gates", "",
        f"- seeds 1/2 rows: `{seeds12['validation']['row_count']}/920`",
        f"- seed 1 rows: `{seeds12['validation']['seed_rows']['1']}`",
        f"- seed 2 rows: `{seeds12['validation']['seed_rows']['2']}`",
        f"- seeds 1/2 result SHA-256: `{seeds12['result_csv_sha256']}`",
        f"- seed-0 source SHA-256: `{seed0_sha}`",
        f"- final rows: `{final_validation['row_count']}/1380`",
        f"- final result SHA-256: `{final_sha}`",
        f"- final duplicate keys: `{final_validation['duplicate_key_count']}`",
        f"- single-class adaptation rows: `{final_validation['single_class_adapt_rows']}`",
        f"- single-class evaluation rows: `{final_validation['single_class_eval_rows']}`",
        f"- adapt/eval overlap failures: `{final_validation['adapt_eval_disjoint_failures']}`",
        f"- missing prediction hashes: `{final_validation['prediction_hash_missing_rows']}`",
        f"- missing logits hashes: `{final_validation['logits_hash_missing_rows']}`", "",
        "P8 seed-0 CSV bytes were not modified. Its 460 parsed rows match the seed-0 subset of the deterministic final merge exactly; no seed-0 model was rerun.", "",
        "## Aggregation Gate", "",
        f"- bootstrap replicates: `{BOOTSTRAP_REPLICATES}`",
        f"- bootstrap seed: `{BOOTSTRAP_SEED}`",
        "- cluster unit: dataset x target subject",
        "- dataset-stratified resampling preserves 9/52/54 target counts",
        "- seeds averaged before aggregation and bootstrap",
        "- methods and paired contrasts preserved within sampled subjects", "",
        "## Internal Validation Review", "",
        "- Eight accepted shard keys cover each seed/subject/method exactly once.",
        "- P100 logs are retained separately and no P100 partial row is accepted.",
        "- Resource partition changes are fully disclosed and do not alter scientific configuration.",
        "- The final label is enabled only because every execution, row, seed-preservation, and inference gate passes.",
        "- No extra seeds, methods, TeX edits, H2CMI reruns, geometry stress, or orthogonal-score work is included.",
    ])
    SEEDS12_AUDIT.write_text("\n".join(lines) + "\n")


def append_command_log(seeds12_sha: str, final_sha: str) -> None:
    marker = "Per PM P9B/P9C, completed the official SPDIM repaired-split three-source-seed baseline"
    text = COMMAND_LOG.read_text()
    if marker in text:
        return
    entry = f"""
- {marker}. Accepted array `892389` used the committed `0-7%4` launcher with
  initial `--partition=A40`, then user-approved in-place partition expansion to
  `H100,A100,L40S,A40,V100`; scientific configuration was unchanged. Eight
  result-carrying tasks produced `920/920` seeds-1/2 rows. P100 array `892385`
  remains a zero-result excluded launch because sm_60 is unsupported. Final
  monitoring used `squeue` only and array `892389` was absent. The seeds-1/2
  SHA-256 is `{seeds12_sha}`; the deterministic 1380-row three-seed SHA-256 is
  `{final_sha}`. Seed-0 source bytes retained SHA-256 `{SEED0_SHA256}`. The
  10,000-replicate paired subject-cluster bootstrap used seed `{BOOTSTRAP_SEED}`
  after averaging seeds within dataset x target subject x method. No extra
  seed, method, TeX, H2CMI, geometry-stress, or orthogonal-score work was run.
"""
    COMMAND_LOG.write_text(text.rstrip() + "\n\n" + entry.lstrip())


def main() -> None:
    monitoring = final_squeue_gate()
    shard_rows, shard_meta, fieldnames, shard_errors = load_shards()
    excluded = excluded_p100_evidence()
    seeds12_validation = validate_result_rows(shard_rows, {1, 2})
    current_hashes = {
        "runner_sha256": sha256_file(ROOT / "h2cmi" / "run_spdim_w1_repaired_seed0.py"),
        "controller_sha256": sha256_file(ROOT / "h2cmi" / "run_spdim_w1_repaired_seeds12.py"),
        "config_sha256": sha256_file(ROOT / "h2cmi" / "config.py"),
        "manifest_file_sha256": sha256_file(OUT_DIR / "w1_repaired_split_manifest.csv"),
        "launcher_sha256": sha256_file(OUT_DIR / "slurm" / "spdim_w1_repaired_seeds12_8task.slurm"),
    }
    freeze_pass = current_hashes == {
        "runner_sha256": P8_RUNNER_SHA256,
        "controller_sha256": P9_CONTROLLER_SHA256,
        "config_sha256": CONFIG_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "launcher_sha256": LAUNCHER_SHA256,
    }
    premerge_pass = (
        monitoring["final_squeue_absent"]
        and not shard_errors
        and all(shard["shard_pass"] for shard in shard_meta)
        and excluded["gate_pass"]
        and seeds12_validation["validation_pass"]
        and freeze_pass
    )
    if not premerge_pass:
        write_json(FAILURE_TRACE, {
            "status": "blocked", "monitoring": monitoring, "shard_errors": shard_errors,
            "shards": shard_meta, "excluded": excluded, "validation": seeds12_validation,
            "current_hashes": current_hashes, "freeze_pass": freeze_pass,
        })
        raise SystemExit(2)
    if FAILURE_TRACE.exists():
        FAILURE_TRACE.unlink()

    clean_shard_rows = [{key: value for key, value in row.items() if key != "_source_shard"} for row in sorted_rows(shard_rows)]
    write_csv(SEEDS12_RESULTS, clean_shard_rows, fieldnames)
    seeds12_sha = sha256_file(SEEDS12_RESULTS)

    seed0_sha = sha256_file(SEED0_RESULTS)
    if seed0_sha != SEED0_SHA256:
        raise RuntimeError(f"seed-0 checksum mismatch: {seed0_sha}")
    seed0_rows, seed0_fields = read_csv(SEED0_RESULTS)
    if seed0_fields != fieldnames:
        raise RuntimeError("seed-0 and seeds-1/2 CSV headers differ")
    final_rows = sorted_rows(seed0_rows + clean_shard_rows)
    write_csv(THREE_RESULTS, final_rows, fieldnames)
    final_sha = sha256_file(THREE_RESULTS)
    final_validation = validate_result_rows(final_rows, {0, 1, 2})
    final_seed0_subset = [row for row in final_rows if int(row["source_seed"]) == 0]
    seed0_rows_preserved = sorted_rows(seed0_rows) == final_seed0_subset
    final_validation["seed0_source_sha256"] = seed0_sha
    final_validation["seed0_source_checksum_preserved"] = seed0_sha == SEED0_SHA256
    final_validation["seed0_rows_preserved_exactly"] = seed0_rows_preserved
    final_validation["validation_pass"] = bool(final_validation["validation_pass"] and seed0_rows_preserved)
    if not final_validation["validation_pass"]:
        write_json(FAILURE_TRACE, {"status": "blocked", "final_validation": final_validation})
        raise SystemExit(2)

    units = seed_averaged_units(final_rows)
    method_rows, contrast_rows = bootstrap_tables(units)
    harms = harm_rows(units)
    stability = stability_rows(final_rows)
    write_csv(METHOD_CI, method_rows, list(method_rows[0].keys()))
    write_csv(CONTRAST_CI, contrast_rows, list(contrast_rows[0].keys()))
    write_csv(HARM_CSV, harms, list(harms[0].keys()))
    write_csv(STABILITY_CSV, stability, list(stability[0].keys()))

    write_evidence_readmes(shard_meta, excluded)
    seeds12_summary = {
        "status": "pass",
        "label": "P9 accepted repaired-split SPDIM seeds 1/2 execution",
        "launch_commit": LAUNCH_COMMIT,
        "accepted_array_job_id": ARRAY_JOB_ID,
        "accepted_job_ids": [shard["job_id"] for shard in shard_meta],
        "submission": {
            "committed_launcher_default_partitions": ["H100", "L40S"],
            "initial_effective_command": "sbatch --partition=A40 h2cmi/results/review_completion/slurm/spdim_w1_repaired_seeds12_8task.slurm",
            "partition_updates": [
                "scontrol update JobId=892389 Partition=H100,A100,L40S,A40",
                "scontrol update JobId=892389 Partition=H100,A100,L40S,A40,V100",
            ],
            "final_eligible_partitions": ["H100", "A100", "L40S", "A40", "V100"],
            "array": "0-7%4",
            "working_directory": "/home/infres/yinwang/CMI_AAAI_spdim_clean_a8b9368",
            "scientific_configuration_changed_by_resource_override": False,
        },
        "monitoring": monitoring,
        "excluded_launches": [excluded],
        "current_frozen_file_hashes": current_hashes,
        "freeze_pass": freeze_pass,
        "validation": seeds12_validation,
        "shards": shard_meta,
        "result_csv": str(SEEDS12_RESULTS.relative_to(ROOT)),
        "result_csv_sha256": seeds12_sha,
    }
    write_audit(seeds12_summary, final_validation, excluded, seed0_sha, final_sha)
    write_digest(method_rows, contrast_rows, harms, stability, final_sha)
    seeds12_summary["artifacts"] = {
        "results_csv": {"path": str(SEEDS12_RESULTS.relative_to(ROOT)), "sha256": seeds12_sha},
        "audit_md": {"path": str(SEEDS12_AUDIT.relative_to(ROOT)), "sha256": sha256_file(SEEDS12_AUDIT)},
        "accepted_shard_readme": {"path": str((SHARD_DIR / "README.md").relative_to(ROOT)), "sha256": sha256_file(SHARD_DIR / "README.md")},
        "excluded_launch_readme": {"path": str((EXCLUDED_DIR / "README.md").relative_to(ROOT)), "sha256": sha256_file(EXCLUDED_DIR / "README.md")},
    }
    write_json(SEEDS12_SUMMARY, seeds12_summary)

    three_summary = {
        "status": "pass",
        "label": "Official SPDIM W1 repaired-split three-source-seed same-split baseline.",
        "source_seeds": [0, 1, 2],
        "datasets": DATASETS,
        "methods": METHODS,
        "split_family": "class_stratified_half",
        "manifest_hash": MANIFEST_HASH,
        "validation": final_validation,
        "aggregation": {
            "seed_average_unit": "dataset_x_target_subject_x_method",
            "biological_cluster_unit": "dataset_x_target_subject",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_interval": "percentile_95",
            "dataset_stratified": True,
            "paired_methods_and_contrasts": True,
            "subject_weighted_estimand": True,
            "dataset_macro_estimand": True,
        },
        "result_csv": str(THREE_RESULTS.relative_to(ROOT)),
        "result_csv_sha256": final_sha,
        "seed0_preservation": {
            "source_path": str(SEED0_RESULTS.relative_to(ROOT)),
            "source_sha256": seed0_sha,
            "expected_sha256": SEED0_SHA256,
            "source_bytes_unchanged": seed0_sha == SEED0_SHA256,
            "parsed_rows_preserved_exactly_in_final_merge": seed0_rows_preserved,
            "seed0_rows_recomputed": False,
        },
        "method_ci": method_rows,
        "contrast_ci": contrast_rows,
        "harm": harms,
        "seed_stability": stability,
        "execution_summary": str(SEEDS12_SUMMARY.relative_to(ROOT)),
        "artifacts": {
            "results_csv": {"path": str(THREE_RESULTS.relative_to(ROOT)), "sha256": final_sha},
            "seeds12_results_csv": {"path": str(SEEDS12_RESULTS.relative_to(ROOT)), "sha256": seeds12_sha},
            "seeds12_summary_json": {"path": str(SEEDS12_SUMMARY.relative_to(ROOT)), "sha256": sha256_file(SEEDS12_SUMMARY)},
            "result_digest_md": {"path": str(THREE_DIGEST.relative_to(ROOT)), "sha256": sha256_file(THREE_DIGEST)},
            "red_team_review_md": {"path": str(RED_TEAM_REVIEW.relative_to(ROOT)), "sha256": sha256_file(RED_TEAM_REVIEW)},
            "method_ci_csv": {"path": str(METHOD_CI.relative_to(ROOT)), "sha256": sha256_file(METHOD_CI)},
            "contrast_ci_csv": {"path": str(CONTRAST_CI.relative_to(ROOT)), "sha256": sha256_file(CONTRAST_CI)},
            "harm_csv": {"path": str(HARM_CSV.relative_to(ROOT)), "sha256": sha256_file(HARM_CSV)},
            "seed_stability_csv": {"path": str(STABILITY_CSV.relative_to(ROOT)), "sha256": sha256_file(STABILITY_CSV)},
        },
    }
    write_json(THREE_SUMMARY, three_summary)
    append_command_log(seeds12_sha, final_sha)
    print(json.dumps({
        "status": "pass",
        "seeds12_rows": seeds12_validation["row_count"],
        "seeds12_sha256": seeds12_sha,
        "final_rows": final_validation["row_count"],
        "final_sha256": final_sha,
        "final_squeue_absent": monitoring["final_squeue_absent"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
