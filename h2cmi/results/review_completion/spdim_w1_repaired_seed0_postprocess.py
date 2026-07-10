"""Merge and validate the four clean P8 SPDIM repaired-split shards.

The postprocessor is intentionally standard-library only. It treats the shard
CSVs, shard summaries, and Slurm logs as immutable inputs, applies the final
squeue/artifact gates, and writes the official seed-0 result packet.
"""
from __future__ import annotations

import csv
import getpass
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "h2cmi" / "results" / "review_completion"
SHARD_DIR = OUT_DIR / "spdim_w1_repaired_seed0_shards"
COMMAND_LOG = OUT_DIR / "COMMAND_LOG.md"
OLD_RESULTS = OUT_DIR / "spdim_w1_seed0_results.csv"

LABEL = "W1 repaired-split seed-0 official SPDIM expansion, not full three-seed baseline."
LAUNCH_COMMIT = "763e11c4412938017f0a7b1be3cfbe9e40ec3d41"
MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
MANIFEST_FILE_SHA256 = "e9ebe6e9421bdcf10f8a952623285cec0842f5cb6b868e8147f13dde23e8a712"
RUNNER_SHA256 = "946b28b93f0ddbce395ade7c6a13d30b20f368fe7a1ae22fbefa01f291e82be8"
CONFIG_SHA256 = "6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b"
EXTERNAL_SHA = "1b0de0ccd4c48a4ff28f087b866a0b671b029c39"
ARRAY_JOB_ID = "891456"
LAUNCHER_RELATIVE_PATH = "h2cmi/results/review_completion/slurm/spdim_w1_repaired_seed0_4shard.slurm"
LAUNCHER_PATH = ROOT / LAUNCHER_RELATIVE_PATH
LAUNCHER_PYTHON = "/home/infres/yinwang/anaconda3/envs/icml/bin/python"

DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
METHODS = ["source_only_tsmnet", "rct", "spdim_geodesic", "spdim_bias"]
EXPECTED_DATASET_ROWS = {"BNCI2014_001": 36, "Cho2017": 208, "Lee2019_MI": 216}
EXPECTED_SUBJECTS = {
    "BNCI2014_001": set(range(1, 10)),
    "Cho2017": set(range(1, 53)),
    "Lee2019_MI": set(range(1, 55)),
}
SHARDS = [
    {
        "id": "shard0",
        "job_id": "891457",
        "array_task_id": 0,
        "target_spec": "BNCI2014_001=1-9;Cho2017=1-20",
        "expected_rows": 116,
        "expected_dataset_rows": {"BNCI2014_001": 36, "Cho2017": 80},
    },
    {
        "id": "shard1",
        "job_id": "891458",
        "array_task_id": 1,
        "target_spec": "Cho2017=21-49",
        "expected_rows": 116,
        "expected_dataset_rows": {"Cho2017": 116},
    },
    {
        "id": "shard2",
        "job_id": "891459",
        "array_task_id": 2,
        "target_spec": "Cho2017=50-52;Lee2019_MI=1-26",
        "expected_rows": 116,
        "expected_dataset_rows": {"Cho2017": 12, "Lee2019_MI": 104},
    },
    {
        "id": "shard3",
        "job_id": "891456",
        "array_task_id": 3,
        "target_spec": "Lee2019_MI=27-54",
        "expected_rows": 112,
        "expected_dataset_rows": {"Lee2019_MI": 112},
    },
]
EXCLUDED_JOBS = {
    "891435": {
        "role": "canceled_monolithic_partial",
        "rows": 56,
        "result_sha256": "ed434d8927b2ee73d6839ca1ba0c724de31118e2ed1e5a8230dc362adae341dc",
        "reason": "Canceled after user-approved four-way sharding; all partial rows are excluded.",
    }
}

RESULT_PATH = OUT_DIR / "spdim_w1_repaired_seed0_results.csv"
SUMMARY_PATH = OUT_DIR / "spdim_w1_repaired_seed0_summary.json"
AUDIT_PATH = OUT_DIR / "spdim_w1_repaired_seed0_audit.md"
DIGEST_PATH = OUT_DIR / "spdim_w1_repaired_seed0_result_digest.md"
LEGACY_PATH = OUT_DIR / "spdim_w1_repaired_seed0_legacy_compare.md"
FAILURE_PATH = OUT_DIR / "spdim_w1_repaired_seed0_failure_trace.txt"
README_PATH = SHARD_DIR / "README.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(sum(clean) / len(clean)) if clean else None


def line_value(text: str, key: str) -> str | None:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


def final_squeue_gate() -> dict[str, Any]:
    user = getpass.getuser()
    command = ["squeue", "-h", "-u", user, "-o", "%i|%A|%a|%t|%j"]
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"squeue failed with code {proc.returncode}: {proc.stderr.strip()}")
    accepted = {shard["job_id"] for shard in SHARDS}
    matching = []
    for line in proc.stdout.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        job_id, array_job_id, array_task_id, state, name = parts
        if job_id in accepted or array_job_id == ARRAY_JOB_ID or job_id.startswith(f"{ARRAY_JOB_ID}_"):
            matching.append({
                "job_id": job_id,
                "array_job_id": array_job_id,
                "array_task_id": array_task_id,
                "state": state,
                "name": name,
            })
    return {
        "command": " ".join(command),
        "user": user,
        "accepted_job_ids": sorted(accepted),
        "array_job_id": ARRAY_JOB_ID,
        "matching_queue_rows": matching,
        "final_squeue_absent": not matching,
        "stderr": proc.stderr.strip(),
    }


def stderr_status(path: Path) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    allowed = {
        "Trials demeaned and stacked with zero buffer to create continuous data -- edge effects present":
            "moabb_zero_buffer_edge_warning",
        "Matplotlib is building the font cache; this may take a moment.":
            "matplotlib_font_cache_notice",
    }
    unexpected = [line for line in lines if line not in allowed]
    patterns = sorted({allowed[line] for line in lines if line in allowed})
    if not path.exists():
        status = "missing"
    elif not lines:
        status = "empty"
    elif not unexpected:
        status = "known_harmless_warnings_only"
    else:
        status = "unexpected_stderr"
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "lines": len(lines),
        "status": status,
        "patterns": patterns,
        "unexpected_lines": unexpected[:20],
        "sha256": sha256_file(path) if path.exists() else "",
    }


def stdout_status(path: Path, shard: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    required_values = {
        "job_id": shard["job_id"],
        "array_task_id": str(shard["array_task_id"]),
        "launch_commit": LAUNCH_COMMIT,
        "manifest_hash": MANIFEST_HASH,
        "target_spec": shard["target_spec"],
        "expected_rows": str(shard["expected_rows"]),
        "runner_sha256": RUNNER_SHA256,
        "config_sha256": CONFIG_SHA256,
        "manifest_csv_sha256": MANIFEST_FILE_SHA256,
        "external_sha": EXTERNAL_SHA,
    }
    observed_values = {key: line_value(text, key) for key in required_values}
    mismatches = {
        key: {"expected": expected, "observed": observed_values[key]}
        for key, expected in required_values.items()
        if observed_values[key] != expected
    }
    clean_block = "repo_status_porcelain_begin\nrepo_status_porcelain_end" in text
    status = (
        "exists_nonempty_clean_launch_header"
        if path.exists() and text and not mismatches and clean_block
        else "invalid_stdout"
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "lines": len(text.splitlines()),
        "status": status,
        "clean_porcelain_block": clean_block,
        "required_header_values": required_values,
        "observed_header_values": observed_values,
        "header_mismatches": mismatches,
        "sha256": sha256_file(path) if path.exists() else "",
    }


def load_shards() -> tuple[list[dict[str, str]], list[dict[str, Any]], list[str]]:
    all_rows: list[dict[str, str]] = []
    shard_meta: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    errors: list[str] = []
    for shard in SHARDS:
        shard_id = shard["id"]
        csv_path = SHARD_DIR / f"{shard_id}_results.csv"
        summary_path = SHARD_DIR / f"{shard_id}_summary.json"
        stdout_path = SHARD_DIR / f"{shard_id}.stdout.txt"
        stderr_path = SHARD_DIR / f"{shard_id}.stderr.txt"
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{csv_path} has no CSV header")
            current_fields = list(reader.fieldnames)
            rows = list(reader)
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            errors.append(f"{shard_id}: CSV header differs from shard0")
        summary = json.loads(summary_path.read_text())
        result_sha = sha256_file(csv_path)
        stdout = stdout_status(stdout_path, shard)
        stderr = stderr_status(stderr_path)
        launch = summary.get("launch_provenance", {})
        validation = summary.get("validation", {})
        checks = {
            "row_count": len(rows) == shard["expected_rows"],
            "summary_status": summary.get("status") == "pass",
            "summary_validation_pass": validation.get("validation_pass") is True,
            "summary_row_count": validation.get("row_count") == shard["expected_rows"],
            "summary_expected_rows": validation.get("expected_rows_total") == shard["expected_rows"],
            "dataset_rows": validation.get("dataset_rows") == shard["expected_dataset_rows"],
            "result_checksum": summary.get("result_csv_sha256") == result_sha,
            "target_spec": summary.get("target_spec") == shard["target_spec"],
            "shard_id": summary.get("shard_id") == shard_id,
            "launch_commit": launch.get("launch_commit") == LAUNCH_COMMIT,
            "clean_worktree": launch.get("clean_worktree") is True,
            "empty_launch_status": launch.get("git_status_porcelain") == "",
            "runner_dirty_disallowed": launch.get("runner_dirty_allowed") is False,
            "runner_checksum": launch.get("runner_file_sha256") == RUNNER_SHA256,
            "config_checksum": launch.get("config_sha256") == CONFIG_SHA256,
            "manifest_file_checksum": launch.get("manifest_file_sha256") == MANIFEST_FILE_SHA256,
            "manifest_hash": launch.get("manifest_hash") == MANIFEST_HASH,
            "manifest_hash_matches_p7": launch.get("manifest_hash_matches_p7") is True,
            "external_commit": launch.get("external_spdim_commit") == EXTERNAL_SHA,
            "slurm_job_id": launch.get("slurm_job_id") == shard["job_id"],
            "stdout": stdout["status"] == "exists_nonempty_clean_launch_header",
            "stderr": stderr["status"] in {"empty", "known_harmless_warnings_only"},
        }
        if not all(checks.values()):
            errors.append(f"{shard_id}: failed checks {sorted(key for key, value in checks.items() if not value)}")
        for row in rows:
            row["_source_shard"] = shard_id
        all_rows.extend(rows)
        shard_meta.append({
            "shard_id": shard_id,
            "job_id": shard["job_id"],
            "array_task_id": shard["array_task_id"],
            "target_spec": shard["target_spec"],
            "expected_rows": shard["expected_rows"],
            "row_count": len(rows),
            "dataset_rows": dict(sorted(Counter(row["dataset"] for row in rows).items())),
            "result_csv": str(csv_path.relative_to(ROOT)),
            "result_csv_sha256": result_sha,
            "summary_json": str(summary_path.relative_to(ROOT)),
            "summary_json_sha256": sha256_file(summary_path),
            "launch_provenance": launch,
            "stdout": stdout,
            "stderr": stderr,
            "checks": checks,
            "shard_pass": all(checks.values()),
        })
    if fieldnames is None:
        raise RuntimeError("no shard CSVs loaded")
    return all_rows, shard_meta, fieldnames


def validate_rows(rows: list[dict[str, str]], squeue_gate: dict[str, Any], shard_meta: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [(row["dataset"], row["target_subject"], row["source_seed"], row["method"]) for row in rows]
    dataset_rows = Counter(row["dataset"] for row in rows)
    method_rows = Counter(row["method"] for row in rows)
    status_rows = Counter(row["status"] for row in rows)
    subjects = {
        dataset: {int(row["target_subject"]) for row in rows if row["dataset"] == dataset}
        for dataset in DATASETS
    }
    eval_counts = [json.loads(row["class_counts_eval"]) for row in rows]
    adapt_counts = [json.loads(row["class_counts_adapt"]) for row in rows]
    launcher_at_launch = subprocess.check_output(
        ["git", "show", f"{LAUNCH_COMMIT}:{LAUNCHER_RELATIVE_PATH}"],
        cwd=ROOT,
    )
    current_checksums = {
        "runner_sha256": sha256_file(ROOT / "h2cmi" / "run_spdim_w1_repaired_seed0.py"),
        "config_sha256": sha256_file(ROOT / "h2cmi" / "config.py"),
        "manifest_file_sha256": sha256_file(OUT_DIR / "w1_repaired_split_manifest.csv"),
        "launcher_sha256": sha256_file(LAUNCHER_PATH),
        "launcher_at_launch_sha256": sha256_bytes(launcher_at_launch),
    }
    recorded_environment_names = sorted({
        meta["launch_provenance"].get("environment_name", "") for meta in shard_meta
    })
    launcher_text = LAUNCHER_PATH.read_text()
    checks = {
        "final_squeue_absent": squeue_gate["final_squeue_absent"],
        "shard_validation_pass": all(meta["shard_pass"] for meta in shard_meta),
        "row_count": len(rows) == 460,
        "dataset_rows": dict(dataset_rows) == EXPECTED_DATASET_ROWS,
        "method_rows": dict(method_rows) == {method: 115 for method in METHODS},
        "status_rows": dict(status_rows) == {"ok": 460},
        "unique_keys": len(keys) == len(set(keys)) == 460,
        "subject_coverage": all(subjects[dataset] == EXPECTED_SUBJECTS[dataset] for dataset in DATASETS),
        "source_seed_zero_only": {row["source_seed"] for row in rows} == {"0"},
        "split_family": {row["split_family"] for row in rows} == {"class_stratified_half"},
        "all_eval_both_classes": all(len(counts) == 2 and min(counts) > 0 for counts in eval_counts),
        "all_adapt_both_classes": all(len(counts) == 2 and min(counts) > 0 for counts in adapt_counts),
        "all_eval_class_balanced": all(counts[0] == counts[1] for counts in eval_counts),
        "all_adapt_eval_disjoint": all(row["adapt_eval_disjoint"] == "True" for row in rows),
        "acc_equals_bacc": all(abs(float(row["acc"]) - float(row["bacc"])) <= 1e-12 for row in rows),
        "prediction_hash_complete": all(bool(row["prediction_hash"]) for row in rows),
        "logits_hash_complete": all(bool(row["logits_hash"]) for row in rows),
        "split_hash_complete": all(bool(row["split_hash"]) for row in rows),
        "trial_index_hashes_complete": all(
            row["source_idx_sha256"] and row["adapt_idx_sha256"] and row["eval_idx_sha256"]
            for row in rows
        ),
        "source_model_hash_complete": all(bool(row["source_model_sha256"]) for row in rows),
        "failure_reasons_empty": all(not row["failure_reason"] for row in rows),
        "manifest_hash_matches_p7": {row["manifest_hash"] for row in rows} == {MANIFEST_HASH},
        "runner_commit_consistent": {row["runner_commit"] for row in rows} == {LAUNCH_COMMIT},
        "official_commit_consistent": {row["official_sha"] for row in rows} == {EXTERNAL_SHA},
        "no_target_label_leakage": all(row["target_label_leakage_detected"] == "False" for row in rows),
        "no_target_performance_selection": all(
            row["method_selection_uses_target_performance"] == "False" for row in rows
        ),
        "no_official_pretrained_weights": all(row["official_pretrained_weight_used"] == "False" for row in rows),
        "no_third_party_vendoring": all(row["third_party_vendored"] == "False" for row in rows),
        "launched_runner_matches_result_commit": current_checksums["runner_sha256"] == RUNNER_SHA256,
        "launched_config_matches_result_commit": current_checksums["config_sha256"] == CONFIG_SHA256,
        "manifest_file_matches_launch": current_checksums["manifest_file_sha256"] == MANIFEST_FILE_SHA256,
        "environment_name_recorded": recorded_environment_names == ["base"],
        "launcher_matches_launch_commit": (
            current_checksums["launcher_sha256"] == current_checksums["launcher_at_launch_sha256"]
        ),
        "launcher_pins_icml_python": f"PY={LAUNCHER_PYTHON}" in launcher_text,
    }
    return {
        "validation_pass": all(checks.values()),
        "checks": checks,
        "row_count": len(rows),
        "expected_rows_total": 460,
        "dataset_rows": dict(sorted(dataset_rows.items())),
        "expected_rows_by_dataset": EXPECTED_DATASET_ROWS,
        "method_rows": dict(sorted(method_rows.items())),
        "status_rows": dict(sorted(status_rows.items())),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "subjects_by_dataset": {dataset: len(subjects[dataset]) for dataset in DATASETS},
        "single_class_eval_rows": sum(min(counts) <= 0 for counts in eval_counts),
        "single_class_adapt_rows": sum(min(counts) <= 0 for counts in adapt_counts),
        "adapt_eval_disjoint_failures": sum(row["adapt_eval_disjoint"] != "True" for row in rows),
        "acc_bacc_unequal_rows": sum(
            abs(float(row["acc"]) - float(row["bacc"])) > 1e-12 for row in rows
        ),
        "prediction_hash_missing_rows": sum(not row["prediction_hash"] for row in rows),
        "logits_hash_missing_rows": sum(not row["logits_hash"] for row in rows),
        "target_label_leakage_detected": any(row["target_label_leakage_detected"] != "False" for row in rows),
        "target_performance_method_selection_detected": any(
            row["method_selection_uses_target_performance"] != "False" for row in rows
        ),
        "pretrained_weight_detected": any(row["official_pretrained_weight_used"] != "False" for row in rows),
        "vendoring_detected": any(row["third_party_vendored"] != "False" for row in rows),
        "current_file_checksums": current_checksums,
        "recorded_environment_names": recorded_environment_names,
    }


def write_results(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    dataset_order = {name: index for index, name in enumerate(DATASETS)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    rows = sorted(
        rows,
        key=lambda row: (
            dataset_order[row["dataset"]],
            int(row["target_subject"]),
            int(row["source_seed"]),
            method_order[row["method"]],
        ),
    )
    with RESULT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(RESULT_PATH)


def result_groups(rows: list[dict[str, str]]) -> tuple[dict, dict, dict]:
    by_dataset: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_dataset[row["dataset"]][row["method"]].append(row)
        by_method[row["method"]].append(row)
        by_key[(row["dataset"], row["target_subject"], row["source_seed"], row["method"])] = row
    return by_dataset, by_method, by_key


def metric_deltas(by_key: dict, method: str, baseline: str, field: str) -> list[float]:
    values = []
    for (dataset, target, seed, row_method), row in by_key.items():
        if row_method != method:
            continue
        base = by_key[(dataset, target, seed, baseline)]
        values.append(float(row[field]) - float(base[field]))
    return values


def summaries(rows: list[dict[str, str]], datasets: list[str]) -> tuple[dict, dict, dict, dict, dict]:
    by_dataset, by_method, by_key = result_groups(rows)
    per_dataset = {}
    for dataset in datasets:
        per_dataset[dataset] = {}
        for method in METHODS:
            method_rows = by_dataset[dataset][method]
            per_dataset[dataset][method] = {
                "n": len(method_rows),
                "mean_acc": mean([float(row["acc"]) for row in method_rows]),
                "mean_bacc": mean([float(row["bacc"]) for row in method_rows]),
                "mean_macro_f1": mean([float(row["macro_f1"]) for row in method_rows]),
            }
    subject_weighted = {}
    dataset_macro = {}
    for method in METHODS:
        method_rows = by_method[method]
        subject_weighted[method] = {
            "n": len(method_rows),
            "mean_acc": mean([float(row["acc"]) for row in method_rows]),
            "mean_bacc": mean([float(row["bacc"]) for row in method_rows]),
        }
        dataset_macro[method] = {
            "mean_acc": mean([per_dataset[dataset][method]["mean_acc"] for dataset in datasets]),
            "mean_bacc": mean([per_dataset[dataset][method]["mean_bacc"] for dataset in datasets]),
        }
    pairs = {
        "rct_minus_source_only_tsmnet": ("rct", "source_only_tsmnet"),
        "spdim_geodesic_minus_source_only_tsmnet": ("spdim_geodesic", "source_only_tsmnet"),
        "spdim_bias_minus_source_only_tsmnet": ("spdim_bias", "source_only_tsmnet"),
        "spdim_geodesic_minus_rct": ("spdim_geodesic", "rct"),
        "spdim_bias_minus_rct": ("spdim_bias", "rct"),
    }
    deltas = {}
    harms = {}
    for name, (method, baseline) in pairs.items():
        acc = metric_deltas(by_key, method, baseline, "acc")
        bacc = metric_deltas(by_key, method, baseline, "bacc")
        harm_count = sum(value < 0 for value in bacc)
        deltas[name] = {
            "n": len(bacc),
            "mean_acc_delta": mean(acc),
            "mean_bacc_delta": mean(bacc),
        }
        harms[name.replace("_minus_", "_vs_")] = {
            "metric": "bacc",
            "n": len(bacc),
            "harm_count": harm_count,
            "harm_rate": float(harm_count / len(bacc)),
        }
    return per_dataset, subject_weighted, dataset_macro, deltas, harms


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_legacy_compare(rows: list[dict[str, str]]) -> dict[str, Any]:
    old_rows = [row for row in read_csv(OLD_RESULTS) if row.get("status") == "ok"]
    old_per, old_sw, old_dm, _, _ = summaries(old_rows, DATASETS)
    new_per, new_sw, new_dm, _, _ = summaries(rows, DATASETS)
    payload = {
        "legacy_source": str(OLD_RESULTS.relative_to(ROOT)),
        "legacy_status": "diagnostic_only_old_contiguous_split",
        "legacy_rows": len(old_rows),
        "repaired_status": "official_seed0_expansion_not_full_baseline",
        "repaired_rows": len(rows),
        "old_per_dataset": old_per,
        "repaired_per_dataset": new_per,
        "old_subject_weighted": old_sw,
        "repaired_subject_weighted": new_sw,
        "old_dataset_macro": old_dm,
        "repaired_dataset_macro": new_dm,
    }
    lines = [
        "# SPDIM W1 Repaired Seed-0 Legacy Compare",
        "",
        "The old P6 SPDIM result used the quarantined contiguous split and remains diagnostic only. The repaired result uses the frozen P7 `class_stratified_half` manifest. This comparison reports magnitude changes; it does not rehabilitate the old split.",
        "",
        "## Per-Dataset Mean bAcc",
        "",
        "| dataset | method | old diagnostic | repaired seed-0 | repaired - old |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            old_value = old_per[dataset][method]["mean_bacc"]
            new_value = new_per[dataset][method]["mean_bacc"]
            lines.append(f"| {dataset} | {method} | {old_value:.6f} | {new_value:.6f} | {new_value - old_value:+.6f} |")
    lines.extend([
        "",
        "## Aggregate Mean bAcc",
        "",
        "| method | old subject-weighted | repaired subject-weighted | old dataset-macro | repaired dataset-macro |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        lines.append(
            f"| {method} | {old_sw[method]['mean_bacc']:.6f} | {new_sw[method]['mean_bacc']:.6f} | "
            f"{old_dm[method]['mean_bacc']:.6f} | {new_dm[method]['mean_bacc']:.6f} |"
        )
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- The split change alters the estimand, especially for Cho2017; differences are not paired treatment effects.",
        "- P6 remains legacy diagnostic only.",
        "- The repaired result is seed 0 only and is not a full three-seed baseline.",
    ])
    LEGACY_PATH.write_text("\n".join(lines) + "\n")
    return payload


def write_digest(summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# SPDIM W1 Repaired-Split Seed-0 Result Digest",
        "",
        f"Label: {LABEL}",
        "",
        f"- status: `{summary['status']}`",
        f"- rows: `{summary['validation']['row_count']}/{summary['validation']['expected_rows_total']}`",
        f"- result_csv_sha256: `{summary['result_csv_sha256']}`",
        f"- prediction_hash_complete: `{summary['validation']['checks']['prediction_hash_complete']}`",
        f"- logits_hash_complete: `{summary['validation']['checks']['logits_hash_complete']}`",
        f"- final_squeue_absent: `{summary['monitoring']['final_squeue_absent']}`",
        "",
        "## Per-Dataset Mean Acc and bAcc",
        "",
        "| dataset | method | n | mean acc | mean bAcc | mean macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            values = summary["per_dataset"][dataset][method]
            lines.append(
                f"| {dataset} | {method} | {values['n']} | {values['mean_acc']:.6f} | "
                f"{values['mean_bacc']:.6f} | {values['mean_macro_f1']:.6f} |"
            )
    lines.extend([
        "",
        "## Overall Subject-Weighted Mean",
        "",
        "| method | n | mean acc | mean bAcc |",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        values = summary["overall_subject_weighted"][method]
        lines.append(f"| {method} | {values['n']} | {values['mean_acc']:.6f} | {values['mean_bacc']:.6f} |")
    lines.extend([
        "",
        "## Dataset-Macro Mean",
        "",
        "| method | mean acc | mean bAcc |",
        "|---|---:|---:|",
    ])
    for method in METHODS:
        values = summary["dataset_macro"][method]
        lines.append(f"| {method} | {values['mean_acc']:.6f} | {values['mean_bacc']:.6f} |")
    lines.extend([
        "",
        "## Deltas",
        "",
        "| contrast | n | mean acc delta | mean bAcc delta |",
        "|---|---:|---:|---:|",
    ])
    for name, values in summary["deltas"].items():
        lines.append(f"| {name} | {values['n']} | {values['mean_acc_delta']:+.6f} | {values['mean_bacc_delta']:+.6f} |")
    lines.extend([
        "",
        "## Harm Counts",
        "",
        "Harm is a strictly negative per-subject bAcc delta.",
        "",
        "| contrast | n | harm count | harm rate |",
        "|---|---:|---:|---:|",
    ])
    for name, values in summary["harm"].items():
        lines.append(f"| {name} | {values['n']} | {values['harm_count']} | {values['harm_rate']:.6f} |")
    lines.extend([
        "",
        "## Per-Subject bAcc",
        "",
        "| dataset | target | source_only_tsmnet | rct | spdim_geodesic | spdim_bias |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_key = {(row["dataset"], int(row["target_subject"]), row["method"]): row for row in rows}
    for dataset in DATASETS:
        for target in sorted(EXPECTED_SUBJECTS[dataset]):
            values = [float(by_key[(dataset, target, method)]["bacc"]) for method in METHODS]
            lines.append(
                f"| {dataset} | {target} | {values[0]:.6f} | {values[1]:.6f} | "
                f"{values[2]:.6f} | {values[3]:.6f} |"
            )
    lines.extend([
        "",
        "## Metric Composition",
        "",
        "All repaired evaluation halves contain equal counts for the two classes: BNCI2014_001 `[36,36]`, Cho2017 `[50,50]` or `[60,60]`, and Lee2019_MI `[25,25]`. Therefore ordinary accuracy equals balanced accuracy for every row by class balance, not by absent-class scorer behavior.",
        "",
        "## Red Team Review",
        "",
        "- Scope is seed 0 only; seeds 1/2 and a full three-seed baseline remain unapproved.",
        "- The canceled monolithic job contributes zero rows to this merge.",
        "- All four methods share each subject's frozen repaired split; target labels enter only final metric evaluation.",
        "- No confidence interval is claimed from this single-seed expansion.",
    ])
    DIGEST_PATH.write_text("\n".join(lines) + "\n")


def write_audit(summary: dict[str, Any]) -> None:
    validation = summary["validation"]
    lines = [
        "# SPDIM W1 Repaired-Split Seed-0 Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- label: {LABEL}",
        "- monitoring policy: `squeue` plus artifact-level completion gates",
        "- result construction: four non-overlapping clean shards; canceled monolithic partial excluded",
        "",
        "## Row Gate",
        "",
        f"- expected_rows_total: `{validation['expected_rows_total']}`",
        f"- result_rows: `{validation['row_count']}`",
        f"- status_rows: `{json.dumps(validation['status_rows'], sort_keys=True)}`",
        f"- duplicate_keys: `{validation['duplicate_key_count']}`",
        f"- result_csv_sha256: `{summary['result_csv_sha256']}`",
        f"- prediction_hash_missing_rows: `{validation['prediction_hash_missing_rows']}`",
        f"- logits_hash_missing_rows: `{validation['logits_hash_missing_rows']}`",
        "",
        "## Dataset Rows",
        "",
        "| dataset | subjects | expected rows | actual rows |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | {validation['subjects_by_dataset'][dataset]} | "
            f"{EXPECTED_DATASET_ROWS[dataset]} | {validation['dataset_rows'][dataset]} |"
        )
    lines.extend([
        "",
        "## Slurm and Shard Evidence",
        "",
        "| shard | job id | target spec | rows | final squeue | stdout | stderr | result sha256 |",
        "|---|---:|---|---:|---|---|---|---|",
    ])
    for shard in summary["shards"]:
        lines.append(
            f"| {shard['shard_id']} | {shard['job_id']} | `{shard['target_spec']}` | "
            f"{shard['row_count']} | absent | {shard['stdout']['status']} | {shard['stderr']['status']} | "
            f"`{shard['result_csv_sha256']}` |"
        )
    lines.extend([
        "",
        "Accepted completion rule: job absent from `squeue` and shard/final CSV parse, row-count, status, provenance, and checksum gates all passed. Stdout is present with the clean launch header. Stderr is empty or contains only the declared MOABB zero-buffer warning and one Matplotlib font-cache notice.",
        "",
        "The canceled monolithic job `891435` wrote 56 partial rows with SHA-256 `ed434d8927b2ee73d6839ca1ba0c724de31118e2ed1e5a8230dc362adae341dc`; those rows are preserved outside the repository and excluded from every result statistic.",
        "",
        "## Clean Provenance Gate",
        "",
        f"- launch_commit: `{LAUNCH_COMMIT}`",
        f"- repaired_manifest_hash: `{MANIFEST_HASH}`",
        f"- runner_sha256: `{RUNNER_SHA256}`",
        f"- config_sha256: `{CONFIG_SHA256}`",
        f"- external_spdim_commit: `{EXTERNAL_SHA}`",
        "- clean_worktree_at_launch: `true` for all four shards",
        "- runner_dirty_allowed: `false` for all four shards",
        "- result commit preserves the launched runner, config, and manifest file checksums",
        f"- launcher_python: `{LAUNCHER_PYTHON}`",
        f"- recorded_environment_name: `{summary['environment_provenance']['recorded_environment_names'][0]}`",
        "- environment note: the runner recorded inherited `CONDA_DEFAULT_ENV=base`; the pinned Slurm launcher invoked the `icml` interpreter path above, and the launcher matches the launch commit byte-for-byte",
        "",
        "## Split and Leakage Gate",
        "",
        f"- single_class_eval_rows: `{validation['single_class_eval_rows']}`",
        f"- single_class_adapt_rows: `{validation['single_class_adapt_rows']}`",
        f"- adapt_eval_disjoint_failures: `{validation['adapt_eval_disjoint_failures']}`",
        f"- target_label_leakage_detected: `{validation['target_label_leakage_detected']}`",
        f"- target_performance_method_selection_detected: `{validation['target_performance_method_selection_detected']}`",
        f"- pretrained_weight_detected: `{validation['pretrained_weight_detected']}`",
        f"- vendoring_detected: `{validation['vendoring_detected']}`",
        "",
        "## Red Team Review",
        "",
        "- Raw shard CSVs, summaries, and normalized text logs are committed with the final packet; the merge is reproducible from those inputs.",
        "- Subject/method keys are exact and non-overlapping across shards; 115 subjects times four methods yields 460 rows.",
        "- The repaired split removes the old Cho2017 single-class evaluation defect, but this run remains seed 0 only.",
        "- The recorded environment name is an inherited shell label; interpreter provenance comes from the pinned launcher path and launch-commit match.",
        "- No seeds 1/2, full SPDIM, H2CMI rerun, TeX edit, geometry stress, or orthogonal-score work is included.",
        "- No inferential CI or full-baseline claim is made.",
    ])
    AUDIT_PATH.write_text("\n".join(lines) + "\n")


def write_shard_readme(summary: dict[str, Any]) -> None:
    lines = [
        "# P8 SPDIM Repaired Seed-0 Shard Evidence",
        "",
        "This directory contains the four result-carrying shard CSVs, their machine-readable summaries, and their Slurm stdout/stderr evidence used by the final postprocessor.",
        "",
        "- Shard CSV and summary bytes are copied unchanged from the repository-external clean-run cache.",
        "- Text logs have trailing horizontal whitespace stripped so repository whitespace validation passes; no non-whitespace content was changed.",
        "- The final summary records a SHA-256 for every committed input.",
        "- Job `891435` is excluded and is not represented in these accepted shard files.",
        "",
        "| shard | rows | result sha256 | summary sha256 | stdout sha256 | stderr sha256 |",
        "|---|---:|---|---|---|---|",
    ]
    for shard in summary["shards"]:
        lines.append(
            f"| {shard['shard_id']} | {shard['row_count']} | `{shard['result_csv_sha256']}` | "
            f"`{shard['summary_json_sha256']}` | `{shard['stdout']['sha256']}` | `{shard['stderr']['sha256']}` |"
        )
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- These are raw execution outputs, not reconstructed predictions.",
        "- Final acceptance depends on the postprocessor's combined coverage, provenance, and leakage gates.",
    ])
    README_PATH.write_text("\n".join(lines) + "\n")


def append_command_log(summary: dict[str, Any]) -> None:
    marker = "Per PM P8B, completed the official SPDIM W1 repaired-split seed-0 expansion"
    if marker in COMMAND_LOG.read_text():
        return
    entry = f"""
- {marker} from four clean, non-overlapping GPU shards (`891456` array;
  result-carrying job IDs `891456`-`891459`). Final monitoring used `squeue`
  only and all accepted jobs were absent. The merged CSV contains
  `{summary['validation']['row_count']}`/`{summary['validation']['expected_rows_total']}`
  `ok` rows with dataset counts BNCI2014_001=`36`, Cho2017=`208`, and
  Lee2019_MI=`216`; prediction/logits hashes are complete, no evaluation split
  is single-class, and all adaptation/evaluation IDs are disjoint. The result
  SHA-256 is `{summary['result_csv_sha256']}`. Canceled monolithic job `891435`
  and its 56 partial rows remain excluded. This is an official repaired-split
  seed-0 expansion, not a full three-seed baseline; no seeds 1/2, TeX edits,
  H2CMI reruns, geometry stress, or orthogonal-score work was performed.
"""
    COMMAND_LOG.write_text(COMMAND_LOG.read_text().rstrip() + "\n\n" + entry.lstrip())


def main() -> None:
    squeue_gate = final_squeue_gate()
    rows, shard_meta, fieldnames = load_shards()
    validation = validate_rows(rows, squeue_gate, shard_meta)
    if not validation["validation_pass"]:
        write_json(FAILURE_PATH, {"status": "blocked", "validation": validation, "monitoring": squeue_gate})
        raise SystemExit(2)
    if FAILURE_PATH.exists():
        FAILURE_PATH.unlink()
    result_sha = write_results(rows, fieldnames)
    merged_rows = read_csv(RESULT_PATH)
    per_dataset, subject_weighted, dataset_macro, deltas, harms = summaries(merged_rows, DATASETS)
    legacy = write_legacy_compare(merged_rows)
    summary = {
        "status": "pass",
        "label": LABEL,
        "scope": {
            "datasets": DATASETS,
            "source_seeds": [0],
            "methods": METHODS,
            "split_family": "class_stratified_half",
            "full_three_seed_baseline": False,
        },
        "launch_commit": LAUNCH_COMMIT,
        "manifest_hash": MANIFEST_HASH,
        "external_spdim_commit": EXTERNAL_SHA,
        "environment_provenance": {
            "recorded_environment_names": validation["recorded_environment_names"],
            "recorded_name_interpretation": "Inherited CONDA_DEFAULT_ENV shell label, not the selected interpreter path.",
            "launcher_python": LAUNCHER_PYTHON,
            "launcher_path": LAUNCHER_RELATIVE_PATH,
            "launcher_sha256": validation["current_file_checksums"]["launcher_sha256"],
            "launcher_matches_launch_commit": validation["checks"]["launcher_matches_launch_commit"],
        },
        "accepted_job_ids": [shard["job_id"] for shard in SHARDS],
        "array_job_id": ARRAY_JOB_ID,
        "excluded_jobs": EXCLUDED_JOBS,
        "monitoring": squeue_gate,
        "validation": validation,
        "shards": shard_meta,
        "result_csv": str(RESULT_PATH.relative_to(ROOT)),
        "result_csv_sha256": result_sha,
        "per_dataset": per_dataset,
        "overall_subject_weighted": subject_weighted,
        "dataset_macro": dataset_macro,
        "deltas": deltas,
        "harm": harms,
        "legacy_compare": legacy,
    }
    write_digest(summary, merged_rows)
    write_audit(summary)
    write_shard_readme(summary)
    summary["artifacts"] = {
        "results_csv": {"path": str(RESULT_PATH.relative_to(ROOT)), "sha256": result_sha},
        "audit_md": {"path": str(AUDIT_PATH.relative_to(ROOT)), "sha256": sha256_file(AUDIT_PATH)},
        "result_digest_md": {"path": str(DIGEST_PATH.relative_to(ROOT)), "sha256": sha256_file(DIGEST_PATH)},
        "legacy_compare_md": {"path": str(LEGACY_PATH.relative_to(ROOT)), "sha256": sha256_file(LEGACY_PATH)},
        "shard_readme_md": {"path": str(README_PATH.relative_to(ROOT)), "sha256": sha256_file(README_PATH)},
    }
    write_json(SUMMARY_PATH, summary)
    append_command_log(summary)
    print(json.dumps({
        "status": summary["status"],
        "rows": validation["row_count"],
        "result_csv_sha256": result_sha,
        "final_squeue_absent": squeue_gate["final_squeue_absent"],
        "single_class_eval_rows": validation["single_class_eval_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
