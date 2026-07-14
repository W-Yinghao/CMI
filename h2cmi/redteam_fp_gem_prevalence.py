"""Independent final red-team validator for the P13 prevalence stress."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from h2cmi.data.real_eeg import load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.w1_repaired_split import indices_from_trial_ids


DATASET = "Lee2019_MI"
METHODS = (
    "source_only_tsmnet",
    "rct",
    "spdim_geodesic",
    "spdim_bias",
    "Joint-GEM",
    "FP-GEM",
)
COMPARATORS = ("Joint-GEM", "rct", "spdim_geodesic", "spdim_bias")
Q_VALUES = (0.1, 0.5, 0.9)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_710


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def assert_close(actual: Any, expected: Any, label: str, atol: float = 1e-15) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def bootstrap_matrices(matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 54, size=(BOOTSTRAP_REPLICATES, 54))
    return {name: matrix[indices].mean(axis=1) for name, matrix in matrices.items()}


def validate_ci_rows(
    rows: list[dict[str, str]],
    point: dict[str, np.ndarray],
    boot: dict[str, np.ndarray],
) -> None:
    method_index = {method: index for index, method in enumerate(METHODS)}
    expected_keys = set()
    for endpoint in point:
        for method in METHODS:
            expected_keys.add((endpoint, "method_summary", method))
        for comparator in COMPARATORS:
            expected_keys.add((endpoint, "paired_contrast", f"FP-GEM minus {comparator}"))
    observed_keys = set()
    for row in rows:
        endpoint = row.get("endpoint", "sensitivity") or "sensitivity"
        row_type = row["row_type"]
        key = row["method"] if row_type == "method_summary" else row["comparison"]
        observed_keys.add((endpoint, row_type, key))
        if row_type == "method_summary":
            index = method_index[row["method"]]
            values = point[endpoint][:, index]
            bootstrap = boot[endpoint][:, index]
        else:
            comparator = row["comparison"].removeprefix("FP-GEM minus ")
            fp_index = method_index["FP-GEM"]
            comparator_index = method_index[comparator]
            values = point[endpoint][:, fp_index] - point[endpoint][:, comparator_index]
            bootstrap = boot[endpoint][:, fp_index] - boot[endpoint][:, comparator_index]
        expected = (
            float(values.mean()),
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        )
        observed = tuple(float(row[name]) for name in ("estimate", "ci_low", "ci_high"))
        assert_close(observed, expected, f"CI row {(endpoint, row_type, key)}")
    if observed_keys != expected_keys:
        raise AssertionError(
            f"CI key mismatch: missing={expected_keys - observed_keys}, extra={observed_keys - expected_keys}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="/home/infres/yinwang/.cache/h2cmi_training_caches/fp_gem_p13",
    )
    parser.add_argument("--out-dir", default="h2cmi/results/fp_gem_prevalence")
    parser.add_argument(
        "--report-json",
        default="h2cmi/results/fp_gem_prevalence/P13_FINAL_RED_TEAM.json",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = Path(args.raw_root)
    out = root / args.out_dir
    report_path = root / args.report_json

    manifest = json.loads((out / "fp_gem_prevalence_manifest.json").read_text())
    units = {
        (int(unit["target_subject"]), int(unit["source_seed"])): unit
        for unit in manifest["units"]
    }
    if len(units) != 162:
        raise AssertionError(f"manifest unit count: {len(units)}")
    if len(read_csv(out / "fp_gem_prevalence_manifest.csv")) != 24_300:
        raise AssertionError("manifest occurrence row count mismatch")

    print("red_team_load_dataset_begin", flush=True)
    epochs = load_dataset(DATASET, MOABB_CLASS[DATASET]().subject_list)
    print(f"red_team_load_dataset_complete n={len(epochs.y)}", flush=True)

    raw_index: dict[tuple[int, int, float, str], dict[str, Any]] = {}
    raw_geometry: dict[tuple[int, int, float, str], dict[str, Any]] = {}
    predictions: dict[tuple[int, int, float, str], np.ndarray] = {}
    metric_errors: list[Any] = []
    evaluation_counts: Counter[tuple[int, int]] = Counter()
    q05_exact_units = 0
    raw_paths = sorted((raw_root / "units").glob("*.json"))
    if len(raw_paths) != 162:
        raise AssertionError(f"raw unit count: {len(raw_paths)}")
    for (target, seed), unit in sorted(units.items()):
        path = raw_root / f"units/{DATASET}_target{target}_seed{seed}.json"
        payload = json.loads(path.read_text())
        if payload["status"] != "ok":
            raise AssertionError(f"raw status failed: {(target, seed)}")
        if payload["q05_center_reproduction"]["p12_center_reuse_approved"]:
            q05_exact_units += 1
        eval_indices = indices_from_trial_ids(unit["eval_trial_ids"])
        y_eval = np.asarray(epochs.y[eval_indices], dtype=np.int64)
        evaluation_counts[tuple(np.bincount(y_eval, minlength=2).tolist())] += 1
        for row in payload["results"]:
            key = (target, seed, float(row["q"]), row["method"])
            if key in raw_index:
                raise AssertionError(f"duplicate raw key: {key}")
            prediction = np.asarray(row["prediction_vector"], dtype=np.int64)
            recomputed = (
                float(accuracy_score(y_eval, prediction)),
                float(balanced_accuracy_score(y_eval, prediction)),
                float(f1_score(y_eval, prediction, average="macro", zero_division=0)),
            )
            observed = tuple(float(row[name]) for name in ("acc", "bacc", "macro_f1"))
            if not np.allclose(recomputed, observed, rtol=0.0, atol=1e-15):
                metric_errors.append((key, recomputed, observed))
            raw_index[key] = row
            predictions[key] = prediction
        for row in payload["geometry"]:
            key = (target, seed, float(row["q"]), row["method"])
            if key in raw_geometry:
                raise AssertionError(f"duplicate raw geometry key: {key}")
            raw_geometry[key] = row
    if metric_errors:
        raise AssertionError(f"raw metric recomputation errors: {metric_errors[:5]}")
    if len(raw_index) != 2_916:
        raise AssertionError(f"raw result row count: {len(raw_index)}")
    if len(raw_geometry) != 972:
        raise AssertionError(f"raw geometry row count: {len(raw_geometry)}")
    if evaluation_counts != Counter({(25, 25): 162}):
        raise AssertionError(f"evaluation composition: {evaluation_counts}")

    result_path = out / "fp_gem_prevalence_results.csv"
    result_rows = read_csv(result_path)
    result_index = {
        (
            int(row["target_subject"]),
            int(row["source_seed"]),
            float(row["q"]),
            row["method"],
        ): row
        for row in result_rows
    }
    if len(result_rows) != 2_916 or set(result_index) != set(raw_index):
        raise AssertionError("result CSV key coverage mismatch")
    for key, row in result_index.items():
        raw = raw_index[key]
        for name in ("acc", "bacc", "macro_f1"):
            if float(row[name]) != float(raw[name]):
                raise AssertionError(f"result metric binding mismatch: {(key, name)}")
        for name in (
            "prediction_hash",
            "logits_hash",
            "checkpoint_hash",
            "adaptation_manifest_hash",
            "status",
        ):
            if row[name] != str(raw[name]):
                raise AssertionError(f"result provenance binding mismatch: {(key, name)}")

    geometry_rows = read_csv(out / "fp_gem_prevalence_geometry_diagnostic.csv")
    geometry_index = {
        (
            int(row["target_subject"]),
            int(row["source_seed"]),
            float(row["q"]),
            row["method"],
        ): row
        for row in geometry_rows
    }
    if len(geometry_rows) != 972 or set(geometry_index) != set(raw_geometry):
        raise AssertionError("geometry CSV key coverage mismatch")
    for key, row in geometry_index.items():
        raw = raw_geometry[key]
        for name in (
            "translation_norm",
            "log_scale_norm",
            "translation_displacement_from_q05",
            "log_scale_displacement_from_q05",
            "geometry_displacement_from_q05",
        ):
            if float(row[name]) != float(raw[name]):
                raise AssertionError(f"geometry numeric binding mismatch: {(key, name)}")
        for name in (
            "checkpoint_hash",
            "adaptation_manifest_hash",
            "translation_vector_hash",
            "log_scale_vector_hash",
        ):
            if row[name] != str(raw[name]):
                raise AssertionError(f"geometry provenance binding mismatch: {(key, name)}")

    method_index = {method: index for index, method in enumerate(METHODS)}
    q_index = {q: index for index, q in enumerate(Q_VALUES)}
    bacc = np.empty((54, len(METHODS), len(Q_VALUES)), dtype=np.float64)
    disagreement = np.empty_like(bacc)
    for subject in range(1, 55):
        for method in METHODS:
            for q in Q_VALUES:
                bacc[subject - 1, method_index[method], q_index[q]] = np.mean([
                    float(raw_index[(subject, seed, q, method)]["bacc"])
                    for seed in (0, 1, 2)
                ])
                disagreement[subject - 1, method_index[method], q_index[q]] = np.mean([
                    np.mean(
                        predictions[(subject, seed, q, method)]
                        != predictions[(subject, seed, 0.5, method)]
                    )
                    for seed in (0, 1, 2)
                ])
    sensitivity = 0.5 * (
        np.abs(bacc[:, :, 0] - bacc[:, :, 1])
        + np.abs(bacc[:, :, 2] - bacc[:, :, 1])
    )
    endpoint_mean = 0.5 * (bacc[:, :, 0] + bacc[:, :, 2])
    worst = np.minimum(bacc[:, :, 0], bacc[:, :, 2])
    endpoint_disagreement = 0.5 * (disagreement[:, :, 0] + disagreement[:, :, 2])
    point = {
        "sensitivity": sensitivity,
        "endpoint_mean_bacc": endpoint_mean,
        "worst_prevalence_bacc": worst,
        "endpoint_mean_prediction_disagreement": endpoint_disagreement,
        "mean_bacc_q01": bacc[:, :, 0],
        "mean_bacc_q05": bacc[:, :, 1],
        "mean_bacc_q09": bacc[:, :, 2],
    }
    boot = bootstrap_matrices(point)
    validate_ci_rows(
        read_csv(out / "fp_gem_prevalence_sensitivity_ci.csv"),
        {"sensitivity": sensitivity},
        {"sensitivity": boot["sensitivity"]},
    )
    validate_ci_rows(
        read_csv(out / "fp_gem_prevalence_endpoint_ci.csv"),
        {name: values for name, values in point.items() if name != "sensitivity"},
        {name: values for name, values in boot.items() if name != "sensitivity"},
    )

    per_subject_rows = read_csv(out / "fp_gem_prevalence_per_subject.csv")
    if len(per_subject_rows) != 324:
        raise AssertionError(f"per-subject row count: {len(per_subject_rows)}")
    for row in per_subject_rows:
        subject = int(row["target_subject"]) - 1
        method = method_index[row["method"]]
        expected = (
            sensitivity[subject, method],
            endpoint_mean[subject, method],
            worst[subject, method],
            endpoint_disagreement[subject, method],
        )
        observed = tuple(float(row[name]) for name in (
            "prevalence_sensitivity",
            "endpoint_mean_bacc",
            "worst_prevalence_bacc",
            "endpoint_mean_prediction_disagreement",
        ))
        assert_close(observed, expected, f"per-subject row {(subject + 1, row['method'])}")

    summary = json.loads((out / "fp_gem_prevalence_summary.json").read_text())
    if summary["validation"]["validation_pass"] is not True:
        raise AssertionError("analyzer validation did not pass")
    if summary["validation"]["target_label_leakage_detected"] is not False:
        raise AssertionError("target-label leakage flag")
    if summary["validation"]["target_performance_selection_detected"] is not False:
        raise AssertionError("target-performance selection flag")
    if len(read_csv(out / "fp_gem_prevalence_job_artifact_manifest.csv")) != 162:
        raise AssertionError("accepted artifact-manifest row count")
    if len(read_csv(out / "fp_gem_prevalence_excluded_artifact_manifest.csv")) != 6:
        raise AssertionError("excluded artifact-manifest row count")
    for name, expected_hash in summary["artifacts"].items():
        if sha256_file(out / name) != expected_hash:
            raise AssertionError(f"summary artifact checksum mismatch: {name}")

    submission = json.loads((out / "fp_gem_prevalence_submission_record.json").read_text())
    job_ids = [str(submission["checkpoint_gate"]["job_id"])] + [
        str(item["job_id"]) for item in submission["arrays"]
    ]
    command = ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T|%P|%R"]
    queue = subprocess.run(command, text=True, capture_output=True, check=True)
    if queue.stdout.strip():
        raise AssertionError(f"jobs remain in squeue: {queue.stdout.strip()}")

    fp_index = method_index["FP-GEM"]
    contrasts = {}
    for comparator in COMPARATORS:
        comparator_index = method_index[comparator]
        values = sensitivity[:, fp_index] - sensitivity[:, comparator_index]
        bootstrap = boot["sensitivity"][:, fp_index] - boot["sensitivity"][:, comparator_index]
        contrasts[f"FP-GEM minus {comparator}"] = {
            "estimate": float(values.mean()),
            "ci_low": float(np.percentile(bootstrap, 2.5)),
            "ci_high": float(np.percentile(bootstrap, 97.5)),
        }
    primary = contrasts["FP-GEM minus Joint-GEM"]
    expected_support = primary["ci_high"] < 0.0
    if summary["primary_method_claim_supported"] is not expected_support:
        raise AssertionError("primary support verdict mismatch")

    report = {
        "status": "pass",
        "review": "independent_final_red_team",
        "dataset": DATASET,
        "raw_units": len(raw_paths),
        "raw_result_rows": len(raw_index),
        "result_csv_rows": len(result_rows),
        "geometry_rows": len(geometry_rows),
        "per_subject_rows": len(per_subject_rows),
        "evaluation_class_counts": {"25,25": evaluation_counts[(25, 25)]},
        "independent_metric_recompute_errors": len(metric_errors),
        "independent_ci_recompute_errors": 0,
        "q05_exact_units": q05_exact_units,
        "final_squeue_absence": True,
        "result_sha256": sha256_file(result_path),
        "primary_comparison": primary,
        "primary_method_claim_supported": expected_support,
        "external_sensitivity_comparisons": {
            name: value for name, value in contrasts.items() if name != "FP-GEM minus Joint-GEM"
        },
        "residual_limit": (
            "New-q logits vectors were not persisted, so their hashes are complete but cannot be "
            "independently regenerated; q=0.5 logits reproduce P12 exactly."
        ),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
