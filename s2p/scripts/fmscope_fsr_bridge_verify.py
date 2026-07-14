#!/usr/bin/env python
"""Independent fail-closed verifier for FMScope-FSR bridge Panel 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


FMSCOPE_COMMIT = "09885016a00db6c7de0074304c455c50685100c9"
LABEL_SEEDS = (42, 123, 2024)
N_SPLITS = 5
N_RANDOM = 100
EXPECTED_RANDOM_ROWS = 2 * 3 * 5 * 100 * 4
DATASETS = {
    "eegmat": {
        "cache": "reproduction/data/features_cache/frozen_cbramod_eegmat_perwindow.npz",
        "sha256": "b4ed9917eeb9cac2eaea911903700da7ce269c40ebb53d0039e93d88403875bc",
        "shape": (1707, 200),
    },
    "sleepdep": {
        "cache": "reproduction/data/features_cache/frozen_cbramod_sleepdep_perwindow.npz",
        "sha256": "da8280e0a469f41c65cea97572dd37e6bd2fd104c05a83d49b26684645a2b091",
        "shape": (4207, 200),
    },
}


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode() + b"\0")
    digest.update(str(tuple(value.shape)).encode() + b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def quantized_array_sha256(array: np.ndarray, decimals: int = 8) -> str:
    return array_sha256(np.round(np.asarray(array, dtype=np.float64), decimals))


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def independent_leace(features: np.ndarray, subjects: np.ndarray):
    values = np.asarray(features, dtype=np.float64)
    mu = values.mean(axis=0)
    centered = values - mu
    covariance, _ = ledoit_wolf(centered, assume_centered=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    roots = np.sqrt(eigenvalues)
    positive = roots > 1e-8 * roots.max()
    inverse = np.zeros_like(roots)
    inverse[positive] = 1.0 / roots[positive]
    whitening = (eigenvectors * inverse) @ eigenvectors.T
    dewhitening = (eigenvectors * roots) @ eigenvectors.T
    unique, inverse_subject = np.unique(subjects, return_inverse=True)
    design = np.eye(len(unique), dtype=np.float64)[inverse_subject]
    design -= design.mean(axis=0)
    cross_covariance = centered.T @ design / len(centered)
    u, singular_values, _ = np.linalg.svd(
        whitening @ cross_covariance, full_matrices=False
    )
    rank = int((singular_values > 1e-6 * singular_values.max()).sum())
    ur = u[:, :rank]
    projection = np.eye(values.shape[1]) - dewhitening @ ur @ ur.T @ whitening
    basis, _ = np.linalg.qr(dewhitening @ ur)
    after = centered @ projection.T + mu
    return {
        "mu": mu,
        "projection": projection,
        "basis": basis[:, :rank],
        "rank": rank,
        "after": after,
    }


class Head:
    def __init__(self, features, labels, seed):
        self.low = np.percentile(features, 1, axis=0)
        self.high = np.percentile(features, 99, axis=0)
        clipped = np.clip(features, self.low, self.high)
        self.scaler = StandardScaler().fit(clipped)
        self.classifier = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            solver="liblinear",
            tol=1e-3,
            random_state=seed,
        )
        self.classifier.fit(self.scaler.transform(clipped), labels)

    def probabilities(self, features):
        clipped = np.clip(features, self.low, self.high)
        return self.classifier.predict_proba(self.scaler.transform(clipped))[:, 1]


def stable_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def metrics(head, features, window_rec_idx, rec_labels, test_recordings):
    probabilities = head.probabilities(features)
    pooled = np.asarray(
        [probabilities[window_rec_idx == rec].mean() for rec in test_recordings]
    )
    labels = rec_labels[test_recordings].astype(int)
    predictions = (pooled >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "nll": float(log_loss(labels, np.column_stack([1 - pooled, pooled]), labels=[0, 1])),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "prediction_sha256": array_sha256(pooled),
    }


def folds(data, seed):
    recordings = np.arange(len(data["rec_labels"]))
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(splitter.split(recordings, data["rec_labels"], groups=data["rec_pids"]))


def close(left, right, tolerance=1e-10):
    return abs(float(left) - float(right)) <= tolerance


def independent_inference(fresh_rows, exact_rows, random_rows):
    summaries = []
    for dataset in DATASETS:
        for regime in ("global", "source"):
            for endpoint, rows, random_prefix in (
                ("fresh_probe", fresh_rows, "fresh"),
                ("exact_head", exact_rows, "exact"),
            ):
                baseline = {
                    (int(row["outer_seed"]), int(row["fold"])): float(
                        row["balanced_accuracy"]
                    )
                    for row in rows
                    if row["dataset"] == dataset and row["protocol"] == "unchanged"
                }
                subject = {
                    (int(row["outer_seed"]), int(row["fold"])): float(
                        row["balanced_accuracy"]
                    )
                    for row in rows
                    if row["dataset"] == dataset
                    and row["protocol"] == f"{regime}_subject_leace"
                }
                observed = float(
                    np.mean([subject[key] - baseline[key] for key in sorted(baseline)])
                )
                nulls = {}
                for null_kind in ("same_rank", "variance_matched"):
                    values_by_draw = defaultdict(list)
                    for row in random_rows:
                        if (
                            row["dataset"] == dataset
                            and row["protocol"] == f"{regime}_random_{null_kind}"
                        ):
                            key = (int(row["outer_seed"]), int(row["fold"]))
                            values_by_draw[int(row["draw"])].append(
                                float(row[f"{random_prefix}_balanced_accuracy"])
                                - baseline[key]
                            )
                    draw_values = np.asarray(
                        [np.mean(values_by_draw[draw]) for draw in range(N_RANDOM)]
                    )
                    nulls[null_kind] = {
                        "mean": float(draw_values.mean()),
                        "raw_p": float(
                            (1 + np.sum(draw_values >= observed)) / (N_RANDOM + 1)
                        ),
                    }
                summaries.append(
                    {
                        "dataset": dataset,
                        "information_regime": regime,
                        "endpoint": endpoint,
                        "observed": observed,
                        "nulls": nulls,
                    }
                )
    for null_kind in ("same_rank", "variance_matched"):
        values = np.asarray([row["nulls"][null_kind]["raw_p"] for row in summaries])
        order = np.argsort(values)
        adjusted = np.empty_like(values)
        running = 0.0
        for position, index in enumerate(order):
            running = max(running, (len(values) - position) * values[index])
            adjusted[index] = min(running, 1.0)
        for index, value in enumerate(adjusted):
            summaries[index]["nulls"][null_kind]["adjusted_p"] = float(value)
    for row in summaries:
        if (
            row["observed"] > 0
            and row["nulls"]["same_rank"]["adjusted_p"] <= 0.05
            and row["nulls"]["variance_matched"]["adjusted_p"] <= 0.05
        ):
            verdict = "IDENTITY_SPECIFIC_BENEFIT_SUPPORTED"
        elif row["observed"] > 0:
            verdict = "POSITIVE_NOT_FAMILYWISE_SPECIFIC"
        elif row["observed"] < 0:
            verdict = "REMOVAL_HARMS_ENDPOINT"
        else:
            verdict = "NO_ENDPOINT_CHANGE"
        row["verdict"] = verdict
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--fmscope-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/s2p_fmscope_fsr_bridge_panel1"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    fmscope_root = args.fmscope_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})

    check("official_commit", git_head(fmscope_root) == FMSCOPE_COMMIT, git_head(fmscope_root))
    assets = read_csv(output_dir / "bridge_external_asset_manifest.csv")
    cache_payload = {}
    for dataset, contract in DATASETS.items():
        path = fmscope_root / contract["cache"]
        actual = sha256_file(path)
        check(f"{dataset}_cache_hash", actual == contract["sha256"], actual)
        data = load_cache(path)
        cache_payload[dataset] = data
        check(f"{dataset}_shape", data["features"].shape == contract["shape"], data["features"].shape)
        row = next((row for row in assets if row["dataset"] == dataset), None)
        check(
            f"{dataset}_asset_manifest",
            row is not None
            and row["cache_sha256"] == actual
            and row["external_commit"] == FMSCOPE_COMMIT,
            row,
        )

    b0 = read_csv(output_dir / "bridge_b0_exact_replication.csv")
    eegmat_b0 = next(row for row in b0 if row["dataset"] == "eegmat")
    sleepdep_b0 = next(row for row in b0 if row["dataset"] == "sleepdep")
    check(
        "b0_eegmat_gate_replay",
        int(eegmat_b0["rank_subject_axis"]) == 35
        and float(eegmat_b0["subj_ba_linear_post"]) <= float(eegmat_b0["chance"]) + 0.01
        and eegmat_b0["interpretable"] == "True"
        and float(eegmat_b0["label_ba_delta"]) > 0
        and abs(float(eegmat_b0["live_minus_historical_delta"])) <= 0.01,
        eegmat_b0,
    )
    check(
        "b0_sleepdep_execution_replay",
        int(sleepdep_b0["rank_subject_axis"]) == 35
        and sleepdep_b0["interpretable"] == "True"
        and np.isfinite(float(sleepdep_b0["label_ba_delta"])),
        sleepdep_b0,
    )

    fold_rows = read_csv(output_dir / "bridge_fold_assignments.csv")
    expected_fold_rows = []
    for dataset, data in cache_payload.items():
        for outer_seed in LABEL_SEEDS:
            for fold, (train_recordings, test_recordings) in enumerate(folds(data, outer_seed)):
                source_subjects = set(data["rec_pids"][train_recordings].tolist())
                target_subjects = set(data["rec_pids"][test_recordings].tolist())
                if source_subjects & target_subjects:
                    check(
                        f"{dataset}_{outer_seed}_{fold}_subject_disjoint",
                        False,
                        source_subjects & target_subjects,
                    )
                for role, recordings in (
                    ("source", train_recordings),
                    ("heldout_final_score", test_recordings),
                ):
                    for recording in recordings:
                        expected_fold_rows.append(
                            (
                                dataset,
                                outer_seed,
                                fold,
                                int(recording),
                                int(data["rec_pids"][recording]),
                                int(data["rec_labels"][recording]),
                                role,
                            )
                        )
    observed_fold_rows = [
        (
            row["dataset"],
            int(row["outer_seed"]),
            int(row["fold"]),
            int(row["recording"]),
            int(row["subject"]),
            int(row["label"]),
            row["role"],
        )
        for row in fold_rows
    ]
    check(
        "fold_assignments_exact",
        sorted(expected_fold_rows) == sorted(observed_fold_rows),
        f"observed={len(observed_fold_rows)} expected={len(expected_fold_rows)}",
    )

    fresh = read_csv(output_dir / "bridge_fresh_probe_results.csv")
    exact = read_csv(output_dir / "bridge_exact_head_results.csv")
    random_rows = read_csv(output_dir / "bridge_random_null_results.csv")
    check("fresh_row_count", len(fresh) == 2 * 3 * 5 * 3, len(fresh))
    check("exact_row_count", len(exact) == 2 * 3 * 5 * 3, len(exact))
    check("random_row_count", len(random_rows) == EXPECTED_RANDOM_ROWS, len(random_rows))

    random_groups = {}
    rank_failures = []
    match_failures = []
    for row in random_rows:
        key = (
            row["dataset"],
            int(row["outer_seed"]),
            int(row["fold"]),
            row["protocol"],
        )
        random_groups.setdefault(key, set()).add(int(row["draw"]))
        if row["protocol"].endswith("same_rank") and int(row["rank"]) != int(
            row["paired_subject_rank"]
        ):
            rank_failures.append((key, row["draw"]))
        if row["protocol"].endswith("variance_matched") and float(
            row["variance_match_abs_error"]
        ) > 1e-10:
            match_failures.append((key, row["draw"], row["variance_match_abs_error"]))
    expected_groups = 2 * 3 * 5 * 4
    draw_failures = [key for key, values in random_groups.items() if values != set(range(N_RANDOM))]
    check(
        "random_draw_contract",
        len(random_groups) == expected_groups and not draw_failures,
        f"groups={len(random_groups)} failures={draw_failures[:3]}",
    )
    check("same_rank_contract", not rank_failures, rank_failures[:3])
    check("variance_match_contract", not match_failures, match_failures[:3])

    reported_inference = json.loads(
        (output_dir / "bridge_panel1_inference.json").read_text()
    )
    reported_by_key = {
        (row["dataset"], row["information_regime"], row["endpoint"]): row
        for row in reported_inference["cell_endpoint_results"]
    }
    inference_failures = []
    for replay in independent_inference(fresh, exact, random_rows):
        key = (
            replay["dataset"],
            replay["information_regime"],
            replay["endpoint"],
        )
        reported = reported_by_key.get(key)
        if reported is None:
            inference_failures.append((key, "missing"))
            continue
        comparisons = (
            (
                "observed",
                replay["observed"],
                reported["subject_leace_mean_delta_balanced_accuracy"],
            ),
            (
                "same_rank_raw_p",
                replay["nulls"]["same_rank"]["raw_p"],
                reported["same_rank_random"][
                    "empirical_one_sided_p_random_ge_subject"
                ],
            ),
            (
                "same_rank_holm_p",
                replay["nulls"]["same_rank"]["adjusted_p"],
                reported["same_rank_random"][
                    "holm_adjusted_p_eight_cell_family"
                ],
            ),
            (
                "variance_raw_p",
                replay["nulls"]["variance_matched"]["raw_p"],
                reported["variance_matched_random"][
                    "empirical_one_sided_p_random_ge_subject"
                ],
            ),
            (
                "variance_holm_p",
                replay["nulls"]["variance_matched"]["adjusted_p"],
                reported["variance_matched_random"][
                    "holm_adjusted_p_eight_cell_family"
                ],
            ),
        )
        for name, left, right in comparisons:
            if not close(left, right, 1e-12):
                inference_failures.append((key, name, left, right))
        if replay["verdict"] != reported["identity_specificity_verdict"]:
            inference_failures.append(
                (
                    key,
                    "verdict",
                    replay["verdict"],
                    reported["identity_specificity_verdict"],
                )
            )
    check(
        "claim_ledger_independent_replay",
        not inference_failures and len(reported_by_key) == 8,
        inference_failures[:5],
    )

    canaries = json.loads((output_dir / "bridge_transform_canaries.json").read_text())[
        "canaries"
    ]
    canary_failures = []
    metric_failures = []
    prediction_hash_differences = []
    for dataset, data in cache_payload.items():
        outer_seed = LABEL_SEEDS[0]
        fold = 0
        train_recordings, test_recordings = folds(data, outer_seed)[fold]
        window_rec = data["window_rec_idx"].astype(int)
        train_mask = np.isin(window_rec, train_recordings)
        test_mask = np.isin(window_rec, test_recordings)
        features = data["features"].astype(np.float64)
        global_operator = independent_leace(features, data["window_pids"])
        source_operator = independent_leace(
            features[train_mask], data["window_pids"][train_mask]
        )
        source_train_after = source_operator["after"]
        source_test_after = (
            (features[test_mask] - source_operator["mu"]) @ source_operator["projection"].T
            + source_operator["mu"]
        )
        expected_canary = next(row for row in canaries if row["dataset"] == dataset)
        observed_hashes = {
            "global_subject_transformed_sha256": quantized_array_sha256(
                global_operator["after"]
            ),
            "source_subject_train_transformed_sha256": quantized_array_sha256(
                source_train_after
            ),
            "source_subject_test_transformed_sha256": quantized_array_sha256(
                source_test_after
            ),
        }
        for name, value in observed_hashes.items():
            if value != expected_canary[name]:
                canary_failures.append((dataset, name, value, expected_canary[name]))

        train_labels = data["window_labels"][train_mask]
        test_window_rec = window_rec[test_mask]
        head_seed = stable_seed("task_head", dataset, outer_seed, fold)
        original = Head(features[train_mask], train_labels, head_seed)
        for protocol, train_after, test_after in (
            ("global_subject_leace", global_operator["after"][train_mask], global_operator["after"][test_mask]),
            ("source_subject_leace", source_train_after, source_test_after),
        ):
            fresh_head = Head(train_after, train_labels, head_seed)
            replay = {
                "fresh": metrics(
                    fresh_head,
                    test_after,
                    test_window_rec,
                    data["rec_labels"],
                    test_recordings,
                ),
                "exact": metrics(
                    original,
                    test_after,
                    test_window_rec,
                    data["rec_labels"],
                    test_recordings,
                ),
            }
            for endpoint, rows in (("fresh", fresh), ("exact", exact)):
                recorded = next(
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and int(row["outer_seed"]) == outer_seed
                    and int(row["fold"]) == fold
                    and row["protocol"] == protocol
                )
                for metric in ("balanced_accuracy", "nll", "cohen_kappa"):
                    tolerance = 1e-5 if metric == "nll" else 1e-12
                    if not close(recorded[metric], replay[endpoint][metric], tolerance):
                        metric_failures.append(
                            (dataset, protocol, endpoint, metric, recorded[metric], replay[endpoint][metric])
                        )
                if recorded["prediction_sha256"] != replay[endpoint]["prediction_sha256"]:
                    prediction_hash_differences.append(
                        (dataset, protocol, endpoint, "prediction_sha256")
                    )
    check("transform_canaries_independent", not canary_failures, canary_failures[:3])
    check(
        "subject_method_metric_canaries",
        not metric_failures,
        {
            "metric_failures": metric_failures[:3],
            "prediction_hash_differences_from_algebraic_order": prediction_hash_differences,
            "nll_tolerance": 1e-5,
            "ba_kappa_tolerance": 1e-12,
        },
    )

    firewall = json.loads((output_dir / "bridge_target_information_firewall.json").read_text())
    check(
        "target_information_firewall",
        firewall.get("pass") is True
        and firewall.get("source_only_eraser_uses_heldout_features") is False
        and firewall.get("source_only_eraser_uses_heldout_subject_ids") is False
        and firewall.get("heldout_information_used_for_rank_or_null_selection") is False
        and firewall.get("best_seed_fold_draw_selection") is False,
        firewall,
    )

    go_nogo = json.loads((output_dir / "bridge_panel1_go_nogo.json").read_text())
    check(
        "analysis_go_nogo_scope",
        go_nogo.get("panel1_compute_complete") is True
        and go_nogo.get("launch_panel2") is False
        and go_nogo.get("launch_phase_d1_training") is False
        and go_nogo.get("fine_tuning_used") is False
        and go_nogo.get("foundation_training_used") is False,
        go_nogo,
    )

    passed = all(row["pass"] for row in checks)
    verdict = {
        "verifier": "independent_low_level_panel1_replay",
        "checks": checks,
        "all_checks_pass": passed,
        "panel1_scientific_closure_recommended": passed,
        "panel2_auto_launch": False,
        "phase_d1_training_auto_launch": False,
        "manuscript_writing_authorized": False,
    }
    (output_dir / "bridge_independent_verification.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    )
    if passed:
        go_nogo["independent_verifier_pass"] = True
        go_nogo["panel1_pm_review_ready"] = True
        go_nogo["panel1_scientific_closure"] = False
        go_nogo["launch_panel2"] = False
        (output_dir / "bridge_panel1_go_nogo.json").write_text(
            json.dumps(go_nogo, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"all_checks_pass": passed, "checks": len(checks)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
