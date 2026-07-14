#!/usr/bin/env python
"""Run the frozen FMScope-FSR Panel-1 diagnostic-to-deployment bridge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FMSCOPE_COMMIT = "09885016a00db6c7de0074304c455c50685100c9"
FMSCOPE_URL = "https://github.com/Jimmy110101013/fmscope"
LABEL_SEEDS = (42, 123, 2024)
N_SPLITS = 5
N_RANDOM = 100
VARIANCE_MATCH_TOLERANCE = 1e-10

DATASETS = {
    "eegmat": {
        "cache": "reproduction/data/features_cache/frozen_cbramod_eegmat_perwindow.npz",
        "sha256": "b4ed9917eeb9cac2eaea911903700da7ce269c40ebb53d0039e93d88403875bc",
        "shape": (1707, 200),
        "historical_delta": 0.06018518518518512,
        "role": "positive_reference",
    },
    "sleepdep": {
        "cache": "reproduction/data/features_cache/frozen_cbramod_sleepdep_perwindow.npz",
        "sha256": "da8280e0a469f41c65cea97572dd37e6bd2fd104c05a83d49b26684645a2b091",
        "shape": (4207, 200),
        "historical_delta": 0.0092592592592593,
        "role": "no_consensus_negative_control",
    },
}


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


def stable_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def load_cache(path: Path, expected_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    required = {
        "features",
        "window_rec_idx",
        "window_labels",
        "window_pids",
        "rec_labels",
        "rec_pids",
        "rec_n_epochs",
    }
    if set(data) != required:
        raise RuntimeError(f"cache keys differ: {sorted(data)}")
    if data["features"].shape != expected_shape or not np.isfinite(data["features"]).all():
        raise RuntimeError(f"feature shape/finiteness failed for {path}")
    rec = data["window_rec_idx"].astype(int)
    if rec.min() != 0 or rec.max() + 1 != len(data["rec_labels"]):
        raise RuntimeError(f"recording indices are not contiguous for {path}")
    if not np.array_equal(data["window_labels"], data["rec_labels"][rec]):
        raise RuntimeError(f"window/recording labels disagree for {path}")
    if not np.array_equal(data["window_pids"], data["rec_pids"][rec]):
        raise RuntimeError(f"window/recording subjects disagree for {path}")
    subjects, counts = np.unique(data["rec_pids"], return_counts=True)
    if len(subjects) != 36 or not np.all(counts == 2):
        raise RuntimeError(f"expected 36 subjects x 2 recordings for {path}")
    return data


@dataclass
class Eraser:
    mu: np.ndarray
    left: np.ndarray
    right: np.ndarray
    basis: np.ndarray
    rank: int
    cond: float

    def apply(self, features: np.ndarray) -> np.ndarray:
        centered = np.asarray(features, dtype=np.float64) - self.mu
        return centered - (centered @ self.left) @ self.right + self.mu


def whiten(features: np.ndarray):
    from sklearn.covariance import ledoit_wolf

    values = np.asarray(features, dtype=np.float64)
    mu = values.mean(axis=0)
    centered = values - mu
    covariance, _ = ledoit_wolf(centered, assume_centered=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    roots = np.sqrt(eigenvalues)
    maximum = roots.max() if roots.size else 0.0
    positive = roots > 1e-8 * maximum if maximum > 0 else np.zeros_like(roots, bool)
    inverse = np.where(positive, 1.0 / np.where(positive, roots, 1.0), 0.0)
    w = (eigenvectors * inverse) @ eigenvectors.T
    w_plus = (eigenvectors * roots) @ eigenvectors.T
    cond = (
        float(eigenvalues[positive].max() / eigenvalues[positive].min())
        if positive.sum() >= 2
        else float("inf")
    )
    return mu, centered, w, w_plus, cond


def eraser_from_whitened_basis(mu, w, w_plus, whitened_basis, cond) -> Eraser:
    q = np.asarray(whitened_basis, dtype=np.float64)
    source_basis, _ = np.linalg.qr(w_plus @ q)
    rank = q.shape[1]
    return Eraser(
        mu=mu,
        left=w.T @ q,
        right=q.T @ w_plus.T,
        basis=source_basis[:, :rank],
        rank=rank,
        cond=float(cond),
    )


def fit_subject_leace(
    features: np.ndarray,
    subjects: np.ndarray,
    whitening_stats=None,
) -> Eraser:
    mu, centered, w, w_plus, cond = whitening_stats or whiten(features)
    subject_values = np.asarray(subjects)
    unique_subjects = np.unique(subject_values)
    index = {value: idx for idx, value in enumerate(unique_subjects)}
    design = np.zeros((len(subject_values), len(unique_subjects)), dtype=np.float64)
    for row, subject in enumerate(subject_values):
        design[row, index[subject]] = 1.0
    design -= design.mean(axis=0)
    cross_covariance = centered.T @ design / len(centered)
    u, singular_values, _ = np.linalg.svd(w @ cross_covariance, full_matrices=False)
    rank = (
        int((singular_values > 1e-6 * singular_values.max()).sum())
        if singular_values.size and singular_values.max() > 0
        else 0
    )
    if rank != len(unique_subjects) - 1:
        raise RuntimeError(
            f"subject-axis rank {rank} != n_subjects-1 {len(unique_subjects)-1}"
        )
    return eraser_from_whitened_basis(mu, w, w_plus, u[:, :rank], cond)


def fit_same_rank_random(
    whitening_stats, rank: int, rng: np.random.Generator
) -> Eraser:
    mu, _, w, w_plus, cond = whitening_stats
    random_basis, _ = np.linalg.qr(rng.standard_normal((w.shape[0], rank)))
    return eraser_from_whitened_basis(mu, w, w_plus, random_basis[:, :rank], cond)


def removed_energy_fraction(
    fit_features: np.ndarray, transformed: np.ndarray, mu: np.ndarray
) -> float:
    denominator = float(np.sum((fit_features - mu) ** 2)) + 1e-12
    return float(np.sum((fit_features - transformed) ** 2) / denominator)


def fit_variance_matched_random(
    fit_features: np.ndarray,
    target_fraction: float,
    rng: np.random.Generator,
) -> tuple[Eraser, float, float]:
    values = np.asarray(fit_features, dtype=np.float64)
    mu = values.mean(axis=0)
    centered = values - mu
    total = float(np.sum(centered**2)) + 1e-12
    q, _ = np.linalg.qr(rng.standard_normal((values.shape[1], values.shape[1])))
    per_direction = np.sum((centered @ q) ** 2, axis=0) / total
    cumulative = np.cumsum(per_direction)
    last = int(np.searchsorted(cumulative, target_fraction, side="left"))
    if last >= values.shape[1]:
        raise RuntimeError(
            f"random orthobasis cannot match removed fraction {target_fraction}"
        )
    previous = float(cumulative[last - 1]) if last else 0.0
    residual = max(target_fraction - previous, 0.0)
    alpha = float(math.sqrt(residual / max(float(per_direction[last]), 1e-15)))
    alpha = min(alpha, 1.0)
    directions = q[:, : last + 1]
    coefficients = np.ones(last + 1, dtype=np.float64)
    coefficients[-1] = alpha
    operator = Eraser(
        mu=mu,
        left=directions,
        right=coefficients[:, None] * directions.T,
        basis=directions,
        rank=last + 1,
        cond=float("nan"),
    )
    after = operator.apply(values)
    achieved = removed_energy_fraction(values, after, mu)
    return operator, achieved, abs(achieved - target_fraction)


@dataclass
class TaskHead:
    low: np.ndarray
    high: np.ndarray
    scaler: StandardScaler
    classifier: LogisticRegression

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        clipped = np.clip(features, self.low, self.high)
        scaled = self.scaler.transform(clipped)
        return self.classifier.predict_proba(scaled)[:, 1]


def fit_task_head(features: np.ndarray, labels: np.ndarray, seed: int) -> TaskHead:
    values = np.asarray(features, dtype=np.float64)
    low = np.percentile(values, 1, axis=0)
    high = np.percentile(values, 99, axis=0)
    clipped = np.clip(values, low, high)
    scaler = StandardScaler().fit(clipped)
    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=1.0,
        solver="liblinear",
        tol=1e-3,
        random_state=seed,
    )
    classifier.fit(scaler.transform(clipped), labels)
    return TaskHead(low, high, scaler, classifier)


def recording_metrics(
    head: TaskHead,
    features: np.ndarray,
    window_rec_idx: np.ndarray,
    rec_labels: np.ndarray,
    test_recordings: np.ndarray,
) -> dict:
    probabilities = head.probabilities(features)
    pooled = np.asarray(
        [probabilities[window_rec_idx == recording].mean() for recording in test_recordings]
    )
    labels = rec_labels[test_recordings].astype(int)
    predictions = (pooled >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "nll": float(log_loss(labels, np.column_stack([1 - pooled, pooled]), labels=[0, 1])),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "n_test_recordings": int(len(test_recordings)),
        "prediction_sha256": array_sha256(pooled),
    }


def direction_geometry(features, subjects, labels):
    directions = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        zero = mask & (labels == 0)
        one = mask & (labels == 1)
        if zero.sum() < 2 or one.sum() < 2:
            continue
        direction = features[one].mean(axis=0) - features[zero].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            directions.append(direction / norm)
    if len(directions) < 2:
        raise RuntimeError("too few paired subjects for direction geometry")
    matrix = np.stack(directions)
    pairwise = matrix @ matrix.T
    upper = pairwise[np.triu_indices(len(matrix), k=1)]
    consensus = matrix.mean(axis=0)
    consensus_norm = np.linalg.norm(consensus)
    if consensus_norm <= 1e-12:
        raise RuntimeError("task consensus direction is degenerate")
    return {
        "direction_consistency_median_cosine": float(np.median(upper)),
        "direction_consistency_iqr_low": float(np.percentile(upper, 25)),
        "direction_consistency_iqr_high": float(np.percentile(upper, 75)),
        "n_paired_subjects": int(len(matrix)),
        "consensus": consensus / consensus_norm,
    }


def subspace_geometry(source_basis: np.ndarray, target_basis: np.ndarray) -> dict:
    singular_values = np.linalg.svd(source_basis.T @ target_basis, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular_values))
    overlap = float(np.sum(singular_values**2) / min(source_basis.shape[1], target_basis.shape[1]))
    return {
        "normalized_projection_overlap": overlap,
        "principal_angle_mean_deg": float(angles.mean()),
        "principal_angle_min_deg": float(angles.min()),
        "principal_angle_max_deg": float(angles.max()),
    }


def subject_mean_scatter_removed(features, transformed, subjects) -> float:
    before = np.stack([features[subjects == value].mean(axis=0) for value in np.unique(subjects)])
    after = np.stack([transformed[subjects == value].mean(axis=0) for value in np.unique(subjects)])
    before -= before.mean(axis=0)
    after -= after.mean(axis=0)
    denominator = float(np.sum(before**2)) + 1e-12
    return float(1.0 - np.sum(after**2) / denominator)


def cross_recording_subject_decode(
    features: np.ndarray,
    window_rec_idx: np.ndarray,
    rec_pids: np.ndarray,
    recordings: np.ndarray,
    seed: int,
    kind: str,
) -> float:
    selected_subjects = np.unique(rec_pids[recordings])
    paired = {}
    for subject in selected_subjects:
        current = np.sort(recordings[rec_pids[recordings] == subject])
        if len(current) != 2:
            raise RuntimeError(f"subject {subject} does not have two recordings in fold")
        paired[subject] = current
    scores = []
    for direction in (0, 1):
        train_recordings = np.asarray([paired[s][direction] for s in selected_subjects])
        test_recordings = np.asarray([paired[s][1 - direction] for s in selected_subjects])
        train_mask = np.isin(window_rec_idx, train_recordings)
        test_mask = np.isin(window_rec_idx, test_recordings)
        x_train = features[train_mask]
        x_test = features[test_mask]
        y_train = rec_pids[window_rec_idx[train_mask]]
        y_test = rec_pids[window_rec_idx[test_mask]]
        scaler = StandardScaler().fit(x_train)
        if kind == "linear":
            classifier = LogisticRegression(
                max_iter=5000,
                C=1.0,
                class_weight="balanced",
                random_state=seed + direction,
            )
        elif kind == "mlp":
            classifier = MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=200,
                early_stopping=True,
                random_state=seed + direction,
            )
        else:
            raise ValueError(kind)
        classifier.fit(scaler.transform(x_train), y_train)
        predictions = classifier.predict(scaler.transform(x_test))
        scores.append(balanced_accuracy_score(y_test, predictions))
    return float(np.mean(scores))


def fold_indices(data: dict[str, np.ndarray], outer_seed: int):
    rec_indices = np.arange(len(data["rec_labels"]))
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=outer_seed
    )
    return list(
        splitter.split(rec_indices, data["rec_labels"], groups=data["rec_pids"])
    )


def method_row(
    dataset,
    outer_seed,
    fold,
    protocol,
    operator,
    fit_features,
    train_features,
    test_features,
    train_labels,
    head_seed,
    original_head,
    window_rec_idx_test,
    rec_labels,
    test_recordings,
):
    transformed_fit = operator.apply(fit_features)
    transformed_train = operator.apply(train_features)
    transformed_test = operator.apply(test_features)
    fresh_head = fit_task_head(transformed_train, train_labels, head_seed)
    fresh = recording_metrics(
        fresh_head, transformed_test, window_rec_idx_test, rec_labels, test_recordings
    )
    exact = recording_metrics(
        original_head, transformed_test, window_rec_idx_test, rec_labels, test_recordings
    )
    base = {
        "dataset": dataset,
        "outer_seed": outer_seed,
        "fold": fold,
        "protocol": protocol,
        "information_regime": "global_oracle" if protocol.startswith("global") else "source_only",
        "removal_kind": "subject_leace",
        "rank": operator.rank,
        "fit_removed_energy_fraction": removed_energy_fraction(
            fit_features, transformed_fit, operator.mu
        ),
        "source_removed_energy_fraction": removed_energy_fraction(
            train_features, transformed_train, operator.mu
        ),
        "heldout_removed_energy_fraction": removed_energy_fraction(
            test_features, transformed_test, operator.mu
        ),
    }
    return {**base, **fresh}, {**base, **exact}, transformed_train, transformed_test


def analyze_seed_task(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    dataset = task["dataset"]
    outer_seed = task["outer_seed"]
    path = Path(task["cache_path"])
    n_random = task["n_random"]
    with threadpool_limits(limits=1):
        data = load_cache(path, tuple(task["expected_shape"]))
        features = data["features"].astype(np.float64)
        window_rec_idx = data["window_rec_idx"].astype(int)
        labels = data["window_labels"].astype(int)
        subjects = data["window_pids"]
        rec_labels = data["rec_labels"].astype(int)
        rec_pids = data["rec_pids"]
        global_whitening = whiten(features)
        global_operator = fit_subject_leace(features, subjects, global_whitening)

        fresh_rows = []
        exact_rows = []
        random_rows = []
        transfer_rows = []
        modifier_rows = []
        fold_rows = []
        canaries = []

        for fold, (train_recordings, test_recordings) in enumerate(
            fold_indices(data, outer_seed)
        ):
            source_subjects = set(rec_pids[train_recordings].tolist())
            heldout_subjects = set(rec_pids[test_recordings].tolist())
            if source_subjects & heldout_subjects:
                raise RuntimeError(f"subject leakage {dataset} seed={outer_seed} fold={fold}")
            train_mask = np.isin(window_rec_idx, train_recordings)
            test_mask = np.isin(window_rec_idx, test_recordings)
            train_indices = np.flatnonzero(train_mask)
            test_indices = np.flatnonzero(test_mask)
            train_features = features[train_indices]
            test_features = features[test_indices]
            train_labels = labels[train_indices]
            train_subject_ids = subjects[train_indices]
            test_subject_ids = subjects[test_indices]
            test_window_rec = window_rec_idx[test_indices]
            head_seed = stable_seed("task_head", dataset, outer_seed, fold)
            original_head = fit_task_head(train_features, train_labels, head_seed)
            baseline = recording_metrics(
                original_head,
                test_features,
                test_window_rec,
                rec_labels,
                test_recordings,
            )
            baseline_base = {
                "dataset": dataset,
                "outer_seed": outer_seed,
                "fold": fold,
                "protocol": "unchanged",
                "information_regime": "source_only",
                "removal_kind": "none",
                "rank": 0,
                "fit_removed_energy_fraction": 0.0,
                "source_removed_energy_fraction": 0.0,
                "heldout_removed_energy_fraction": 0.0,
            }
            fresh_rows.append({**baseline_base, **baseline})
            exact_rows.append({**baseline_base, **baseline})

            source_whitening = whiten(train_features)
            source_operator = fit_subject_leace(
                train_features, train_subject_ids, source_whitening
            )
            subject_methods = (
                ("global_subject_leace", global_operator, features),
                ("source_subject_leace", source_operator, train_features),
            )
            method_outputs = {}
            for protocol, operator, fit_features in subject_methods:
                fresh, exact, transformed_train, transformed_test = method_row(
                    dataset,
                    outer_seed,
                    fold,
                    protocol,
                    operator,
                    fit_features,
                    train_features,
                    test_features,
                    train_labels,
                    head_seed,
                    original_head,
                    test_window_rec,
                    rec_labels,
                    test_recordings,
                )
                fresh_rows.append(fresh)
                exact_rows.append(exact)
                method_outputs[protocol] = (transformed_train, transformed_test)

            target_operator = fit_subject_leace(test_features, test_subject_ids)
            geometry = subspace_geometry(source_operator.basis, target_operator.basis)
            source_after_train, source_after_test = method_outputs["source_subject_leace"]
            pre_target_decode = cross_recording_subject_decode(
                features,
                window_rec_idx,
                rec_pids,
                test_recordings,
                stable_seed("target_subject", dataset, outer_seed, fold),
                "linear",
            )
            full_source_after = source_operator.apply(features)
            post_target_decode = cross_recording_subject_decode(
                full_source_after,
                window_rec_idx,
                rec_pids,
                test_recordings,
                stable_seed("target_subject", dataset, outer_seed, fold),
                "linear",
            )
            transfer_rows.append(
                {
                    "dataset": dataset,
                    "outer_seed": outer_seed,
                    "fold": fold,
                    "source_subjects": len(source_subjects),
                    "heldout_subjects": len(heldout_subjects),
                    "source_axis_rank": source_operator.rank,
                    "heldout_axis_rank": target_operator.rank,
                    **geometry,
                    "heldout_subject_mean_scatter_removed_fraction": subject_mean_scatter_removed(
                        test_features, source_after_test, test_subject_ids
                    ),
                    "heldout_cross_recording_subject_ba_pre": pre_target_decode,
                    "heldout_cross_recording_subject_ba_post_source_erasure": post_target_decode,
                    "heldout_subject_chance": 1.0 / len(heldout_subjects),
                }
            )

            source_direction = direction_geometry(
                train_features, train_subject_ids, train_labels
            )
            task_overlap = float(
                np.sum((source_operator.basis.T @ source_direction["consensus"]) ** 2)
            )
            nonlinear_post = cross_recording_subject_decode(
                full_source_after,
                window_rec_idx,
                rec_pids,
                train_recordings,
                stable_seed("source_mlp", dataset, outer_seed, fold),
                "mlp",
            )
            modifier_rows.append(
                {
                    "dataset": dataset,
                    "outer_seed": outer_seed,
                    "fold": fold,
                    "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                    "baseline_nll": baseline["nll"],
                    "source_subject_task_overlap": task_overlap,
                    "source_post_erasure_cross_recording_mlp_subject_ba": nonlinear_post,
                    **{key: value for key, value in source_direction.items() if key != "consensus"},
                }
            )

            for recording in train_recordings:
                fold_rows.append(
                    {
                        "dataset": dataset,
                        "outer_seed": outer_seed,
                        "fold": fold,
                        "recording": int(recording),
                        "subject": int(rec_pids[recording]),
                        "label": int(rec_labels[recording]),
                        "role": "source",
                    }
                )
            for recording in test_recordings:
                fold_rows.append(
                    {
                        "dataset": dataset,
                        "outer_seed": outer_seed,
                        "fold": fold,
                        "recording": int(recording),
                        "subject": int(rec_pids[recording]),
                        "label": int(rec_labels[recording]),
                        "role": "heldout_final_score",
                    }
                )

            global_fit_after = global_operator.apply(features)
            source_fit_after = source_operator.apply(train_features)
            target_global_fraction = removed_energy_fraction(
                features, global_fit_after, global_operator.mu
            )
            target_source_fraction = removed_energy_fraction(
                train_features, source_fit_after, source_operator.mu
            )

            for draw in range(n_random):
                operator_specs = []
                for regime, fit_values, whitening_stats, rank, target_fraction in (
                    (
                        "global",
                        features,
                        global_whitening,
                        global_operator.rank,
                        target_global_fraction,
                    ),
                    (
                        "source",
                        train_features,
                        source_whitening,
                        source_operator.rank,
                        target_source_fraction,
                    ),
                ):
                    rank_rng = np.random.default_rng(
                        stable_seed("same_rank", dataset, outer_seed, fold, regime, draw)
                    )
                    rank_operator = fit_same_rank_random(whitening_stats, rank, rank_rng)
                    operator_specs.append(
                        (f"{regime}_random_same_rank", rank_operator, fit_values, 0.0)
                    )
                    variance_rng = np.random.default_rng(
                        stable_seed("variance_match", dataset, outer_seed, fold, regime, draw)
                    )
                    variance_operator, _, match_error = fit_variance_matched_random(
                        fit_values, target_fraction, variance_rng
                    )
                    if match_error > VARIANCE_MATCH_TOLERANCE:
                        raise RuntimeError(
                            f"variance match failed {dataset}/{outer_seed}/{fold}/{regime}/{draw}: "
                            f"{match_error}"
                        )
                    operator_specs.append(
                        (
                            f"{regime}_random_variance_matched",
                            variance_operator,
                            fit_values,
                            match_error,
                        )
                    )

                for protocol, operator, fit_values, match_error in operator_specs:
                    fit_after = operator.apply(fit_values)
                    train_after = operator.apply(train_features)
                    test_after = operator.apply(test_features)
                    fresh_head = fit_task_head(train_after, train_labels, head_seed)
                    fresh = recording_metrics(
                        fresh_head,
                        test_after,
                        test_window_rec,
                        rec_labels,
                        test_recordings,
                    )
                    exact = recording_metrics(
                        original_head,
                        test_after,
                        test_window_rec,
                        rec_labels,
                        test_recordings,
                    )
                    random_rows.append(
                        {
                            "dataset": dataset,
                            "outer_seed": outer_seed,
                            "fold": fold,
                            "draw": draw,
                            "protocol": protocol,
                            "rank": operator.rank,
                            "paired_subject_rank": (
                                global_operator.rank
                                if protocol.startswith("global")
                                else source_operator.rank
                            ),
                            "fit_removed_energy_fraction": removed_energy_fraction(
                                fit_values, fit_after, operator.mu
                            ),
                            "source_removed_energy_fraction": removed_energy_fraction(
                                train_features, train_after, operator.mu
                            ),
                            "heldout_removed_energy_fraction": removed_energy_fraction(
                                test_features, test_after, operator.mu
                            ),
                            "variance_match_abs_error": match_error,
                            **{f"fresh_{key}": value for key, value in fresh.items()},
                            **{f"exact_{key}": value for key, value in exact.items()},
                        }
                    )

            if outer_seed == LABEL_SEEDS[0] and fold == 0:
                canaries.append(
                    {
                        "dataset": dataset,
                        "outer_seed": outer_seed,
                        "fold": fold,
                        "quantization_decimals": 8,
                        "global_subject_transformed_sha256": quantized_array_sha256(
                            global_operator.apply(features)
                        ),
                        "source_subject_train_transformed_sha256": quantized_array_sha256(
                            source_after_train
                        ),
                        "source_subject_test_transformed_sha256": quantized_array_sha256(
                            source_after_test
                        ),
                    }
                )

        return {
            "fresh": fresh_rows,
            "exact": exact_rows,
            "random": random_rows,
            "transfer": transfer_rows,
            "modifiers": modifier_rows,
            "folds": fold_rows,
            "canaries": canaries,
        }


def run_b0(fmscope_root: Path, caches: dict[str, Path]) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(fmscope_root))
    from fmscope.diagnostics.erasure import subject_axis_erasure

    rows = []
    for dataset, path in caches.items():
        data = load_cache(path, DATASETS[dataset]["shape"])
        result = subject_axis_erasure(
            data["features"], data["window_pids"], data["window_labels"]
        )
        row = {
            "dataset": dataset,
            "role": DATASETS[dataset]["role"],
            **asdict(result),
            "historical_bundled_delta": DATASETS[dataset]["historical_delta"],
            "live_minus_historical_delta": (
                result.label_ba_delta - DATASETS[dataset]["historical_delta"]
            ),
        }
        rows.append(row)
    eegmat = next(row for row in rows if row["dataset"] == "eegmat")
    sleepdep = next(row for row in rows if row["dataset"] == "sleepdep")
    eegmat_pass = bool(
        eegmat["rank_subject_axis"] == 35
        and eegmat["subj_ba_linear_post"] <= eegmat["chance"] + 0.01
        and np.isfinite(eegmat["subj_ba_mlp_post"])
        and eegmat["interpretable"]
        and eegmat["label_ba_delta"] > 0
        and abs(eegmat["live_minus_historical_delta"]) <= 0.01
    )
    sleepdep_pass = bool(
        sleepdep["rank_subject_axis"] == 35
        and np.isfinite(sleepdep["label_ba_delta"])
        and sleepdep["interpretable"]
    )
    return rows, {
        "eegmat_positive_reference_pass": eegmat_pass,
        "sleepdep_negative_control_execution_pass": sleepdep_pass,
        "b0_exact_replication_pass": eegmat_pass and sleepdep_pass,
    }


def verify_operator_equivalence(
    fmscope_root: Path, caches: dict[str, Path]
) -> dict:
    sys.path.insert(0, str(fmscope_root))
    from fmscope.diagnostics.erasure import (
        apply_eraser as public_apply_eraser,
        subject_eraser as public_subject_eraser,
        whiten as public_whiten,
    )

    rows = []
    for dataset, path in caches.items():
        data = load_cache(path, DATASETS[dataset]["shape"])
        features = data["features"].astype(np.float64)
        subjects = data["window_pids"]
        public_mu, public_centered, public_w, public_w_plus, _ = public_whiten(
            features
        )
        _, public_projection, public_rank = public_subject_eraser(
            public_centered, public_w, public_w_plus, subjects
        )
        public_after = public_apply_eraser(
            features, public_mu, public_projection
        )
        local_operator = fit_subject_leace(features, subjects)
        local_after = local_operator.apply(features)
        rows.append(
            {
                "dataset": dataset,
                "public_rank": int(public_rank),
                "local_rank": int(local_operator.rank),
                "transformed_max_abs_diff": float(
                    np.max(np.abs(public_after - local_after))
                ),
            }
        )
    passed = all(
        row["public_rank"] == row["local_rank"]
        and row["transformed_max_abs_diff"] <= 1e-10
        for row in rows
    )
    return {
        "public_commit": FMSCOPE_COMMIT,
        "tolerance": 1e-10,
        "datasets": rows,
        "operator_equivalence_pass": passed,
    }


def summarize_inference(fresh_rows, exact_rows, random_rows):
    summaries = []
    for dataset in DATASETS:
        for regime in ("global", "source"):
            subject_protocol = f"{regime}_subject_leace"
            for endpoint, rows, metric_prefix in (
                ("fresh_probe", fresh_rows, ""),
                ("exact_head", exact_rows, ""),
            ):
                relevant = [row for row in rows if row["dataset"] == dataset]
                baseline = {
                    (int(row["outer_seed"]), int(row["fold"])): float(
                        row["balanced_accuracy"]
                    )
                    for row in relevant
                    if row["protocol"] == "unchanged"
                }
                subject = {
                    (int(row["outer_seed"]), int(row["fold"])): float(
                        row["balanced_accuracy"]
                    )
                    for row in relevant
                    if row["protocol"] == subject_protocol
                }
                observed = float(np.mean([subject[key] - baseline[key] for key in sorted(baseline)]))
                null_payload = {}
                for null_kind in ("same_rank", "variance_matched"):
                    random_protocol = f"{regime}_random_{null_kind}"
                    draw_deltas = []
                    for draw in range(N_RANDOM):
                        current = []
                        for row in random_rows:
                            if (
                                row["dataset"] == dataset
                                and row["protocol"] == random_protocol
                                and int(row["draw"]) == draw
                            ):
                                key = (int(row["outer_seed"]), int(row["fold"]))
                                value = float(row[f"{metric_prefix}{'fresh' if endpoint == 'fresh_probe' else 'exact'}_balanced_accuracy"])
                                current.append(value - baseline[key])
                        if len(current) != len(baseline):
                            raise RuntimeError(
                                f"random null is incomplete for {dataset}/{regime}/{endpoint}/{draw}"
                            )
                        draw_deltas.append(float(np.mean(current)))
                    array = np.asarray(draw_deltas)
                    null_payload[null_kind] = {
                        "mean_delta_balanced_accuracy": float(array.mean()),
                        "std_delta_balanced_accuracy": float(array.std(ddof=1)),
                        "empirical_one_sided_p_random_ge_subject": float(
                            (1 + np.sum(array >= observed)) / (len(array) + 1)
                        ),
                    }
                summaries.append(
                    {
                        "dataset": dataset,
                        "information_regime": regime,
                        "endpoint": endpoint,
                        "subject_leace_mean_delta_balanced_accuracy": observed,
                        "same_rank_random": null_payload["same_rank"],
                        "variance_matched_random": null_payload["variance_matched"],
                    }
                )
    for null_kind in ("same_rank", "variance_matched"):
        pvalues = np.asarray(
            [
                row[f"{null_kind}_random"]["empirical_one_sided_p_random_ge_subject"]
                for row in summaries
            ],
            dtype=np.float64,
        )
        order = np.argsort(pvalues)
        adjusted = np.empty_like(pvalues)
        running = 0.0
        for position, index in enumerate(order):
            running = max(running, (len(pvalues) - position) * pvalues[index])
            adjusted[index] = min(running, 1.0)
        for index, value in enumerate(adjusted):
            summaries[index][f"{null_kind}_random"][
                "holm_adjusted_p_eight_cell_family"
            ] = float(value)
    for row in summaries:
        observed = row["subject_leace_mean_delta_balanced_accuracy"]
        same_pass = row["same_rank_random"][
            "holm_adjusted_p_eight_cell_family"
        ] <= 0.05
        variance_pass = row["variance_matched_random"][
            "holm_adjusted_p_eight_cell_family"
        ] <= 0.05
        if observed > 0 and same_pass and variance_pass:
            verdict = "IDENTITY_SPECIFIC_BENEFIT_SUPPORTED"
        elif observed > 0:
            verdict = "POSITIVE_NOT_FAMILYWISE_SPECIFIC"
        elif observed < 0:
            verdict = "REMOVAL_HARMS_ENDPOINT"
        else:
            verdict = "NO_ENDPOINT_CHANGE"
        row["identity_specificity_verdict"] = verdict
    by_key = {
        (row["dataset"], row["endpoint"], row["information_regime"]): row
        for row in summaries
    }
    gaps = []
    for dataset in DATASETS:
        for endpoint in ("fresh_probe", "exact_head"):
            global_value = by_key[(dataset, endpoint, "global")][
                "subject_leace_mean_delta_balanced_accuracy"
            ]
            source_value = by_key[(dataset, endpoint, "source")][
                "subject_leace_mean_delta_balanced_accuracy"
            ]
            gaps.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "global_minus_source_delta_balanced_accuracy": global_value - source_value,
                }
            )
    return {"cell_endpoint_results": summaries, "diagnostic_to_deployment_gaps": gaps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--fmscope-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/s2p_fmscope_fsr_bridge_panel1"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--random-repetitions", type=int, default=N_RANDOM)
    args = parser.parse_args()
    if args.random_repetitions != N_RANDOM:
        raise RuntimeError(f"frozen protocol requires exactly {N_RANDOM} random repetitions")
    repo_root = args.repo_root.resolve()
    fmscope_root = args.fmscope_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if git_head(fmscope_root) != FMSCOPE_COMMIT:
        raise RuntimeError("FMScope commit differs from frozen authority")

    caches = {}
    asset_rows = []
    for dataset, contract in DATASETS.items():
        path = fmscope_root / contract["cache"]
        actual_hash = sha256_file(path)
        if actual_hash != contract["sha256"]:
            raise RuntimeError(f"cache hash differs for {dataset}")
        data = load_cache(path, contract["shape"])
        caches[dataset] = path
        asset_rows.append(
            {
                "dataset": dataset,
                "role": contract["role"],
                "external_repository": FMSCOPE_URL,
                "external_commit": FMSCOPE_COMMIT,
                "cache_relative_path": contract["cache"],
                "cache_sha256": actual_hash,
                "cache_size_bytes": path.stat().st_size,
                "feature_rows": data["features"].shape[0],
                "feature_dim": data["features"].shape[1],
                "subjects": len(np.unique(data["window_pids"])),
                "recordings": len(data["rec_labels"]),
                "labels": len(np.unique(data["rec_labels"])),
                "raw_eeg_in_repository": False,
            }
        )
    write_csv(output_dir / "bridge_external_asset_manifest.csv", asset_rows)

    operator_equivalence = verify_operator_equivalence(fmscope_root, caches)
    write_json(
        output_dir / "bridge_operator_equivalence.json", operator_equivalence
    )
    if not operator_equivalence["operator_equivalence_pass"]:
        raise RuntimeError("local bridge LEACE operator differs from pinned public code")

    b0_rows, b0_gate = run_b0(fmscope_root, caches)
    write_csv(output_dir / "bridge_b0_exact_replication.csv", b0_rows)
    if not b0_gate["b0_exact_replication_pass"]:
        write_json(
            output_dir / "bridge_panel1_go_nogo.json",
            {
                **b0_gate,
                "panel1_compute_complete": False,
                "launch_panel2": False,
                "stop_reason": "B0_EXACT_REPLICATION_GATE_FAILED",
            },
        )
        raise RuntimeError("B0 exact-replication gate failed")

    tasks = [
        {
            "dataset": dataset,
            "outer_seed": outer_seed,
            "cache_path": str(caches[dataset]),
            "expected_shape": DATASETS[dataset]["shape"],
            "n_random": N_RANDOM,
        }
        for dataset in DATASETS
        for outer_seed in LABEL_SEEDS
    ]
    combined = {
        "fresh": [],
        "exact": [],
        "random": [],
        "transfer": [],
        "modifiers": [],
        "folds": [],
        "canaries": [],
    }
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for payload in executor.map(analyze_seed_task, tasks):
            for key in combined:
                combined[key].extend(payload[key])

    sort_key = lambda row: tuple(
        row.get(key, "")
        for key in ("dataset", "outer_seed", "fold", "protocol", "draw", "recording")
    )
    for rows in combined.values():
        rows.sort(key=sort_key)
    write_csv(output_dir / "bridge_fold_assignments.csv", combined["folds"])
    write_csv(output_dir / "bridge_fresh_probe_results.csv", combined["fresh"])
    write_csv(output_dir / "bridge_exact_head_results.csv", combined["exact"])
    write_csv(output_dir / "bridge_random_null_results.csv", combined["random"])
    write_csv(output_dir / "bridge_transferability.csv", combined["transfer"])
    write_csv(output_dir / "bridge_effect_modifiers.csv", combined["modifiers"])
    write_json(output_dir / "bridge_transform_canaries.json", {"canaries": combined["canaries"]})

    inference = summarize_inference(
        combined["fresh"], combined["exact"], combined["random"]
    )
    inference.update(
        {
            "primary_metric": "recording_balanced_accuracy",
            "fresh_probe_and_exact_head_separate": True,
            "random_repetitions": N_RANDOM,
            "same_rank_random_primary": True,
            "variance_matched_random_sensitivity": True,
            "no_cross_cell_pooled_inference": True,
            "multiplicity": "Holm across 8 dataset_x_regime_x_endpoint cells per null family",
        }
    )
    write_json(output_dir / "bridge_panel1_inference.json", inference)

    firewall = {
        "global_oracle_uses_complete_cohort_features": True,
        "global_oracle_uses_complete_cohort_subject_ids": True,
        "global_oracle_classification": "TRANSDUCTIVE_DIAGNOSTIC_ONLY",
        "source_only_eraser_uses_heldout_features": False,
        "source_only_eraser_uses_heldout_subject_ids": False,
        "source_task_head_uses_heldout_features_or_labels": False,
        "heldout_task_labels_use": "FROZEN_STRATIFIED_OUTER_SPLIT_AND_FINAL_SCORING_ONLY",
        "heldout_subject_ids_use": "POST_HOC_TRANSFERABILITY_DIAGNOSTIC_ONLY",
        "heldout_information_used_for_rank_or_null_selection": False,
        "best_seed_fold_draw_selection": False,
        "pass": True,
    }
    write_json(output_dir / "bridge_target_information_firewall.json", firewall)
    write_json(
        output_dir / "bridge_panel1_go_nogo.json",
        {
            **b0_gate,
            "external_assets_hash_pinned": True,
            "four_protocols_complete": True,
            "fresh_probe_complete": True,
            "exact_head_complete": True,
            "same_rank_random_draws_per_fold": N_RANDOM,
            "variance_matched_random_draws_per_fold": N_RANDOM,
            "variance_match_tolerance": VARIANCE_MATCH_TOLERANCE,
            "target_information_firewall_pass": True,
            "panel1_compute_complete": True,
            "independent_verifier_pass": False,
            "panel1_scientific_closure": False,
            "launch_panel2": False,
            "launch_phase_d1_training": False,
            "fine_tuning_used": False,
            "foundation_training_used": False,
        },
    )
    print(json.dumps({"b0": b0_gate, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
