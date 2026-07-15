#!/usr/bin/env python
"""Run one FACED or SEED-V Panel-2 representation object."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fmscope_fsr_bridge_panel1 import (
    fit_same_rank_random,
    fit_subject_leace,
    fit_variance_matched_random,
    removed_energy_fraction,
    stable_seed,
    subject_mean_scatter_removed,
    subspace_geometry,
    whiten,
)


N_RANDOM = 100
CLASS_TO_CLIPS = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10, 11],
    4: [12, 13, 14, 15],
    5: [16, 17, 18],
    6: [19, 20, 21],
    7: [22, 23, 24],
    8: [25, 26, 27],
}
CLIP_TO_FOLD = {
    clip: index % 3 for clips in CLASS_TO_CLIPS.values() for index, clip in enumerate(clips)
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode() + b"\0")
    digest.update(str(tuple(value.shape)).encode() + b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_feature_payload(dataset, path):
    with np.load(path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    if dataset == "faced":
        required = {"features", "labels", "subjects", "clips", "segments", "splits", "keys"}
        if set(data) != required or data["features"].shape[1] != 200:
            raise RuntimeError("FACED Panel-2 payload contract mismatch")
        unit_ids = np.asarray(
            [f"s{subject:03d}_c{clip:02d}" for subject, clip in zip(data["subjects"], data["clips"])]
        )
        decode_folds = np.asarray([CLIP_TO_FOLD[int(clip)] for clip in data["clips"]])
        n_classes = 9
    elif dataset == "seedv":
        required = {"features", "labels", "subjects", "sessions", "splits", "trial_ids", "window_counts"}
        if set(data) != required or data["features"].shape[1] % 200:
            raise RuntimeError("SEED-V Panel-2 payload contract mismatch")
        channels = data["features"].shape[1] // 200
        data["features"] = np.ascontiguousarray(
            data["features"].reshape(len(data["features"]), channels, 200).mean(axis=1)
        )
        unit_ids = data["trial_ids"].astype(str)
        decode_folds = data["splits"].astype(str)
        n_classes = 5
    else:
        raise ValueError(dataset)
    if data["features"].shape[1] != 200 or not np.isfinite(data["features"]).all():
        raise RuntimeError("pooled feature shape/finiteness failed")
    return {
        **data,
        "features": data["features"].astype(np.float64),
        "labels": data["labels"].astype(int),
        "subjects": data["subjects"].astype(int),
        "unit_ids": unit_ids,
        "decode_folds": decode_folds,
        "n_classes": n_classes,
    }


@dataclass
class MulticlassHead:
    low: np.ndarray
    high: np.ndarray
    scaler: StandardScaler
    classifier: LogisticRegression

    def probabilities(self, features):
        clipped = np.clip(np.asarray(features, dtype=np.float64), self.low, self.high)
        return self.classifier.predict_proba(self.scaler.transform(clipped))


def fit_head(features, labels, seed):
    values = np.asarray(features, dtype=np.float64)
    low = np.percentile(values, 1, axis=0)
    high = np.percentile(values, 99, axis=0)
    clipped = np.clip(values, low, high)
    scaler = StandardScaler().fit(clipped)
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        tol=1e-5,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        classifier.fit(scaler.transform(clipped), labels)
    return MulticlassHead(low, high, scaler, classifier)


def pooled_unit_predictions(head, features, labels, subjects, unit_ids, n_classes):
    probabilities = head.probabilities(features)
    ordered = np.unique(unit_ids)
    pooled = []
    unit_labels = []
    unit_subjects = []
    for unit in ordered:
        mask = unit_ids == unit
        label_values = np.unique(labels[mask])
        subject_values = np.unique(subjects[mask])
        if len(label_values) != 1 or len(subject_values) != 1:
            raise RuntimeError(f"task unit is not label/subject constant: {unit}")
        pooled.append(probabilities[mask].mean(axis=0))
        unit_labels.append(int(label_values[0]))
        unit_subjects.append(int(subject_values[0]))
    result = np.asarray(pooled)
    if result.shape != (len(ordered), n_classes):
        raise RuntimeError("pooled probability shape mismatch")
    return result, np.asarray(unit_labels), np.asarray(unit_subjects), ordered


def task_metrics(head, features, labels, subjects, unit_ids, n_classes):
    probabilities, pooled_labels, pooled_subjects, pooled_units = pooled_unit_predictions(
        head, features, labels, subjects, unit_ids, n_classes
    )
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "cohen_kappa": float(cohen_kappa_score(pooled_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(pooled_labels, predictions)),
        "nll": float(log_loss(pooled_labels, probabilities, labels=list(range(n_classes)))),
        "n_units": len(pooled_labels),
        "prediction_sha256": array_sha(probabilities),
    }
    support = {
        "probabilities": np.ascontiguousarray(probabilities.astype(np.float64)),
        "labels": pooled_labels.astype(np.int64),
        "subjects": pooled_subjects.astype(np.int64),
        "units": pooled_units,
    }
    return metrics, support


def discriminant_basis(features, labels, rank):
    values = np.asarray(features, dtype=np.float64)
    classes = np.unique(labels)
    means = np.stack([values[labels == label].mean(axis=0) for label in classes])
    means -= means.mean(axis=0)
    _, singular, vt = np.linalg.svd(means, full_matrices=False)
    if len(singular) < rank or singular[rank - 1] <= 1e-10 * singular[0]:
        raise RuntimeError("task discriminant subspace is rank deficient")
    return vt[:rank].T


def projection_overlap(left, right):
    values = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.sum(np.clip(values, 0, 1) ** 2) / right.shape[1])


def target_subject_decode(dataset, features, transformed, data, train_mask, test_mask):
    scores = {"pre": [], "post": []}
    if dataset == "faced":
        for fold in range(3):
            fit = test_mask & (data["decode_folds"] != fold)
            hold = test_mask & (data["decode_folds"] == fold)
            for name, values in (("pre", features), ("post", transformed)):
                scaler = StandardScaler().fit(values[fit])
                clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
                clf.fit(scaler.transform(values[fit]), data["subjects"][fit])
                scores[name].append(
                    balanced_accuracy_score(
                        data["subjects"][hold], clf.predict(scaler.transform(values[hold]))
                    )
                )
    else:
        for name, values in (("pre", features), ("post", transformed)):
            scaler = StandardScaler().fit(values[train_mask])
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(scaler.transform(values[train_mask]), data["subjects"][train_mask])
            scores[name].append(
                balanced_accuracy_score(
                    data["subjects"][test_mask], clf.predict(scaler.transform(values[test_mask]))
                )
            )
    return float(np.mean(scores["pre"])), float(np.mean(scores["post"]))


def metric_row(tag, dataset, endpoint, regime, removal, rank, metric, baseline, energy):
    return {
        "tag": tag,
        "dataset": dataset,
        "endpoint": endpoint,
        "information_regime": regime,
        "removal_kind": removal,
        "rank": rank,
        **metric,
        "delta_kappa_vs_unchanged": metric["cohen_kappa"] - baseline["cohen_kappa"],
        "delta_bacc_vs_unchanged": metric["balanced_accuracy"] - baseline["balanced_accuracy"],
        "delta_nll_vs_unchanged": metric["nll"] - baseline["nll"],
        **energy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("faced", "seedv"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--feature-payload", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-random", type=int, default=N_RANDOM)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if args.canary and args.n_random != 2:
        raise RuntimeError("Panel-2 object canary requires exactly two random draws")
    if not args.canary and args.n_random != N_RANDOM:
        raise RuntimeError("Panel-2 FACED/SEED-V requires exactly 100 random draws")
    data = load_feature_payload(args.dataset, args.feature_payload)
    features = data["features"]
    labels = data["labels"]
    subjects = data["subjects"]
    splits = data["splits"].astype(str)
    train = splits == ("source_train" if args.dataset == "faced" else "train")
    val = splits == ("source_val" if args.dataset == "faced" else "val")
    test = splits == ("target_test" if args.dataset == "faced" else "test")
    if not train.any() or not val.any() or not test.any():
        raise RuntimeError("dataset split masks are incomplete")
    n_classes = data["n_classes"]
    task_seed = stable_seed("panel2_task_head", args.dataset, args.tag)

    with threadpool_limits(limits=1):
        original_head = fit_head(features[train], labels[train], task_seed)
        baseline_test, baseline_support = task_metrics(
            original_head,
            features[test],
            labels[test],
            subjects[test],
            data["unit_ids"][test],
            n_classes,
        )
        baseline_val, _ = task_metrics(
            original_head,
            features[val],
            labels[val],
            subjects[val],
            data["unit_ids"][val],
            n_classes,
        )
        metrics_rows = []
        null_rows = []
        transfer_rows = []
        modifier_rows = []
        support = {
            "baseline_probabilities": baseline_support["probabilities"],
            "unit_labels": baseline_support["labels"],
            "unit_subjects": baseline_support["subjects"],
            "unit_ids": baseline_support["units"],
        }
        baseline_energy = {
            "fit_removed_energy_fraction": 0.0,
            "source_removed_energy_fraction": 0.0,
            "target_removed_energy_fraction": 0.0,
        }
        for endpoint in ("fresh_head", "exact_head"):
            metrics_rows.append(
                metric_row(
                    args.tag,
                    args.dataset,
                    endpoint,
                    "source_only",
                    "unchanged",
                    0,
                    baseline_test,
                    baseline_test,
                    baseline_energy,
                )
            )

        global_stats = whiten(features)
        source_stats = whiten(features[train])
        operators = {
            "global_oracle": (fit_subject_leace(features, subjects, global_stats), features, global_stats),
            "source_only": (
                fit_subject_leace(features[train], subjects[train], source_stats),
                features[train],
                source_stats,
            ),
        }
        target_operator = fit_subject_leace(features[test], subjects[test])
        for regime, (operator, fit_features, whitening_stats) in operators.items():
            transformed = operator.apply(features)
            transformed_train = transformed[train]
            transformed_test = transformed[test]
            fresh_head = fit_head(transformed_train, labels[train], task_seed)
            fresh, fresh_support = task_metrics(
                fresh_head,
                transformed_test,
                labels[test],
                subjects[test],
                data["unit_ids"][test],
                n_classes,
            )
            exact, exact_support = task_metrics(
                original_head,
                transformed_test,
                labels[test],
                subjects[test],
                data["unit_ids"][test],
                n_classes,
            )
            energy = {
                "fit_removed_energy_fraction": removed_energy_fraction(
                    fit_features, operator.apply(fit_features), operator.mu
                ),
                "source_removed_energy_fraction": removed_energy_fraction(
                    features[train], transformed_train, operator.mu
                ),
                "target_removed_energy_fraction": removed_energy_fraction(
                    features[test], transformed_test, operator.mu
                ),
            }
            metrics_rows.append(
                metric_row(
                    args.tag,
                    args.dataset,
                    "fresh_head",
                    regime,
                    "subject_leace",
                    operator.rank,
                    fresh,
                    baseline_test,
                    energy,
                )
            )
            metrics_rows.append(
                metric_row(
                    args.tag,
                    args.dataset,
                    "exact_head",
                    regime,
                    "subject_leace",
                    operator.rank,
                    exact,
                    baseline_test,
                    energy,
                )
            )
            support[f"{regime}_fresh_probabilities"] = fresh_support["probabilities"]
            support[f"{regime}_exact_probabilities"] = exact_support["probabilities"]

            if regime == "source_only":
                pre_decode, post_decode = target_subject_decode(
                    args.dataset, features, transformed, data, train, test
                )
                geometry = subspace_geometry(operator.basis, target_operator.basis)
                transfer_rows.append(
                    {
                        "tag": args.tag,
                        "dataset": args.dataset,
                        "information_regime": regime,
                        "source_axis_rank": operator.rank,
                        "target_axis_rank": target_operator.rank,
                        **geometry,
                        "target_subject_scatter_removed_fraction": subject_mean_scatter_removed(
                            features[test], transformed_test, subjects[test]
                        ),
                        "target_subject_decode_pre": pre_decode,
                        "target_subject_decode_post": post_decode,
                        **energy,
                    }
                )

            task_basis = discriminant_basis(features[train], labels[train], n_classes - 1)
            modifier_rows.append(
                {
                    "tag": args.tag,
                    "dataset": args.dataset,
                    "information_regime": regime,
                    "subject_task_projection_overlap": projection_overlap(operator.basis, task_basis),
                    "task_subspace_rank": n_classes - 1,
                    "subject_subspace_rank": operator.rank,
                    "baseline_source_val_kappa": baseline_val["cohen_kappa"],
                    "baseline_target_kappa": baseline_test["cohen_kappa"],
                }
            )

            target_fraction = energy["fit_removed_energy_fraction"]
            for draw in range(args.n_random):
                same_rng = np.random.default_rng(
                    stable_seed("panel2_same_rank", args.dataset, args.tag, regime, draw)
                )
                variance_rng = np.random.default_rng(
                    stable_seed("panel2_variance", args.dataset, args.tag, regime, draw)
                )
                random_operators = {
                    "same_rank_random": (
                        fit_same_rank_random(whitening_stats, operator.rank, same_rng),
                        0.0,
                    ),
                    "variance_matched_random": fit_variance_matched_random(
                        fit_features, target_fraction, variance_rng
                    )[:2],
                }
                for null_kind, (random_operator, match_error) in random_operators.items():
                    random_transformed = random_operator.apply(features)
                    random_head = fit_head(
                        random_transformed[train], labels[train], task_seed
                    )
                    random_fresh, _ = task_metrics(
                        random_head,
                        random_transformed[test],
                        labels[test],
                        subjects[test],
                        data["unit_ids"][test],
                        n_classes,
                    )
                    random_exact, _ = task_metrics(
                        original_head,
                        random_transformed[test],
                        labels[test],
                        subjects[test],
                        data["unit_ids"][test],
                        n_classes,
                    )
                    achieved = removed_energy_fraction(
                        fit_features,
                        random_operator.apply(fit_features),
                        random_operator.mu,
                    )
                    for endpoint, metric in (
                        ("fresh_head", random_fresh),
                        ("exact_head", random_exact),
                    ):
                        null_rows.append(
                            {
                                "tag": args.tag,
                                "dataset": args.dataset,
                                "draw": draw,
                                "endpoint": endpoint,
                                "information_regime": regime,
                                "removal_kind": null_kind,
                                "rank": random_operator.rank,
                                "fit_removed_energy_fraction": achieved,
                                "variance_match_abs_error": abs(achieved - target_fraction)
                                if null_kind == "variance_matched_random"
                                else match_error,
                                **metric,
                                "delta_kappa_vs_unchanged": metric["cohen_kappa"]
                                - baseline_test["cohen_kappa"],
                                "delta_bacc_vs_unchanged": metric["balanced_accuracy"]
                                - baseline_test["balanced_accuracy"],
                                "delta_nll_vs_unchanged": metric["nll"] - baseline_test["nll"],
                            }
                        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.tag}_panel2_metrics.csv", metrics_rows)
    write_csv(args.out_dir / f"{args.tag}_panel2_random_null.csv", null_rows)
    write_csv(args.out_dir / f"{args.tag}_panel2_transferability.csv", transfer_rows)
    write_csv(args.out_dir / f"{args.tag}_panel2_modifiers.csv", modifier_rows)
    np.savez_compressed(args.out_dir / f"{args.tag}_panel2_support.npz", **support)
    contract = {
        "phase": "FMScope_FSR_Bridge_Panel2",
        "dataset": args.dataset,
        "tag": args.tag,
        "representation": "final_channel_and_patch_mean_200d",
        "feature_payload": (
            "${PANEL2_FEATURE_ROOT}/" if args.dataset == "faced" else "${PHASE_C_FEATURE_ROOT}/"
        )
        + args.feature_payload.name,
        "feature_payload_sha256": sha256_file(args.feature_payload),
        "feature_shape": list(features.shape),
        "random_draws_per_null": args.n_random,
        "canary": args.canary,
        "four_arm_contract": True,
        "fresh_and_exact_separate": True,
        "target_labels_used_for_selection": False,
        "status": "PASS_CANARY" if args.canary else "PASS_OBJECT",
    }
    write_json(args.out_dir / f"{args.tag}_panel2_contract.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
