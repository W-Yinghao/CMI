#!/usr/bin/env python
"""Confirmatory subject-cluster bootstrap for the Phase-C SEED-V fleet."""

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, log_loss

from route_b_cross_task_common import (
    TAGS,
    canonical_sha,
    manifest_row,
    read_csv,
    sha256_file,
    validate_manifest,
    write_csv,
    write_json,
)


PCA_DIM = 128
PROBE_C = 1.0
RANK = 4
N_CLASSES = 5
N_SUBJECTS = 16
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260714
N_NULL = 200
NULL_SEED = 20260731
MATCH_TOLERANCE = 1e-10
CONTINUOUS_RECOMPUTE_ATOL = 1e-4
GEOMETRY_RECOMPUTE_ATOL = 1e-5
EMPIRICAL_P_RECOMPUTE_ATOL = 1.0 / (N_NULL + 1) + 1e-12
HIGH_BUDGETS = (500, 1000, 2000)


def close(left, right, atol=1e-10):
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=0.0))


def budget_tags(budget):
    return ["H%d_s0" % budget, "H%d_s1" % budget]


def fixed_probe(x, y):
    pca = PCA(
        n_components=PCA_DIM,
        whiten=True,
        svd_solver="randomized",
        random_state=0,
    )
    z = pca.fit_transform(x)
    clf = LogisticRegression(
        C=PROBE_C,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-6,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(z, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("frozen SEED-V logistic probe did not converge")
    return pca, clf


def load_cache(feature_dir, tag, checkpoint_row):
    contract_path = feature_dir / (tag + "_feature_contract.json")
    payload_path = feature_dir / (tag + "_features.npz")
    contract = json.loads(contract_path.read_text())
    if (
        contract.get("status") != "PASS"
        or contract.get("dataset") != "SEED-V"
        or contract.get("tag") != tag
        or contract.get("checkpoint_sha256_before") != checkpoint_row["immutable_sha256"]
        or contract.get("checkpoint_sha256_after") != checkpoint_row["immutable_sha256"]
        or contract.get("canary_repeat_max_abs_diff") != 0.0
        or contract.get("encoder_frozen") is not True
        or contract.get("fine_tuning_used") is not False
        or sha256_file(payload_path) != contract.get("payload_sha256")
    ):
        raise RuntimeError("SEED-V feature contract failed for %s" % tag)
    with np.load(payload_path, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    if data["features"].shape != (705, 12400):
        raise RuntimeError("SEED-V feature shape mismatch for %s" % tag)
    return data, contract


def probabilities_and_margin(clf, z, labels):
    probability = clf.predict_proba(z)
    prediction = clf.predict(z)
    decision = clf.decision_function(z)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    class_to_index = {int(value): index for index, value in enumerate(clf.classes_)}
    true_index = np.asarray([class_to_index[int(value)] for value in labels])
    masked = decision.copy()
    masked[np.arange(len(labels)), true_index] = -np.inf
    margin = decision[np.arange(len(labels)), true_index] - masked.max(axis=1)
    nll = -np.log(np.clip(probability[np.arange(len(labels)), true_index], 1e-15, 1.0))
    return probability, prediction, nll, margin


def point_metrics(labels, probability):
    prediction = probability.argmax(axis=1)
    return {
        "kappa": float(cohen_kappa_score(labels, prediction)),
        "nll": float(log_loss(labels, probability, labels=np.arange(N_CLASSES))),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def subject_confusion(labels, predictions, subject_index):
    result = np.zeros((N_SUBJECTS, N_CLASSES, N_CLASSES), dtype=np.float64)
    for subject in range(N_SUBJECTS):
        mask = subject_index == subject
        np.add.at(result[subject], (labels[mask], predictions[mask]), 1.0)
    return result


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    total = confusion.sum(axis=(-2, -1))
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1).sum(axis=-1)
    observed = np.divide(diagonal, total, out=np.zeros_like(diagonal), where=total > 0)
    row = confusion.sum(axis=-1)
    column = confusion.sum(axis=-2)
    expected = np.divide(
        (row * column).sum(axis=-1),
        total ** 2,
        out=np.zeros_like(total),
        where=total > 0,
    )
    kappa = np.divide(
        observed - expected,
        1.0 - expected,
        out=np.zeros_like(observed),
        where=np.abs(1.0 - expected) > 1e-15,
    )
    recall = np.divide(
        np.diagonal(confusion, axis1=-2, axis2=-1),
        row,
        out=np.zeros_like(row),
        where=row > 0,
    )
    bacc = recall.mean(axis=-1)
    return kappa, bacc


def subject_nll_support(features, subjects, splits):
    sums = np.zeros(N_SUBJECTS, dtype=np.float64)
    counts = np.zeros(N_SUBJECTS, dtype=np.float64)
    for holdout in ("train", "val", "test"):
        fit = splits != holdout
        hold = splits == holdout
        pca, clf = fixed_probe(features[fit], subjects[fit])
        probability = clf.predict_proba(pca.transform(features[hold]))
        class_to_index = {int(value): index for index, value in enumerate(clf.classes_)}
        true_index = np.asarray([class_to_index[int(value)] for value in subjects[hold]])
        values = -np.log(np.clip(probability[np.arange(hold.sum()), true_index], 1e-15, 1.0))
        held_subjects = subjects[hold].astype(int) - 1
        np.add.at(sums, held_subjects, values)
        np.add.at(counts, held_subjects, 1.0)
    if (counts == 0).any():
        raise RuntimeError("SEED-V subject NLL support lost a subject")
    return sums, counts


def effect_cells(z, subjects, labels):
    cells = np.empty((N_SUBJECTS, N_CLASSES, z.shape[1]), dtype=np.float64)
    for subject in range(N_SUBJECTS):
        for label in range(N_CLASSES):
            mask = (subjects == subject + 1) & (labels == label)
            if not mask.any():
                raise RuntimeError("missing SEED-V subject-class geometry cell")
            cells[subject, label] = z[mask].mean(axis=0)
    return cells


def top_basis(matrix):
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if len(singular) < RANK or singular[RANK - 1] <= 1e-12:
        raise RuntimeError("rank-4 SEED-V geometry is degenerate")
    return vt[:RANK]


def geometry_from_cells(cells, subject_counts=None):
    if subject_counts is None:
        subject_counts = np.ones(N_SUBJECTS, dtype=np.float64)
    subject_counts = np.asarray(subject_counts, dtype=np.float64)
    if subject_counts.sum() <= 0:
        raise RuntimeError("empty subject bootstrap")
    weights = subject_counts / subject_counts.sum()
    subject_means = cells.mean(axis=1)
    task_means = np.einsum("s,sck->ck", weights, cells, optimize=True)
    grand = np.einsum("s,sk->k", weights, subject_means, optimize=True)
    subject_effect = (subject_means - grand) * np.sqrt(subject_counts[:, None])
    task_effect = task_means - grand
    subject_basis = top_basis(subject_effect)
    task_basis = top_basis(task_effect)
    overlap = float(np.sum((subject_basis @ task_basis.T) ** 2) / RANK)
    return overlap, subject_basis, task_basis


def captured_energy(cells, basis):
    subject_means = cells.mean(axis=1)
    grand = subject_means.mean(axis=0)
    effect = subject_means - grand
    return float(np.sum((effect @ basis.T) ** 2) / (np.sum(effect ** 2) + 1e-12))


def task_captured_energy(cells, basis):
    task_means = cells.mean(axis=0)
    grand = task_means.mean(axis=0)
    effect = task_means - grand
    return float(np.sum((effect @ basis.T) ** 2) / (np.sum(effect ** 2) + 1e-12))


def remove_subspace(z, basis):
    return z - (z @ basis.T) @ basis


def removed_fraction(z, after):
    return float(np.sum((z - after) ** 2) / (np.sum(z ** 2) + 1e-12))


def subject_basis(z, subjects):
    means = np.stack([z[subjects == value].mean(axis=0) for value in sorted(np.unique(subjects))])
    return top_basis(means - means.mean(axis=0))


def l5_support(clf, z_val, z_test, labels, subjects, basis):
    base_prediction = clf.predict(z_test)
    subject_after = remove_subspace(z_test, basis)
    subject_prediction = clf.predict(subject_after)
    subject_val = remove_subspace(z_val, basis)
    target_fraction = removed_fraction(z_val, subject_val)
    subject_index = subjects.astype(int) - 1
    base_confusion = subject_confusion(labels, base_prediction, subject_index)
    erased_confusion = subject_confusion(labels, subject_prediction, subject_index)
    null_confusion = np.empty(
        (N_NULL, N_SUBJECTS, N_CLASSES, N_CLASSES), dtype=np.float64
    )
    match_error = np.empty(N_NULL, dtype=np.float64)
    rng = np.random.default_rng(NULL_SEED)
    total_energy = float(np.sum(z_val ** 2)) + 1e-12
    for null_index in range(N_NULL):
        q, _ = np.linalg.qr(rng.standard_normal((z_val.shape[1], z_val.shape[1])))
        direction_energy = np.sum((z_val @ q) ** 2, axis=0) / total_energy
        cumulative = np.cumsum(direction_energy)
        last = int(np.searchsorted(cumulative, target_fraction, side="left"))
        if last >= z_val.shape[1]:
            raise RuntimeError("random basis cannot match removed variance")
        previous = float(cumulative[last - 1]) if last else 0.0
        alpha = np.sqrt(
            max(target_fraction - previous, 0.0)
            / max(float(direction_energy[last]), 1e-12)
        )
        alpha = min(float(alpha), 1.0)
        directions = q[:, : last + 1]
        coefficient = np.ones(last + 1)
        coefficient[-1] = alpha

        def erase(z):
            return z - ((z @ directions) * coefficient) @ directions.T

        val_after = erase(z_val)
        test_prediction = clf.predict(erase(z_test))
        match_error[null_index] = abs(removed_fraction(z_val, val_after) - target_fraction)
        null_confusion[null_index] = subject_confusion(
            labels, test_prediction, subject_index
        )
    if match_error.max() > MATCH_TOLERANCE:
        raise RuntimeError("SEED-V L5 removed-energy match failed")
    return base_confusion, erased_confusion, null_confusion, target_fraction, match_error


def bootstrap_l5(counts, base, erased, nulls):
    base_boot = np.einsum("bs,sij->bij", counts, base, optimize=True)
    erased_boot = np.einsum("bs,sij->bij", counts, erased, optimize=True)
    base_kappa = metrics_from_confusion(base_boot)[0]
    erased_kappa = metrics_from_confusion(erased_boot)[0]
    subject_delta = base_kappa - erased_kappa
    null_mean = np.zeros(len(counts), dtype=np.float64)
    chunk = 100
    for start in range(0, len(counts), chunk):
        stop = min(start + chunk, len(counts))
        null_boot = np.einsum(
            "bs,nsij->bnij", counts[start:stop], nulls, optimize=True
        )
        null_kappa = metrics_from_confusion(null_boot)[0]
        null_mean[start:stop] = (
            base_kappa[start:stop, None] - null_kappa
        ).mean(axis=1)
    return subject_delta - null_mean


def summarize(point, samples, alternative="two_sided"):
    samples = np.asarray(samples, dtype=np.float64)
    if alternative == "greater":
        pvalue = float((1 + np.sum(samples <= 0.0)) / (len(samples) + 1))
    elif alternative == "less":
        pvalue = float((1 + np.sum(samples >= 0.0)) / (len(samples) + 1))
    else:
        pvalue = float(
            min(
                1.0,
                2.0
                * min(
                    (1 + np.sum(samples <= 0.0)) / (len(samples) + 1),
                    (1 + np.sum(samples >= 0.0)) / (len(samples) + 1),
                ),
            )
        )
    return {
        "point": float(point),
        "bootstrap_mean": float(samples.mean()),
        "bootstrap_se": float(samples.std(ddof=1)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_raw": pvalue,
        "alternative": alternative,
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def holm(pvalues):
    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def apply_holm(rows):
    adjusted = holm([row["p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = float(value)
        row["reject_holm_0p05"] = bool(value < 0.05)


def mean_arrays(values, tags):
    return np.mean([values[tag] for tag in tags], axis=0)


def budget_mean(values, budget):
    return mean_arrays(values, budget_tags(budget))


def pooled_high(values):
    return np.mean([budget_mean(values, budget) for budget in HIGH_BUDGETS], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=Path(
            "results/s2p_route_b_phase_b_checkpoint_closure/"
            "phase_b_checkpoint_immutable_manifest.csv"
        ),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("results/s2p_route_b_cross_task_seedv/features"),
    )
    parser.add_argument(
        "--fleet-dir",
        type=Path,
        default=Path("results/s2p_route_b_cross_task_seedv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/s2p_route_b_phase_c_closure"),
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if args.code_commit != actual_commit or len(actual_commit) != 40:
        raise RuntimeError("SEED-V bootstrap code-commit contract failed")

    manifest = validate_manifest(args.checkpoint_manifest)
    fleet_task = {
        row["tag"]: row
        for row in read_csv(args.fleet_dir / "seedv_fleet_task_performance.csv")
    }
    fleet_subject = {
        row["tag"]: row
        for row in read_csv(args.fleet_dir / "seedv_fleet_subject_metrics.csv")
    }
    fleet_geometry = {
        row["tag"]: row
        for row in read_csv(args.fleet_dir / "seedv_fleet_geometry.csv")
    }
    fleet_l5 = {
        row["tag"]: row
        for row in read_csv(args.fleet_dir / "seedv_fleet_l5_reliance.csv")
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = rng.multinomial(
        N_SUBJECTS,
        np.full(N_SUBJECTS, 1.0 / N_SUBJECTS),
        size=N_BOOTSTRAP,
    ).astype(np.float64)
    task_kappa = {}
    task_nll = {}
    task_bacc = {}
    subject_nll = {}
    overlap = {}
    point = {}
    support = {
        "tags": np.asarray(TAGS),
        "subject_counts": counts.astype(np.int16),
    }
    l5_rows = []
    identity_sha = None
    payload_hashes = {}
    point_checks = []

    for tag_index, tag in enumerate(TAGS):
        checkpoint_row = manifest_row(manifest, tag)
        data, contract = load_cache(args.feature_dir, tag, checkpoint_row)
        payload_hashes[tag] = contract["payload_sha256"]
        current_identity = canonical_sha(
            {
                key: data[key].tolist()
                for key in (
                    "labels",
                    "subjects",
                    "sessions",
                    "splits",
                    "trial_ids",
                    "window_counts",
                )
            }
        )
        if identity_sha is None:
            identity_sha = current_identity
        elif current_identity != identity_sha:
            raise RuntimeError("SEED-V identities differ across representation objects")

        x = data["features"]
        y = data["labels"].astype(int)
        subjects = data["subjects"].astype(int)
        subject_index = subjects - 1
        splits = data["splits"].astype(str)
        train = splits == "train"
        val = splits == "val"
        test = splits == "test"
        # One of 48 subject-sessions is unavailable. Its identity is pinned by
        # the manifest, while this assertion checks the aggregate 47-session shape.
        if sorted(np.bincount(subject_index, minlength=N_SUBJECTS)) != [30] + [45] * 15:
            raise RuntimeError("SEED-V subject/session support differs from contract")

        pca, clf = fixed_probe(x[train], y[train])
        z_train = pca.transform(x[train])
        z_val = pca.transform(x[val])
        z_test = pca.transform(x[test])
        test_probability, test_prediction, test_nll_values, _ = probabilities_and_margin(
            clf, z_test, y[test]
        )
        val_probability, _, _, _ = probabilities_and_margin(clf, z_val, y[val])
        test_point = point_metrics(y[test], test_probability)
        val_point = point_metrics(y[val], val_probability)
        confusion = subject_confusion(y[test], test_prediction, subject_index[test])
        nll_sum = np.bincount(
            subject_index[test], weights=test_nll_values, minlength=N_SUBJECTS
        )
        nll_count = np.bincount(subject_index[test], minlength=N_SUBJECTS).astype(float)
        boot_confusion = np.einsum("bs,sij->bij", counts, confusion, optimize=True)
        task_kappa[tag], task_bacc[tag] = metrics_from_confusion(boot_confusion)
        task_nll[tag] = (counts @ nll_sum) / (counts @ nll_count)

        subject_sum, subject_count = subject_nll_support(x, subjects, splits)
        subject_nll[tag] = (counts @ subject_sum) / (counts @ subject_count)

        train_cells = effect_cells(z_train, subjects[train], y[train])
        val_cells = effect_cells(z_val, subjects[val], y[val])
        point_overlap, subject_direction, task_direction = geometry_from_cells(train_cells)
        subject_capture = captured_energy(val_cells, subject_direction)
        task_capture = task_captured_energy(val_cells, task_direction)
        if tag != "random" and (subject_capture < 0.05 or task_capture < 0.05):
            raise RuntimeError("pretrained SEED-V geometry stability gate failed for %s" % tag)
        overlap[tag] = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for replicate in range(N_BOOTSTRAP):
            overlap[tag][replicate] = geometry_from_cells(
                train_cells, counts[replicate]
            )[0]

        task_gate = (
            val_point["kappa"] >= 0.05
            and val_point["kappa"]
            >= float(fleet_task["random"]["source_val_kappa"]) + 0.02
        )
        if task_gate:
            basis = subject_basis(z_train, subjects[train])
            base_confusion, erased_confusion, null_confusion, removed, match_error = l5_support(
                clf, z_val, z_test, y[test], subjects[test], basis
            )
            base_kappa = float(metrics_from_confusion(base_confusion.sum(axis=0))[0])
            erased_kappa = float(metrics_from_confusion(erased_confusion.sum(axis=0))[0])
            null_kappa = metrics_from_confusion(null_confusion.sum(axis=1))[0]
            subject_delta = base_kappa - erased_kappa
            null_delta = base_kappa - null_kappa
            l5_bootstrap = bootstrap_l5(
                counts, base_confusion, erased_confusion, null_confusion
            )
            l5_row = {
                "tag": tag,
                "task_gate_pass": True,
                "subject_delta_kappa": subject_delta,
                "null_delta_kappa_mean": float(null_delta.mean()),
                "subject_minus_null_kappa": float(subject_delta - null_delta.mean()),
                "empirical_one_sided_p": float(
                    (1 + np.sum(null_delta >= subject_delta)) / (N_NULL + 1)
                ),
                "source_val_removed_variance_fraction": removed,
                "null_match_abs_error_max": float(match_error.max()),
                "null_repetitions": N_NULL,
                **{
                    "cluster_" + key: value
                    for key, value in summarize(
                        subject_delta - null_delta.mean(), l5_bootstrap, "greater"
                    ).items()
                },
            }
            l5_rows.append(l5_row)
            support["l5_bootstrap__" + tag] = l5_bootstrap

        point[tag] = {
            "target_kappa": test_point["kappa"],
            "target_nll": test_point["nll"],
            "target_bacc": test_point["balanced_accuracy"],
            "subject_nll": float(subject_sum.sum() / subject_count.sum()),
            "overlap": point_overlap,
            "subject_capture": subject_capture,
            "task_capture": task_capture,
        }
        expected_values = {
            "target_kappa": float(fleet_task[tag]["target_test_kappa"]),
            "target_nll": float(fleet_task[tag]["target_test_nll"]),
            "target_bacc": float(fleet_task[tag]["target_test_balanced_accuracy"]),
            "subject_nll": float(fleet_subject[tag]["subject_nll"]),
            "overlap": float(fleet_geometry[tag]["projection_overlap_rank4"]),
            "subject_capture": float(
                fleet_geometry[tag]["subject_heldout_captured_energy"]
            ),
            "task_capture": float(fleet_geometry[tag]["task_heldout_captured_energy"]),
        }
        checks = {
            "target_kappa": close(test_point["kappa"], fleet_task[tag]["target_test_kappa"]),
            "target_nll": close(
                test_point["nll"],
                fleet_task[tag]["target_test_nll"],
                atol=CONTINUOUS_RECOMPUTE_ATOL,
            ),
            "target_bacc": close(
                test_point["balanced_accuracy"],
                fleet_task[tag]["target_test_balanced_accuracy"],
            ),
            "subject_nll": close(
                point[tag]["subject_nll"],
                fleet_subject[tag]["subject_nll"],
                atol=CONTINUOUS_RECOMPUTE_ATOL,
            ),
            "overlap": close(
                point_overlap,
                fleet_geometry[tag]["projection_overlap_rank4"],
                atol=GEOMETRY_RECOMPUTE_ATOL,
            ),
            "subject_capture": close(
                subject_capture,
                fleet_geometry[tag]["subject_heldout_captured_energy"],
                atol=GEOMETRY_RECOMPUTE_ATOL,
            ),
            "task_capture": close(
                task_capture,
                fleet_geometry[tag]["task_heldout_captured_energy"],
                atol=GEOMETRY_RECOMPUTE_ATOL,
            ),
        }
        point_checks.append(
            {
                "tag": tag,
                **checks,
                **{
                    key + "_abs_diff": abs(float(point[tag][key]) - expected_values[key])
                    for key in expected_values
                },
                "all_pass": all(checks.values()),
            }
        )
        if not all(checks.values()):
            raise RuntimeError("SEED-V independent point recomputation failed for %s" % tag)
        support["task_kappa__" + tag] = task_kappa[tag]
        support["task_nll__" + tag] = task_nll[tag]
        support["task_bacc__" + tag] = task_bacc[tag]
        support["subject_nll__" + tag] = subject_nll[tag]
        support["overlap__" + tag] = overlap[tag]

    h_family = [row for row in l5_rows if row["tag"].startswith("H")]
    adjusted = holm([row["empirical_one_sided_p"] for row in h_family])
    for row, value in zip(h_family, adjusted):
        row["holm_adjusted_p"] = float(value)
        row["exceeds_matched_null_holm_0p05"] = bool(value < 0.05)
        expected = fleet_l5[row["tag"]]
        for key in (
            "subject_delta_kappa",
            "null_delta_kappa_mean",
            "subject_minus_null_kappa",
            "empirical_one_sided_p",
            "source_val_removed_variance_fraction",
            "holm_adjusted_p",
        ):
            atol = (
                CONTINUOUS_RECOMPUTE_ATOL
                if key
                in (
                    "null_delta_kappa_mean",
                    "subject_minus_null_kappa",
                    "source_val_removed_variance_fraction",
                )
                else EMPIRICAL_P_RECOMPUTE_ATOL
                if key == "empirical_one_sided_p"
                else 1e-10
            )
            if not close(row[key], expected[key], atol=atol):
                raise RuntimeError("SEED-V L5 reproduction failed for %s:%s" % (row["tag"], key))
        if float(row["null_match_abs_error_max"]) > MATCH_TOLERANCE:
            raise RuntimeError("SEED-V L5 null energy matching failed for %s" % row["tag"])

    budget_rows = []
    for tag in TAGS:
        budget_rows.append(
            {
                "tag": tag,
                "target_kappa": point[tag]["target_kappa"],
                "target_kappa_ci95_low": float(np.quantile(task_kappa[tag], 0.025)),
                "target_kappa_ci95_high": float(np.quantile(task_kappa[tag], 0.975)),
                "target_nll": point[tag]["target_nll"],
                "target_nll_ci95_low": float(np.quantile(task_nll[tag], 0.025)),
                "target_nll_ci95_high": float(np.quantile(task_nll[tag], 0.975)),
                "target_bacc": point[tag]["target_bacc"],
                "target_bacc_ci95_low": float(np.quantile(task_bacc[tag], 0.025)),
                "target_bacc_ci95_high": float(np.quantile(task_bacc[tag], 0.975)),
                "subject_nll": point[tag]["subject_nll"],
                "subject_nll_ci95_low": float(np.quantile(subject_nll[tag], 0.025)),
                "subject_nll_ci95_high": float(np.quantile(subject_nll[tag], 0.975)),
                "projection_overlap_rank4": point[tag]["overlap"],
                "overlap_ci95_low": float(np.quantile(overlap[tag], 0.025)),
                "overlap_ci95_high": float(np.quantile(overlap[tag], 0.975)),
            }
        )

    task_rows = []
    for metric_name, values, point_key, direction in (
        ("target_kappa", task_kappa, "target_kappa", "greater"),
        ("target_nll_improvement", task_nll, "target_nll", "greater"),
        ("target_bacc", task_bacc, "target_bacc", "greater"),
    ):
        family = []
        for budget in HIGH_BUDGETS:
            if metric_name == "target_nll_improvement":
                samples = budget_mean(values, 200) - budget_mean(values, budget)
                point_value = np.mean([point[tag][point_key] for tag in budget_tags(200)]) - np.mean(
                    [point[tag][point_key] for tag in budget_tags(budget)]
                )
                contrast = "H200_minus_H%d" % budget
            else:
                samples = budget_mean(values, budget) - budget_mean(values, 200)
                point_value = np.mean([point[tag][point_key] for tag in budget_tags(budget)]) - np.mean(
                    [point[tag][point_key] for tag in budget_tags(200)]
                )
                contrast = "H%d_minus_H200" % budget
            family.append(
                {
                    "metric": metric_name,
                    "contrast": contrast,
                    **summarize(point_value, samples, direction),
                }
            )
        if metric_name == "target_nll_improvement":
            samples = budget_mean(values, 200) - pooled_high(values)
            point_value = np.mean([point[tag][point_key] for tag in budget_tags(200)]) - np.mean(
                [
                    np.mean([point[tag][point_key] for tag in budget_tags(budget)])
                    for budget in HIGH_BUDGETS
                ]
            )
            contrast = "H200_minus_pooled_higher"
        else:
            samples = pooled_high(values) - budget_mean(values, 200)
            point_value = np.mean(
                [
                    np.mean([point[tag][point_key] for tag in budget_tags(budget)])
                    for budget in HIGH_BUDGETS
                ]
            ) - np.mean([point[tag][point_key] for tag in budget_tags(200)])
            contrast = "pooled_higher_minus_H200"
        family.append(
            {
                "metric": metric_name,
                "contrast": contrast,
                **summarize(point_value, samples, direction),
            }
        )
        apply_holm(family)
        task_rows.extend(family)

    subject_rows = []
    subject_contrasts = (
        (
            "random_minus_H200",
            subject_nll["random"] - budget_mean(subject_nll, 200),
            point["random"]["subject_nll"]
            - np.mean([point[tag]["subject_nll"] for tag in budget_tags(200)]),
        ),
        (
            "H200_minus_pooled_higher",
            budget_mean(subject_nll, 200) - pooled_high(subject_nll),
            np.mean([point[tag]["subject_nll"] for tag in budget_tags(200)])
            - np.mean(
                [
                    np.mean([point[tag]["subject_nll"] for tag in budget_tags(budget)])
                    for budget in HIGH_BUDGETS
                ]
            ),
        ),
    )
    for name, samples, point_value in subject_contrasts:
        subject_rows.append(
            {
                "metric": "subject_probe_nll_improvement",
                "contrast": name,
                **summarize(point_value, samples, "greater"),
            }
        )
    apply_holm(subject_rows)

    geometry_rows = []
    for budget in HIGH_BUDGETS:
        samples = budget_mean(overlap, budget) - budget_mean(overlap, 200)
        point_value = np.mean([point[tag]["overlap"] for tag in budget_tags(budget)]) - np.mean(
            [point[tag]["overlap"] for tag in budget_tags(200)]
        )
        geometry_rows.append(
            {
                "metric": "projection_overlap_rank4",
                "contrast": "H%d_minus_H200" % budget,
                **summarize(point_value, samples, "two_sided"),
            }
        )
    samples = pooled_high(overlap) - budget_mean(overlap, 200)
    point_value = np.mean(
        [
            np.mean([point[tag]["overlap"] for tag in budget_tags(budget)])
            for budget in HIGH_BUDGETS
        ]
    ) - np.mean([point[tag]["overlap"] for tag in budget_tags(200)])
    geometry_rows.append(
        {
            "metric": "projection_overlap_rank4",
            "contrast": "pooled_higher_minus_H200",
            **summarize(point_value, samples, "two_sided"),
        }
    )
    apply_holm(geometry_rows)

    support_path = args.out_dir / "seedv_bootstrap_support.npz"
    np.savez_compressed(support_path, **support)
    write_csv(args.out_dir / "seedv_bootstrap_budget_metrics.csv", budget_rows)
    write_csv(args.out_dir / "seedv_bootstrap_task_contrasts.csv", task_rows)
    write_csv(args.out_dir / "seedv_bootstrap_subject_contrasts.csv", subject_rows)
    write_csv(args.out_dir / "seedv_bootstrap_geometry_contrasts.csv", geometry_rows)
    write_csv(args.out_dir / "seedv_bootstrap_l5.csv", l5_rows)
    write_csv(args.out_dir / "seedv_bootstrap_point_reproduction.csv", point_checks)
    verification = {
        "phase": "C1_SEED-V_5000_subject_cluster_bootstrap",
        "status": "PASS",
        "code_commit": actual_commit,
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "biological_cluster": "subject",
        "sessions_and_trials_nested_within_subject": True,
        "windows_treated_as_independent": False,
        "training_seed_means_used": True,
        "best_seed_or_budget_selected": False,
        "identity_sha256": identity_sha,
        "feature_payload_sha256": payload_hashes,
        "all_point_metrics_recomputed_from_feature_payloads": True,
        "all_point_reproduction_checks_pass": True,
        "continuous_recompute_atol": CONTINUOUS_RECOMPUTE_ATOL,
        "geometry_recompute_atol": GEOMETRY_RECOMPUTE_ATOL,
        "empirical_p_recompute_atol": EMPIRICAL_P_RECOMPUTE_ATOL,
        "empirical_p_tolerance_interpretation": "at_most_one_of_200_null_draws",
        "subject_geometry_and_l5_refit_from_feature_payloads": True,
        "l5_holm_family": [row["tag"] for row in h_family],
        "l5_holm_positive_cells": [
            row["tag"]
            for row in h_family
            if row["exceeds_matched_null_holm_0p05"]
        ],
        "target_labels_used_for_selection": False,
        "unseen_subject_claim_allowed": False,
        "support_payload": support_path.name,
        "support_payload_sha256": sha256_file(support_path),
    }
    write_json(args.out_dir / "seedv_bootstrap_verification.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
