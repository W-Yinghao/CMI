#!/usr/bin/env python
"""Independent adversarial verifier for S2P Phase B1 outputs."""
import argparse
import csv
import hashlib
import json
import math
import stat
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score


TAGS = [
    "random", "released", "H200_s0", "H200_s1", "H500_s0", "H500_s1",
    "H1000_s0", "H1000_s1", "H2000_s0", "H2000_s1",
]
BUDGETS = [200, 500, 1000, 2000]
HIGH_BUDGETS = [500, 1000, 2000]
N_SOURCE_SUBJECTS = 80
N_CLASSES = 9
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260711
SUBSPACE_RANK = 8
CLIP_TO_FOLD = {
    clip: position % 3
    for clips in (
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14, 15],
        [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27],
    )
    for position, clip in enumerate(clips)
}


def read_csv(path):
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value):
    return str(value).lower() == "true"


def close(left, right, atol=1e-10, rtol=1e-8):
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))


def budget_tags(budget):
    return [f"H{budget}_s0", f"H{budget}_s1"]


def mean_arrays(mapping, tags):
    return np.mean([mapping[tag] for tag in tags], axis=0)


def pool_high(mapping, budgets=HIGH_BUDGETS, seed=None):
    if seed is None:
        return np.mean([mean_arrays(mapping, budget_tags(budget)) for budget in budgets], axis=0)
    return np.mean([mapping[f"H{budget}_s{seed}"] for budget in budgets], axis=0)


def effect_geometry(cell, weights=None):
    if weights is None:
        weights = np.full(cell.shape[0], 1 / cell.shape[0])
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    cell = cell.astype(np.float64)
    subject_mean = cell.mean(axis=1)
    class_mean = np.einsum("s,skd->kd", weights, cell)
    grand = np.einsum("s,sd->d", weights, subject_mean)
    subject_effect = subject_mean - grand
    task_effect = class_mean - grand
    _, subject_singular, subject_vt = np.linalg.svd(
        np.sqrt(weights)[:, None] * subject_effect, full_matrices=False
    )
    _, task_singular, task_vt = np.linalg.svd(task_effect, full_matrices=False)
    if subject_singular[7] <= max(1e-12, subject_singular[0] * 1e-10):
        raise RuntimeError("verifier subject rank below 8")
    if task_singular[7] <= max(1e-12, task_singular[0] * 1e-10):
        raise RuntimeError("verifier task rank below 8")
    subject_basis = subject_vt[:8]
    task_basis = task_vt[:8]
    canonical = np.linalg.svd(subject_basis @ task_basis.T, compute_uv=False)
    return {
        "subject_basis": subject_basis,
        "task_basis": task_basis,
        "subject_effect": subject_effect,
        "task_effect": task_effect,
        "canonical": canonical,
        "overlap": float(np.mean(canonical ** 2)),
    }


def captured_energy(effect, basis, weights=None):
    projection = effect @ basis.T
    if weights is None:
        numerator = np.sum(projection ** 2)
        denominator = np.sum(effect ** 2)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        numerator = np.sum(weights[:, None] * projection ** 2)
        denominator = np.sum(weights[:, None] * effect ** 2)
    return float(numerator / max(float(denominator), 1e-12))


def variance_components(sufficient, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    weights /= weights.sum(axis=1, keepdims=True)

    def quadratic(matrix):
        return np.einsum("bi,ij,bj->b", weights, matrix, weights, optimize=True)

    grand = quadratic(sufficient["grand_cross"])
    subject = weights @ sufficient["subject_diag_cross"] - grand
    between = weights @ sufficient["between_diag_cross"] - grand
    class_component = np.stack(
        [quadratic(matrix) for matrix in sufficient["class_cross"]], axis=1
    ).mean(axis=1) - grand
    interaction = between - subject - class_component
    total = (
        weights @ sufficient["hold_second_subject"]
        - quadratic(sufficient["hold_grand_square"])
    )
    residual = total - subject - class_component - interaction
    denominator = np.where(np.abs(total) > 1e-12, total, np.nan)
    return {
        "subject": subject,
        "class": class_component,
        "interaction": interaction,
        "residual": residual,
        "total": total,
        "subject_frac": subject / denominator,
        "class_frac": class_component / denominator,
        "interaction_frac": interaction / denominator,
        "residual_frac": residual / denominator,
    }


def weighted_pair_bootstrap(rows, weights, value_key):
    left = np.asarray([int(row["subject_left"]) - 1 for row in rows])
    right = np.asarray([int(row["subject_right"]) - 1 for row in rows])
    values = np.asarray([float(row[value_key]) for row in rows])
    output = np.empty(len(weights))
    for start in range(0, len(weights), 250):
        current = weights[start:start + 250]
        pair_weights = current[:, left] * current[:, right]
        output[start:start + len(current)] = (
            pair_weights @ values
        ) / np.maximum(pair_weights.sum(axis=1), 1e-12)
    return output


def metrics_from_confusion(confusion):
    total = confusion.sum(axis=(-2, -1))
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    observed = diagonal.sum(axis=-1) / np.maximum(total, 1)
    row = confusion.sum(axis=-1)
    column = confusion.sum(axis=-2)
    expected = (row * column).sum(axis=-1) / np.maximum(total ** 2, 1)
    kappa = (observed - expected) / np.maximum(1 - expected, 1e-12)
    bacc = (diagonal / np.maximum(row, 1)).mean(axis=-1)
    return kappa, bacc


def one_sided_positive_p(samples):
    return float((1 + np.sum(np.asarray(samples) <= 0)) / (len(samples) + 1))


def two_sided_sign_p(samples):
    samples = np.asarray(samples)
    low = (1 + np.sum(samples <= 0)) / (len(samples) + 1)
    high = (1 + np.sum(samples >= 0)) / (len(samples) + 1)
    return float(min(1.0, 2 * min(low, high)))


def holm_adjust(values):
    order = np.argsort(values)
    output = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        output[index] = running
    return output


def summarize(point, samples):
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_sd": float(np.std(samples)),
    }


def run_self_tests():
    if not np.allclose(holm_adjust([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06]):
        raise RuntimeError("verifier Holm self-test failed")
    rng = np.random.default_rng(321)
    subject = rng.normal(size=(80, 8))
    subject -= subject.mean(axis=0)
    task = rng.normal(size=(9, 8))
    task -= task.mean(axis=0)
    cell = np.zeros((80, 9, 16))
    cell[:, :, :8] = subject[:, None]
    cell[:, :, 8:] = task[None]
    if effect_geometry(cell)["overlap"] > 1e-20:
        raise RuntimeError("verifier orthogonal-geometry self-test failed")
    print("Phase B1 independent verifier self-tests: PASS")


def run(args):
    out = Path(args.out_dir)
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    required = [
        "phase_b1_run_manifest.csv", "phase_b1_checkpoint_hash_recheck.csv",
        "phase_b1_feature_manifest.csv", "phase_b1_subject_metrics.csv",
        "phase_b1_subject_cluster_scores.csv", "phase_b1_subject_pair_metrics.csv",
        "phase_b1_task_metrics.csv", "phase_b1_task_sample_scores.csv",
        "phase_b1_subject_task_geometry.csv", "phase_b1_variance_partition.csv",
        "phase_b1_primary_inference.json", "phase_b1_sensitivity_results.csv",
        "phase_b1_target_label_firewall.json", "phase_b1_verdict.json",
        "phase_b1_primary_bootstrap_samples.csv", "phase_b1_bootstrap_support.npz",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"missing B1 files: {missing}")

    closure_rows = read_csv(args.checkpoint_manifest)
    check([row["tag"] for row in closure_rows] == TAGS, "closure_tag_set")
    closure = {row["tag"]: row for row in closure_rows}
    hash_rows = read_csv(out / "phase_b1_checkpoint_hash_recheck.csv")
    check([row["tag"] for row in hash_rows] == TAGS, "hash_row_tag_set")
    for row in hash_rows:
        tag = row["tag"]
        expected = closure[tag]["immutable_sha256"]
        check(as_bool(row["hash_recheck_pass"]), f"hash_flag:{tag}")
        check(row["expected_sha256"] == expected, f"hash_expected:{tag}")
        if tag == "random":
            check(all(row[key] == "logical_contract" for key in (
                "sha256_at_object_start", "sha256_at_object_end", "sha256_at_global_end"
            )), "random_logical_hash")
        else:
            path = Path(row["immutable_path"])
            check(path.is_file() and not path.is_symlink(), f"immutable_direct:{tag}")
            check(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0, f"immutable_writable:{tag}")
            current = sha256_file(path)
            check(current == expected, f"current_hash:{tag}")
            check(all(row[key] == expected for key in (
                "sha256_at_object_start", "sha256_at_object_end", "sha256_at_global_end"
            )), f"three_hash_rechecks:{tag}")

    feature_rows = read_csv(out / "phase_b1_feature_manifest.csv")
    check([row["tag"] for row in feature_rows] == TAGS, "feature_tag_set")
    for row in feature_rows:
        tag = row["tag"]
        check(as_bool(row["feature_contract_pass"]), f"feature_flag:{tag}")
        check(float(row["checksum_start_end_max_abs_diff"]) == 0, f"feature_repeat:{tag}")
        check(
            row["checksum_feature_sha256_start"]
            == row["checksum_feature_sha256_end"]
            == row["closure_feature_sha256"]
            == closure[tag]["feature_hash"],
            f"feature_closure_hash:{tag}",
        )

    subject_rows = read_csv(out / "phase_b1_subject_metrics.csv")
    subject_clusters = read_csv(out / "phase_b1_subject_cluster_scores.csv")
    subject_pairs = read_csv(out / "phase_b1_subject_pair_metrics.csv")
    check(len(subject_rows) == 10, "subject_checkpoint_count")
    check(len(subject_clusters) == 10 * 80, "subject_cluster_count")
    check(len(subject_pairs) == 10 * 3160, "subject_pair_count")
    subject_lookup = {row["tag"]: row for row in subject_rows}
    pair_by_tag = {}
    cluster_by_tag = {}
    for tag in TAGS:
        current_clusters = [row for row in subject_clusters if row["tag"] == tag]
        current_pairs = [row for row in subject_pairs if row["tag"] == tag]
        cluster_by_tag[tag] = current_clusters
        pair_by_tag[tag] = current_pairs
        check([int(row["subject"]) for row in current_clusters] == list(range(1, 81)), f"subject_order:{tag}")
        for output_key, support_key in (
            ("heldout_clip_subject_nll", "heldout_clip_subject_nll"),
            ("heldout_clip_subject_accuracy_diagnostic", "heldout_clip_subject_accuracy_diagnostic"),
            ("heldout_clip_true_subject_probability", "heldout_clip_true_subject_probability"),
            ("true_subject_probability_gt_0p999999_frac", "true_subject_probability_gt_0p999999_frac"),
            ("retrieval_map", "retrieval_map"),
            ("class_conditional_subject_nll", "class_conditional_subject_nll_macro"),
        ):
            value = np.mean([float(row[support_key]) for row in current_clusters])
            check(close(subject_lookup[tag][output_key], value), f"subject_metric:{tag}:{output_key}")
        for output_key, support_key in (
            ("pairwise_subject_auc", "pairwise_auc"),
            ("pairwise_subject_margin", "pairwise_standardized_margin"),
            ("pairwise_subject_bacc_diagnostic", "pairwise_bacc_diagnostic"),
        ):
            value = np.mean([float(row[support_key]) for row in current_pairs])
            check(close(subject_lookup[tag][output_key], value), f"pair_metric:{tag}:{output_key}")

    task_rows = read_csv(out / "phase_b1_task_metrics.csv")
    sample_rows = read_csv(out / "phase_b1_task_sample_scores.csv")
    check(len(task_rows) == 20, "task_checkpoint_split_count")
    task_lookup = {(row["tag"], row["split"]): row for row in task_rows}
    subject_support = [row for row in sample_rows if row["record_type"] == "subject_probe_support"]
    task_support = [row for row in sample_rows if row["record_type"] == "task_probe_support"]
    check(len(subject_support) == 10 * 80 * 28 * 3, "subject_probe_support_count")
    check(len(task_support) == 10 * (20 + 23) * 28 * 3, "task_probe_support_count")
    for tag in TAGS:
        current = [row for row in subject_support if row["tag"] == tag]
        keys = {(int(row["subject"]), int(row["clip_id"]), int(row["segment_id"])) for row in current}
        check(len(keys) == 80 * 28 * 3, f"subject_support_unique:{tag}")
        check(all(int(row["fold"]) == CLIP_TO_FOLD[int(row["clip_id"])] for row in current), f"clip_fold:{tag}")
        check(all(row["label_final_scoring_only"] == "" for row in current), f"subject_support_target_label:{tag}")
        for split, expected_count in (("source_val", 20 * 28 * 3), ("target_test", 23 * 28 * 3)):
            rows = [row for row in task_support if row["tag"] == tag and row["split"] == split]
            check(len(rows) == expected_count, f"task_support_count:{tag}:{split}")
            truth = np.asarray([int(row["label_final_scoring_only"]) for row in rows])
            prediction = np.asarray([int(row["prediction"]) for row in rows])
            nll = np.asarray([float(row["nll"]) for row in rows])
            margin = np.asarray([float(row["margin"]) for row in rows])
            output = task_lookup[(tag, split)]
            checks = {
                "task_nll": np.mean(nll),
                "cohen_kappa": cohen_kappa_score(truth, prediction),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "class_margin_mean": np.mean(margin),
                "class_margin_median": np.median(margin),
                "class_margin_positive_frac": np.mean(margin > 0),
            }
            for key, value in checks.items():
                check(close(output[key], value), f"task_metric:{tag}:{split}:{key}")

    support = np.load(out / "phase_b1_bootstrap_support.npz", allow_pickle=False)
    check(support["tags"].tolist() == TAGS, "bootstrap_tag_set")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    source_counts = rng.multinomial(80, np.full(80, 1 / 80), size=N_BOOTSTRAP)
    target_counts = rng.multinomial(23, np.full(23, 1 / 23), size=N_BOOTSTRAP)
    check(np.array_equal(support["source_counts"], source_counts), "source_bootstrap_counts")
    check(np.array_equal(support["target_counts"], target_counts), "target_bootstrap_counts")
    source_weights = source_counts / source_counts.sum(axis=1, keepdims=True)
    target_weights = target_counts / target_counts.sum(axis=1, keepdims=True)

    subject_nll = {}
    subject_margin = {}
    subject_auc = {}
    target_nll = {}
    target_kappa = {}
    target_bacc = {}
    for tag_index, tag in enumerate(TAGS):
        cluster_values = np.asarray([float(row["heldout_clip_subject_nll"]) for row in cluster_by_tag[tag]])
        subject_nll[tag] = source_weights @ cluster_values
        subject_margin[tag] = weighted_pair_bootstrap(pair_by_tag[tag], source_counts, "pairwise_standardized_margin")
        subject_auc[tag] = weighted_pair_bootstrap(pair_by_tag[tag], source_counts, "pairwise_auc")
        check(np.allclose(subject_nll[tag], support["subject_nll_samples"][tag_index]), f"subject_nll_bootstrap:{tag}")
        check(np.allclose(subject_margin[tag], support["subject_margin_samples"][tag_index]), f"subject_margin_bootstrap:{tag}")
        check(np.allclose(subject_auc[tag], support["subject_auc_samples"][tag_index]), f"subject_auc_bootstrap:{tag}")

        target_rows = [row for row in task_support if row["tag"] == tag and row["split"] == "target_test"]
        target_subjects = list(range(101, 124))
        nll_by_subject = np.asarray([
            np.mean([float(row["nll"]) for row in target_rows if int(row["subject"]) == subject])
            for subject in target_subjects
        ])
        target_nll[tag] = target_weights @ nll_by_subject
        confusion = np.zeros((23, N_CLASSES, N_CLASSES))
        for subject_index, subject in enumerate(target_subjects):
            rows = [row for row in target_rows if int(row["subject"]) == subject]
            for row in rows:
                confusion[subject_index, int(row["label_final_scoring_only"]), int(row["prediction"])] += 1
        boot_confusion = np.einsum("bs,sij->bij", target_counts, confusion, optimize=True)
        target_kappa[tag], target_bacc[tag] = metrics_from_confusion(boot_confusion)
        check(np.allclose(target_nll[tag], support["target_nll_samples"][tag_index]), f"target_nll_bootstrap:{tag}")
        check(np.allclose(target_kappa[tag], support["target_kappa_samples"][tag_index]), f"target_kappa_bootstrap:{tag}")
        check(np.allclose(target_bacc[tag], support["target_bacc_samples"][tag_index]), f"target_bacc_bootstrap:{tag}")

    geometry_rows = read_csv(out / "phase_b1_subject_task_geometry.csv")
    geometry_samples = {}
    for tag_index, tag in enumerate(TAGS):
        fold_rows = sorted(
            [row for row in geometry_rows if row["tag"] == tag and row["scope"] == "fold"],
            key=lambda row: int(row["fold"]),
        )
        check(len(fold_rows) == 3, f"geometry_fold_count:{tag}")
        geometry_samples[tag] = np.empty(N_BOOTSTRAP)
        for fold, row in enumerate(fold_rows):
            fit = support[f"geometry_cell__{tag}__fold{fold}"]
            hold = support[f"geometry_hold_cell__{tag}__fold{fold}"]
            fit_geometry = effect_geometry(fit)
            hold_geometry = effect_geometry(hold)
            angles = np.degrees(np.arccos(np.clip(fit_geometry["canonical"], -1, 1)))
            check(close(row["projection_overlap"], fit_geometry["overlap"]), f"geometry_overlap:{tag}:{fold}")
            check(close(row["max_canonical_correlation"], fit_geometry["canonical"].max()), f"geometry_max_cc:{tag}:{fold}")
            check(close(row["median_principal_angle_deg"], np.median(angles)), f"geometry_angle:{tag}:{fold}")
            checks = {
                "heldout_subject_self_capture": captured_energy(hold_geometry["subject_effect"], fit_geometry["subject_basis"]),
                "heldout_task_self_capture": captured_energy(hold_geometry["task_effect"], fit_geometry["task_basis"]),
                "heldout_subject_on_task_capture": captured_energy(hold_geometry["subject_effect"], fit_geometry["task_basis"]),
                "heldout_task_on_subject_capture": captured_energy(hold_geometry["task_effect"], fit_geometry["subject_basis"]),
            }
            for key, value in checks.items():
                check(close(row[key], value), f"geometry_capture:{tag}:{fold}:{key}")
            expected_stable = checks["heldout_subject_self_capture"] >= 0.05 and checks["heldout_task_self_capture"] >= 0.05
            check(as_bool(row["subspace_stable"]) == expected_stable, f"geometry_stability:{tag}:{fold}")
        for replicate in range(N_BOOTSTRAP):
            geometry_samples[tag][replicate] = np.mean([
                effect_geometry(support[f"geometry_cell__{tag}__fold{fold}"], source_weights[replicate])["overlap"]
                for fold in range(3)
            ])
        check(np.allclose(geometry_samples[tag], support["geometry_overlap_samples"][tag_index]), f"geometry_bootstrap:{tag}")
        mean_row = [row for row in geometry_rows if row["tag"] == tag and row["scope"] == "checkpoint_mean"]
        check(len(mean_row) == 1, f"geometry_mean_count:{tag}")
        check(close(mean_row[0]["projection_overlap"], np.mean([float(row["projection_overlap"]) for row in fold_rows])), f"geometry_mean:{tag}")

    variance_rows = read_csv(out / "phase_b1_variance_partition.csv")
    variance_samples = {tag: {} for tag in TAGS}
    variance_keys = ("subject", "class", "interaction", "residual", "total", "subject_frac", "class_frac", "interaction_frac", "residual_frac")
    for tag in TAGS:
        fold_rows = sorted(
            [row for row in variance_rows if row["tag"] == tag and row["scope"] == "fold"],
            key=lambda row: int(row["fold"]),
        )
        check(len(fold_rows) == 3, f"variance_fold_count:{tag}")
        fold_boot = []
        for fold, row in enumerate(fold_rows):
            sufficient = {}
            prefix = f"variance__{tag}__fold{fold}__"
            for key in ("grand_cross", "subject_diag_cross", "between_diag_cross", "class_cross", "hold_grand_square", "hold_second_subject"):
                sufficient[key] = support[prefix + key]
            point = variance_components(sufficient, np.full(80, 1 / 80))
            boot = variance_components(sufficient, source_weights)
            fold_boot.append(boot)
            for key in variance_keys:
                check(close(row[key], point[key][0]), f"variance_point:{tag}:{fold}:{key}")
            check(close(float(row["subject"]) + float(row["class"]) + float(row["interaction"]) + float(row["residual"]), row["total"]), f"variance_sum:{tag}:{fold}")
        instability = []
        for component in ("subject_frac", "class_frac", "interaction_frac", "residual_frac"):
            values = np.mean([entry[component] for entry in fold_boot], axis=0)
            variance_samples[tag][component] = values
            check(np.allclose(values, support[f"variance_samples__{tag}__{component}"]), f"variance_bootstrap:{tag}:{component}")
            fold_point = np.asarray([float(row[component]) for row in fold_rows])
            if np.max(np.abs(fold_point - fold_point.mean())) > 0.10 or np.quantile(values, 0.975) - np.quantile(values, 0.025) > 0.20:
                instability.append(component)
        if min(float(row["residual_frac"]) for row in fold_rows) < -0.01:
            instability.append("negative_residual")
        expected = "UNSTABLE_UNDER_CLIP_CROSSFIT" if instability else "PASS"
        check(all(row["variance_stability"] == expected for row in fold_rows), f"variance_stability:{tag}")
        mean_rows = [row for row in variance_rows if row["tag"] == tag and row["scope"] == "checkpoint_mean"]
        check(len(mean_rows) == 1, f"variance_mean_count:{tag}")
        for key in variance_keys:
            check(close(mean_rows[0][key], np.mean([float(row[key]) for row in fold_rows])), f"variance_mean:{tag}:{key}")

    primary = json.loads((out / "phase_b1_primary_inference.json").read_text())
    subject_lookup = {row["tag"]: row for row in subject_rows}
    task_lookup = {row["tag"]: row for row in task_rows if row["split"] == "target_test"}
    geometry_lookup = {row["tag"]: row for row in geometry_rows if row["scope"] == "checkpoint_mean"}
    if primary["primary_subject_metric"] == "pairwise_subject_margin_fallback":
        p1 = mean_arrays(subject_margin, budget_tags(200)) - subject_margin["random"]
        p1_point = np.mean([float(subject_lookup[tag]["pairwise_subject_margin"]) for tag in budget_tags(200)]) - float(subject_lookup["random"]["pairwise_subject_margin"])
    else:
        p1 = subject_nll["random"] - mean_arrays(subject_nll, budget_tags(200))
        p1_point = float(subject_lookup["random"]["heldout_clip_subject_nll"]) - np.mean([float(subject_lookup[tag]["heldout_clip_subject_nll"]) for tag in budget_tags(200)])
    p2 = mean_arrays(target_nll, budget_tags(200)) - pool_high(target_nll)
    p2_point = np.mean([float(task_lookup[tag]["task_nll"]) for tag in budget_tags(200)]) - np.mean([
        np.mean([float(task_lookup[tag]["task_nll"]) for tag in budget_tags(budget)]) for budget in HIGH_BUDGETS
    ])
    p3 = pool_high(geometry_samples) - mean_arrays(geometry_samples, budget_tags(200))
    p3_point = np.mean([
        np.mean([float(geometry_lookup[tag]["projection_overlap"]) for tag in budget_tags(budget)]) for budget in HIGH_BUDGETS
    ]) - np.mean([float(geometry_lookup[tag]["projection_overlap"]) for tag in budget_tags(200)])
    samples = [p1, p2, p3]
    points = [p1_point, p2_point, p3_point]
    raw_p = [one_sided_positive_p(p1), one_sided_positive_p(p2), two_sided_sign_p(p3)]
    corrected = holm_adjust(raw_p)
    bootstrap_rows = read_csv(out / "phase_b1_primary_bootstrap_samples.csv")
    check(len(bootstrap_rows) == N_BOOTSTRAP, "primary_bootstrap_count")
    for key, expected in zip(("p1_subject_delta", "p2_task_nll_delta", "p3_overlap_delta"), samples):
        observed = np.asarray([float(row[key]) for row in bootstrap_rows])
        check(np.allclose(observed, expected), f"primary_bootstrap:{key}")
    check(len(primary["contrasts"]) == 3, "primary_contrast_count")
    for index, row in enumerate(primary["contrasts"]):
        expected = summarize(points[index], samples[index])
        for key, value in expected.items():
            check(close(row[key], value), f"primary_summary:{index}:{key}")
        check(close(row["p_raw"], raw_p[index]), f"primary_p_raw:{index}")
        check(close(row["p_holm"], corrected[index]), f"primary_p_holm:{index}")
        check(as_bool(row["reject_holm_0p05"]) == (corrected[index] < 0.05), f"primary_reject:{index}")

    sensitivity_rows = read_csv(out / "phase_b1_sensitivity_results.csv")
    sensitivity = {row["analysis"]: row for row in sensitivity_rows}
    required_sensitivity = {
        "subject_nll_H200_minus_pooled_high", "pooled_high_minus_H200_target_kappa",
        "pooled_high_minus_H200_target_bacc",
        *{f"leave_high_budget_{budget}_out_task_nll" for budget in HIGH_BUDGETS},
        *{f"leave_high_budget_{budget}_out_overlap" for budget in HIGH_BUDGETS},
        *{f"retain_training_seed_{seed}_{name}" for seed in (0, 1) for name in ("subject_early", "task_nll", "overlap")},
    }
    check(required_sensitivity.issubset(sensitivity), "sensitivity_required_rows")

    firewall = json.loads((out / "phase_b1_target_label_firewall.json").read_text())
    selection_keys = [key for key in firewall if "selection" in key or key in (
        "target_labels_used_for_pca", "target_labels_used_for_probe_fit", "target_labels_used_for_rank_selection",
    )]
    check(all(firewall[key] is False for key in selection_keys), "target_label_firewall")
    check(firewall.get("target_labels_used_for_final_task_scoring") is True, "target_final_scoring_flag")

    protocol = json.loads((out / "phase_b1_protocol.json").read_text())
    check(protocol.get("pm_b1_authorized") is True, "pm_authorization_record")
    check(protocol.get("layerwise") is False, "no_layerwise")
    check(all(protocol.get(key) is False for key in ("new_pretraining", "fine_tuning", "h4000", "codebrain")), "scope_exclusions")
    verdict = json.loads((out / "phase_b1_verdict.json").read_text())
    check(verdict.get("status") == "PASS", "primary_verdict_status")
    check(verdict.get("phase_b2_authorized") is False, "phase_b2_held")
    check(verdict.get("recommend_phase_b2_layerwise") is False, "phase_b2_not_recommended_automatically")
    check(verdict.get("target_labels_used_for_selection") is False, "verdict_firewall")

    verification = {
        "phase": "B1_independent_adversarial_verification",
        "status": "PASS" if not failures else "NO_GO",
        "independent_metric_recomputation": True,
        "checkpoint_objects": 10,
        "checkpoint_hashes_reverified": not any(item.startswith(("current_hash", "three_hash", "hash_")) for item in failures),
        "feature_contracts_reverified": not any(item.startswith("feature") for item in failures),
        "clip_grouping_reverified": not any(item.startswith(("clip_fold", "subject_support")) for item in failures),
        "primary_inference_recomputed": not any(item.startswith("primary") for item in failures),
        "geometry_recomputed_from_sufficient_statistics": not any(item.startswith("geometry") for item in failures),
        "variance_recomputed_from_sufficient_statistics": not any(item.startswith("variance") for item in failures),
        "target_label_firewall_clean": "target_label_firewall" not in failures,
        "phase_b2_authorized": False,
        "failures": failures,
    }
    write_json(out / "phase_b1_adversarial_verification.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"B1 independent verification failed: {failures[:10]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/s2p_route_b_representation_emergence_b1")
    parser.add_argument(
        "--checkpoint-manifest",
        default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    run(args)


if __name__ == "__main__":
    main()
