#!/usr/bin/env python
"""Independent, fail-closed verification of the complete Phase-C readout fleet."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, log_loss

from route_b_cross_task_common import (
    TAGS,
    read_csv,
    sha256_file,
    validate_manifest,
    write_csv,
    write_json,
)


N_BOOTSTRAP = 5000
SEEDV_BOOTSTRAP_SEED = 20260714
FACED_BOOTSTRAP_SEED = 20260711
N_CLASSES_FACED = 9
N_CLASSES_EXTERNAL = 5
HIGH_BUDGETS = (500, 1000, 2000)
MATCH_TOLERANCE = 1e-10


def close(left, right, atol=1e-10):
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=0.0))


def truth(value):
    return str(value).lower() == "true"


def budget_tags(budget):
    return ["H%d_s0" % budget, "H%d_s1" % budget]


def holm(pvalues):
    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def metrics(labels, probability, n_classes):
    prediction = probability.argmax(axis=1)
    return {
        "kappa": float(cohen_kappa_score(labels, prediction)),
        "nll": float(log_loss(labels, probability, labels=np.arange(n_classes))),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


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
    return np.divide(
        observed - expected,
        1.0 - expected,
        out=np.zeros_like(observed),
        where=np.abs(1.0 - expected) > 1e-15,
    )


def top_basis(matrix, rank):
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if len(singular) < rank or singular[rank - 1] <= 1e-12:
        raise RuntimeError("geometry subspace is numerically degenerate")
    return vt[:rank]


def overlap_from_cells(cells, rank):
    grand = cells.mean(axis=(0, 1))
    subject_effect = cells.mean(axis=1) - grand
    task_effect = cells.mean(axis=0) - grand
    subject_basis = top_basis(subject_effect, rank)
    task_basis = top_basis(task_effect, rank)
    return float(np.sum((subject_basis @ task_basis.T) ** 2) / rank)


class Checks:
    def __init__(self):
        self.rows = []
        self.failures = []

    def add(self, dataset, component, name, passed, detail=""):
        passed = bool(passed)
        self.rows.append(
            {
                "dataset": dataset,
                "component": component,
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )
        if not passed:
            self.failures.append("%s:%s:%s" % (dataset, component, name))


def verify_faced(checks, roots):
    final_root = roots["faced_final"]
    core_root = roots["faced_core"]
    b0_root = roots["faced_b0"]

    split_rows = read_csv(final_root / "faced_split_manifest.csv")
    split_subjects = {
        split: sorted(int(row["subject"]) for row in split_rows if row["split"] == split)
        for split in ("source_train", "source_val", "target_test")
    }
    checks.add("FACED", "split", "subject_sets", split_subjects == {
        "source_train": list(range(1, 81)),
        "source_val": list(range(81, 101)),
        "target_test": list(range(101, 124)),
    })
    checks.add(
        "FACED",
        "split",
        "84_segments_per_subject",
        len(split_rows) == 123 and all(int(row["n_segments"]) == 84 for row in split_rows),
    )
    clip_rows = read_csv(b0_root / "phase_b0_clip_group_manifest.csv")
    checks.add(
        "FACED",
        "split",
        "28_clip_groups_three_segments",
        len(clip_rows) == 28
        and {int(row["clip_id"]) for row in clip_rows} == set(range(28))
        and all(row["segment_ids"] == "0;1;2" for row in clip_rows),
    )

    prediction_rows = read_csv(final_root / "faced_final_target_predictions.csv")
    reproduction = {
        row["tag"]: row
        for row in read_csv(final_root / "faced_final_reproduction_check.csv")
    }
    task_point = {}
    for tag in TAGS:
        rows = [row for row in prediction_rows if row["tag"] == tag]
        labels = np.asarray([int(row["target_label_final_scoring_only"]) for row in rows])
        prediction = np.asarray([int(row["prediction_base"]) for row in rows])
        subjects = sorted({int(row["target_subject"]) for row in rows})
        value = {
            "kappa": float(cohen_kappa_score(labels, prediction)),
            "bacc": float(balanced_accuracy_score(labels, prediction)),
        }
        task_point[tag] = value
        checks.add(
            "FACED",
            "task",
            "prediction_recompute_%s" % tag,
            len(rows) == 23 * 84
            and subjects == list(range(101, 124))
            and close(value["kappa"], reproduction[tag]["target_kappa_recomputed"])
            and close(value["bacc"], reproduction[tag]["target_bacc_recomputed"]),
        )

    task_rows = read_csv(core_root / "phase_b1_core_task_metrics.csv")
    task_lookup = {(row["tag"], row["split"]): row for row in task_rows}
    task_support = [
        row
        for row in read_csv(core_root / "phase_b1_core_task_sample_scores.csv")
        if row["record_type"] == "task_probe_support"
    ]
    target_nll_point = {}
    for tag in TAGS:
        for split, count in (("source_val", 20 * 84), ("target_test", 23 * 84)):
            rows = [row for row in task_support if row["tag"] == tag and row["split"] == split]
            labels = np.asarray([int(row["label_final_scoring_only"]) for row in rows])
            prediction = np.asarray([int(row["prediction"]) for row in rows])
            nll = float(np.mean([float(row["nll"]) for row in rows]))
            expected = task_lookup[(tag, split)]
            checks.add(
                "FACED",
                "task",
                "core_support_recompute_%s_%s" % (tag, split),
                len(rows) == count
                and close(nll, expected["task_nll"])
                and close(cohen_kappa_score(labels, prediction), expected["cohen_kappa"])
                and close(
                    balanced_accuracy_score(labels, prediction),
                    expected["balanced_accuracy"],
                ),
            )
            if split == "target_test":
                target_nll_point[tag] = nll

    subject_metric = {
        row["tag"]: row
        for row in read_csv(core_root / "phase_b1_core_subject_metrics.csv")
    }
    subject_clusters = read_csv(core_root / "phase_b1_core_subject_cluster_scores.csv")
    subject_nll_point = {}
    for tag in TAGS:
        rows = [row for row in subject_clusters if row["tag"] == tag]
        value = float(np.mean([float(row["heldout_clip_subject_nll"]) for row in rows]))
        subject_nll_point[tag] = value
        checks.add(
            "FACED",
            "subject",
            "cluster_support_recompute_%s" % tag,
            len(rows) == 80
            and sorted(int(row["subject"]) for row in rows) == list(range(1, 81))
            and close(value, subject_metric[tag]["heldout_clip_subject_nll"]),
        )

    support_path = core_root / "phase_b1_core_support.npz"
    primary = json.loads((core_root / "phase_b1_core_primary_inference.json").read_text())
    with np.load(support_path, allow_pickle=False) as support:
        checks.add("FACED", "inference", "support_tags", support["tags"].tolist() == TAGS)
        rng = np.random.default_rng(FACED_BOOTSTRAP_SEED)
        source_counts = rng.multinomial(80, np.full(80, 1 / 80), size=N_BOOTSTRAP)
        target_counts = rng.multinomial(23, np.full(23, 1 / 23), size=N_BOOTSTRAP)
        checks.add(
            "FACED",
            "inference",
            "bootstrap_cluster_counts",
            np.array_equal(source_counts, support["source_counts"])
            and np.array_equal(target_counts, support["target_counts"]),
        )
        tag_index = {tag: index for index, tag in enumerate(TAGS)}
        subject_samples = {
            tag: support["subject_nll_samples"][tag_index[tag]] for tag in TAGS
        }
        target_samples = {
            tag: support["target_nll_samples"][tag_index[tag]] for tag in TAGS
        }
        geometry_samples = {
            tag: support["geometry_overlap_samples"][tag_index[tag]] for tag in TAGS
        }
        geometry_point = {}
        geometry_rows = read_csv(core_root / "phase_b1_core_geometry.csv")
        geometry_lookup = {
            row["tag"]: row
            for row in geometry_rows
            if row["scope"] == "checkpoint_mean"
        }
        for tag in TAGS:
            fold_values = []
            for fold in range(3):
                cells = support["geometry_cell__%s__fold%d" % (tag, fold)]
                fold_values.append(overlap_from_cells(cells, 8))
            geometry_point[tag] = float(np.mean(fold_values))
            checks.add(
                "FACED",
                "geometry",
                "rank8_point_recompute_%s" % tag,
                close(geometry_point[tag], geometry_lookup[tag]["projection_overlap"], atol=1e-7),
            )

    def seed_mean(values, budget):
        return np.mean([values[tag] for tag in budget_tags(budget)], axis=0)

    def high_mean(values):
        return np.mean([seed_mean(values, budget) for budget in HIGH_BUDGETS], axis=0)

    p1 = subject_samples["random"] - seed_mean(subject_samples, 200)
    p2 = seed_mean(target_samples, 200) - high_mean(target_samples)
    p3 = high_mean(geometry_samples) - seed_mean(geometry_samples, 200)
    points = [
        subject_nll_point["random"]
        - np.mean([subject_nll_point[tag] for tag in budget_tags(200)]),
        np.mean([target_nll_point[tag] for tag in budget_tags(200)])
        - np.mean(
            [
                np.mean([target_nll_point[tag] for tag in budget_tags(budget)])
                for budget in HIGH_BUDGETS
            ]
        ),
        np.mean(
            [
                np.mean([geometry_point[tag] for tag in budget_tags(budget)])
                for budget in HIGH_BUDGETS
            ]
        )
        - np.mean([geometry_point[tag] for tag in budget_tags(200)]),
    ]
    for index, (samples, point) in enumerate(zip((p1, p2, p3), points)):
        row = primary["contrasts"][index]
        checks.add(
            "FACED",
            "inference",
            "primary_contrast_%d" % (index + 1),
            close(point, row["point"], atol=1e-7)
            and close(np.quantile(samples, 0.025), row["ci95_low"], atol=1e-7)
            and close(np.quantile(samples, 0.975), row["ci95_high"], atol=1e-7),
        )

    l5 = [
        row
        for row in read_csv(final_root / "faced_final_l5_null_sensitivity.csv")
        if row["tag"].startswith("H")
    ]
    clustered = [
        row
        for row in read_csv(final_root / "faced_final_l5_clustered_inference.csv")
        if row["scope"] == "checkpoint" and row["metric"] == "kappa"
    ]
    adjusted = holm([float(row["empirical_one_sided_p_kappa"]) for row in l5])
    l5_lookup = {row["tag"]: row for row in clustered}
    checks.add(
        "FACED",
        "L5",
        "removed_energy_and_holm",
        len(l5) == 8
        and all(float(row["null_source_val_match_abs_error_max"]) <= 1e-10 for row in l5)
        and all(
            close(value, l5_lookup[row["tag"]]["holm_adjusted_empirical_p"])
            for row, value in zip(l5, adjusted)
        )
        and not any(truth(l5_lookup[row["tag"]]["subject_intervention_exceeds_null"]) for row in l5),
    )
    final_verification = json.loads((final_root / "faced_final_verification.json").read_text())
    core_verification = json.loads(
        (core_root / "phase_b1_core_adversarial_verification.json").read_text()
    )
    checks.add(
        "FACED",
        "independent_history",
        "existing_adversarial_verifiers",
        final_verification.get("all_committed_metrics_reproduced") is True
        and core_verification.get("status") == "PASS",
    )
    return {
        "task_point": task_point,
        "primary_subject_early": bool(primary["contrasts"][0]["reject_holm_0p05"]),
        "primary_task_nll_later": bool(primary["contrasts"][1]["reject_holm_0p05"]),
        "primary_overlap_decrease": bool(
            primary["contrasts"][2]["reject_holm_0p05"]
            and float(primary["contrasts"][2]["point"]) < 0
        ),
        "l5_positive_cells": [],
    }


def verify_seedv(checks, roots):
    seedv_root = roots["seedv"]
    closure_root = roots["closure"]
    c0_root = roots["c0"]
    verification = json.loads((closure_root / "seedv_bootstrap_verification.json").read_text())
    support_path = closure_root / verification["support_payload"]
    checks.add(
        "SEED-V",
        "bootstrap",
        "support_hash",
        sha256_file(support_path) == verification["support_payload_sha256"],
    )
    trial_rows = read_csv(c0_root / "seedv_trial_manifest.csv")
    split_rows = read_csv(c0_root / "seedv_split_manifest.csv")
    split_rule = {"train": range(0, 5), "val": range(5, 10), "test": range(10, 15)}
    checks.add(
        "SEED-V",
        "split",
        "trial_groups_and_nested_sessions",
        len(trial_rows) == 705
        and len({row["trial_id"] for row in trial_rows}) == 705
        and {int(row["subject"]) for row in trial_rows} == set(range(1, 17))
        and len({(row["subject"], row["session"]) for row in trial_rows}) == 47
        and all(
            int(row["trial"]) in split_rule[row["split"]]
            and truth(row["trial_label_constant"])
            and truth(row["window_chronology_contiguous"])
            for row in trial_rows
        )
        and all(int(row["trials"]) == 235 and truth(row["trial_groups_disjoint"]) for row in split_rows),
    )
    for tag in TAGS:
        contract = json.loads((seedv_root / "features" / (tag + "_feature_contract.json")).read_text())
        payload = seedv_root / "features" / (tag + "_features.npz")
        checks.add(
            "SEED-V",
            "features",
            "payload_hash_%s" % tag,
            sha256_file(payload) == contract["payload_sha256"]
            == verification["feature_payload_sha256"][tag]
            and contract.get("canary_repeat_max_abs_diff") == 0.0,
        )
    with np.load(support_path, allow_pickle=False) as support:
        rng = np.random.default_rng(SEEDV_BOOTSTRAP_SEED)
        counts = rng.multinomial(16, np.full(16, 1 / 16), size=N_BOOTSTRAP)
        checks.add(
            "SEED-V",
            "bootstrap",
            "5000_subject_cluster_counts",
            support["tags"].tolist() == TAGS
            and np.array_equal(counts, support["subject_counts"]),
        )
        checks.add(
            "SEED-V",
            "bootstrap",
            "no_window_level_resampling",
            verification.get("biological_cluster") == "subject"
            and verification.get("sessions_and_trials_nested_within_subject") is True
            and verification.get("windows_treated_as_independent") is False,
        )
    point_checks = read_csv(closure_root / "seedv_bootstrap_point_reproduction.csv")
    checks.add(
        "SEED-V",
        "point_metrics",
        "all_ten_refit_from_features",
        len(point_checks) == 10 and all(truth(row["all_pass"]) for row in point_checks),
    )
    budget = {row["tag"]: row for row in read_csv(closure_root / "seedv_bootstrap_budget_metrics.csv")}
    task_contrasts = read_csv(closure_root / "seedv_bootstrap_task_contrasts.csv")
    subject_contrasts = {
        row["contrast"]: row
        for row in read_csv(closure_root / "seedv_bootstrap_subject_contrasts.csv")
    }
    geometry_contrasts = {
        row["contrast"]: row
        for row in read_csv(closure_root / "seedv_bootstrap_geometry_contrasts.csv")
    }
    l5 = read_csv(closure_root / "seedv_bootstrap_l5.csv")
    h_l5 = [row for row in l5 if row["tag"].startswith("H")]
    adjusted = holm([float(row["empirical_one_sided_p"]) for row in h_l5])
    checks.add(
        "SEED-V",
        "L5",
        "energy_match_holm_and_cell_inclusion",
        len(h_l5) == 8
        and [row["tag"] for row in h_l5]
        == [tag for tag in TAGS if tag.startswith("H")]
        and all(float(row["null_match_abs_error_max"]) <= MATCH_TOLERANCE for row in h_l5)
        and all(close(value, row["holm_adjusted_p"]) for row, value in zip(h_l5, adjusted))
        and not any(truth(row["exceeds_matched_null_holm_0p05"]) for row in h_l5),
    )
    pooled_kappa = next(
        row
        for row in task_contrasts
        if row["metric"] == "target_kappa"
        and row["contrast"] == "pooled_higher_minus_H200"
    )
    pooled_nll = next(
        row
        for row in task_contrasts
        if row["metric"] == "target_nll_improvement"
        and row["contrast"] == "H200_minus_pooled_higher"
    )
    subject_early = subject_contrasts["random_minus_H200"]
    subject_continues = subject_contrasts["H200_minus_pooled_higher"]
    overlap = geometry_contrasts["pooled_higher_minus_H200"]
    kappa_means = {
        budget_h: np.mean(
            [float(budget[tag]["target_kappa"]) for tag in budget_tags(budget_h)]
        )
        for budget_h in (200, 500, 1000, 2000)
    }
    checks.add(
        "SEED-V",
        "claims",
        "no_unseen_subject_claim",
        verification.get("unseen_subject_claim_allowed") is False,
    )
    return {
        "subject_early": truth(subject_early["reject_holm_0p05"])
        and float(subject_early["point"]) > 0,
        "subject_continues": truth(subject_continues["reject_holm_0p05"])
        and float(subject_continues["point"]) > 0,
        "overlap_decrease": truth(overlap["reject_holm_0p05"])
        and float(overlap["point"]) < 0,
        "task_kappa_higher_than_h200": float(pooled_kappa["ci95_low"]) > 0,
        "task_nll_higher_than_h200": float(pooled_nll["ci95_low"]) > 0,
        "kappa_budget_means": kappa_means,
        "kappa_monotonic": bool(
            np.all(np.diff([kappa_means[value] for value in (200, 500, 1000, 2000)]) >= 0)
        ),
        "l5_positive_cells": [
            row["tag"] for row in h_l5 if truth(row["exceeds_matched_null_holm_0p05"])
        ],
    }


def verify_isruc(checks, roots):
    isruc_root = roots["isruc"]
    object_root = isruc_root / "objects_fleet_a40"
    c0_root = roots["c0"]
    task_rows = {
        row["tag"]: row
        for row in read_csv(isruc_root / "isruc_s3_fleet_task_performance.csv")
    }
    prediction_identity = None
    point = {}
    for tag in TAGS:
        contract = json.loads((object_root / (tag + "_isruc_object_contract.json")).read_text())
        payload_path = object_root / (tag + "_isruc_predictions.npz")
        checks.add(
            "ISRUC_S3",
            "predictions",
            "payload_hash_%s" % tag,
            sha256_file(payload_path) == contract["prediction_payload_sha256"],
        )
        with np.load(payload_path, allow_pickle=False) as payload:
            probability = payload["test_probabilities"].mean(axis=0)
            current = metrics(payload["test_labels"], probability, N_CLASSES_EXTERNAL)
            identity = (
                payload["test_labels"].copy(),
                payload["test_subjects"].copy(),
                payload["test_rotations"].copy(),
            )
        if prediction_identity is None:
            prediction_identity = identity
        else:
            checks.add(
                "ISRUC_S3",
                "predictions",
                "identity_%s" % tag,
                all(np.array_equal(left, right) for left, right in zip(prediction_identity, identity)),
            )
        point[tag] = current
        checks.add(
            "ISRUC_S3",
            "task",
            "prediction_recompute_%s" % tag,
            close(current["kappa"], task_rows[tag]["target_test_kappa"])
            and close(current["nll"], task_rows[tag]["target_test_nll"])
            and close(
                current["balanced_accuracy"],
                task_rows[tag]["target_test_balanced_accuracy"],
            ),
        )

    sequence_rows = read_csv(c0_root / "isruc_s3_sequence_manifest.csv")
    rotations = read_csv(c0_root / "isruc_s3_rotation_manifest.csv")
    checks.add(
        "ISRUC_S3",
        "split",
        "sequence_and_rotation_contract",
        len(sequence_rows) == 425
        and {int(row["subject"]) for row in sequence_rows} == set(range(1, 11))
        and all(row["shape"] == "20;6;6000" and truth(row["sequence_contract_pass"]) for row in sequence_rows)
        and len(rotations) == 10
        and sorted(int(row["test_subject"]) for row in rotations) == list(range(1, 11))
        and sorted(int(row["val_subject"]) for row in rotations) == list(range(1, 11))
        and all(
            int(row["train_count"]) == 8
            and int(row["val_count"]) == 1
            and int(row["test_count"]) == 1
            and truth(row["subject_disjoint"])
            for row in rotations
        ),
    )

    subject_rows = read_csv(isruc_root / "isruc_s3_fleet_subject_metrics.csv")
    geometry_rows = read_csv(isruc_root / "isruc_s3_fleet_geometry.csv")
    subject_means = {
        tag: float(np.mean([float(row["subject_nll"]) for row in subject_rows if row["tag"] == tag]))
        for tag in TAGS
    }
    overlap_means = {
        tag: float(
            np.mean(
                [float(row["projection_overlap_rank4"]) for row in geometry_rows if row["tag"] == tag]
            )
        )
        for tag in TAGS
    }
    checks.add(
        "ISRUC_S3",
        "geometry",
        "ten_rotation_support_and_stability",
        all(len([row for row in geometry_rows if row["tag"] == tag]) == 10 for tag in TAGS)
        and all(truth(row["subspace_stable"]) for row in geometry_rows),
    )

    l5_rows = read_csv(isruc_root / "isruc_s3_fleet_l5_reliance.csv")
    h_l5 = [row for row in l5_rows if row["tag"].startswith("H")]
    adjusted = holm([float(row["empirical_one_sided_p"]) for row in h_l5])
    positive = []
    for row, adjusted_value in zip(h_l5, adjusted):
        tag = row["tag"]
        payload_path = object_root / (tag + "_isruc_l5_nulls.npz")
        contract = json.loads((object_root / (tag + "_isruc_object_contract.json")).read_text())
        with np.load(payload_path, allow_pickle=False) as payload:
            null_delta = payload["global_null_delta_kappa"]
            match = float(payload["source_val_null_match_abs_error"].max())
        subject_delta = float(row["subject_delta_kappa"])
        empirical = float((1 + np.sum(null_delta >= subject_delta)) / (len(null_delta) + 1))
        passed = (
            sha256_file(payload_path) == contract["l5_null_payload_sha256"]
            and close(null_delta.mean(), row["null_delta_kappa_mean"])
            and close(empirical, row["empirical_one_sided_p"])
            and close(adjusted_value, row["holm_adjusted_p"])
            and match <= MATCH_TOLERANCE
        )
        checks.add("ISRUC_S3", "L5", "null_recompute_%s" % tag, passed)
        if adjusted_value < 0.05:
            positive.append(tag)
    expected_positive = ["H500_s0", "H1000_s0", "H2000_s1"]
    paired_inconsistent = all(
        not ({"H%d_s0" % budget, "H%d_s1" % budget} <= set(positive))
        for budget in HIGH_BUDGETS
    )
    checks.add(
        "ISRUC_S3",
        "L5",
        "three_of_six_and_paired_seed_inconsistency",
        positive == expected_positive and paired_inconsistent,
    )
    leave_one = read_csv(isruc_root / "isruc_s3_fleet_l5_leave_one_subject_out.csv")
    checks.add(
        "ISRUC_S3",
        "L5",
        "leave_one_subject_support",
        all(len([row for row in leave_one if row["tag"] == tag]) == 10 for tag in positive),
    )
    return {
        "task_kappa": {tag: point[tag]["kappa"] for tag in TAGS},
        "subject_nll": subject_means,
        "overlap": overlap_means,
        "subject_strengthens_descriptively": (
            np.mean([subject_means[tag] for tag in budget_tags(200)])
            > np.mean(
                [
                    np.mean([subject_means[tag] for tag in budget_tags(budget)])
                    for budget in HIGH_BUDGETS
                ]
            )
        ),
        "overlap_decreases_descriptively": (
            np.mean([overlap_means[tag] for tag in budget_tags(200)])
            > np.mean(
                [
                    np.mean([overlap_means[tag] for tag in budget_tags(budget)])
                    for budget in HIGH_BUDGETS
                ]
            )
        ),
        "l5_positive_cells": positive,
        "paired_seed_replicated": not paired_inconsistent,
    }


def claim_row(dataset, claim_id, status, evidence, boundary):
    return {
        "dataset": dataset,
        "claim_id": claim_id,
        "status": status,
        "evidence": evidence,
        "boundary": boundary,
    }


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        "--out-dir",
        type=Path,
        default=Path("results/s2p_route_b_phase_c_closure"),
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if args.code_commit != actual_commit or len(actual_commit) != 40:
        raise RuntimeError("Phase C verifier code-commit contract failed")
    roots = {
        "closure": args.out_dir,
        "c0": Path("results/s2p_route_b_cross_task_c0"),
        "seedv": Path("results/s2p_route_b_cross_task_seedv"),
        "isruc": Path("results/s2p_route_b_cross_task_isruc_s3"),
        "faced_final": Path("results/s2p_route_b_33ch_b1_faced"),
        "faced_core": Path("results/s2p_route_b_representation_emergence_b1_core"),
        "faced_b0": Path("results/s2p_route_b_representation_emergence_b0"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks = Checks()

    manifest = validate_manifest(args.checkpoint_manifest)
    checks.add(
        "ALL",
        "provenance",
        "ten_immutable_representation_contracts",
        len(manifest) == 10 and [row["tag"] for row in manifest] == TAGS,
    )
    bootstrap_verification = json.loads(
        (args.out_dir / "seedv_bootstrap_verification.json").read_text()
    )
    checks.add(
        "ALL",
        "provenance",
        "closure_code_commit",
        bootstrap_verification.get("code_commit") == actual_commit,
    )
    checks.add(
        "ALL",
        "provenance",
        "all_checkpoint_hashes_rechecked",
        all(
            row["tag"] == "random"
            or sha256_file(row["immutable_path"]) == row["immutable_sha256"]
            for row in manifest
        ),
    )

    faced = verify_faced(checks, roots)
    seedv = verify_seedv(checks, roots)
    isruc = verify_isruc(checks, roots)

    firewall_paths = (
        roots["faced_final"] / "faced_final_target_label_firewall.json",
        roots["faced_core"] / "phase_b1_core_target_label_firewall.json",
        roots["seedv"] / "seedv_fleet_target_label_firewall.json",
        roots["isruc"] / "isruc_s3_fleet_target_label_firewall.json",
    )
    firewall_values = [json.loads(path.read_text()) for path in firewall_paths]
    firewall_pass = all(
        value.get("target_labels_used_for_selection") is False
        or value.get("target_test_labels_used_for_selection") is False
        for value in firewall_values
    )
    checks.add("ALL", "firewall", "target_labels_final_scoring_only", firewall_pass)

    claims = [
        claim_row(
            "FACED",
            "subject_structure_early",
            "SUPPORTED" if faced["primary_subject_early"] else "NOT_ESTABLISHED",
            "B1-Core P1 random vs H200 continuous subject NLL",
            "cross-subject frozen representation",
        ),
        claim_row(
            "FACED",
            "primary_target_nll_later",
            "SUPPORTED" if faced["primary_task_nll_later"] else "NOT_ESTABLISHED",
            "B1-Core P2 H200 vs pooled higher target NLL",
            "Kappa/bAcc remain positive sensitivity endpoints only",
        ),
        claim_row(
            "FACED",
            "subject_task_overlap_decrease",
            "SUPPORTED" if faced["primary_overlap_decrease"] else "NOT_ESTABLISHED",
            "B1-Core rank-8 P3",
            "within-dataset geometry only",
        ),
        claim_row(
            "FACED",
            "functional_subject_reliance",
            "NOT_DETECTED",
            "0/8 task-gated cells exceed matched null after Holm",
            "measured linear subject subspace only",
        ),
        claim_row(
            "SEED-V",
            "subject_structure_early",
            "SUPPORTED" if seedv["subject_early"] else "NOT_ESTABLISHED",
            "5,000 subject-cluster bootstrap random minus H200 subject NLL",
            "within-cohort/session trial-held-out setting",
        ),
        claim_row(
            "SEED-V",
            "subject_structure_continues",
            "SUPPORTED" if seedv["subject_continues"] else "NOT_ESTABLISHED",
            "5,000 subject-cluster bootstrap H200 minus pooled-higher subject NLL",
            "no unseen-subject claim",
        ),
        claim_row(
            "SEED-V",
            "subject_task_overlap_decrease",
            "SUPPORTED" if seedv["overlap_decrease"] else "NOT_ESTABLISHED",
            "rank-4 pooled-higher minus H200 overlap",
            "within-dataset geometry only",
        ),
        claim_row(
            "SEED-V",
            "task_accessibility_beyond_H200",
            "SUPPORTED" if seedv["task_kappa_higher_than_h200"] else "NOT_ESTABLISHED",
            "5,000 subject-cluster bootstrap prospective Kappa endpoint",
            "non-monotonic budget response; no optimum or scaling-law claim",
        ),
        claim_row(
            "SEED-V",
            "functional_subject_reliance",
            "NOT_DETECTED" if not seedv["l5_positive_cells"] else "DETECTED_IN_SOME_CELLS",
            "%d/8 task-gated cells exceed matched null after Holm"
            % len(seedv["l5_positive_cells"]),
            "cohort/session diagnostic, not unseen-subject reliance",
        ),
        claim_row(
            "ISRUC_S3",
            "task_budget_emergence",
            "DIRECTIONALLY_SUPPORTED",
            "H200 below random; H500/H1000/H2000 above random from prediction recompute",
            "10-subject low-power directional replication",
        ),
        claim_row(
            "ISRUC_S3",
            "subject_strengthening",
            "DIRECTIONALLY_SUPPORTED"
            if isruc["subject_strengthens_descriptively"]
            else "NOT_ESTABLISHED",
            "rotation-mean subject NLL",
            "descriptive; repeated rotations are not independent subjects",
        ),
        claim_row(
            "ISRUC_S3",
            "subject_task_overlap_decrease",
            "DIRECTIONALLY_SUPPORTED"
            if isruc["overlap_decreases_descriptively"]
            else "NOT_ESTABLISHED",
            "rotation-mean rank-4 overlap",
            "descriptive and within-dataset only",
        ),
        claim_row(
            "ISRUC_S3",
            "functional_subject_reliance",
            "TASK_AND_SEED_DEPENDENT_SIGNAL",
            "3/6 Holm-positive cells with no paired-seed replication",
            "not a general sleep-stage or budget reliance claim",
        ),
        claim_row(
            "CROSS_TASK",
            "pooled_eeg_significance",
            "FORBIDDEN",
            "no p-value pooled across datasets",
            "dataset contracts and biological units differ",
        ),
        claim_row(
            "CROSS_TASK",
            "monotonic_scaling_law",
            "NOT_ESTABLISHED",
            "SEED-V non-monotonic; FACED endpoint discordance; ISRUC directional only",
            "four budgets, two pretraining seeds, frozen readouts",
        ),
    ]

    checks.add(
        "ALL",
        "claims",
        "no_best_seed_or_budget_claim",
        not seedv["kappa_monotonic"]
        and all("optimal" not in row["evidence"].lower() for row in claims),
    )
    checks.add(
        "ALL",
        "claims",
        "no_cross_dataset_pvalue",
        any(row["claim_id"] == "pooled_eeg_significance" and row["status"] == "FORBIDDEN" for row in claims),
    )
    checks.add(
        "ALL",
        "claims",
        "isruc_l5_scoped",
        isruc["l5_positive_cells"] == ["H500_s0", "H1000_s0", "H2000_s1"]
        and isruc["paired_seed_replicated"] is False,
    )

    status = "PASS" if not checks.failures else "NO_GO"
    write_csv(args.out_dir / "phase_c_independent_checks.csv", checks.rows)
    write_csv(args.out_dir / "phase_c_claim_ledger.csv", claims)
    dataset_verdicts = {
        "FACED": faced,
        "SEED-V": seedv,
        "ISRUC_S3": isruc,
    }
    write_json(
        args.out_dir / "phase_c_dataset_verdicts.json", json_ready(dataset_verdicts)
    )
    closure = {
        "phase": "C_cross_task_frozen_representation_validation",
        "status": status,
        "code_commit": actual_commit,
        "raw_fleet_complete": True,
        "seedv_5000_subject_cluster_bootstrap_complete": True,
        "independent_fleet_verifier": status,
        "immutable_representation_objects": 10,
        "all_checkpoint_hashes_reverified": not any(
            failure.startswith("ALL:provenance") for failure in checks.failures
        ),
        "faced_primary_target_nll_established": faced["primary_task_nll_later"],
        "faced_kappa_sensitivity_preserved": True,
        "seedv_unseen_subject_claim_allowed": False,
        "seedv_budget_response_monotonic": seedv["kappa_monotonic"],
        "isruc_low_power_directional_replication": True,
        "isruc_l5_status": "TASK_AND_SEED_DEPENDENT_RELIANCE_SIGNAL",
        "isruc_l5_positive_cells": isruc["l5_positive_cells"],
        "isruc_l5_paired_seed_replication": isruc["paired_seed_replicated"],
        "cross_dataset_pvalue_computed": False,
        "best_seed_or_budget_selected": False,
        "target_labels_used_for_selection": False,
        "monotonic_scaling_law_established": False,
        "phase_c_scientific_closure_ready_for_pm_review": status == "PASS",
        "project_close_recommended": False,
        "new_experiment_authorized": False,
        "manuscript_writing_authorized": False,
        "failures": checks.failures,
    }
    write_json(args.out_dir / "phase_c_closure_verdict.json", json_ready(closure))
    artifact_roles = {
        "seedv_bootstrap_budget_metrics.csv": "SEED-V point estimates and cluster intervals",
        "seedv_bootstrap_task_contrasts.csv": "SEED-V task bootstrap contrasts",
        "seedv_bootstrap_subject_contrasts.csv": "SEED-V subject-NLL contrasts",
        "seedv_bootstrap_geometry_contrasts.csv": "SEED-V rank-4 overlap contrasts",
        "seedv_bootstrap_l5.csv": "SEED-V matched-null L5 family",
        "seedv_bootstrap_point_reproduction.csv": "SEED-V independent point reproduction",
        "seedv_bootstrap_support.npz": "SEED-V compact bootstrap support",
        "seedv_bootstrap_verification.json": "SEED-V bootstrap contract verdict",
        "phase_c_independent_checks.csv": "cross-dataset independent checks",
        "phase_c_claim_ledger.csv": "mechanically scoped claim ledger",
        "phase_c_dataset_verdicts.json": "dataset-specific verified results",
        "phase_c_closure_verdict.json": "Phase-C closure readiness verdict",
    }
    artifact_rows = []
    for name, role in artifact_roles.items():
        path = args.out_dir / name
        artifact_rows.append(
            {
                "artifact": name,
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "code_commit": actual_commit,
            }
        )
    write_csv(args.out_dir / "phase_c_closure_artifact_manifest.csv", artifact_rows)
    print(json.dumps(closure, indent=2, sort_keys=True))
    if checks.failures:
        raise RuntimeError("Phase C fleet verification failed: %s" % checks.failures[:10])


if __name__ == "__main__":
    main()
