#!/usr/bin/env python
"""Independent verification of variance-excluded Phase-B1-Core outputs."""
import argparse
import json
import stat
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

import route_b_phase_b1_verify as base


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def run(args):
    out = Path(args.out_dir)
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    required = [
        "phase_b1_variance_forensic_replay.csv",
        "phase_b1_variance_gate_components.json",
        "phase_b1_variance_reproducibility.json",
        "phase_b1_variance_family_disposition.json",
        "phase_b1_core_run_manifest.csv",
        "phase_b1_core_checkpoint_hash_recheck.csv",
        "phase_b1_core_feature_manifest.csv",
        "phase_b1_core_subject_metrics.csv",
        "phase_b1_core_subject_cluster_scores.csv",
        "phase_b1_core_subject_pair_metrics.csv",
        "phase_b1_core_task_metrics.csv",
        "phase_b1_core_task_sample_scores.csv",
        "phase_b1_core_geometry.csv",
        "phase_b1_core_primary_inference.json",
        "phase_b1_core_sensitivity.csv",
        "phase_b1_core_target_label_firewall.json",
        "phase_b1_core_verdict.json",
        "phase_b1_core_primary_bootstrap_samples.csv",
        "phase_b1_core_support.npz",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"missing B1-Core files: {missing}")

    disposition = json.loads((out / "phase_b1_variance_family_disposition.json").read_text())
    reproducibility = json.loads((out / "phase_b1_variance_reproducibility.json").read_text())
    check(disposition.get("status") == "FAILED_STABILITY_NOT_INTERPRETABLE", "variance_disposition")
    check(disposition.get("all_checkpoints_excluded_from_variance_claims") is True, "variance_all_excluded")
    check(disposition.get("variance_used_in_primary_inference") is False, "variance_not_primary")
    check(disposition.get("variance_used_in_mechanism_verdict") is False, "variance_not_mechanism")
    check(reproducibility.get("status") in ("REPRODUCED_AT_TAG_GATE_LEVEL", "NONDETERMINISTIC_UNRELIABLE"), "variance_replay_status")

    closure_rows = base.read_csv(args.checkpoint_manifest)
    check([row["tag"] for row in closure_rows] == base.TAGS, "closure_tags")
    closure = {row["tag"]: row for row in closure_rows}
    hash_rows = base.read_csv(out / "phase_b1_core_checkpoint_hash_recheck.csv")
    check([row["tag"] for row in hash_rows] == base.TAGS, "hash_tags")
    for row in hash_rows:
        tag = row["tag"]
        expected = closure[tag]["immutable_sha256"]
        check(base.as_bool(row["hash_recheck_pass"]), f"hash_flag:{tag}")
        check(row["expected_sha256"] == expected, f"hash_expected:{tag}")
        if tag == "random":
            check(row["sha256_at_global_end"] == "logical_contract", "random_hash_contract")
        else:
            path = Path(row["immutable_path"])
            check(path.is_file() and not path.is_symlink(), f"immutable_direct:{tag}")
            check(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0, f"immutable_writable:{tag}")
            check(base.sha256_file(path) == expected, f"immutable_hash:{tag}")
            check(all(row[key] == expected for key in (
                "sha256_at_object_start", "sha256_at_object_end", "sha256_at_global_end"
            )), f"three_hash_checks:{tag}")

    feature_rows = base.read_csv(out / "phase_b1_core_feature_manifest.csv")
    check([row["tag"] for row in feature_rows] == base.TAGS, "feature_tags")
    for row in feature_rows:
        tag = row["tag"]
        check(base.as_bool(row["feature_contract_pass"]), f"feature_flag:{tag}")
        check(float(row["checksum_start_end_max_abs_diff"]) == 0, f"feature_repeat:{tag}")
        check(
            row["checksum_feature_sha256_start"]
            == row["checksum_feature_sha256_end"]
            == row["closure_feature_sha256"]
            == closure[tag]["feature_hash"],
            f"feature_closure:{tag}",
        )

    subject_rows = base.read_csv(out / "phase_b1_core_subject_metrics.csv")
    subject_clusters = base.read_csv(out / "phase_b1_core_subject_cluster_scores.csv")
    subject_pairs = base.read_csv(out / "phase_b1_core_subject_pair_metrics.csv")
    subject_lookup = {row["tag"]: row for row in subject_rows}
    cluster_by_tag = {}
    pair_by_tag = {}
    check(len(subject_rows) == 10, "subject_metric_count")
    check(len(subject_clusters) == 800, "subject_cluster_count")
    check(len(subject_pairs) == 31600, "subject_pair_count")
    for tag in base.TAGS:
        clusters = [row for row in subject_clusters if row["tag"] == tag]
        pairs = [row for row in subject_pairs if row["tag"] == tag]
        cluster_by_tag[tag] = clusters
        pair_by_tag[tag] = pairs
        check([int(row["subject"]) for row in clusters] == list(range(1, 81)), f"subject_order:{tag}")
        for output_key, support_key in (
            ("heldout_clip_subject_nll", "heldout_clip_subject_nll"),
            ("heldout_clip_subject_accuracy_diagnostic", "heldout_clip_subject_accuracy_diagnostic"),
            ("heldout_clip_true_subject_probability", "heldout_clip_true_subject_probability"),
            ("retrieval_map", "retrieval_map"),
            ("class_conditional_subject_nll", "class_conditional_subject_nll_macro"),
        ):
            observed = np.mean([float(row[support_key]) for row in clusters])
            check(base.close(subject_lookup[tag][output_key], observed), f"subject_metric:{tag}:{output_key}")
        for output_key, support_key in (
            ("pairwise_subject_auc", "pairwise_auc"),
            ("pairwise_subject_margin", "pairwise_standardized_margin"),
            ("pairwise_subject_bacc_diagnostic", "pairwise_bacc_diagnostic"),
        ):
            observed = np.mean([float(row[support_key]) for row in pairs])
            check(base.close(subject_lookup[tag][output_key], observed), f"pair_metric:{tag}:{output_key}")

    task_rows = base.read_csv(out / "phase_b1_core_task_metrics.csv")
    task_samples = base.read_csv(out / "phase_b1_core_task_sample_scores.csv")
    task_lookup = {(row["tag"], row["split"]): row for row in task_rows}
    subject_support = [row for row in task_samples if row["record_type"] == "subject_probe_support"]
    task_support = [row for row in task_samples if row["record_type"] == "task_probe_support"]
    check(len(subject_support) == 10 * 80 * 28 * 3, "subject_support_count")
    check(len(task_support) == 10 * 43 * 28 * 3, "task_support_count")
    for tag in base.TAGS:
        current_subject = [row for row in subject_support if row["tag"] == tag]
        unique = {(int(row["subject"]), int(row["clip_id"]), int(row["segment_id"])) for row in current_subject}
        check(len(unique) == 80 * 28 * 3, f"subject_support_unique:{tag}")
        check(all(int(row["fold"]) == base.CLIP_TO_FOLD[int(row["clip_id"])] for row in current_subject), f"clip_fold:{tag}")
        check(all(row["label_final_scoring_only"] == "" for row in current_subject), f"subject_label_firewall:{tag}")
        for split, expected_count in (("source_val", 20 * 28 * 3), ("target_test", 23 * 28 * 3)):
            rows = [row for row in task_support if row["tag"] == tag and row["split"] == split]
            check(len(rows) == expected_count, f"task_support:{tag}:{split}")
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
                check(base.close(output[key], value), f"task_metric:{tag}:{split}:{key}")

    support = np.load(out / "phase_b1_core_support.npz", allow_pickle=False)
    check(support["tags"].tolist() == base.TAGS, "support_tags")
    rng = np.random.default_rng(base.BOOTSTRAP_SEED)
    source_counts = rng.multinomial(80, np.full(80, 1 / 80), size=base.N_BOOTSTRAP)
    target_counts = rng.multinomial(23, np.full(23, 1 / 23), size=base.N_BOOTSTRAP)
    check(np.array_equal(support["source_counts"], source_counts), "source_counts")
    check(np.array_equal(support["target_counts"], target_counts), "target_counts")
    source_weights = source_counts / source_counts.sum(axis=1, keepdims=True)
    target_weights = target_counts / target_counts.sum(axis=1, keepdims=True)

    subject_nll = {}
    subject_margin = {}
    subject_auc = {}
    target_nll = {}
    target_kappa = {}
    target_bacc = {}
    for tag_index, tag in enumerate(base.TAGS):
        cluster_values = np.asarray([float(row["heldout_clip_subject_nll"]) for row in cluster_by_tag[tag]])
        subject_nll[tag] = source_weights @ cluster_values
        subject_margin[tag] = base.weighted_pair_bootstrap(pair_by_tag[tag], source_counts, "pairwise_standardized_margin")
        subject_auc[tag] = base.weighted_pair_bootstrap(pair_by_tag[tag], source_counts, "pairwise_auc")
        check(np.allclose(subject_nll[tag], support["subject_nll_samples"][tag_index]), f"subject_nll_boot:{tag}")
        check(np.allclose(subject_margin[tag], support["subject_margin_samples"][tag_index]), f"subject_margin_boot:{tag}")
        check(np.allclose(subject_auc[tag], support["subject_auc_samples"][tag_index]), f"subject_auc_boot:{tag}")

        rows = [row for row in task_support if row["tag"] == tag and row["split"] == "target_test"]
        subjects = list(range(101, 124))
        nll_by_subject = np.asarray([
            np.mean([float(row["nll"]) for row in rows if int(row["subject"]) == subject])
            for subject in subjects
        ])
        target_nll[tag] = target_weights @ nll_by_subject
        confusion = np.zeros((23, base.N_CLASSES, base.N_CLASSES))
        for index, subject in enumerate(subjects):
            for row in rows:
                if int(row["subject"]) == subject:
                    confusion[index, int(row["label_final_scoring_only"]), int(row["prediction"])] += 1
        boot_confusion = np.einsum("bs,sij->bij", target_counts, confusion, optimize=True)
        target_kappa[tag], target_bacc[tag] = base.metrics_from_confusion(boot_confusion)
        check(np.allclose(target_nll[tag], support["target_nll_samples"][tag_index]), f"target_nll_boot:{tag}")
        check(np.allclose(target_kappa[tag], support["target_kappa_samples"][tag_index]), f"target_kappa_boot:{tag}")
        check(np.allclose(target_bacc[tag], support["target_bacc_samples"][tag_index]), f"target_bacc_boot:{tag}")

    geometry_rows = base.read_csv(out / "phase_b1_core_geometry.csv")
    geometry_samples = {}
    for tag_index, tag in enumerate(base.TAGS):
        folds = sorted(
            [row for row in geometry_rows if row["tag"] == tag and row["scope"] == "fold"],
            key=lambda row: int(row["fold"]),
        )
        check(len(folds) == 3, f"geometry_folds:{tag}")
        geometry_samples[tag] = np.empty(base.N_BOOTSTRAP)
        for fold, row in enumerate(folds):
            fit = support[f"geometry_cell__{tag}__fold{fold}"]
            hold = support[f"geometry_hold_cell__{tag}__fold{fold}"]
            fit_geometry = base.effect_geometry(fit)
            hold_geometry = base.effect_geometry(hold)
            check(base.close(row["projection_overlap"], fit_geometry["overlap"]), f"geometry_overlap:{tag}:{fold}")
            subject_capture = base.captured_energy(hold_geometry["subject_effect"], fit_geometry["subject_basis"])
            task_capture = base.captured_energy(hold_geometry["task_effect"], fit_geometry["task_basis"])
            check(base.close(row["heldout_subject_self_capture"], subject_capture), f"subject_capture:{tag}:{fold}")
            check(base.close(row["heldout_task_self_capture"], task_capture), f"task_capture:{tag}:{fold}")
            check(subject_capture >= 0.05 and task_capture >= 0.05, f"geometry_stability:{tag}:{fold}")
        for replicate in range(base.N_BOOTSTRAP):
            geometry_samples[tag][replicate] = np.mean([
                base.effect_geometry(
                    support[f"geometry_cell__{tag}__fold{fold}"], source_weights[replicate]
                )["overlap"]
                for fold in range(3)
            ])
        check(np.allclose(geometry_samples[tag], support["geometry_overlap_samples"][tag_index]), f"geometry_boot:{tag}")
        mean_rows = [row for row in geometry_rows if row["tag"] == tag and row["scope"] == "checkpoint_mean"]
        check(len(mean_rows) == 1, f"geometry_mean_count:{tag}")
        check(base.close(mean_rows[0]["projection_overlap"], np.mean([float(row["projection_overlap"]) for row in folds])), f"geometry_mean:{tag}")

    primary = json.loads((out / "phase_b1_core_primary_inference.json").read_text())
    target_lookup = {row["tag"]: row for row in task_rows if row["split"] == "target_test"}
    geometry_lookup = {row["tag"]: row for row in geometry_rows if row["scope"] == "checkpoint_mean"}
    if primary["primary_subject_metric"] == "pairwise_subject_margin_fallback":
        p1 = base.mean_arrays(subject_margin, base.budget_tags(200)) - subject_margin["random"]
        p1_point = np.mean([float(subject_lookup[tag]["pairwise_subject_margin"]) for tag in base.budget_tags(200)]) - float(subject_lookup["random"]["pairwise_subject_margin"])
    else:
        p1 = subject_nll["random"] - base.mean_arrays(subject_nll, base.budget_tags(200))
        p1_point = float(subject_lookup["random"]["heldout_clip_subject_nll"]) - np.mean([float(subject_lookup[tag]["heldout_clip_subject_nll"]) for tag in base.budget_tags(200)])
    p2 = base.mean_arrays(target_nll, base.budget_tags(200)) - base.pool_high(target_nll)
    p2_point = np.mean([float(target_lookup[tag]["task_nll"]) for tag in base.budget_tags(200)]) - np.mean([
        np.mean([float(target_lookup[tag]["task_nll"]) for tag in base.budget_tags(budget)])
        for budget in base.HIGH_BUDGETS
    ])
    p3 = base.pool_high(geometry_samples) - base.mean_arrays(geometry_samples, base.budget_tags(200))
    p3_point = np.mean([
        np.mean([float(geometry_lookup[tag]["projection_overlap"]) for tag in base.budget_tags(budget)])
        for budget in base.HIGH_BUDGETS
    ]) - np.mean([float(geometry_lookup[tag]["projection_overlap"]) for tag in base.budget_tags(200)])
    samples = [p1, p2, p3]
    points = [p1_point, p2_point, p3_point]
    raw_p = [base.one_sided_positive_p(p1), base.one_sided_positive_p(p2), base.two_sided_sign_p(p3)]
    adjusted = base.holm_adjust(raw_p)
    check(len(primary["contrasts"]) == 3, "primary_count")
    for index, row in enumerate(primary["contrasts"]):
        expected = base.summarize(points[index], samples[index])
        for key, value in expected.items():
            check(base.close(row[key], value), f"primary_summary:{index}:{key}")
        check(base.close(row["p_raw"], raw_p[index]), f"primary_raw_p:{index}")
        check(base.close(row["p_holm"], adjusted[index]), f"primary_holm:{index}")
        check(base.as_bool(row["reject_holm_0p05"]) == (adjusted[index] < 0.05), f"primary_reject:{index}")

    primary_bootstrap = base.read_csv(out / "phase_b1_core_primary_bootstrap_samples.csv")
    check(len(primary_bootstrap) == base.N_BOOTSTRAP, "primary_bootstrap_count")
    for key, expected in zip(("p1_subject_delta", "p2_task_nll_delta", "p3_overlap_delta"), samples):
        check(np.allclose([float(row[key]) for row in primary_bootstrap], expected), f"primary_bootstrap:{key}")

    sensitivity_rows = base.read_csv(out / "phase_b1_core_sensitivity.csv")
    sensitivity = {row["analysis"]: row for row in sensitivity_rows}
    check(not any(name.startswith("variance_") for name in sensitivity), "variance_in_core_sensitivity")
    required_sensitivity = {
        "subject_nll_H200_minus_pooled_high",
        "pooled_high_minus_H200_target_kappa",
        "pooled_high_minus_H200_target_bacc",
        *{f"leave_high_budget_{budget}_out_task_nll" for budget in base.HIGH_BUDGETS},
        *{f"leave_high_budget_{budget}_out_overlap" for budget in base.HIGH_BUDGETS},
        *{f"retain_training_seed_{seed}_{name}" for seed in (0, 1) for name in ("subject_early", "task_nll", "overlap")},
    }
    check(required_sensitivity.issubset(sensitivity), "sensitivity_rows")

    firewall = json.loads((out / "phase_b1_core_target_label_firewall.json").read_text())
    selection_keys = [key for key in firewall if "selection" in key or key in (
        "target_labels_used_for_pca", "target_labels_used_for_probe_fit", "target_labels_used_for_rank_selection",
    )]
    check(all(firewall[key] is False for key in selection_keys), "target_firewall")

    verdict_path = out / "phase_b1_core_verdict.json"
    verdict_pre_verification_sha256 = base.sha256_file(verdict_path)
    verdict = json.loads(verdict_path.read_text())
    subject_uninformative = bool(primary["primary_subject_metric_uninformative"])
    subject_early = None if subject_uninformative else bool(adjusted[0] < 0.05 and p1_point > 0)
    task_later = bool(adjusted[1] < 0.05 and p2_point > 0)
    overlap_trend = (
        "increase" if adjusted[2] < 0.05 and p3_point > 0
        else "decrease" if adjusted[2] < 0.05 and p3_point < 0
        else "no_detectable_change"
    )
    continuing_row = sensitivity.get("subject_nll_H200_minus_pooled_high", {})
    continuing = None if subject_uninformative else bool(
        float(continuing_row.get("ci95_low", "-inf")) > 0
        and float(continuing_row.get("point", "-inf")) > 0
    )
    if subject_early is not True or not task_later or not verdict.get("all_subject_task_subspaces_stable"):
        mechanism = "D"
    elif overlap_trend == "increase":
        mechanism = "C"
    elif continuing:
        mechanism = "B"
    else:
        mechanism = "A"
    check(verdict.get("subject_structure_early") == subject_early, "verdict_subject_early")
    check(verdict.get("subject_continues_after_h200") == continuing, "verdict_subject_continues")
    check(verdict.get("task_structure_later") == task_later, "verdict_task_later")
    check(verdict.get("subject_task_overlap_trend") == overlap_trend, "verdict_overlap")
    check(verdict.get("mechanism_verdict") == mechanism, "verdict_mechanism")
    mechanism_labels = {
        "A": "SUBJECT_EARLY_TASK_LATER_LOW_OVERLAP",
        "B": "SUBJECT_CONTINUES_TASK_LATER_LOW_OVERLAP",
        "C": "TASK_LATER_WITH_INCREASING_SUBJECT_TASK_OVERLAP",
        "D": "CORE_GEOMETRY_UNRESOLVED",
    }
    check(verdict.get("mechanism_label") == mechanism_labels[mechanism], "verdict_mechanism_label")
    check(verdict.get("variance_used_in_mechanism_verdict") is False, "verdict_variance_excluded")
    check(verdict.get("interaction_claim_allowed") is False, "interaction_claim_forbidden")
    check(verdict.get("target_labels_used_for_selection") is False, "verdict_firewall")
    check(verdict.get("phase_b2_authorized") is False, "b2_held")

    verification = {
        "phase": "B1_core_independent_adversarial_verification",
        "status": "PASS" if not failures else "NO_GO",
        "checkpoint_objects": 10,
        "all_hashes_reverified": not any("hash" in item for item in failures),
        "all_features_match_closure": not any(item.startswith("feature") for item in failures),
        "clip_group_crossfit_pass": not any(item.startswith(("clip_fold", "subject_support")) for item in failures),
        "primary_subject_claim_recomputed": not any(item.startswith("primary") for item in failures),
        "primary_task_claim_recomputed": not any(item.startswith("primary") for item in failures),
        "primary_geometry_claim_recomputed": not any(item.startswith(("primary", "geometry")) for item in failures),
        "variance_partition_status": "FAILED_STABILITY_NOT_INTERPRETABLE",
        "variance_used_in_verdict": False,
        "target_labels_used_for_selection": False,
        "phase_b2_authorized": False,
        "core_verdict_pre_verification_sha256": verdict_pre_verification_sha256,
        "failures": failures,
    }
    write_json(out / "phase_b1_core_adversarial_verification.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"B1-Core independent verification failed: {failures[:10]}")
    verdict["independent_verification_status"] = "PASS"
    verdict["independent_verification_file"] = "phase_b1_core_adversarial_verification.json"
    write_json(verdict_path, verdict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/s2p_route_b_representation_emergence_b1_core")
    parser.add_argument("--checkpoint-manifest", default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        base.run_self_tests()
        summary = base.summarize(1.0, np.asarray([0.0, 1.0, 2.0]))
        if summary["point"] != 1.0:
            raise RuntimeError("B1-Core verifier summary API self-test failed")
        print("Phase B1-Core verifier self-tests: PASS")
        return
    run(args)


if __name__ == "__main__":
    main()
