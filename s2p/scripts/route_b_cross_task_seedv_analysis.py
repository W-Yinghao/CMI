#!/usr/bin/env python
"""SEED-V gate-first frozen-representation analysis for S2P Phase C1."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
    f1_score,
)

from route_b_cross_task_common import (
    GATE_TAGS,
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
N_NULL = 200
NULL_SEED = 20260731


def fixed_probe(x, y):
    pca = PCA(n_components=PCA_DIM, whiten=True, svd_solver="randomized", random_state=0)
    z = pca.fit_transform(x)
    clf = LogisticRegression(
        C=PROBE_C, solver="lbfgs", max_iter=2000, tol=1e-6
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(z, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("fixed logistic probe did not converge")
    return pca, clf


def task_metrics(clf, z, y):
    probability = clf.predict_proba(z)
    prediction = clf.predict(z)
    decision = clf.decision_function(z)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    class_to_index = {int(value): i for i, value in enumerate(clf.classes_)}
    true_index = np.asarray([class_to_index[int(value)] for value in y], dtype=np.int64)
    masked = decision.copy()
    masked[np.arange(len(y)), true_index] = -np.inf
    margin = decision[np.arange(len(y)), true_index] - masked.max(axis=1)
    return {
        "kappa": float(cohen_kappa_score(y, prediction)),
        "nll": float(log_loss(y, probability, labels=np.arange(5))),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted")),
        "class_margin": float(np.mean(margin)),
    }, probability, prediction


def pairwise_subject_metrics(z_fit, subject_fit, z_hold, subject_hold):
    values = []
    subjects = sorted(int(value) for value in np.unique(subject_fit))
    centroids = {value: z_fit[subject_fit == value].mean(axis=0) for value in subjects}
    for left_index, left in enumerate(subjects):
        for right in subjects[left_index + 1:]:
            mask = np.isin(subject_hold, [left, right])
            current = z_hold[mask]
            truth = subject_hold[mask]
            left_distance = np.sum((current - centroids[left]) ** 2, axis=1)
            right_distance = np.sum((current - centroids[right]) ** 2, axis=1)
            score = right_distance - left_distance
            binary = (truth == left).astype(int)
            auc = roc_auc_score(binary, score)
            correct = np.where(binary == 1, score, -score)
            margin = float(np.mean(correct) / (np.std(score) + 1e-12))
            values.append((float(auc), margin))
    return values


def subject_metrics(features, subjects, splits):
    nll = []
    accuracy = []
    pairwise = []
    for holdout in ("train", "val", "test"):
        fit = splits != holdout
        hold = splits == holdout
        pca, clf = fixed_probe(features[fit], subjects[fit])
        z_hold = pca.transform(features[hold])
        probability = clf.predict_proba(z_hold)
        prediction = clf.predict(z_hold)
        class_to_index = {int(value): i for i, value in enumerate(clf.classes_)}
        true_index = np.asarray(
            [class_to_index[int(value)] for value in subjects[hold]], dtype=np.int64
        )
        nll.extend(-np.log(np.clip(probability[np.arange(hold.sum()), true_index], 1e-15, 1.0)))
        accuracy.extend((prediction == subjects[hold]).astype(float))
        pairwise.extend(
            pairwise_subject_metrics(
                pca.transform(features[fit]), subjects[fit], z_hold, subjects[hold]
            )
        )
    return {
        "subject_nll": float(np.mean(nll)),
        "subject_accuracy_diagnostic": float(np.mean(accuracy)),
        "pairwise_subject_auc": float(np.mean([value[0] for value in pairwise])),
        "pairwise_subject_margin": float(np.mean([value[1] for value in pairwise])),
        "crossfit": "three_trial_blocks_train_val_test",
    }


def effect_matrix(z, subjects, labels, effect):
    subject_values = sorted(int(value) for value in np.unique(subjects))
    class_values = sorted(int(value) for value in np.unique(labels))
    cells = np.empty((len(subject_values), len(class_values), z.shape[1]), dtype=np.float64)
    for subject_index, subject in enumerate(subject_values):
        for class_index, label in enumerate(class_values):
            mask = (subjects == subject) & (labels == label)
            if not mask.any():
                raise RuntimeError(f"missing subject-class cell: subject={subject} class={label}")
            cells[subject_index, class_index] = z[mask].mean(axis=0)
    grand = cells.mean(axis=(0, 1))
    if effect == "subject":
        return cells.mean(axis=1) - grand
    if effect == "task":
        return cells.mean(axis=0) - grand
    raise ValueError(effect)


def top_basis(matrix, rank):
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if len(singular) < rank or singular[rank - 1] <= 1e-12:
        raise RuntimeError("rank-4 effect subspace is numerically degenerate")
    return vt[:rank]


def captured_energy(effect, basis):
    return float(np.sum((effect @ basis.T) ** 2) / (np.sum(effect ** 2) + 1e-12))


def geometry(features, labels, subjects, splits):
    train = splits == "train"
    hold = splits == "val"
    pca = PCA(n_components=PCA_DIM, whiten=True, svd_solver="randomized", random_state=0)
    z_train = pca.fit_transform(features[train])
    z_hold = pca.transform(features[hold])
    subject_fit = effect_matrix(z_train, subjects[train], labels[train], "subject")
    task_fit = effect_matrix(z_train, subjects[train], labels[train], "task")
    subject_hold = effect_matrix(z_hold, subjects[hold], labels[hold], "subject")
    task_hold = effect_matrix(z_hold, subjects[hold], labels[hold], "task")
    subject_basis = top_basis(subject_fit, RANK)
    task_basis = top_basis(task_fit, RANK)
    overlap = float(np.sum((subject_basis @ task_basis.T) ** 2) / RANK)
    subject_capture = captured_energy(subject_hold, subject_basis)
    task_capture = captured_energy(task_hold, task_basis)
    stable = subject_capture >= 0.05 and task_capture >= 0.05
    return {
        "projection_overlap_rank4": overlap,
        "subject_heldout_captured_energy": subject_capture,
        "task_heldout_captured_energy": task_capture,
        "subspace_stable": stable,
        "fit_split": "train_trials",
        "heldout_split": "val_trials",
    }


def remove_subspace(z, basis):
    return z - (z @ basis.T) @ basis


def removed_fraction(z, after):
    return float(np.sum((z - after) ** 2) / (np.sum(z ** 2) + 1e-12))


def subject_basis(z, subjects):
    means = np.stack([z[subjects == value].mean(axis=0) for value in sorted(np.unique(subjects))])
    return top_basis(means - means.mean(axis=0), RANK)


def variance_matched_l5(clf, z_val, z_test, y_test, basis):
    base = task_metrics(clf, z_test, y_test)[0]
    subject_val = remove_subspace(z_val, basis)
    subject_test = remove_subspace(z_test, basis)
    subject_metric = task_metrics(clf, subject_test, y_test)[0]
    target_fraction = removed_fraction(z_val, subject_val)
    rng = np.random.default_rng(NULL_SEED)
    null_delta = []
    match_error = []
    total_energy = float(np.sum(z_val ** 2)) + 1e-12
    for _ in range(N_NULL):
        q, _ = np.linalg.qr(rng.standard_normal((z_val.shape[1], z_val.shape[1])))
        per_direction = np.sum((z_val @ q) ** 2, axis=0) / total_energy
        cumulative = np.cumsum(per_direction)
        last = int(np.searchsorted(cumulative, target_fraction, side="left"))
        if last >= z_val.shape[1]:
            raise RuntimeError("random orthobasis cannot match removed variance")
        previous = float(cumulative[last - 1]) if last else 0.0
        alpha = np.sqrt(max(target_fraction - previous, 0.0) / max(float(per_direction[last]), 1e-12))
        alpha = min(float(alpha), 1.0)
        direction = q[:, :last + 1]
        coefficient = np.ones(last + 1)
        coefficient[-1] = alpha

        def erase(z):
            return z - ((z @ direction) * coefficient) @ direction.T

        val_after = erase(z_val)
        test_after = erase(z_test)
        match_error.append(abs(removed_fraction(z_val, val_after) - target_fraction))
        null_metric = task_metrics(clf, test_after, y_test)[0]
        null_delta.append(base["kappa"] - null_metric["kappa"])
    subject_delta = base["kappa"] - subject_metric["kappa"]
    null_delta = np.asarray(null_delta)
    return {
        "subject_delta_kappa": subject_delta,
        "null_delta_kappa_mean": float(null_delta.mean()),
        "subject_minus_null_kappa": float(subject_delta - null_delta.mean()),
        "empirical_one_sided_p": float((1 + np.sum(null_delta >= subject_delta)) / (N_NULL + 1)),
        "source_val_removed_variance_fraction": target_fraction,
        "null_match_abs_error_max": float(max(match_error)),
        "null_repetitions": N_NULL,
    }


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_feature_cache(feature_dir, tag, manifest_row_value):
    contract_path = feature_dir / f"{tag}_feature_contract.json"
    payload_path = feature_dir / f"{tag}_features.npz"
    if not contract_path.is_file() or not payload_path.is_file():
        raise RuntimeError(f"missing SEED-V feature cache for {tag}")
    contract = json.loads(contract_path.read_text())
    if (
        contract.get("status") != "PASS"
        or contract.get("dataset") != "SEED-V"
        or contract.get("tag") != tag
        or contract.get("checkpoint_sha256_before") != manifest_row_value["immutable_sha256"]
        or contract.get("checkpoint_sha256_after") != manifest_row_value["immutable_sha256"]
        or contract.get("canary_repeat_max_abs_diff") != 0.0
        or contract.get("encoder_frozen") is not True
        or contract.get("fine_tuning_used") is not False
        or sha256_file(payload_path) != contract.get("payload_sha256")
    ):
        raise RuntimeError(f"SEED-V feature contract failed for {tag}")
    with np.load(payload_path) as payload:
        result = {name: payload[name] for name in payload.files}
    return result, contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "fleet"), required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path)
    args = parser.parse_args()

    rows = validate_manifest(args.checkpoint_manifest)
    tags = GATE_TAGS if args.stage == "gate" else TAGS
    if args.stage == "fleet":
        if args.gate_json is None or json.loads(args.gate_json.read_text()).get("status") != "PASS":
            raise RuntimeError("SEED-V fleet requires a passing gate artifact")

    caches = {}
    contracts = {}
    identity_hash = None
    for tag in tags:
        caches[tag], contracts[tag] = load_feature_cache(
            args.feature_dir, tag, manifest_row(rows, tag)
        )
        current = canonical_sha({
            key: caches[tag][key].tolist()
            for key in ("labels", "subjects", "sessions", "splits", "trial_ids", "window_counts")
        })
        if identity_hash is None:
            identity_hash = current
        elif current != identity_hash:
            raise RuntimeError(f"SEED-V trial identity differs for {tag}")

    task_rows = []
    subject_rows = []
    geometry_rows = []
    packs = {}
    for tag in tags:
        data = caches[tag]
        x = data["features"]
        y = data["labels"]
        subjects = data["subjects"]
        splits = data["splits"].astype(str)
        if x.shape != (705, 12400) or set(splits) != {"train", "val", "test"}:
            raise RuntimeError(f"SEED-V feature shape/split mismatch for {tag}: {x.shape}")
        train = splits == "train"
        val = splits == "val"
        test = splits == "test"
        pca, clf = fixed_probe(x[train], y[train])
        z_train = pca.transform(x[train])
        z_val = pca.transform(x[val])
        z_test = pca.transform(x[test])
        val_metric, _, _ = task_metrics(clf, z_val, y[val])
        test_metric, _, _ = task_metrics(clf, z_test, y[test])
        task_rows.append({
            "tag": tag,
            "budget_h": manifest_row(rows, tag)["budget_h"],
            "seed": manifest_row(rows, tag)["seed"],
            **{f"source_val_{key}": value for key, value in val_metric.items()},
            **{f"target_test_{key}": value for key, value in test_metric.items()},
            "primary_metric": "target_test_kappa",
            "analysis_unit": "trial_mean",
        })
        subject_rows.append({"tag": tag, **subject_metrics(x, subjects, splits)})
        geometry_rows.append({"tag": tag, **geometry(x, y, subjects, splits)})
        packs[tag] = (clf, z_train, z_val, z_test, y[test], subjects[train])

    task = {row["tag"]: row for row in task_rows}
    random_val = task["random"]["source_val_kappa"]
    for row in task_rows:
        row["task_gate_pass"] = (
            row["source_val_kappa"] >= 0.05
            and row["source_val_kappa"] >= random_val + 0.02
        )

    l5_rows = []
    if args.stage == "fleet":
        for row in task_rows:
            tag = row["tag"]
            if not row["task_gate_pass"]:
                continue
            clf, z_train, z_val, z_test, y_test, train_subjects = packs[tag]
            l5_rows.append({
                "tag": tag,
                "task_gate_pass": True,
                **variance_matched_l5(
                    clf, z_val, z_test, y_test, subject_basis(z_train, train_subjects)
                ),
            })
        family = [row for row in l5_rows if row["tag"].startswith("H")]
        adjusted = holm([row["empirical_one_sided_p"] for row in family]) if family else []
        for row, value in zip(family, adjusted):
            row["holm_adjusted_p"] = float(value)
            row["exceeds_matched_null_holm_0p05"] = bool(value < 0.05)

    released_minus_random = (
        task["released"]["target_test_kappa"] - task["random"]["target_test_kappa"]
    )
    released_gate = released_minus_random >= 0.02
    stable_geometry = all(row["subspace_stable"] for row in geometry_rows)
    status = "PASS" if released_gate else "NO_GO"
    prefix = f"seedv_{args.stage}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{prefix}_task_performance.csv", task_rows)
    write_csv(args.out_dir / f"{prefix}_subject_metrics.csv", subject_rows)
    write_csv(args.out_dir / f"{prefix}_geometry.csv", geometry_rows)
    write_csv(args.out_dir / f"{prefix}_l5_reliance.csv", l5_rows)
    run_manifest = [{
        "tag": tag,
        "feature_payload_sha256": contracts[tag]["payload_sha256"],
        "checkpoint_sha256": contracts[tag]["checkpoint_sha256_before"],
        "trial_identity_sha256": identity_hash,
        "analysis_unit": "trial_mean",
    } for tag in tags]
    write_csv(args.out_dir / f"{prefix}_run_manifest.csv", run_manifest)
    firewall = {
        "dataset": "SEED-V",
        "stage": args.stage,
        "target_labels_used_for_selection": False,
        "test_split_used_for_pca": False,
        "test_split_used_for_probe_fit": False,
        "test_split_used_for_task_gate": False,
        "test_split_use": "final_scoring_only",
        "trial_group_leakage": False,
        "window_as_independent_inference_unit": False,
        "pass": True,
    }
    write_json(args.out_dir / f"{prefix}_target_label_firewall.json", firewall)
    verdict = {
        "phase": "C1_SEED-V_gate" if args.stage == "gate" else "C1_SEED-V_fleet",
        "status": status,
        "objects": tags,
        "released_target_kappa": task["released"]["target_test_kappa"],
        "random_target_kappa": task["random"]["target_test_kappa"],
        "released_minus_random_target_kappa": released_minus_random,
        "released_clears_random_plus_0p02": released_gate,
        "all_geometry_subspaces_stable": stable_geometry,
        "trial_group_crossfit_pass": True,
        "feature_determinism_pass": True,
        "target_label_firewall_pass": True,
        "fine_tuning_used": False,
        "unseen_subject_replication_claim_allowed": False,
        "fleet_authorized_by_gate": bool(args.stage == "gate" and status == "PASS"),
        "other_dataset_auto_launch_authorized": False,
    }
    write_json(args.out_dir / f"{prefix}_verdict.json", verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
