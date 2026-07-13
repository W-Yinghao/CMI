#!/usr/bin/env python
"""Run one immutable CBraMod object through the frozen ISRUC sequence protocol."""

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch import nn

from route_b_cross_task_common import (
    TAGS,
    canonical_sha,
    configure_determinism,
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
HEAD_SEEDS = (0, 1, 2)


class SequenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1200, 512)
        self.activation = nn.GELU()
        layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=2048,
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(512, 5)

    def forward(self, x):
        return self.classifier(self.sequence(self.activation(self.projection(x))))


def metric_block(labels, probability):
    prediction = probability.argmax(axis=1)
    return {
        "kappa": float(cohen_kappa_score(labels, prediction)),
        "nll": float(log_loss(labels, probability, labels=np.arange(5))),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "weighted_f1": float(f1_score(labels, prediction, average="weighted")),
    }


@torch.inference_mode()
def predict(head, features, device, batch_size=32):
    output = []
    head.eval()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).to(device)
        output.append(torch.softmax(head(batch), dim=-1).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(output).astype(np.float32))


def train_head(train_x, train_y, val_x, val_y, test_x, seed, device):
    configure_determinism(seed)
    head = SequenceHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = None
    best_epoch = None
    best_kappa = -np.inf
    history = []
    for epoch in range(1, 51):
        head.train()
        order = torch.randperm(len(train_x), generator=generator).numpy()
        losses = []
        for start in range(0, len(order), 16):
            indices = order[start:start + 16]
            x = torch.from_numpy(train_x[indices]).to(device)
            y = torch.from_numpy(train_y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = loss_fn(logits.reshape(-1, 5), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite ISRUC sequence-head loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probability = predict(head, val_x, device)
        val_metric = metric_block(val_y.reshape(-1), val_probability.reshape(-1, 5))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_kappa": val_metric["kappa"],
            "val_nll": val_metric["nll"],
        })
        if val_metric["kappa"] > best_kappa + 1e-12:
            best_kappa = val_metric["kappa"]
            best_epoch = epoch
            best = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if best is None:
        raise RuntimeError("ISRUC sequence head did not select an epoch")
    head.load_state_dict(best, strict=True)
    val_probability = predict(head, val_x, device)
    test_probability = predict(head, test_x, device)
    return head, val_probability, test_probability, best_epoch, best_kappa, history


def fit_logistic(x, y):
    pca = PCA(n_components=PCA_DIM, whiten=True, svd_solver="randomized", random_state=0)
    z = pca.fit_transform(x)
    clf = LogisticRegression(C=PROBE_C, solver="lbfgs", max_iter=2000, tol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(z, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("ISRUC subject probe did not converge")
    return pca, clf


def pairwise(z_fit, subject_fit, z_hold, subject_hold):
    values = []
    subjects = sorted(int(value) for value in np.unique(subject_fit))
    centers = {value: z_fit[subject_fit == value].mean(axis=0) for value in subjects}
    for left_index, left in enumerate(subjects):
        for right in subjects[left_index + 1:]:
            mask = np.isin(subject_hold, [left, right])
            current = z_hold[mask]
            truth = subject_hold[mask]
            score = (
                np.sum((current - centers[right]) ** 2, axis=1)
                - np.sum((current - centers[left]) ** 2, axis=1)
            )
            binary = (truth == left).astype(int)
            correct = np.where(binary == 1, score, -score)
            values.append((
                float(roc_auc_score(binary, score)),
                float(np.mean(correct) / (np.std(score) + 1e-12)),
            ))
    return values


def subject_metrics(features, subjects, source_subjects):
    sequence_mean = features.mean(axis=1)
    source_mask = np.isin(subjects, source_subjects)
    current_x = sequence_mean[source_mask]
    current_subject = subjects[source_mask]
    current_index = np.arange(len(subjects))[source_mask]
    losses = []
    accuracy = []
    pairs = []
    for half in (0, 1):
        fit_indices = []
        hold_indices = []
        for subject in source_subjects:
            local = current_index[current_subject == subject]
            midpoint = len(local) // 2
            first, second = local[:midpoint], local[midpoint:]
            fit_global, hold_global = (first, second) if half == 0 else (second, first)
            fit_indices.extend(np.where(np.isin(current_index, fit_global))[0])
            hold_indices.extend(np.where(np.isin(current_index, hold_global))[0])
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        hold_indices = np.asarray(hold_indices, dtype=np.int64)
        pca, clf = fit_logistic(current_x[fit_indices], current_subject[fit_indices])
        z_fit = pca.transform(current_x[fit_indices])
        z_hold = pca.transform(current_x[hold_indices])
        probability = clf.predict_proba(z_hold)
        prediction = clf.predict(z_hold)
        class_to_index = {int(value): i for i, value in enumerate(clf.classes_)}
        true_index = np.asarray([class_to_index[int(value)] for value in current_subject[hold_indices]])
        losses.extend(-np.log(np.clip(probability[np.arange(len(hold_indices)), true_index], 1e-15, 1.0)))
        accuracy.extend((prediction == current_subject[hold_indices]).astype(float))
        pairs.extend(pairwise(
            z_fit, current_subject[fit_indices], z_hold, current_subject[hold_indices]
        ))
    return {
        "subject_nll": float(np.mean(losses)),
        "subject_accuracy_diagnostic": float(np.mean(accuracy)),
        "pairwise_subject_auc": float(np.mean([value[0] for value in pairs])),
        "pairwise_subject_margin": float(np.mean([value[1] for value in pairs])),
    }


def effects(z, labels, subjects, effect):
    subject_values = sorted(int(value) for value in np.unique(subjects))
    cells = np.empty((len(subject_values), 5, z.shape[1]), dtype=np.float64)
    for subject_index, subject in enumerate(subject_values):
        for label in range(5):
            mask = (subjects == subject) & (labels == label)
            if not mask.any():
                raise RuntimeError(f"missing ISRUC subject-class cell subject={subject} class={label}")
            cells[subject_index, label] = z[mask].mean(axis=0)
    grand = cells.mean(axis=(0, 1))
    return cells.mean(axis=1) - grand if effect == "subject" else cells.mean(axis=0) - grand


def basis(matrix):
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if len(singular) < RANK or singular[RANK - 1] <= 1e-12:
        raise RuntimeError("ISRUC rank-4 geometry is degenerate")
    return vt[:RANK]


def capture(effect, directions):
    return float(np.sum((effect @ directions.T) ** 2) / (np.sum(effect ** 2) + 1e-12))


def geometry(features, labels, subjects, source_subjects):
    rows = []
    for half in (0, 1):
        fit_x = []
        fit_y = []
        fit_s = []
        hold_x = []
        hold_y = []
        hold_s = []
        for subject in source_subjects:
            indices = np.flatnonzero(subjects == subject)
            midpoint = len(indices) // 2
            first, second = indices[:midpoint], indices[midpoint:]
            fit_index, hold_index = (first, second) if half == 0 else (second, first)
            for target_x, target_y, target_s, selected in (
                (fit_x, fit_y, fit_s, fit_index),
                (hold_x, hold_y, hold_s, hold_index),
            ):
                target_x.append(features[selected].reshape(-1, features.shape[-1]))
                target_y.append(labels[selected].reshape(-1))
                target_s.append(np.repeat(subject, len(selected) * 20))
        fit_x = np.concatenate(fit_x)
        fit_y = np.concatenate(fit_y)
        fit_s = np.concatenate(fit_s)
        hold_x = np.concatenate(hold_x)
        hold_y = np.concatenate(hold_y)
        hold_s = np.concatenate(hold_s)
        pca = PCA(n_components=PCA_DIM, whiten=True, svd_solver="randomized", random_state=0)
        z_fit = pca.fit_transform(fit_x)
        z_hold = pca.transform(hold_x)
        subject_fit = effects(z_fit, fit_y, fit_s, "subject")
        task_fit = effects(z_fit, fit_y, fit_s, "task")
        subject_hold = effects(z_hold, hold_y, hold_s, "subject")
        task_hold = effects(z_hold, hold_y, hold_s, "task")
        subject_basis = basis(subject_fit)
        task_basis = basis(task_fit)
        rows.append({
            "half": half,
            "projection_overlap_rank4": float(np.sum((subject_basis @ task_basis.T) ** 2) / RANK),
            "subject_heldout_captured_energy": capture(subject_hold, subject_basis),
            "task_heldout_captured_energy": capture(task_hold, task_basis),
        })
    return {
        "projection_overlap_rank4": float(np.mean([row["projection_overlap_rank4"] for row in rows])),
        "subject_heldout_captured_energy": float(np.mean([row["subject_heldout_captured_energy"] for row in rows])),
        "task_heldout_captured_energy": float(np.mean([row["task_heldout_captured_energy"] for row in rows])),
        "subspace_stable": all(
            row["subject_heldout_captured_energy"] >= 0.05
            and row["task_heldout_captured_energy"] >= 0.05
            for row in rows
        ),
    }


def load_feature_cache(feature_dir, tag, manifest_value):
    contract = json.loads((feature_dir / f"{tag}_feature_contract.json").read_text())
    payload_path = feature_dir / f"{tag}_features.npz"
    if (
        contract.get("status") != "PASS"
        or contract.get("dataset") != "ISRUC_S3_Group_III"
        or contract.get("tag") != tag
        or contract.get("checkpoint_sha256_before") != manifest_value["immutable_sha256"]
        or contract.get("checkpoint_sha256_after") != manifest_value["immutable_sha256"]
        or contract.get("canary_repeat_max_abs_diff") != 0.0
        or contract.get("fine_tuning_used") is not False
        or sha256_file(payload_path) != contract.get("payload_sha256")
    ):
        raise RuntimeError(f"ISRUC feature cache contract failed for {tag}")
    with np.load(payload_path) as payload:
        data = {name: payload[name] for name in payload.files}
    return data, contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", choices=TAGS, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rows = validate_manifest(args.checkpoint_manifest)
    data, feature_contract = load_feature_cache(
        args.feature_dir, args.tag, manifest_row(rows, args.tag)
    )
    features = data["features"]
    labels = data["labels"]
    subjects = data["subjects"]
    sequence_ids = data["sequence_ids"]
    if features.shape != (425, 20, 1200) or labels.shape != (425, 20):
        raise RuntimeError(f"ISRUC feature payload shape mismatch: {features.shape}")
    if canonical_sha(list(zip(subjects.tolist(), sequence_ids.tolist()))) != feature_contract["subject_sequence_sha256"]:
        raise RuntimeError("ISRUC subject/sequence identity hash mismatch")
    rotations = read_csv(args.rotation_manifest)
    if len(rotations) != 10:
        raise RuntimeError("ISRUC requires ten rotating folds")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ISRUC sequence-head audit requires CUDA")

    test_probabilities = []
    test_labels = []
    test_subjects = []
    test_rotations = []
    val_probabilities = []
    val_labels = []
    val_subjects = []
    val_rotations = []
    head_rows = []
    subject_rows = []
    geometry_rows = []
    for rotation in rotations:
        fold = int(rotation["fold"])
        source = [int(value) for value in rotation["train_subjects"].split(";")]
        val_subject = int(rotation["val_subject"])
        test_subject = int(rotation["test_subject"])
        if set(source) & {val_subject, test_subject} or val_subject == test_subject:
            raise RuntimeError(f"ISRUC subject leakage in rotation {fold}")
        train = np.isin(subjects, source)
        val = subjects == val_subject
        test = subjects == test_subject
        seed_val = []
        seed_test = []
        for seed in HEAD_SEEDS:
            _, val_probability, test_probability, best_epoch, best_kappa, history = train_head(
                features[train], labels[train], features[val], labels[val],
                features[test], seed, device
            )
            seed_val.append(val_probability)
            seed_test.append(test_probability)
            selected = history[best_epoch - 1]
            head_rows.append({
                "tag": args.tag,
                "rotation": fold,
                "source_subjects": ";".join(str(value) for value in source),
                "val_subject": val_subject,
                "test_subject": test_subject,
                "downstream_seed": seed,
                "epochs_run": 50,
                "selected_epoch": best_epoch,
                "selection_metric": "source_val_kappa_only",
                "selected_val_kappa": best_kappa,
                "selected_val_nll": selected["val_nll"],
                "encoder_frozen": True,
                "encoder_optimizer_created": False,
            })
        test_probabilities.append(np.stack(seed_test))
        val_probabilities.append(np.stack(seed_val))
        test_labels.append(labels[test])
        val_labels.append(labels[val])
        test_subjects.append(np.full(test.sum() * 20, test_subject, dtype=np.int64))
        val_subjects.append(np.full(val.sum() * 20, val_subject, dtype=np.int64))
        test_rotations.append(np.full(test.sum() * 20, fold, dtype=np.int64))
        val_rotations.append(np.full(val.sum() * 20, fold, dtype=np.int64))
        subject_rows.append({"tag": args.tag, "rotation": fold, **subject_metrics(features, subjects, source)})
        geometry_rows.append({"tag": args.tag, "rotation": fold, **geometry(features, labels, subjects, source)})
        print(f"ISRUC tag={args.tag} rotation={fold}/9 complete", flush=True)

    # Concatenate variable subject sequence counts along the epoch dimension.
    test_probability = np.concatenate(
        [value.reshape(len(HEAD_SEEDS), -1, 5) for value in test_probabilities], axis=1
    )
    val_probability = np.concatenate(
        [value.reshape(len(HEAD_SEEDS), -1, 5) for value in val_probabilities], axis=1
    )
    y_test = np.concatenate([value.reshape(-1) for value in test_labels])
    y_val = np.concatenate([value.reshape(-1) for value in val_labels])
    test_subject = np.concatenate(test_subjects)
    val_subject = np.concatenate(val_subjects)
    test_rotation = np.concatenate(test_rotations)
    val_rotation = np.concatenate(val_rotations)
    if len(y_test) != 8500 or len(y_val) != 8500:
        raise RuntimeError("ISRUC rotating aggregate must score 8,500 test and validation epochs")
    averaged_test = test_probability.mean(axis=0)
    averaged_val = val_probability.mean(axis=0)
    test_metric = metric_block(y_test, averaged_test)
    val_metric = metric_block(y_val, averaged_val)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out_dir / f"{args.tag}_isruc_predictions.npz"
    if prediction_path.exists():
        raise RuntimeError(f"refusing to overwrite ISRUC predictions: {prediction_path}")
    np.savez_compressed(
        prediction_path,
        test_probabilities=test_probability,
        test_labels=y_test,
        test_subjects=test_subject,
        test_rotations=test_rotation,
        val_probabilities=val_probability,
        val_labels=y_val,
        val_subjects=val_subject,
        val_rotations=val_rotation,
    )
    write_csv(args.out_dir / f"{args.tag}_isruc_head_manifest.csv", head_rows)
    write_csv(args.out_dir / f"{args.tag}_isruc_subject_metrics.csv", subject_rows)
    write_csv(args.out_dir / f"{args.tag}_isruc_geometry.csv", geometry_rows)
    contract = {
        "status": "PASS",
        "dataset": "ISRUC_S3_Group_III",
        "tag": args.tag,
        "feature_payload_sha256": feature_contract["payload_sha256"],
        "prediction_payload": f"objects/{prediction_path.name}",
        "prediction_payload_sha256": sha256_file(prediction_path),
        "rotations": 10,
        "downstream_seeds": list(HEAD_SEEDS),
        "head_architecture": "linear1200_512_gelu_transformer1_h4_ff2048_linear5",
        "selection": "source_val_kappa_only_earliest_tie",
        "source_val": val_metric,
        "target_test": test_metric,
        "all_geometry_subspaces_stable": all(row["subspace_stable"] for row in geometry_rows),
        "encoder_frozen": True,
        "encoder_optimizer_created": False,
        "fine_tuning_used": False,
        "target_test_labels_used_for_selection": False,
    }
    write_json(args.out_dir / f"{args.tag}_isruc_object_contract.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
