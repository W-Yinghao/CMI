#!/usr/bin/env python
"""S2P Phase B1 final-representation confirmatory decomposition.

The implementation is restricted to the immutable ten-object manifest and the
pre-registered final-layer subject/task/geometry/variance measurements.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import stat
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import lmdb
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, log_loss, roc_auc_score

sys.path.insert(0, os.path.expanduser("~/eeg2025/CBraMod"))
from models.cbramod import CBraMod


TAGS = [
    "random",
    "released",
    "H200_s0",
    "H200_s1",
    "H500_s0",
    "H500_s1",
    "H1000_s0",
    "H1000_s1",
    "H2000_s0",
    "H2000_s1",
]
BUDGETS = [200, 500, 1000, 2000]
HIGH_BUDGETS = [500, 1000, 2000]
KEY_RE = re.compile(r"sub(?P<subject>[0-9]+)[.]pkl-(?P<clip>[0-9]+)-(?P<segment>[0-9]+)$")
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
CLIP_TO_CLASS = {
    clip: class_id for class_id, clips in CLASS_TO_CLIPS.items() for clip in clips
}
CLIP_TO_FOLD = {
    clip: index % 3 for clips in CLASS_TO_CLIPS.values() for index, clip in enumerate(clips)
}
CHECKSUM_KEYS = [
    f"sub{subject:03d}.pkl-{clip}-{class_id % 3}"
    for subject in (0, 79)
    for class_id, clip in enumerate((0, 3, 6, 9, 12, 16, 19, 22, 25))
]
N_CLASSES = 9
N_SOURCE_SUBJECTS = 80
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260711
PCA_DIM = 128
PROBE_C = 1.0
SUBSPACE_RANK = 8


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def as_bool(value):
    return str(value).lower() == "true"


def budget_seed(tag):
    if not tag.startswith("H"):
        return None, None
    left, right = tag.split("_s")
    return int(left[1:]), int(right)


def make_model(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=10,
        n_layer=12,
        nhead=8,
    )


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint is not a dictionary")
    for key in ("model_state", "model", "state_dict", "model_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict) or not obj:
        raise RuntimeError("checkpoint does not contain a state dictionary")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in obj.items()}


def build_model(tag, immutable_path, device):
    if tag == "random":
        model = make_model(seed=0)
    else:
        obj = torch.load(immutable_path, map_location="cpu", weights_only=False)
        state = unwrap_state_dict(obj)
        model = make_model(seed=0)
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"{tag} strict state mismatch")
    return model.to(device).eval()


@torch.inference_mode()
def extract_features(model, data, device, batch_size, label):
    chunks = []
    for start in range(0, len(data), batch_size):
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"{label} feature start={start}/{len(data)}", flush=True)
        batch = torch.from_numpy(data[start:start + batch_size]).to(device)
        patch = model.patch_embedding(batch, None)
        encoded = model.encoder(patch)
        chunks.append(encoded.mean(2).reshape(encoded.shape[0], -1).float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(chunks).astype(np.float32))


@torch.inference_mode()
def checksum_feature(model, batch, device):
    tensor = torch.from_numpy(batch).to(device)
    patch = model.patch_embedding(tensor, None)
    encoded = model.encoder(patch)
    result = encoded.mean(2).reshape(encoded.shape[0], -1).float().cpu().numpy()
    return np.ascontiguousarray(result.astype(np.float32))


def load_faced(lmdb_path):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        keys_by_split = pickle.loads(txn.get(b"__keys__"))
        ordered = []
        for lmdb_split, protocol_split in (
            ("train", "source_train"),
            ("val", "source_val"),
            ("test", "target_test"),
        ):
            for key in keys_by_split[lmdb_split]:
                key_text = key.decode() if isinstance(key, bytes) else key
                ordered.append((key_text, protocol_split))
        n_items = len(ordered)
        data = np.empty((n_items, 32, 10, 200), dtype=np.float32)
        labels = np.empty(n_items, dtype=np.int64)
        subjects = np.empty(n_items, dtype=np.int64)
        clips = np.empty(n_items, dtype=np.int64)
        segments = np.empty(n_items, dtype=np.int64)
        splits = np.empty(n_items, dtype="U12")
        for index, (key, split_name) in enumerate(ordered):
            if index % 1000 == 0:
                print(f"FACED load={index}/{n_items}", flush=True)
            match = KEY_RE.fullmatch(key)
            if match is None:
                raise RuntimeError(f"unparseable FACED key: {key}")
            subject = int(match.group("subject")) + 1
            clip = int(match.group("clip"))
            segment = int(match.group("segment"))
            obj = pickle.loads(txn.get(key.encode()))
            sample = np.asarray(obj["sample"], dtype=np.float32)
            label = int(obj["label"])
            if sample.shape != (32, 10, 200):
                raise RuntimeError(f"FACED sample shape mismatch: {key} {sample.shape}")
            if CLIP_TO_CLASS[clip] != label:
                raise RuntimeError(f"FACED clip/class mismatch: {key} label={label}")
            data[index] = sample
            labels[index] = label
            subjects[index] = subject
            clips[index] = clip
            segments[index] = segment
            splits[index] = split_name
    env.close()
    data -= data.mean(-1, keepdims=True)
    data /= data.std(-1, keepdims=True) + 1e-6
    fold = np.asarray([CLIP_TO_FOLD[int(clip)] for clip in clips], dtype=np.int64)
    return data, labels, subjects, clips, segments, splits, fold, [key for key, _ in ordered]


def load_checksum_from_full(data, keys):
    lookup = {key: index for index, key in enumerate(keys)}
    missing = [key for key in CHECKSUM_KEYS if key not in lookup]
    if missing:
        raise RuntimeError(f"checksum keys missing from FACED load: {missing}")
    return np.ascontiguousarray(np.stack([data[lookup[key]] for key in CHECKSUM_KEYS]))


def validate_manifest(path):
    rows = read_csv(path)
    if len(rows) != 10 or [row["tag"] for row in rows] != TAGS:
        raise RuntimeError("B1 requires the ordered ten-object immutable manifest")
    for row in rows:
        if not as_bool(row["strict_reload_pass"]):
            raise RuntimeError(f"strict reload not closed: {row['tag']}")
        if not as_bool(row["parameter_exact_pass"]):
            raise RuntimeError(f"parameter exactness not closed: {row['tag']}")
        if not as_bool(row["feature_equivalence_pass"]):
            raise RuntimeError(f"feature equivalence not closed: {row['tag']}")
        if float(row["feature_max_abs_diff"]) != 0.0:
            raise RuntimeError(f"nonzero closure feature difference: {row['tag']}")
        if row["tag"] == "random":
            if not row["immutable_path"].startswith("random_init_contract://sha256_"):
                raise RuntimeError("random immutable contract mismatch")
            continue
        immutable = Path(row["immutable_path"])
        if not immutable.is_file() or immutable.is_symlink():
            raise RuntimeError(f"B1 checkpoint is not a direct payload: {row['tag']}")
        if immutable.resolve() == Path(row["source_path"]).resolve():
            raise RuntimeError(f"B1 checkpoint resolves to mutable source: {row['tag']}")
        if stat.S_IMODE(immutable.stat().st_mode) & 0o222:
            raise RuntimeError(f"B1 checkpoint is writable: {row['tag']}")
        digest = sha256_file(immutable)
        if digest != row["immutable_sha256"] or digest not in immutable.name:
            raise RuntimeError(f"B1 checkpoint content address mismatch: {row['tag']}")
    return rows


def fit_logistic(features, labels):
    model = LogisticRegression(
        C=PROBE_C,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-6,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f"fixed logistic probe did not converge: {convergence[0].message}")
    return model


def true_probabilities(model, features, labels):
    probability = model.predict_proba(features)
    class_to_index = {int(value): index for index, value in enumerate(model.classes_)}
    indices = np.asarray([class_to_index[int(value)] for value in labels], dtype=np.int64)
    return np.clip(probability[np.arange(len(labels)), indices], 1e-15, 1.0), probability


def retrieval_average_precision(z_fit, subject_fit, z_hold, subject_hold, chunk_size=128):
    gallery = z_fit / (np.linalg.norm(z_fit, axis=1, keepdims=True) + 1e-12)
    query = z_hold / (np.linalg.norm(z_hold, axis=1, keepdims=True) + 1e-12)
    values = []
    for start in range(0, len(query), chunk_size):
        similarity = query[start:start + chunk_size] @ gallery.T
        order = np.argsort(-similarity, axis=1)
        for local, ranked in enumerate(order):
            subject = subject_hold[start + local]
            relevant = (subject_fit[ranked] == subject).astype(np.float64)
            precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
            values.append(float(np.sum(precision * relevant) / max(1.0, relevant.sum())))
    return np.asarray(values, dtype=np.float64)


def pairwise_fold_metrics(z_fit, subject_fit, z_hold, subject_hold):
    subjects = sorted(int(value) for value in np.unique(subject_fit))
    centroids = {
        subject: z_fit[subject_fit == subject].mean(axis=0) for subject in subjects
    }
    rows = []
    for left_index, left in enumerate(subjects):
        for right in subjects[left_index + 1:]:
            mask = np.isin(subject_hold, [left, right])
            current = z_hold[mask]
            truth = subject_hold[mask]
            left_distance = np.sum((current - centroids[left]) ** 2, axis=1)
            right_distance = np.sum((current - centroids[right]) ** 2, axis=1)
            score = right_distance - left_distance
            binary = (truth == left).astype(int)
            auc = float(roc_auc_score(binary, score))
            correct_margin = np.where(binary == 1, score, -score)
            scale = float(np.std(score)) + 1e-12
            margin = float(np.mean(correct_margin) / scale)
            prediction = np.where(score >= 0, left, right)
            bacc = float(balanced_accuracy_score(truth, prediction))
            rows.append((left, right, auc, margin, bacc, int(mask.sum())))
    return rows


def cell_means(features, labels, subjects, mask):
    result = np.empty((N_SOURCE_SUBJECTS, N_CLASSES, features.shape[1]), dtype=np.float32)
    second_moment = np.empty((N_SOURCE_SUBJECTS, N_CLASSES), dtype=np.float64)
    for subject_index, subject in enumerate(range(1, N_SOURCE_SUBJECTS + 1)):
        for class_id in range(N_CLASSES):
            cell = mask & (subjects == subject) & (labels == class_id)
            if not cell.any():
                raise RuntimeError(f"empty source subject/class cell: {subject}/{class_id}")
            selected = features[cell]
            result[subject_index, class_id] = selected.mean(axis=0)
            second_moment[subject_index, class_id] = float(np.mean(np.sum(selected ** 2, axis=1)))
    return result, second_moment


def effect_geometry(cell, weights=None):
    n_subjects = cell.shape[0]
    if weights is None:
        weights = np.full(n_subjects, 1.0 / n_subjects)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    subject_mean = cell.mean(axis=1).astype(np.float64)
    class_mean = np.einsum("s,skd->kd", weights, cell.astype(np.float64))
    grand = np.einsum("s,sd->d", weights, subject_mean)
    subject_effect = subject_mean - grand
    task_effect = class_mean - grand
    weighted_subject = np.sqrt(weights)[:, None] * subject_effect
    _, subject_singular, subject_vt = np.linalg.svd(weighted_subject, full_matrices=False)
    _, task_singular, task_vt = np.linalg.svd(task_effect, full_matrices=False)
    if subject_singular[SUBSPACE_RANK - 1] <= max(1e-12, subject_singular[0] * 1e-10):
        raise RuntimeError("subject effect rank below 8")
    if task_singular[SUBSPACE_RANK - 1] <= max(1e-12, task_singular[0] * 1e-10):
        raise RuntimeError("task effect rank below 8")
    subject_basis = subject_vt[:SUBSPACE_RANK]
    task_basis = task_vt[:SUBSPACE_RANK]
    canonical = np.linalg.svd(subject_basis @ task_basis.T, compute_uv=False)
    overlap = float(np.mean(canonical ** 2))
    return {
        "subject_basis": subject_basis,
        "task_basis": task_basis,
        "canonical": canonical,
        "overlap": overlap,
        "subject_effect": subject_effect,
        "task_effect": task_effect,
    }


def captured_energy(effect, basis, subject_weights=None):
    projection = effect @ basis.T
    if subject_weights is None:
        numerator = float(np.sum(projection ** 2))
        denominator = float(np.sum(effect ** 2)) + 1e-12
    else:
        weights = np.asarray(subject_weights, dtype=np.float64)
        weights /= weights.sum()
        numerator = float(np.sum(weights[:, None] * projection ** 2))
        denominator = float(np.sum(weights[:, None] * effect ** 2)) + 1e-12
    return numerator / denominator


def variance_sufficient(fit_cell, hold_cell, hold_second):
    fit = fit_cell.astype(np.float64)
    hold = hold_cell.astype(np.float64)
    fit_subject = fit.mean(axis=1)
    hold_subject = hold.mean(axis=1)
    return {
        "grand_cross": fit_subject @ hold_subject.T,
        "subject_diag_cross": np.sum(fit_subject * hold_subject, axis=1),
        "between_diag_cross": np.mean(np.sum(fit * hold, axis=2), axis=1),
        "class_cross": np.stack([fit[:, class_id] @ hold[:, class_id].T for class_id in range(N_CLASSES)]),
        "hold_grand_square": hold_subject @ hold_subject.T,
        "hold_second_subject": hold_second.mean(axis=1),
    }


def variance_components(sufficient, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    weights = weights / weights.sum(axis=1, keepdims=True)

    def quadratic(matrix):
        return np.einsum("bi,ij,bj->b", weights, matrix, weights, optimize=True)

    grand_cross = quadratic(sufficient["grand_cross"])
    subject = weights @ sufficient["subject_diag_cross"] - grand_cross
    between = weights @ sufficient["between_diag_cross"] - grand_cross
    class_terms = np.stack(
        [quadratic(matrix) for matrix in sufficient["class_cross"]], axis=1
    )
    class_component = class_terms.mean(axis=1) - grand_cross
    interaction = between - subject - class_component
    hold_grand_square = quadratic(sufficient["hold_grand_square"])
    total = weights @ sufficient["hold_second_subject"] - hold_grand_square
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


def confusion_by_subject(labels, predictions, subjects, subject_values):
    result = np.zeros((len(subject_values), N_CLASSES, N_CLASSES), dtype=np.float64)
    for index, subject in enumerate(subject_values):
        mask = subjects == subject
        np.add.at(result[index], (labels[mask], predictions[mask]), 1)
    return result


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    total = confusion.sum(axis=(-2, -1))
    observed = np.diagonal(confusion, axis1=-2, axis2=-1).sum(-1) / np.maximum(total, 1)
    row = confusion.sum(axis=-1)
    column = confusion.sum(axis=-2)
    expected = (row * column).sum(-1) / np.maximum(total ** 2, 1)
    kappa = (observed - expected) / np.maximum(1 - expected, 1e-12)
    recall = np.diagonal(confusion, axis1=-2, axis2=-1) / np.maximum(row, 1)
    bacc = recall.mean(-1)
    return kappa, bacc


def summarize_samples(point, samples):
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_sd": float(np.std(samples)),
    }


def one_sided_positive_p(samples):
    return float((1 + np.sum(np.asarray(samples) <= 0)) / (len(samples) + 1))


def two_sided_sign_p(samples):
    samples = np.asarray(samples)
    lower = (1 + np.sum(samples <= 0)) / (len(samples) + 1)
    upper = (1 + np.sum(samples >= 0)) / (len(samples) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    n_values = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (n_values - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def budget_tags(budget):
    return [f"H{budget}_s0", f"H{budget}_s1"]


def mean_arrays(mapping, tags):
    return np.mean([mapping[tag] for tag in tags], axis=0)


def pool_high(mapping, budgets=HIGH_BUDGETS, seed=None):
    if seed is None:
        return np.mean([mean_arrays(mapping, budget_tags(budget)) for budget in budgets], axis=0)
    return np.mean([mapping[f"H{budget}_s{seed}"] for budget in budgets], axis=0)


def weighted_pair_bootstrap(pair_rows, weights, value_key):
    left = np.asarray([int(row["subject_left"]) - 1 for row in pair_rows], dtype=np.int64)
    right = np.asarray([int(row["subject_right"]) - 1 for row in pair_rows], dtype=np.int64)
    values = np.asarray([float(row[value_key]) for row in pair_rows], dtype=np.float64)
    result = np.empty(len(weights), dtype=np.float64)
    for start in range(0, len(weights), 250):
        current = weights[start:start + 250]
        pair_weight = current[:, left] * current[:, right]
        denominator = pair_weight.sum(axis=1)
        result[start:start + len(current)] = (pair_weight @ values) / np.maximum(denominator, 1e-12)
    return result


def run_self_tests():
    adjusted = holm_adjust([0.01, 0.03, 0.04])
    if not np.allclose(adjusted, [0.03, 0.06, 0.06]):
        raise RuntimeError(f"Holm self-test failed: {adjusted}")

    rng = np.random.default_rng(123)
    subject_score = rng.normal(size=(N_SOURCE_SUBJECTS, SUBSPACE_RANK))
    subject_score -= subject_score.mean(axis=0, keepdims=True)
    class_score = rng.normal(size=(N_CLASSES, SUBSPACE_RANK))
    class_score -= class_score.mean(axis=0, keepdims=True)
    orthogonal = np.zeros((N_SOURCE_SUBJECTS, N_CLASSES, 2 * SUBSPACE_RANK))
    orthogonal[:, :, :SUBSPACE_RANK] = subject_score[:, None, :]
    orthogonal[:, :, SUBSPACE_RANK:] = class_score[None, :, :]
    orthogonal_geometry = effect_geometry(orthogonal)
    if orthogonal_geometry["overlap"] > 1e-20:
        raise RuntimeError("orthogonal geometry self-test failed")
    aligned = subject_score[:, None, :] + class_score[None, :, :]
    aligned_geometry = effect_geometry(aligned)
    if not np.isclose(aligned_geometry["overlap"], 1.0, atol=1e-12):
        raise RuntimeError("aligned geometry self-test failed")

    second = np.sum(orthogonal ** 2, axis=2)
    sufficient = variance_sufficient(orthogonal, orthogonal, second)
    components = variance_components(
        sufficient, np.full(N_SOURCE_SUBJECTS, 1 / N_SOURCE_SUBJECTS)
    )
    if abs(float(components["interaction"][0])) > 1e-10:
        raise RuntimeError("additive variance interaction self-test failed")
    if abs(float(components["residual"][0])) > 1e-10:
        raise RuntimeError("zero-residual variance self-test failed")
    fractions = sum(float(components[key][0]) for key in (
        "subject_frac", "class_frac", "interaction_frac", "residual_frac"
    ))
    if not np.isclose(fractions, 1.0, atol=1e-12):
        raise RuntimeError("variance fraction self-test failed")

    labels = np.repeat(np.arange(N_CLASSES), 2)
    predictions = labels.copy()
    predictions[::4] = (predictions[::4] + 1) % N_CLASSES
    subjects = np.tile([1, 2, 3], 6)
    confusion = confusion_by_subject(labels, predictions, subjects, [1, 2, 3])
    kappa, bacc = metrics_from_confusion(confusion.sum(axis=0, keepdims=True))
    if not np.isclose(kappa[0], cohen_kappa_score(labels, predictions)):
        raise RuntimeError("Kappa self-test failed")
    if not np.isclose(bacc[0], balanced_accuracy_score(labels, predictions)):
        raise RuntimeError("balanced-accuracy self-test failed")

    for class_id, clips in CLASS_TO_CLIPS.items():
        if {CLIP_TO_FOLD[clip] for clip in clips} != {0, 1, 2}:
            raise RuntimeError(f"clip-fold self-test failed for class {class_id}")
    print("Phase B1 synthetic self-tests: PASS", flush=True)


def run(args):
    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = validate_manifest(args.checkpoint_manifest)
    manifest = {row["tag"]: row for row in manifest_rows}
    closure = json.loads(Path(args.closure_json).read_text())
    b0 = json.loads(Path(args.b0_go_nogo).read_text())
    redteam = json.loads(Path(args.provenance_redteam).read_text())
    if closure.get("status") != "PASS" or redteam.get("status") != "PASS":
        raise RuntimeError("B1 provenance authority is not PASS")
    if b0.get("phase_b1_go_recommended") is not True:
        raise RuntimeError("B0 does not recommend B1")
    if b0.get("phase_b1_compute_authorized") is not False:
        raise RuntimeError("B1 launcher expects the pre-launch authorization flag to remain false in B0")
    if not args.pm_authorized:
        raise RuntimeError("B1 requires the explicit post-closure PM authorization flag")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B1 requires the A100 CUDA path used by provenance feature closure")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data, labels, subjects, clips, segments, splits, folds, keys = load_faced(args.faced_lmdb)
    checksum_batch = load_checksum_from_full(data, keys)
    source_mask = splits == "source_train"
    val_mask = splits == "source_val"
    target_mask = splits == "target_test"
    if sorted(np.unique(subjects[source_mask]).tolist()) != list(range(1, 81)):
        raise RuntimeError("source subject split mismatch")
    if sorted(np.unique(subjects[target_mask]).tolist()) != list(range(101, 124)):
        raise RuntimeError("target subject split mismatch")
    for fold in range(3):
        fit_clips = {clip for clip, assigned in CLIP_TO_FOLD.items() if assigned != fold}
        hold_clips = {clip for clip, assigned in CLIP_TO_FOLD.items() if assigned == fold}
        if fit_clips & hold_clips or fit_clips | hold_clips != set(range(28)):
            raise RuntimeError(f"clip fold contract failed: {fold}")
        for clip in range(28):
            current = source_mask & (clips == clip)
            if set(np.unique(folds[current]).tolist()) != {CLIP_TO_FOLD[clip]}:
                raise RuntimeError(f"clip segments split across folds: {clip}")

    run_manifest = {
        "phase": "B1_final_representation_decomposition",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_manifest": str(Path(args.checkpoint_manifest).resolve()),
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
        "protocol_sha256": sha256_file(args.protocol_doc),
        "redteam_sha256": sha256_file(args.redteam_doc),
        "checkpoint_objects": TAGS,
        "feature_layer": "final_encoder_patch_mean_flatten_32x200",
        "subject_probe": "PCA128_whiten_C1_lbfgs_clip_group_3fold",
        "task_probe": "PCA128_whiten_C1_lbfgs_source_train_fit",
        "geometry": "fold_fit_PCA128_whiten_equal_rank8_projection_overlap",
        "variance": "raw6400_cross_fitted_equal_cell_trace",
        "bootstrap_reps": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_family": [
            "random_minus_H200_subject_nll",
            "H200_minus_pooled_high_target_nll",
            "pooled_high_minus_H200_projection_overlap",
        ],
        "holm_family_size": 3,
        "target_labels_used_for_selection": False,
        "new_pretraining": False,
        "fine_tuning": False,
        "h4000": False,
        "codebrain": False,
        "layerwise": False,
        "pm_b1_authorized": True,
    }
    write_json(out / "phase_b1_protocol.json", run_manifest)

    subject_metric_rows = []
    subject_cluster_rows = []
    subject_pair_rows = []
    task_metric_rows = []
    task_sample_rows = []
    geometry_rows = []
    variance_rows = []
    feature_rows = []
    hash_rows = []
    subject_clusters = {}
    pairs_by_tag = {}
    task_target_clusters = {}
    task_confusions = {}
    geometry_cells = {}
    geometry_hold_cells = {}
    variance_suff = {}

    for tag in TAGS:
        print(f"B1 object={tag}", flush=True)
        row = manifest[tag]
        immutable_path = row["immutable_path"]
        physical_sha_start = "logical_contract"
        if tag != "random":
            physical_sha_start = sha256_file(immutable_path)
            if physical_sha_start != row["immutable_sha256"]:
                raise RuntimeError(f"{tag} start SHA mismatch")
        model = build_model(tag, immutable_path, device)
        checksum_start = checksum_feature(model, checksum_batch, device)
        checksum_start_sha = hashlib.sha256(checksum_start.tobytes()).hexdigest()
        if checksum_start_sha != row["feature_hash"]:
            raise RuntimeError(f"{tag} checksum feature no longer matches closure")
        checksum_repeat = checksum_feature(model, checksum_batch, device)
        if float(np.max(np.abs(checksum_start - checksum_repeat))) != 0.0:
            raise RuntimeError(f"{tag} checksum feature is nondeterministic")
        features = extract_features(model, data, device, args.batch_size, tag)
        full_feature_sha = hashlib.sha256(features.tobytes()).hexdigest()

        subject_nll = defaultdict(list)
        subject_probability = defaultdict(list)
        subject_correct = defaultdict(list)
        subject_retrieval = defaultdict(list)
        subject_class_nll = defaultdict(lambda: defaultdict(list))
        pair_accumulator = defaultdict(lambda: defaultdict(float))
        geometry_cells[tag] = {}
        geometry_hold_cells[tag] = {}
        variance_suff[tag] = {}

        for fold in range(3):
            fit_mask = source_mask & (folds != fold)
            hold_mask = source_mask & (folds == fold)
            if set(clips[fit_mask]) & set(clips[hold_mask]):
                raise RuntimeError(f"{tag} fold {fold} clip leakage")
            pca = PCA(
                n_components=PCA_DIM,
                whiten=True,
                svd_solver="randomized",
                random_state=0,
            )
            z_fit = pca.fit_transform(features[fit_mask]).astype(np.float32)
            z_hold = pca.transform(features[hold_mask]).astype(np.float32)
            subject_fit = subjects[fit_mask]
            subject_hold = subjects[hold_mask]
            label_fit = labels[fit_mask]
            label_hold = labels[hold_mask]
            clip_hold = clips[hold_mask]
            segment_hold = segments[hold_mask]

            subject_probe = fit_logistic(z_fit, subject_fit)
            true_prob, _ = true_probabilities(subject_probe, z_hold, subject_hold)
            prediction = subject_probe.predict(z_hold)
            nll = -np.log(true_prob)
            retrieval = retrieval_average_precision(z_fit, subject_fit, z_hold, subject_hold)
            for index, subject in enumerate(subject_hold):
                subject = int(subject)
                subject_nll[subject].append(float(nll[index]))
                subject_probability[subject].append(float(true_prob[index]))
                subject_correct[subject].append(float(prediction[index] == subject))
                subject_retrieval[subject].append(float(retrieval[index]))

            for class_id in range(N_CLASSES):
                class_fit = label_fit == class_id
                class_hold = label_hold == class_id
                class_probe = fit_logistic(z_fit[class_fit], subject_fit[class_fit])
                class_prob, _ = true_probabilities(
                    class_probe, z_hold[class_hold], subject_hold[class_hold]
                )
                for subject, value in zip(subject_hold[class_hold], -np.log(class_prob)):
                    subject_class_nll[int(subject)][class_id].append(float(value))

            for left, right, auc, margin, bacc, n_items in pairwise_fold_metrics(
                z_fit, subject_fit, z_hold, subject_hold
            ):
                key = (left, right)
                pair_accumulator[key]["auc"] += auc * n_items
                pair_accumulator[key]["margin"] += margin * n_items
                pair_accumulator[key]["bacc"] += bacc * n_items
                pair_accumulator[key]["weight"] += n_items

            fit_cell_probe, _ = cell_means(z_fit, label_fit, subject_fit, np.ones(len(z_fit), dtype=bool))
            hold_cell_probe, _ = cell_means(z_hold, label_hold, subject_hold, np.ones(len(z_hold), dtype=bool))
            point_geometry = effect_geometry(fit_cell_probe)
            hold_geometry = effect_geometry(hold_cell_probe)
            subject_capture = captured_energy(
                hold_geometry["subject_effect"], point_geometry["subject_basis"]
            )
            task_capture = captured_energy(
                hold_geometry["task_effect"], point_geometry["task_basis"]
            )
            subject_on_task = captured_energy(
                hold_geometry["subject_effect"], point_geometry["task_basis"]
            )
            task_on_subject = captured_energy(
                hold_geometry["task_effect"], point_geometry["subject_basis"]
            )
            if subject_capture < 0.05 or task_capture < 0.05:
                raise RuntimeError(
                    f"{tag} fold {fold} subject/task subspace failed held-out capture gate: "
                    f"subject={subject_capture:.6g} task={task_capture:.6g}"
                )
            angles = np.degrees(np.arccos(np.clip(point_geometry["canonical"], -1, 1)))
            geometry_cells[tag][fold] = fit_cell_probe
            geometry_hold_cells[tag][fold] = hold_cell_probe
            geometry_rows.append({
                "scope": "fold",
                "tag": tag,
                "budget_h": budget_seed(tag)[0] or "",
                "seed": budget_seed(tag)[1] if budget_seed(tag)[1] is not None else "",
                "fold": fold,
                "pca_dim": PCA_DIM,
                "rank": SUBSPACE_RANK,
                "projection_overlap": point_geometry["overlap"],
                "canonical_correlations": ";".join(f"{value:.12g}" for value in point_geometry["canonical"]),
                "principal_angles_deg": ";".join(f"{value:.12g}" for value in angles),
                "max_canonical_correlation": float(point_geometry["canonical"].max()),
                "median_principal_angle_deg": float(np.median(angles)),
                "heldout_subject_self_capture": subject_capture,
                "heldout_task_self_capture": task_capture,
                "heldout_subject_on_task_capture": subject_on_task,
                "heldout_task_on_subject_capture": task_on_subject,
                "subspace_stable": subject_capture >= 0.05 and task_capture >= 0.05,
            })

            fit_cell_raw, _ = cell_means(features, labels, subjects, fit_mask)
            hold_cell_raw, hold_second_raw = cell_means(features, labels, subjects, hold_mask)
            sufficient = variance_sufficient(fit_cell_raw, hold_cell_raw, hold_second_raw)
            variance_suff[tag][fold] = sufficient
            point_variance = variance_components(
                sufficient, np.full(N_SOURCE_SUBJECTS, 1 / N_SOURCE_SUBJECTS)
            )
            variance_rows.append({
                "scope": "fold",
                "tag": tag,
                "budget_h": budget_seed(tag)[0] or "",
                "seed": budget_seed(tag)[1] if budget_seed(tag)[1] is not None else "",
                "fold": fold,
                **{key: float(value[0]) for key, value in point_variance.items()},
            })

            hold_indices = np.where(hold_mask)[0]
            for local, global_index in enumerate(hold_indices):
                task_sample_rows.append({
                    "record_type": "subject_probe_support",
                    "tag": tag,
                    "split": "source_train_clip_holdout",
                    "subject": int(subjects[global_index]),
                    "clip_id": int(clip_hold[local]),
                    "segment_id": int(segment_hold[local]),
                    "fold": fold,
                    "label_final_scoring_only": "",
                    "prediction": int(prediction[local]),
                    "nll": float(nll[local]),
                    "margin": "",
                })

        pair_rows_current = []
        for (left, right), values in sorted(pair_accumulator.items()):
            weight = values["weight"]
            pair_row = {
                "tag": tag,
                "subject_left": left,
                "subject_right": right,
                "pairwise_auc": values["auc"] / weight,
                "pairwise_standardized_margin": values["margin"] / weight,
                "pairwise_bacc_diagnostic": values["bacc"] / weight,
                "heldout_items": int(weight),
            }
            pair_rows_current.append(pair_row)
            subject_pair_rows.append(pair_row)
        pairs_by_tag[tag] = pair_rows_current

        cluster_current = []
        for subject in range(1, N_SOURCE_SUBJECTS + 1):
            class_values = [
                float(np.mean(subject_class_nll[subject][class_id]))
                for class_id in range(N_CLASSES)
            ]
            cluster_row = {
                "tag": tag,
                "subject": subject,
                "heldout_clip_subject_nll": float(np.mean(subject_nll[subject])),
                "heldout_clip_subject_accuracy_diagnostic": float(np.mean(subject_correct[subject])),
                "heldout_clip_true_subject_probability": float(np.mean(subject_probability[subject])),
                "true_subject_probability_gt_0p999999_frac": float(
                    np.mean(np.asarray(subject_probability[subject]) > 0.999999)
                ),
                "retrieval_map": float(np.mean(subject_retrieval[subject])),
                "class_conditional_subject_nll_macro": float(np.mean(class_values)),
            }
            cluster_current.append(cluster_row)
            subject_cluster_rows.append(cluster_row)
        subject_clusters[tag] = cluster_current
        metric_row = {
            "scope": "checkpoint",
            "tag": tag,
            "budget_h": budget_seed(tag)[0] or "",
            "seed": budget_seed(tag)[1] if budget_seed(tag)[1] is not None else "",
            "heldout_clip_subject_nll": float(np.mean([row["heldout_clip_subject_nll"] for row in cluster_current])),
            "heldout_clip_subject_accuracy_diagnostic": float(np.mean([row["heldout_clip_subject_accuracy_diagnostic"] for row in cluster_current])),
            "heldout_clip_true_subject_probability": float(np.mean([row["heldout_clip_true_subject_probability"] for row in cluster_current])),
            "true_subject_probability_gt_0p999999_frac": float(np.mean([row["true_subject_probability_gt_0p999999_frac"] for row in cluster_current])),
            "pairwise_subject_auc": float(np.mean([row["pairwise_auc"] for row in pair_rows_current])),
            "pairwise_subject_margin": float(np.mean([row["pairwise_standardized_margin"] for row in pair_rows_current])),
            "pairwise_subject_bacc_diagnostic": float(np.mean([row["pairwise_bacc_diagnostic"] for row in pair_rows_current])),
            "retrieval_map": float(np.mean([row["retrieval_map"] for row in cluster_current])),
            "class_conditional_subject_nll": float(np.mean([row["class_conditional_subject_nll_macro"] for row in cluster_current])),
        }
        subject_metric_rows.append(metric_row)

        task_pca = PCA(
            n_components=PCA_DIM,
            whiten=True,
            svd_solver="randomized",
            random_state=0,
        )
        z_train = task_pca.fit_transform(features[source_mask]).astype(np.float32)
        task_probe = fit_logistic(z_train, labels[source_mask])
        task_target_clusters[tag] = {}
        task_confusions[tag] = {}
        for split_name, mask in (("source_val", val_mask), ("target_test", target_mask)):
            z_score = task_pca.transform(features[mask]).astype(np.float32)
            truth = labels[mask]
            probability_true, _ = true_probabilities(task_probe, z_score, truth)
            prediction = task_probe.predict(z_score)
            decision = task_probe.decision_function(z_score)
            class_to_index = {int(value): index for index, value in enumerate(task_probe.classes_)}
            true_index = np.asarray([class_to_index[int(value)] for value in truth], dtype=np.int64)
            true_score = decision[np.arange(len(truth)), true_index]
            masked_decision = decision.copy()
            masked_decision[np.arange(len(truth)), true_index] = -np.inf
            margin = true_score - masked_decision.max(axis=1)
            nll_values = -np.log(probability_true)
            task_metric_rows.append({
                "scope": "checkpoint",
                "tag": tag,
                "budget_h": budget_seed(tag)[0] or "",
                "seed": budget_seed(tag)[1] if budget_seed(tag)[1] is not None else "",
                "split": split_name,
                "task_nll": float(np.mean(nll_values)),
                "cohen_kappa": float(cohen_kappa_score(truth, prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
                "class_margin_mean": float(np.mean(margin)),
                "class_margin_median": float(np.median(margin)),
                "class_margin_positive_frac": float(np.mean(margin > 0)),
                "pca_dim": PCA_DIM,
                "probe_C": PROBE_C,
            })
            selected_indices = np.where(mask)[0]
            for local, global_index in enumerate(selected_indices):
                task_sample_rows.append({
                    "record_type": "task_probe_support",
                    "tag": tag,
                    "split": split_name,
                    "subject": int(subjects[global_index]),
                    "clip_id": int(clips[global_index]),
                    "segment_id": int(segments[global_index]),
                    "fold": "",
                    "label_final_scoring_only": int(truth[local]),
                    "prediction": int(prediction[local]),
                    "nll": float(nll_values[local]),
                    "margin": float(margin[local]),
                })
            split_subjects = sorted(int(value) for value in np.unique(subjects[mask]))
            cluster_nll = {
                subject: float(np.mean(nll_values[subjects[mask] == subject]))
                for subject in split_subjects
            }
            task_target_clusters[tag][split_name] = cluster_nll
            task_confusions[tag][split_name] = confusion_by_subject(
                truth, prediction, subjects[mask], split_subjects
            )

        checksum_end = checksum_feature(model, checksum_batch, device)
        checksum_end_sha = hashlib.sha256(checksum_end.tobytes()).hexdigest()
        checksum_max_diff = float(np.max(np.abs(checksum_start - checksum_end)))
        if checksum_end_sha != row["feature_hash"] or checksum_max_diff != 0.0:
            raise RuntimeError(f"{tag} feature contract changed during B1")
        physical_sha_end = "logical_contract"
        if tag != "random":
            physical_sha_end = sha256_file(immutable_path)
            if physical_sha_end != physical_sha_start:
                raise RuntimeError(f"{tag} checkpoint changed during B1")
        feature_rows.append({
            "tag": tag,
            "full_feature_shape": "x".join(map(str, features.shape)),
            "full_feature_sha256": full_feature_sha,
            "checksum_feature_sha256_start": checksum_start_sha,
            "checksum_feature_sha256_end": checksum_end_sha,
            "closure_feature_sha256": row["feature_hash"],
            "checksum_start_end_max_abs_diff": checksum_max_diff,
            "feature_contract_pass": True,
        })
        hash_rows.append({
            "tag": tag,
            "immutable_path": immutable_path,
            "expected_sha256": row["immutable_sha256"],
            "sha256_at_object_start": physical_sha_start,
            "sha256_at_object_end": physical_sha_end,
            "sha256_at_global_end": "pending",
            "hash_recheck_pass": True,
        })
        del model, features
        torch.cuda.empty_cache()

    global_hash = {row["tag"]: row for row in hash_rows}
    for tag in TAGS:
        if tag == "random":
            global_hash[tag]["sha256_at_global_end"] = "logical_contract"
            continue
        digest = sha256_file(manifest[tag]["immutable_path"])
        global_hash[tag]["sha256_at_global_end"] = digest
        if digest != manifest[tag]["immutable_sha256"]:
            raise RuntimeError(f"{tag} global-end SHA mismatch")

    subject_metric = {row["tag"]: row for row in subject_metric_rows}
    all_pretrained_auc_saturated = all(
        float(subject_metric[tag]["pairwise_subject_auc"]) > 0.9999
        for tag in TAGS if tag.startswith("H")
    )
    h200_nll_saturated = all(
        float(subject_metric[tag]["heldout_clip_subject_nll"]) < 1e-6
        for tag in budget_tags(200)
    )
    h200_probability_saturated = all(
        float(subject_metric[tag]["true_subject_probability_gt_0p999999_frac"]) > 0.99
        for tag in budget_tags(200)
    )
    primary_subject_uninformative = (
        all_pretrained_auc_saturated or h200_nll_saturated or h200_probability_saturated
    )
    saturation_reasons = []
    if all_pretrained_auc_saturated:
        saturation_reasons.append("all_pretrained_pairwise_auc_gt_0p9999")
    if h200_nll_saturated:
        saturation_reasons.append("both_H200_subject_nll_lt_1e-6")
    if h200_probability_saturated:
        saturation_reasons.append("both_H200_true_probability_gt_0p999999_frac_gt_0p99")
    for row in subject_metric_rows:
        row["primary_subject_metric_uninformative"] = primary_subject_uninformative
        row["saturation_reasons"] = ";".join(saturation_reasons)

    for tag in TAGS:
        fold_geometry = [
            row for row in geometry_rows if row["tag"] == tag and row["scope"] == "fold"
        ]
        geometry_rows.append({
            "scope": "checkpoint_mean",
            "tag": tag,
            "budget_h": budget_seed(tag)[0] or "",
            "seed": budget_seed(tag)[1] if budget_seed(tag)[1] is not None else "",
            "fold": "mean_0_1_2",
            "pca_dim": PCA_DIM,
            "rank": SUBSPACE_RANK,
            "projection_overlap": float(np.mean([row["projection_overlap"] for row in fold_geometry])),
            "max_canonical_correlation": float(np.mean([row["max_canonical_correlation"] for row in fold_geometry])),
            "median_principal_angle_deg": float(np.mean([row["median_principal_angle_deg"] for row in fold_geometry])),
            "heldout_subject_self_capture": float(np.mean([row["heldout_subject_self_capture"] for row in fold_geometry])),
            "heldout_task_self_capture": float(np.mean([row["heldout_task_self_capture"] for row in fold_geometry])),
            "heldout_subject_on_task_capture": float(np.mean([row["heldout_subject_on_task_capture"] for row in fold_geometry])),
            "heldout_task_on_subject_capture": float(np.mean([row["heldout_task_on_subject_capture"] for row in fold_geometry])),
            "subspace_stable": all(as_bool(row["subspace_stable"]) for row in fold_geometry),
        })
        fold_variance = [
            row for row in variance_rows if row["tag"] == tag and row["scope"] == "fold"
        ]
        variance_rows.append({
            "scope": "checkpoint_mean",
            "tag": tag,
            "budget_h": budget_seed(tag)[0] or "",
            "seed": budget_seed(tag)[1] if budget_seed(tag)[1] is not None else "",
            "fold": "mean_0_1_2",
            **{
                key: float(np.mean([row[key] for row in fold_variance]))
                for key in (
                    "subject", "class", "interaction", "residual", "total",
                    "subject_frac", "class_frac", "interaction_frac", "residual_frac",
                )
            },
        })

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    source_counts = rng.multinomial(
        N_SOURCE_SUBJECTS,
        np.full(N_SOURCE_SUBJECTS, 1 / N_SOURCE_SUBJECTS),
        size=N_BOOTSTRAP,
    ).astype(np.float64)
    source_weights = source_counts / source_counts.sum(axis=1, keepdims=True)
    target_subject_values = list(range(101, 124))
    target_counts = rng.multinomial(
        len(target_subject_values),
        np.full(len(target_subject_values), 1 / len(target_subject_values)),
        size=N_BOOTSTRAP,
    ).astype(np.float64)
    target_weights = target_counts / target_counts.sum(axis=1, keepdims=True)

    subject_nll_samples = {}
    subject_margin_samples = {}
    subject_auc_samples = {}
    for tag in TAGS:
        cluster_values = np.asarray(
            [row["heldout_clip_subject_nll"] for row in subject_clusters[tag]], dtype=np.float64
        )
        subject_nll_samples[tag] = source_weights @ cluster_values
        subject_margin_samples[tag] = weighted_pair_bootstrap(
            pairs_by_tag[tag], source_counts, "pairwise_standardized_margin"
        )
        subject_auc_samples[tag] = weighted_pair_bootstrap(
            pairs_by_tag[tag], source_counts, "pairwise_auc"
        )

    target_nll_samples = {}
    target_kappa_samples = {}
    target_bacc_samples = {}
    for tag in TAGS:
        nll_by_subject = np.asarray(
            [task_target_clusters[tag]["target_test"][subject] for subject in target_subject_values]
        )
        target_nll_samples[tag] = target_weights @ nll_by_subject
        confusion = task_confusions[tag]["target_test"]
        boot_confusion = np.einsum("bs,sij->bij", target_counts, confusion, optimize=True)
        target_kappa_samples[tag], target_bacc_samples[tag] = metrics_from_confusion(boot_confusion)

    print("B1 geometry bootstrap start", flush=True)
    geometry_samples = {tag: np.empty(N_BOOTSTRAP, dtype=np.float64) for tag in TAGS}
    for tag in TAGS:
        for replicate in range(N_BOOTSTRAP):
            fold_values = [
                effect_geometry(geometry_cells[tag][fold], source_weights[replicate])["overlap"]
                for fold in range(3)
            ]
            geometry_samples[tag][replicate] = float(np.mean(fold_values))
        print(f"B1 geometry bootstrap complete tag={tag}", flush=True)

    print("B1 variance bootstrap start", flush=True)
    variance_samples = {
        tag: {key: np.zeros(N_BOOTSTRAP, dtype=np.float64) for key in (
            "subject_frac", "class_frac", "interaction_frac", "residual_frac"
        )}
        for tag in TAGS
    }
    for tag in TAGS:
        fold_results = [
            variance_components(variance_suff[tag][fold], source_weights) for fold in range(3)
        ]
        for key in variance_samples[tag]:
            variance_samples[tag][key] = np.mean([result[key] for result in fold_results], axis=0)

    if primary_subject_uninformative:
        primary_subject_metric = "pairwise_subject_margin_fallback"
        p1_samples = mean_arrays(subject_margin_samples, budget_tags(200)) - subject_margin_samples["random"]
        p1_point = float(
            np.mean([subject_metric[tag]["pairwise_subject_margin"] for tag in budget_tags(200)])
            - subject_metric["random"]["pairwise_subject_margin"]
        )
    else:
        primary_subject_metric = "heldout_clip_subject_nll"
        p1_samples = subject_nll_samples["random"] - mean_arrays(
            subject_nll_samples, budget_tags(200)
        )
        p1_point = float(
            subject_metric["random"]["heldout_clip_subject_nll"]
            - np.mean([subject_metric[tag]["heldout_clip_subject_nll"] for tag in budget_tags(200)])
        )
    task_metric_lookup = {
        row["tag"]: row for row in task_metric_rows if row["split"] == "target_test"
    }
    p2_samples = mean_arrays(target_nll_samples, budget_tags(200)) - pool_high(target_nll_samples)
    p2_point = float(
        np.mean([task_metric_lookup[tag]["task_nll"] for tag in budget_tags(200)])
        - np.mean([
            np.mean([task_metric_lookup[tag]["task_nll"] for tag in budget_tags(budget)])
            for budget in HIGH_BUDGETS
        ])
    )
    geometry_lookup = {
        row["tag"]: row for row in geometry_rows if row["scope"] == "checkpoint_mean"
    }
    p3_samples = pool_high(geometry_samples) - mean_arrays(geometry_samples, budget_tags(200))
    p3_point = float(
        np.mean([
            np.mean([geometry_lookup[tag]["projection_overlap"] for tag in budget_tags(budget)])
            for budget in HIGH_BUDGETS
        ])
        - np.mean([geometry_lookup[tag]["projection_overlap"] for tag in budget_tags(200)])
    )
    primary_names = [
        "P1_random_vs_H200_subject",
        "P2_H200_vs_pooled_high_target_nll",
        "P3_H200_vs_pooled_high_overlap",
    ]
    primary_points = [p1_point, p2_point, p3_point]
    primary_samples = [p1_samples, p2_samples, p3_samples]
    raw_p = [
        one_sided_positive_p(p1_samples),
        one_sided_positive_p(p2_samples),
        two_sided_sign_p(p3_samples),
    ]
    adjusted = holm_adjust(raw_p)
    primary_rows = []
    for name, point, samples, raw, corrected in zip(
        primary_names, primary_points, primary_samples, raw_p, adjusted
    ):
        primary_rows.append({
            "contrast": name,
            **summarize_samples(point, samples),
            "p_raw": raw,
            "p_holm": float(corrected),
            "reject_holm_0p05": bool(corrected < 0.05),
        })

    sensitivity_rows = []
    subject_continuing_samples = mean_arrays(subject_nll_samples, budget_tags(200)) - pool_high(subject_nll_samples)
    sensitivity_rows.append({
        "analysis": "subject_nll_H200_minus_pooled_high",
        **summarize_samples(
            float(np.mean([subject_metric[tag]["heldout_clip_subject_nll"] for tag in budget_tags(200)])
                  - np.mean([
                      np.mean([subject_metric[tag]["heldout_clip_subject_nll"] for tag in budget_tags(budget)])
                      for budget in HIGH_BUDGETS
                  ])),
            subject_continuing_samples,
        ),
    })
    for metric_name, samples_map, point_key in (
        ("pairwise_auc", subject_auc_samples, "pairwise_subject_auc"),
        ("pairwise_margin", subject_margin_samples, "pairwise_subject_margin"),
    ):
        early_samples = mean_arrays(samples_map, budget_tags(200)) - samples_map["random"]
        early_point = float(
            np.mean([subject_metric[tag][point_key] for tag in budget_tags(200)])
            - subject_metric["random"][point_key]
        )
        sensitivity_rows.append({
            "analysis": f"{metric_name}_H200_minus_random",
            **summarize_samples(early_point, early_samples),
        })

    for metric_name, samples_map, point_key in (
        ("target_kappa", target_kappa_samples, "cohen_kappa"),
        ("target_bacc", target_bacc_samples, "balanced_accuracy"),
    ):
        delta_samples = pool_high(samples_map) - mean_arrays(samples_map, budget_tags(200))
        point = float(
            np.mean([
                np.mean([task_metric_lookup[tag][point_key] for tag in budget_tags(budget)])
                for budget in HIGH_BUDGETS
            ])
            - np.mean([task_metric_lookup[tag][point_key] for tag in budget_tags(200)])
        )
        sensitivity_rows.append({
            "analysis": f"pooled_high_minus_H200_{metric_name}",
            **summarize_samples(point, delta_samples),
        })

    for omitted in HIGH_BUDGETS:
        retained = [budget for budget in HIGH_BUDGETS if budget != omitted]
        sensitivity_rows.extend([
            {
                "analysis": f"leave_high_budget_{omitted}_out_task_nll",
                **summarize_samples(
                    p2_point if retained == HIGH_BUDGETS else float(
                        np.mean([task_metric_lookup[tag]["task_nll"] for tag in budget_tags(200)])
                        - np.mean([
                            np.mean([task_metric_lookup[tag]["task_nll"] for tag in budget_tags(budget)])
                            for budget in retained
                        ])
                    ),
                    mean_arrays(target_nll_samples, budget_tags(200))
                    - pool_high(target_nll_samples, budgets=retained),
                ),
            },
            {
                "analysis": f"leave_high_budget_{omitted}_out_overlap",
                **summarize_samples(
                    float(
                        np.mean([
                            np.mean([geometry_lookup[tag]["projection_overlap"] for tag in budget_tags(budget)])
                            for budget in retained
                        ])
                        - np.mean([geometry_lookup[tag]["projection_overlap"] for tag in budget_tags(200)])
                    ),
                    pool_high(geometry_samples, budgets=retained)
                    - mean_arrays(geometry_samples, budget_tags(200)),
                ),
            },
        ])

    for retained_seed in (0, 1):
        sensitivity_rows.extend([
            {
                "analysis": f"retain_training_seed_{retained_seed}_subject_early",
                **summarize_samples(
                    float(subject_metric[f"H200_s{retained_seed}"]["pairwise_subject_margin"]
                          - subject_metric["random"]["pairwise_subject_margin"])
                    if primary_subject_uninformative else
                    float(subject_metric["random"]["heldout_clip_subject_nll"]
                          - subject_metric[f"H200_s{retained_seed}"]["heldout_clip_subject_nll"]),
                    subject_margin_samples[f"H200_s{retained_seed}"] - subject_margin_samples["random"]
                    if primary_subject_uninformative else
                    subject_nll_samples["random"] - subject_nll_samples[f"H200_s{retained_seed}"],
                ),
            },
            {
                "analysis": f"retain_training_seed_{retained_seed}_task_nll",
                **summarize_samples(
                    float(task_metric_lookup[f"H200_s{retained_seed}"]["task_nll"]
                          - np.mean([task_metric_lookup[f"H{budget}_s{retained_seed}"]["task_nll"] for budget in HIGH_BUDGETS])),
                    target_nll_samples[f"H200_s{retained_seed}"]
                    - pool_high(target_nll_samples, seed=retained_seed),
                ),
            },
            {
                "analysis": f"retain_training_seed_{retained_seed}_overlap",
                **summarize_samples(
                    float(np.mean([geometry_lookup[f"H{budget}_s{retained_seed}"]["projection_overlap"] for budget in HIGH_BUDGETS])
                          - geometry_lookup[f"H200_s{retained_seed}"]["projection_overlap"]),
                    pool_high(geometry_samples, seed=retained_seed)
                    - geometry_samples[f"H200_s{retained_seed}"],
                ),
            },
        ])

    variance_lookup = {
        row["tag"]: row for row in variance_rows if row["scope"] == "checkpoint_mean"
    }
    variance_contrasts = {}
    for component in ("subject_frac", "class_frac", "interaction_frac", "residual_frac"):
        samples = pool_high(
            {tag: variance_samples[tag][component] for tag in TAGS}
        ) - mean_arrays(
            {tag: variance_samples[tag][component] for tag in TAGS}, budget_tags(200)
        )
        point = float(
            np.mean([
                np.mean([variance_lookup[tag][component] for tag in budget_tags(budget)])
                for budget in HIGH_BUDGETS
            ])
            - np.mean([variance_lookup[tag][component] for tag in budget_tags(200)])
        )
        variance_contrasts[component] = summarize_samples(point, samples)
        sensitivity_rows.append({
            "analysis": f"variance_pooled_high_minus_H200_{component}",
            **variance_contrasts[component],
        })

    for tag in TAGS:
        fold_values = [
            row for row in variance_rows if row["tag"] == tag and row["scope"] == "fold"
        ]
        instability_reasons = []
        for component in ("subject_frac", "class_frac", "interaction_frac", "residual_frac"):
            mean_value = float(np.mean([row[component] for row in fold_values]))
            max_deviation = float(max(abs(row[component] - mean_value) for row in fold_values))
            bootstrap_width = float(
                np.quantile(variance_samples[tag][component], 0.975)
                - np.quantile(variance_samples[tag][component], 0.025)
            )
            if max_deviation > 0.10 or bootstrap_width > 0.20:
                instability_reasons.append(
                    f"{component}:max_fold_deviation={max_deviation:.6g},ci_width={bootstrap_width:.6g}"
                )
        minimum_residual = min(float(row["residual_frac"]) for row in fold_values)
        if minimum_residual < -0.01:
            instability_reasons.append(f"residual_frac_min={minimum_residual:.6g}")
        stability = "UNSTABLE_UNDER_CLIP_CROSSFIT" if instability_reasons else "PASS"
        for row in variance_rows:
            if row["tag"] == tag:
                row["variance_stability"] = stability
                row["variance_stability_reason"] = ";".join(instability_reasons)

    high_component_means = {
        component: float(np.mean([
            np.mean([variance_lookup[tag][component] for tag in budget_tags(budget)])
            for budget in HIGH_BUDGETS
        ]))
        for component in ("subject_frac", "class_frac", "interaction_frac", "residual_frac")
    }
    variance_dominant = max(high_component_means, key=high_component_means.get)
    subject_early = primary_rows[0]["reject_holm_0p05"] and p1_point > 0
    task_later = primary_rows[1]["reject_holm_0p05"] and p2_point > 0
    overlap_significant = primary_rows[2]["reject_holm_0p05"]
    overlap_trend = (
        "increase" if overlap_significant and p3_point > 0
        else "decrease" if overlap_significant and p3_point < 0
        else "no_detectable_change"
    )
    continuing_subject = (
        sensitivity_rows[0]["ci95_low"] > 0
        and sensitivity_rows[0]["point"] > 0
        and not primary_subject_uninformative
    )
    interaction_increases = variance_contrasts["interaction_frac"]["ci95_low"] > 0
    all_subspaces_stable = all(
        as_bool(row["subspace_stable"])
        for row in geometry_rows if row["scope"] == "checkpoint_mean"
    )
    any_variance_unstable = any(
        row.get("variance_stability") != "PASS"
        for row in variance_rows if row["scope"] == "checkpoint_mean"
    )
    if any_variance_unstable:
        unstable_tags = sorted({
            row["tag"] for row in variance_rows
            if row["scope"] == "checkpoint_mean" and row.get("variance_stability") != "PASS"
        })
        raise RuntimeError(
            "variance partition failed the frozen stability gate: " + ",".join(unstable_tags)
        )
    if task_later and interaction_increases and not any_variance_unstable:
        mechanism = "C_subject_class_interaction"
    elif task_later and not overlap_significant and all_subspaces_stable:
        mechanism = (
            "B_continuing_subject_plus_distinct_task" if continuing_subject
            else "A_distinct_task_structure"
        )
    else:
        mechanism = "D_unresolved"

    bootstrap_rows = []
    for index in range(N_BOOTSTRAP):
        bootstrap_rows.append({
            "replicate": index,
            "p1_subject_delta": float(p1_samples[index]),
            "p2_task_nll_delta": float(p2_samples[index]),
            "p3_overlap_delta": float(p3_samples[index]),
        })

    bootstrap_support = {
        "tags": np.asarray(TAGS),
        "source_counts": source_counts.astype(np.int16),
        "target_counts": target_counts.astype(np.int16),
        "subject_nll_samples": np.stack([subject_nll_samples[tag] for tag in TAGS]),
        "subject_margin_samples": np.stack([subject_margin_samples[tag] for tag in TAGS]),
        "subject_auc_samples": np.stack([subject_auc_samples[tag] for tag in TAGS]),
        "target_nll_samples": np.stack([target_nll_samples[tag] for tag in TAGS]),
        "target_kappa_samples": np.stack([target_kappa_samples[tag] for tag in TAGS]),
        "target_bacc_samples": np.stack([target_bacc_samples[tag] for tag in TAGS]),
        "geometry_overlap_samples": np.stack([geometry_samples[tag] for tag in TAGS]),
    }
    for tag in TAGS:
        for fold in range(3):
            bootstrap_support[f"geometry_cell__{tag}__fold{fold}"] = geometry_cells[tag][fold]
            bootstrap_support[f"geometry_hold_cell__{tag}__fold{fold}"] = (
                geometry_hold_cells[tag][fold]
            )
            for key, value in variance_suff[tag][fold].items():
                bootstrap_support[f"variance__{tag}__fold{fold}__{key}"] = value
        for component in ("subject_frac", "class_frac", "interaction_frac", "residual_frac"):
            bootstrap_support[f"variance_samples__{tag}__{component}"] = (
                variance_samples[tag][component]
            )

    primary_inference = {
        "phase": "B1_primary_confirmatory_inference",
        "bootstrap_reps": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_subject_metric": primary_subject_metric,
        "primary_subject_metric_uninformative": primary_subject_uninformative,
        "subject_metric_saturation_reasons": saturation_reasons,
        "holm_family_size": 3,
        "contrasts": primary_rows,
        "target_labels_used_for_selection": False,
    }
    verdict = {
        "phase": "B1_final_representation_decomposition",
        "status": "PASS",
        "checkpoint_objects": 10,
        "all_hashes_reverified": True,
        "all_features_match_closure": True,
        "clip_group_crossfit_pass": True,
        "primary_subject_metric": primary_subject_metric,
        "primary_subject_metric_uninformative": primary_subject_uninformative,
        "subject_structure_early": bool(subject_early),
        "subject_structure_continues_to_strengthen": bool(continuing_subject),
        "task_structure_later": bool(task_later),
        "subject_task_overlap_trend": overlap_trend,
        "all_subject_task_subspaces_stable": all_subspaces_stable,
        "variance_dominant_component": variance_dominant,
        "variance_analysis_unstable": any_variance_unstable,
        "interaction_component_increases": bool(interaction_increases),
        "mechanism_verdict": mechanism,
        "target_labels_used_for_selection": False,
        "recommend_phase_b2_layerwise": False,
        "phase_b2_authorized": False,
        "new_pretraining": False,
        "fine_tuning": False,
        "h4000": False,
        "codebrain": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    firewall = {
        "target_labels_used_for_pca": False,
        "target_labels_used_for_probe_fit": False,
        "target_labels_used_for_probe_selection": False,
        "target_labels_used_for_rank_selection": False,
        "target_labels_used_for_metric_selection": False,
        "target_labels_used_for_checkpoint_selection": False,
        "target_labels_used_for_final_task_scoring": True,
        "target_labels_used_for_target_subject_cluster_bootstrap_scoring": True,
        "source_labels_used_for_subject_and_task_fits": True,
        "target_labels_used_for_selection": False,
    }

    write_csv(out / "phase_b1_run_manifest.csv", [run_manifest])
    write_csv(out / "phase_b1_checkpoint_hash_recheck.csv", hash_rows)
    write_csv(out / "phase_b1_feature_manifest.csv", feature_rows)
    write_csv(out / "phase_b1_subject_metrics.csv", subject_metric_rows)
    write_csv(out / "phase_b1_subject_cluster_scores.csv", subject_cluster_rows)
    write_csv(out / "phase_b1_subject_pair_metrics.csv", subject_pair_rows)
    write_csv(out / "phase_b1_task_metrics.csv", task_metric_rows)
    write_csv(out / "phase_b1_task_sample_scores.csv", task_sample_rows)
    write_csv(out / "phase_b1_subject_task_geometry.csv", geometry_rows)
    write_csv(out / "phase_b1_variance_partition.csv", variance_rows)
    write_csv(out / "phase_b1_sensitivity_results.csv", sensitivity_rows)
    write_csv(out / "phase_b1_primary_bootstrap_samples.csv", bootstrap_rows)
    np.savez_compressed(out / "phase_b1_bootstrap_support.npz", **bootstrap_support)
    write_json(out / "phase_b1_primary_inference.json", primary_inference)
    write_json(out / "phase_b1_target_label_firewall.json", firewall)
    write_json(out / "phase_b1_verdict.json", verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-manifest",
        default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_checkpoint_immutable_manifest.csv",
    )
    parser.add_argument(
        "--closure-json",
        default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_provenance_closure.json",
    )
    parser.add_argument(
        "--provenance-redteam",
        default="results/s2p_route_b_phase_b_checkpoint_closure/phase_b_provenance_redteam_verdict.json",
    )
    parser.add_argument(
        "--b0-go-nogo",
        default="results/s2p_route_b_representation_emergence_b0/phase_b0_go_nogo.json",
    )
    parser.add_argument(
        "--protocol-doc",
        default="docs/S2P_19_REPRESENTATION_EMERGENCE_PROTOCOL.md",
    )
    parser.add_argument(
        "--redteam-doc",
        default="docs/S2P_20_REPRESENTATION_EMERGENCE_REDTEAM.md",
    )
    parser.add_argument("--faced-lmdb", default="/projects/EEG-foundation-model/FACED_data/processed")
    parser.add_argument(
        "--out-dir",
        default="results/s2p_route_b_representation_emergence_b1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--pm-authorized", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as exc:
        write_json(out / "phase_b1_verdict.json", {
            "phase": "B1_final_representation_decomposition",
            "status": "NO_GO",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "target_labels_used_for_selection": False,
            "phase_b2_authorized": False,
        })
        raise


if __name__ == "__main__":
    main()
