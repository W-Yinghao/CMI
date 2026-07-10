#!/usr/bin/env python
"""Final confirmatory verification for the Route-B FACED frozen audit.

This script performs no pretraining or fine-tuning. It refits only the already
selected source-side frozen probes, reproduces their committed aggregate
metrics, and then evaluates target-subject-clustered uncertainty and task-gated
L4/L5/L6 mechanism diagnostics. FACED test labels are used only for scoring.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import route_b_faced_downstream_audit as faced


TAGS_THROUGH_1000 = [
    "random",
    "released",
    "H200_s0",
    "H200_s1",
    "H500_s0",
    "H500_s1",
    "H1000_s0",
    "H1000_s1",
]
TAGS_FULL = [
    *TAGS_THROUGH_1000,
    "H2000_s0",
    "H2000_s1",
]
BUDGETS_THROUGH_1000 = [200, 500, 1000]
BUDGETS_FULL = [200, 500, 1000, 2000]
N_CLASSES = 9


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    faced.write_csv(Path(path), rows, fieldnames=fieldnames)


def write_json(path, obj):
    faced.write_json(Path(path), obj)


def as_float(row, key):
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    return float(value)


def as_bool(value):
    return str(value).lower() == "true"


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_for(tag, ckpt_root, h2000_ckpt_root=None):
    if tag == "random":
        return "random"
    if tag == "released":
        return faced.RELEASED_CKPT
    if tag.startswith("H2000"):
        if h2000_ckpt_root is None:
            raise RuntimeError("full verification requires --h2000-ckpt-root")
        return str(Path(h2000_ckpt_root) / tag / "best.pth")
    return str(Path(ckpt_root) / tag / "best.pth")


def load_h2000_immutable_contract(root, manifest_path, go_nogo_path):
    go_nogo = json.loads(Path(go_nogo_path).read_text())
    if go_nogo.get("status") != "PASS" or go_nogo.get("faced_reaudit_unlocked") is not True:
        raise RuntimeError("H2000 immutable closure is not PASS")
    rows = {
        row["tag"]: row
        for row in json.loads(Path(manifest_path).read_text()).get("checkpoints", [])
    }
    expected = {"H2000_s0", "H2000_s1"}
    if set(rows) != expected:
        raise RuntimeError(f"H2000 immutable manifest mismatch: {sorted(rows)}")
    for tag, row in rows.items():
        link = Path(root) / tag / "best.pth"
        payload = Path(row["immutable_checkpoint"])
        if not link.is_symlink() or link.resolve() != payload.resolve():
            raise RuntimeError(f"H2000 immutable path mismatch for {tag}")
        if payload.stat().st_mode & 0o222:
            raise RuntimeError(f"H2000 immutable checkpoint is writable for {tag}")
        if sha256_file(payload) != row["sha256"]:
            raise RuntimeError(f"H2000 immutable SHA mismatch for {tag}")
    return rows


def metric_from_confusion(cm):
    """Return kappa, balanced accuracy and weighted F1 for (..., K, K) CMs."""
    cm = np.asarray(cm, dtype=np.float64)
    rows = cm.sum(axis=-1)
    cols = cm.sum(axis=-2)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    total = rows.sum(axis=-1)
    observed = diag.sum(axis=-1) / np.maximum(total, 1.0)
    expected = (rows * cols).sum(axis=-1) / np.maximum(total * total, 1.0)
    kappa = (observed - expected) / np.maximum(1.0 - expected, 1e-12)
    recall = np.divide(diag, rows, out=np.full_like(diag, np.nan), where=rows > 0)
    bacc = np.nanmean(recall, axis=-1)
    denom = 2.0 * diag + (cols - diag) + (rows - diag)
    f1 = np.divide(2.0 * diag, denom, out=np.zeros_like(diag), where=denom > 0)
    weighted_f1 = (f1 * rows).sum(axis=-1) / np.maximum(total, 1.0)
    return {"kappa": kappa, "bacc": bacc, "weighted_f1": weighted_f1}


def confusion_by_subject(y, pred, subject, subjects):
    out = np.zeros((len(subjects), N_CLASSES, N_CLASSES), dtype=np.int32)
    index = {int(s): i for i, s in enumerate(subjects)}
    for yt, yp, sid in zip(y, pred, subject):
        out[index[int(sid)], int(yt), int(yp)] += 1
    return out


def bootstrap_metrics(subject_cm, weights):
    aggregate = np.einsum("bs,skl->bkl", weights, subject_cm, optimize=True)
    return metric_from_confusion(aggregate)


def summarize(point, samples):
    samples = np.asarray(samples, dtype=np.float64)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "point": float(point),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_sd": float(samples.std()),
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p_values[idx]))
        adjusted[idx] = min(running, 1.0)
    return adjusted


def fit_fixed_probe(feat, y, split, pca_dim, c_value):
    masks = faced.masks(split)
    pca = PCA(n_components=int(pca_dim), svd_solver="randomized", random_state=0)
    ztr = pca.fit_transform(feat[masks["source_train"]])
    zva = pca.transform(feat[masks["source_val"]])
    zte = pca.transform(feat[masks["target_test"]])
    clf = LogisticRegression(C=float(c_value), max_iter=2000, n_jobs=1)
    clf.fit(ztr, y[masks["source_train"]])
    return {
        "pca": pca,
        "clf": clf,
        "ztr": ztr,
        "zva": zva,
        "zte": zte,
        "pred_val": clf.predict(zva),
        "pred_test": clf.predict(zte),
    }


def rank_null_subject_confusions(clf, zte, yte, ste, subjects, rank, n_null):
    """Historical FACED control: rank-matched random projection erasure."""
    rng = np.random.default_rng(92014)
    out = np.empty((n_null, len(subjects), N_CLASSES, N_CLASSES), dtype=np.int16)
    for i in range(n_null):
        q, _ = np.linalg.qr(rng.standard_normal((zte.shape[1], rank)))
        pred = clf.predict(faced.remove_subspace(zte, q.T))
        out[i] = confusion_by_subject(yte, pred, ste, subjects)
    return out


def variance_fraction_removed(z, z_after):
    return float(((z - z_after) ** 2).sum() / ((z ** 2).sum() + 1e-12))


def variance_matched_null_subject_confusions(
    clf, zva, zte, yte, ste, subjects, subject_basis, n_null
):
    """Build source-val-energy-matched random erasures without target labels.

    Each random orthobasis is traversed until its cumulative source-val energy
    reaches the subject-erasure energy. The final direction is only partially
    erased, making the removed source-val variance equal to the subject
    intervention while leaving direction and effective rank random.
    """
    rng = np.random.default_rng(92015)
    subject_after_val = faced.remove_subspace(zva, subject_basis)
    subject_after_test = faced.remove_subspace(zte, subject_basis)
    target_fraction = variance_fraction_removed(zva, subject_after_val)
    d = zva.shape[1]
    out = np.empty((n_null, len(subjects), N_CLASSES, N_CLASSES), dtype=np.int16)
    diagnostics = []
    total_val_energy = float((zva ** 2).sum()) + 1e-12
    for i in range(n_null):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        per_direction = ((zva @ q) ** 2).sum(axis=0) / total_val_energy
        cumulative = np.cumsum(per_direction)
        last = int(np.searchsorted(cumulative, target_fraction, side="left"))
        if last >= d:
            raise RuntimeError("random orthobasis could not reach subject removed variance")
        previous = float(cumulative[last - 1]) if last else 0.0
        residual = max(target_fraction - previous, 0.0)
        alpha = float(np.sqrt(residual / max(float(per_direction[last]), 1e-12)))
        alpha = min(alpha, 1.0)
        directions = q[:, :last + 1]
        coefficients = np.ones(last + 1, dtype=np.float64)
        coefficients[-1] = alpha

        def erase(z):
            projected = z @ directions
            return z - (projected * coefficients) @ directions.T

        zva_after = erase(zva)
        zte_after = erase(zte)
        pred = clf.predict(zte_after)
        out[i] = confusion_by_subject(yte, pred, ste, subjects)
        diagnostics.append({
            "null_index": i,
            "effective_rank": last + 1,
            "partial_last_direction_coefficient": alpha,
            "source_val_removed_variance_frac": variance_fraction_removed(zva, zva_after),
            "source_val_match_abs_error": abs(variance_fraction_removed(zva, zva_after) - target_fraction),
            "target_removed_variance_frac": variance_fraction_removed(zte, zte_after),
        })
    subject_diagnostics = {
        "subject_source_val_removed_variance_frac": target_fraction,
        "subject_target_removed_variance_frac": variance_fraction_removed(zte, subject_after_test),
    }
    return out, diagnostics, subject_diagnostics


def null_bootstrap_mean(null_cm, weights, metric, chunk_size=10):
    total = np.zeros(weights.shape[0], dtype=np.float64)
    for start in range(0, len(null_cm), chunk_size):
        chunk = null_cm[start:start + chunk_size]
        aggregate = np.einsum("bs,nskl->bnkl", weights, chunk, optimize=True)
        total += metric_from_confusion(aggregate)[metric].sum(axis=1)
    return total / len(null_cm)


def regression_summary(budget_series, metric, selected_budgets):
    budgets = np.asarray(selected_budgets, dtype=np.float64)
    x = np.log2(budgets)
    y_point = np.asarray([budget_series[h][metric]["point"] for h in selected_budgets])
    y_boot = np.stack([budget_series[h][metric]["samples"] for h in selected_budgets], axis=1)
    linear_x = np.column_stack([np.ones(len(x)), x])
    linear_point = np.linalg.lstsq(linear_x, y_point, rcond=None)[0]
    linear_boot = (np.linalg.pinv(linear_x) @ y_boot.T).T
    xc = x - x.mean()
    quad_x = np.column_stack([np.ones(len(x)), xc, xc * xc])
    quad_point = np.linalg.lstsq(quad_x, y_point, rcond=None)[0]
    quad_boot = (np.linalg.pinv(quad_x) @ y_boot.T).T
    loo = []
    for omitted in selected_budgets:
        keep = budgets != omitted
        xx = np.column_stack([np.ones(int(keep.sum())), x[keep]])
        coef_point = np.linalg.lstsq(xx, y_point[keep], rcond=None)[0]
        coef_boot = (np.linalg.pinv(xx) @ y_boot[:, keep].T).T
        row = {"omitted_budget_h": omitted}
        row.update({f"slope_{k}": v for k, v in summarize(coef_point[1], coef_boot[:, 1]).items()})
        row["bootstrap_probability_positive"] = float((coef_boot[:, 1] > 0).mean())
        loo.append(row)
    return {
        "metric": metric,
        "x": "log2_budget_hours",
        "linear_slope": summarize(linear_point[1], linear_boot[:, 1]),
        "linear_slope_bootstrap_probability_positive": float((linear_boot[:, 1] > 0).mean()),
        "quadratic_term_centered_log2": summarize(quad_point[2], quad_boot[:, 2]),
        "leave_one_budget_out": loo,
        "leave_one_point_sign_stable": len({np.sign(r["slope_point"]) for r in loo}) == 1,
        "observed_budget_means_monotone": bool(np.all(np.diff(y_point) >= 0)),
        "monotonic_scaling_established": False,
    }


def self_test():
    y = np.asarray([0, 1, 0, 1, 0, 1])
    pred = np.asarray([0, 1, 1, 1, 0, 0])
    subject = np.asarray([1, 1, 2, 2, 3, 3])
    cm = confusion_by_subject(y, pred, subject, [1, 2, 3]).sum(axis=0)
    got = metric_from_confusion(cm)
    assert abs(float(got["kappa"]) - cohen_kappa_score(y, pred)) < 1e-12
    assert abs(float(got["bacc"]) - balanced_accuracy_score(y, pred)) < 1e-12
    rng = np.random.default_rng(7)
    zva = rng.normal(size=(20, 4))
    zte = rng.normal(size=(12, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 1)))

    class DummyHead:
        @staticmethod
        def predict(z):
            return (z[:, 0] > 0).astype(int)

    _, diagnostics, _ = variance_matched_null_subject_confusions(
        DummyHead(), zva, zte, np.arange(12) % 2, np.repeat([1, 2, 3], 4),
        [1, 2, 3], q.T, 3
    )
    assert max(r["source_val_match_abs_error"] for r in diagnostics) < 1e-10
    print("self-test PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=faced.B1_CKPT_ROOT)
    ap.add_argument("--h2000-ckpt-root")
    ap.add_argument("--immutable-manifest")
    ap.add_argument("--closure-go-nogo")
    ap.add_argument("--out-dir", default="results/s2p_route_b_33ch_b1_faced")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--bootstrap-seed", type=int, default=20260710)
    ap.add_argument("--scope", choices=["through_1000", "full"], default="through_1000")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.bootstrap_reps < 1000 or args.null_reps < 50:
        raise ValueError("final verification requires >=1000 bootstraps and >=50 null subspaces")

    selected_tags = TAGS_THROUGH_1000 if args.scope == "through_1000" else TAGS_FULL
    selected_budgets = BUDGETS_THROUGH_1000 if args.scope == "through_1000" else BUDGETS_FULL
    h2000_immutable = {}
    if args.scope == "full":
        if not all([args.h2000_ckpt_root, args.immutable_manifest, args.closure_go_nogo]):
            raise RuntimeError("full scope requires H2000 immutable root, manifest, and closure go/no-go")
        h2000_immutable = load_h2000_immutable_contract(
            args.h2000_ckpt_root, args.immutable_manifest, args.closure_go_nogo
        )
    out = Path(args.out_dir)
    task_rows = read_csv(out / "faced_task_performance.csv")
    l1_rows = read_csv(out / "faced_pairwise_subject_separability.csv")
    l4_rows = read_csv(out / "faced_l4_task_alignment.csv")
    l5_rows = read_csv(out / "faced_l5_replay.csv")
    task = {r["tag"]: r for r in task_rows}
    l1 = {r["tag"]: r for r in l1_rows}
    l4 = {r["tag"]: r for r in l4_rows}
    l5 = {r["tag"]: r for r in l5_rows}
    missing = [tag for tag in selected_tags if tag not in task]
    if missing:
        raise RuntimeError(f"missing committed task rows: {missing}")

    random_gate_threshold = max(0.05, as_float(task["random"], "source_val_kappa") + 0.02)
    pretrained_tags = [tag for tag in selected_tags if tag.startswith("H")]
    if not all(as_bool(task[tag]["task_gate_pass"]) for tag in pretrained_tags):
        raise RuntimeError("not all pretrained FACED cells pass the frozen task gate")
    if any(as_bool(task[tag]["target_labels_used_for_selection"]) for tag in selected_tags):
        raise RuntimeError("target-label firewall violation in committed task table")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    X, y, subject, split, segment_id, item_index, _ = faced.load_faced_lmdb(faced.FACED_LMDB)
    target_mask = split == "target_test"
    val_mask = split == "source_val"
    yte = y[target_mask]
    ste = subject[target_mask]
    target_item = item_index[target_mask]
    subjects = sorted(int(s) for s in np.unique(ste))
    if subjects != list(range(101, 124)):
        raise RuntimeError(f"unexpected FACED test subjects: {subjects}")
    rng = np.random.default_rng(args.bootstrap_seed)
    weights = rng.multinomial(len(subjects), np.full(len(subjects), 1.0 / len(subjects)), size=args.bootstrap_reps)

    base_cm = {}
    removed_cm = {}
    null_cm = {}
    reproduction_rows = []
    prediction_rows = []
    l5_empirical_rows = []

    for tag in selected_tags:
        ckpt = checkpoint_for(tag, args.ckpt_root, args.h2000_ckpt_root)
        if ckpt != "random" and not Path(ckpt).exists():
            raise FileNotFoundError(ckpt)
        print(f"final verification tag={tag} checkpoint={ckpt}", flush=True)
        checkpoint_sha_before = sha256_file(Path(ckpt).resolve()) if tag.startswith("H2000") else None
        if tag.startswith("H2000") and checkpoint_sha_before != h2000_immutable[tag]["sha256"]:
            raise RuntimeError(f"{tag} immutable SHA changed before final verification")
        model, _ = faced.build_encoder(tag, ckpt, device)
        det = faced.deterministic_batch_check(model, X, device)
        feat = faced.extract_features(model, X, device, args.batch_size)
        if tag.startswith("H2000") and sha256_file(Path(ckpt).resolve()) != checkpoint_sha_before:
            raise RuntimeError(f"{tag} immutable SHA changed during final verification")
        pack = fit_fixed_probe(
            feat,
            y,
            split,
            int(float(task[tag]["selected_pca_dim"])),
            float(task[tag]["selected_C"]),
        )
        val_kappa = float(cohen_kappa_score(y[val_mask], pack["pred_val"]))
        val_bacc = float(balanced_accuracy_score(y[val_mask], pack["pred_val"]))
        target_kappa = float(cohen_kappa_score(yte, pack["pred_test"]))
        target_bacc = float(balanced_accuracy_score(yte, pack["pred_test"]))
        max_diff = max(
            abs(val_kappa - as_float(task[tag], "source_val_kappa")),
            abs(val_bacc - as_float(task[tag], "source_val_bacc")),
            abs(target_kappa - as_float(task[tag], "target_kappa")),
            abs(target_bacc - as_float(task[tag], "target_bacc")),
        )
        if max_diff > 1e-8:
            raise RuntimeError(f"{tag} committed metric reproduction failed: max_diff={max_diff}")

        basis, _ = faced.subject_subspace_source_pca(
            pack["ztr"], subject[split == "source_train"], k=5
        )
        pred_removed = pack["clf"].predict(faced.remove_subspace(pack["zte"], basis))
        base_cm[tag] = confusion_by_subject(yte, pack["pred_test"], ste, subjects)
        removed_cm[tag] = confusion_by_subject(yte, pred_removed, ste, subjects)
        historical_rank_null = rank_null_subject_confusions(
            pack["clf"], pack["zte"], yte, ste, subjects, basis.shape[0], 50
        )
        null_cm[tag], null_diagnostics, subject_variance = variance_matched_null_subject_confusions(
            pack["clf"], pack["zva"], pack["zte"], yte, ste, subjects, basis, args.null_reps
        )
        base_point = metric_from_confusion(base_cm[tag].sum(axis=0))
        removed_point = metric_from_confusion(removed_cm[tag].sum(axis=0))
        null_point = metric_from_confusion(null_cm[tag].sum(axis=1))
        null_delta_kappa = float(base_point["kappa"]) - null_point["kappa"]
        null_delta_bacc = float(base_point["bacc"]) - null_point["bacc"]
        subj_delta_kappa = float(base_point["kappa"] - removed_point["kappa"])
        subj_delta_bacc = float(base_point["bacc"] - removed_point["bacc"])
        historical_null_point = metric_from_confusion(historical_rank_null.sum(axis=1))
        historical_null_delta_kappa = float(base_point["kappa"]) - historical_null_point["kappa"]
        historical_null_delta_bacc = float(base_point["bacc"]) - historical_null_point["bacc"]
        original50_checks = {}
        if tag in l5:
            original50_checks = {
                "historical_rank50_null_kappa_mean_diff": float(historical_null_delta_kappa.mean() - as_float(l5[tag], "l5_null_delta_kappa_mean")),
                "historical_rank50_null_bacc_mean_diff": float(historical_null_delta_bacc.mean() - as_float(l5[tag], "l5_null_delta_bacc_mean")),
            }
            if max(abs(v) for v in original50_checks.values()) > 1e-8:
                raise RuntimeError(f"{tag} original L5 null reproduction failed: {original50_checks}")
        l5_empirical_rows.append({
            "tag": tag,
            "budget_h": task[tag].get("budget_h", ""),
            "seed": task[tag].get("seed", ""),
            "task_gate_pass": task[tag].get("task_gate_pass", ""),
            "subject_delta_kappa": subj_delta_kappa,
            "null_delta_kappa_mean": float(null_delta_kappa.mean()),
            "subject_minus_null_kappa": float(subj_delta_kappa - null_delta_kappa.mean()),
            "empirical_one_sided_p_kappa": float((1 + np.sum(null_delta_kappa >= subj_delta_kappa)) / (args.null_reps + 1)),
            "subject_delta_bacc": subj_delta_bacc,
            "null_delta_bacc_mean": float(null_delta_bacc.mean()),
            "subject_minus_null_bacc": float(subj_delta_bacc - null_delta_bacc.mean()),
            "empirical_one_sided_p_bacc": float((1 + np.sum(null_delta_bacc >= subj_delta_bacc)) / (args.null_reps + 1)),
            "n_variance_matched_nulls": args.null_reps,
            "null_control": "source_val_energy_matched_random_orthobasis_partial_last_direction",
            **subject_variance,
            "null_source_val_removed_variance_frac_mean": float(np.mean([r["source_val_removed_variance_frac"] for r in null_diagnostics])),
            "null_source_val_match_abs_error_max": float(max(r["source_val_match_abs_error"] for r in null_diagnostics)),
            "null_target_removed_variance_frac_mean": float(np.mean([r["target_removed_variance_frac"] for r in null_diagnostics])),
            "null_effective_rank_mean": float(np.mean([r["effective_rank"] for r in null_diagnostics])),
            "null_effective_rank_min": int(min(r["effective_rank"] for r in null_diagnostics)),
            "null_effective_rank_max": int(max(r["effective_rank"] for r in null_diagnostics)),
            **original50_checks,
        })
        reproduction_rows.append({
            "tag": tag,
            "checkpoint": ckpt,
            "deterministic_repeat_max_abs": det,
            "selected_pca_dim_reused": int(float(task[tag]["selected_pca_dim"])),
            "selected_C_reused": float(task[tag]["selected_C"]),
            "source_val_kappa_recomputed": val_kappa,
            "target_kappa_recomputed": target_kappa,
            "target_bacc_recomputed": target_bacc,
            "max_committed_metric_abs_diff": max_diff,
            "reproduction_pass": max_diff <= 1e-8 and det <= 1e-6,
            "checkpoint_sha256": checkpoint_sha_before or "",
            "immutable_checkpoint": tag.startswith("H2000"),
        })
        for i in range(len(yte)):
            prediction_rows.append({
                "tag": tag,
                "target_subject": int(ste[i]),
                "target_item_index": int(target_item[i]),
                "target_label_final_scoring_only": int(yte[i]),
                "prediction_base": int(pack["pred_test"][i]),
                "prediction_subject_removed": int(pred_removed[i]),
            })
        del model, feat, pack
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_boot = {tag: bootstrap_metrics(base_cm[tag], weights) for tag in selected_tags}
    removed_boot = {tag: bootstrap_metrics(removed_cm[tag], weights) for tag in selected_tags}
    point_metrics = {tag: metric_from_confusion(base_cm[tag].sum(axis=0)) for tag in selected_tags}
    budget_series = {}
    for budget in selected_budgets:
        tags = [f"H{budget}_s0", f"H{budget}_s1"]
        budget_series[budget] = {}
        for metric in ["kappa", "bacc", "weighted_f1"]:
            samples = np.mean([base_boot[tag][metric] for tag in tags], axis=0)
            point = float(np.mean([point_metrics[tag][metric] for tag in tags]))
            budget_series[budget][metric] = {"point": point, "samples": samples}

    uncertainty_rows = []
    for tag in selected_tags:
        for metric in ["kappa", "bacc", "weighted_f1"]:
            rec = {
                "scope": "checkpoint",
                "tag": tag,
                "budget_h": task[tag].get("budget_h", ""),
                "seed": task[tag].get("seed", ""),
                "metric": metric,
                "n_test_subject_clusters": len(subjects),
                "bootstrap_reps": args.bootstrap_reps,
            }
            rec.update(summarize(point_metrics[tag][metric], base_boot[tag][metric]))
            uncertainty_rows.append(rec)
    for budget in selected_budgets:
        for metric in ["kappa", "bacc", "weighted_f1"]:
            rec = {
                "scope": "budget_seed_mean",
                "tag": f"H{budget}_mean",
                "budget_h": budget,
                "seed": "mean_0_1",
                "metric": metric,
                "n_test_subject_clusters": len(subjects),
                "bootstrap_reps": args.bootstrap_reps,
            }
            rec.update(summarize(
                budget_series[budget][metric]["point"], budget_series[budget][metric]["samples"]
            ))
            uncertainty_rows.append(rec)

    comparison_rows = []
    for budget in selected_budgets:
        for reference in ["random", "released"]:
            for metric in ["kappa", "bacc"]:
                point = budget_series[budget][metric]["point"] - float(point_metrics[reference][metric])
                samples = budget_series[budget][metric]["samples"] - base_boot[reference][metric]
                rec = {
                    "scope": "budget_seed_mean",
                    "budget_h": budget,
                    "metric": metric,
                    "reference": reference,
                    "floor_margin": 0.02 if reference == "random" else "",
                    "point_crosses_random_plus_0p02": bool(point >= 0.02) if reference == "random" else "",
                    "bootstrap_probability_delta_positive": float((samples > 0).mean()),
                    "bootstrap_probability_delta_ge_0p02": float((samples >= 0.02).mean()) if reference == "random" else "",
                    "single_released_checkpoint_descriptive": reference == "released",
                }
                rec.update({f"delta_{k}": v for k, v in summarize(point, samples).items()})
                comparison_rows.append(rec)
    for tag in pretrained_tags:
        for metric in ["kappa", "bacc"]:
            point = float(point_metrics[tag][metric] - point_metrics["random"][metric])
            samples = base_boot[tag][metric] - base_boot["random"][metric]
            rec = {
                "scope": "checkpoint",
                "tag": tag,
                "budget_h": int(float(task[tag]["budget_h"])),
                "seed": int(float(task[tag]["seed"])),
                "metric": metric,
                "reference": "random",
                "floor_margin": 0.02,
                "point_crosses_random_plus_0p02": bool(point >= 0.02),
                "bootstrap_probability_delta_positive": float((samples > 0).mean()),
                "bootstrap_probability_delta_ge_0p02": float((samples >= 0.02).mean()),
                "single_released_checkpoint_descriptive": False,
            }
            rec.update({f"delta_{k}": v for k, v in summarize(point, samples).items()})
            comparison_rows.append(rec)

    l5_cluster_rows = []
    l5_series = {}
    empirical_by_tag = {r["tag"]: r for r in l5_empirical_rows}
    for tag in pretrained_tags:
        l5_series[tag] = {}
        for metric in ["kappa", "bacc"]:
            base = base_boot[tag][metric]
            removed = removed_boot[tag][metric]
            null_mean_metric = null_bootstrap_mean(null_cm[tag], weights, metric)
            subject_delta = base - removed
            null_delta = base - null_mean_metric
            contrast = subject_delta - null_delta
            point = empirical_by_tag[tag][f"subject_minus_null_{metric}"]
            rec = {
                "scope": "checkpoint",
                "tag": tag,
                "budget_h": int(float(task[tag]["budget_h"])),
                "seed": int(float(task[tag]["seed"])),
                "metric": metric,
                "task_gate_pass": True,
                "empirical_one_sided_p": empirical_by_tag[tag][f"empirical_one_sided_p_{metric}"],
                "subject_delta_point": empirical_by_tag[tag][f"subject_delta_{metric}"],
                "null_delta_point": empirical_by_tag[tag][f"null_delta_{metric}_mean"],
            }
            rec.update({f"subject_minus_null_{k}": v for k, v in summarize(point, contrast).items()})
            rec["subject_intervention_exceeds_null"] = bool(
                rec["empirical_one_sided_p"] < 0.05 and rec["subject_minus_null_ci95_low"] > 0
            )
            l5_cluster_rows.append(rec)
            l5_series[tag][metric] = {
                "subject_delta": subject_delta,
                "null_delta": null_delta,
                "contrast": contrast,
            }
    for budget in selected_budgets:
        tags = [f"H{budget}_s0", f"H{budget}_s1"]
        for metric in ["kappa", "bacc"]:
            contrast = np.mean([l5_series[tag][metric]["contrast"] for tag in tags], axis=0)
            point = float(np.mean([empirical_by_tag[tag][f"subject_minus_null_{metric}"] for tag in tags]))
            rec = {
                "scope": "budget_seed_mean",
                "tag": f"H{budget}_mean",
                "budget_h": budget,
                "seed": "mean_0_1",
                "metric": metric,
                "task_gate_pass": True,
                "empirical_one_sided_p": "",
                "subject_delta_point": float(np.mean([empirical_by_tag[tag][f"subject_delta_{metric}"] for tag in tags])),
                "null_delta_point": float(np.mean([empirical_by_tag[tag][f"null_delta_{metric}_mean"] for tag in tags])),
            }
            rec.update({f"subject_minus_null_{k}": v for k, v in summarize(point, contrast).items()})
            rec["subject_intervention_exceeds_null"] = bool(rec["subject_minus_null_ci95_low"] > 0)
            l5_cluster_rows.append(rec)

    for metric in ["kappa", "bacc"]:
        checkpoint_rows = [
            r for r in l5_cluster_rows
            if r["scope"] == "checkpoint" and r["metric"] == metric
        ]
        adjusted = holm_adjust([r["empirical_one_sided_p"] for r in checkpoint_rows])
        for row, p_adjusted in zip(checkpoint_rows, adjusted):
            row["holm_adjusted_empirical_p"] = float(p_adjusted)
            row["subject_intervention_exceeds_null"] = bool(
                p_adjusted < 0.05 and row["subject_minus_null_ci95_low"] > 0
            )
    for row in l5_cluster_rows:
        if row["scope"] == "budget_seed_mean":
            row["holm_adjusted_empirical_p"] = ""
            row["subject_intervention_exceeds_null"] = "descriptive_not_cellwise_tested"

    response = {
        metric: regression_summary(budget_series, metric, selected_budgets)
        for metric in ["kappa", "bacc"]
    }
    response["analysis_scope"] = "descriptive_budget_response_with_target_subject_cluster_bootstrap"
    response["n_budget_levels"] = len(selected_budgets)
    response["training_seeds_per_budget"] = 2
    response["monotonic_scaling_established"] = False

    mechanism_rows = []
    l5_kappa = {(r["scope"], r["tag"]): r for r in l5_cluster_rows if r["metric"] == "kappa"}
    for tag in pretrained_tags:
        key = ("checkpoint", tag)
        mechanism_rows.append({
            "scope": "checkpoint",
            "tag": tag,
            "budget_h": int(float(task[tag]["budget_h"])),
            "seed": int(float(task[tag]["seed"])),
            "source_val_kappa": as_float(task[tag], "source_val_kappa"),
            "task_gate_threshold": random_gate_threshold,
            "task_gate_pass": True,
            "l1_subject_bacc": as_float(l1[tag], "l1_pairwise_subject_bacc_mean"),
            "l4_task_head_subject_energy": as_float(l4[tag], "l4_task_head_subject_subspace_energy"),
            "l5_subject_delta_kappa": l5_kappa[key]["subject_delta_point"],
            "l5_null_delta_kappa": l5_kappa[key]["null_delta_point"],
            "l5_subject_minus_null_kappa": l5_kappa[key]["subject_minus_null_point"],
            "l5_subject_minus_null_ci95_low": l5_kappa[key]["subject_minus_null_ci95_low"],
            "l5_subject_minus_null_ci95_high": l5_kappa[key]["subject_minus_null_ci95_high"],
            "l5_subject_intervention_exceeds_null": l5_kappa[key]["subject_intervention_exceeds_null"],
            "l6_target_kappa_consequence": l5_kappa[key]["subject_delta_point"],
        })

    final_table = []
    random_k = float(point_metrics["random"]["kappa"])
    random_b = float(point_metrics["random"]["bacc"])
    table_rows = [("Random", None)] + [(f"{h}h", h) for h in selected_budgets] + [("Released", None)]
    for label, budget in table_rows:
        if label == "Random":
            k_point, k_samples = random_k, base_boot["random"]["kappa"]
            b_point, b_samples = random_b, base_boot["random"]["bacc"]
            l1_value, gate, l5_value = as_float(l1["random"], "l1_pairwise_subject_bacc_mean"), "reference", ""
        elif label == "Released":
            k_point, k_samples = float(point_metrics["released"]["kappa"]), base_boot["released"]["kappa"]
            b_point, b_samples = float(point_metrics["released"]["bacc"]), base_boot["released"]["bacc"]
            l1_value, gate, l5_value = as_float(l1["released"], "l1_pairwise_subject_bacc_mean"), "reference", ""
        else:
            k_point, k_samples = budget_series[budget]["kappa"]["point"], budget_series[budget]["kappa"]["samples"]
            b_point, b_samples = budget_series[budget]["bacc"]["point"], budget_series[budget]["bacc"]["samples"]
            tags = [f"H{budget}_s0", f"H{budget}_s1"]
            l1_value = float(np.mean([as_float(l1[tag], "l1_pairwise_subject_bacc_mean") for tag in tags]))
            gate = "2/2 PASS"
            l5_value = l5_kappa[("budget_seed_mean", f"H{budget}_mean")]["subject_minus_null_point"]
        k_summary = summarize(k_point, k_samples)
        b_summary = summarize(b_point, b_samples)
        final_table.append({
            "row": label,
            "budget_h": "" if budget is None else budget,
            "target_kappa_mean": k_point,
            "target_kappa_ci95_low": k_summary["ci95_low"],
            "target_kappa_ci95_high": k_summary["ci95_high"],
            "target_bacc_mean": b_point,
            "target_bacc_ci95_low": b_summary["ci95_low"],
            "target_bacc_ci95_high": b_summary["ci95_high"],
            "delta_kappa_vs_random": "" if label == "Random" else k_point - random_k,
            "l1_subject_bacc": l1_value,
            "task_gate": gate,
            "l5_subject_minus_variance_null_kappa": l5_value,
        })

    write_csv(out / "faced_final_reproduction_check.csv", reproduction_rows)
    write_csv(out / "faced_final_target_predictions.csv", prediction_rows)
    write_csv(out / "faced_final_subject_cluster_uncertainty.csv", uncertainty_rows)
    write_csv(out / "faced_final_paired_comparisons.csv", comparison_rows)
    write_csv(out / "faced_final_l5_null_sensitivity.csv", l5_empirical_rows)
    write_csv(out / "faced_final_l5_clustered_inference.csv", l5_cluster_rows)
    write_csv(out / "faced_final_task_gated_mechanism.csv", mechanism_rows)
    write_csv(out / "faced_final_results_table.csv", final_table)
    write_json(out / "faced_final_budget_response.json", response)

    budget_floor = {
        str(h): {
            metric: {
                "delta_vs_random": float(budget_series[h][metric]["point"] - point_metrics["random"][metric]),
                "point_crosses_random_plus_0p02": bool(budget_series[h][metric]["point"] >= point_metrics["random"][metric] + 0.02),
            }
            for metric in ["kappa", "bacc"]
        }
        for h in selected_budgets
    }
    any_l5_exceeds = any(
        r["subject_intervention_exceeds_null"]
        for r in l5_cluster_rows
        if r["scope"] == "checkpoint" and r["metric"] == "kappa"
    )
    verification = {
        "phase": "D2_final_confirmatory_verification",
        "scope": args.scope,
        "downstream_dataset": "FACED",
        "protocol": "frozen_encoder_source_only_probe",
        "all_committed_metrics_reproduced": all(r["reproduction_pass"] for r in reproduction_rows),
        "n_target_subject_clusters": len(subjects),
        "cluster_bootstrap_reps": args.bootstrap_reps,
        "variance_matched_null_reps": args.null_reps,
        "n_evaluated_pretrained_cells": len(pretrained_tags),
        "all_evaluated_pretrained_cells_task_gate_pass": True,
        "task_gate_threshold_source_val_kappa": random_gate_threshold,
        "any_pretrained_l5_subject_intervention_exceeds_variance_null": any_l5_exceeds,
        "variance_null_contract": "source_val_energy_matched_random_orthobasis_partial_last_direction",
        "l5_confirmatory_metric": "cohen_kappa",
        "l5_multiple_comparison_control": f"Holm across {len(pretrained_tags)} evaluated pretrained checkpoints",
        "budget_floor": budget_floor,
        "subject_separability_near_ceiling": min(
            as_float(l1[tag], "l1_pairwise_subject_bacc_mean") for tag in pretrained_tags
        ) >= 0.95,
        "monotonic_scaling_established": False,
        "released_comparison_is_descriptive_single_checkpoint": True,
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_scoring_and_clustered_uncertainty_only": True,
        "fine_tuning_used": False,
        "new_pretraining_used": False,
        "h2000_included": args.scope == "full",
        "h2000_immutable_checkpoints": args.scope == "full",
        "h2000_checkpoint_sha256_by_tag": (
            {tag: row["sha256"] for tag, row in h2000_immutable.items()}
            if args.scope == "full" else {}
        ),
        "h2000_confirmatory_exclusion_reason": (
            None if args.scope == "full"
            else "D2-2 checkpoint mutated after audit; current H2000 runs incomplete at epochs 29/31"
        ),
        "h4000_included": False,
        "codebrain_used": False,
        "claim_status": (
            "task_gated_subject_subspace_effect_exceeds_variance_matched_null_in_at_least_one_cell"
            if any_l5_exceeds
            else "task_gated_subject_subspace_effect_does_not_exceed_variance_matched_null"
        ),
    }
    write_json(out / "faced_final_verification.json", verification)
    write_json(out / "faced_final_target_label_firewall.json", {
        "target_labels_used_for_selection": False,
        "target_labels_used_for_pca": False,
        "target_labels_used_for_head_fit": False,
        "target_labels_used_for_head_selection": False,
        "target_labels_used_for_subject_subspace": False,
        "target_labels_used_for_rank_or_null_selection": False,
        "target_labels_used_for_final_scoring": True,
        "target_labels_used_for_subject_cluster_bootstrap_scoring": True,
        "bootstrap_resampling_unit": "FACED target subject",
        "bootstrap_seed": args.bootstrap_seed,
    })
    print(json.dumps(verification, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
