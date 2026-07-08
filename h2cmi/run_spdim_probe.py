"""Bounded official-SPDIM feasibility probe on frozen H2CMI W1-style splits.

This adapter imports the official external SPDIM checkout via --external-spdim-path.
It does not vendor or reimplement the official SPDIM model/trainer code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from h2cmi.data.real_eeg import contiguous_split, load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import hash_state, require_clean_git

OFFICIAL_REPO = "https://github.com/fightlesliefigt/SPDIM"
OFFICIAL_SHA = "1b0de0ccd4c48a4ff28f087b866a0b671b029c39"


class TensorDomainDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, domains: np.ndarray):
        self.features = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.domains = torch.from_numpy(np.asarray(domains, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return {
            "inputs": self.features[index],
            "domains": self.domains[index],
        }, self.labels[index]


def _sha_indices(idx: np.ndarray) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in idx).encode()).hexdigest()


def _hash_ndarray(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(tuple(x.shape)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _git_sha(path: str) -> str:
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def _has_license(path: str) -> bool:
    return any((Path(path) / name).exists() for name in ("LICENSE", "LICENSE.md", "COPYING"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_train_val_split(indices: np.ndarray, y: np.ndarray, domains: np.ndarray, val_fraction: float):
    train, val = [], []
    for domain in sorted(int(d) for d in np.unique(domains[indices])):
        for cls in sorted(int(c) for c in np.unique(y[indices])):
            group = indices[(domains[indices] == domain) & (y[indices] == cls)]
            if len(group) == 0:
                continue
            n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
            train_group = group[:-n_val] if n_val else group
            val_group = group[-n_val:] if n_val else []
            train.extend(int(i) for i in train_group)
            if n_val:
                val.extend(int(i) for i in val_group)
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _predict(model, loader: DataLoader, *, device, dtype, parameter_t, fm_mean=None):
    model.eval()
    ys, preds, logits_all = [], [], []
    with torch.no_grad():
        for features, y in loader:
            features["inputs"] = features["inputs"].to(device=device, dtype=dtype)
            # Official SPDIM keeps SPD layers on CPU by default; domains must stay CPU.
            logits = model(**features, parameter_t=parameter_t, fm_mean=fm_mean)
            ys.append(y.detach().cpu())
            preds.append(logits.argmax(1).detach().cpu())
            logits_all.append(logits.detach().cpu().float())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    logits_np = torch.cat(logits_all).numpy()
    return y_true, y_pred, logits_np


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _class_counts(y_true: np.ndarray) -> str:
    return json.dumps(np.bincount(y_true, minlength=2).astype(int).tolist(), separators=(",", ":"))


def _prediction_fields(y_true: np.ndarray, y_pred: np.ndarray, logits: np.ndarray) -> dict:
    return {
        "n_eval": int(len(y_true)),
        "class_counts_eval": _class_counts(y_true),
        "prediction_hash": _hash_ndarray(y_pred.astype(np.int64)),
        "logits_hash": _hash_ndarray(logits.astype(np.float32)),
    }


def _result_fieldnames(schema: str) -> list[str]:
    if schema == "bnci001":
        return [
            "dataset", "target_subject", "source_seed", "method", "n_eval",
            "class_counts_eval", "acc", "bacc", "prediction_hash", "logits_hash",
            "status", "failure_reason", "protocol", "official_repo", "official_sha",
            "runner_commit", "source_subjects", "source_n", "target_session", "adapt_n",
            "eval_n", "source_idx_sha256", "adapt_idx_sha256", "eval_idx_sha256",
            "parameter_t", "source_model_sha256", "train_seconds", "adapt_seconds",
            "eval_seconds", "macro_f1",
        ]
    return [
        "status", "dataset", "protocol", "official_repo", "official_sha", "runner_commit",
        "seed", "target_subject", "source_subjects", "source_n", "target_session",
        "adapt_n", "eval_n", "source_idx_sha256", "adapt_idx_sha256", "eval_idx_sha256",
        "mode", "bacc", "acc", "macro_f1", "parameter_t", "source_model_sha256",
        "train_seconds", "adapt_seconds", "eval_seconds", "failure",
    ]


def _append_rows(path: str, rows: list[dict], schema: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(path).exists()
    fieldnames = _result_fieldnames(schema)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_audit(path: str, *, args, rows: list[dict], command: str, started: float, finished: float,
                 external_sha: str, runner_commit: str, dry_run: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    failed_rows = [r for r in rows if r.get("status") != "ok"]
    lines = [
        "# SPDIM Probe Audit",
        "",
        f"- status: {'DRY-RUN PASS' if dry_run else ('PASS' if rows and not failed_rows else 'FAIL')}",
        f"- runner_commit: `{runner_commit}`",
        f"- official_repo: `{OFFICIAL_REPO}`",
        f"- official_sha: `{external_sha}`",
        f"- external_path: `{args.external_spdim_path}`",
        f"- external_license_file_present: `{_has_license(args.external_spdim_path)}`",
        f"- dataset: `{args.dataset}`",
        f"- protocol: `W1-style LOSO, target first-session contiguous_split`",
        f"- subjects: `{args.subjects}`",
        f"- seed: `{args.seed}`",
        f"- source_epochs: `{args.epochs}`",
        f"- adapt_epochs: `{args.adapt_epochs}`",
        f"- device: `{args.device}`",
        f"- elapsed_seconds: `{finished - started:.3f}`",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Target Label Policy",
        "",
        "Target subject IDs are selected from sorted subject metadata only. Target adaptation datasets use dummy labels; target labels are read only by the evaluation metric code after adaptation/refit has completed.",
        "",
        "## Results",
        "",
    ]
    if not rows:
        lines.append("No result rows were produced.")
    else:
        lines.append("| target | mode | status | bAcc | acc | macro-F1 | failure |")
        lines.append("|---:|---|---|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| {row.get('target_subject', '')} | {row.get('mode', '')} | {row.get('status', '')} | "
                f"{row.get('bacc', '')} | {row.get('acc', '')} | {row.get('macro_f1', '')} | {row.get('failure', '')} |"
            )
    lines.extend([
        "",
        "## Gate",
        "",
        f"- ok_rows: `{len(ok_rows)}`",
        f"- failed_rows: `{len(failed_rows)}`",
        f"- expected_ok_rows_for_success: `{len(set(r.get('target_subject') for r in rows)) * 4 if rows else 0}`",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def _import_official(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)
    import spdnets.batchnorm as bn  # noqa: PLC0415
    from spdnets.models import TSMNet  # noqa: PLC0415
    from spdnets.trainer import Trainer  # noqa: PLC0415

    return bn, TSMNet, Trainer


def _build_model(TSMNet, bn, *, ep, domain_values, args, device, dtype):
    return TSMNet(
        temporal_filters=args.temporal_filters,
        spatial_filters=args.spatial_filters,
        subspacedims=args.subspace_dims,
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        nclasses=2,
        nchannels=int(ep.X.shape[1]),
        nsamples=int(ep.X.shape[2]),
        domain_adaptation=True,
        domains=torch.tensor(domain_values, dtype=torch.long),
        spd_device=args.spd_device,
        spd_dtype=torch.double,
    ).to(device=device, dtype=dtype)


def _run_target(ep, target: int, *, args, bn, TSMNet, Trainer, runner_commit: str, external_sha: str):
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    subjects = sorted(int(s) for s in np.unique(ep.subject))
    source_subjects = [s for s in subjects if s != target]
    source_idx = np.where(ep.subject != target)[0]
    target_session = int(ep.session[ep.subject == target].min())
    adapt_idx, eval_idx = contiguous_split(ep, target, target_session)
    domain_values = sorted(source_subjects + [target])
    train_idx, val_idx = _source_train_val_split(source_idx, ep.y, ep.subject, args.val_fraction)

    meta = {
        "status": "ok",
        "dataset": args.dataset,
        "protocol": "W1_LOSO_first_session_contiguous_split",
        "official_repo": OFFICIAL_REPO,
        "official_sha": external_sha,
        "runner_commit": runner_commit,
        "seed": int(args.seed),
        "source_seed": int(args.seed),
        "target_subject": int(target),
        "source_subjects": " ".join(str(s) for s in source_subjects),
        "source_n": int(len(source_idx)),
        "target_session": int(target_session),
        "adapt_n": int(len(adapt_idx)),
        "eval_n": int(len(eval_idx)),
        "source_idx_sha256": _sha_indices(source_idx),
        "adapt_idx_sha256": _sha_indices(adapt_idx),
        "eval_idx_sha256": _sha_indices(eval_idx),
        "failure": "",
        "failure_reason": "",
    }

    train_ds = TensorDomainDataset(ep.X[train_idx], ep.y[train_idx], ep.subject[train_idx])
    val_ds = TensorDomainDataset(ep.X[val_idx], ep.y[val_idx], ep.subject[val_idx])
    # Dummy labels deliberately prevent target labels from entering adaptation loaders.
    adapt_ds = TensorDomainDataset(ep.X[adapt_idx], np.zeros(len(adapt_idx), dtype=np.int64),
                                   np.full(len(adapt_idx), target, dtype=np.int64))
    eval_ds = TensorDomainDataset(ep.X[eval_idx], ep.y[eval_idx],
                                  np.full(len(eval_idx), target, dtype=np.int64))

    train_loader = _loader(train_ds, args.batch_size, True, args.seed + target)
    val_loader = _loader(val_ds, len(val_ds), False, args.seed)
    adapt_loader = _loader(adapt_ds, len(adapt_ds), False, args.seed)
    eval_loader = _loader(eval_ds, len(eval_ds), False, args.seed)

    _set_seed(args.seed)
    model = _build_model(TSMNet, bn, ep=ep, domain_values=domain_values, args=args, device=device, dtype=dtype)
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[],
        loss=torch.nn.CrossEntropyLoss(),
        device=device,
        dtype=dtype,
    )
    parameter_t = torch.tensor(1.0, dtype=torch.float64, device="cpu")
    train_start = time.time()
    trainer.fit(model, train_dataloader=train_loader, val_dataloader=val_loader, parameter_t=parameter_t)
    train_seconds = time.time() - train_start
    source_state = deepcopy(model.state_dict())
    source_model_sha = hash_state(model)

    def fresh_refit():
        m = _build_model(TSMNet, bn, ep=ep, domain_values=domain_values, args=args, device=device, dtype=dtype)
        m.load_state_dict(source_state)
        x_adapt = adapt_ds.features.to(device=device, dtype=dtype)
        dummy_y = adapt_ds.labels.to(device=device)
        d_adapt = adapt_ds.domains
        start = time.time()
        m.eval()
        m.domainadapt_finetune(x_adapt, dummy_y, d_adapt, "refit")
        return m, time.time() - start

    rows = []
    eval_start = time.time()
    y_true, y_pred, logits = _predict(model, eval_loader, device=device, dtype=dtype, parameter_t=parameter_t)
    source_eval_seconds = time.time() - eval_start
    rows.append(meta | _metrics(y_true, y_pred) | _prediction_fields(y_true, y_pred, logits) | {
        "mode": "source_only",
        "method": "source_only_tsmnet",
        "parameter_t": float(parameter_t.item()),
        "source_model_sha256": source_model_sha,
        "train_seconds": train_seconds,
        "adapt_seconds": 0.0,
        "eval_seconds": source_eval_seconds,
    })

    rct_model, rct_seconds = fresh_refit()
    eval_start = time.time()
    y_true, y_pred, logits = _predict(rct_model, eval_loader, device=device, dtype=dtype, parameter_t=parameter_t)
    rows.append(meta | _metrics(y_true, y_pred) | _prediction_fields(y_true, y_pred, logits) | {
        "mode": "rct_refit",
        "method": "rct",
        "parameter_t": float(parameter_t.item()),
        "source_model_sha256": source_model_sha,
        "train_seconds": train_seconds,
        "adapt_seconds": rct_seconds,
        "eval_seconds": time.time() - eval_start,
    })

    geo_model, refit_seconds = fresh_refit()
    geo_start = time.time()
    best_t = trainer.get_information_maximization_geodesic(
        geo_model,
        test_dataloader=adapt_loader,
        parameter_t=torch.tensor(1.0, dtype=torch.float64, device="cpu"),
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
    )
    geo_seconds = refit_seconds + (time.time() - geo_start)
    eval_start = time.time()
    y_true, y_pred, logits = _predict(geo_model, eval_loader, device=device, dtype=dtype, parameter_t=best_t.detach())
    rows.append(meta | _metrics(y_true, y_pred) | _prediction_fields(y_true, y_pred, logits) | {
        "mode": "spdim_geodesic",
        "method": "spdim_geodesic",
        "parameter_t": float(best_t.detach().cpu().item()),
        "source_model_sha256": source_model_sha,
        "train_seconds": train_seconds,
        "adapt_seconds": geo_seconds,
        "eval_seconds": time.time() - eval_start,
    })

    bias_model, refit_seconds = fresh_refit()
    bias_start = time.time()
    # Official bias optimization initializes from model.get_spdnet_data(); set it using target adapt inputs.
    trainer.predict(bias_model, adapt_loader, parameter_t=torch.tensor(1.0, dtype=torch.float64, device="cpu"))
    best_mean = trainer.get_information_maximization_bias(
        bias_model,
        test_dataloader=adapt_loader,
        parameter_t=torch.tensor(1.0, dtype=torch.float64, device="cpu"),
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
    )
    bias_seconds = refit_seconds + (time.time() - bias_start)
    eval_start = time.time()
    y_true, y_pred, logits = _predict(
        bias_model,
        eval_loader,
        device=device,
        dtype=dtype,
        parameter_t=torch.tensor(1.0, dtype=torch.float64, device="cpu"),
        fm_mean=best_mean.detach(),
    )
    rows.append(meta | _metrics(y_true, y_pred) | _prediction_fields(y_true, y_pred, logits) | {
        "mode": "spdim_bias",
        "method": "spdim_bias",
        "parameter_t": 1.0,
        "source_model_sha256": source_model_sha,
        "train_seconds": train_seconds,
        "adapt_seconds": bias_seconds,
        "eval_seconds": time.time() - eval_start,
    })
    return rows


def _dry_run(ep, *, args, bn, TSMNet) -> None:
    subjects = sorted(int(s) for s in np.unique(ep.subject))
    target = subjects[0]
    domain_values = subjects
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    model = _build_model(TSMNet, bn, ep=ep, domain_values=domain_values, args=args, device=device, dtype=dtype)
    idx = np.where(ep.subject != target)[0][: min(8, int((ep.subject != target).sum()))]
    ds = TensorDomainDataset(ep.X[idx], ep.y[idx], ep.subject[idx])
    loader = _loader(ds, len(ds), False, args.seed)
    parameter_t = torch.tensor(1.0, dtype=torch.float64, device="cpu")
    y_true, y_pred, _logits = _predict(model, loader, device=device, dtype=dtype, parameter_t=parameter_t)
    if len(y_true) != len(y_pred):
        raise RuntimeError("dry-run prediction length mismatch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-spdim-path", required=True)
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--subjects", default="1,9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--adapt-epochs", type=int, default=30)
    ap.add_argument("--adapt-lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--temporal-filters", type=int, default=4)
    ap.add_argument("--spatial-filters", type=int, default=40)
    ap.add_argument("--subspace-dims", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--spd-device", default="cpu")
    ap.add_argument("--out", default="h2cmi/results/review_completion/spdim_probe_results.csv")
    ap.add_argument("--audit", default="h2cmi/results/review_completion/spdim_probe_audit.md")
    ap.add_argument("--failure-trace", default="h2cmi/results/review_completion/spdim_probe_failure_trace.txt")
    ap.add_argument("--result-schema", choices=["probe", "bnci001"], default="probe")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    started = time.time()
    command = " ".join(sys.argv)
    rows: list[dict] = []
    runner_commit = "unknown"
    external_sha = "unknown"
    try:
        if args.overwrite and not args.dry_run:
            for path in (args.out, args.audit, args.failure_trace):
                try:
                    Path(path).unlink()
                except FileNotFoundError:
                    pass
        runner_commit = require_clean_git(
            allow_dirty=args.allow_dirty,
            ignore_prefixes=["results/h2cmi", "h2cmi/results/review_completion"],
        )
        external_sha = _git_sha(args.external_spdim_path)
        if external_sha != OFFICIAL_SHA:
            raise RuntimeError(f"external SPDIM SHA mismatch: {external_sha} != {OFFICIAL_SHA}")
        bn, TSMNet, Trainer = _import_official(args.external_spdim_path)
        _set_seed(args.seed)
        ep = load_dataset(args.dataset, MOABB_CLASS(args.dataset)().subject_list)
        if args.dry_run:
            _dry_run(ep, args=args, bn=bn, TSMNet=TSMNet)
            _write_audit(
                args.audit,
                args=args,
                rows=[],
                command=command,
                started=started,
                finished=time.time(),
                external_sha=external_sha,
                runner_commit=runner_commit,
                dry_run=True,
            )
            return 0
        if args.subjects.strip().lower() == "all":
            targets = sorted(int(s) for s in np.unique(ep.subject))
        else:
            targets = [int(s) for s in args.subjects.split(",") if s]
        for target in targets:
            try:
                target_rows = _run_target(
                    ep,
                    target,
                    args=args,
                    bn=bn,
                    TSMNet=TSMNet,
                    Trainer=Trainer,
                    runner_commit=runner_commit,
                    external_sha=external_sha,
                )
                rows.extend(target_rows)
                _append_rows(args.out, target_rows, args.result_schema)
            except Exception as exc:  # fail loud but preserve per-target trace rows
                failure = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                row = {
                    "status": "failed",
                    "dataset": args.dataset,
                    "protocol": "W1_LOSO_first_session_contiguous_split",
                    "official_repo": OFFICIAL_REPO,
                    "official_sha": external_sha,
                    "runner_commit": runner_commit,
                    "seed": int(args.seed),
                    "source_seed": int(args.seed),
                    "target_subject": int(target),
                    "mode": "target_failed",
                    "method": "target_failed",
                    "failure": failure,
                    "failure_reason": failure,
                }
                rows.append(row)
                _append_rows(args.out, [row], args.result_schema)
                Path(args.failure_trace).parent.mkdir(parents=True, exist_ok=True)
                with open(args.failure_trace, "a") as f:
                    f.write(f"\n## target {target}\n")
                    f.write(traceback.format_exc())
        _write_audit(
            args.audit,
            args=args,
            rows=rows,
            command=command,
            started=started,
            finished=time.time(),
            external_sha=external_sha,
            runner_commit=runner_commit,
        )
        return 0 if rows and all(r.get("status") == "ok" for r in rows) and len(rows) == len(targets) * 4 else 2
    except Exception:
        Path(args.failure_trace).parent.mkdir(parents=True, exist_ok=True)
        Path(args.failure_trace).write_text(traceback.format_exc())
        _write_audit(
            args.audit,
            args=args,
            rows=rows,
            command=command,
            started=started,
            finished=time.time(),
            external_sha=external_sha,
            runner_commit=runner_commit,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
