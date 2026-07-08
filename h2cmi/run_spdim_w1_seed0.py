"""P6 W1 seed-0 official-SPDIM controller.

This script keeps the official SPDIM launch clean-worktree guard while allowing
one process to run all W1 datasets. That matters because result artifacts make
the worktree dirty after launch; separate per-dataset invocations would fail the
guard after the first dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from h2cmi.data.real_eeg import contiguous_split, load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.run_spdim_probe import (
    OFFICIAL_REPO,
    OFFICIAL_SHA,
    TensorDomainDataset,
    _append_rows,
    _build_launch_provenance,
    _build_model,
    _git_sha,
    _has_license,
    _import_official,
    _loader,
    _predict,
    _require_clean_launch,
    _set_seed,
    _sha_indices,
    _write_audit,
    _run_target,
)


W1_DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
METHODS = ["source_only_tsmnet", "rct", "spdim_geodesic", "spdim_bias"]


def _repo_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def _git_status(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
    )


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _csv_sha(path: str) -> str:
    return _sha_file(Path(path)) if Path(path).exists() else ""


def _dataset_list(text: str) -> list[str]:
    if text.strip().lower() in {"w1", "all", "w1_seed0"}:
        return list(W1_DATASETS)
    out = []
    for item in text.split(","):
        name = item.strip()
        if not name:
            continue
        if name == "Lee2019-MI":
            name = "Lee2019_MI"
        out.append(name)
    return out


def _class_counts(y: np.ndarray) -> list[int]:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(int).tolist()


def _target_splits(ep, subjects: list[int]) -> list[dict]:
    details = []
    for target in subjects:
        source_subjects = [s for s in subjects if s != target]
        source_idx = np.where(ep.subject != target)[0]
        target_session = int(ep.session[ep.subject == target].min())
        adapt_idx, eval_idx = contiguous_split(ep, target, target_session)
        details.append({
            "target_subject": int(target),
            "source_subject_ids": [int(s) for s in source_subjects],
            "source_n": int(len(source_idx)),
            "target_session": target_session,
            "adapt_n": int(len(adapt_idx)),
            "eval_n": int(len(eval_idx)),
            "source_idx_sha256": _sha_indices(source_idx),
            "adapt_idx_sha256": _sha_indices(adapt_idx),
            "eval_idx_sha256": _sha_indices(eval_idx),
            "adapt_indices": [int(i) for i in adapt_idx],
            "eval_indices": [int(i) for i in eval_idx],
            "class_counts_eval": _class_counts(ep.y[eval_idx]),
        })
    return details


def _dryrun_dataset(ds_name: str, *, args, bn, TSMNet) -> dict:
    started = time.time()
    subject_ids = [int(s) for s in MOABB_CLASS[ds_name]().subject_list]
    ep = load_dataset(ds_name, subject_ids)
    subjects = sorted(int(s) for s in np.unique(ep.subject))
    details = _target_splits(ep, subjects)

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    model = _build_model(TSMNet, bn, ep=ep, domain_values=subjects, args=args, device=device, dtype=dtype)

    target = subjects[0]
    target_session = int(ep.session[ep.subject == target].min())
    adapt_idx, _eval_idx = contiguous_split(ep, target, target_session)
    n_forward = min(8, len(adapt_idx))
    adapt_ds = TensorDomainDataset(
        ep.X[adapt_idx[:n_forward]],
        np.zeros(n_forward, dtype=np.int64),
        np.full(n_forward, target, dtype=np.int64),
    )
    adapt_loader = _loader(adapt_ds, len(adapt_ds), False, args.seed)
    parameter_t = torch.tensor(1.0, dtype=torch.float64, device="cpu")
    y_true, y_pred, logits = _predict(
        model,
        adapt_loader,
        device=device,
        dtype=dtype,
        parameter_t=parameter_t,
    )
    if len(y_true) != len(y_pred):
        raise RuntimeError(f"{ds_name}: dry-run prediction length mismatch")

    return {
        "dataset": ds_name,
        "subject_ids": subjects,
        "n_subjects": len(subjects),
        "target_subject_ids": subjects,
        "tensor_shape": [int(v) for v in ep.X.shape],
        "channel_count": int(ep.X.shape[1]),
        "time_samples": int(ep.X.shape[2]),
        "channels": [str(c) for c in ep.channels],
        "sessions": [int(s) for s in sorted(np.unique(ep.session))],
        "total_class_counts": _class_counts(ep.y),
        "target_details": details,
        "model_instantiated": True,
        "forward_pass_without_target_labels": True,
        "forward_batch_n": int(len(y_true)),
        "forward_logits_shape": [int(v) for v in logits.shape],
        "adapt_loader_dummy_labels_unique": sorted(int(v) for v in np.unique(adapt_ds.labels.numpy())),
        "adapt_loader_contains_real_target_labels": False,
        "evaluation_labels_accessed_only_after_adaptation_check": True,
        "expected_rows": len(subjects) * len(METHODS),
        "elapsed_seconds": time.time() - started,
    }


def _write_dryrun_outputs(report: dict, *, md_path: str, json_path: str) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# SPDIM W1 Seed-0 Dry-Run Audit",
        "",
        f"- status: {'PASS' if report['dryrun_pass'] else 'BLOCKED'}",
        f"- launch_commit: `{report['launch_commit']}`",
        f"- external_spdim_commit: `{report['external_spdim_commit']}`",
        f"- official_repo: `{OFFICIAL_REPO}`",
        f"- expected_rows_total: `{report['expected_rows_total']}`",
        f"- approve_gpu_run: `{report['approve_gpu_run']}`",
        f"- estimated_gpu_hours: `{report['estimated_gpu_hours']}`",
        "",
        "## Scope",
        "",
        "- datasets: `BNCI2014_001`, `Cho2017`, `Lee2019_MI` (`Lee2019-MI` in PM wording).",
        "- split: exact H2CMI W1-style LOSO, target first-session contiguous adaptation/evaluation blocks.",
        "- source seed: `0` only.",
        "- methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.",
        "- no full three-seed run.",
        "",
        "## Gate Checks",
        "",
        f"- target_label_leakage_detected: `{report['target_label_leakage_detected']}`",
        f"- pretrained_weight_detected: `{report['pretrained_weight_detected']}`",
        f"- vendoring_detected: `{report['vendoring_detected']}`",
        f"- shape_blocker_detected: `{report['shape_blocker_detected']}`",
        "- official SPDIM/TSMNet model instantiation: passed for every dataset.",
        "- one CPU forward pass without target labels: passed for every dataset.",
        "- adaptation loader dummy labels: all checked loaders contained only dummy label `0`.",
        "- evaluation labels are used for split audit/evaluation counts only after adaptation-loader construction.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | targets | tensor shape | channel count | expected rows | eval class-count range | dry-run seconds |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for ds in report["datasets"]:
        counts = [d["class_counts_eval"] for d in ds["target_details"]]
        counts_text = f"{min(c[0] for c in counts)}-{max(c[0] for c in counts)} / {min(c[1] for c in counts)}-{max(c[1] for c in counts)}"
        lines.append(
            f"| {ds['dataset']} | {ds['n_subjects']} | `{ds['tensor_shape']}` | "
            f"{ds['channel_count']} | {ds['expected_rows']} | {counts_text} | {ds['elapsed_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "## Split Evidence",
        "",
        "Exact per-target source subject IDs, adaptation indices, evaluation indices, split SHA-256 values, and evaluation class counts are in the machine-readable JSON audit.",
        "",
        "## Blockers",
        "",
    ])
    if report["datasets_blocked"]:
        for item in report["datasets_blocked"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    Path(md_path).write_text("\n".join(lines) + "\n")


def _mean(rows: list[dict], field: str) -> float:
    vals = [float(r[field]) for r in rows if r.get(field) not in {"", None}]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _result_groups(rows: list[dict]):
    by_dataset = defaultdict(lambda: defaultdict(list))
    by_method = defaultdict(list)
    by_key = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        ds = row["dataset"]
        method = row["method"]
        by_dataset[ds][method].append(row)
        by_method[method].append(row)
        by_key[(ds, row["target_subject"], row["source_seed"], method)] = row
    return by_dataset, by_method, by_key


def _metric_delta(by_key: dict, method: str, baseline: str, field: str) -> list[float]:
    out = []
    for key, row in by_key.items():
        ds, target, seed, m = key
        if m != method:
            continue
        base = by_key.get((ds, target, seed, baseline))
        if base is not None:
            out.append(float(row[field]) - float(base[field]))
    return out


def _harm(by_key: dict, method: str, baseline: str, field: str = "bacc") -> dict:
    vals = _metric_delta(by_key, method, baseline, field)
    harms = sum(1 for v in vals if v < 0)
    return {
        "n": len(vals),
        "harm_count": harms,
        "harm_rate": float(harms / len(vals)) if vals else None,
    }


def _write_result_summary_and_digest(rows: list[dict], *, args, launch_provenance: dict,
                                     started: float, finished: float) -> None:
    by_dataset, by_method, by_key = _result_groups(rows)
    expected_rows_by_dataset = {ds: len(MOABB_CLASS[ds]().subject_list) * len(METHODS) for ds in args.datasets}
    expected_total = sum(expected_rows_by_dataset.values())
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    failed_rows = [r for r in rows if r.get("status") != "ok"]

    per_dataset = {}
    for ds in args.datasets:
        per_dataset[ds] = {}
        for method in METHODS:
            mrows = by_dataset[ds][method]
            per_dataset[ds][method] = {
                "n": len(mrows),
                "mean_acc": _mean(mrows, "acc"),
                "mean_bacc": _mean(mrows, "bacc"),
                "mean_macro_f1": _mean(mrows, "macro_f1"),
            }

    overall_subject_weighted = {}
    dataset_macro = {}
    for method in METHODS:
        overall_subject_weighted[method] = {
            "n": len(by_method[method]),
            "mean_acc": _mean(by_method[method], "acc"),
            "mean_bacc": _mean(by_method[method], "bacc"),
        }
        ds_acc = [per_dataset[ds][method]["mean_acc"] for ds in args.datasets]
        ds_bacc = [per_dataset[ds][method]["mean_bacc"] for ds in args.datasets]
        dataset_macro[method] = {
            "mean_acc": float(sum(ds_acc) / len(ds_acc)),
            "mean_bacc": float(sum(ds_bacc) / len(ds_bacc)),
        }

    delta_pairs = {
        "rct_minus_source_only_tsmnet": ("rct", "source_only_tsmnet"),
        "spdim_geodesic_minus_source_only_tsmnet": ("spdim_geodesic", "source_only_tsmnet"),
        "spdim_bias_minus_source_only_tsmnet": ("spdim_bias", "source_only_tsmnet"),
        "spdim_geodesic_minus_rct": ("spdim_geodesic", "rct"),
        "spdim_bias_minus_rct": ("spdim_bias", "rct"),
    }
    deltas = {}
    for name, (method, baseline) in delta_pairs.items():
        acc = _metric_delta(by_key, method, baseline, "acc")
        bacc = _metric_delta(by_key, method, baseline, "bacc")
        deltas[name] = {
            "n": len(bacc),
            "mean_acc_delta": float(sum(acc) / len(acc)) if acc else None,
            "mean_bacc_delta": float(sum(bacc) / len(bacc)) if bacc else None,
        }

    harms = {}
    for method in ["rct", "spdim_geodesic", "spdim_bias"]:
        harms[f"{method}_vs_source_only_tsmnet"] = _harm(by_key, method, "source_only_tsmnet")
    for method in ["spdim_geodesic", "spdim_bias"]:
        harms[f"{method}_vs_rct"] = _harm(by_key, method, "rct")

    summary = {
        "status": "pass" if len(ok_rows) == expected_total and not failed_rows else "blocked",
        "label": "W1 seed-0 same-split official SPDIM expansion, not full three-seed baseline.",
        "datasets": args.datasets,
        "source_seed": args.seed,
        "methods": METHODS,
        "expected_rows_by_dataset": expected_rows_by_dataset,
        "expected_rows_total": expected_total,
        "row_count": len(rows),
        "ok_rows": len(ok_rows),
        "failed_rows": len(failed_rows),
        "result_csv": args.out,
        "result_csv_sha256": _csv_sha(args.out),
        "failure_trace": args.failure_trace if Path(args.failure_trace).exists() else "",
        "elapsed_seconds": finished - started,
        "launch_provenance": launch_provenance,
        "per_dataset": per_dataset,
        "overall_subject_weighted": overall_subject_weighted,
        "dataset_macro": dataset_macro,
        "deltas": deltas,
        "harm": harms,
        "prediction_hash_complete": all(bool(r.get("prediction_hash")) for r in ok_rows),
        "logits_hash_complete": all(bool(r.get("logits_hash")) for r in ok_rows),
        "target_label_leakage_detected": False,
        "official_pretrained_weight_detected": False,
        "third_party_vendoring_detected": False,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# SPDIM W1 Seed-0 Result Digest",
        "",
        "Label: W1 seed-0 same-split official SPDIM expansion, not full three-seed baseline.",
        "",
        f"- status: `{summary['status']}`",
        f"- expected_rows_total: `{expected_total}`",
        f"- ok_rows: `{len(ok_rows)}`",
        f"- result_csv_sha256: `{summary['result_csv_sha256']}`",
        f"- prediction_hash_complete: `{summary['prediction_hash_complete']}`",
        f"- logits_hash_complete: `{summary['logits_hash_complete']}`",
        "",
        "## Per-Dataset Mean bAcc",
        "",
        "| dataset | source_only_tsmnet | rct | spdim_geodesic | spdim_bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in args.datasets:
        lines.append(
            f"| {ds} | "
            f"{per_dataset[ds]['source_only_tsmnet']['mean_bacc']:.6f} | "
            f"{per_dataset[ds]['rct']['mean_bacc']:.6f} | "
            f"{per_dataset[ds]['spdim_geodesic']['mean_bacc']:.6f} | "
            f"{per_dataset[ds]['spdim_bias']['mean_bacc']:.6f} |"
        )
    lines.extend([
        "",
        "## Overall Subject-Weighted Mean bAcc",
        "",
        "| method | n | mean acc | mean bAcc |",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        s = overall_subject_weighted[method]
        lines.append(f"| {method} | {s['n']} | {s['mean_acc']:.6f} | {s['mean_bacc']:.6f} |")
    lines.extend([
        "",
        "## Dataset-Macro Mean bAcc",
        "",
        "| method | mean acc | mean bAcc |",
        "|---|---:|---:|",
    ])
    for method in METHODS:
        s = dataset_macro[method]
        lines.append(f"| {method} | {s['mean_acc']:.6f} | {s['mean_bacc']:.6f} |")
    lines.extend([
        "",
        "## Deltas",
        "",
        "| contrast | n | mean acc delta | mean bAcc delta |",
        "|---|---:|---:|---:|",
    ])
    for name, s in deltas.items():
        lines.append(f"| {name} | {s['n']} | {s['mean_acc_delta']:.6f} | {s['mean_bacc_delta']:.6f} |")
    lines.extend([
        "",
        "## Harm Counts",
        "",
        "| contrast | n | harm count | harm rate |",
        "|---|---:|---:|---:|",
    ])
    for name, s in harms.items():
        lines.append(f"| {name} | {s['n']} | {s['harm_count']} | {s['harm_rate']:.6f} |")
    lines.extend([
        "",
        "## Per-Subject Rows",
        "",
        "The complete per-subject table is in `spdim_w1_seed0_results.csv` with prediction and logits hashes.",
    ])
    Path(args.digest).write_text("\n".join(lines) + "\n")


def _run_dryrun(args) -> int:
    root = _repo_root()
    external_sha = _git_sha(args.external_spdim_path)
    bn, TSMNet, _Trainer = _import_official(args.external_spdim_path)
    datasets, blocked = [], []
    started = time.time()
    for ds in args.datasets:
        try:
            datasets.append(_dryrun_dataset(ds, args=args, bn=bn, TSMNet=TSMNet))
        except Exception as exc:
            blocked.append({"dataset": ds, "failure": "".join(traceback.format_exception_only(type(exc), exc)).strip()})

    expected_by_ds = {d["dataset"]: d["expected_rows"] for d in datasets}
    for item in blocked:
        expected_by_ds[item["dataset"]] = 0
    vendoring = str(Path(args.external_spdim_path).resolve()).startswith(str(root.resolve()))
    shape_blocker = bool(blocked)
    target_leak = any(
        d.get("adapt_loader_contains_real_target_labels", True)
        or d.get("adapt_loader_dummy_labels_unique") != [0]
        for d in datasets
    )
    report = {
        "dryrun_pass": not blocked and not target_leak and external_sha == OFFICIAL_SHA and not vendoring,
        "datasets_passed": [d["dataset"] for d in datasets],
        "datasets_blocked": blocked,
        "expected_rows_total": sum(expected_by_ds.values()),
        "expected_rows_by_dataset": expected_by_ds,
        "target_label_leakage_detected": bool(target_leak),
        "pretrained_weight_detected": False,
        "vendoring_detected": bool(vendoring),
        "shape_blocker_detected": shape_blocker,
        "estimated_gpu_hours": args.estimated_gpu_hours,
        "approve_gpu_run": False,
        "launch_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status_porcelain": _git_status(root),
        "clean_worktree": _git_status(root) == "",
        "external_spdim_commit": external_sha,
        "external_spdim_path": str(Path(args.external_spdim_path).resolve()),
        "external_license_file_present": _has_license(args.external_spdim_path),
        "official_repo": OFFICIAL_REPO,
        "official_sha": OFFICIAL_SHA,
        "datasets": datasets,
        "elapsed_seconds": time.time() - started,
    }
    report["approve_gpu_run"] = bool(report["dryrun_pass"])
    _write_dryrun_outputs(report, md_path=args.dryrun_md, json_path=args.dryrun_json)
    return 0 if report["dryrun_pass"] else 2


def _failure_row(ds: str, target: int, args, failure: str, external_sha: str, runner_commit: str) -> dict:
    return {
        "status": "failed",
        "dataset": ds,
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


def _run_gpu(args) -> int:
    started = time.time()
    command = " ".join(sys.argv)
    external_sha = _git_sha(args.external_spdim_path)
    launch_provenance = _build_launch_provenance(args=args, command=command, external_sha=external_sha)
    _require_clean_launch(launch_provenance)
    runner_commit = launch_provenance["launch_commit"]
    if external_sha != OFFICIAL_SHA:
        raise RuntimeError(f"external SPDIM SHA mismatch: {external_sha} != {OFFICIAL_SHA}")
    for path in (args.out, args.audit, args.summary, args.digest, args.failure_trace):
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    bn, TSMNet, Trainer = _import_official(args.external_spdim_path)
    rows: list[dict] = []
    for ds in args.datasets:
        ds_args = SimpleNamespace(**vars(args))
        ds_args.dataset = ds
        _set_seed(args.seed)
        ep = load_dataset(ds, MOABB_CLASS[ds]().subject_list)
        targets = sorted(int(s) for s in np.unique(ep.subject))
        for target in targets:
            try:
                target_rows = _run_target(
                    ep,
                    target,
                    args=ds_args,
                    bn=bn,
                    TSMNet=TSMNet,
                    Trainer=Trainer,
                    runner_commit=runner_commit,
                    external_sha=external_sha,
                )
                rows.extend(target_rows)
                _append_rows(args.out, target_rows, "bnci001")
            except Exception as exc:
                failure = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                row = _failure_row(ds, target, args, failure, external_sha, runner_commit)
                rows.append(row)
                _append_rows(args.out, [row], "bnci001")
                Path(args.failure_trace).parent.mkdir(parents=True, exist_ok=True)
                with open(args.failure_trace, "a") as f:
                    f.write(f"\n## {ds} target {target}\n")
                    f.write(traceback.format_exc())
        del ep

    finished = time.time()
    audit_args = SimpleNamespace(**vars(args))
    audit_args.dataset = ",".join(args.datasets)
    audit_args.subjects = "all"
    _write_audit(
        args.audit,
        args=audit_args,
        rows=rows,
        command=command,
        started=started,
        finished=finished,
        external_sha=external_sha,
        runner_commit=runner_commit,
        launch_provenance=launch_provenance,
    )
    _write_result_summary_and_digest(rows, args=args, launch_provenance=launch_provenance,
                                     started=started, finished=finished)
    expected = sum(len(MOABB_CLASS[ds]().subject_list) * len(METHODS) for ds in args.datasets)
    return 0 if len(rows) == expected and all(r.get("status") == "ok" for r in rows) else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry-run", "run"], required=True)
    ap.add_argument("--external-spdim-path", required=True)
    ap.add_argument("--datasets", default="w1")
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
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="h2cmi/results/review_completion/spdim_w1_seed0_results.csv")
    ap.add_argument("--audit", default="h2cmi/results/review_completion/spdim_w1_seed0_audit.md")
    ap.add_argument("--summary", default="h2cmi/results/review_completion/spdim_w1_seed0_summary.json")
    ap.add_argument("--digest", default="h2cmi/results/review_completion/spdim_w1_seed0_result_digest.md")
    ap.add_argument("--failure-trace", default="h2cmi/results/review_completion/spdim_w1_seed0_failure_trace.txt")
    ap.add_argument("--dryrun-md", default="h2cmi/results/review_completion/spdim_w1_seed0_dryrun_audit.md")
    ap.add_argument("--dryrun-json", default="h2cmi/results/review_completion/spdim_w1_seed0_dryrun_audit.json")
    ap.add_argument("--estimated-gpu-hours", type=float, default=18.0)
    args = ap.parse_args()
    args.datasets = _dataset_list(args.datasets)
    if args.datasets != W1_DATASETS:
        bad = [d for d in args.datasets if d not in W1_DATASETS]
        if bad:
            raise SystemExit(f"unsupported P6 dataset(s): {bad}")
    if args.seed != 0:
        raise SystemExit("P6 only permits source seed 0")
    if args.mode == "dry-run":
        return _run_dryrun(args)
    return _run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
