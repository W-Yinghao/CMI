"""P8 official SPDIM W1 repaired-split seed-0 controller.

This runner imports the external official SPDIM checkout and uses the frozen P7
class-stratified repaired W1 split manifest. Runtime adaptation receives target
features with dummy labels only; target labels are used by this script for split
audit and final evaluation metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from h2cmi.data.real_eeg import load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import hash_state
from h2cmi.run_spdim_probe import (
    OFFICIAL_REPO,
    OFFICIAL_SHA,
    TensorDomainDataset,
    _build_model,
    _git_sha,
    _has_license,
    _import_official,
    _loader,
    _metrics,
    _predict,
    _prediction_fields,
    _set_seed,
    _sha_indices,
    _source_train_val_split,
)
from h2cmi.w1_repaired_split import (
    SPLIT_FAMILY,
    indices_from_trial_ids,
    load_manifest_csv,
    manifest_hash,
    sha256_file,
)


W1_DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
METHODS = ["source_only_tsmnet", "rct", "spdim_geodesic", "spdim_bias"]
P7_MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
EXPECTED_ROWS_BY_DATASET = {"BNCI2014_001": 36, "Cho2017": 208, "Lee2019_MI": 216}
EXPECTED_TOTAL_ROWS = 460
OUT_DIR = Path("h2cmi/results/review_completion")
OLD_P6_RESULTS = OUT_DIR / "spdim_w1_seed0_results.csv"


def _repo_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def _git_status(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
    )


def _env_name() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=int)


def _hash_ndarray(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(tuple(x.shape)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _dataset_list(text: str) -> list[str]:
    if text.strip().lower() in {"w1", "all", "w1_repaired"}:
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


def _parse_target_spec(text: str) -> dict[str, set[int]] | None:
    if not text.strip():
        return None
    out: dict[str, set[int]] = {ds: set() for ds in W1_DATASETS}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            dataset, spec = part.split("=", 1)
        elif ":" in part:
            dataset, spec = part.split(":", 1)
        else:
            raise ValueError(f"target spec item lacks dataset separator: {part}")
        dataset = dataset.strip()
        if dataset == "Lee2019-MI":
            dataset = "Lee2019_MI"
        if dataset not in out:
            raise ValueError(f"unsupported target-spec dataset: {dataset}")
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lo, hi = token.split("-", 1)
                out[dataset].update(range(int(lo), int(hi) + 1))
            else:
                out[dataset].add(int(token))
    return out


def _selected_targets(dataset: str, available: list[int], target_spec: dict[str, set[int]] | None) -> list[int]:
    if target_spec is None:
        return list(available)
    requested = target_spec.get(dataset, set())
    available_set = set(int(v) for v in available)
    missing = sorted(requested - available_set)
    if missing:
        raise ValueError(f"{dataset}: target spec requested unavailable subjects {missing}")
    return [int(v) for v in available if int(v) in requested]


def _expected_rows_from_targets(targets_by_dataset: dict[str, list[int]]) -> dict[str, int]:
    return {ds: len(targets_by_dataset.get(ds, [])) * len(METHODS) for ds in W1_DATASETS}


def _active_datasets(expected_rows_by_dataset: dict[str, int]) -> list[str]:
    return [ds for ds in W1_DATASETS if int(expected_rows_by_dataset.get(ds, 0)) > 0]


def _class_counts_array(y: np.ndarray) -> list[int]:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(int).tolist()


def _class_counts_text(counts: list[int]) -> str:
    return json.dumps([int(v) for v in counts], separators=(",", ":"))


def _load_manifest(path: str, datasets: list[str], seed: int) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], str]:
    rows_all = load_manifest_csv(path)
    frozen_hash = manifest_hash(rows_all)
    rows = [
        row for row in rows_all
        if row["dataset"] in datasets and int(row["source_seed"]) == int(seed)
    ]
    row_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], int(row["target_subject"]))
        if key in row_map:
            raise ValueError(f"duplicate manifest row for {key}")
        row_map[key] = row
    return rows, row_map, frozen_hash


def _run_config(args, external_sha: str) -> dict[str, Any]:
    return {
        "datasets": args.datasets,
        "target_spec": args.target_spec,
        "shard_id": args.shard_id,
        "seed": int(args.seed),
        "manifest": args.manifest,
        "manifest_hash": args.manifest_hash,
        "methods": METHODS,
        "epochs": int(args.epochs),
        "adapt_epochs": int(args.adapt_epochs),
        "adapt_lr": float(args.adapt_lr),
        "batch_size": int(args.batch_size),
        "val_fraction": float(args.val_fraction),
        "temporal_filters": int(args.temporal_filters),
        "spatial_filters": int(args.spatial_filters),
        "subspace_dims": int(args.subspace_dims),
        "dtype": args.dtype,
        "spd_device": args.spd_device,
        "external_spdim_commit": external_sha,
    }


def _launch_provenance(args, command: str, external_sha: str, frozen_hash: str) -> dict[str, Any]:
    root = _repo_root()
    runner = Path(__file__).resolve()
    config = root / "h2cmi" / "config.py"
    status = _git_status(root)
    run_config = _run_config(args, external_sha)
    run_config["frozen_manifest_hash"] = frozen_hash
    return {
        "launch_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status_porcelain": status,
        "clean_worktree": status == "",
        "runner_dirty_allowed": bool(args.allow_dirty),
        "runner_file": str(runner.relative_to(root)),
        "runner_file_sha256": sha256_file(runner),
        "config_file": str(config.relative_to(root)),
        "config_sha256": sha256_file(config),
        "run_config_sha256": _sha_text(_json_compact(run_config)),
        "external_spdim_commit": external_sha,
        "environment_name": _env_name(),
        "command_line": command,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "manifest": args.manifest,
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_hash": frozen_hash,
        "manifest_hash_matches_p7": frozen_hash == P7_MANIFEST_HASH == args.manifest_hash,
    }


def _require_clean_launch(provenance: dict[str, Any]) -> None:
    if provenance["runner_dirty_allowed"]:
        raise RuntimeError("refusing P8 official SPDIM run: --allow-dirty is exploratory-only and blocked")
    if not provenance["clean_worktree"]:
        raise RuntimeError("refusing P8 official SPDIM run: git status --porcelain is nonempty")
    if not provenance["manifest_hash_matches_p7"]:
        raise RuntimeError("refusing P8 official SPDIM run: manifest hash does not match P7")


def _allowed_p8a_status(status: str) -> bool:
    allowed_exact = {
        "?? h2cmi/run_spdim_w1_repaired_seed0.py",
        "?? h2cmi/results/review_completion/p7_training_cache_hygiene.json",
        "?? h2cmi/results/review_completion/p7_training_cache_hygiene.md",
        "?? h2cmi/results/review_completion/spdim_w1_repaired_seed0_protocol.md",
        "?? h2cmi/results/review_completion/spdim_w1_repaired_seed0_dryrun_audit.json",
        "?? h2cmi/results/review_completion/spdim_w1_repaired_seed0_dryrun_audit.md",
        "?? h2cmi/results/review_completion/slurm/spdim_w1_repaired_seed0.slurm",
    }
    allowed_prefix = (" M h2cmi/results/review_completion/COMMAND_LOG.md",)
    for line in status.splitlines():
        if line in allowed_exact:
            continue
        if any(line.startswith(prefix) for prefix in allowed_prefix):
            continue
        return False
    return True


def _manifest_detail(ep, row: dict[str, Any]) -> dict[str, Any]:
    target = int(row["target_subject"])
    adapt_idx = indices_from_trial_ids(row["adapt_trial_ids"])
    eval_idx = indices_from_trial_ids(row["eval_trial_ids"])
    source_idx = np.where(np.isin(ep.subject, np.asarray(row["source_subject_ids"], dtype=np.int64)))[0]
    adapt_counts = _class_counts_array(ep.y[adapt_idx])
    eval_counts = _class_counts_array(ep.y[eval_idx])
    overlap = sorted(set(int(i) for i in adapt_idx) & set(int(i) for i in eval_idx))
    return {
        "target_subject": target,
        "source_subject_ids": [int(v) for v in row["source_subject_ids"]],
        "source_n": int(len(source_idx)),
        "target_session": int(row["target_session"]),
        "adapt_n": int(len(adapt_idx)),
        "eval_n": int(len(eval_idx)),
        "class_counts_adapt": adapt_counts,
        "class_counts_eval": eval_counts,
        "adapt_eval_disjoint": not overlap,
        "both_classes_adapt": min(adapt_counts) > 0,
        "both_classes_eval": min(eval_counts) > 0,
        "source_idx_sha256": _sha_indices(source_idx),
        "adapt_idx_sha256": _sha_indices(adapt_idx),
        "eval_idx_sha256": _sha_indices(eval_idx),
        "split_hash": row["split_hash"],
        "adapt_trial_ids": row["adapt_trial_ids"],
        "eval_trial_ids": row["eval_trial_ids"],
    }


def _dryrun_dataset(ds_name: str, *, args, row_map, bn, TSMNet) -> dict[str, Any]:
    started = time.time()
    subject_ids = [int(s) for s in MOABB_CLASS[ds_name]().subject_list]
    ep = load_dataset(ds_name, subject_ids)
    subjects = sorted(int(s) for s in np.unique(ep.subject))
    details = []
    for target in subjects:
        row = row_map[(ds_name, target)]
        details.append(_manifest_detail(ep, row))

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    model = _build_model(TSMNet, bn, ep=ep, domain_values=subjects, args=args, device=device, dtype=dtype)

    first_row = row_map[(ds_name, subjects[0])]
    adapt_idx = indices_from_trial_ids(first_row["adapt_trial_ids"])
    n_forward = min(int(args.dryrun_forward_samples), len(adapt_idx))
    adapt_ds = TensorDomainDataset(
        ep.X[adapt_idx[: max(1, n_forward)]],
        np.zeros(max(1, n_forward), dtype=np.int64),
        np.full(max(1, n_forward), subjects[0], dtype=np.int64),
    )
    logits_shape: list[int] = []
    if n_forward:
        adapt_loader = _loader(adapt_ds, len(adapt_ds), False, args.seed)
        parameter_t = torch.tensor(1.0, dtype=torch.float64, device="cpu")
        y_true, y_pred, logits = _predict(model, adapt_loader, device=device, dtype=dtype, parameter_t=parameter_t)
        if len(y_true) != len(y_pred):
            raise RuntimeError(f"{ds_name}: dry-run prediction length mismatch")
        logits_shape = [int(v) for v in logits.shape]

    return {
        "dataset": ds_name,
        "subject_ids": subjects,
        "target_subject_ids": subjects,
        "n_subjects": len(subjects),
        "source_seed": int(args.seed),
        "tensor_shape": [int(v) for v in ep.X.shape],
        "channel_count": int(ep.X.shape[1]),
        "time_samples": int(ep.X.shape[2]),
        "channels": [str(c) for c in ep.channels],
        "sessions": [int(s) for s in sorted(np.unique(ep.session))],
        "total_class_counts": _class_counts_array(ep.y),
        "target_details": details,
        "model_instantiated": True,
        "dryrun_forward_check_performed": bool(n_forward),
        "source_training_command_buildable": True,
        "source_training_command_template": (
            "python -m h2cmi.run_spdim_w1_repaired_seed0 --mode run "
            f"--dataset {ds_name} --seed {args.seed} --manifest {args.manifest}"
        ),
        "forward_pass_without_target_labels": True,
        "forward_batch_n": int(n_forward),
        "forward_logits_shape": logits_shape,
        "adapt_loader_dummy_labels_unique": sorted(int(v) for v in np.unique(adapt_ds.labels.numpy())),
        "adapt_loader_contains_real_target_labels": False,
        "evaluation_labels_accessed_only_after_adaptation_check": True,
        "expected_rows": len(subjects) * len(METHODS),
        "elapsed_seconds": time.time() - started,
    }


def _write_protocol(report: dict[str, Any], path: str) -> None:
    lines = [
        "# SPDIM W1 Repaired-Split Seed-0 Protocol",
        "",
        "Label: W1 repaired-split seed-0 official SPDIM expansion, not full three-seed baseline.",
        "",
        "## Scope",
        "",
        "- datasets: `BNCI2014_001`, `Cho2017`, `Lee2019_MI`.",
        "- split: frozen P7 `class_stratified_half` repaired W1 manifest.",
        f"- manifest_hash: `{report['manifest_hash']}`.",
        "- source seed: `0` only.",
        "- methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.",
        "- no SPDIM seeds 1/2.",
        "- no full three-seed SPDIM baseline.",
        "- no official pretrained weights.",
        "- no vendored third-party SPDIM code.",
        "",
        "## Runtime Label Policy",
        "",
        "Target labels are used only by the frozen split manifest and final evaluation metrics. Target adaptation loaders are built with dummy labels, and method selection is not based on target performance.",
        "",
        "## Expected Rows",
        "",
        "| dataset | targets | methods | expected rows |",
        "|---|---:|---:|---:|",
    ]
    for ds, n in report["expected_rows_by_dataset"].items():
        lines.append(f"| {ds} | {n // len(METHODS)} | {len(METHODS)} | {n} |")
    lines.extend([
        f"| total | 115 | {len(METHODS)} | {report['expected_rows_total']} |",
        "",
        "## Monitoring And Validation",
        "",
        "- use `squeue` only; do not use Slurm accounting commands.",
        "- final job state must be absent from `squeue`.",
        "- stderr must be empty or contain only declared harmless warnings.",
        "- stdout must exist and record clean launch provenance.",
        "- CSV parse, row count, dataset row counts, JSON parse, checksums, no single-class eval, adapt/eval disjointness, P7 manifest hash, no target-label leakage, no target-performance method selection, no official pretrained weights, no vendoring, prediction/logits hash completeness, and `git show --check` must pass.",
        "",
        "## Red Team Review",
        "",
        "- Old W1 and old SPDIM P6 remain legacy diagnostic only.",
        "- This protocol does not approve SPDIM seeds 1/2 or full SPDIM.",
        "- Launch is blocked unless the P8A dry-run gate passes and the post-P8A worktree is clean.",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def _write_dryrun(report: dict[str, Any], *, json_path: str, md_path: str, protocol_path: str) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_protocol(report, protocol_path)
    lines = [
        "# SPDIM W1 Repaired-Split Seed-0 Dry-Run Audit",
        "",
        f"- status: `{'PASS' if report['dryrun_pass'] else 'BLOCKED'}`",
        f"- approve_gpu_run: `{report['approve_gpu_run']}`",
        f"- manifest_hash: `{report['manifest_hash']}`",
        f"- manifest_hash_matches_p7: `{report['manifest_hash_matches_p7']}`",
        f"- expected_rows_total: `{report['expected_rows_total']}`",
        f"- worktree_clean_for_launch: `{report['worktree_clean_for_launch']}`",
        f"- external_spdim_commit: `{report['external_spdim_commit']}`",
        "",
        "## Gate Checks",
        "",
        f"- all_eval_both_classes: `{report['all_eval_both_classes']}`",
        f"- all_adapt_both_classes: `{report['all_adapt_both_classes']}`",
        f"- all_adapt_eval_disjoint: `{report['all_adapt_eval_disjoint']}`",
        f"- target_label_leakage_detected: `{report['target_label_leakage_detected']}`",
        f"- pretrained_weight_detected: `{report['pretrained_weight_detected']}`",
        f"- vendoring_detected: `{report['vendoring_detected']}`",
        f"- shape_blocker_detected: `{report['shape_blocker_detected']}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | targets | expected rows | adapt counts | eval counts | tensor shape |",
        "|---|---:|---:|---|---|---|",
    ]
    for ds in report["datasets"]:
        adapt_counts = sorted({tuple(d["class_counts_adapt"]) for d in ds["target_details"]})
        eval_counts = sorted({tuple(d["class_counts_eval"]) for d in ds["target_details"]})
        lines.append(
            f"| {ds['dataset']} | {ds['n_subjects']} | {ds['expected_rows']} | "
            f"`{adapt_counts}` | `{eval_counts}` | `{ds['tensor_shape']}` |"
        )
    lines.extend([
        "",
        "## Split Evidence",
        "",
        "Exact source subject IDs, adaptation trial IDs, evaluation trial IDs, and split hashes for every target subject are recorded in the JSON audit.",
        "",
        "## Red Team Review",
        "",
        "- The dry-run does not train or adapt on GPU.",
        "- The expensive official SPD forward check is optional and defaults to zero samples; the required gate is model instantiation plus loader/manifest validation.",
        "- Target adaptation loaders use dummy labels; target labels are not available to adaptation or method selection.",
        "- `worktree_clean_for_launch` is evaluated after P7 cache hygiene allowing only expected P8A files pending commit; the GPU launch still requires an actually clean post-commit worktree.",
    ])
    if report["datasets_blocked"]:
        lines.extend(["", "## Blockers", ""])
        for item in report["datasets_blocked"]:
            lines.append(f"- `{item['dataset']}`: {item['failure']}")
    Path(md_path).write_text("\n".join(lines) + "\n")


def _run_dryrun(args) -> int:
    root = _repo_root()
    status = _git_status(root)
    rows, row_map, frozen_hash = _load_manifest(args.manifest, args.datasets, args.seed)
    external_sha = _git_sha(args.external_spdim_path)
    bn, TSMNet, _Trainer = _import_official(args.external_spdim_path)
    datasets, blocked = [], []
    started = time.time()
    for ds in args.datasets:
        try:
            datasets.append(_dryrun_dataset(ds, args=args, row_map=row_map, bn=bn, TSMNet=TSMNet))
        except Exception as exc:
            blocked.append({"dataset": ds, "failure": "".join(traceback.format_exception_only(type(exc), exc)).strip()})

    expected_by_dataset = {ds: EXPECTED_ROWS_BY_DATASET[ds] for ds in args.datasets}
    all_details = [detail for ds in datasets for detail in ds["target_details"]]
    all_eval = bool(all_details) and all(d["both_classes_eval"] for d in all_details)
    all_adapt = bool(all_details) and all(d["both_classes_adapt"] for d in all_details)
    all_disjoint = bool(all_details) and all(d["adapt_eval_disjoint"] for d in all_details)
    target_leak = any(
        ds.get("adapt_loader_contains_real_target_labels", True)
        or ds.get("adapt_loader_dummy_labels_unique") != [0]
        for ds in datasets
    )
    vendoring = str(Path(args.external_spdim_path).resolve()).startswith(str(root.resolve()))
    expected_rows_total = sum(expected_by_dataset.values())
    worktree_clean_for_launch = _allowed_p8a_status(status) and not (root / "results/h2cmi/p7_w1_repaired_bundles").exists()
    dryrun_pass = (
        not blocked
        and len(rows) == 115
        and frozen_hash == P7_MANIFEST_HASH == args.manifest_hash
        and expected_rows_total == EXPECTED_TOTAL_ROWS
        and all_eval
        and all_adapt
        and all_disjoint
        and not target_leak
        and external_sha == OFFICIAL_SHA
        and not vendoring
        and worktree_clean_for_launch
    )
    report = {
        "dryrun_pass": bool(dryrun_pass),
        "expected_rows_total": expected_rows_total,
        "expected_rows_by_dataset": expected_by_dataset,
        "datasets_passed": [d["dataset"] for d in datasets],
        "datasets_blocked": blocked,
        "manifest": args.manifest,
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_hash": frozen_hash,
        "manifest_hash_matches_p7": frozen_hash == P7_MANIFEST_HASH == args.manifest_hash,
        "split_family": SPLIT_FAMILY,
        "all_eval_both_classes": all_eval,
        "all_adapt_both_classes": all_adapt,
        "all_adapt_eval_disjoint": all_disjoint,
        "target_label_leakage_detected": bool(target_leak),
        "pretrained_weight_detected": False,
        "vendoring_detected": bool(vendoring),
        "shape_blocker_detected": bool(blocked),
        "worktree_clean_for_launch": bool(worktree_clean_for_launch),
        "worktree_status_porcelain_before_outputs": status,
        "approve_gpu_run": bool(dryrun_pass),
        "source_seed": int(args.seed),
        "methods": METHODS,
        "official_repo": OFFICIAL_REPO,
        "official_sha": OFFICIAL_SHA,
        "external_spdim_commit": external_sha,
        "external_spdim_path": str(Path(args.external_spdim_path).resolve()),
        "external_license_file_present": _has_license(args.external_spdim_path),
        "no_official_pretrained_weights_policy": "model initialized from scratch; no checkpoint/pretrained-weight argument exists in this runner",
        "no_vendoring_policy": "official SPDIM code imported only from external path outside this git repository",
        "evaluation_labels_accessed_only_after_adaptation": all(
            ds.get("evaluation_labels_accessed_only_after_adaptation_check", False) for ds in datasets
        ),
        "source_training_command_buildable": all(ds.get("source_training_command_buildable", False) for ds in datasets),
        "datasets": datasets,
        "elapsed_seconds": time.time() - started,
    }
    _write_dryrun(report, json_path=args.dryrun_json, md_path=args.dryrun_md, protocol_path=args.protocol)
    return 0 if dryrun_pass else 2


def _fieldnames() -> list[str]:
    return [
        "dataset", "target_subject", "source_seed", "split_family", "method",
        "n_adapt", "n_eval", "class_counts_adapt", "class_counts_eval",
        "acc", "bacc", "macro_f1", "prediction_hash", "logits_hash",
        "status", "failure_reason", "protocol", "official_repo", "official_sha",
        "runner_commit", "manifest_hash", "split_hash", "source_subjects", "source_n",
        "target_session", "adapt_eval_disjoint", "both_classes_adapt", "both_classes_eval",
        "source_idx_sha256", "adapt_idx_sha256", "eval_idx_sha256", "parameter_t",
        "source_model_sha256", "train_seconds", "adapt_seconds", "eval_seconds",
        "target_label_leakage_detected", "method_selection_uses_target_performance",
        "official_pretrained_weight_used", "third_party_vendored",
    ]


def _append_rows(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(), lineterminator="\n")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in _fieldnames()})


def _run_target_repaired(ep, split_row: dict[str, Any], *, args, bn, TSMNet, Trainer,
                         runner_commit: str, external_sha: str) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    target = int(split_row["target_subject"])
    source_subjects = [int(v) for v in split_row["source_subject_ids"]]
    source_idx = np.where(np.isin(ep.subject, np.asarray(source_subjects, dtype=np.int64)))[0]
    adapt_idx = indices_from_trial_ids(split_row["adapt_trial_ids"])
    eval_idx = indices_from_trial_ids(split_row["eval_trial_ids"])
    domain_values = sorted(source_subjects + [target])
    train_idx, val_idx = _source_train_val_split(source_idx, ep.y, ep.subject, args.val_fraction)
    adapt_counts = _class_counts_array(ep.y[adapt_idx])
    eval_counts = _class_counts_array(ep.y[eval_idx])

    meta = {
        "status": "ok",
        "dataset": args.dataset,
        "protocol": "W1_repaired_class_stratified_half_manifest",
        "official_repo": OFFICIAL_REPO,
        "official_sha": external_sha,
        "runner_commit": runner_commit,
        "manifest_hash": args.manifest_hash,
        "split_hash": split_row["split_hash"],
        "source_seed": int(args.seed),
        "target_subject": target,
        "split_family": SPLIT_FAMILY,
        "source_subjects": " ".join(str(s) for s in source_subjects),
        "source_n": int(len(source_idx)),
        "target_session": int(split_row["target_session"]),
        "n_adapt": int(len(adapt_idx)),
        "n_eval": int(len(eval_idx)),
        "class_counts_adapt": _class_counts_text(adapt_counts),
        "class_counts_eval": _class_counts_text(eval_counts),
        "adapt_eval_disjoint": bool(set(adapt_idx.tolist()).isdisjoint(set(eval_idx.tolist()))),
        "both_classes_adapt": bool(min(adapt_counts) > 0),
        "both_classes_eval": bool(min(eval_counts) > 0),
        "source_idx_sha256": _sha_indices(source_idx),
        "adapt_idx_sha256": _sha_indices(adapt_idx),
        "eval_idx_sha256": _sha_indices(eval_idx),
        "failure_reason": "",
        "target_label_leakage_detected": False,
        "method_selection_uses_target_performance": False,
        "official_pretrained_weight_used": False,
        "third_party_vendored": False,
    }

    train_ds = TensorDomainDataset(ep.X[train_idx], ep.y[train_idx], ep.subject[train_idx])
    val_ds = TensorDomainDataset(ep.X[val_idx], ep.y[val_idx], ep.subject[val_idx])
    adapt_ds = TensorDomainDataset(
        ep.X[adapt_idx],
        np.zeros(len(adapt_idx), dtype=np.int64),
        np.full(len(adapt_idx), target, dtype=np.int64),
    )
    eval_ds = TensorDomainDataset(ep.X[eval_idx], ep.y[eval_idx], np.full(len(eval_idx), target, dtype=np.int64))

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
    rows.append(meta | _metrics(y_true, y_pred) | _prediction_fields(y_true, y_pred, logits) | {
        "method": "source_only_tsmnet",
        "parameter_t": float(parameter_t.item()),
        "source_model_sha256": source_model_sha,
        "train_seconds": train_seconds,
        "adapt_seconds": 0.0,
        "eval_seconds": time.time() - eval_start,
    })

    rct_model, rct_seconds = fresh_refit()
    eval_start = time.time()
    y_true, y_pred, logits = _predict(rct_model, eval_loader, device=device, dtype=dtype, parameter_t=parameter_t)
    rows.append(meta | _metrics(y_true, y_pred) | _prediction_fields(y_true, y_pred, logits) | {
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
        "method": "spdim_geodesic",
        "parameter_t": float(best_t.detach().cpu().item()),
        "source_model_sha256": source_model_sha,
        "train_seconds": train_seconds,
        "adapt_seconds": geo_seconds,
        "eval_seconds": time.time() - eval_start,
    })

    bias_model, refit_seconds = fresh_refit()
    bias_start = time.time()
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
        "method": "spdim_bias",
        "parameter_t": 1.0,
        "source_model_sha256": source_model_sha,
        "train_seconds": train_seconds,
        "adapt_seconds": bias_seconds,
        "eval_seconds": time.time() - eval_start,
    })
    return rows


def _failure_row(ds: str, target: int, args, failure: str, external_sha: str, runner_commit: str,
                 split_row: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "dataset": ds,
        "target_subject": int(target),
        "source_seed": int(args.seed),
        "split_family": SPLIT_FAMILY,
        "method": "target_failed",
        "status": "failed",
        "failure_reason": failure,
        "protocol": "W1_repaired_class_stratified_half_manifest",
        "official_repo": OFFICIAL_REPO,
        "official_sha": external_sha,
        "runner_commit": runner_commit,
        "manifest_hash": args.manifest_hash,
    }
    if split_row:
        base |= {
            "split_hash": split_row["split_hash"],
            "n_adapt": int(split_row["n_adapt"]),
            "n_eval": int(split_row["n_eval"]),
            "class_counts_adapt": _class_counts_text(split_row["class_counts_adapt"]),
            "class_counts_eval": _class_counts_text(split_row["class_counts_eval"]),
            "adapt_eval_disjoint": bool(split_row["adapt_eval_disjoint"]),
            "both_classes_adapt": bool(split_row["both_classes_adapt"]),
            "both_classes_eval": bool(split_row["both_classes_eval"]),
        }
    return base


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _mean(rows: list[dict[str, str]], field: str) -> float | None:
    vals = [float(row[field]) for row in rows if row.get(field) not in {"", None}]
    return float(sum(vals) / len(vals)) if vals else None


def _result_groups(rows: list[dict[str, str]]):
    by_dataset = defaultdict(lambda: defaultdict(list))
    by_method = defaultdict(list)
    by_key = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_dataset[row["dataset"]][row["method"]].append(row)
        by_method[row["method"]].append(row)
        by_key[(row["dataset"], row["target_subject"], row["source_seed"], row["method"])] = row
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


def _harm(by_key: dict, method: str, baseline: str, field: str = "bacc") -> dict[str, Any]:
    vals = _metric_delta(by_key, method, baseline, field)
    harms = sum(1 for val in vals if val < 0)
    return {"n": len(vals), "harm_count": harms, "harm_rate": float(harms / len(vals)) if vals else None}


def _summaries(rows: list[dict[str, str]], datasets: list[str]) -> tuple[dict, dict, dict, dict, dict]:
    by_dataset, by_method, by_key = _result_groups(rows)
    per_dataset = {}
    for ds in datasets:
        per_dataset[ds] = {}
        for method in METHODS:
            mrows = by_dataset[ds][method]
            per_dataset[ds][method] = {
                "n": len(mrows),
                "mean_acc": _mean(mrows, "acc"),
                "mean_bacc": _mean(mrows, "bacc"),
                "mean_macro_f1": _mean(mrows, "macro_f1"),
            }
    subject_weighted = {}
    dataset_macro = {}
    for method in METHODS:
        subject_weighted[method] = {
            "n": len(by_method[method]),
            "mean_acc": _mean(by_method[method], "acc"),
            "mean_bacc": _mean(by_method[method], "bacc"),
        }
        ds_acc = [per_dataset[ds][method]["mean_acc"] for ds in datasets if per_dataset[ds][method]["mean_acc"] is not None]
        ds_bacc = [per_dataset[ds][method]["mean_bacc"] for ds in datasets if per_dataset[ds][method]["mean_bacc"] is not None]
        dataset_macro[method] = {
            "mean_acc": float(sum(ds_acc) / len(ds_acc)) if ds_acc else None,
            "mean_bacc": float(sum(ds_bacc) / len(ds_bacc)) if ds_bacc else None,
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
    return per_dataset, subject_weighted, dataset_macro, deltas, harms


def _validation(rows: list[dict[str, str]], datasets: list[str], frozen_hash: str,
                expected_rows_by_dataset: dict[str, int] | None = None) -> dict[str, Any]:
    dataset_counts = defaultdict(int)
    method_counts = defaultdict(int)
    failed = 0
    single_class_eval = 0
    single_class_adapt = 0
    disjoint_fail = 0
    pred_missing = 0
    logits_missing = 0
    leakage = False
    method_selection = False
    pretrained = False
    vendored = False
    for row in rows:
        dataset_counts[row["dataset"]] += 1
        method_counts[row["method"]] += 1
        if row.get("status") != "ok":
            failed += 1
        if row.get("status") == "ok":
            eval_counts = json.loads(row["class_counts_eval"])
            adapt_counts = json.loads(row["class_counts_adapt"])
            single_class_eval += int(min(eval_counts) <= 0)
            single_class_adapt += int(min(adapt_counts) <= 0)
            disjoint_fail += int(str(row.get("adapt_eval_disjoint")) != "True")
            pred_missing += int(not row.get("prediction_hash"))
            logits_missing += int(not row.get("logits_hash"))
        leakage = leakage or row.get("target_label_leakage_detected") == "True"
        method_selection = method_selection or row.get("method_selection_uses_target_performance") == "True"
        pretrained = pretrained or row.get("official_pretrained_weight_used") == "True"
        vendored = vendored or row.get("third_party_vendored") == "True"
    expected = (
        {ds: int(expected_rows_by_dataset.get(ds, 0)) for ds in datasets}
        if expected_rows_by_dataset is not None
        else {ds: EXPECTED_ROWS_BY_DATASET[ds] for ds in datasets}
    )
    expected_nonzero = {ds: n for ds, n in expected.items() if n}
    dataset_counts_ok = all(int(dataset_counts.get(ds, 0)) == n for ds, n in expected.items())
    return {
        "row_count": len(rows),
        "expected_rows_total": sum(expected.values()),
        "dataset_rows": dict(sorted(dataset_counts.items())),
        "expected_rows_by_dataset": expected,
        "method_rows": dict(sorted(method_counts.items())),
        "failed_rows": failed,
        "single_class_eval_rows": single_class_eval,
        "single_class_adapt_rows": single_class_adapt,
        "adapt_eval_disjoint_failures": disjoint_fail,
        "prediction_hash_missing_rows": pred_missing,
        "logits_hash_missing_rows": logits_missing,
        "prediction_hash_complete": pred_missing == 0,
        "logits_hash_complete": logits_missing == 0,
        "manifest_hash": frozen_hash,
        "manifest_hash_matches_p7": frozen_hash == P7_MANIFEST_HASH,
        "target_label_leakage_detected": leakage,
        "target_performance_method_selection_detected": method_selection,
        "pretrained_weight_detected": pretrained,
        "vendoring_detected": vendored,
        "validation_pass": (
            len(rows) == sum(expected.values())
            and dataset_counts_ok
            and set(dataset_counts.keys()) == set(expected_nonzero.keys())
            and failed == 0
            and single_class_eval == 0
            and single_class_adapt == 0
            and disjoint_fail == 0
            and pred_missing == 0
            and logits_missing == 0
            and frozen_hash == P7_MANIFEST_HASH
            and not leakage
            and not method_selection
            and not pretrained
            and not vendored
        ),
    }


def _write_legacy_compare(new_rows: list[dict[str, str]], path: str, datasets: list[str]) -> dict[str, Any]:
    old_rows = _read_csv_rows(OLD_P6_RESULTS)
    old_ok = [r for r in old_rows if r.get("status") == "ok"]
    old_per, old_sw, old_dm, _old_deltas, _old_harms = _summaries(old_ok, datasets)
    new_per, new_sw, new_dm, _new_deltas, _new_harms = _summaries(new_rows, datasets)
    payload = {
        "legacy_source": str(OLD_P6_RESULTS),
        "legacy_status": "diagnostic_only_old_split",
        "repaired_status": "w1_repaired_seed0_official_spdim",
        "legacy_rows": len(old_ok),
        "new_rows": len(new_rows),
        "old_subject_weighted": old_sw,
        "new_subject_weighted": new_sw,
        "old_dataset_macro": old_dm,
        "new_dataset_macro": new_dm,
        "old_per_dataset": old_per,
        "new_per_dataset": new_per,
    }
    lines = [
        "# SPDIM W1 Repaired Seed-0 Legacy Compare",
        "",
        "Old SPDIM P6 rows remain legacy diagnostic only because they used the old W1 split. This file compares magnitudes only and does not rehabilitate the old split.",
        "",
        "| method | old subject-weighted bAcc | repaired subject-weighted bAcc | old dataset-macro bAcc | repaired dataset-macro bAcc |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {old_sw[method]['mean_bacc']:.6f} | {new_sw[method]['mean_bacc']:.6f} | "
            f"{old_dm[method]['mean_bacc']:.6f} | {new_dm[method]['mean_bacc']:.6f} |"
        )
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- Legacy rows are diagnostic-only.",
        "- Repaired rows use the P7 `class_stratified_half` manifest.",
        "- This comparison is not a full three-seed SPDIM baseline.",
    ])
    Path(path).write_text("\n".join(lines) + "\n")
    return payload


def _write_digest(path: str, summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# SPDIM W1 Repaired-Split Seed-0 Result Digest",
        "",
        "Label: W1 repaired-split seed-0 official SPDIM expansion, not full three-seed baseline.",
        "",
        f"- status: `{summary['status']}`",
        f"- row_count: `{summary['validation']['row_count']}`",
        f"- result_csv_sha256: `{summary['result_csv_sha256']}`",
        f"- prediction_hash_complete: `{summary['validation']['prediction_hash_complete']}`",
        f"- logits_hash_complete: `{summary['validation']['logits_hash_complete']}`",
        "",
        "## Per-Dataset Mean bAcc",
        "",
        "| dataset | source_only_tsmnet | rct | spdim_geodesic | spdim_bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in summary["datasets"]:
        vals = summary["per_dataset"][ds]
        lines.append(
            f"| {ds} | {vals['source_only_tsmnet']['mean_bacc']:.6f} | {vals['rct']['mean_bacc']:.6f} | "
            f"{vals['spdim_geodesic']['mean_bacc']:.6f} | {vals['spdim_bias']['mean_bacc']:.6f} |"
        )
    lines.extend([
        "",
        "## Overall Subject-Weighted Mean",
        "",
        "| method | n | mean acc | mean bAcc |",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        vals = summary["overall_subject_weighted"][method]
        lines.append(f"| {method} | {vals['n']} | {vals['mean_acc']:.6f} | {vals['mean_bacc']:.6f} |")
    lines.extend([
        "",
        "## Dataset-Macro Mean",
        "",
        "| method | mean acc | mean bAcc |",
        "|---|---:|---:|",
    ])
    for method in METHODS:
        vals = summary["dataset_macro"][method]
        lines.append(f"| {method} | {vals['mean_acc']:.6f} | {vals['mean_bacc']:.6f} |")
    lines.extend([
        "",
        "## Deltas",
        "",
        "| contrast | n | mean acc delta | mean bAcc delta |",
        "|---|---:|---:|---:|",
    ])
    for name, vals in summary["deltas"].items():
        lines.append(f"| {name} | {vals['n']} | {vals['mean_acc_delta']:.6f} | {vals['mean_bacc_delta']:.6f} |")
    lines.extend([
        "",
        "## Harm Counts",
        "",
        "| contrast | n | harm count | harm rate |",
        "|---|---:|---:|---:|",
    ])
    for name, vals in summary["harm"].items():
        lines.append(f"| {name} | {vals['n']} | {vals['harm_count']} | {vals['harm_rate']:.6f} |")
    lines.extend([
        "",
        "## Per-Subject Table",
        "",
        "| dataset | target | source_only_tsmnet | rct | spdim_geodesic | spdim_bias |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_key = {(r["dataset"], r["target_subject"], r["method"]): r for r in rows if r.get("status") == "ok"}
    for ds in summary["datasets"]:
        subjects = sorted({int(r["target_subject"]) for r in rows if r["dataset"] == ds})
        for target in subjects:
            vals = [float(by_key[(ds, str(target), method)]["bacc"]) for method in METHODS]
            lines.append(f"| {ds} | {target} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {vals[3]:.6f} |")
    lines.extend([
        "",
        "## Red Team Review",
        "",
        "- This is seed 0 only, not a full three-seed baseline.",
        "- Old SPDIM P6 remains legacy diagnostic only.",
        "- No target labels enter adaptation, early stopping, or method selection.",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def _write_audit(path: str, summary: dict[str, Any]) -> None:
    lines = [
        "# SPDIM W1 Repaired-Split Seed-0 Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- launch_commit: `{summary['launch_provenance']['launch_commit']}`",
        f"- clean_worktree_at_launch: `{summary['launch_provenance']['clean_worktree']}`",
        f"- runner_dirty_allowed: `{summary['launch_provenance']['runner_dirty_allowed']}`",
        f"- manifest_hash: `{summary['manifest_hash']}`",
        f"- expected_rows_total: `{summary['validation']['expected_rows_total']}`",
        f"- row_count: `{summary['validation']['row_count']}`",
        f"- prediction_hash_complete: `{summary['validation']['prediction_hash_complete']}`",
        f"- logits_hash_complete: `{summary['validation']['logits_hash_complete']}`",
        f"- target_label_leakage_detected: `{summary['validation']['target_label_leakage_detected']}`",
        f"- target_performance_method_selection_detected: `{summary['validation']['target_performance_method_selection_detected']}`",
        f"- pretrained_weight_detected: `{summary['validation']['pretrained_weight_detected']}`",
        f"- vendoring_detected: `{summary['validation']['vendoring_detected']}`",
        "",
        "## Launch Provenance",
        "",
        "```json",
        json.dumps(summary["launch_provenance"], indent=2, sort_keys=True),
        "```",
        "",
        "## Validation",
        "",
        "- Final `squeue` absence is validated after Slurm completion before commit.",
        "- Result CSV parse and row-count gates are recorded in the summary JSON.",
        "- Adapt/eval split disjointness and both-class gates are inherited from the frozen P7 manifest and rechecked on every row.",
        "",
        "## Red Team Review",
        "",
        "- This run uses P7 repaired split, not the legacy contiguous W1 split.",
        "- The external SPDIM path is outside the repository; no third-party code was vendored.",
        "- The runner has no checkpoint/pretrained-weight input and initializes models from scratch.",
        "- This does not approve seeds 1/2 or full SPDIM.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def _write_summary_and_digest(rows: list[dict[str, str]], *, args, launch_provenance: dict[str, Any],
                              started: float, finished: float, frozen_hash: str) -> None:
    expected_rows_by_dataset = getattr(
        args,
        "expected_rows_by_dataset",
        {ds: EXPECTED_ROWS_BY_DATASET[ds] for ds in args.datasets},
    )
    active_datasets = _active_datasets(expected_rows_by_dataset)
    validation = _validation(rows, active_datasets, frozen_hash, expected_rows_by_dataset)
    per_dataset, subject_weighted, dataset_macro, deltas, harms = _summaries(rows, active_datasets)
    legacy = _write_legacy_compare(rows, args.legacy_compare, active_datasets)
    summary = {
        "status": "pass" if validation["validation_pass"] else "blocked",
        "label": "W1 repaired-split seed-0 official SPDIM expansion, not full three-seed baseline.",
        "datasets": active_datasets,
        "all_requested_datasets": args.datasets,
        "target_spec": args.target_spec,
        "shard_id": args.shard_id,
        "expected_rows_by_dataset": expected_rows_by_dataset,
        "source_seed": int(args.seed),
        "methods": METHODS,
        "split_family": SPLIT_FAMILY,
        "manifest": args.manifest,
        "manifest_hash": frozen_hash,
        "result_csv": args.out,
        "result_csv_sha256": sha256_file(args.out) if Path(args.out).exists() else "",
        "elapsed_seconds": finished - started,
        "launch_provenance": launch_provenance,
        "validation": validation,
        "per_dataset": per_dataset,
        "overall_subject_weighted": subject_weighted,
        "dataset_macro": dataset_macro,
        "deltas": deltas,
        "harm": harms,
        "legacy_compare": legacy,
        "final_squeue_absent": None,
        "stderr_status": "pending_post_slurm_validation",
        "stdout_status": "pending_post_slurm_validation",
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_digest(args.digest, summary, rows)
    _write_audit(args.audit, summary)


def _run_gpu(args) -> int:
    started = time.time()
    command = " ".join(sys.argv)
    rows_manifest, row_map, frozen_hash = _load_manifest(args.manifest, args.datasets, args.seed)
    targets_by_dataset = {
        ds: _selected_targets(
            ds,
            [int(s) for s in MOABB_CLASS[ds]().subject_list],
            args.target_spec_map,
        )
        for ds in args.datasets
    }
    args.expected_rows_by_dataset = _expected_rows_from_targets(targets_by_dataset)
    expected_total = sum(args.expected_rows_by_dataset.values())
    if expected_total <= 0:
        raise RuntimeError("target spec selected zero P8 rows")
    external_sha = _git_sha(args.external_spdim_path)
    launch_provenance = _launch_provenance(args, command, external_sha, frozen_hash)
    _require_clean_launch(launch_provenance)
    if external_sha != OFFICIAL_SHA:
        raise RuntimeError(f"external SPDIM SHA mismatch: {external_sha} != {OFFICIAL_SHA}")
    for path in (args.out, args.audit, args.summary, args.digest, args.legacy_compare, args.failure_trace):
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
    bn, TSMNet, Trainer = _import_official(args.external_spdim_path)
    result_rows: list[dict[str, Any]] = []
    runner_commit = launch_provenance["launch_commit"]
    for ds in args.datasets:
        selected_targets = targets_by_dataset[ds]
        if not selected_targets:
            continue
        ds_args = SimpleNamespace(**vars(args))
        ds_args.dataset = ds
        ep = load_dataset(ds, MOABB_CLASS[ds]().subject_list)
        targets = _selected_targets(ds, sorted(int(s) for s in np.unique(ep.subject)), args.target_spec_map)
        for target in targets:
            split_row = row_map.get((ds, target))
            try:
                if split_row is None:
                    raise RuntimeError(f"missing repaired split manifest row for {ds} target {target}")
                target_rows = _run_target_repaired(
                    ep,
                    split_row,
                    args=ds_args,
                    bn=bn,
                    TSMNet=TSMNet,
                    Trainer=Trainer,
                    runner_commit=runner_commit,
                    external_sha=external_sha,
                )
                result_rows.extend(target_rows)
                _append_rows(args.out, target_rows)
            except Exception as exc:
                failure = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                fail_row = _failure_row(ds, target, args, failure, external_sha, runner_commit, split_row)
                result_rows.append(fail_row)
                _append_rows(args.out, [fail_row])
                Path(args.failure_trace).parent.mkdir(parents=True, exist_ok=True)
                with open(args.failure_trace, "a") as f:
                    f.write(f"\n## {ds} target {target}\n")
                    f.write(traceback.format_exc())
        del ep
    finished = time.time()
    rows = _read_csv_rows(args.out)
    _write_summary_and_digest(rows, args=args, launch_provenance=launch_provenance,
                              started=started, finished=finished, frozen_hash=frozen_hash)
    return 0 if len(rows) == expected_total and all(row.get("status") == "ok" for row in rows) else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry-run", "run"], required=True)
    ap.add_argument("--external-spdim-path", required=True)
    ap.add_argument("--datasets", default="w1_repaired")
    ap.add_argument("--manifest", default="h2cmi/results/review_completion/w1_repaired_split_manifest.csv")
    ap.add_argument("--manifest-hash", default=P7_MANIFEST_HASH)
    ap.add_argument("--target-spec", default="")
    ap.add_argument("--shard-id", default="")
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
    ap.add_argument("--dryrun-forward-samples", type=int, default=0)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--protocol", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_protocol.md")
    ap.add_argument("--dryrun-md", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_dryrun_audit.md")
    ap.add_argument("--dryrun-json", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_dryrun_audit.json")
    ap.add_argument("--out", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_results.csv")
    ap.add_argument("--audit", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_audit.md")
    ap.add_argument("--summary", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_summary.json")
    ap.add_argument("--digest", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_result_digest.md")
    ap.add_argument("--legacy-compare", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_legacy_compare.md")
    ap.add_argument("--failure-trace", default="h2cmi/results/review_completion/spdim_w1_repaired_seed0_failure_trace.txt")
    args = ap.parse_args()
    args.datasets = _dataset_list(args.datasets)
    args.target_spec_map = _parse_target_spec(args.target_spec)
    if args.datasets != W1_DATASETS:
        bad = [d for d in args.datasets if d not in W1_DATASETS]
        if bad:
            raise SystemExit(f"unsupported P8 dataset(s): {bad}")
        raise SystemExit("P8 must run all three W1 datasets together")
    if int(args.seed) != 0:
        raise SystemExit("P8 only permits source seed 0")
    if args.manifest_hash != P7_MANIFEST_HASH:
        raise SystemExit("P8 manifest hash must match P7 frozen repaired split hash")
    if args.mode == "dry-run":
        return _run_dryrun(args)
    return _run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
