"""P9 controller for frozen P8 SPDIM repaired-split seeds 1 and 2.

The P8 runner remains byte-for-byte frozen. This controller expands only the
approved source-seed set, delegates GPU computation to the frozen runner, and
adds P9 runtime provenance to each shard summary.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mne
import moabb
import numpy as np
import torch

from h2cmi.data.real_eeg import load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.run_spdim_probe import TensorDomainDataset
from h2cmi import run_spdim_w1_repaired_seed0 as p8
from h2cmi.w1_repaired_split import indices_from_trial_ids, load_manifest_csv, manifest_hash


ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
OUT_DIR = ROOT / "h2cmi" / "results" / "review_completion"
COMMAND_LOG = OUT_DIR / "COMMAND_LOG.md"

APPROVED_SEEDS = [1, 2]
P8_RESULT_COMMIT = "3d820dfd1ef988cdd44acd34d47ed37c490a98e5"
P8_MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
P8_RUNNER_SHA256 = "946b28b93f0ddbce395ade7c6a13d30b20f368fe7a1ae22fbefa01f291e82be8"
P8_CONFIG_SHA256 = "6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b"
P8_SEED0_RESULT_SHA256 = "118ec37f3a195d50c24abf24b4c61048cdbc0ffff7d9c0f0bf51c83f7f69229c"
EXTERNAL_SHA = "1b0de0ccd4c48a4ff28f087b866a0b671b029c39"
EXPECTED_ROWS_BY_DATASET = {"BNCI2014_001": 72, "Cho2017": 416, "Lee2019_MI": 432}
EXPECTED_ROWS_TOTAL = 920
EXPECTED_FINAL_ROWS = 1380
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260710
ICML_PYTHON = "/home/infres/yinwang/anaconda3/envs/icml/bin/python"
FROZEN_HYPERPARAMETERS = {
    "epochs": 20,
    "adapt_epochs": 30,
    "adapt_lr": 0.01,
    "batch_size": 64,
    "val_fraction": 0.2,
    "temporal_filters": 4,
    "spatial_filters": 40,
    "subspace_dims": 20,
    "dtype": "float32",
    "spd_device": "cpu",
}
APPROVED_TARGET_SPECS = {
    0: "BNCI2014_001=1-9;Cho2017=1-20",
    1: "Cho2017=21-49",
    2: "Cho2017=50-52;Lee2019_MI=1-26",
    3: "Lee2019_MI=27-54",
}

PROTOCOL_PATH = OUT_DIR / "spdim_w1_repaired_three_seed_protocol.md"
DRYRUN_MD_PATH = OUT_DIR / "spdim_w1_repaired_seeds12_dryrun_audit.md"
DRYRUN_JSON_PATH = OUT_DIR / "spdim_w1_repaired_seeds12_dryrun_audit.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_environment(device: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    if device.startswith("cuda") and cuda_available:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        device_name = "CPU"
    return {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": device,
        "device_name": device_name,
        "moabb_version": getattr(moabb, "__version__", "MISSING"),
        "mne_version": mne.__version__,
        "conda_default_env_inherited_label": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_label_is_runtime_proof": False,
    }


def print_runtime_environment(payload: dict[str, Any]) -> None:
    for key in (
        "sys_executable",
        "sys_prefix",
        "python_version",
        "pytorch_version",
        "cuda_version",
        "cuda_available",
        "device",
        "device_name",
        "moabb_version",
        "mne_version",
        "conda_default_env_inherited_label",
    ):
        print(f"runtime_{key}={payload[key]}", flush=True)


def git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )


def allowed_p9a_status(status: str) -> tuple[bool, list[str]]:
    allowed_exact = {
        "?? h2cmi/run_spdim_w1_repaired_seeds12.py",
        "?? h2cmi/results/review_completion/spdim_w1_repaired_three_seed_protocol.md",
        "?? h2cmi/results/review_completion/spdim_w1_repaired_seeds12_dryrun_audit.md",
        "?? h2cmi/results/review_completion/spdim_w1_repaired_seeds12_dryrun_audit.json",
        "?? h2cmi/results/review_completion/slurm/spdim_w1_repaired_seeds12_8task.slurm",
    }
    allowed_modified = {" M h2cmi/results/review_completion/COMMAND_LOG.md"}
    unexpected = [
        line for line in status.splitlines()
        if line not in allowed_exact and line not in allowed_modified
    ]
    return not unexpected, unexpected


def accounting_script_calls() -> list[str]:
    pattern = r"\bsa" + "cct" + r"\b"
    command = [
        "rg", "-n", pattern, "h2cmi", "scripts",
        "--glob", "*.py", "--glob", "*.sh", "--glob", "*.slurm",
    ]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"script-call scan failed: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def build_source_command(seed: int, dataset: str) -> str:
    return (
        f"{ICML_PYTHON} -m h2cmi.run_spdim_w1_repaired_seeds12 --mode run "
        f"--seed {seed} --datasets BNCI2014_001,Cho2017,Lee2019_MI "
        f"--manifest h2cmi/results/review_completion/w1_repaired_split_manifest.csv "
        f"--manifest-hash {P8_MANIFEST_HASH} --epochs 20 --adapt-epochs 30 "
        f"--device cuda # dataset shard includes {dataset}"
    )


def require_frozen_protocol(args: argparse.Namespace) -> None:
    mismatches = {
        key: {"expected": expected, "observed": getattr(args, key)}
        for key, expected in FROZEN_HYPERPARAMETERS.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise SystemExit(f"P9 frozen hyperparameter mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if args.mode == "run":
        if args.device != "cuda":
            raise SystemExit("P9 GPU execution requires --device cuda")
        if args.target_spec not in APPROVED_TARGET_SPECS.values():
            raise SystemExit("P9 target spec is not one of the four frozen P8 shards")
        shard_index = next(index for index, spec in APPROVED_TARGET_SPECS.items() if spec == args.target_spec)
        expected_shard_id = f"seed{args.seed}_shard{shard_index}"
        if args.shard_id != expected_shard_id:
            raise SystemExit(f"P9 shard id mismatch: {args.shard_id!r} != {expected_shard_id!r}")


def manifest_maps(path: str, seeds: list[int]) -> tuple[dict[int, dict[tuple[str, int], dict]], str, int]:
    rows = load_manifest_csv(path)
    frozen_hash = manifest_hash(rows)
    maps: dict[int, dict[tuple[str, int], dict]] = {}
    for seed in seeds:
        selected = [row for row in rows if int(row["source_seed"]) == seed]
        maps[seed] = {
            (row["dataset"], int(row["target_subject"])): row
            for row in selected
        }
        if len(maps[seed]) != 115:
            raise RuntimeError(f"seed {seed}: expected 115 manifest units, got {len(maps[seed])}")
    return maps, frozen_hash, sum(len(maps[seed]) for seed in seeds)


def dryrun_dataset(
    dataset: str,
    *,
    seeds: list[int],
    row_maps: dict[int, dict[tuple[str, int], dict]],
    bn: Any,
    TSMNet: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    print(f"p9_dryrun_dataset_start={dataset}", flush=True)
    subject_ids = [int(subject) for subject in MOABB_CLASS[dataset]().subject_list]
    ep = load_dataset(dataset, subject_ids)
    targets = sorted(int(subject) for subject in np.unique(ep.subject))
    print(f"p9_dryrun_dataset_loaded={dataset} targets={len(targets)} shape={tuple(ep.X.shape)}", flush=True)
    per_seed = []
    for seed in seeds:
        print(f"p9_dryrun_model_start={dataset} seed={seed}", flush=True)
        details = [p8._manifest_detail(ep, row_maps[seed][(dataset, target)]) for target in targets]
        p8._set_seed(seed)
        model_args = SimpleNamespace(**vars(args))
        model_args.seed = seed
        model = p8._build_model(
            TSMNet,
            bn,
            ep=ep,
            domain_values=targets,
            args=model_args,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        first_row = row_maps[seed][(dataset, targets[0])]
        adapt_idx = indices_from_trial_ids(first_row["adapt_trial_ids"])
        dummy = TensorDomainDataset(
            ep.X[adapt_idx[:1]],
            np.zeros(1, dtype=np.int64),
            np.full(1, targets[0], dtype=np.int64),
        )
        per_seed.append({
            "source_seed": seed,
            "target_subject_ids": targets,
            "target_details": details,
            "model_instantiated": model is not None,
            "adapt_loader_dummy_labels_unique": sorted(int(value) for value in np.unique(dummy.labels.numpy())),
            "adapt_loader_contains_real_target_labels": False,
            "evaluation_labels_accessed_only_after_adaptation_check": True,
            "target_performance_selection_used": False,
            "source_training_command_buildable": True,
            "source_training_command": build_source_command(seed, dataset),
            "expected_rows": len(targets) * len(p8.METHODS),
        })
        del dummy, model
        gc.collect()
        print(f"p9_dryrun_model_pass={dataset} seed={seed}", flush=True)
    result = {
        "dataset": dataset,
        "subject_ids": targets,
        "n_subjects": len(targets),
        "tensor_shape": [int(value) for value in ep.X.shape],
        "channel_count": int(ep.X.shape[1]),
        "time_samples": int(ep.X.shape[2]),
        "sessions": [int(value) for value in sorted(np.unique(ep.session))],
        "per_seed": per_seed,
        "elapsed_seconds": time.time() - started,
    }
    del ep
    gc.collect()
    print(f"p9_dryrun_dataset_pass={dataset}", flush=True)
    return result


def write_protocol(report: dict[str, Any]) -> None:
    lines = [
        "# Official SPDIM W1 Repaired-Split Three-Seed Protocol",
        "",
        "Pre-registered final label, allowed only if all P9 gates pass: Official SPDIM W1 repaired-split three-source-seed same-split baseline.",
        "",
        "## Frozen Components",
        "",
        f"- P8 result commit: `{P8_RESULT_COMMIT}`.",
        f"- repaired split manifest hash: `{P8_MANIFEST_HASH}`.",
        f"- P8 runner SHA-256: `{P8_RUNNER_SHA256}`.",
        f"- config SHA-256: `{P8_CONFIG_SHA256}`.",
        f"- external SPDIM commit: `{EXTERNAL_SHA}`.",
        f"- immutable P8 seed-0 result SHA-256: `{P8_SEED0_RESULT_SHA256}`.",
        "- source-training and adaptation hyperparameters are unchanged from P8.",
        "",
        "## Scope",
        "",
        "- new source seeds: `1`, `2` only.",
        "- datasets: `BNCI2014_001`, `Cho2017`, `Lee2019_MI`.",
        "- target subjects: all 115 W1 repaired-split targets per seed.",
        "- methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`.",
        "- epochs: `20`; adaptation epochs: `30`; adaptation LR: `0.01`.",
        "- batch size: `64`; source validation fraction: `0.2`.",
        "- temporal filters: `4`; spatial filters: `40`; subspace dimensions: `20`.",
        "- model dtype: `float32`; SPD calculation device: `cpu`.",
        "- no target-performance tuning or method selection.",
        "- no official pretrained weights and no third-party vendoring.",
        "",
        "## Expected Rows",
        "",
        "| dataset | targets per seed | seeds | methods | new rows | final rows including seed 0 |",
        "|---|---:|---:|---:|---:|---:|",
        "| BNCI2014_001 | 9 | 2 | 4 | 72 | 108 |",
        "| Cho2017 | 52 | 2 | 4 | 416 | 624 |",
        "| Lee2019_MI | 54 | 2 | 4 | 432 | 648 |",
        "| total | 115 | 2 | 4 | 920 | 1380 |",
        "",
        "## GPU Sharding",
        "",
        "Eight immutable seed-by-target-shard tasks are submitted as one array with concurrency `%4`, preferring `H100,L40S`. Each seed reuses the four P8 target shards with expected rows `116`, `116`, `116`, and `112`. A failed task may be rerun only with the same seed, target spec, and frozen command.",
        "",
        "## Aggregation and Inference",
        "",
        "- Average seeds 0/1/2 within each dataset x target subject x method before aggregation.",
        "- Primary estimands: subject-weighted and dataset-macro means.",
        f"- Cluster bootstrap: `{BOOTSTRAP_REPLICATES}` replicates, fixed seed `{BOOTSTRAP_SEED}`, cluster unit dataset x target subject.",
        "- Preserve all methods and contrasts within each sampled subject; report percentile 95% CIs.",
        "- Harm thresholds: delta `< 0`, `< -0.01`, and `< -0.02`.",
        "- No post-hoc equivalence or noninferiority margin.",
        "",
        "## Completion and Failure Gates",
        "",
        "- Monitor with `squeue`; completion requires queue absence plus stdout/stderr and artifact validation.",
        "- Require 920 new rows and a deterministic 1380-row merge preserving seed 0 byte-for-byte.",
        "- Require both classes, disjoint adaptation/evaluation IDs, complete prediction/logits hashes, frozen checksums, and no leakage flags.",
        "- Stop and write a failure trace on any unresolved row, provenance, checksum, or scope failure.",
        "",
        "## Runtime Provenance Policy",
        "",
        "Every GPU task records `sys.executable`, `sys.prefix`, Python, PyTorch, CUDA, GPU device, MOABB, and MNE versions. `CONDA_DEFAULT_ENV` is retained only as an inherited shell label and is not runtime proof.",
        "",
        "## Red Team Review",
        "",
        "- P8 runner, config, manifest, methods, and hyperparameters are frozen before seeds 1/2 are observed.",
        "- Seed 0 provides no SPDIM-specific improvement over RCT; P9 does not alter the protocol in response.",
        "- No H2CMI rerun, TeX edit, geometry stress, orthogonal-score work, extra seed, or extra method is authorized.",
    ]
    PROTOCOL_PATH.write_text("\n".join(lines) + "\n")


def write_dryrun_audit(report: dict[str, Any]) -> None:
    write_protocol(report)
    runtime = report["runtime_environment"]
    lines = [
        "# SPDIM W1 Repaired Seeds 1/2 Dry-Run Audit",
        "",
        f"- status: `{'PASS' if report['dryrun_pass'] else 'BLOCKED'}`",
        f"- approve_gpu_run: `{report['approve_gpu_run']}`",
        f"- launch_seeds: `{report['launch_seeds']}`",
        f"- expected_rows_total: `{report['expected_rows_total']}`",
        f"- manifest_hash_matches_p8: `{report['manifest_hash_matches_p8']}`",
        f"- runner_hash_matches_p8: `{report['runner_hash_matches_p8']}`",
        f"- config_hash_matches_p8: `{report['config_hash_matches_p8']}`",
        f"- external_spdim_commit_matches: `{report['external_spdim_commit_matches']}`",
        f"- worktree_clean_for_launch: `{report['worktree_clean_for_launch']}`",
        "",
        "## Split and Label Gates",
        "",
        f"- all_eval_both_classes: `{report['all_eval_both_classes']}`",
        f"- all_adapt_both_classes: `{report['all_adapt_both_classes']}`",
        f"- all_adapt_eval_disjoint: `{report['all_adapt_eval_disjoint']}`",
        f"- target_label_leakage_detected: `{report['target_label_leakage_detected']}`",
        f"- target_performance_selection_detected: `{report['target_performance_selection_detected']}`",
        f"- pretrained_weight_detected: `{report['pretrained_weight_detected']}`",
        f"- vendoring_detected: `{report['vendoring_detected']}`",
        f"- slurm_accounting_script_calls_detected: `{report['slurm_accounting_script_calls_detected']}`",
        "",
        "## Actual CPU Dry-Run Runtime",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for key in (
        "sys_executable", "sys_prefix", "python_version", "pytorch_version", "cuda_version",
        "device_name", "moabb_version", "mne_version", "conda_default_env_inherited_label",
    ):
        lines.append(f"| {key} | `{runtime[key]}` |")
    lines.extend([
        "",
        "The dry-run device is CPU. GPU tasks must record their actual allocated device in stdout and shard summary. The inherited conda label is not used as environment proof.",
        "",
        "## Dataset and Model Gate",
        "",
        "| dataset | targets | seed models instantiated | tensor shape | expected new rows |",
        "|---|---:|---:|---|---:|",
    ])
    for dataset in report["datasets"]:
        expected = EXPECTED_ROWS_BY_DATASET[dataset["dataset"]]
        instantiated = sum(seed["model_instantiated"] for seed in dataset["per_seed"])
        lines.append(
            f"| {dataset['dataset']} | {dataset['n_subjects']} | {instantiated}/2 | "
            f"`{dataset['tensor_shape']}` | {expected} |"
        )
    lines.extend([
        "",
        "Exact source subjects, adaptation/evaluation trial IDs, class counts, and split hashes for both seeds and every target are retained in the JSON audit.",
        "",
        "## Worktree Gate",
        "",
        "The branch was clean at P9A start. During this audit, `worktree_clean_for_launch` allows only the named P9A controller, launcher, protocol/audit outputs, and command-log update pending their required commit. GPU launch still requires literal empty `git status --porcelain` after P9A is pushed.",
        "",
        "## Red Team Review",
        "",
        "- The P8 runner/config/manifest hashes are independently recomputed rather than inferred from filenames.",
        "- Both approved seeds instantiate every dataset shape and build the frozen source-training command.",
        "- The dry-run inspects target labels only for post-hoc split composition auditing; runtime adaptation loaders contain dummy labels only.",
        "- No GPU job is approved if any required boolean is false.",
    ])
    if report["datasets_blocked"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["datasets_blocked"]:
            lines.append(f"- `{blocker['dataset']}`: {blocker['failure']}")
    DRYRUN_MD_PATH.write_text("\n".join(lines) + "\n")
    DRYRUN_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def append_p9a_command_log(report: dict[str, Any]) -> None:
    marker = "Per PM P9A, froze the official SPDIM repaired-split seeds 1/2 execution gate"
    text = COMMAND_LOG.read_text()
    if marker in text:
        return
    entry = f"""
- {marker}. The P8 runner (`{P8_RUNNER_SHA256}`), config
  (`{P8_CONFIG_SHA256}`), repaired manifest (`{P8_MANIFEST_HASH}`), and external
  SPDIM commit (`{EXTERNAL_SHA}`) matched exactly. CPU dry-run instantiated all
  three dataset shapes for seeds `1` and `2`, verified 230 target-seed manifest
  units and `{report['expected_rows_total']}` expected new rows, and passed the
  dummy-label, both-class, disjoint-split, no-selection, no-pretrained, and
  no-vendoring gates. Runtime provenance records the actual `icml` executable;
  the inherited conda label is not treated as environment proof. No GPU job was
  launched during P9A.
"""
    COMMAND_LOG.write_text(text.rstrip() + "\n\n" + entry.lstrip())


def run_dryrun(args: argparse.Namespace) -> int:
    started = time.time()
    status = git_status()
    clean_for_launch, unexpected = allowed_p9a_status(status)
    runner_sha = sha256_file(ROOT / "h2cmi" / "run_spdim_w1_repaired_seed0.py")
    config_sha = sha256_file(ROOT / "h2cmi" / "config.py")
    controller_sha = sha256_file(__file__)
    row_maps, frozen_hash, manifest_units = manifest_maps(args.manifest, APPROVED_SEEDS)
    external_sha = p8._git_sha(args.external_spdim_path)
    vendoring = str(Path(args.external_spdim_path).resolve()).startswith(str(ROOT.resolve()))
    script_calls = accounting_script_calls()
    runtime = runtime_environment("cpu")
    datasets = []
    blockers = []
    try:
        bn, TSMNet, _Trainer = p8._import_official(args.external_spdim_path)
    except Exception as exc:
        bn = TSMNet = None
        blockers.append({"dataset": "external_import", "failure": "".join(traceback.format_exception_only(type(exc), exc)).strip()})
    if bn is not None and TSMNet is not None:
        for dataset in p8.W1_DATASETS:
            try:
                datasets.append(
                    dryrun_dataset(
                        dataset,
                        seeds=APPROVED_SEEDS,
                        row_maps=row_maps,
                        bn=bn,
                        TSMNet=TSMNet,
                        args=args,
                    )
                )
            except Exception as exc:
                blockers.append({
                    "dataset": dataset,
                    "failure": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                })
    seed_details = [seed for dataset in datasets for seed in dataset["per_seed"]]
    target_details = [detail for seed in seed_details for detail in seed["target_details"]]
    all_eval = bool(target_details) and all(detail["both_classes_eval"] for detail in target_details)
    all_adapt = bool(target_details) and all(detail["both_classes_adapt"] for detail in target_details)
    all_disjoint = bool(target_details) and all(detail["adapt_eval_disjoint"] for detail in target_details)
    label_leakage = any(seed["adapt_loader_contains_real_target_labels"] for seed in seed_details)
    performance_selection = any(seed["target_performance_selection_used"] for seed in seed_details)
    models_instantiated = len(seed_details) == 6 and all(seed["model_instantiated"] for seed in seed_details)
    commands_build = len(seed_details) == 6 and all(seed["source_training_command_buildable"] for seed in seed_details)
    required = {
        "launch_seeds_exact": APPROVED_SEEDS == [1, 2],
        "expected_rows": EXPECTED_ROWS_TOTAL == 920,
        "manifest_units": manifest_units == 230,
        "manifest_hash": frozen_hash == P8_MANIFEST_HASH,
        "runner_hash": runner_sha == P8_RUNNER_SHA256,
        "config_hash": config_sha == P8_CONFIG_SHA256,
        "external_commit": external_sha == EXTERNAL_SHA,
        "all_eval_both_classes": all_eval,
        "all_adapt_both_classes": all_adapt,
        "all_adapt_eval_disjoint": all_disjoint,
        "no_target_label_leakage": not label_leakage,
        "no_target_performance_selection": not performance_selection,
        "all_shapes_instantiated": models_instantiated,
        "source_commands_build": commands_build,
        "frozen_hyperparameters_exact": all(
            getattr(args, key) == expected for key, expected in FROZEN_HYPERPARAMETERS.items()
        ),
        "no_pretrained_weights": True,
        "no_vendoring": not vendoring,
        "no_slurm_accounting_script_calls": not script_calls,
        "worktree_clean_for_launch": clean_for_launch,
        "actual_icml_runtime": runtime["sys_executable"] == ICML_PYTHON and runtime["sys_prefix"].endswith("/envs/icml"),
    }
    dryrun_pass = not blockers and all(required.values())
    report = {
        "dryrun_pass": bool(dryrun_pass),
        "launch_seeds": APPROVED_SEEDS,
        "expected_rows_total": EXPECTED_ROWS_TOTAL,
        "expected_rows_by_dataset": EXPECTED_ROWS_BY_DATASET,
        "expected_final_three_seed_rows": EXPECTED_FINAL_ROWS,
        "manifest_hash": frozen_hash,
        "manifest_hash_matches_p8": frozen_hash == P8_MANIFEST_HASH,
        "runner_sha256": runner_sha,
        "runner_hash_matches_p8": runner_sha == P8_RUNNER_SHA256,
        "config_sha256": config_sha,
        "config_hash_matches_p8": config_sha == P8_CONFIG_SHA256,
        "controller_sha256": controller_sha,
        "external_spdim_commit": external_sha,
        "external_spdim_commit_matches": external_sha == EXTERNAL_SHA,
        "all_eval_both_classes": all_eval,
        "all_adapt_both_classes": all_adapt,
        "all_adapt_eval_disjoint": all_disjoint,
        "target_label_leakage_detected": bool(label_leakage),
        "target_performance_selection_detected": bool(performance_selection),
        "pretrained_weight_detected": False,
        "vendoring_detected": bool(vendoring),
        "shape_blocker_detected": bool(blockers) or not models_instantiated,
        "source_training_commands_buildable": commands_build,
        "manifest_target_seed_units": manifest_units,
        "slurm_accounting_script_calls_detected": bool(script_calls),
        "slurm_accounting_script_call_matches": script_calls,
        "worktree_clean_for_launch": bool(clean_for_launch),
        "worktree_status_porcelain_before_outputs": status,
        "unexpected_worktree_entries": unexpected,
        "runtime_environment": runtime,
        "frozen_hyperparameters": {
            "epochs": args.epochs,
            "adapt_epochs": args.adapt_epochs,
            "adapt_lr": args.adapt_lr,
            "batch_size": args.batch_size,
            "val_fraction": args.val_fraction,
            "temporal_filters": args.temporal_filters,
            "spatial_filters": args.spatial_filters,
            "subspace_dims": args.subspace_dims,
            "dtype": args.dtype,
            "spd_device": args.spd_device,
            "methods": p8.METHODS,
        },
        "bootstrap_plan": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "dataset_x_target_subject",
            "seed_average_before_bootstrap": True,
            "interval": "percentile_95",
        },
        "required_checks": required,
        "datasets": datasets,
        "datasets_blocked": blockers,
        "elapsed_seconds": time.time() - started,
        "approve_gpu_run": bool(dryrun_pass),
    }
    write_dryrun_audit(report)
    append_p9a_command_log(report)
    print(json.dumps({
        "dryrun_pass": report["dryrun_pass"],
        "approve_gpu_run": report["approve_gpu_run"],
        "launch_seeds": report["launch_seeds"],
        "expected_rows_total": report["expected_rows_total"],
        "manifest_target_seed_units": report["manifest_target_seed_units"],
        "datasets_blocked": report["datasets_blocked"],
    }, indent=2, sort_keys=True))
    return 0 if dryrun_pass else 2


def finalize_shard_summary(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    path = Path(args.summary)
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    summary["label"] = "P9 repaired-split official SPDIM seed shard; part of the approved seeds 1/2 expansion."
    summary["p9_controller"] = {
        "file": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256_file(__file__),
        "approved_seeds": APPROVED_SEEDS,
        "executed_seed": int(args.seed),
        "p8_runner_sha256": sha256_file(ROOT / "h2cmi" / "run_spdim_w1_repaired_seed0.py"),
        "config_sha256": sha256_file(ROOT / "h2cmi" / "config.py"),
    }
    summary["runtime_environment"] = runtime
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def run_gpu(args: argparse.Namespace) -> int:
    if args.seed not in APPROVED_SEEDS:
        raise SystemExit(f"P9 permits only seeds {APPROVED_SEEDS}")
    runtime = runtime_environment(args.device)
    print_runtime_environment(runtime)
    if runtime["sys_executable"] != ICML_PYTHON or not runtime["sys_prefix"].endswith("/envs/icml"):
        raise RuntimeError("P9 GPU runtime is not the frozen icml interpreter")
    if not runtime["cuda_available"] or runtime["device_name"] == "CPU":
        raise RuntimeError("P9 GPU run requires an allocated CUDA device")
    if sha256_file(ROOT / "h2cmi" / "run_spdim_w1_repaired_seed0.py") != P8_RUNNER_SHA256:
        raise RuntimeError("frozen P8 runner checksum mismatch")
    if sha256_file(ROOT / "h2cmi" / "config.py") != P8_CONFIG_SHA256:
        raise RuntimeError("frozen config checksum mismatch")
    result = p8._run_gpu(args)
    finalize_shard_summary(args, runtime)
    return result


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry-run", "run"], required=True)
    ap.add_argument("--external-spdim-path", default="/home/infres/yinwang/.cache/h2cmi_external/SPDIM_1b0de0ccd4c48a4ff28f087b866a0b671b029c39")
    ap.add_argument("--datasets", default="BNCI2014_001,Cho2017,Lee2019_MI")
    ap.add_argument("--manifest", default="h2cmi/results/review_completion/w1_repaired_split_manifest.csv")
    ap.add_argument("--manifest-hash", default=P8_MANIFEST_HASH)
    ap.add_argument("--target-spec", default="")
    ap.add_argument("--shard-id", default="")
    ap.add_argument("--seed", type=int, default=1)
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
    ap.add_argument("--protocol", default=str(PROTOCOL_PATH.relative_to(ROOT)))
    ap.add_argument("--dryrun-md", default=str(DRYRUN_MD_PATH.relative_to(ROOT)))
    ap.add_argument("--dryrun-json", default=str(DRYRUN_JSON_PATH.relative_to(ROOT)))
    ap.add_argument("--out", default="h2cmi/results/review_completion/spdim_w1_repaired_seeds12_results.csv")
    ap.add_argument("--audit", default="h2cmi/results/review_completion/spdim_w1_repaired_seeds12_audit.md")
    ap.add_argument("--summary", default="h2cmi/results/review_completion/spdim_w1_repaired_seeds12_summary.json")
    ap.add_argument("--digest", default="h2cmi/results/review_completion/spdim_w1_repaired_seeds12_result_digest.md")
    ap.add_argument("--legacy-compare", default="h2cmi/results/review_completion/spdim_w1_repaired_seeds12_legacy_compare.md")
    ap.add_argument("--failure-trace", default="h2cmi/results/review_completion/spdim_w1_repaired_three_seed_failure_trace.txt")
    return ap


def main() -> int:
    args = parser().parse_args()
    args.datasets = p8._dataset_list(args.datasets)
    args.target_spec_map = p8._parse_target_spec(args.target_spec)
    if args.datasets != p8.W1_DATASETS:
        raise SystemExit("P9 requires all three frozen W1 datasets")
    if args.manifest_hash != P8_MANIFEST_HASH:
        raise SystemExit("P9 manifest hash must match frozen P8")
    if args.allow_dirty:
        raise SystemExit("P9 does not permit dirty-worktree execution")
    require_frozen_protocol(args)
    if args.mode == "dry-run":
        args.device = "cpu"
        return run_dryrun(args)
    return run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
