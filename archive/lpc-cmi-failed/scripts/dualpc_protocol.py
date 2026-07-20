"""Generate the DualPC AAAI experiment protocol commands without submitting jobs.

The goal is reproducibility, not scheduling. The script prints shell commands for:
  1. current synthetic concept/null gates;
  2. source-only guarded selection;
  3. real-data LOSO instrumentation/null-safety runs;
  4. SCPS cohort-domain runs;
  5. a final DualPC readiness summary command.

Examples:
  python scripts/dualpc_protocol.py --profile smoke --device cpu
  python scripts/dualpc_protocol.py --profile paper --device cuda --seeds 0 1 2 --write scripts/dualpc_paper_cmds.sh
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path


PY = "/home/infres/yinwang/anaconda3/envs/icml/bin/python"


def q(x):
    return shlex.quote(str(x))


def cmd(parts, env=None):
    prefix = ""
    if env:
        prefix = " ".join(f"{k}={q(v)}" for k, v in env.items()) + " "
    return prefix + " ".join(q(p) for p in parts)


def out_path(out_dir, name):
    return str(Path(out_dir) / f"{name}.json")


def synthetic_cmd(out_dir, profile):
    if profile == "smoke":
        args = ["--n", 240, "--seeds", 2, "--epochs", 25, "--probe_epochs", 50,
                "--warmup", 8, "--bs", 128, "--n_inner", 2]
    else:
        args = ["--n", 700, "--seeds", 5, "--epochs", 80, "--probe_epochs", 150,
                "--warmup", 20, "--bs", 128, "--n_inner", 2]
    return cmd([PY, "synthetic/dualpc_validation.py",
                *args, "--lam", 0.1, "--gamma", 1.0,
                "--dgps", "null_prior", "concept", "all_three",
                "--out", out_path(out_dir, f"dualpc_{profile}_synthetic_gate")])


def regression_cmd():
    return cmd([PY, "scripts/dualpc_regression_checks.py"])


def selector_cmd(out_dir, dataset, seed, profile, device):
    if profile == "smoke":
        train = ["--epochs", 1, "--select_epochs", 1, "--select_probe_epochs", 1,
                 "--final_probe_epochs", 1, "--warmup", 1, "--n_inner", 1,
                 "--max_subjects", 4, "--resample", 64]
        backbone = "LogCov"
    else:
        train = ["--epochs", 200, "--select_epochs", 80, "--select_probe_epochs", 80,
                 "--final_probe_epochs", 80, "--warmup", 40, "--n_inner", 2, "--resample", 128]
        backbone = "EEGNet"
    tag = f"dualpc_{profile}_select_{dataset}_s{seed}"
    return cmd([PY, "-m", "cmi.run_lambda_select",
                "--dataset", dataset, "--backbone", backbone, "--method", "dualpc",
                "--lams", 0, 0.05, 0.1, "--gammas", 0.05, 0.1,
                "--select_rule", "guarded_probe", "--select_tolerance", 0.05,
                "--select_cond_weight", 1.0, "--select_pz_weight", 1.0, "--select_py_weight", 1.0,
                *train, "--bs", 64, "--device", device, "--seed", seed,
                "--out", out_path(out_dir, tag)],
               env={"MPLCONFIGDIR": "/tmp/matplotlib"})


def loso_cmd(out_dir, dataset, seed, profile, device):
    if profile == "smoke":
        train = ["--epochs", 1, "--warmup", 1, "--n_inner", 1, "--max_subjects", 3, "--resample", 64]
        backbone = "LogCov"
        null = ["--decoder_null_perms", 0]
    else:
        train = ["--epochs", 200, "--warmup", 40, "--n_inner", 2, "--resample", 128]
        backbone = "EEGNet"
        null = ["--decoder_null_perms", 20, "--decoder_null_quantile", 0.95]
    tag = f"dualpc_{profile}_loso_{dataset}_s{seed}"
    return cmd([PY, "-m", "cmi.run_loso",
                "--dataset", dataset, "--backbone", backbone,
                "--configs", "erm:0", "lpc_prior:0.1", "dualc:0.1:0.05",
                "dualpc:0.1:0.05", "dualpc_marginal:0.1:0.05",
                *train, *null, "--bs", 64, "--device", device, "--seed", seed,
                "--out", out_path(out_dir, tag)],
               env={"MPLCONFIGDIR": "/tmp/matplotlib"})


def scps_cmd(out_dir, condition, seed, profile, device):
    if profile == "smoke":
        train = ["--epochs", 1, "--warmup", 1, "--n_inner", 1]
        backbone = "LogCov"
        null = ["--decoder_null_perms", 0]
    else:
        train = ["--epochs", 120, "--warmup", 30, "--n_inner", 2]
        backbone = "EEGNet"
        null = ["--decoder_null_perms", 20, "--decoder_null_quantile", 0.95]
    tag = f"dualpc_{profile}_scps_{condition}_s{seed}"
    return cmd([PY, "-m", "cmi.run_scps_crossdataset",
                "--condition", condition, "--backbone", backbone,
                "--domain", "cohort", "--dec_domain", "cohort",
                "--configs", "erm:0", "lpc_prior:0.1", "dualc:0.1:0.05",
                "dualpc:0.1:0.05", "dualpc_marginal:0.1:0.05",
                *train, *null, "--bs", 64, "--device", device, "--seed", seed,
                "--out", out_path(out_dir, tag)],
               env={"MPLCONFIGDIR": "/tmp/matplotlib"})


def readiness_cmd(out_dir):
    return cmd([PY, "analysis/dualpc_readiness.py", str(Path(out_dir) / "*.json"),
                "--out-json", out_path(out_dir, "dualpc_readiness")])


def paper_summary_cmd(out_dir):
    return cmd([PY, "analysis/dualpc_paper_summary.py", str(Path(out_dir) / "*.json"),
                "--out-json", out_path(out_dir, "dualpc_paper_summary")])


def decision_cmd(out_dir, min_comparison_tasks, min_selector_tasks):
    return cmd([PY, "analysis/dualpc_decision_gate.py",
                "--readiness", out_path(out_dir, "dualpc_readiness"),
                "--paper-summary", out_path(out_dir, "dualpc_paper_summary"),
                "--min-comparison-tasks", min_comparison_tasks,
                "--min-selector-tasks", min_selector_tasks,
                "--required-baselines", "erm", "lpc_prior",
                "--required-synthetic-groups", "null_prior", "concept", "all_three",
                "--out-json", out_path(out_dir, "dualpc_decision")])


def build(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    commands = []
    commands.append(("# Fast DualPC regression checks", regression_cmd()))
    commands.append(("# Synthetic current-code gate", synthetic_cmd(args.out_dir, args.profile)))
    for seed in args.seeds:
        for dataset in args.loso_datasets:
            commands.append((f"# LOSO {dataset} seed {seed}",
                             loso_cmd(args.out_dir, dataset, seed, args.profile, args.device)))
            commands.append((f"# Source-only selector {dataset} seed {seed}",
                             selector_cmd(args.out_dir, dataset, seed, args.profile, args.device)))
        for condition in args.scps_conditions:
            commands.append((f"# SCPS {condition} seed {seed}",
                             scps_cmd(args.out_dir, condition, seed, args.profile, args.device)))
    commands.append(("# Readiness summary after the JSONs above exist", readiness_cmd(args.out_dir)))
    commands.append(("# Paper aggregate summary after readiness", paper_summary_cmd(args.out_dir)))
    min_tasks = len(args.loso_datasets) + len(args.scps_conditions)
    commands.append(("# Headline decision gate after summaries",
                     decision_cmd(args.out_dir, min_tasks, len(args.loso_datasets))))
    return commands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default="results/dualpc_protocol")
    ap.add_argument("--loso-datasets", nargs="+", default=["BNCI2014_001"])
    ap.add_argument("--scps-conditions", nargs="+", default=["PD", "SCZ"])
    ap.add_argument("--write", default="", help="Optional shell script path to write")
    args = ap.parse_args()

    commands = build(args)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for comment, command in commands:
        lines.extend([comment, command, ""])
    text = "\n".join(lines)
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(text)
        print(f"wrote -> {args.write}")
    else:
        print(text)


if __name__ == "__main__":
    main()
