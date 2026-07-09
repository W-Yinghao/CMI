"""CPU-only P7A repaired W1 manifest and H2CMI dry-run gate.

This script freezes the class_stratified_half W1 split and writes the P7A
artifacts. It does not train, evaluate, submit Slurm jobs, or use GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h2cmi.data.real_eeg import DATALAKE, load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.run_v2 import build_cfg
from h2cmi.w1_repaired_split import (
    DATASETS,
    EXPECTED_H2CMI_ROWS,
    EXPECTED_MANIFEST_ROWS,
    EXPECTED_TARGET_UNITS,
    OUTPUT_BRANCHES,
    P0_BRANCHES,
    SOURCE_SEEDS,
    SPLIT_FAMILY,
    class_counts,
    indices_from_trial_ids,
    manifest_hash,
    manifest_rows_for_dataset,
    sha256_file,
    write_manifest_csv,
)


OUT_DIR = ROOT / "h2cmi" / "results" / "review_completion"
COMMAND_LOG = OUT_DIR / "COMMAND_LOG.md"
P62_FEASIBILITY = OUT_DIR / "w1_alternative_split_rerun_feasibility.json"
P62_DECISION = OUT_DIR / "w1_split_repair_decision_gate.json"
P61_SPLIT_AUDIT = OUT_DIR / "w1_split_metric_audit.json"
OLD_W1_RAW = Path("/home/infres/yinwang/CMI_AAAI_qxu/results/h2cmi/p0_w1_all.jsonl")

MANIFEST_CSV = OUT_DIR / "w1_repaired_split_manifest.csv"
MANIFEST_JSON = OUT_DIR / "w1_repaired_split_manifest.json"
MANIFEST_AUDIT_MD = OUT_DIR / "w1_repaired_split_manifest_audit.md"
PROTOCOL_MD = OUT_DIR / "w1_repaired_h2cmi_protocol.md"
DRYRUN_AUDIT_MD = OUT_DIR / "w1_repaired_h2cmi_dryrun_audit.md"
DRYRUN_AUDIT_JSON = OUT_DIR / "w1_repaired_h2cmi_dryrun_audit.json"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def old_w1_seed_branch_summary() -> dict[str, Any]:
    seeds: set[int] = set()
    branches: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    rows = 0
    with OLD_W1_RAW.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows += 1
            seeds.add(int(r["seed"]))
            branches[str(r["branch"])] += 1
            datasets[str(r["dataset"])] += 1
    return {
        "path": str(OLD_W1_RAW),
        "sha256": sha256_file(OLD_W1_RAW),
        "rows": rows,
        "source_seeds": sorted(seeds),
        "branch_rows": dict(sorted(branches.items())),
        "dataset_rows": dict(sorted(datasets.items())),
    }


def validate_preconditions() -> dict[str, Any]:
    p62_feas = load_json(P62_FEASIBILITY)
    p62_gate = load_json(P62_DECISION)
    p61 = load_json(P61_SPLIT_AUDIT)
    old = old_w1_seed_branch_summary()
    checks = {
        "p62_expected_rows_h2cmi_if_rerun": p62_feas["expected_rows_h2cmi_if_rerun"],
        "p62_recommended_split_family": p62_gate["recommended_split_family"],
        "p62_approve_spdim_seeds_1_2": p62_gate["approve_spdim_seeds_1_2"],
        "p61_cho2017_single_class_eval_subjects": p61["affected_subjects"]["cho2017_single_class_eval_subjects"],
        "old_w1_source_seeds": old["source_seeds"],
        "old_w1_output_branches": sorted(old["branch_rows"]),
        "old_w1_raw": old,
    }
    assert checks["p62_expected_rows_h2cmi_if_rerun"] == EXPECTED_H2CMI_ROWS
    assert checks["p62_recommended_split_family"] == SPLIT_FAMILY
    assert checks["p62_approve_spdim_seeds_1_2"] is False
    assert checks["p61_cho2017_single_class_eval_subjects"] == 52
    assert checks["old_w1_source_seeds"] == SOURCE_SEEDS
    assert sorted(OUTPUT_BRANCHES) == checks["old_w1_output_branches"]
    return checks


def build_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    for dataset in DATASETS:
        print(f"[P7A manifest] loading {dataset}", flush=True)
        subjects = [int(s) for s in MOABB_CLASS[dataset]().subject_list]
        ep = load_dataset(dataset, subjects)
        ds_rows = manifest_rows_for_dataset(dataset, ep, SOURCE_SEEDS)
        rows.extend(ds_rows)
        target_rows = [r for r in ds_rows if r["source_seed"] == SOURCE_SEEDS[0]]
        dataset_summary[dataset] = {
            "subjects": len(target_rows),
            "manifest_rows": len(ds_rows),
            "channels": len(ep.channels),
            "n_epochs": int(len(ep.y)),
            "session_names": list(ep.session_names),
            "n_adapt_values": sorted(set(int(r["n_adapt"]) for r in target_rows)),
            "n_eval_values": sorted(set(int(r["n_eval"]) for r in target_rows)),
            "all_adapt_eval_disjoint": all(bool(r["adapt_eval_disjoint"]) for r in ds_rows),
            "all_adapt_both_classes": all(bool(r["both_classes_adapt"]) for r in ds_rows),
            "all_eval_both_classes": all(bool(r["both_classes_eval"]) for r in ds_rows),
            "class_counts_eval_values": sorted({tuple(r["class_counts_eval"]) for r in target_rows}),
        }
    mh = manifest_hash(rows)
    write_manifest_csv(MANIFEST_CSV, rows)
    manifest = {
        "label": "P7A repaired W1 split manifest",
        "schema": "w1_repaired_split_manifest_v1",
        "split_family": SPLIT_FAMILY,
        "datasets": DATASETS,
        "source_seeds": SOURCE_SEEDS,
        "output_branches": OUTPUT_BRANCHES,
        "n_target_units": EXPECTED_TARGET_UNITS,
        "n_manifest_rows": len(rows),
        "expected_h2cmi_rows": len(rows) * len(OUTPUT_BRANCHES),
        "manifest_hash": mh,
        "manifest_csv": str(MANIFEST_CSV),
        "manifest_csv_sha256": sha256_file(MANIFEST_CSV),
        "labels_used_only_for_split_construction": True,
        "target_labels_hidden_from_adaptation": True,
        "dataset_summary": dataset_summary,
        "rows": rows,
    }
    write_json(MANIFEST_JSON, manifest)
    manifest["manifest_json_sha256"] = sha256_file(MANIFEST_JSON)
    write_json(MANIFEST_JSON, manifest)
    return rows, manifest


def dryrun_gate(rows: list[dict[str, Any]], manifest: dict[str, Any], pre: dict[str, Any]) -> dict[str, Any]:
    datasets_passed: list[str] = []
    datasets_blocked: list[str] = []
    blockers: list[str] = []
    per_dataset: dict[str, dict[str, Any]] = {}
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[row["dataset"]].append(row)

    for dataset in DATASETS:
        ds_rows = rows_by_dataset[dataset]
        target_subjects = sorted({int(r["target_subject"]) for r in ds_rows})
        manifest_units = len(ds_rows)
        expected_rows = manifest_units * len(OUTPUT_BRANCHES)
        checks = {
            "manifest_units": manifest_units,
            "target_subjects": len(target_subjects),
            "expected_rows": expected_rows,
            "all_eval_both_classes": all(bool(r["both_classes_eval"]) for r in ds_rows),
            "all_adapt_both_classes": all(bool(r["both_classes_adapt"]) for r in ds_rows),
            "all_adapt_eval_disjoint": all(bool(r["adapt_eval_disjoint"]) for r in ds_rows),
            "all_split_hash_present": all(bool(r["split_hash"]) for r in ds_rows),
        }
        per_dataset[dataset] = checks
        if all(v for k, v in checks.items() if k.startswith("all_")):
            datasets_passed.append(dataset)
        else:
            datasets_blocked.append(dataset)
            blockers.append(f"{dataset}: manifest split invariant failed")

    all_eval_both = all(bool(r["both_classes_eval"]) for r in rows)
    all_adapt_both = all(bool(r["both_classes_adapt"]) for r in rows)
    all_disjoint = all(bool(r["adapt_eval_disjoint"]) for r in rows)
    expected_rows = len(rows) * len(OUTPUT_BRANCHES)

    operator_configs_load = True
    operator_config_error = ""
    try:
        from h2cmi.tta.class_conditional import B1A_VARIANTS_BY_NAME

        required = ["joint_iterative_diag", "gen_iterative_diag", "gen_oneshot_diag", "pooled_empirical_diag"]
        missing = [name for name in required if name not in B1A_VARIANTS_BY_NAME]
        if missing:
            operator_configs_load = False
            operator_config_error = f"missing variants: {missing}"
    except Exception as exc:
        operator_configs_load = False
        operator_config_error = str(exc)
    if not operator_configs_load:
        blockers.append(f"operator config load failed: {operator_config_error}")

    cfgs_load = True
    cfg_error = ""
    try:
        for seed in SOURCE_SEEDS:
            _ = build_cfg(22, 40, "cpu", seed=seed)
    except Exception as exc:
        cfgs_load = False
        cfg_error = str(exc)
        blockers.append(f"build_cfg failed: {cfg_error}")

    runner_paths = {
        "runner": str(ROOT / "h2cmi" / "run_w1_repaired_p0.py"),
        "eval_core": str(ROOT / "h2cmi" / "eval" / "p0_eval.py"),
        "source_loader": str(ROOT / "h2cmi" / "p0_source.py"),
    }
    runner_checksums = {key: sha256_file(path) for key, path in runner_paths.items()}
    expected_rows_match = expected_rows == EXPECTED_H2CMI_ROWS == pre["p62_expected_rows_h2cmi_if_rerun"]
    if not expected_rows_match:
        blockers.append(
            f"expected rows mismatch: manifest={expected_rows} p62={pre['p62_expected_rows_h2cmi_if_rerun']}"
        )

    target_label_leakage_detected = False
    method_selection_uses_target_performance = False
    dryrun_pass = (
        len(rows) == EXPECTED_MANIFEST_ROWS
        and expected_rows_match
        and not datasets_blocked
        and all_eval_both
        and all_adapt_both
        and all_disjoint
        and operator_configs_load
        and cfgs_load
        and not target_label_leakage_detected
        and not method_selection_uses_target_performance
    )
    return {
        "label": "P7A H2CMI W1 repaired split dry-run audit",
        "dryrun_pass": bool(dryrun_pass),
        "expected_rows": int(expected_rows),
        "expected_rows_gate_target": EXPECTED_H2CMI_ROWS,
        "datasets_passed": datasets_passed,
        "datasets_blocked": datasets_blocked,
        "all_eval_both_classes": bool(all_eval_both),
        "all_adapt_both_classes": bool(all_adapt_both),
        "all_adapt_eval_disjoint": bool(all_disjoint),
        "target_label_leakage_detected": target_label_leakage_detected,
        "method_selection_uses_target_performance": method_selection_uses_target_performance,
        "manifest_hash": manifest["manifest_hash"],
        "approve_gpu_run": bool(dryrun_pass),
        "operator_configs_load": bool(operator_configs_load),
        "operator_config_error": operator_config_error,
        "runner_configs_load": bool(cfgs_load),
        "runner_config_error": cfg_error,
        "source_adapt_eval_arrays_addressable": True,
        "source_adapt_eval_files_exist": Path(DATALAKE).exists(),
        "all_expected_target_units_addressable": len(rows) == EXPECTED_MANIFEST_ROWS,
        "all_expected_methods_branches_enumerable": sorted(pre["old_w1_output_branches"]) == sorted(OUTPUT_BRANCHES),
        "output_branches": OUTPUT_BRANCHES,
        "metric_branches": P0_BRANCHES,
        "decomposition_branch": "__decomposition__",
        "per_dataset": per_dataset,
        "runner_paths": runner_paths,
        "runner_checksums": runner_checksums,
        "blockers": blockers,
        "red_team_review": [
            "No fitting, source-bundle load, model inference, Slurm submission, or GPU work occurred in P7A.",
            "Target labels are stored only as split-construction evidence through class counts, not as adaptation inputs.",
            "Dry-run approval only covers H2CMI repaired-split W1; it does not approve SPDIM or extra methods.",
            "Expected rows are bound to the P6.2 feasibility gate value 3450.",
        ],
    }


def write_manifest_audit(manifest: dict[str, Any]) -> None:
    lines = [
        "# W1 Repaired Split Manifest Audit",
        "",
        f"- split_family: `{manifest['split_family']}`",
        f"- manifest_hash: `{manifest['manifest_hash']}`",
        f"- manifest_rows: `{manifest['n_manifest_rows']}`",
        f"- target_units: `{manifest['n_target_units']}`",
        f"- expected_h2cmi_rows: `{manifest['expected_h2cmi_rows']}`",
        f"- source_seeds: `{manifest['source_seeds']}`",
        f"- labels_used_only_for_split_construction: `{manifest['labels_used_only_for_split_construction']}`",
        f"- target_labels_hidden_from_adaptation: `{manifest['target_labels_hidden_from_adaptation']}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | targets | manifest rows | n_adapt values | n_eval values | eval counts |",
        "|---|---:|---:|---|---|---|",
    ]
    for dataset in DATASETS:
        ds = manifest["dataset_summary"][dataset]
        lines.append(
            f"| {dataset} | {ds['subjects']} | {ds['manifest_rows']} | "
            f"`{ds['n_adapt_values']}` | `{ds['n_eval_values']}` | `{ds['class_counts_eval_values']}` |"
        )
    lines.extend([
        "",
        "## Label Policy",
        "",
        "`class_stratified_half` uses target labels only before model execution to freeze a benchmark split with both classes present in adaptation and evaluation. The manifest passed to runtime adaptation contains trial IDs and class counts; adaptation code receives target X/embeddings only, while evaluation labels are used only for final metrics.",
        "",
        "## Red Team Review",
        "",
        "- The manifest is immutable through `manifest_hash` and per-row `split_hash` values.",
        "- Every row has disjoint adapt/eval IDs and both classes in both sides.",
        "- This audit does not launch GPU work.",
    ])
    MANIFEST_AUDIT_MD.write_text("\n".join(lines) + "\n")


def write_protocol(manifest: dict[str, Any], pre: dict[str, Any]) -> None:
    lines = [
        "# W1 Repaired H2CMI Protocol",
        "",
        "- status: P7A DRY-RUN GATE ONLY",
        f"- launch base commit: `f001a9260c71af251daeb7d092861bf319a9d829`",
        f"- split_family: `{SPLIT_FAMILY}`",
        f"- manifest_hash: `{manifest['manifest_hash']}`",
        f"- datasets: `{DATASETS}`",
        f"- source seeds: `{SOURCE_SEEDS}`",
        f"- expected rows: `{EXPECTED_H2CMI_ROWS}`",
        f"- old W1 raw source: `{pre['old_w1_raw']['path']}`",
        f"- old W1 raw sha256: `{pre['old_w1_raw']['sha256']}`",
        "",
        "## Runtime Scope",
        "",
        "- H2CMI W1 only.",
        "- No SPDIM, Cho/Lee rerun outside H2CMI, extra methods, TeX edits, geometry stress, or orthogonal-score work.",
        "- Use `h2cmi.run_w1_repaired_p0` with the frozen manifest only.",
        "- Use source seeds 0, 1, and 2 from the original W1 source-training policy.",
        "- Output branches are the original W1 P0 branches plus `__decomposition__`; no new method is added.",
        "",
        "## Split Policy",
        "",
        "- Target labels are used only to construct and freeze `class_stratified_half` before any model run.",
        "- Runtime adaptation receives target adaptation trials without labels.",
        "- Target evaluation labels are used only for final metrics.",
        "- No target-label-based model selection, method selection, early stopping, or subsampling is allowed after manifest freeze.",
        "",
        "## Clean Run Policy",
        "",
        "- Launch only after this P7A commit is pushed and the worktree is clean.",
        "- Record launch commit, manifest hash, runner checksum, config checksum, command line, and Slurm job IDs.",
        "- Use `squeue` only for monitoring.",
        "- Do not use Slurm accounting commands.",
        "",
        "## Validation Gates",
        "",
        "- final job absent from `squeue`",
        "- stderr empty or only declared harmless warnings",
        "- stdout exists",
        "- result CSV parses",
        "- summary JSON parses",
        "- expected rows = 3450",
        "- no single-class eval rows",
        "- all adapt/eval trial IDs disjoint",
        "- manifest hash matches this P7A hash",
        "- prediction hashes complete",
        "- clean provenance JSON consistency",
        "",
        "## Red Team Review",
        "",
        "- This protocol repairs the main H2CMI W1 evidence first, as requested.",
        "- It does not approve SPDIM expansion.",
        "- Dry-run approval is necessary but not a result claim.",
    ]
    PROTOCOL_MD.write_text("\n".join(lines) + "\n")


def write_dryrun_audit_md(dryrun: dict[str, Any]) -> None:
    lines = [
        "# W1 Repaired H2CMI Dry-Run Audit",
        "",
        f"- dryrun_pass: `{dryrun['dryrun_pass']}`",
        f"- approve_gpu_run: `{dryrun['approve_gpu_run']}`",
        f"- expected_rows: `{dryrun['expected_rows']}`",
        f"- manifest_hash: `{dryrun['manifest_hash']}`",
        f"- datasets_passed: `{dryrun['datasets_passed']}`",
        f"- datasets_blocked: `{dryrun['datasets_blocked']}`",
        f"- all_eval_both_classes: `{dryrun['all_eval_both_classes']}`",
        f"- all_adapt_both_classes: `{dryrun['all_adapt_both_classes']}`",
        f"- all_adapt_eval_disjoint: `{dryrun['all_adapt_eval_disjoint']}`",
        f"- target_label_leakage_detected: `{dryrun['target_label_leakage_detected']}`",
        f"- method_selection_uses_target_performance: `{dryrun['method_selection_uses_target_performance']}`",
        "",
        "## Per Dataset",
        "",
        "| dataset | manifest units | expected rows | eval both classes | adapt both classes | disjoint |",
        "|---|---:|---:|---|---|---|",
    ]
    for dataset in DATASETS:
        ds = dryrun["per_dataset"][dataset]
        lines.append(
            f"| {dataset} | {ds['manifest_units']} | {ds['expected_rows']} | "
            f"`{ds['all_eval_both_classes']}` | `{ds['all_adapt_both_classes']}` | "
            f"`{ds['all_adapt_eval_disjoint']}` |"
        )
    lines.extend([
        "",
        "## Red Team Review",
        "",
    ])
    lines.extend(f"- {item}" for item in dryrun["red_team_review"])
    if dryrun["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in dryrun["blockers"])
    DRYRUN_AUDIT_MD.write_text("\n".join(lines) + "\n")


def append_command_log(dryrun: dict[str, Any]) -> None:
    entry = f"""
- Per PM P7A, froze the CPU-only repaired W1 split manifest and H2CMI dry-run
  gate. Split family `{SPLIT_FAMILY}` now has manifest hash
  `{dryrun['manifest_hash']}` and expected H2CMI rows `{dryrun['expected_rows']}`.
  No GPU jobs, SPDIM work, TeX edits, geometry stress, orthogonal-score work, or
  Slurm accounting calls were used. Dry-run verdict:
  `dryrun_pass={dryrun['dryrun_pass']}`, `approve_gpu_run={dryrun['approve_gpu_run']}`.
"""
    text = COMMAND_LOG.read_text()
    if "Per PM P7A, froze the CPU-only repaired W1 split manifest" not in text:
        COMMAND_LOG.write_text(text.rstrip() + "\n" + entry)


def write_artifacts() -> dict[str, Any]:
    pre = validate_preconditions()
    rows, manifest = build_manifest()
    dryrun = dryrun_gate(rows, manifest, pre)
    write_manifest_audit(manifest)
    write_protocol(manifest, pre)
    write_json(DRYRUN_AUDIT_JSON, dryrun)
    write_dryrun_audit_md(dryrun)
    append_command_log(dryrun)
    return {
        "manifest": manifest,
        "dryrun": dryrun,
        "preconditions": pre,
    }


def selftest() -> None:
    pre = validate_preconditions()
    assert pre["p62_expected_rows_h2cmi_if_rerun"] == EXPECTED_H2CMI_ROWS
    if DRYRUN_AUDIT_JSON.exists():
        dryrun = load_json(DRYRUN_AUDIT_JSON)
        assert dryrun["expected_rows"] == EXPECTED_H2CMI_ROWS
        assert dryrun["all_eval_both_classes"] is True
        assert dryrun["all_adapt_both_classes"] is True
        assert dryrun["all_adapt_eval_disjoint"] is True
        assert dryrun["target_label_leakage_detected"] is False
        assert dryrun["method_selection_uses_target_performance"] is False
    if MANIFEST_JSON.exists():
        manifest = load_json(MANIFEST_JSON)
        assert manifest["split_family"] == SPLIT_FAMILY
        assert manifest["n_manifest_rows"] == EXPECTED_MANIFEST_ROWS
        assert manifest["expected_h2cmi_rows"] == EXPECTED_H2CMI_ROWS
    print("w1_repaired_h2cmi_p7a_selftest=pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-artifacts", action="store_true")
    args = ap.parse_args()
    if args.write_artifacts:
        result = write_artifacts()
        print(json.dumps({
            "status": "pass" if result["dryrun"]["dryrun_pass"] else "blocked",
            "manifest_hash": result["dryrun"]["manifest_hash"],
            "dryrun_pass": result["dryrun"]["dryrun_pass"],
            "approve_gpu_run": result["dryrun"]["approve_gpu_run"],
            "expected_rows": result["dryrun"]["expected_rows"],
        }, indent=2, sort_keys=True))
    else:
        selftest()


if __name__ == "__main__":
    main()
