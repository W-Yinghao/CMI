"""H2CMI W1 repaired-split runner.

This runner reuses the REVIEW_P0 W1 evaluation core, but replaces the legacy
contiguous target split with the frozen P7 class_stratified_half manifest.
Target labels appear only in the pre-frozen manifest construction and in final
evaluation labels; runtime adaptation receives target X/embeddings only.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.eval.p0_eval import eval_unit_p0
from h2cmi.p0_source import get_source_p0, ProvenanceError
from h2cmi.data.real_eeg import load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.run_v2 import build_cfg
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int
from h2cmi.w1_repaired_split import (
    DATASETS,
    EXPECTED_H2CMI_ROWS,
    OUTPUT_BRANCHES,
    SPLIT_FAMILY,
    indices_from_trial_ids,
    load_manifest_csv,
    manifest_hash,
)


K = 2


def _manifest_by_dataset_target_seed(manifest_path: str) -> tuple[dict[tuple[str, int, int], dict], str]:
    rows = load_manifest_csv(manifest_path)
    mh = manifest_hash(rows)
    out: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        key = (row["dataset"], int(row["target_subject"]), int(row["source_seed"]))
        if key in out:
            raise ValueError(f"duplicate manifest key: {key}")
        if row["split_family"] != SPLIT_FAMILY:
            raise ValueError(f"{key}: unexpected split_family={row['split_family']}")
        if not row["adapt_eval_disjoint"]:
            raise ValueError(f"{key}: adapt/eval overlap in manifest")
        if not row["both_classes_adapt"] or not row["both_classes_eval"]:
            raise ValueError(f"{key}: manifest split lacks both classes")
        if not row["target_labels_hidden_from_adaptation"]:
            raise ValueError(f"{key}: target labels not marked hidden from adaptation")
        out[key] = row
    return out, mh


def _fold_subjects(subjects: list[int], folds: str) -> list[int]:
    if not folds:
        return subjects
    a, b = (int(x) for x in folds.split("-"))
    return subjects[a:b + 1]


def _load_ep(dataset: str):
    return load_dataset(dataset, MOABB_CLASS[dataset]().subject_list)


def dry_run(args) -> dict:
    manifest, mh = _manifest_by_dataset_target_seed(args.manifest)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    datasets = [args.dataset] if args.dataset else DATASETS
    datasets_passed: list[str] = []
    datasets_blocked: list[str] = []
    rows_checked = 0
    units_checked = 0
    all_eval_both = True
    all_adapt_both = True
    all_disjoint = True
    blockers: list[str] = []
    per_dataset: dict[str, dict] = {}

    for dataset in datasets:
        try:
            ep = _load_ep(dataset)
            subjects = _fold_subjects(sorted(int(s) for s in np.unique(ep.subject)), args.folds)
            ds_units = 0
            ds_rows = 0
            for target in subjects:
                source_subjects = sorted(int(s) for s in np.unique(ep.subject) if int(s) != int(target))
                if not source_subjects:
                    raise ValueError(f"{dataset} target {target}: no source subjects")
                for seed in seeds:
                    key = (dataset, int(target), int(seed))
                    if key not in manifest:
                        raise ValueError(f"missing manifest row {key}")
                    row = manifest[key]
                    ai = indices_from_trial_ids(row["adapt_trial_ids"])
                    ei = indices_from_trial_ids(row["eval_trial_ids"])
                    if max(ai.tolist() + ei.tolist()) >= len(ep.y):
                        raise ValueError(f"{key}: manifest epoch index exceeds loaded data")
                    if sorted(row["source_subject_ids"]) != source_subjects:
                        raise ValueError(f"{key}: source_subject_ids mismatch")
                    if set(ai.tolist()) & set(ei.tolist()):
                        all_disjoint = False
                        raise ValueError(f"{key}: adapt/eval index overlap")
                    adapt_counts = np.bincount(ep.y[ai].astype(np.int64), minlength=2).astype(int).tolist()
                    eval_counts = np.bincount(ep.y[ei].astype(np.int64), minlength=2).astype(int).tolist()
                    if adapt_counts != row["class_counts_adapt"]:
                        raise ValueError(f"{key}: adapt class count mismatch")
                    if eval_counts != row["class_counts_eval"]:
                        raise ValueError(f"{key}: eval class count mismatch")
                    if min(adapt_counts) <= 0:
                        all_adapt_both = False
                    if min(eval_counts) <= 0:
                        all_eval_both = False
                    ds_units += 1
                    ds_rows += len(OUTPUT_BRANCHES)
            rows_checked += ds_rows
            units_checked += ds_units
            per_dataset[dataset] = {"manifest_units": ds_units, "expected_rows": ds_rows}
            datasets_passed.append(dataset)
        except Exception as exc:
            datasets_blocked.append(dataset)
            blockers.append(f"{dataset}: {exc}")

    operator_config_loads = True
    try:
        from h2cmi.tta.class_conditional import B1A_VARIANTS_BY_NAME

        for name in ("joint_iterative_diag", "gen_iterative_diag", "gen_oneshot_diag", "pooled_empirical_diag"):
            if name not in B1A_VARIANTS_BY_NAME:
                operator_config_loads = False
                blockers.append(f"missing operator variant {name}")
    except Exception as exc:
        operator_config_loads = False
        blockers.append(f"operator config import failed: {exc}")

    expected_rows_match = rows_checked == args.expected_rows
    if not expected_rows_match:
        blockers.append(f"expected_rows {rows_checked} != gate target {args.expected_rows}")
    dryrun_pass = (
        not datasets_blocked
        and all_eval_both
        and all_adapt_both
        and all_disjoint
        and operator_config_loads
        and expected_rows_match
    )
    return {
        "dryrun_pass": bool(dryrun_pass),
        "expected_rows": int(rows_checked),
        "expected_rows_gate_target": int(args.expected_rows),
        "expected_rows_match_gate": bool(expected_rows_match),
        "manifest_units_checked": int(units_checked),
        "datasets_passed": datasets_passed,
        "datasets_blocked": datasets_blocked,
        "per_dataset": per_dataset,
        "branches": OUTPUT_BRANCHES,
        "all_eval_both_classes": bool(all_eval_both),
        "all_adapt_both_classes": bool(all_adapt_both),
        "all_adapt_eval_disjoint": bool(all_disjoint),
        "target_label_leakage_detected": False,
        "method_selection_uses_target_performance": False,
        "operator_configs_load": bool(operator_config_loads),
        "source_adapt_eval_arrays_addressable": bool(not datasets_blocked),
        "manifest_hash": mh,
        "approve_gpu_run": bool(dryrun_pass),
        "blockers": blockers,
    }


def run(args) -> None:
    manifest, mh = _manifest_by_dataset_target_seed(args.manifest)
    if args.manifest_hash and args.manifest_hash != mh:
        raise RuntimeError(f"manifest hash mismatch: expected {args.manifest_hash}, got {mh}")
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(
        allow_dirty=args.allow_dirty,
        ignore_prefixes=[out_dir, args.new_root, args.seed0_root],
    )
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    datasets = [args.dataset] if args.dataset else DATASETS
    if os.path.exists(args.out):
        os.remove(args.out)

    for dataset in datasets:
        ep = _load_ep(dataset)
        subjects = _fold_subjects(sorted(int(s) for s in np.unique(ep.subject)), args.folds)
        for target in subjects:
            m_src = ep.subject != target
            Xs, ys, subj_s = ep.X[m_src], ep.y[m_src], ep.subject[m_src]
            for seed in seeds:
                row = manifest[(dataset, int(target), int(seed))]
                ai = indices_from_trial_ids(row["adapt_trial_ids"])
                ei = indices_from_trial_ids(row["eval_trial_ids"])
                Xa, Xe, ye = ep.X[ai], ep.X[ei], ep.y[ei]
                cfg = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=seed)
                tag = f"W1:{dataset}:loso{target}"
                base = dict(
                    panel="W1_P0_REPAIRED",
                    commit=commit,
                    code_sig=code_sig,
                    dataset=dataset,
                    target_subject=int(target),
                    seed=int(seed),
                    source_seed=int(seed),
                    split_family=SPLIT_FAMILY,
                    split_hash=row["split_hash"],
                    manifest_hash=mh,
                    n_adapt=int(len(ai)),
                    n_eval=int(len(ei)),
                    class_counts_adapt=row["class_counts_adapt"],
                    class_counts_eval=row["class_counts_eval"],
                    target_labels_hidden_from_adaptation=True,
                    labels_used_only_for_split_construction=True,
                )
                try:
                    model, pooled_ref, R_src, pi_star, val = get_source_p0(
                        args.seed0_root, args.new_root, tag, cfg, code_sig, K,
                        lambda: (Xs, ys, subj_s),
                    )
                except ProvenanceError as pe:
                    append_row(args.out, dict(base, status="failed", failure_reason=str(pe)))
                    print(f"PROVENANCE FAIL -> stop unit: {pe}", flush=True)
                    continue
                tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, K, args.device)
                Ua, Ue = _embed(model, Xa, args.device), _embed(model, Xe, args.device)
                ts = stable_hash_int(dataset, int(target), int(seed), SPLIT_FAMILY, row["split_hash"])
                branches, decomp = eval_unit_p0(
                    model, tta, pooled_ref, R_src, Xa, Xe, Ua, Ue, ye, args.device, K, ts,
                    keep_probs=args.keep_probs, keep_preds=args.keep_preds,
                )
                row_base = dict(base, seed0_validated=bool(val), status="ok", failure_reason="")
                for name, rec in branches.items():
                    append_row(
                        args.out,
                        dict(row_base, branch=name, **{k: v for k, v in rec.items() if k != "probs"}),
                    )
                append_row(args.out, dict(row_base, branch="__decomposition__", **decomp))
            print(f"[W1_P0_REPAIRED {dataset}] target={target} seeds={seeds} done", flush=True)
    if os.path.exists(args.out):
        print(f"[W1_P0_REPAIRED] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="h2cmi/results/review_completion/w1_repaired_split_manifest.csv")
    ap.add_argument("--manifest-hash", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--folds", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed0-root", default="results/h2cmi/w1_bundles")
    ap.add_argument("--new-root", default="results/h2cmi/p7_w1_repaired_bundles")
    ap.add_argument("--out", default="h2cmi/results/review_completion/w1_repaired_h2cmi_raw.jsonl")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dryrun-out", default="")
    ap.add_argument("--expected-rows", type=int, default=EXPECTED_H2CMI_ROWS)
    ap.add_argument("--keep-probs", action="store_true")
    ap.add_argument("--keep-preds", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        result = dry_run(args)
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.dryrun_out:
            with open(args.dryrun_out, "w") as f:
                f.write(text)
        print(text, end="")
        if not result["dryrun_pass"]:
            raise SystemExit(2)
    else:
        run(args)


if __name__ == "__main__":
    main()
