#!/usr/bin/env python
"""CPU/metadata preflight for S2P-CodeBrain-Bounded.

This script never constructs a trainer, reads target test labels for selection,
or launches another process.  It records source, corpus, budget, and downstream
asset contracts before the GPU tokenizer-target gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import subprocess
from pathlib import Path

import lmdb
import numpy as np

import codebrain_bounded_data as CBD


CODEBRAIN = Path("/home/infres/yinwang/CodeBrain")
TOKENIZER = Path("/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain_Tokenizer.pth")
RELEASED = Path("/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth")
FACED = Path("/projects/EEG-foundation-model/FACED_data/processed")
SEEDV = Path("/projects/EEG-foundation-model/SEED-V_processed_0219")
ISRUC_FLAT = Path("/projects/EEG-foundation-model/tdoan-24/isruc_200hz/isruc_complete.lmdb")
ISRUC_SEQUENCE_CANDIDATES = (
    Path("/projects/EEG-foundation-model/ISRUC_S3/precessed_filter_35"),
    Path("/projects/EEG-foundation-model/datalake/raw/ISRUC_S3/precessed_filter_35"),
    Path("/projects/EEG-foundation-model/datalake/raw/ISRUC/precessed_filter_35"),
    CODEBRAIN / "Preprocessing" / "ISRUC_S3" / "precessed_filter_35",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_provenance() -> dict:
    files = [
        CODEBRAIN / "Pretrain" / "pretrain_EEGSSM.py",
        CODEBRAIN / "Pretrain" / "Trainer.py",
        CODEBRAIN / "Models" / "SSSM.py",
        CODEBRAIN / "Models" / "modeling_tokenizer.py",
        CODEBRAIN / "Models" / "model_for_faced.py",
        CODEBRAIN / "Models" / "model_for_seedv.py",
        CODEBRAIN / "Models" / "model_for_isruc.py",
    ]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODEBRAIN, text=True).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=CODEBRAIN, text=True
    ).strip()
    return {
        "codebrain_repo": str(CODEBRAIN),
        "codebrain_commit": head,
        "tracked_worktree_clean": not bool(tracked),
        "tracked_status": tracked,
        "source_file_sha256": {str(p.relative_to(CODEBRAIN)): sha256(p) for p in files},
        "tokenizer_path": str(TOKENIZER),
        "tokenizer_size_bytes": TOKENIZER.stat().st_size,
        "tokenizer_sha256": sha256(TOKENIZER),
        "released_encoder_path": str(RELEASED),
        "released_encoder_size_bytes": RELEASED.stat().st_size,
        "released_encoder_sha256": sha256(RELEASED),
        "tokenizer_role": "released_frozen_target_generator",
        "released_encoder_role": "path_validity_external_reference",
        "stage2_scope": "fixed_released_tokenizer_native_EEGSSM_only",
    }


def lmdb_keys(path: Path):
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False, max_readers=32)
    with env.begin(write=False) as txn:
        raw = txn.get(b"__keys__")
        keys = pickle.loads(raw) if raw is not None else None
    return env, keys


def inspect_faced() -> dict:
    env, keys = lmdb_keys(FACED)
    pattern = re.compile(r"^sub(\d+)\.pkl-(\d+)-(\d+)$")
    expected = {"train": set(range(0, 80)), "val": set(range(80, 100)), "test": set(range(100, 123))}
    split_details = {}
    valid = isinstance(keys, dict) and set(keys) == set(expected)
    for split in ("train", "val", "test"):
        parsed = [pattern.match(k) for k in keys.get(split, [])] if isinstance(keys, dict) else []
        valid = valid and all(parsed)
        subjects = {int(m.group(1)) for m in parsed if m}
        clips = {int(m.group(2)) for m in parsed if m}
        segments = {int(m.group(3)) for m in parsed if m}
        split_details[split] = {
            "keys": len(parsed), "subjects": len(subjects), "subject_ids": sorted(subjects),
            "clips": sorted(clips), "segments": sorted(segments),
        }
        valid = valid and subjects == expected[split] and clips == set(range(28)) and segments == {0, 1, 2}
    first = keys["train"][0]
    with env.begin(write=False) as txn:
        item = pickle.loads(txn.get(first.encode()))
    shape = list(np.asarray(item["sample"]).shape)
    env.close()
    valid = valid and shape == [32, 10, 200]
    return {
        "dataset": "FACED", "path": str(FACED), "asset_exists": FACED.exists(),
        "native_shape": shape, "split_contract": "subjects_1-80_81-100_101-123 (zero-based 0-79/80-99/100-122)",
        "split_details": split_details, "asset_contract_pass": bool(valid),
        "frozen_role": "primary_cross_subject_mechanism", "fine_tuning_role": "primary_adaptability_control",
    }


def inspect_seedv() -> dict:
    env, keys = lmdb_keys(SEEDV)
    pattern = re.compile(r"^(.*)-(\d+)-(\d+)$")
    expected_trials = {"train": set(range(0, 5)), "val": set(range(5, 10)), "test": set(range(10, 15))}
    split_details = {}
    valid = isinstance(keys, dict) and set(keys) == set(expected_trials)
    for split in ("train", "val", "test"):
        parsed = [pattern.match(k) for k in keys.get(split, [])] if isinstance(keys, dict) else []
        valid = valid and all(parsed)
        records = {m.group(1) for m in parsed if m}
        subjects = {r.split("_")[0] for r in records}
        sessions = {tuple(r.split("_")[:2]) for r in records}
        trials = {int(m.group(2)) for m in parsed if m}
        split_details[split] = {
            "keys": len(parsed), "recordings": len(records), "subjects": len(subjects),
            "subject_sessions": len(sessions), "trials": sorted(trials),
        }
        valid = valid and trials == expected_trials[split] and len(subjects) == 16 and len(sessions) == 47
    first = keys["train"][0]
    with env.begin(write=False) as txn:
        item = pickle.loads(txn.get(first.encode()))
    shape = list(np.asarray(item["sample"]).shape)
    env.close()
    valid = valid and shape == [62, 1, 200]
    return {
        "dataset": "SEED-V", "path": str(SEEDV), "asset_exists": SEEDV.exists(),
        "native_shape": shape, "split_contract": "same 16 subjects/session; trials 0-4/5-9/10-14",
        "split_details": split_details, "asset_contract_pass": bool(valid),
        "frozen_role": "same_task_domain_external_validation",
        "fine_tuning_role": "same_task_domain_adaptability_control",
        "cross_subject_reliance_confirmation_allowed": False,
    }


def inspect_isruc() -> dict:
    sequence_root = None
    for candidate in ISRUC_SEQUENCE_CANDIDATES:
        seq = candidate / "seq"
        lab = candidate / "labels"
        if all((seq / f"ISRUC-group3-{i}").is_dir() for i in range(1, 11)) and all(
            (lab / f"ISRUC-group3-{i}").is_dir() for i in range(1, 11)
        ):
            sequence_root = candidate
            break
    flat = {"path": str(ISRUC_FLAT), "exists": ISRUC_FLAT.exists()}
    if ISRUC_FLAT.exists():
        env, keys = lmdb_keys(ISRUC_FLAT)
        flat["has_keys_manifest"] = keys is not None
        flat["entries"] = int(env.stat()["entries"])
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            _, raw = next(iter(cursor))
            item = pickle.loads(raw)
        flat["sample_shape"] = list(np.asarray(item["x"]).shape) if isinstance(item, dict) and "x" in item else None
        flat["fields"] = sorted(item) if isinstance(item, dict) else None
        env.close()
    valid = sequence_root is not None
    return {
        "dataset": "ISRUC_S3", "path": str(sequence_root) if sequence_root else None,
        "asset_exists": bool(sequence_root), "native_shape": [20, 6, 30, 200] if valid else None,
        "split_contract": "10-subject leave-one-subject-out with adjacent validation subject",
        "sequence_aware_head": "frozen 20-epoch context; 512-d head + one-layer Transformer",
        "asset_contract_pass": bool(valid), "frozen_role": "cross_task_sleep_validation",
        "fine_tuning_role": "sequence_aware_adaptability_control",
        "flat_epoch_asset_diagnostic": flat,
        "blocker": None if valid else (
            "No CodeBrain-compatible ISRUC_S3 10-subject sequence tree was found. The available flat LMDB has "
            "global epoch keys and no auditable subject/20-epoch sequence manifest; it cannot be silently substituted."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    budgets, corpus = CBD.budget_summaries()
    source = source_provenance()
    downstream = [inspect_faced(), inspect_seedv(), inspect_isruc()]
    downstream_rows = [{
        "dataset": d["dataset"], "path": d["path"], "asset_exists": d["asset_exists"],
        "native_shape": json.dumps(d["native_shape"]), "split_contract": d["split_contract"],
        "asset_contract_pass": d["asset_contract_pass"], "frozen_role": d["frozen_role"],
        "fine_tuning_role": d["fine_tuning_role"], "blocker": d.get("blocker"),
    } for d in downstream]

    protocol = {
        "phase": "S2P_CodeBrain_Bounded_cross_architecture_representation_emergence",
        "not_full_codebrain_replication": True,
        "tokenizer": "released_TFDual_frozen",
        "encoder": "native_EEGSSM_Stage2_8layer_hidden200",
        "objective": "native_masked_temporal_CE_plus_frequency_CE",
        "mask_shape": ["B", 19, 30], "mask_ratio": 0.5,
        "budgets_h": list(CBD.BUDGETS_H), "pretraining_init_seeds": [0, 1],
        "subset_seed": CBD.SUBSET_SEED, "same_subset_across_init_seeds": True,
        "budget_subsets_nested": True, "epochs": 10,
        "optimizer_updates_matched_across_budgets": False,
        "downstream_datasets": ["FACED", "SEED-V", "ISRUC_S3"],
        "evaluation_tracks": ["frozen_representation_audit", "unified_full_finetuning_control"],
        "fine_tuning_seeds": [0, 1, 2, 3, 4],
        "frozen_primary_task_metric": "target_negative_log_likelihood",
        "fine_tuning_primary_metric": "cohen_kappa",
        "target_labels_used_for_selection": False,
        "training_authorized_by_this_preflight": False,
    }
    metadata_pass = bool(
        source["tracked_worktree_clean"]
        and all(r["exact_window_budget_feasible"] and r["no_window_reuse"] for r in budgets)
        and all(r["nested_with_previous_budget"] for r in budgets)
        and corpus["train_val_subject_disjoint"]
    )
    downstream_pass = all(d["asset_contract_pass"] for d in downstream)
    status = {
        "phase": protocol["phase"], "source_provenance_pass": bool(source["tracked_worktree_clean"]),
        "budget_metadata_pass": metadata_pass, "downstream_asset_contract_pass": downstream_pass,
        "tokenizer_target_gate_pass": None, "native_shape_canary_pass": None,
        "launch_bounded_stage2": False,
        "launch_blockers": [d["blocker"] for d in downstream if d.get("blocker")],
        "training_launched": False, "fine_tuning_launched": False,
        "target_labels_used_for_selection": False,
    }
    firewall = {
        "downstream_test_labels_read_for_selection": False,
        "test_labels_used_for_budget_or_checkpoint_selection": False,
        "test_labels_used_for_probe_or_hyperparameter_selection": False,
        "preflight_reads_dataset_keys_and_tensor_shapes_only": True,
        "target_labels_final_scoring_only": True,
    }

    write_csv(out / "codebrain_19common_budget_feasibility.csv", budgets)
    dump_json(out / "codebrain_19common_corpus_contract.json", corpus)
    dump_json(out / "codebrain_source_provenance.json", source)
    dump_json(out / "codebrain_downstream_asset_details.json", downstream)
    write_csv(out / "codebrain_downstream_contract_check.csv", downstream_rows)
    dump_json(out / "codebrain_bounded_protocol.json", protocol)
    dump_json(out / "codebrain_target_label_firewall.json", firewall)
    dump_json(out / "codebrain_metadata_preflight.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
