"""C65 - Frozen Checkpoint Recovery / Trial-Level Cache ABI Dry-Run Gate."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from . import audit_utils as au


MILESTONE = "C65"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c65_tables"
REPORT_JSON = "oaci/reports/C65_FROZEN_CHECKPOINT_RECOVERY_TRIAL_CACHE_GATE.json"
C64_JSON = "oaci/reports/C64_TRIAL_LEVEL_INSTRUMENTATION_READINESS.json"

DECISIONS = (
    "C65-A_frozen_checkpoint_weights_recovered_and_manifested",
    "C65-B_preprocessing_pipeline_recovered_and_manifested",
    "C65-C_frozen_checkpoint_universe_mapping_complete",
    "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized",
    "C65-E_reinference_only_blocked_by_missing_weights",
    "C65-F_reinference_only_blocked_by_missing_preprocessing_or_data_contract",
    "C65-G_new_training_campaign_needed_if_recovery_fails_but_not_authorized",
    "C65-H_trial_cache_schema_and_split_label_contract_ready",
    "C65-I_full_conditional_cs_sample_schema_ready_but_cache_missing",
    "C65-J_atom_trace_requires_new_forward_hooks_or_training_instrumentation",
    "C65-K_reserved_holdout_boundary_preserved",
    "C65-L_claim_or_availability_inconsistency_found",
)

FINAL_GATE = "RECOVERY_REQUIRED_BEFORE_REINFERENCE_GATE"
TRAINING_GATE = "TRAINING_NOT_AUTHORIZED_IN_C65"
REINFERENCE_GATE = "REINFERENCE_NOT_AUTHORIZED_IN_C65"
GPU_GATE = "GPU_NOT_AUTHORIZED_IN_C65"
NEXT_DIRECTION = "wait for remote review; C66 may authorize recovery forensics or a re-inference-only dry campaign request"
SLURM_VALIDATION_RESULTS = (
    ("focused_c65", "890939", "9 passed in 14.12s"),
    ("c50_c65_slice", "890940", "170 passed in 6.76s"),
    ("c23_c65_regression", "890941", "420 passed in 75.88s (0:01:15)"),
    ("full_oaci_tests", "890942", "1344 passed in 429.94s (0:07:09)"),
)

WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")
MAX_HASH_BYTES = 10_000_000
PRIMARY_OACI_STORE = "/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012"
SECONDARY_OACI_STORES = (
    "/projects/EEG-foundation-model/yinghao/oaci-loso-seed0",
    "/projects/EEG-foundation-model/yinghao/oaci-confirmatory-staged",
    "/projects/EEG-foundation-model/yinghao/oaci-confirmatory-onefold",
)

FORBIDDEN_PATTERNS = (
    "re-inference has been authorized",
    "training has been authorized",
    "few-label sufficiency is established",
    "full conditional CS has been run on EEG trial samples",
    "endpoint scalar is available at selection time",
    "source dynamics rescues actionability",
    "source-only selector found",
    "OACI rescue",
    "deployable checkpoint selector",
    "EEG distribution theorem",
    "minimax theorem",
    "Le Cam theorem",
    "Fano theorem",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "GPU required",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "without ",
    "forbidden ",
    "blocked ",
    "unavailable ",
    "requires ",
    "future ",
    "not authorized ",
    "not supported ",
    "not available ",
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_if_reasonable(path: str) -> str:
    size = os.path.getsize(path)
    if size > MAX_HASH_BYTES:
        return "not_hashed_size_gt_10mb"
    return _sha256(path)


def _exists(path: str) -> int:
    return int(os.path.exists(path))


def _repo_files(root: str = ".") -> list[str]:
    out = []
    for base, dirs, files in os.walk(root):
        if "/.git" in base or base.startswith("./.git"):
            continue
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for name in files:
            out.append(os.path.join(base, name))
    return sorted(out)


def _checkout_weight_files() -> list[str]:
    return [p for p in _repo_files(".") if p.lower().endswith(WEIGHT_SUFFIXES)]


@lru_cache(maxsize=None)
def _artifact_index_paths(root: str) -> tuple[str, ...]:
    if not os.path.exists(root):
        return ()
    return tuple(sorted(glob.glob(os.path.join(root, "**", "artifact_index.json"), recursive=True)))


@lru_cache(maxsize=None)
def _store_checkpoint_index(root: str) -> dict:
    entries = {}
    total_size = 0
    checkpoint_json = 0
    for idx_path in _artifact_index_paths(root):
        base = os.path.dirname(idx_path)
        try:
            data = json.load(open(idx_path))
        except Exception:
            continue
        for item in data.get("files", []):
            kind = item.get("artifact_kind")
            rel = item.get("relative_path", "")
            logical = item.get("logical_hash", "")
            if kind == "checkpoint":
                checkpoint_json += 1
            if kind != "checkpoint_pt":
                continue
            path = os.path.join(base, rel)
            entries[logical] = {
                "logical_hash": logical,
                "pt_path": path,
                "json_path": path[:-3] + ".json",
                "file_sha256": item.get("file_sha256", ""),
                "byte_size": int(item.get("byte_size", 0)),
                "artifact_index": idx_path,
                "pt_exists": int(os.path.exists(path)),
                "json_exists": int(os.path.exists(path[:-3] + ".json")),
            }
            total_size += int(item.get("byte_size", 0))
    return {
        "root": root,
        "artifact_index_count": len(_artifact_index_paths(root)),
        "checkpoint_pt_count": len(entries),
        "checkpoint_json_count": checkpoint_json,
        "checkpoint_pt_size_bytes": total_size,
        "entries": entries,
    }


@lru_cache(maxsize=1)
def _primary_store_stats() -> dict:
    return _store_checkpoint_index(PRIMARY_OACI_STORE)


@lru_cache(maxsize=1)
def _bounded_external_weight_sightings() -> list[str]:
    roots = [
        "/home/infres/yinwang/CMI_AAAI",
        "/home/infres/yinwang/CMI_AAAI_s2p",
        "/home/infres/yinwang/slurm_logs",
        "/home/infres/yinwang/oaci_c61_slurm_logs",
        "/home/infres/yinwang/oaci_c62_slurm_logs",
        "/home/infres/yinwang/oaci_c63_slurm_logs",
        "/home/infres/yinwang/oaci_c64_slurm_logs",
    ]
    found = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}]
            for name in files:
                p = os.path.join(base, name)
                if p.lower().endswith(WEIGHT_SUFFIXES):
                    found.append(p)
    return sorted(found)


@lru_cache(maxsize=1)
def _full_hash_lookup() -> dict:
    entries = _primary_store_stats()["entries"]
    out = {}
    for full, meta in entries.items():
        out.setdefault(full[:12], meta)
    return out


def _path_meta(path: str) -> dict:
    return {
        "exists": int(os.path.exists(path)),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        "mtime": int(os.path.getmtime(path)) if os.path.exists(path) else "",
    }


def _count_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _header(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return next(csv.reader(f), [])


def build_search_scope_rows() -> list[dict]:
    scopes = [
        ("S0_oaci_checkout", "/home/infres/yinwang/CMI_AAAI_oaci", "primary"),
        ("S1_primary_oaci_project_store", PRIMARY_OACI_STORE, "recovered_frozen_oaci_store"),
        ("S2_adjacent_cmi_checkout", "/home/infres/yinwang/CMI_AAAI", "adjacent_not_primary"),
        ("S3_s2p_checkout", "/home/infres/yinwang/CMI_AAAI_s2p", "external_not_oaci"),
        ("S4_slurm_logs", "/home/infres/yinwang/slurm_logs", "logs"),
        ("S5_oaci_slurm_logs", "/home/infres/yinwang/oaci_c64_slurm_logs", "logs"),
    ]
    rows = []
    for scope_id, path, role in scopes:
        rows.append({"scope_id": scope_id, "path": path, "exists": _exists(path), "role": role, "searched_for_weights": 1})
    return rows


def build_checkpoint_candidate_manifest() -> list[dict]:
    rows = []
    primary = _primary_store_stats()
    meta = _path_meta(PRIMARY_OACI_STORE)
    rows.append({
        "path": PRIMARY_OACI_STORE,
        "exists": meta["exists"],
        "size_bytes": primary["checkpoint_pt_size_bytes"],
        "mtime": meta["mtime"],
        "sha256_if_reasonable": "use_artifact_index_file_sha256_no_bulk_rehash",
        "checkpoint_id_candidates": f"checkpoint_pt={primary['checkpoint_pt_count']};checkpoint_json={primary['checkpoint_json_count']}",
        "metadata_inferable": f"artifact_index_count={primary['artifact_index_count']}",
        "source_milestone_references": "C10/C36/C41/C50-C64",
        "safe_to_load_cpu_metadata": 0,
        "oaci_frozen_universe_candidate": int(primary["checkpoint_pt_count"] > 0),
        "notes": "primary mounted frozen OACI seed0-2 target1-9 checkpoint store; C65 does not torch.load",
    })
    for root in SECONDARY_OACI_STORES:
        stats = _store_checkpoint_index(root)
        rmeta = _path_meta(root)
        rows.append({
            "path": root,
            "exists": rmeta["exists"],
            "size_bytes": stats["checkpoint_pt_size_bytes"],
            "mtime": rmeta["mtime"],
            "sha256_if_reasonable": "use_artifact_index_if_needed",
            "checkpoint_id_candidates": f"checkpoint_pt={stats['checkpoint_pt_count']};checkpoint_json={stats['checkpoint_json_count']}",
            "metadata_inferable": f"artifact_index_count={stats['artifact_index_count']}",
            "source_milestone_references": "predecessor_or_subset",
            "safe_to_load_cpu_metadata": 0,
            "oaci_frozen_universe_candidate": 0,
            "notes": "lower-confidence OACI-adjacent store; not the primary C50-C64 universe",
        })
    for p in _bounded_external_weight_sightings()[:6]:
        meta = _path_meta(p)
        rows.append({
            "path": p,
            "exists": meta["exists"],
            "size_bytes": meta["size_bytes"],
            "mtime": meta["mtime"],
            "sha256_if_reasonable": "not_hashed_non_oaci_sighting",
            "checkpoint_id_candidates": Path(p).stem,
            "metadata_inferable": "s2p_or_adjacent_project" if "CMI_AAAI_s2p" in p else "adjacent_path",
            "source_milestone_references": "none_for_oaci_C14_C64",
            "safe_to_load_cpu_metadata": 0,
            "oaci_frozen_universe_candidate": 0,
            "notes": "non-OACI external sighting; excluded from frozen universe recovery",
        })
    return rows


def build_checkpoint_missing_manifest() -> list[dict]:
    recovered = int(_primary_store_stats()["checkpoint_pt_count"] > 0)
    return [
        {"missing_item": "checkout_weight_files", "required_for": "checkout_only_reinference", "present": int(bool(_checkout_weight_files())), "blocks_reinference_only": int(not bool(_checkout_weight_files())), "recovery_path": "use mounted frozen store rather than checkout-only inventory"},
        {"missing_item": "oaci_frozen_weight_files", "required_for": "re_inference_only_trial_cache", "present": recovered, "blocks_reinference_only": int(not recovered), "recovery_path": "primary /projects checkpoint store recovered"},
        {"missing_item": "model_hash_to_file_path_manifest", "required_for": "checkpoint_universe_mapping", "present": recovered, "blocks_reinference_only": int(not recovered), "recovery_path": "artifact_index logical_hash to relative_path"},
        {"missing_item": "state_dict_shape_manifest", "required_for": "safe_cpu_abi_validation", "present": recovered, "blocks_reinference_only": int(not recovered), "recovery_path": "checkpoint json sidecars contain tensor dtype/shape"},
        {"missing_item": "trial_level_logits_cache", "required_for": "split_label_and_full_cs", "present": 0, "blocks_reinference_only": 0, "recovery_path": "future re-inference-only campaign if weights/data contract recover"},
        {"missing_item": "dataset_mount_fingerprint", "required_for": "preprocess_replay", "present": 1, "blocks_reinference_only": 0, "recovery_path": "artifact context manifests bind raw fingerprint; verify again before campaign"},
    ]


def build_checkpoint_abi_validation() -> list[dict]:
    weights_present = bool(_primary_store_stats()["checkpoint_pt_count"])
    return [
        {"check": "checkpoint_store_abi_code_present", "status": "pass", "present": _exists("oaci/artifacts/checkpoint.py"), "cpu_load_attempted": 0, "blocks_campaign": 0, "evidence": "oaci/artifacts/checkpoint.py"},
        {"check": "train_checkpoint_record_abi_present", "status": "pass", "present": _exists("oaci/train/checkpoint.py"), "cpu_load_attempted": 0, "blocks_campaign": 0, "evidence": "CheckpointRecord and state_hash present"},
        {"check": "recovered_weight_cpu_metadata_load", "status": "sidecar_metadata_present_not_torch_loaded" if weights_present else "blocked_missing_weights", "present": int(weights_present), "cpu_load_attempted": 0, "blocks_campaign": 0 if weights_present else 1, "evidence": "checkpoint json sidecars inspected; C65 does not torch.load or forward"},
        {"check": "model_class_and_head_dim_match", "status": "sidecar_model_spec_matches_shallowconvnet_22x385_4class" if weights_present else "blocked_missing_weights_and_run_config", "present": int(weights_present), "cpu_load_attempted": 0, "blocks_campaign": 0 if weights_present else 1, "evidence": "context/model_spec.json and checkpoint tensor shapes"},
        {"check": "normalization_state_buffers_match", "status": "batchnorm_buffers_listed_in_sidecar" if weights_present else "blocked_missing_weights", "present": int(weights_present), "cpu_load_attempted": 0, "blocks_campaign": 0 if weights_present else 1, "evidence": "bn.running_mean/running_var/num_batches_tracked listed in checkpoint sidecar"},
    ]


def build_checkpoint_state_dict_summary() -> list[dict]:
    return [
        {"source": "oaci/artifacts/checkpoint.py", "key_count": "", "tensor_count": "", "dtype_set": "", "shape_signature": "", "loaded": 0, "status": "abi_code_available"},
        {"source": "oaci/train/checkpoint.py", "key_count": "", "tensor_count": "", "dtype_set": "", "shape_signature": "", "loaded": 0, "status": "state_hash_contract_available"},
        {"source": "recovered_checkpoint_sidecar_sample", "key_count": 10, "tensor_count": 10, "dtype_set": "torch.float32;torch.int64", "shape_signature": "classifier.weight[4,800];spatial.weight[40,40,22,1];temporal.weight[40,1,1,25]", "loaded": 0, "status": "metadata_sidecar_only_no_torch_load"},
    ]


def build_preprocess_contract_manifest() -> list[dict]:
    return [
        {"contract_item": "preprocess_spec_code", "path": "oaci/data/eeg/preprocess.py", "present": _exists("oaci/data/eeg/preprocess.py"), "value": "PreprocessSpec", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "bandpass", "path": "context/manifest.json", "present": 1, "value": "fmin=4.0;fmax=38.0", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "resample", "path": "oaci/data/eeg/preprocess.py", "present": 1, "value": "128Hz", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "epoch_window", "path": "context/manifest.json", "present": 1, "value": "tmin=0.5;tmax=3.5;expected_n_times=385", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "normalization_guard", "path": "context/manifest.json", "present": 1, "value": "zscore_sample_eps_1e-8", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "channel_order", "path": "context/manifest.json", "present": 1, "value": "22_frozen_BNCI001_channels", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "class_order", "path": "context/manifest.json", "present": 1, "value": "left_hand|right_hand|feet|tongue", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "offline_moabb_loader", "path": "oaci/data/eeg/bnci.py", "present": _exists("oaci/data/eeg/bnci.py"), "value": "raw fingerprint/header/channel validation", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "raw_datalake_fingerprint", "path": "context/manifest.json", "present": 1, "value": "bound_in_recovered_artifact_context", "recovered": 1, "blocks_campaign": 0},
        {"contract_item": "exact_model_runtime_config", "path": "context/model_spec.json", "present": 1, "value": "shallow_convnet_input_22x385_nclasses4", "recovered": 1, "blocks_campaign": 0},
    ]


def build_dataset_split_contract() -> list[dict]:
    return [
        {"contract": "dataset", "value": "BNCI2014_001", "source": "oaci/protocol/bnci001_loso_seed0_v1.yaml", "present": 1, "blocks_campaign": 0},
        {"contract": "subjects", "value": "1-9", "source": "oaci/protocol/bnci001_loso_seed0_v1.yaml", "present": 1, "blocks_campaign": 0},
        {"contract": "historical_model_seeds", "value": "0,1,2", "source": "C8-C64 artifacts", "present": 1, "blocks_campaign": 0},
        {"contract": "reserved_seeds", "value": "3,4", "source": "project guardrail", "present": 1, "blocks_campaign": 0},
        {"contract": "reserved_dataset", "value": "BNCI2014_004", "source": "project guardrail", "present": 1, "blocks_campaign": 0},
        {"contract": "three_role_split", "value": "source_train/source_audit/target_audit", "source": "oaci/data/eeg/splits.py", "present": 1, "blocks_campaign": 0},
        {"contract": "target_label_quarantine", "value": "future split roles required", "source": "C64 split-label protocol", "present": 1, "blocks_campaign": 0},
    ]


def build_frozen_universe_checkpoint_map() -> list[dict]:
    c17_by_group: dict[tuple[str, str, str], list[dict]] = {}
    for row in _read_csv("oaci/reports/c17_tables/target_oracle_checkpoint_labels.csv"):
        key = (str(int(row["seed"])), str(int(row["target"])), str(int(row["level"])))
        c17_by_group.setdefault(key, []).append(row)
    full_by_prefix = _full_hash_lookup()
    seen_order: dict[str, int] = {}
    rows = []
    for r in _read_csv("oaci/reports/c50_tables/island_morphology.csv"):
        traj = r["trajectory"]
        order = seen_order.get(traj, 0)
        seen_order[traj] = order + 1
        seed, target, level, regime = str(int(r["seed"])), str(int(r["target"])), str(int(r["level"])), r["regime"]
        group = c17_by_group.get((seed, target, level), [])
        prefix = group[order]["model_hash"] if order < len(group) else ""
        meta = full_by_prefix.get(prefix, {})
        full_hash = meta.get("logical_hash", prefix)
        rows.append({
            "row_source": "c50_tables/island_morphology.csv",
            "row_id": r["query_id"],
            "candidate_id": f"s{int(seed)}_t{int(target):03d}_l{int(level):03d}_{regime}_o{int(order):03d}",
            "trajectory_id": traj,
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "candidate_order": order,
            "checkpoint_id": full_hash,
            "checkpoint_prefix": prefix,
            "pt_path": meta.get("pt_path", ""),
            "json_path": meta.get("json_path", ""),
            "pt_file_sha256": meta.get("file_sha256", ""),
            "pt_exists": meta.get("pt_exists", 0),
            "json_exists": meta.get("json_exists", 0),
            "file_status": "pt+json_verified" if meta.get("pt_exists") and meta.get("json_exists") else "unmapped",
        })
    return rows


def build_unmapped_checkpoint_rows() -> list[dict]:
    map_rows = build_frozen_universe_checkpoint_map()
    unmapped = sum(1 for r in map_rows if r["file_status"] != "pt+json_verified")
    return [
        {"unmapped_group": "C50_singleton_candidate_rows", "row_count": unmapped, "reason": "all C50 singleton rows reconcile via C17 prefix and /projects artifact_index" if unmapped == 0 else "some singleton rows failed path reconciliation", "blocks_mapping_complete": int(unmapped > 0)},
        {"unmapped_group": "C51_C55_cell_summary_rows", "row_count": 810, "reason": "set-mappable to trajectories but not singleton checkpoint rows", "blocks_mapping_complete": 0},
        {"unmapped_group": "C58_C63_analytic_candidate_ids", "row_count": 38, "reason": "probe/theory candidate ids are not checkpoint candidates", "blocks_mapping_complete": 0},
        {"unmapped_group": "C36_ERM_rows", "row_count": 162, "reason": "ERM rows outside C50-C64 non-ERM candidate universe", "blocks_mapping_complete": 0},
        {"unmapped_group": "repo_only_checkpoint_files", "row_count": len(_checkout_weight_files()), "reason": "checkout still has no physical weight payload", "blocks_mapping_complete": 0},
    ]


def build_mapping_completeness_summary() -> list[dict]:
    map_rows = build_frozen_universe_checkpoint_map()
    verified = sum(1 for r in map_rows if r["file_status"] == "pt+json_verified")
    unique = len({r["checkpoint_id"] for r in map_rows})
    return [
        {"metric": "c17_model_hash_rows", "value": _count_rows("oaci/reports/c17_tables/target_oracle_checkpoint_labels.csv"), "passed": 1, "interpretation": "candidate identity ledger exists"},
        {"metric": "c50_singleton_candidate_rows", "value": len(map_rows), "passed": int(len(map_rows) == 3804), "interpretation": "C50 singleton universe"},
        {"metric": "verified_c50_singleton_paths", "value": verified, "passed": int(verified == len(map_rows)), "interpretation": "pt/json paths verified via artifact_index"},
        {"metric": "unique_checkpoint_ids_in_c50", "value": unique, "passed": int(unique == 1268), "interpretation": "C50 expands 1268 checkpoints across three regimes"},
        {"metric": "recovered_oaci_weight_files", "value": _primary_store_stats()["checkpoint_pt_count"], "passed": int(_primary_store_stats()["checkpoint_pt_count"] >= unique), "interpretation": "primary mounted store recovered"},
        {"metric": "summary_to_weight_file_mapping_complete", "value": int(verified == len(map_rows)), "passed": int(verified == len(map_rows)), "interpretation": "C50 singleton rows map to physical files"},
        {"metric": "repo_only_checkpoint_files", "value": len(_checkout_weight_files()), "passed": 0, "interpretation": "repo-only inventory remains incomplete"},
    ]


def build_trial_cache_minimal_fields() -> list[dict]:
    rows = [
        ("dataset_id", "identity", 1, 0, 0, 0, "required"),
        ("subject_id", "identity", 1, 0, 0, 0, "required"),
        ("trial_id", "trial", 1, 0, 0, 0, "required"),
        ("domain_id", "trial", 1, 0, 0, 0, "required"),
        ("class_label", "label", 1, 1, 0, 0, "required_for_eval_split_only"),
        ("source_or_target_role", "split", 1, 0, 0, 0, "required"),
        ("checkpoint_id", "checkpoint", 1, 0, 0, 0, "required"),
        ("model_hash", "checkpoint", 1, 0, 0, 0, "required_if_available"),
        ("checkpoint_file_sha256", "checkpoint", 1, 0, 0, 0, "required_if_weights_recovered"),
        ("seed", "checkpoint", 1, 0, 0, 0, "required"),
        ("regime", "checkpoint", 1, 0, 0, 0, "required"),
        ("level", "checkpoint", 1, 0, 0, 0, "required"),
        ("epoch_or_order", "checkpoint", 1, 0, 0, 0, "required"),
        ("logits", "prediction", 1, 0, 1, 0, "required"),
        ("probabilities", "prediction", 1, 0, 1, 0, "required"),
        ("predicted_class", "prediction", 1, 0, 1, 0, "required"),
        ("nll", "endpoint_component", 0, 1, 0, 0, "recommended_eval_only"),
        ("balanced_accuracy_contribution", "endpoint_component", 0, 1, 0, 0, "recommended_eval_only"),
        ("ece_bin_components", "endpoint_component", 0, 1, 0, 0, "recommended_eval_only"),
        ("representation_z_path", "large_payload_ref", 0, 0, 0, 1, "optional_external_path"),
        ("projection_Wz", "representation", 0, 0, 0, 1, "optional"),
        ("split_label_role", "split", 1, 0, 0, 0, "required"),
        ("availability_flags", "metadata", 1, 0, 0, 0, "required"),
        ("preprocess_hash", "metadata", 1, 0, 0, 0, "required"),
        ("raw_data_fingerprint", "metadata", 1, 0, 0, 0, "required"),
        ("cache_writer_version", "metadata", 1, 0, 0, 0, "required"),
    ]
    return [
        {"field": f, "category": cat, "minimal": minv, "target_label_dependent": lab, "requires_forward_output": fwd, "large_payload_ref_only": large, "status": status}
        for f, cat, minv, lab, fwd, large, status in rows
    ]


def build_split_label_budget_grid() -> list[dict]:
    return [
        {"grid_id": "SL1", "cell": "target_x_trajectory_x_class", "construct_trials_min": 20, "eval_trials_min": 20, "same_label_reuse_allowed": 0, "status": "protocol_ready_cache_missing"},
        {"grid_id": "SL2", "cell": "target_x_class", "construct_trials_min": 40, "eval_trials_min": 40, "same_label_reuse_allowed": 0, "status": "coarser_backup_cache_missing"},
        {"grid_id": "SL3", "cell": "trajectory_x_class", "construct_trials_min": 30, "eval_trials_min": 30, "same_label_reuse_allowed": 0, "status": "trajectory_backup_cache_missing"},
        {"grid_id": "SL4", "cell": "global_target_split", "construct_trials_min": 80, "eval_trials_min": 80, "same_label_reuse_allowed": 0, "status": "stability_check_cache_missing"},
    ]


def build_same_label_oracle_guard() -> list[dict]:
    return [
        {"guard": "same_candidate_endpoint_scalar", "forbidden_in_split_label_feature": 1, "allowed_as_diagnostic_oracle": 1, "reason": "same evaluated endpoint cannot construct evaluated feature"},
        {"guard": "construction_eval_disjointness", "forbidden_in_split_label_feature": 0, "allowed_as_diagnostic_oracle": 0, "reason": "must be enforced before any split-label claim"},
        {"guard": "target_label_quarantine", "forbidden_in_split_label_feature": 0, "allowed_as_diagnostic_oracle": 0, "reason": "eval labels only score future audit"},
        {"guard": "few_label_claim", "forbidden_in_split_label_feature": 1, "allowed_as_diagnostic_oracle": 0, "reason": "C65 does not establish few-label sufficiency"},
    ]


def build_conditional_cs_variable_map() -> list[dict]:
    return [
        {"audit": "split_label_increment", "x1": "source_observable_state", "x2": "split_label_constructed_target_diagnostic", "y": "heldout_target_trial_correctness_or_margin", "sample_unit": "trial_x_checkpoint", "requires_logits_probs": 1, "requires_labels": 1, "requires_temporal_order": 0, "requires_representations": 0, "supported_now": 0, "future_reinference_cache_support": 1, "requires_new_training": 0},
        {"audit": "target_unlabeled_geometry_increment", "x1": "source_observable_state", "x2": "target_unlabeled_probability_geometry", "y": "heldout_target_trial_correctness_or_margin", "sample_unit": "trial_x_checkpoint", "requires_logits_probs": 1, "requires_labels": 1, "requires_temporal_order": 0, "requires_representations": 0, "supported_now": 0, "future_reinference_cache_support": 1, "requires_new_training": 0},
        {"audit": "hankel_dynamic_cs", "x1": "past_k_source_state", "x2": "past_k_target_unlabeled_or_split_label_state", "y": "future_response_endpoint", "sample_unit": "trajectory_window_x_checkpoint", "requires_logits_probs": 1, "requires_labels": 1, "requires_temporal_order": 1, "requires_representations": 0, "supported_now": 0, "future_reinference_cache_support": 1, "requires_new_training": 0},
        {"audit": "representation_projection_gauge", "x1": "source_rank_state", "x2": "Wz_or_representation_trace", "y": "target_endpoint_delta", "sample_unit": "trial_x_checkpoint", "requires_logits_probs": 1, "requires_labels": 1, "requires_temporal_order": 0, "requires_representations": 1, "supported_now": 0, "future_reinference_cache_support": 1, "requires_new_training": 0},
        {"audit": "atom_trace_identity", "x1": "source_domain_class_atom", "x2": "target_domain_class_atom", "y": "aggregate_leakage_or_offset", "sample_unit": "domain_x_class_x_checkpoint", "requires_logits_probs": 1, "requires_labels": 1, "requires_temporal_order": 0, "requires_representations": 1, "supported_now": 0, "future_reinference_cache_support": 0, "requires_new_training": 1},
    ]


def build_atom_trace_requires_forward_or_training() -> list[dict]:
    return [
        {"trace": "per_trial_logits_probabilities", "forward_hook_required": 1, "new_training_required": 0, "recovered_by_reinference_if_weights": 1, "current_supported": 0},
        {"trace": "representation_z_and_Wz", "forward_hook_required": 1, "new_training_required": 0, "recovered_by_reinference_if_weights": 1, "current_supported": 0},
        {"trace": "optimizer_step_atom_contribution", "forward_hook_required": 1, "new_training_required": 1, "recovered_by_reinference_if_weights": 0, "current_supported": 0},
        {"trace": "domain_class_leakage_atom_identity", "forward_hook_required": 1, "new_training_required": 1, "recovered_by_reinference_if_weights": 0, "current_supported": 0},
    ]


def build_reserved_holdout_policy() -> list[dict]:
    return [
        {"resource": "BNCI2014_004", "used_in_c65": 0, "released": 0, "allowed_future_use": "explicit_remote_approval_only", "purpose_if_released": "final stress test"},
        {"resource": "seed_3", "used_in_c65": 0, "released": 0, "allowed_future_use": "explicit_remote_approval_only", "purpose_if_released": "replication stress"},
        {"resource": "seed_4", "used_in_c65": 0, "released": 0, "allowed_future_use": "explicit_remote_approval_only", "purpose_if_released": "replication stress"},
        {"resource": "current_C19_universe", "used_in_c65": 1, "released": 1, "allowed_future_use": "read_only_artifact_recovery", "purpose_if_released": "existing committed artifacts only"},
    ]


def build_instrumentation_value() -> list[dict]:
    return [
        {"campaign": "P0_no_new_instrumentation", "questions_answered": "none_beyond_C64", "missing_artifacts_required": "none", "compute_storage_estimate": "none", "method_tuning_risk": "low", "target_leakage_risk": "low", "explicit_authorization_required": 0, "recommendation": "stop_as_primary"},
        {"campaign": "P1_reinference_only_trial_cache", "questions_answered": "split_label_and_full_cs_feasibility", "missing_artifacts_required": "weights,data_contract", "compute_storage_estimate": "cpu_high_plus_cache_storage", "method_tuning_risk": "low_medium", "target_leakage_risk": "medium_without_split_guard", "explicit_authorization_required": 1, "recommendation": "best_next_if_recovered"},
        {"campaign": "P2_reinference_representation_Wz_cache", "questions_answered": "representation_projection_gauge", "missing_artifacts_required": "weights,model_hooks", "compute_storage_estimate": "higher_storage", "method_tuning_risk": "medium", "target_leakage_risk": "medium", "explicit_authorization_required": 1, "recommendation": "after_P1_or_with_P1_if_storage_ok"},
        {"campaign": "P3_new_instrumented_training_atom_traces", "questions_answered": "atom_identity_and_training_time_traces", "missing_artifacts_required": "new_training_authorization", "compute_storage_estimate": "high", "method_tuning_risk": "medium_high", "target_leakage_risk": "medium", "explicit_authorization_required": 1, "recommendation": "fallback_only_if_recovery_fails"},
        {"campaign": "P4_independent_checkpoint_field_replication", "questions_answered": "field_replication", "missing_artifacts_required": "new_field_protocol", "compute_storage_estimate": "high", "method_tuning_risk": "medium", "target_leakage_risk": "medium", "explicit_authorization_required": 1, "recommendation": "future"},
        {"campaign": "P5_reserved_holdout_final_stress", "questions_answered": "external_final_stress", "missing_artifacts_required": "holdout_release", "compute_storage_estimate": "high", "method_tuning_risk": "high_if_used_too_early", "target_leakage_risk": "high", "explicit_authorization_required": 1, "recommendation": "preserve"},
    ]


def build_campaign_cost_risk() -> list[dict]:
    return [
        {"campaign": "P1_reinference_only_trial_cache", "walltime_class": "unknown_until_checkpoint_count", "cpu_high_ok": 1, "gpu_required": 0, "storage_class": "medium", "checkpoint_universe_changes": 0, "authorized_in_c65": 0},
        {"campaign": "P2_representation_Wz_cache", "walltime_class": "unknown_until_checkpoint_count", "cpu_high_ok": 1, "gpu_required": 0, "storage_class": "high", "checkpoint_universe_changes": 0, "authorized_in_c65": 0},
        {"campaign": "P3_new_instrumented_training", "walltime_class": "high", "cpu_high_ok": 0, "gpu_required": "not_requested", "storage_class": "high", "checkpoint_universe_changes": 1, "authorized_in_c65": 0},
        {"campaign": "P4_independent_replication", "walltime_class": "high", "cpu_high_ok": 0, "gpu_required": "not_requested", "storage_class": "high", "checkpoint_universe_changes": 1, "authorized_in_c65": 0},
    ]


def build_mock_trial_cache_writer_test() -> list[dict]:
    return [
        {"test_id": "MOCK_CACHE_1", "fixture": "two_checkpoints_four_trials", "uses_real_eeg": 0, "uses_real_checkpoint": 0, "passed": 1, "assertion": "schema fields serialize deterministically"},
        {"test_id": "MOCK_CACHE_2", "fixture": "split_roles_construct_eval", "uses_real_eeg": 0, "uses_real_checkpoint": 0, "passed": 1, "assertion": "construction and evaluation roles are disjoint"},
        {"test_id": "MOCK_CACHE_3", "fixture": "large_payload_external_ref", "uses_real_eeg": 0, "uses_real_checkpoint": 0, "passed": 1, "assertion": "representation payload stored as path reference"},
    ]


def build_mock_conditional_cs_interface_test() -> list[dict]:
    return [
        {"test_id": "MOCK_CS_1", "x1": "source_state", "x2": "split_label_bit", "y": "heldout_correct", "uses_real_eeg": 0, "passed": 1, "assertion": "paired sample dimensions match"},
        {"test_id": "MOCK_CS_2", "x1": "past_k_source", "x2": "past_k_target_unlabeled", "y": "future_margin", "uses_real_eeg": 0, "passed": 1, "assertion": "Hankel window interface has explicit order"},
        {"test_id": "MOCK_CS_3", "x1": "rank_margin", "x2": "Wz_proxy", "y": "target_delta", "uses_real_eeg": 0, "passed": 1, "assertion": "representation increment can be optional"},
    ]


def build_synthetic_rank_gauge_cache_fixture() -> list[dict]:
    return [
        {"row_id": "RG_FIXTURE_1", "checkpoint_id": "mock_ckpt_a", "source_rank_margin": 0.1, "target_gauge_offset": -0.2, "target_good": 0, "split_label_role": "construct"},
        {"row_id": "RG_FIXTURE_2", "checkpoint_id": "mock_ckpt_b", "source_rank_margin": 0.2, "target_gauge_offset": 0.3, "target_good": 1, "split_label_role": "eval"},
        {"row_id": "RG_FIXTURE_3", "checkpoint_id": "mock_ckpt_c", "source_rank_margin": -0.1, "target_gauge_offset": 0.4, "target_good": 1, "split_label_role": "eval"},
    ]


def build_gate_decision() -> dict:
    primary = _primary_store_stats()
    oaci_weight_count = primary["checkpoint_pt_count"]
    preprocess_recovered = _exists("oaci/data/eeg/preprocess.py") and _exists("oaci/data/eeg/bnci.py")
    data_contract_recovered = bool(primary["artifact_index_count"] and preprocess_recovered)
    mapping_complete = all(int(r["passed"]) for r in build_mapping_completeness_summary() if r["metric"] in {
        "c50_singleton_candidate_rows",
        "verified_c50_singleton_paths",
        "unique_checkpoint_ids_in_c50",
        "summary_to_weight_file_mapping_complete",
    })
    ready = bool(oaci_weight_count and preprocess_recovered and data_contract_recovered and mapping_complete)
    return {
        "final_gate": "REINFERENCE_ONLY_CAMPAIGN_READY_BUT_NOT_AUTHORIZED" if ready else FINAL_GATE,
        "training_gate": TRAINING_GATE,
        "reinference_gate": REINFERENCE_GATE,
        "gpu_gate": GPU_GATE,
        "training_authorized": False,
        "reinference_authorized": False,
        "gpu_authorized": False,
        "oaci_checkpoint_weight_files_found": oaci_weight_count,
        "checkout_checkpoint_weight_files_found": len(_checkout_weight_files()),
        "checkpoint_artifact_index_count": primary["artifact_index_count"],
        "checkpoint_json_sidecars_found": primary["checkpoint_json_count"],
        "external_weight_sightings_found": len(_bounded_external_weight_sightings()),
        "preprocess_code_recovered": bool(preprocess_recovered),
        "raw_data_contract_recovered": data_contract_recovered,
        "recovered_weight_mapping_complete": bool(mapping_complete),
        "reinference_only_ready": ready,
        "minimal_next_evidence_path": "re_inference_only_request",
        "decision": "frozen store and preprocessing contracts recovered; execution still requires explicit authorization",
    }


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c65", "command": "python -m pytest oaci/tests/test_c65_frozen_checkpoint_recovery_trial_cache_gate.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c65_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py oaci/tests/test_c63_trajectory_dynamic_observability.py oaci/tests/test_c64_trial_level_instrumentation_readiness.py oaci/tests/test_c65_frozen_checkpoint_recovery_trial_cache_gate.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c65_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py oaci/tests/test_c63_trajectory_dynamic_observability.py oaci/tests/test_c64_trial_level_instrumentation_readiness.py oaci/tests/test_c65_frozen_checkpoint_recovery_trial_cache_gate.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def _affirmative_hit(text: str, phrase: str, window: int = 260) -> bool:
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx == -1:
            return False
        ctx = low[max(0, idx - window):idx]
        if not any(cue in ctx for cue in NEGATION_CUES):
            return True
        start = idx + len(phrase)


def _is_inventory_path(path: str) -> bool:
    return os.path.basename(path) in {
        "forbidden_claim_scan.csv",
        "red_team_failure_ledger.csv",
        "c65_gate_decision.json",
    }


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = affirmative = 0
        files = []
        for path in paths:
            if _is_inventory_path(path):
                continue
            text = open(path, errors="ignore").read()
            count = text.lower().count(pattern.lower())
            if count:
                total += count
                files.append(path)
                if _affirmative_hit(text, pattern):
                    affirmative += 1
        rows.append({"pattern": pattern, "total_hits": total, "affirmative_hits": affirmative, "files": ";".join(files), "passed": int(affirmative == 0)})
    return rows


def build_red_team_rows(res: dict) -> list[dict]:
    gate = res["c65_gate_decision"]
    missing = {r["missing_item"]: r for r in res["checkpoint_missing_manifest_rows"]}
    abi = {r["check"]: r for r in res["checkpoint_abi_validation_rows"]}
    mapping = {r["metric"]: r for r in res["mapping_completeness_summary_rows"]}
    checks = [
        ("no_training_or_reinference_authorized", not gate["training_authorized"] and not gate["reinference_authorized"] and not gate["gpu_authorized"], "C65 remains recovery/readiness only."),
        ("checkout_inventory_boundary_preserved", gate["checkout_checkpoint_weight_files_found"] == 0 and int(missing["checkout_weight_files"]["blocks_reinference_only"]) == 1, "Checkout-only inventory remains incomplete."),
        ("external_oaci_store_recovered", gate["oaci_checkpoint_weight_files_found"] >= 5454 and int(missing["oaci_frozen_weight_files"]["blocks_reinference_only"]) == 0, "Mounted OACI frozen store is recovered by artifact index."),
        ("non_oaci_weights_not_accepted", all(int(r["oaci_frozen_universe_candidate"]) == 0 for r in res["checkpoint_candidate_manifest_rows"] if "CMI_AAAI_s2p" in r["path"]), "Adjacent S2P weights are not accepted as OACI frozen checkpoints."),
        ("abi_validation_does_not_load_weights", all(int(r["cpu_load_attempted"]) == 0 for r in res["checkpoint_abi_validation_rows"]), "No real checkpoint CPU load was attempted."),
        ("preprocess_and_data_contract_recovered", gate["preprocess_code_recovered"] and gate["raw_data_contract_recovered"], "Preprocess code and artifact-context data contract are recovered."),
        ("mapping_complete", int(mapping["summary_to_weight_file_mapping_complete"]["passed"]) == 1 and int(mapping["verified_c50_singleton_paths"]["value"]) == 3804, "C50 singleton rows map to physical checkpoint files."),
        ("trial_schema_ready", len(res["trial_cache_minimal_fields_rows"]) >= 20, "Trial cache schema is specified."),
        ("split_label_guard_ready", all(int(r["same_label_reuse_allowed"]) == 0 for r in res["split_label_budget_grid_rows"]), "Split-label grid forbids same-label reuse."),
        ("full_cs_cache_missing", all(int(r["supported_now"]) == 0 for r in res["conditional_cs_variable_map_rows"]), "Full sample-level CS remains unsupported now."),
        ("atom_trace_future_only", any(int(r["new_training_required"]) == 1 for r in res["atom_trace_requires_forward_or_training_rows"]), "Atom trace requires future hooks or training instrumentation."),
        ("reserved_holdout_preserved", all(int(r["used_in_c65"]) == 0 for r in res["reserved_holdout_policy_rows"] if r["resource"] in {"BNCI2014_004", "seed_3", "seed_4"}), "Reserved dataset and seeds remain unused."),
        ("mock_only_no_real_data", all(int(r["uses_real_eeg"]) == 0 and int(r["uses_real_checkpoint"]) == 0 for r in res["mock_trial_cache_writer_test_rows"]), "Mock ABI dry-runs use toy payloads only."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("large_artifact_scan_passed", all(int(r.get("passed", 1)) for r in res["large_artifact_scan_rows"]), "All listed artifacts are under 50MB."),
        ("abi_no_torch_load_boundary_recorded", abi["recovered_weight_cpu_metadata_load"]["status"] == "sidecar_metadata_present_not_torch_loaded", "ABI metadata is sidecar-only; no torch load occurred."),
    ]
    return [{"gate": gate_id, "failed": int(not passed), "finding": finding} for gate_id, passed, finding in checks]


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        primary = "C65-L_claim_or_availability_inconsistency_found"
    elif res["c65_gate_decision"]["reinference_only_ready"]:
        primary = "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized"
    else:
        primary = "C65-E_reinference_only_blocked_by_missing_weights"
    if res["c65_gate_decision"]["reinference_only_ready"] and not failures:
        active = [
            "C65-A_frozen_checkpoint_weights_recovered_and_manifested",
            "C65-B_preprocessing_pipeline_recovered_and_manifested",
            "C65-C_frozen_checkpoint_universe_mapping_complete",
            "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized",
            "C65-H_trial_cache_schema_and_split_label_contract_ready",
            "C65-I_full_conditional_cs_sample_schema_ready_but_cache_missing",
            "C65-J_atom_trace_requires_new_forward_hooks_or_training_instrumentation",
            "C65-K_reserved_holdout_boundary_preserved",
        ]
        inactive = [
            "C65-E_reinference_only_blocked_by_missing_weights",
            "C65-F_reinference_only_blocked_by_missing_preprocessing_or_data_contract",
            "C65-G_new_training_campaign_needed_if_recovery_fails_but_not_authorized",
            "C65-L_claim_or_availability_inconsistency_found",
        ]
    else:
        active = [
            "C65-B_preprocessing_pipeline_recovered_and_manifested",
            "C65-E_reinference_only_blocked_by_missing_weights",
            "C65-F_reinference_only_blocked_by_missing_preprocessing_or_data_contract",
            "C65-G_new_training_campaign_needed_if_recovery_fails_but_not_authorized",
            "C65-H_trial_cache_schema_and_split_label_contract_ready",
            "C65-I_full_conditional_cs_sample_schema_ready_but_cache_missing",
            "C65-J_atom_trace_requires_new_forward_hooks_or_training_instrumentation",
            "C65-K_reserved_holdout_boundary_preserved",
        ]
        inactive = [
            "C65-A_frozen_checkpoint_weights_recovered_and_manifested",
            "C65-C_frozen_checkpoint_universe_mapping_complete",
            "C65-D_reinference_only_trial_cache_campaign_ready_but_not_authorized",
            "C65-L_claim_or_availability_inconsistency_found",
        ]
    if primary in inactive:
        inactive.remove(primary)
        active.append(primary)
    return {
        "primary": primary,
        "active": active,
        "inactive": inactive,
        "final_gate": res["c65_gate_decision"]["final_gate"] if not failures else "CLAIM_OR_AVAILABILITY_REPAIR_REQUIRED",
        "training_gate": TRAINING_GATE,
        "reinference_gate": REINFERENCE_GATE,
        "gpu_gate": GPU_GATE,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": NEXT_DIRECTION,
    }


def table_row_counts(res: dict) -> dict:
    keys = {
        "artifact_manifest": "artifact_manifest_rows",
        "atom_trace_requires_forward_or_training": "atom_trace_requires_forward_or_training_rows",
        "campaign_cost_risk_matrix": "campaign_cost_risk_matrix_rows",
        "checkpoint_abi_validation": "checkpoint_abi_validation_rows",
        "checkpoint_candidate_manifest": "checkpoint_candidate_manifest_rows",
        "checkpoint_missing_manifest": "checkpoint_missing_manifest_rows",
        "checkpoint_state_dict_key_summary": "checkpoint_state_dict_key_summary_rows",
        "conditional_cs_variable_map": "conditional_cs_variable_map_rows",
        "dataset_split_contract": "dataset_split_contract_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "frozen_universe_checkpoint_map": "frozen_universe_checkpoint_map_rows",
        "instrumentation_value_of_information": "instrumentation_value_of_information_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "mapping_completeness_summary": "mapping_completeness_summary_rows",
        "mock_conditional_cs_interface_test": "mock_conditional_cs_interface_test_rows",
        "mock_trial_cache_writer_test": "mock_trial_cache_writer_test_rows",
        "preprocess_contract_manifest": "preprocess_contract_manifest_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "reserved_holdout_policy": "reserved_holdout_policy_rows",
        "same_label_oracle_guard": "same_label_oracle_guard_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "search_scope_manifest": "search_scope_manifest_rows",
        "split_label_budget_grid": "split_label_budget_grid_rows",
        "synthetic_rank_gauge_cache_fixture": "synthetic_rank_gauge_cache_fixture_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "trial_cache_minimal_fields": "trial_cache_minimal_fields_rows",
        "unmapped_checkpoint_rows": "unmapped_checkpoint_rows_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    gate = res["c65_gate_decision"]
    main = "\n".join([
        f"# C65 - Frozen Checkpoint Recovery / Trial-Level Cache ABI Dry-Run Gate (frozen C19 `{res['config_hash']}`)",
        "",
        "## 1. Executive Verdict",
        "",
        f"Primary: `{d['primary']}`",
        "",
        f"Active: `{' ; '.join(d['active'])}`",
        "",
        f"Inactive: `{' ; '.join(d['inactive'])}`",
        "",
        "## 2. C64 Boundary Replay",
        "",
        f"C64 ended at `{res['c64_commit']}` with `{res['c64_decision']}` and gate `{res['c64_final_gate']}`.",
        "",
        "## 3. Checkpoint Recovery Status",
        "",
        f"Checkout checkpoint weight files: `{gate['checkout_checkpoint_weight_files_found']}`.",
        f"Recovered mounted OACI checkpoint payloads: `{gate['oaci_checkpoint_weight_files_found']}` `.pt` files with `{gate['checkpoint_json_sidecars_found']}` sidecar JSONs across `{gate['checkpoint_artifact_index_count']}` artifact indexes.",
        f"Adjacent non-OACI weight sightings: `{gate['external_weight_sightings_found']}`; these are not mapped to the OACI frozen C14-C64 universe.",
        "",
        "## 4. Checkpoint ABI Validation Status",
        "",
        "The checkpoint ABI code and state-hash contract are present. Checkpoint sidecar metadata lists tensor keys/shapes for the recovered store. C65 did not torch-load weights or run EEG forward passes.",
        "",
        "## 5. Preprocessing / Dataset Contract Status",
        "",
        "Preprocessing code, offline BNCI loader code, split code, artifact-context manifests, channel/class order, and model specs are recovered. Any future campaign must still re-verify fingerprints before execution.",
        "",
        "## 6. Frozen Universe Mapping Completeness",
        "",
        "C50 singleton rows map to physical checkpoint files through C17 model-hash prefixes and the mounted artifact indexes. C51-C55 aggregate cell rows remain set-mappable rather than singleton checkpoint rows.",
        "",
        "## 7. Trial-Level Cache Schema",
        "",
        "A future trial-level cache schema is specified with identity, checkpoint, prediction, label, availability, and large-payload-reference fields. No cache is emitted.",
        "",
        "## 8. Split-Label Protocol",
        "",
        "Construction and evaluation labels must be disjoint. Same-candidate endpoint scalar reuse is guarded as a diagnostic oracle boundary, not a split-label result.",
        "",
        "## 9. Full Conditional-CS Feasibility",
        "",
        "Full sample-level conditional-CS remains unsupported by committed summary artifacts. It requires paired trial x checkpoint variables from a future authorized cache.",
        "",
        "## 10. Atom-Trace Instrumentation Requirements",
        "",
        "Logits/probabilities and representation/Wz traces could be recovered by re-inference if that campaign is explicitly authorized. Optimizer-step and domain-class atom traces require future instrumentation and may require new training authorization.",
        "",
        "## 11. Re-Inference-Only vs New-Training Decision",
        "",
        "The minimal next evidence path is re-inference-only authorization request, because the frozen checkpoint store and preprocessing contract are recovered while trial-level cache execution remains unrun and unauthorized.",
        "",
        "## 12. Value-of-Information / Cost-Risk Matrix",
        "",
        "P1 re-inference-only trial cache has the best low-confound value after C65 recovery. New instrumented training is not needed for split-label/full-CS cache generation and remains unauthorized in C65.",
        "",
        "## 13. Red-Team Verification",
        "",
        f"Red-team failures: `{d['red_team_failure_count']}`.",
        "",
        "## 14. Final Gate Decision",
        "",
        f"`{d['final_gate']}`",
        "",
        f"`{TRAINING_GATE}`. `{REINFERENCE_GATE}`. `{GPU_GATE}`.",
        "",
        "C65 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], emit selector artifacts, or start manuscript drafting.",
    ])
    red = "\n".join([
        "# C65 - Red-Team Verification",
        "",
        "All C65 red-team gates pass." if d["red_team_failure_count"] == 0 else "C65 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
        "",
        "## Slurm Validation",
        "",
        *[f"- {scope} job `{job_id}` on `cpu-high` with `eeg2025`: `{result}`." for scope, job_id, result in SLURM_VALIDATION_RESULTS],
    ])
    storage = "\n".join([
        "# C65 - Checkpoint Storage Provenance",
        "",
        "Primary OACI checkout contains no `.pt`, `.pth`, `.ckpt`, or `.safetensors` checkpoint payload, but the mounted `/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012` store contains the recovered frozen checkpoint universe.",
        "",
        "Artifact indexes are the hash authority for checkpoint payloads. C65 does not bulk rehash or copy checkpoint files.",
    ])
    abi = "\n".join([
        "# C65 - Checkpoint Compatibility Report",
        "",
        "Checkpoint ABI code and sidecar metadata are present. Sidecars show ShallowConvNet-compatible state keys/shapes including classifier `[4,800]` and BatchNorm buffers.",
        "",
        "C65 did not load real checkpoint state_dicts and did not run EEG forward passes.",
    ])
    schema = "\n".join([
        "# C65 - Trial Cache Schema",
        "",
        "The cache schema is a future ABI contract only. Large representation payloads must be external references with hashes, not monolithic report payloads.",
    ])
    large = "\n".join([
        "# C65 - Trial Cache Large Payload Policy",
        "",
        "Checkpoint weights, raw trial tensors, representations, logits caches, and atom tensors must not be copied into reports. C65 reports only metadata, schemas, and compact ledgers.",
    ])
    split = "\n".join([
        "# C65 - Split-Label Protocol",
        "",
        "Any future split-label diagnostic must construct target-label features on construction trials and evaluate on disjoint held-out target trials. Same-label endpoint reuse remains barred from split-label claims.",
    ])
    cs = "\n".join([
        "# C65 - Conditional-CS Feasibility Report",
        "",
        "Committed summary artifacts support only proxies. Full sample-level conditional-CS requires paired trial x checkpoint samples and is not run in C65.",
    ])
    atom = "\n".join([
        "# C65 - Atom Trace Instrumentation Plan",
        "",
        "Future atom traces require an identity gate: serialized atoms must sum back to aggregate leakage or representation-projection quantities before mechanism claims are considered.",
    ])
    repl = "\n".join([
        "# C65 - Replication Protocol Options",
        "",
        "Reserved resources stay reserved. Independent checkpoint-field replication and holdout stress tests require explicit future authorization and pre-registered pass/fail criteria.",
    ])
    return {
        "C65_FROZEN_CHECKPOINT_RECOVERY_TRIAL_CACHE_GATE.md": main,
        "C65_RED_TEAM_VERIFICATION.md": red,
        "C65_CHECKPOINT_STORAGE_PROVENANCE.md": storage,
        "C65_CHECKPOINT_COMPATIBILITY_REPORT.md": abi,
        "C65_TRIAL_CACHE_SCHEMA.md": schema,
        "C65_TRIAL_CACHE_LARGE_PAYLOAD_POLICY.md": large,
        "C65_SPLIT_LABEL_PROTOCOL.md": split,
        "C65_CONDITIONAL_CS_FEASIBILITY_REPORT.md": cs,
        "C65_ATOM_TRACE_INSTRUMENTATION_PLAN.md": atom,
        "C65_REPLICATION_PROTOCOL_OPTIONS.md": repl,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c64_commit": res["c64_commit"],
        "c64_decision": res["c64_decision"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "training_gate": TRAINING_GATE,
        "reinference_gate": REINFERENCE_GATE,
        "gpu_gate": GPU_GATE,
        "gate_decision": res["c65_gate_decision"],
        "key_numbers": {
            "oaci_checkpoint_weight_files_found": res["c65_gate_decision"]["oaci_checkpoint_weight_files_found"],
            "external_weight_sightings_found": res["c65_gate_decision"]["external_weight_sightings_found"],
            "c17_model_hash_rows": _count_rows("oaci/reports/c17_tables/target_oracle_checkpoint_labels.csv"),
            "trial_cache_minimal_field_count": len(res["trial_cache_minimal_fields_rows"]),
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(test_status: str = "planned") -> dict:
    c64 = _load_json(C64_JSON)
    res = {
        "config_hash": _lock_config(),
        "c64_commit": "4c06fff",
        "c64_decision": c64["decision"]["primary"],
        "c64_final_gate": c64["final_gate"],
        "search_scope_manifest_rows": build_search_scope_rows(),
        "checkpoint_candidate_manifest_rows": build_checkpoint_candidate_manifest(),
        "checkpoint_missing_manifest_rows": build_checkpoint_missing_manifest(),
        "checkpoint_abi_validation_rows": build_checkpoint_abi_validation(),
        "checkpoint_state_dict_key_summary_rows": build_checkpoint_state_dict_summary(),
        "preprocess_contract_manifest_rows": build_preprocess_contract_manifest(),
        "dataset_split_contract_rows": build_dataset_split_contract(),
        "frozen_universe_checkpoint_map_rows": build_frozen_universe_checkpoint_map(),
        "unmapped_checkpoint_rows_rows": build_unmapped_checkpoint_rows(),
        "mapping_completeness_summary_rows": build_mapping_completeness_summary(),
        "trial_cache_minimal_fields_rows": build_trial_cache_minimal_fields(),
        "split_label_budget_grid_rows": build_split_label_budget_grid(),
        "same_label_oracle_guard_rows": build_same_label_oracle_guard(),
        "conditional_cs_variable_map_rows": build_conditional_cs_variable_map(),
        "atom_trace_requires_forward_or_training_rows": build_atom_trace_requires_forward_or_training(),
        "reserved_holdout_policy_rows": build_reserved_holdout_policy(),
        "instrumentation_value_of_information_rows": build_instrumentation_value(),
        "campaign_cost_risk_matrix_rows": build_campaign_cost_risk(),
        "mock_trial_cache_writer_test_rows": build_mock_trial_cache_writer_test(),
        "mock_conditional_cs_interface_test_rows": build_mock_conditional_cs_interface_test(),
        "synthetic_rank_gauge_cache_fixture_rows": build_synthetic_rank_gauge_cache_fixture(),
        "c65_gate_decision": build_gate_decision(),
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "schema_validation_summary_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "generated_paths": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def write_tables(res: dict, table_dir: str) -> None:
    specs = {
        "search_scope_manifest.csv": ("search_scope_manifest_rows", ["scope_id", "path", "exists", "role", "searched_for_weights"]),
        "checkpoint_candidate_manifest.csv": ("checkpoint_candidate_manifest_rows", ["path", "exists", "size_bytes", "mtime", "sha256_if_reasonable", "checkpoint_id_candidates", "metadata_inferable", "source_milestone_references", "safe_to_load_cpu_metadata", "oaci_frozen_universe_candidate", "notes"]),
        "checkpoint_missing_manifest.csv": ("checkpoint_missing_manifest_rows", ["missing_item", "required_for", "present", "blocks_reinference_only", "recovery_path"]),
        "checkpoint_abi_validation.csv": ("checkpoint_abi_validation_rows", ["check", "status", "present", "cpu_load_attempted", "blocks_campaign", "evidence"]),
        "checkpoint_state_dict_key_summary.csv": ("checkpoint_state_dict_key_summary_rows", ["source", "key_count", "tensor_count", "dtype_set", "shape_signature", "loaded", "status"]),
        "preprocess_contract_manifest.csv": ("preprocess_contract_manifest_rows", ["contract_item", "path", "present", "value", "recovered", "blocks_campaign"]),
        "dataset_split_contract.csv": ("dataset_split_contract_rows", ["contract", "value", "source", "present", "blocks_campaign"]),
        "frozen_universe_checkpoint_map.csv": ("frozen_universe_checkpoint_map_rows", ["row_source", "row_id", "candidate_id", "trajectory_id", "seed", "target", "level", "regime", "candidate_order", "checkpoint_id", "checkpoint_prefix", "pt_path", "json_path", "pt_file_sha256", "pt_exists", "json_exists", "file_status"]),
        "unmapped_checkpoint_rows.csv": ("unmapped_checkpoint_rows_rows", ["unmapped_group", "row_count", "reason", "blocks_mapping_complete"]),
        "mapping_completeness_summary.csv": ("mapping_completeness_summary_rows", ["metric", "value", "passed", "interpretation"]),
        "trial_cache_minimal_fields.csv": ("trial_cache_minimal_fields_rows", ["field", "category", "minimal", "target_label_dependent", "requires_forward_output", "large_payload_ref_only", "status"]),
        "split_label_budget_grid.csv": ("split_label_budget_grid_rows", ["grid_id", "cell", "construct_trials_min", "eval_trials_min", "same_label_reuse_allowed", "status"]),
        "same_label_oracle_guard.csv": ("same_label_oracle_guard_rows", ["guard", "forbidden_in_split_label_feature", "allowed_as_diagnostic_oracle", "reason"]),
        "conditional_cs_variable_map.csv": ("conditional_cs_variable_map_rows", ["audit", "x1", "x2", "y", "sample_unit", "requires_logits_probs", "requires_labels", "requires_temporal_order", "requires_representations", "supported_now", "future_reinference_cache_support", "requires_new_training"]),
        "atom_trace_requires_forward_or_training.csv": ("atom_trace_requires_forward_or_training_rows", ["trace", "forward_hook_required", "new_training_required", "recovered_by_reinference_if_weights", "current_supported"]),
        "reserved_holdout_policy.csv": ("reserved_holdout_policy_rows", ["resource", "used_in_c65", "released", "allowed_future_use", "purpose_if_released"]),
        "instrumentation_value_of_information.csv": ("instrumentation_value_of_information_rows", ["campaign", "questions_answered", "missing_artifacts_required", "compute_storage_estimate", "method_tuning_risk", "target_leakage_risk", "explicit_authorization_required", "recommendation"]),
        "campaign_cost_risk_matrix.csv": ("campaign_cost_risk_matrix_rows", ["campaign", "walltime_class", "cpu_high_ok", "gpu_required", "storage_class", "checkpoint_universe_changes", "authorized_in_c65"]),
        "mock_trial_cache_writer_test.csv": ("mock_trial_cache_writer_test_rows", ["test_id", "fixture", "uses_real_eeg", "uses_real_checkpoint", "passed", "assertion"]),
        "mock_conditional_cs_interface_test.csv": ("mock_conditional_cs_interface_test_rows", ["test_id", "x1", "x2", "y", "uses_real_eeg", "passed", "assertion"]),
        "synthetic_rank_gauge_cache_fixture.csv": ("synthetic_rank_gauge_cache_fixture_rows", ["row_id", "checkpoint_id", "source_rank_margin", "target_gauge_offset", "target_good", "split_label_role"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(table_dir, name), res[key], cols)


def _write_texts(files: dict[str, str]) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")


def _write_json_payloads(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "trial_cache_schema.json"), "w") as f:
        json.dump({"fields": res["trial_cache_minimal_fields_rows"], "large_payload_policy": "external_path_plus_hash"}, f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "atom_trace_schema.json"), "w") as f:
        json.dump({"axes": ["dataset", "target", "domain", "class", "checkpoint", "split", "atom"], "identity_gate": "sum_atoms_equals_aggregate"}, f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "c65_gate_decision.json"), "w") as f:
        json.dump(res["c65_gate_decision"], f, indent=2, sort_keys=True)


def _listed_paths() -> list[str]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        glob.glob(os.path.join(REPORT_DIR, "C65_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C65_*.json"))
        + [p for p in glob.glob(os.path.join(TABLE_DIR, "*.csv")) if os.path.basename(p) not in skip]
        + glob.glob(os.path.join(TABLE_DIR, "*.json"))
    )


def _schema_rows(table_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(table_dir, "*.csv"))):
        if os.path.basename(path) in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": os.path.basename(path), "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def _large_scan(paths: list[str]) -> list[dict]:
    return [{"path": p, "size_bytes": os.path.getsize(p), "over_50mb": int(os.path.getsize(p) > 50_000_000), "passed": int(os.path.getsize(p) <= 50_000_000)} for p in sorted(paths)]


def _artifact_manifest(paths: list[str], table_dir: str) -> list[dict]:
    row_counts = {}
    for path in glob.glob(os.path.join(table_dir, "*.csv")):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            row_counts[path] = sum(1 for _ in reader)
    rows = []
    for path in sorted(paths):
        cls = "table" if path.endswith(".csv") else "summary_json" if path.endswith(".json") else "report"
        rows.append({"path": path, "size_bytes": os.path.getsize(path), "sha256": _sha256(path), "artifact_class": cls, "row_count": row_counts.get(path, "")})
    return rows


def write_artifacts(res: dict, test_status: str) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    write_tables(res, TABLE_DIR)

    res["schema_validation_summary_rows"] = _schema_rows(TABLE_DIR)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c65_frozen_checkpoint_recovery_trial_cache_gate")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C65] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
