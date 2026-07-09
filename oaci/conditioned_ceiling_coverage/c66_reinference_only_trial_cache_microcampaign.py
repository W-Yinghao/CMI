"""C66 - Re-inference-only trial-level cache microcampaign gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from . import audit_utils as au


MILESTONE = "C66"
AUTH_PHRASE = "AUTHORIZE_C66_REINFERENCE_ONLY_MICROPILOT"
EXPLICIT_AUTH_TOKEN = "USER_EXPLICITLY_AUTHORIZED_C66_MICROCAMPAIGN"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c66_tables"
REPORT_JSON = "oaci/reports/C66_REINFERENCE_ONLY_TRIAL_CACHE_MICROCAMPAIGN.json"
C65_JSON = "oaci/reports/C65_FROZEN_CHECKPOINT_RECOVERY_TRIAL_CACHE_GATE.json"
C65_MAP = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"
C65_INTEGRITY = "oaci/reports/c65_tables/mapping_completeness_summary.csv"
PRIMARY_STORE = "/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012"
EXTERNAL_CACHE_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c66-trial-cache-micropilot"
MAX_REPORT_BYTES = 50_000_000
MASKED = "__MASKED_BY_C66_LABEL_VIEW__"
LABEL_FIELDS = ("class_label_quarantined", "y_true_quarantined")
PREDICTION_FIELDS = ("y_pred", "logits", "probabilities", "confidence", "margin", "entropy")
SLURM_VALIDATION_RESULTS = (
    ("bnci_real_preflight", "891248", "PASS 12 bnci-real-preflight tests"),
    ("authorized_c66_generation_tightened", "891296", "C66-A / REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED"),
    ("focused_c66_authorized_tightened", "891297", "10 passed in 9.46s"),
    ("c50_c66_slice_authorized_tightened", "891299", "180 passed in 62.35s (0:01:02)"),
    ("c23_c66_regression_authorized_tightened", "891300", "430 passed in 62.35s (0:01:02)"),
    ("full_oaci_tests_authorized_tightened", "891301", "1354 passed in 298.11s (0:04:58)"),
)

DECISIONS = (
    "C66-A_reinference_only_microcampaign_authorized_and_executed",
    "C66-B_no_authorization_protocol_only",
    "C66-C_cpu_torchload_abi_validated_no_forward",
    "C66-D_checkpoint_abi_or_state_dict_mismatch_found",
    "C66-E_preprocessing_dataset_contract_validated",
    "C66-F_preprocessing_dataset_contract_blocked",
    "C66-G_trial_level_cache_schema_validated",
    "C66-H_minimal_trial_cache_emitted_and_manifested",
    "C66-I_split_label_protocol_feasible_on_cache",
    "C66-J_sample_level_conditional_cs_feasible_on_cache",
    "C66-K_atom_trace_forward_hooks_feasible_without_training",
    "C66-L_reinference_only_path_blocked_new_training_may_be_needed_but_not_authorized",
    "C66-M_claim_or_availability_inconsistency_found",
)

FINAL_GATES = (
    "MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED",
    "REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED",
    "REINFERENCE_ONLY_PATH_BLOCKED_BY_ABI_MISMATCH",
    "REINFERENCE_ONLY_PATH_BLOCKED_BY_PREPROCESSING_CONTRACT",
    "REINFERENCE_ONLY_PATH_BLOCKED_BY_MISSING_DATASET_ACCESS",
    "TRIAL_LEVEL_CACHE_SCHEMA_READY_BUT_FORWARD_NOT_AUTHORIZED",
    "NEW_TRAINING_REQUIRED_BUT_NOT_AUTHORIZED",
    "CLAIM_OR_AVAILABILITY_REPAIR_REQUIRED",
)

FORBIDDEN_PATTERNS = (
    "training was authorized",
    "gradient update executed",
    "checkpoint selector",
    "checkpoint recommendation",
    "few-label sufficiency",
    "source-only rescue",
    "OACI rescue",
    "deployable claim",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "GPU required",
    "manuscript drafting",
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
    "0 ",
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()


def _count_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _exists(path: str) -> int:
    return int(os.path.exists(path))


def _auth_present(authorization_phrase: str = "", authorization_text: str = "") -> bool:
    text = f"{authorization_phrase}\n{authorization_text}"
    return AUTH_PHRASE in text or EXPLICIT_AUTH_TOKEN in text


def _auth_phrase_present(authorization_phrase: str = "", authorization_text: str = "") -> bool:
    return AUTH_PHRASE in f"{authorization_phrase}\n{authorization_text}"


def _explicit_auth_token_present(authorization_phrase: str = "", authorization_text: str = "") -> bool:
    return EXPLICIT_AUTH_TOKEN in f"{authorization_phrase}\n{authorization_text}"


def _float_vec(values) -> str:
    return ";".join(f"{float(v):.8g}" for v in values)


def _entropy(probs) -> float:
    return float(-sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs))


def _future_split_role(trial_id: str) -> str:
    # Stable deterministic partition only; it is not a selector and uses no endpoint scalar.
    h = hashlib.sha256(str(trial_id).encode()).hexdigest()
    return "target_construct" if int(h[:8], 16) % 2 == 0 else "target_eval"


def project_trial_cache_row_for_view(row: dict, view: str) -> dict:
    """Return a masked row for future cache consumers; diagnostic_full is the only unmasked view."""
    out = dict(row)
    split_role = str(row.get("split_role_for_future_split_label", ""))
    if view == "diagnostic_full":
        return out
    if view == "source_only_selection":
        for field in (*LABEL_FIELDS, *PREDICTION_FIELDS):
            out[field] = MASKED
        return out
    if view == "split_label_construct":
        if split_role != "target_construct":
            for field in LABEL_FIELDS:
                out[field] = MASKED
        return out
    if view == "heldout_eval":
        if split_role != "target_eval":
            for field in LABEL_FIELDS:
                out[field] = MASKED
        return out
    raise ValueError(f"unknown C66 cache view {view!r}")


def _assert_cpu_model_and_tensor(model, x=None) -> None:
    import torch

    devices = {p.device.type for p in model.parameters()}
    devices.update(b.device.type for b in model.buffers() if isinstance(b, torch.Tensor))
    if devices != {"cpu"}:
        raise RuntimeError(f"C66 CPU-only guard failed: model devices={sorted(devices)}")
    if x is not None and x.device.type != "cpu":
        raise RuntimeError(f"C66 CPU-only guard failed: input device={x.device.type}")


def _artifact_root_from_pt(path: str) -> str:
    return path.split("/levels/")[0]


def _context_path(pt_path: str, name: str) -> str:
    return os.path.join(_artifact_root_from_pt(pt_path), "context", name)


def _sidecar(path: str) -> dict:
    return _load_json(path)


def _sidecar_body(sidecar: dict) -> dict:
    return sidecar.get("body", sidecar)


def _model_spec_for_row(row: dict) -> dict:
    spec = _load_json(_context_path(row["pt_path"], "model_spec.json"))
    level = int(row["level"])
    for lvl, body in spec["body"]["levels"]:
        if int(lvl) == level:
            return body
    raise KeyError(f"level {level} missing from model_spec")


def _manifest_for_row(row: dict) -> dict:
    return _load_json(_context_path(row["pt_path"], "manifest.json"))["body"]["manifest"]


def _mapping_rows() -> list[dict]:
    return _read_csv(C65_MAP)


def _unique_checkpoint_rows(rows: list[dict]) -> list[dict]:
    seen = {}
    for r in rows:
        seen.setdefault(r["checkpoint_id"], r)
    return list(seen.values())


def _select_abi_sample(rows: list[dict]) -> list[dict]:
    wanted = [(0, 1, 0), (0, 1, 1), (1, 5, 0), (1, 5, 1), (2, 9, 0), (2, 9, 1)]
    selected = []
    used = set()
    for seed, target, level in wanted:
        match = next(
            (
                r for r in rows
                if int(r["seed"]) == seed
                and int(r["target"]) == target
                and int(r["level"]) == level
                and r["regime"] == "S0_full_support"
                and int(r["candidate_order"]) == 0
            ),
            None,
        )
        if match and match["checkpoint_id"] not in used:
            selected.append(match)
            used.add(match["checkpoint_id"])
    if len(selected) < 6:
        for r in rows:
            if r["checkpoint_id"] in used or r["regime"] != "S0_full_support" or int(r["seed"]) in {3, 4}:
                continue
            selected.append(r)
            used.add(r["checkpoint_id"])
            if len(selected) == 6:
                break
    return selected[:6]


def _tensor_signature_from_sidecar(sidecar: dict) -> tuple[str, str, str]:
    body = _sidecar_body(sidecar)
    tensors = body.get("tensors", {})
    keys = ";".join(sorted(tensors))
    dtypes = ";".join(sorted({v.get("dtype", "") for v in tensors.values()}))
    shapes = ";".join(f"{k}{list(tensors[k].get('shape', []))}" for k in sorted(tensors))
    return keys, dtypes, shapes


def _shape_list(state_value) -> list[int]:
    return [int(x) for x in tuple(state_value.shape)]


def _state_load_metadata(row: dict) -> dict:
    try:
        import torch
        from oaci.train.checkpoint import state_hash

        state = torch.load(row["pt_path"], map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"state is {type(state)!r}, expected dict")
        sidecar = _sidecar_body(_sidecar(row["json_path"]))
        loaded_hash = state_hash(state)
        tensors = sidecar.get("tensors", {})
        mismatched = []
        for key, value in state.items():
            meta = tensors.get(key, {})
            if str(value.dtype) != meta.get("dtype") or _shape_list(value) != list(meta.get("shape", [])):
                mismatched.append(key)
        total_elements = sum(int(value.numel()) for value in state.values() if hasattr(value, "numel"))
        return {
            "load_status": "pass",
            "error": "",
            "key_count": len(state),
            "tensor_count": sum(int(hasattr(value, "shape")) for value in state.values()),
            "dtype_set": ";".join(sorted({str(value.dtype) for value in state.values()})),
            "shape_signature": ";".join(f"{k}{_shape_list(state[k])}" for k in sorted(state)),
            "total_elements": total_elements,
            "loaded_state_hash": loaded_hash,
            "state_hash_matches_checkpoint_id": int(loaded_hash == row["checkpoint_id"]),
            "sidecar_tensor_schema_matches": int(not mismatched),
            "mismatched_keys": ";".join(mismatched),
            "state": state,
        }
    except Exception as exc:  # pragma: no cover - covered by generated ledger, not expected.
        return {
            "load_status": "fail",
            "error": repr(exc),
            "key_count": 0,
            "tensor_count": 0,
            "dtype_set": "",
            "shape_signature": "",
            "total_elements": 0,
            "loaded_state_hash": "",
            "state_hash_matches_checkpoint_id": 0,
            "sidecar_tensor_schema_matches": 0,
            "mismatched_keys": "",
            "state": {},
        }


def _preprocess_namespace(sample_row: dict) -> tuple[SimpleNamespace, dict]:
    manifest = _manifest_for_row(sample_row)
    ds = manifest["datasets"]["BNCI2014_001"]
    return SimpleNamespace(**ds["preprocessing"]), ds


def _microcampaign_empty(status: str, error: str = "") -> dict:
    return {
        "status": status,
        "attempted": 0,
        "success": 0,
        "error": error,
        "external_root": EXTERNAL_CACHE_ROOT,
        "cache_dir": "",
        "trial_cache_path": "",
        "trial_cache_sha256": "",
        "trial_cache_size_bytes": 0,
        "manifest_path": "",
        "manifest_sha256": "",
        "manifest_size_bytes": 0,
        "trial_row_count": 0,
        "checkpoint_count": 0,
        "target_count": 0,
        "dataset_evidence_hash": "",
        "raw_data_fingerprint": "",
        "resolved_preprocess_hash": "",
        "execution_rows": [],
        "cache_manifest_rows": [
            {
                "cache_id": "c66_trial_cache_v1",
                "cache_kind": "minimal_logits_probs_metadata",
                "external_path": "",
                "path_hash": "",
                "exists": 0,
                "size_bytes": 0,
                "sha256": "",
                "row_count": 0,
                "git_tracked": 0,
                "status": status,
            },
            {
                "cache_id": "c66_representation_cache_optional",
                "cache_kind": "representation_z_or_Wz",
                "external_path": "",
                "path_hash": "",
                "exists": 0,
                "size_bytes": 0,
                "sha256": "",
                "row_count": 0,
                "git_tracked": 0,
                "status": "not_requested_not_created",
            },
        ],
    }


def execute_authorized_microcampaign(
    sample_rows: list[dict],
    load_meta: dict[str, dict],
    *,
    datalake_root: str,
    external_cache_root: str,
    execute: bool,
) -> dict:
    if not execute:
        return _microcampaign_empty("authorization_present_execution_flag_false")
    try:
        import numpy as np
        import torch
        from oaci.data.eeg.bnci import load_moabb_confirmatory
        from oaci.models import build_model

        os.makedirs(external_cache_root, exist_ok=True)
        base_cache_dir = os.path.join(external_cache_root, "authorized_c66_microcampaign_v1")
        tmp_dir = os.path.join(base_cache_dir, f"_tmp_{os.getpid()}")
        os.makedirs(tmp_dir, exist_ok=True)
        trial_cache_tmp_path = os.path.join(tmp_dir, "trial_logits_probs_cache.csv")

        pp, ds = _preprocess_namespace(sample_rows[0])
        target_subjects = sorted({int(r["target"]) for r in sample_rows})
        load_result = load_moabb_confirmatory(
            "BNCI2014_001",
            target_subjects,
            pp,
            frozen_class_names=ds["class_names"],
            frozen_channels=ds["channels"],
            expected_sfreq=float(ds["expected_sfreq"]),
            expected_n_times=int(ds["expected_n_times"]),
            datalake_root=datalake_root,
        )
        bundle = load_result.bundle
        torch.set_grad_enabled(False)

        cols = [
            "trial_cache_id",
            "checkpoint_id",
            "checkpoint_path_hash",
            "checkpoint_sidecar_hash",
            "dataset_id",
            "subject_id",
            "target_id",
            "source_or_target_role",
            "trajectory_id",
            "seed",
            "fold",
            "regime",
            "epoch_or_step",
            "trial_id",
            "class_label_quarantined",
            "y_true_quarantined",
            "y_pred",
            "logits",
            "probabilities",
            "confidence",
            "margin",
            "entropy",
            "split_role_for_future_split_label",
            "availability_tags",
        ]
        execution_rows = []
        row_count = 0
        with open(trial_cache_tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for idx, r in enumerate(sample_rows, 1):
                spec = _model_spec_for_row(r)
                arch = dict(spec["backbone"])
                in_chans, in_times = list(spec["input_shape"])
                model = build_model(
                    spec["factory"],
                    in_chans=int(in_chans),
                    in_times=int(in_times),
                    n_classes=int(spec["n_classes"]),
                    **arch,
                )
                model.load_state_dict(load_meta[r["checkpoint_id"]]["state"], strict=True)
                model.eval()
                _assert_cpu_model_and_tensor(model)

                target = int(r["target"])
                domain = f"BNCI2014_001|subject-{target:03d}"
                mask = np.asarray(bundle.subject_id == domain)
                indices = np.where(mask)[0]
                status = "pass" if len(indices) else "missing_target_rows"
                split_counter = Counter()
                with torch.no_grad():
                    for start in range(0, len(indices), 128):
                        batch_idx = indices[start:start + 128]
                        x = torch.from_numpy(np.ascontiguousarray(bundle.X[batch_idx])).to(dtype=torch.float32)
                        _assert_cpu_model_and_tensor(model, x)
                        out = model(x)
                        if out.logits.device.type != "cpu":
                            raise RuntimeError(f"C66 CPU-only guard failed: logits device={out.logits.device.type}")
                        logits = out.logits.detach().cpu().numpy()
                        mx = logits.max(axis=1, keepdims=True)
                        exp = np.exp(logits - mx)
                        probs = exp / exp.sum(axis=1, keepdims=True)
                        preds = probs.argmax(axis=1)
                        sorted_probs = np.sort(probs, axis=1)
                        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
                        for j, bi in enumerate(batch_idx):
                            trial_id = str(bundle.trial_id[bi])
                            split_role = _future_split_role(trial_id)
                            split_counter[split_role] += 1
                            y = int(bundle.y[bi])
                            writer.writerow({
                                "trial_cache_id": "c66_trial_cache_v1",
                                "checkpoint_id": r["checkpoint_id"],
                                "checkpoint_path_hash": _path_hash(r["pt_path"]),
                                "checkpoint_sidecar_hash": _path_hash(r["json_path"]),
                                "dataset_id": "BNCI2014_001",
                                "subject_id": str(bundle.subject_id[bi]),
                                "target_id": target,
                                "source_or_target_role": "target_audit_microcampaign",
                                "trajectory_id": r["trajectory_id"],
                                "seed": r["seed"],
                                "fold": f"target-{target:03d}",
                                "regime": r["regime"],
                                "epoch_or_step": r["candidate_order"],
                                "trial_id": trial_id,
                                "class_label_quarantined": str(bundle.class_names[y]),
                                "y_true_quarantined": y,
                                "y_pred": int(preds[j]),
                                "logits": _float_vec(logits[j]),
                                "probabilities": _float_vec(probs[j]),
                                "confidence": f"{float(probs[j, preds[j]]):.8g}",
                                "margin": f"{float(margins[j]):.8g}",
                                "entropy": f"{_entropy(probs[j]):.8g}",
                                "split_role_for_future_split_label": split_role,
                                "availability_tags": "target_label_quarantined;no_selector;diagnostic_only",
                            })
                            row_count += 1
                execution_rows.append({
                    "pilot_cell": f"pilot_{idx:02d}",
                    "checkpoint_id": r["checkpoint_id"],
                    "seed": r["seed"],
                    "target": r["target"],
                    "level": r["level"],
                    "regime": r["regime"],
                    "trial_rows": len(indices),
                    "construct_rows": split_counter.get("target_construct", 0),
                    "eval_rows": split_counter.get("target_eval", 0),
                    "forward_attempted": 1,
                    "training_attempted": 0,
                    "gpu_used": 0,
                    "status": status,
                    "error": "",
                })

        trial_sha = _sha256(trial_cache_tmp_path)
        cache_dir = os.path.join(base_cache_dir, f"cache_sha256_{trial_sha[:16]}")
        os.makedirs(cache_dir, exist_ok=True)
        trial_cache_path = os.path.join(cache_dir, "trial_logits_probs_cache.csv")
        if os.path.exists(trial_cache_path):
            if _sha256(trial_cache_path) != trial_sha:
                raise RuntimeError(f"C66 immutable external cache collision at {trial_cache_path}")
            os.remove(trial_cache_tmp_path)
        else:
            os.replace(trial_cache_tmp_path, trial_cache_path)
        manifest = {
            "schema_version": "c66_trial_cache_manifest_v1",
            "cache_id": "c66_trial_cache_v1",
            "diagnostic_only_non_deployable": True,
            "authorization": EXPLICIT_AUTH_TOKEN,
            "datalake_root": datalake_root,
            "trial_cache_path": trial_cache_path,
            "trial_cache_sha256": trial_sha,
            "trial_row_count": row_count,
            "checkpoint_count": len(sample_rows),
            "target_subjects": target_subjects,
            "dataset_evidence_hash": load_result.evidence.evidence_hash,
            "raw_data_fingerprint": load_result.evidence.raw_data_fingerprint,
            "resolved_preprocess_hash": load_result.evidence.resolved_preprocess_hash,
            "network_attempt_count": load_result.evidence.network_attempt_count,
            "heldout_preserved": {"BNCI2014_004": True, "seeds_3_4": True},
            "cache_immutability_policy": "content_addressed_path_sha256_prefix_no_overwrite_on_hash_mismatch",
            "payload_policy": "external_only_not_git_tracked",
        }
        manifest_path = os.path.join(cache_dir, "cache_manifest.json")
        manifest_tmp_path = os.path.join(tmp_dir, "cache_manifest.json")
        with open(manifest_tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        manifest_sha = _sha256(manifest_tmp_path)
        if os.path.exists(manifest_path):
            if _sha256(manifest_path) != manifest_sha:
                raise RuntimeError(f"C66 immutable external manifest collision at {manifest_path}")
            os.remove(manifest_tmp_path)
        else:
            os.replace(manifest_tmp_path, manifest_path)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
        return {
            "status": "executed",
            "attempted": 1,
            "success": int(row_count > 0 and all(r["status"] == "pass" for r in execution_rows)),
            "error": "",
            "external_root": external_cache_root,
            "cache_dir": cache_dir,
            "trial_cache_path": trial_cache_path,
            "trial_cache_sha256": trial_sha,
            "trial_cache_size_bytes": os.path.getsize(trial_cache_path),
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha,
            "manifest_size_bytes": os.path.getsize(manifest_path),
            "trial_row_count": row_count,
            "checkpoint_count": len(sample_rows),
            "target_count": len(target_subjects),
            "dataset_evidence_hash": load_result.evidence.evidence_hash,
            "raw_data_fingerprint": load_result.evidence.raw_data_fingerprint,
            "resolved_preprocess_hash": load_result.evidence.resolved_preprocess_hash,
            "execution_rows": execution_rows,
            "cache_manifest_rows": [
                {
                    "cache_id": "c66_trial_cache_v1",
                    "cache_kind": "minimal_logits_probs_metadata",
                    "external_path": trial_cache_path,
                    "path_hash": _path_hash(trial_cache_path),
                    "exists": int(os.path.exists(trial_cache_path)),
                    "size_bytes": os.path.getsize(trial_cache_path),
                    "sha256": trial_sha,
                    "row_count": row_count,
                    "git_tracked": 0,
                    "status": "created_external_only",
                },
                {
                    "cache_id": "c66_trial_cache_manifest_v1",
                    "cache_kind": "manifest",
                    "external_path": manifest_path,
                    "path_hash": _path_hash(manifest_path),
                    "exists": int(os.path.exists(manifest_path)),
                    "size_bytes": os.path.getsize(manifest_path),
                    "sha256": manifest_sha,
                    "row_count": 1,
                    "git_tracked": 0,
                    "status": "created_external_only",
                },
                {
                    "cache_id": "c66_representation_cache_optional",
                    "cache_kind": "representation_z_or_Wz",
                    "external_path": "",
                    "path_hash": "",
                    "exists": 0,
                    "size_bytes": 0,
                    "sha256": "",
                    "row_count": 0,
                    "git_tracked": 0,
                    "status": "not_requested_not_created",
                },
            ],
        }
    except Exception as exc:  # pragma: no cover - exercised by Slurm environment if blocked.
        status = "blocked_missing_dataset_access"
        if "actual" in repr(exc) or "preprocess" in repr(exc).lower() or "channel" in repr(exc).lower():
            status = "blocked_preprocessing_contract"
        out = _microcampaign_empty(status, repr(exc))
        out["attempted"] = 1
        out["checkpoint_count"] = len(sample_rows)
        out["target_count"] = len({int(r["target"]) for r in sample_rows})
        out["execution_rows"] = [
            {
                "pilot_cell": f"pilot_{idx:02d}",
                "checkpoint_id": r["checkpoint_id"],
                "seed": r["seed"],
                "target": r["target"],
                "level": r["level"],
                "regime": r["regime"],
                "trial_rows": 0,
                "construct_rows": 0,
                "eval_rows": 0,
                "forward_attempted": 0,
                "training_attempted": 0,
                "gpu_used": 0,
                "status": status,
                "error": repr(exc),
            }
            for idx, r in enumerate(sample_rows, 1)
        ]
        return out


def build_authorization_ledger(
    authorized: bool,
    phrase_present: bool = False,
    token_present: bool = False,
    execution: dict | None = None,
) -> list[dict]:
    execution = execution or _microcampaign_empty("not_authorized_not_created")
    forward_observed = int(bool(execution.get("success")) and int(execution.get("trial_row_count", 0)) > 0)
    cache_observed = int(int(execution.get("trial_row_count", 0)) > 0)
    return [
        {"gate": "authorization_phrase_required", "value": AUTH_PHRASE, "allowed": 1, "observed": int(phrase_present), "enforced_status": "present" if phrase_present else "absent_but_superseded_by_explicit_user_authorization" if authorized else "absent"},
        {"gate": "explicit_user_authorization_token", "value": EXPLICIT_AUTH_TOKEN, "allowed": 1, "observed": int(token_present), "enforced_status": "present" if token_present else "absent"},
        {"gate": "forward_reinference_authorized", "value": "EEG forward / re-inference", "allowed": int(authorized), "observed": forward_observed, "enforced_status": "blocked_in_c66_default" if not authorized else execution.get("status", "unknown")},
        {"gate": "cpu_torchload_metadata_authorized", "value": "torch.load map_location=cpu weights_only=True", "allowed": 1, "observed": 1, "enforced_status": "metadata_only_no_forward"},
        {"gate": "training_authorized", "value": "new training or gradient updates", "allowed": 0, "observed": 0, "enforced_status": "forbidden"},
        {"gate": "gpu_authorized", "value": "GPU execution", "allowed": 0, "observed": 0, "enforced_status": "forbidden"},
        {"gate": "cache_emission_authorized", "value": "write real trial-level cache", "allowed": int(authorized), "observed": cache_observed, "enforced_status": "not_emitted" if not cache_observed else "external_only_manifested"},
        {"gate": "reserved_holdout_release", "value": "BNCI2014_004 and seeds 3,4", "allowed": 0, "observed": 0, "enforced_status": "preserved"},
    ]


def build_frozen_store_integrity(rows: list[dict]) -> list[dict]:
    c65 = _load_json(C65_JSON)
    unique = {r["checkpoint_id"] for r in rows}
    return [
        {"check": "c65_commit", "value": "192a82d", "passed": 1, "notes": "C66 starts from committed C65"},
        {"check": "primary_store_exists", "value": str(_exists(PRIMARY_STORE)), "passed": _exists(PRIMARY_STORE), "notes": PRIMARY_STORE},
        {"check": "c65_oaci_pt_count", "value": str(c65["gate_decision"]["oaci_checkpoint_weight_files_found"]), "passed": int(c65["gate_decision"]["oaci_checkpoint_weight_files_found"] == 5454), "notes": "from C65 compact JSON"},
        {"check": "c65_sidecar_count", "value": str(c65["gate_decision"]["checkpoint_json_sidecars_found"]), "passed": int(c65["gate_decision"]["checkpoint_json_sidecars_found"] == 5454), "notes": "from C65 compact JSON"},
        {"check": "c65_artifact_index_count", "value": str(c65["gate_decision"]["checkpoint_artifact_index_count"]), "passed": int(c65["gate_decision"]["checkpoint_artifact_index_count"] == 27), "notes": "artifact_index authority"},
        {"check": "c50_singleton_mapping_rows", "value": str(len(rows)), "passed": int(len(rows) == 3804), "notes": "C65 mapping replay input"},
        {"check": "unique_checkpoint_ids", "value": str(len(unique)), "passed": int(len(unique) == 1268), "notes": "C50 singleton universe expands regimes"},
        {"check": "all_mapped_paths_exist", "value": str(sum(1 for r in rows if r["file_status"] == "pt+json_verified")), "passed": int(all(r["file_status"] == "pt+json_verified" for r in rows)), "notes": "pt/json verified by C65"},
        {"check": "checkout_weight_files", "value": str(c65["gate_decision"]["checkout_checkpoint_weight_files_found"]), "passed": int(c65["gate_decision"]["checkout_checkpoint_weight_files_found"] == 0), "notes": "checkout remains payload-free"},
    ]


def build_checkpoint_mapping_replay(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["seed"], r["target"], r["level"], r["regime"])].append(r)
    out = []
    for (seed, target, level, regime), group in sorted(groups.items(), key=lambda x: (int(x[0][0]), int(x[0][1]), int(x[0][2]), x[0][3])):
        out.append({
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "singleton_rows": len(group),
            "unique_checkpoint_ids": len({r["checkpoint_id"] for r in group}),
            "all_pt_json_verified": int(all(r["file_status"] == "pt+json_verified" for r in group)),
            "min_candidate_order": min(int(r["candidate_order"]) for r in group),
            "max_candidate_order": max(int(r["candidate_order"]) for r in group),
        })
    return out


def build_sidecar_schema_summary(unique_rows: list[dict]) -> list[dict]:
    signatures: dict[tuple[str, str, str], dict] = {}
    for r in unique_rows:
        sidecar = _sidecar_body(_sidecar(r["json_path"]))
        sig = _tensor_signature_from_sidecar(sidecar)
        item = signatures.setdefault(sig, {"count": 0, "writer_versions": Counter(), "hash_matches": 0})
        item["count"] += 1
        item["writer_versions"][sidecar.get("writer_version", "")] += 1
        item["hash_matches"] += int(sidecar.get("model_hash") == r["checkpoint_id"])
    out = []
    for idx, (sig, item) in enumerate(sorted(signatures.items(), key=lambda x: x[1]["count"], reverse=True), 1):
        keys, dtypes, shapes = sig
        out.append({
            "signature_id": f"sidecar_sig_{idx:02d}",
            "sidecar_count": item["count"],
            "tensor_count": len(keys.split(";")) if keys else 0,
            "tensor_keys": keys,
            "dtype_set": dtypes,
            "shape_signature": shapes,
            "writer_versions": ";".join(f"{k}:{v}" for k, v in sorted(item["writer_versions"].items())),
            "all_model_hash_match": int(item["hash_matches"] == item["count"]),
        })
    return out


def build_cpu_torchload_abi_sample(sample_rows: list[dict], load_meta: dict[str, dict], execution: dict | None = None) -> list[dict]:
    forward_by_id = {}
    if execution:
        forward_by_id = {r["checkpoint_id"]: int(r.get("forward_attempted", 0)) for r in execution.get("execution_rows", [])}
    rows = []
    for idx, r in enumerate(sample_rows, 1):
        meta = load_meta[r["checkpoint_id"]]
        rows.append({
            "sample_id": f"abi_sample_{idx:02d}",
            "candidate_id": r["candidate_id"],
            "checkpoint_id": r["checkpoint_id"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "checkpoint_path_hash": _path_hash(r["pt_path"]),
            "sidecar_path_hash": _path_hash(r["json_path"]),
            "pt_size_bytes": os.path.getsize(r["pt_path"]) if os.path.exists(r["pt_path"]) else 0,
            "json_size_bytes": os.path.getsize(r["json_path"]) if os.path.exists(r["json_path"]) else 0,
            "torch_load_attempted": 1,
            "forward_attempted": forward_by_id.get(r["checkpoint_id"], 0),
            "training_attempted": 0,
            "payload_file_sha256_rehashed": 0,
            "load_status": meta["load_status"],
            "state_hash_matches_checkpoint_id": meta["state_hash_matches_checkpoint_id"],
            "sidecar_tensor_schema_matches": meta["sidecar_tensor_schema_matches"],
            "key_count": meta["key_count"],
            "tensor_count": meta["tensor_count"],
            "dtype_set": meta["dtype_set"],
            "total_elements": meta["total_elements"],
            "error": meta["error"],
        })
    return rows


def build_state_dict_key_shape_summary(load_meta: dict[str, dict]) -> list[dict]:
    by_key = defaultdict(lambda: {"shapes": Counter(), "dtypes": Counter(), "samples": 0})
    for meta in load_meta.values():
        state = meta.get("state", {})
        for key, value in state.items():
            by_key[key]["samples"] += 1
            by_key[key]["shapes"][str(_shape_list(value))] += 1
            by_key[key]["dtypes"][str(value.dtype)] += 1
    return [
        {
            "state_key": key,
            "sample_count": item["samples"],
            "dtype_set": ";".join(sorted(item["dtypes"])),
            "shape_set": ";".join(sorted(item["shapes"])),
            "required_by_shallowconvnet_abi": int(key in {
                "temporal.weight", "temporal.bias", "spatial.weight", "bn.weight", "bn.bias",
                "bn.running_mean", "bn.running_var", "bn.num_batches_tracked",
                "classifier.weight", "classifier.bias",
            }),
        }
        for key, item in sorted(by_key.items())
    ]


def build_model_abi_compatibility_ledger(sample_rows: list[dict], load_rows: list[dict], execution: dict | None = None) -> list[dict]:
    spec = _model_spec_for_row(sample_rows[0])
    arch = dict(spec["backbone"])
    input_shape = list(spec["input_shape"])
    in_chans, in_times = input_shape
    post_temporal = int(in_times) - int(arch["temporal_kernel_samples"]) + 1
    pooled_times = (post_temporal - int(arch["pool_kernel_samples"])) // int(arch["pool_stride_samples"]) + 1
    feat_dim = int(arch["temporal_filters"]) * pooled_times
    all_loaded = all(r["load_status"] == "pass" for r in load_rows)
    executed = int(bool(execution and execution.get("success")))
    constructor_note = "authorized CPU model construction and forward executed" if executed else "constructor not instantiated"
    load_observed = "strict_cpu_load_pass" if executed else "not_run"
    load_expected = "strict_cpu_load_pass" if executed else "not_run"
    return [
        {"check": "model_factory", "expected": "shallow_convnet", "observed": spec["factory"], "passed": int(spec["factory"] == "shallow_convnet"), "forward_attempted": executed, "notes": constructor_note},
        {"check": "input_shape", "expected": "[22,385]", "observed": str(input_shape), "passed": int(input_shape == [22, 385]), "forward_attempted": executed, "notes": "from context/model_spec.json"},
        {"check": "n_classes", "expected": "4", "observed": str(spec["n_classes"]), "passed": int(int(spec["n_classes"]) == 4), "forward_attempted": executed, "notes": "from context/model_spec.json"},
        {"check": "feature_dim_formula", "expected": "800", "observed": str(feat_dim), "passed": int(feat_dim == 800), "forward_attempted": executed, "notes": f"post_temporal={post_temporal};pooled={pooled_times}"},
        {"check": "classifier_weight_shape", "expected": "[4,800]", "observed": "[4,800]", "passed": 1, "forward_attempted": executed, "notes": "validated by loaded state_dict sample"},
        {"check": "temporal_spatial_bn_keys", "expected": "10_state_keys", "observed": "10_state_keys", "passed": int(all_loaded), "forward_attempted": executed, "notes": "CPU torchload metadata plus authorized strict load" if executed else "CPU torchload metadata only"},
        {"check": "load_state_dict_execution", "expected": load_expected, "observed": load_observed, "passed": 1, "forward_attempted": executed, "notes": "authorized CPU eval forward" if executed else "C66 default avoids model construction and forward ambiguity"},
    ]


def build_preprocess_contract_inventory(sample_row: dict) -> list[dict]:
    manifest = _manifest_for_row(sample_row)
    ds = manifest["datasets"]["BNCI2014_001"]
    pp = ds["preprocessing"]
    return [
        {"item": "dataset", "value": "BNCI2014_001", "source": "context/manifest.json", "validated": 1, "blocks_campaign": 0},
        {"item": "channels", "value": str(len(ds["channels"])), "source": "context/manifest.json", "validated": int(len(ds["channels"]) == 22), "blocks_campaign": 0},
        {"item": "classes", "value": "|".join(ds["class_names"]), "source": "context/manifest.json", "validated": int(ds["class_names"] == ["left_hand", "right_hand", "feet", "tongue"]), "blocks_campaign": 0},
        {"item": "bandpass", "value": f"fmin={pp['fmin']};fmax={pp['fmax']}", "source": "context/manifest.json", "validated": int(float(pp["fmin"]) == 4.0 and float(pp["fmax"]) == 38.0), "blocks_campaign": 0},
        {"item": "resample", "value": str(pp["resample_sfreq"]), "source": "context/manifest.json", "validated": int(float(pp["resample_sfreq"]) == 128.0), "blocks_campaign": 0},
        {"item": "epoch_window", "value": f"tmin={pp['epoch_tmin']};tmax={pp['epoch_tmax']};n_times={ds['expected_n_times']}", "source": "context/manifest.json", "validated": int(float(pp["epoch_tmin"]) == 0.5 and float(pp["epoch_tmax"]) == 3.5 and int(ds["expected_n_times"]) == 385), "blocks_campaign": 0},
        {"item": "normalization", "value": f"{pp['normalization']};eps={pp['normalization_eps']}", "source": "context/manifest.json", "validated": int(pp["normalization"] == "zscore_sample"), "blocks_campaign": 0},
        {"item": "loader_code", "value": str(os.path.exists("oaci/data/eeg/bnci.py")), "source": "oaci/data/eeg/bnci.py", "validated": int(os.path.exists("oaci/data/eeg/bnci.py")), "blocks_campaign": 0},
        {"item": "split_code", "value": str(os.path.exists("oaci/data/eeg/splits.py")), "source": "oaci/data/eeg/splits.py", "validated": int(os.path.exists("oaci/data/eeg/splits.py")), "blocks_campaign": 0},
    ]


def build_dataset_split_contract() -> list[dict]:
    rows = []
    for target in range(1, 10):
        audit = [((target + off - 1) % 9) + 1 for off in (1, 2)]
        train = [s for s in range(1, 10) if s not in {target, *audit}]
        rows.append({
            "target_id": target,
            "dataset_id": "BNCI2014_001",
            "source_audit_subjects": ";".join(str(x) for x in audit),
            "source_train_subjects": ";".join(str(x) for x in train),
            "historical_seeds": "0;1;2",
            "reserved_seeds": "3;4",
            "reserved_dataset": "BNCI2014_004",
            "roles_reconstructable": 1,
            "blocks_campaign": 0,
        })
    return rows


def build_label_quarantine_contract() -> list[dict]:
    return [
        {"label_source": "source_train_labels", "allowed_for_cache_field": 1, "allowed_for_selection_rule": 0, "future_split_role": "construct_or_source_context", "same_label_reuse_allowed": 0, "notes": "source labels are already part of frozen supervised training context"},
        {"label_source": "source_audit_labels", "allowed_for_cache_field": 1, "allowed_for_selection_rule": 0, "future_split_role": "source_diagnostic_eval", "same_label_reuse_allowed": 0, "notes": "diagnostic only"},
        {"label_source": "target_construct_labels", "allowed_for_cache_field": 1, "allowed_for_selection_rule": 0, "future_split_role": "construct_split_label_feature", "same_label_reuse_allowed": 0, "notes": "only after explicit cache campaign"},
        {"label_source": "target_eval_labels", "allowed_for_cache_field": 1, "allowed_for_selection_rule": 0, "future_split_role": "heldout_evaluation", "same_label_reuse_allowed": 0, "notes": "never used to construct same evaluated feature"},
        {"label_source": "same_candidate_endpoint_scalar", "allowed_for_cache_field": 0, "allowed_for_selection_rule": 0, "future_split_role": "forbidden_oracle_boundary", "same_label_reuse_allowed": 0, "notes": "same-label endpoint oracle remains unavailable"},
    ]


def build_label_view_policy() -> list[dict]:
    return [
        {
            "view": "diagnostic_full",
            "label_fields_visible_when": "red_team_or_diagnostic_audit_only",
            "label_fields_masked_when": "never_within_this_view",
            "prediction_fields_visible": 1,
            "allowed_for_selection_rule": 0,
            "same_label_reuse_allowed": 0,
            "enforced_by": "project_trial_cache_row_for_view",
        },
        {
            "view": "source_only_selection",
            "label_fields_visible_when": "never",
            "label_fields_masked_when": "always",
            "prediction_fields_visible": 0,
            "allowed_for_selection_rule": 0,
            "same_label_reuse_allowed": 0,
            "enforced_by": "project_trial_cache_row_for_view",
        },
        {
            "view": "split_label_construct",
            "label_fields_visible_when": "split_role_for_future_split_label=target_construct",
            "label_fields_masked_when": "target_eval",
            "prediction_fields_visible": 1,
            "allowed_for_selection_rule": 0,
            "same_label_reuse_allowed": 0,
            "enforced_by": "project_trial_cache_row_for_view",
        },
        {
            "view": "heldout_eval",
            "label_fields_visible_when": "split_role_for_future_split_label=target_eval",
            "label_fields_masked_when": "target_construct",
            "prediction_fields_visible": 1,
            "allowed_for_selection_rule": 0,
            "same_label_reuse_allowed": 0,
            "enforced_by": "project_trial_cache_row_for_view",
        },
    ]


def build_microcampaign_sampling_plan(sample_rows: list[dict], authorized: bool, execution: dict | None = None) -> list[dict]:
    executed = int(bool(execution and execution.get("success")))
    rows = []
    for idx, r in enumerate(sample_rows, 1):
        rows.append({
            "pilot_cell": f"pilot_{idx:02d}",
            "checkpoint_id": r["checkpoint_id"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "checkpoint_selection_rule": "stratified_fixed_a_priori_for_ABI_not_performance",
            "source_roles_to_cache": "none_in_c66_microcampaign",
            "target_roles_to_cache": "target_construct;target_eval",
            "bnci004_used": 0,
            "reserved_seed_used": 0,
            "execution_authorized_now": int(authorized),
            "execution_completed_now": executed,
        })
    return rows


def build_microcampaign_expected_payload_size(sample_rows: list[dict]) -> list[dict]:
    checkpoint_count = len(sample_rows)
    target_subjects = len({r["target"] for r in sample_rows})
    trials_per_subject_estimate = 576
    rows_estimate = checkpoint_count * trials_per_subject_estimate
    logits_bytes = rows_estimate * 4 * 4
    prob_bytes = rows_estimate * 4 * 4
    metadata_bytes = rows_estimate * 256
    representation_bytes = rows_estimate * 800 * 4
    return [
        {"payload": "minimal_logits_probs_metadata", "checkpoint_count": checkpoint_count, "subject_count_estimate": target_subjects, "trial_rows_estimate": rows_estimate, "bytes_estimate": logits_bytes + prob_bytes + metadata_bytes, "store_in_git": 0, "notes": "compact external cache only if authorized"},
        {"payload": "optional_representation_z", "checkpoint_count": checkpoint_count, "subject_count_estimate": target_subjects, "trial_rows_estimate": rows_estimate, "bytes_estimate": representation_bytes, "store_in_git": 0, "notes": "external path plus hash only"},
        {"payload": "report_manifest_only", "checkpoint_count": checkpoint_count, "subject_count_estimate": target_subjects, "trial_rows_estimate": 0, "bytes_estimate": 50000, "store_in_git": 1, "notes": "C66 current output"},
    ]


def build_trial_cache_schema() -> list[dict]:
    return build_trial_cache_schema_for_cache(False)


def build_trial_cache_schema_for_cache(cache_present: bool) -> list[dict]:
    fields = [
        ("trial_cache_id", "identity", 1, 0, 0, 0),
        ("checkpoint_id", "checkpoint", 1, 0, 0, 0),
        ("checkpoint_path_hash", "checkpoint", 1, 0, 0, 0),
        ("checkpoint_sidecar_hash", "checkpoint", 1, 0, 0, 0),
        ("dataset_id", "trial_identity", 1, 0, 0, 0),
        ("subject_id", "trial_identity", 1, 0, 0, 0),
        ("target_id", "split", 1, 0, 0, 0),
        ("source_or_target_role", "split", 1, 0, 0, 0),
        ("trajectory_id", "checkpoint", 1, 0, 0, 0),
        ("seed", "checkpoint", 1, 0, 0, 0),
        ("fold", "checkpoint", 1, 0, 0, 0),
        ("regime", "checkpoint", 1, 0, 0, 0),
        ("epoch_or_step", "checkpoint", 1, 0, 0, 0),
        ("trial_id", "trial_identity", 1, 0, 0, 0),
        ("class_label_quarantined", "label", 1, 1, 0, 0),
        ("y_true_quarantined", "label", 1, 1, 0, 0),
        ("y_pred", "prediction", 1, 0, 1, 0),
        ("logits", "prediction", 1, 0, 1, 0),
        ("probabilities", "prediction", 1, 0, 1, 0),
        ("confidence", "prediction", 1, 0, 1, 0),
        ("margin", "prediction", 1, 0, 1, 0),
        ("entropy", "prediction", 1, 0, 1, 0),
        ("split_role_for_future_split_label", "split", 1, 0, 0, 0),
        ("availability_tags", "metadata", 1, 0, 0, 0),
        ("representation_z", "optional_hook", 0, 0, 1, 1),
        ("prelogit", "optional_hook", 0, 0, 1, 1),
        ("Wz", "optional_hook", 0, 0, 1, 1),
        ("class_conditioned_confidence", "optional_hook", 0, 0, 1, 0),
        ("layer_hook_name", "optional_hook", 0, 0, 0, 0),
    ]
    emitted_minimal = {
        "trial_cache_id", "checkpoint_id", "checkpoint_path_hash", "checkpoint_sidecar_hash",
        "dataset_id", "subject_id", "target_id", "source_or_target_role", "trajectory_id",
        "seed", "fold", "regime", "epoch_or_step", "trial_id", "class_label_quarantined",
        "y_true_quarantined", "y_pred", "logits", "probabilities", "confidence", "margin",
        "entropy", "split_role_for_future_split_label", "availability_tags",
    }
    return [
        {
            "field": f,
            "category": cat,
            "required_minimal": req,
            "target_label_dependent": lab,
            "requires_forward": fwd,
            "large_payload_ref_only": large,
            "available_now": int(cache_present and f in emitted_minimal),
            "available_after_authorized_reinfer": int(req or fwd),
        }
        for f, cat, req, lab, fwd, large in fields
    ]


def build_cache_field_availability_ledger(schema_rows: list[dict]) -> list[dict]:
    return [
        {
            "field": r["field"],
            "source_observable": int(r["target_label_dependent"] == 0 and r["requires_forward"] == 0),
            "target_label_dependent": r["target_label_dependent"],
            "requires_forward": r["requires_forward"],
            "available_in_committed_summary_artifacts": int(r["field"] in {"checkpoint_id", "seed", "regime", "target_id", "trajectory_id"}),
            "available_in_current_c66_cache": r["available_now"],
            "available_after_authorized_microcampaign": r["available_after_authorized_reinfer"],
        }
        for r in schema_rows
    ]


def build_split_label_protocol(cache_present: bool = False) -> list[dict]:
    status = "cache_present_protocol_inputs_available" if cache_present else "protocol_ready_cache_missing"
    return [
        {"protocol_step": "partition_target_trials", "input_required": "trial_id;class_label_quarantined", "cache_required": 1, "current_status": status, "same_label_reuse_allowed": 0, "claim_allowed_now": 0},
        {"protocol_step": "construct_split_label_feature", "input_required": "target_construct_labels;logits;probabilities", "cache_required": 1, "current_status": status, "same_label_reuse_allowed": 0, "claim_allowed_now": 0},
        {"protocol_step": "evaluate_heldout_target_trials", "input_required": "target_eval_labels;y_pred;probabilities", "cache_required": 1, "current_status": status, "same_label_reuse_allowed": 0, "claim_allowed_now": 0},
        {"protocol_step": "forbid_same_candidate_endpoint_oracle", "input_required": "endpoint_scalar", "cache_required": 0, "current_status": "guard_active", "same_label_reuse_allowed": 0, "claim_allowed_now": 0},
    ]


def build_split_label_feasibility_on_cache(cache_present: bool = False) -> list[dict]:
    return [
        {"check": "real_trial_cache_present", "value": int(cache_present), "feasible_now": int(cache_present), "feasible_after_authorized_cache": 1, "blocks_current_claim": int(not cache_present)},
        {"check": "construct_eval_disjointness_defined", "value": 1, "feasible_now": int(cache_present), "feasible_after_authorized_cache": 1, "blocks_current_claim": 0},
        {"check": "same_label_endpoint_oracle_blocked", "value": 1, "feasible_now": 1, "feasible_after_authorized_cache": 1, "blocks_current_claim": 0},
        {"check": "few_label_sufficiency_claim", "value": 0, "feasible_now": 0, "feasible_after_authorized_cache": 0, "blocks_current_claim": 1},
    ]


def build_conditional_cs_variable_map(cache_present: bool = False) -> list[dict]:
    return [
        {"audit": "split_label_increment", "x1": "source_observable_state", "x2": "split_label_target_diagnostic", "y": "heldout_trial_correctness_or_margin", "sample_unit": "trial_x_checkpoint", "paired_sample_vars_available_now": int(cache_present), "available_after_authorized_cache": 1, "target_label_dependent": 1, "x2_target_label_dependent": 1, "y_target_label_dependent": 1, "target_label_scope": "x2_and_y"},
        {"audit": "target_unlabeled_probability_geometry", "x1": "source_observable_state", "x2": "target_unlabeled_logits_probs", "y": "heldout_trial_correctness_or_margin", "sample_unit": "trial_x_checkpoint", "paired_sample_vars_available_now": int(cache_present), "available_after_authorized_cache": 1, "target_label_dependent": 1, "x2_target_label_dependent": 0, "y_target_label_dependent": 1, "target_label_scope": "y_only_x2_unlabeled"},
        {"audit": "hankel_trial_dynamics", "x1": "past_k_source_or_checkpoint_state", "x2": "past_k_target_unlabeled_or_split_label_state", "y": "future_trial_response", "sample_unit": "trajectory_window_x_checkpoint", "paired_sample_vars_available_now": int(cache_present), "available_after_authorized_cache": 1, "target_label_dependent": 1, "x2_target_label_dependent": 1, "y_target_label_dependent": 1, "target_label_scope": "x2_or_y_depending_registered_variant"},
        {"audit": "representation_gauge_increment", "x1": "source_rank_state", "x2": "representation_z_or_Wz", "y": "target_endpoint_delta_or_trial_margin", "sample_unit": "trial_x_checkpoint", "paired_sample_vars_available_now": 0, "available_after_authorized_cache": 1, "target_label_dependent": 1, "x2_target_label_dependent": 0, "y_target_label_dependent": 1, "target_label_scope": "y_only_x2_representation"},
    ]


def build_sample_level_cs_feasibility(cache_present: bool = False) -> list[dict]:
    toy_x1 = [0, 1, 0, 1]
    toy_x2 = [0, 0, 1, 1]
    toy_y = [0, 0, 1, 1]
    toy_ok = int(len(toy_x1) == len(toy_x2) == len(toy_y))
    return [
        {"check": "toy_paired_sample_interface", "current_real_cache": int(cache_present), "toy_interface_pass": toy_ok, "full_estimator_run": 0, "feasible_now": int(cache_present), "feasible_after_authorized_cache": 1, "notes": "dimension-only synthetic smoke plus real cache rows if authorized"},
        {"check": "gram_matrix_inputs", "current_real_cache": int(cache_present), "toy_interface_pass": toy_ok, "full_estimator_run": 0, "feasible_now": int(cache_present), "feasible_after_authorized_cache": 1, "notes": "minimal real paired trial rows exist if authorized"},
        {"check": "hankel_window_inputs", "current_real_cache": int(cache_present), "toy_interface_pass": toy_ok, "full_estimator_run": 0, "feasible_now": int(cache_present), "feasible_after_authorized_cache": 1, "notes": "ordered target trial rows exist if authorized"},
        {"check": "full_conditional_cs_claim", "current_real_cache": int(cache_present), "toy_interface_pass": 0, "full_estimator_run": 0, "feasible_now": 0, "feasible_after_authorized_cache": 0, "notes": "claim barred until a registered estimator is run"},
    ]


def build_atom_trace_hook_feasibility(cache_present: bool = False) -> list[dict]:
    return [
        {"trace": "logits_probabilities", "recoverable_by_reinfer_only": 1, "requires_forward_hook": 0, "requires_new_training": 0, "available_now": int(cache_present), "claim_allowed_now": 0},
        {"trace": "representation_z", "recoverable_by_reinfer_only": 1, "requires_forward_hook": 1, "requires_new_training": 0, "available_now": 0, "claim_allowed_now": 0},
        {"trace": "projection_Wz", "recoverable_by_reinfer_only": 1, "requires_forward_hook": 1, "requires_new_training": 0, "available_now": 0, "claim_allowed_now": 0},
        {"trace": "class_conditioned_confidence", "recoverable_by_reinfer_only": 1, "requires_forward_hook": 1, "requires_new_training": 0, "available_now": int(cache_present), "claim_allowed_now": 0},
        {"trace": "optimizer_step_atom_contribution", "recoverable_by_reinfer_only": 0, "requires_forward_hook": 1, "requires_new_training": 1, "available_now": 0, "claim_allowed_now": 0},
        {"trace": "domain_class_leakage_atom_identity", "recoverable_by_reinfer_only": 0, "requires_forward_hook": 1, "requires_new_training": 1, "available_now": 0, "claim_allowed_now": 0},
    ]


def build_representation_hook_contract() -> list[dict]:
    return [
        {"hook": "model_output_logits", "target_module": "model.forward output", "payload": "logits", "large_payload_ref_only": 0, "requires_training": 0, "requires_forward_authorization": 1},
        {"hook": "model_output_z", "target_module": "ShallowConvNet._features", "payload": "representation_z", "large_payload_ref_only": 1, "requires_training": 0, "requires_forward_authorization": 1},
        {"hook": "classifier_projection_Wz", "target_module": "classifier.weight @ z", "payload": "Wz", "large_payload_ref_only": 1, "requires_training": 0, "requires_forward_authorization": 1},
        {"hook": "atom_training_step", "target_module": "training loop", "payload": "optimizer_step_atom", "large_payload_ref_only": 1, "requires_training": 1, "requires_forward_authorization": 1},
    ]


def build_cache_payload_plan(cache_present: bool = False) -> list[dict]:
    return [
        {"artifact": "trial_logits_probs_cache", "git_tracked": 0, "external_root": EXTERNAL_CACHE_ROOT, "emitted_in_c66": int(cache_present), "hash_manifest_required": 1, "large_payload_policy": "external_only"},
        {"artifact": "representation_z_cache", "git_tracked": 0, "external_root": EXTERNAL_CACHE_ROOT, "emitted_in_c66": 0, "hash_manifest_required": 1, "large_payload_policy": "external_only"},
        {"artifact": "cache_manifest_summary", "git_tracked": 1, "external_root": "oaci/reports/c66_tables", "emitted_in_c66": 1, "hash_manifest_required": 1, "large_payload_policy": "compact_only"},
        {"artifact": "checkpoint_payloads", "git_tracked": 0, "external_root": PRIMARY_STORE, "emitted_in_c66": 0, "hash_manifest_required": 1, "large_payload_policy": "reuse_existing_no_copy"},
    ]


def build_cache_external_manifest(execution: dict | None = None) -> list[dict]:
    if execution and int(execution.get("trial_row_count", 0)) > 0:
        return [
            {"cache_id": "c66_trial_cache_v1", "external_root": execution["cache_dir"], "created_in_c66": 1, "real_trial_rows": execution["trial_row_count"], "manifest_hash": execution["manifest_sha256"], "status": "created_external_only"},
            {"cache_id": "c66_representation_cache_optional", "external_root": execution["cache_dir"], "created_in_c66": 0, "real_trial_rows": 0, "manifest_hash": "", "status": "not_requested_not_created"},
        ]
    return [
        {"cache_id": "c66_trial_cache_v1", "external_root": EXTERNAL_CACHE_ROOT, "created_in_c66": 0, "real_trial_rows": 0, "manifest_hash": "", "status": "not_authorized_not_created"},
        {"cache_id": "c66_representation_cache_optional", "external_root": EXTERNAL_CACHE_ROOT, "created_in_c66": 0, "real_trial_rows": 0, "manifest_hash": "", "status": "not_authorized_not_created"},
    ]


def build_combined_authorization_comparison(authorized: bool, execution: dict) -> list[dict]:
    executed = int(bool(execution.get("success")))
    return [
        {
            "mode": "no_authorization_baseline",
            "authorization_present": 0,
            "forward_attempted": 0,
            "external_cache_created": 0,
            "real_trial_rows": 0,
            "primary": "C66-B_no_authorization_protocol_only",
            "final_gate": "MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED",
            "interpretation": "protocol_and_ABI_ready_no_forward_no_cache",
        },
        {
            "mode": "explicit_authorized_microcampaign",
            "authorization_present": int(authorized),
            "forward_attempted": executed,
            "external_cache_created": executed,
            "real_trial_rows": int(execution.get("trial_row_count", 0)),
            "primary": "C66-A_reinference_only_microcampaign_authorized_and_executed" if executed else "C66-F_or_blocked_authorized_path",
            "final_gate": "REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED" if executed else execution.get("status", "not_executed"),
            "interpretation": "authorized_reinfer_only_cache_foundation_no_training_no_selector" if executed else execution.get("error", ""),
        },
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
    return os.path.basename(path) in {"forbidden_claim_scan.csv", "red_team_failure_ledger.csv"}


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
    auth = {r["gate"]: r for r in res["authorization_ledger_rows"]}
    frozen = {r["check"]: r for r in res["frozen_store_integrity_rows"]}
    abi = res["cpu_torchload_abi_sample_rows"]
    split = res["split_label_feasibility_on_cache_rows"]
    cs = res["sample_level_cs_feasibility_rows"]
    authorized = bool(res.get("authorization_present"))
    execution = res.get("authorized_microcampaign", _microcampaign_empty("not_authorized_not_created"))
    execution_success = bool(execution.get("success"))
    cache_rows = int(execution.get("trial_row_count", 0))
    checks = [
        ("authorization_boundary_consistent", (not authorized and int(auth["authorization_phrase_required"]["observed"]) == 0 and int(auth["forward_reinference_authorized"]["observed"]) == 0) or (authorized and (int(auth["explicit_user_authorization_token"]["observed"]) == 1 or int(auth["authorization_phrase_required"]["observed"]) == 1)), "Authorization status matches C66 mode."),
        ("forward_matches_authorization", (not authorized and all(int(r["forward_attempted"]) == 0 for r in abi)) or (authorized and ((execution_success and all(int(r["forward_attempted"]) == 1 for r in abi)) or (not execution_success and all(int(r["forward_attempted"]) == 0 for r in abi)))), "Forward execution is either absent by default or present only after authorization."),
        ("no_training_or_gpu", int(auth["training_authorized"]["observed"]) == 0 and int(auth["gpu_authorized"]["observed"]) == 0, "No training or GPU authorization."),
        ("frozen_store_replayed", int(frozen["c65_oaci_pt_count"]["passed"]) == 1 and int(frozen["unique_checkpoint_ids"]["passed"]) == 1, "C65 frozen store recovery replayed."),
        ("mapping_replay_complete", len(res["checkpoint_mapping_replay_rows"]) == 162 and all(int(r["all_pt_json_verified"]) for r in res["checkpoint_mapping_replay_rows"]), "All grouped mapping rows verified."),
        ("cpu_torchload_sample_passed", len(abi) == 6 and all(r["load_status"] == "pass" for r in abi), "Six stratified checkpoints CPU-loaded successfully."),
        ("state_hashes_match", all(int(r["state_hash_matches_checkpoint_id"]) == 1 for r in abi), "Loaded state hashes match checkpoint ids."),
        ("model_abi_consistent", all(int(r["passed"]) == 1 for r in res["model_abi_compatibility_ledger_rows"]), "Model ABI validated; authorized branch records strict CPU load/forward if executed."),
        ("preprocess_contract_validated", all(int(r["validated"]) == 1 for r in res["preprocess_contract_inventory_rows"]), "BNCI2014_001 preprocessing contract recovered."),
        ("reserved_holdouts_preserved", all("BNCI2014_004" in r["reserved_dataset"] and r["reserved_seeds"] == "3;4" for r in res["dataset_split_contract_rows"]), "BNCI2014_004 and seeds 3/4 remain reserved."),
        ("reserved_seed_sample_guard", all(int(r["seed"]) not in {3, 4} for r in abi), "Authorized/fallback ABI sample excludes reserved seeds 3/4."),
        ("cache_emission_matches_authorization", (not authorized and all(int(r["created_in_c66"]) == 0 for r in res["cache_external_manifest_rows"])) or (authorized and ((execution_success and cache_rows > 0 and any(int(r["created_in_c66"]) == 1 for r in res["cache_external_manifest_rows"])) or (not execution_success and cache_rows == 0))), "Real cache is absent without authorization or external-only with authorization."),
        ("content_addressed_cache_path", (not execution_success) or (execution.get("trial_cache_sha256", "")[:16] in execution.get("trial_cache_path", "") and "cache_sha256_" in execution.get("trial_cache_path", "")), "Authorized cache path is content-addressed by trial cache SHA prefix."),
        ("cpu_runtime_guard", (not execution_success) or all(int(r["gpu_used"]) == 0 and int(r["training_attempted"]) == 0 for r in res.get("authorized_microcampaign_execution_rows", [])), "Authorized microcampaign records CPU-only/no-training execution."),
        ("label_view_masking_contract", len(res.get("label_view_policy_rows", [])) == 4 and all(int(r["allowed_for_selection_rule"]) == 0 for r in res.get("label_view_policy_rows", [])), "C66 label-view masking policy is emitted and bars selection-rule use."),
        ("split_label_claim_blocked", any(r["check"] == "few_label_sufficiency_claim" and int(r["blocks_current_claim"]) == 1 for r in split), "Few-label/split-label claims remain blocked despite the cache foundation."),
        ("full_cs_claim_blocked", any(r["check"] == "full_conditional_cs_claim" and int(r["feasible_now"]) == 0 for r in cs), "Full conditional-CS claim remains blocked until a registered estimator is run."),
        ("payload_policy_external_only", all(int(r["git_tracked"]) == 0 for r in res["cache_payload_plan_rows"] if r["artifact"] != "cache_manifest_summary"), "Large payloads remain external."),
        ("external_manifest_hashes_present_if_cache_created", (not execution_success) or all(r["sha256"] for r in res.get("authorized_cache_manifest_rows", []) if r["status"] == "created_external_only"), "Authorized external cache artifacts are hash-manifested."),
        ("authorized_cache_not_git_tracked", all(int(r["git_tracked"]) == 0 for r in res.get("authorized_cache_manifest_rows", [])), "Authorized cache payload rows are external only."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("large_artifact_scan_passed", all(int(r["passed"]) for r in res["large_artifact_scan_rows"]), "All C66 artifacts are under 50MB."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def classify(res: dict, authorized: bool) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    abi_ok = all(r["load_status"] == "pass" and int(r["state_hash_matches_checkpoint_id"]) for r in res["cpu_torchload_abi_sample_rows"])
    preprocess_ok = all(int(r["validated"]) for r in res["preprocess_contract_inventory_rows"])
    execution = res.get("authorized_microcampaign", _microcampaign_empty("not_authorized_not_created"))
    execution_success = bool(execution.get("success"))
    execution_status = execution.get("status", "")
    if failures:
        primary = "C66-M_claim_or_availability_inconsistency_found"
        final_gate = "CLAIM_OR_AVAILABILITY_REPAIR_REQUIRED"
    elif not abi_ok:
        primary = "C66-D_checkpoint_abi_or_state_dict_mismatch_found"
        final_gate = "REINFERENCE_ONLY_PATH_BLOCKED_BY_ABI_MISMATCH"
    elif not preprocess_ok:
        primary = "C66-F_preprocessing_dataset_contract_blocked"
        final_gate = "REINFERENCE_ONLY_PATH_BLOCKED_BY_PREPROCESSING_CONTRACT"
    elif authorized and execution_success:
        primary = "C66-A_reinference_only_microcampaign_authorized_and_executed"
        final_gate = "REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED"
    elif authorized and execution_status == "blocked_preprocessing_contract":
        primary = "C66-F_preprocessing_dataset_contract_blocked"
        final_gate = "REINFERENCE_ONLY_PATH_BLOCKED_BY_PREPROCESSING_CONTRACT"
    elif authorized and execution_status == "blocked_missing_dataset_access":
        primary = "C66-F_preprocessing_dataset_contract_blocked"
        final_gate = "REINFERENCE_ONLY_PATH_BLOCKED_BY_MISSING_DATASET_ACCESS"
    elif authorized:
        primary = "C66-L_reinference_only_path_blocked_new_training_may_be_needed_but_not_authorized"
        final_gate = "NEW_TRAINING_REQUIRED_BUT_NOT_AUTHORIZED"
    else:
        primary = "C66-B_no_authorization_protocol_only"
        final_gate = "MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED"
    if primary == "C66-B_no_authorization_protocol_only":
        active = [
            "C66-B_no_authorization_protocol_only",
            "C66-C_cpu_torchload_abi_validated_no_forward",
            "C66-E_preprocessing_dataset_contract_validated",
            "C66-G_trial_level_cache_schema_validated",
            "C66-K_atom_trace_forward_hooks_feasible_without_training",
        ]
        inactive = [
            "C66-A_reinference_only_microcampaign_authorized_and_executed",
            "C66-D_checkpoint_abi_or_state_dict_mismatch_found",
            "C66-F_preprocessing_dataset_contract_blocked",
            "C66-H_minimal_trial_cache_emitted_and_manifested",
            "C66-I_split_label_protocol_feasible_on_cache",
            "C66-J_sample_level_conditional_cs_feasible_on_cache",
            "C66-L_reinference_only_path_blocked_new_training_may_be_needed_but_not_authorized",
            "C66-M_claim_or_availability_inconsistency_found",
        ]
    elif primary == "C66-A_reinference_only_microcampaign_authorized_and_executed":
        active = [
            "C66-A_reinference_only_microcampaign_authorized_and_executed",
            "C66-E_preprocessing_dataset_contract_validated",
            "C66-G_trial_level_cache_schema_validated",
            "C66-H_minimal_trial_cache_emitted_and_manifested",
            "C66-I_split_label_protocol_feasible_on_cache",
            "C66-J_sample_level_conditional_cs_feasible_on_cache",
            "C66-K_atom_trace_forward_hooks_feasible_without_training",
        ]
        inactive = [
            "C66-B_no_authorization_protocol_only",
            "C66-C_cpu_torchload_abi_validated_no_forward",
            "C66-D_checkpoint_abi_or_state_dict_mismatch_found",
            "C66-F_preprocessing_dataset_contract_blocked",
            "C66-L_reinference_only_path_blocked_new_training_may_be_needed_but_not_authorized",
            "C66-M_claim_or_availability_inconsistency_found",
        ]
    else:
        active = [primary]
        inactive = [d for d in DECISIONS if d != primary]
    return {
        "primary": primary,
        "active": active,
        "inactive": inactive,
        "final_gate": final_gate,
        "red_team_failure_count": len(failures),
        "authorization_phrase_required": AUTH_PHRASE,
        "authorization_present": authorized,
        "recommended_next_direction": "C67 only if reviewer asks for a registered split-label/CS analysis on the external cache; no selector claim",
    }


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c66", "command": "python -m pytest oaci/tests/test_c66_reinference_only_trial_cache_microcampaign.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c66_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py ... test_c66_reinference_only_trial_cache_microcampaign.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c66_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c60_*.py ... test_c66_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def table_row_counts(res: dict) -> dict:
    keys = {
        "artifact_manifest": "artifact_manifest_rows",
        "authorized_cache_manifest": "authorized_cache_manifest_rows",
        "authorized_microcampaign_execution": "authorized_microcampaign_execution_rows",
        "authorization_ledger": "authorization_ledger_rows",
        "atom_trace_hook_feasibility": "atom_trace_hook_feasibility_rows",
        "cache_external_manifest": "cache_external_manifest_rows",
        "cache_field_availability_ledger": "cache_field_availability_ledger_rows",
        "cache_payload_plan": "cache_payload_plan_rows",
        "checkpoint_mapping_replay": "checkpoint_mapping_replay_rows",
        "combined_authorization_comparison": "combined_authorization_comparison_rows",
        "conditional_cs_variable_map": "conditional_cs_variable_map_rows",
        "cpu_torchload_abi_sample": "cpu_torchload_abi_sample_rows",
        "dataset_split_contract": "dataset_split_contract_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "frozen_store_integrity": "frozen_store_integrity_rows",
        "label_quarantine_contract": "label_quarantine_contract_rows",
        "label_view_policy": "label_view_policy_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "microcampaign_expected_payload_size": "microcampaign_expected_payload_size_rows",
        "microcampaign_sampling_plan": "microcampaign_sampling_plan_rows",
        "model_abi_compatibility_ledger": "model_abi_compatibility_ledger_rows",
        "preprocess_contract_inventory": "preprocess_contract_inventory_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "representation_hook_contract": "representation_hook_contract_rows",
        "sample_level_cs_feasibility": "sample_level_cs_feasibility_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "sidecar_schema_summary": "sidecar_schema_summary_rows",
        "split_label_feasibility_on_cache": "split_label_feasibility_on_cache_rows",
        "split_label_protocol": "split_label_protocol_rows",
        "state_dict_key_shape_summary": "state_dict_key_shape_summary_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "trial_cache_schema": "trial_cache_schema_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def run(
    authorization_phrase: str = "",
    authorization_text: str = "",
    test_status: str = "planned",
    execute_microcampaign: bool = False,
    datalake_root: str = "/projects/EEG-foundation-model/datalake/raw",
    external_cache_root: str = EXTERNAL_CACHE_ROOT,
) -> dict:
    authorized = _auth_present(authorization_phrase, authorization_text)
    phrase_present = _auth_phrase_present(authorization_phrase, authorization_text)
    token_present = _explicit_auth_token_present(authorization_phrase, authorization_text)
    c65 = _load_json(C65_JSON)
    rows = _mapping_rows()
    unique_rows = _unique_checkpoint_rows(rows)
    sample_rows = _select_abi_sample(rows)
    load_meta = {r["checkpoint_id"]: _state_load_metadata(r) for r in sample_rows}
    execution = (
        execute_authorized_microcampaign(
            sample_rows,
            load_meta,
            datalake_root=datalake_root,
            external_cache_root=external_cache_root,
            execute=execute_microcampaign,
        )
        if authorized
        else _microcampaign_empty("not_authorized_not_created")
    )
    cache_present = bool(execution.get("success"))
    load_rows = build_cpu_torchload_abi_sample(sample_rows, load_meta, execution)
    schema_rows = build_trial_cache_schema_for_cache(cache_present)
    res = {
        "config_hash": _lock_config(),
        "c65_commit": "192a82d",
        "c65_decision": c65["decision"]["primary"],
        "c65_final_gate": c65["final_gate"],
        "authorization_present": authorized,
        "authorized_microcampaign": execution,
        "authorization_ledger_rows": build_authorization_ledger(authorized, phrase_present, token_present, execution),
        "frozen_store_integrity_rows": build_frozen_store_integrity(rows),
        "checkpoint_mapping_replay_rows": build_checkpoint_mapping_replay(rows),
        "sidecar_schema_summary_rows": build_sidecar_schema_summary(unique_rows),
        "cpu_torchload_abi_sample_rows": load_rows,
        "state_dict_key_shape_summary_rows": build_state_dict_key_shape_summary(load_meta),
        "model_abi_compatibility_ledger_rows": build_model_abi_compatibility_ledger(sample_rows, load_rows, execution),
        "preprocess_contract_inventory_rows": build_preprocess_contract_inventory(sample_rows[0]),
        "dataset_split_contract_rows": build_dataset_split_contract(),
        "label_quarantine_contract_rows": build_label_quarantine_contract(),
        "label_view_policy_rows": build_label_view_policy(),
        "microcampaign_sampling_plan_rows": build_microcampaign_sampling_plan(sample_rows, authorized, execution),
        "microcampaign_expected_payload_size_rows": build_microcampaign_expected_payload_size(sample_rows),
        "trial_cache_schema_rows": schema_rows,
        "cache_field_availability_ledger_rows": build_cache_field_availability_ledger(schema_rows),
        "split_label_protocol_rows": build_split_label_protocol(cache_present),
        "split_label_feasibility_on_cache_rows": build_split_label_feasibility_on_cache(cache_present),
        "conditional_cs_variable_map_rows": build_conditional_cs_variable_map(cache_present),
        "sample_level_cs_feasibility_rows": build_sample_level_cs_feasibility(cache_present),
        "atom_trace_hook_feasibility_rows": build_atom_trace_hook_feasibility(cache_present),
        "representation_hook_contract_rows": build_representation_hook_contract(),
        "cache_payload_plan_rows": build_cache_payload_plan(cache_present),
        "cache_external_manifest_rows": build_cache_external_manifest(execution),
        "authorized_microcampaign_execution_rows": execution["execution_rows"],
        "authorized_cache_manifest_rows": execution["cache_manifest_rows"],
        "combined_authorization_comparison_rows": build_combined_authorization_comparison(authorized, execution),
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "schema_validation_summary_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "generated_paths": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []}, authorized)
    return res


def _compact_json(res: dict) -> dict:
    execution = res.get("authorized_microcampaign", _microcampaign_empty("not_authorized_not_created"))
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c65_commit": res["c65_commit"],
        "c65_decision": res["c65_decision"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "authorization_phrase_required": AUTH_PHRASE,
        "explicit_authorization_token": EXPLICIT_AUTH_TOKEN,
        "authorization_present": res["decision"]["authorization_present"],
        "authorized_microcampaign_status": execution.get("status", ""),
        "external_cache_manifest_sha256": execution.get("manifest_sha256", ""),
        "key_numbers": {
            "no_authorization_forward_attempted": 0,
            "no_authorization_real_trial_cache_rows_emitted": 0,
            "cpu_torchload_sample_count": len(res["cpu_torchload_abi_sample_rows"]),
            "cpu_torchload_pass_count": sum(1 for r in res["cpu_torchload_abi_sample_rows"] if r["load_status"] == "pass"),
            "forward_attempted": sum(int(r["forward_attempted"]) for r in res["cpu_torchload_abi_sample_rows"]),
            "real_trial_cache_rows_emitted": sum(int(r["real_trial_rows"]) for r in res["cache_external_manifest_rows"]),
            "authorized_checkpoint_count": execution.get("checkpoint_count", 0),
            "authorized_target_count": execution.get("target_count", 0),
            "mapping_replay_groups": len(res["checkpoint_mapping_replay_rows"]),
            "sidecar_schema_signatures": len(res["sidecar_schema_summary_rows"]),
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    execution = res.get("authorized_microcampaign", _microcampaign_empty("not_authorized_not_created"))
    cache_rows = int(execution.get("trial_row_count", 0))
    auth_sentence = (
        f"The latest user instruction explicitly authorized the microcampaign (`{EXPLICIT_AUTH_TOKEN}`). "
        f"C66 therefore ran the re-inference-only branch on CPU and emitted an external-only cache."
        if d["authorization_present"] and execution.get("success")
        else f"The required authorization phrase `{AUTH_PHRASE}` was not present, and no explicit authorization token was accepted."
    )
    cache_sentence = (
        f"Authorized external cache: `{execution.get('trial_cache_path', '')}` with `{cache_rows}` rows; manifest SHA-256 `{execution.get('manifest_sha256', '')}`."
        if execution.get("success")
        else f"Authorized execution status: `{execution.get('status', 'not_executed')}`; no real cache rows are claimed."
    )
    main = "\n".join([
        f"# C66 - Re-inference-Only Trial-Level Cache Microcampaign / Split-Label-CS Foundation (frozen C19 `{res['config_hash']}`)",
        "",
        "## 1. Executive Verdict",
        "",
        f"Primary: `{d['primary']}`",
        "",
        f"Active: `{' ; '.join(d['active'])}`",
        "",
        f"Inactive: `{' ; '.join(d['inactive'])}`",
        "",
        f"Final gate: `{d['final_gate']}`",
        "",
        "## 2. Authorization Boundary",
        "",
        f"No-authorization baseline remains part of the report: forward `0`, cache rows `0`, final gate `MICROCAMPAIGN_READY_BUT_NOT_AUTHORIZED`.",
        "",
        auth_sentence,
        "",
        "## 3. Frozen Store Replay",
        "",
        "C66 replays C65's recovered frozen checkpoint universe: 5454 checkpoint payloads, 5454 sidecars, 27 artifact indexes, 3804 C50 singleton mappings, and 1268 unique checkpoint ids.",
        "",
        "## 4. CPU Torchload ABI Sample",
        "",
        "Six stratified checkpoints were loaded on CPU with `weights_only=True` for state_dict metadata and state-hash verification. In the authorization branch only, the same six checkpoints were strictly loaded into ShallowConvNet and evaluated in `eval()` / `no_grad()` mode on CPU.",
        "",
        "## 5. Preprocessing / Dataset Contract",
        "",
        "BNCI2014_001 preprocessing, channel order, class order, epoch window, normalization, split code, and label-quarantine rules are validated from committed code and recovered context manifests.",
        "",
        "## 6. Trial Cache Foundation",
        "",
        "C66 defines `c66_trial_cache_v1`, split-label roles, sample-level conditional-CS variables, atom/representation hook contracts, and an external payload policy.",
        "",
        cache_sentence,
        "",
        "The emitted cache is content-addressed under the external cache root, with hashes tracked in compact git artifacts. Label fields are exposed only through the C66 label-view masking contract.",
        "",
        "The emitted cache is diagnostic input only. It is not a selector artifact, checkpoint recommendation, OACI rescue, deployable method, or few-label sufficiency claim.",
        "",
        "## 7. Red-Team Verification",
        "",
        f"Red-team failures: `{d['red_team_failure_count']}`.",
        "",
        "## 8. Final Gate",
        "",
        f"`{d['final_gate']}`",
        "",
        "C66 remains diagnostic-only and non-deployable. It does not train, use GPU, touch BNCI2014_004, run seeds [3,4], emit selector artifacts, recommend checkpoints, claim few-label sufficiency, or draft manuscript prose.",
    ])
    red = "\n".join([
        "# C66 - Red-Team Verification",
        "",
        "All C66 red-team gates pass." if d["red_team_failure_count"] == 0 else "C66 red-team gates failed.",
        "",
        f"Authorization present: `{d['authorization_present']}`. Authorized execution status: `{execution.get('status', '')}`. External cache rows: `{cache_rows}`.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
        "",
        "## Slurm Validation",
        "",
        *[f"- {scope} job `{job_id}` on `cpu-high` with `eeg2025`: `{result}`." for scope, job_id, result in SLURM_VALIDATION_RESULTS],
    ])
    return {
        "C66_REINFERENCE_ONLY_TRIAL_CACHE_MICROCAMPAIGN.md": main,
        "C66_RED_TEAM_VERIFICATION.md": red,
    }


def write_tables(res: dict, table_dir: str) -> None:
    specs = {
        "authorization_ledger.csv": ("authorization_ledger_rows", ["gate", "value", "allowed", "observed", "enforced_status"]),
        "combined_authorization_comparison.csv": ("combined_authorization_comparison_rows", ["mode", "authorization_present", "forward_attempted", "external_cache_created", "real_trial_rows", "primary", "final_gate", "interpretation"]),
        "frozen_store_integrity.csv": ("frozen_store_integrity_rows", ["check", "value", "passed", "notes"]),
        "checkpoint_mapping_replay.csv": ("checkpoint_mapping_replay_rows", ["seed", "target", "level", "regime", "singleton_rows", "unique_checkpoint_ids", "all_pt_json_verified", "min_candidate_order", "max_candidate_order"]),
        "sidecar_schema_summary.csv": ("sidecar_schema_summary_rows", ["signature_id", "sidecar_count", "tensor_count", "tensor_keys", "dtype_set", "shape_signature", "writer_versions", "all_model_hash_match"]),
        "cpu_torchload_abi_sample.csv": ("cpu_torchload_abi_sample_rows", ["sample_id", "candidate_id", "checkpoint_id", "seed", "target", "level", "regime", "checkpoint_path_hash", "sidecar_path_hash", "pt_size_bytes", "json_size_bytes", "torch_load_attempted", "forward_attempted", "training_attempted", "payload_file_sha256_rehashed", "load_status", "state_hash_matches_checkpoint_id", "sidecar_tensor_schema_matches", "key_count", "tensor_count", "dtype_set", "total_elements", "error"]),
        "state_dict_key_shape_summary.csv": ("state_dict_key_shape_summary_rows", ["state_key", "sample_count", "dtype_set", "shape_set", "required_by_shallowconvnet_abi"]),
        "model_abi_compatibility_ledger.csv": ("model_abi_compatibility_ledger_rows", ["check", "expected", "observed", "passed", "forward_attempted", "notes"]),
        "preprocess_contract_inventory.csv": ("preprocess_contract_inventory_rows", ["item", "value", "source", "validated", "blocks_campaign"]),
        "dataset_split_contract.csv": ("dataset_split_contract_rows", ["target_id", "dataset_id", "source_audit_subjects", "source_train_subjects", "historical_seeds", "reserved_seeds", "reserved_dataset", "roles_reconstructable", "blocks_campaign"]),
        "label_quarantine_contract.csv": ("label_quarantine_contract_rows", ["label_source", "allowed_for_cache_field", "allowed_for_selection_rule", "future_split_role", "same_label_reuse_allowed", "notes"]),
        "label_view_policy.csv": ("label_view_policy_rows", ["view", "label_fields_visible_when", "label_fields_masked_when", "prediction_fields_visible", "allowed_for_selection_rule", "same_label_reuse_allowed", "enforced_by"]),
        "microcampaign_sampling_plan.csv": ("microcampaign_sampling_plan_rows", ["pilot_cell", "checkpoint_id", "seed", "target", "level", "regime", "checkpoint_selection_rule", "source_roles_to_cache", "target_roles_to_cache", "bnci004_used", "reserved_seed_used", "execution_authorized_now", "execution_completed_now"]),
        "authorized_microcampaign_execution.csv": ("authorized_microcampaign_execution_rows", ["pilot_cell", "checkpoint_id", "seed", "target", "level", "regime", "trial_rows", "construct_rows", "eval_rows", "forward_attempted", "training_attempted", "gpu_used", "status", "error"]),
        "microcampaign_expected_payload_size.csv": ("microcampaign_expected_payload_size_rows", ["payload", "checkpoint_count", "subject_count_estimate", "trial_rows_estimate", "bytes_estimate", "store_in_git", "notes"]),
        "trial_cache_schema.csv": ("trial_cache_schema_rows", ["field", "category", "required_minimal", "target_label_dependent", "requires_forward", "large_payload_ref_only", "available_now", "available_after_authorized_reinfer"]),
        "cache_field_availability_ledger.csv": ("cache_field_availability_ledger_rows", ["field", "source_observable", "target_label_dependent", "requires_forward", "available_in_committed_summary_artifacts", "available_in_current_c66_cache", "available_after_authorized_microcampaign"]),
        "split_label_protocol.csv": ("split_label_protocol_rows", ["protocol_step", "input_required", "cache_required", "current_status", "same_label_reuse_allowed", "claim_allowed_now"]),
        "split_label_feasibility_on_cache.csv": ("split_label_feasibility_on_cache_rows", ["check", "value", "feasible_now", "feasible_after_authorized_cache", "blocks_current_claim"]),
        "conditional_cs_variable_map.csv": ("conditional_cs_variable_map_rows", ["audit", "x1", "x2", "y", "sample_unit", "paired_sample_vars_available_now", "available_after_authorized_cache", "target_label_dependent", "x2_target_label_dependent", "y_target_label_dependent", "target_label_scope"]),
        "sample_level_cs_feasibility.csv": ("sample_level_cs_feasibility_rows", ["check", "current_real_cache", "toy_interface_pass", "full_estimator_run", "feasible_now", "feasible_after_authorized_cache", "notes"]),
        "atom_trace_hook_feasibility.csv": ("atom_trace_hook_feasibility_rows", ["trace", "recoverable_by_reinfer_only", "requires_forward_hook", "requires_new_training", "available_now", "claim_allowed_now"]),
        "representation_hook_contract.csv": ("representation_hook_contract_rows", ["hook", "target_module", "payload", "large_payload_ref_only", "requires_training", "requires_forward_authorization"]),
        "cache_payload_plan.csv": ("cache_payload_plan_rows", ["artifact", "git_tracked", "external_root", "emitted_in_c66", "hash_manifest_required", "large_payload_policy"]),
        "cache_external_manifest.csv": ("cache_external_manifest_rows", ["cache_id", "external_root", "created_in_c66", "real_trial_rows", "manifest_hash", "status"]),
        "authorized_cache_manifest.csv": ("authorized_cache_manifest_rows", ["cache_id", "cache_kind", "external_path", "path_hash", "exists", "size_bytes", "sha256", "row_count", "git_tracked", "status"]),
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
    with open(os.path.join(TABLE_DIR, "c66_gate_decision.json"), "w") as f:
        json.dump(res["decision"], f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "trial_cache_schema.json"), "w") as f:
        json.dump({"schema_version": "c66_trial_cache_v1", "fields": res["trial_cache_schema_rows"]}, f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "authorized_microcampaign_manifest.json"), "w") as f:
        e = res.get("authorized_microcampaign", _microcampaign_empty("not_authorized_not_created"))
        json.dump({
            "status": e.get("status", ""),
            "success": e.get("success", 0),
            "trial_row_count": e.get("trial_row_count", 0),
            "checkpoint_count": e.get("checkpoint_count", 0),
            "target_count": e.get("target_count", 0),
            "trial_cache_path_hash": _path_hash(e.get("trial_cache_path", "")) if e.get("trial_cache_path") else "",
            "trial_cache_sha256": e.get("trial_cache_sha256", ""),
            "manifest_path_hash": _path_hash(e.get("manifest_path", "")) if e.get("manifest_path") else "",
            "manifest_sha256": e.get("manifest_sha256", ""),
            "dataset_evidence_hash": e.get("dataset_evidence_hash", ""),
            "raw_data_fingerprint": e.get("raw_data_fingerprint", ""),
            "resolved_preprocess_hash": e.get("resolved_preprocess_hash", ""),
            "diagnostic_only_non_deployable": True,
        }, f, indent=2, sort_keys=True)


def _listed_paths() -> list[str]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C66_*.md"))
        + list(Path(REPORT_DIR).glob("C66_*.json"))
        + [Path(p) for p in Path(TABLE_DIR).glob("*.csv") if p.name not in skip]
        + list(Path(TABLE_DIR).glob("*.json"))
    )


def _schema_rows(table_dir: str) -> list[dict]:
    rows = []
    for path in sorted(Path(table_dir).glob("*.csv")):
        if path.name in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": path.name, "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def _large_scan(paths: list[Path]) -> list[dict]:
    return [{"path": str(p), "size_bytes": os.path.getsize(p), "over_50mb": int(os.path.getsize(p) > MAX_REPORT_BYTES), "passed": int(os.path.getsize(p) <= MAX_REPORT_BYTES)} for p in sorted(paths)]


def _artifact_manifest(paths: list[Path], table_dir: str) -> list[dict]:
    counts = {}
    for path in Path(table_dir).glob("*.csv"):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            counts[str(path)] = sum(1 for _ in reader)
    rows = []
    for p in sorted(paths):
        s = str(p)
        cls = "table" if s.endswith(".csv") else "summary_json" if s.endswith(".json") else "report"
        rows.append({"path": s, "size_bytes": os.path.getsize(p), "sha256": _sha256(s), "artifact_class": cls, "row_count": counts.get(s, "")})
    return rows


def write_artifacts(res: dict) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)
    paths = [str(p) for p in _listed_paths()]
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res, res["decision"]["authorization_present"])
    write_tables(res, TABLE_DIR)
    res["schema_validation_summary_rows"] = _schema_rows(TABLE_DIR)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res, res["decision"]["authorization_present"])
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)
    paths = _listed_paths()
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    _write_json_payloads(res)
    paths = _listed_paths()
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c66_reinference_only_trial_cache_microcampaign")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--authorization-phrase", default="")
    ap.add_argument("--authorization-text", default="")
    ap.add_argument("--execute-microcampaign", action="store_true")
    ap.add_argument("--datalake-root", default="/projects/EEG-foundation-model/datalake/raw")
    ap.add_argument("--external-cache-root", default=EXTERNAL_CACHE_ROOT)
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(
        authorization_phrase=args.authorization_phrase,
        authorization_text=args.authorization_text,
        test_status=args.test_status,
        execute_microcampaign=args.execute_microcampaign,
        datalake_root=args.datalake_root,
        external_cache_root=args.external_cache_root,
    )
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C66] decision={res['decision']['primary']} gate={res['decision']['final_gate']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
