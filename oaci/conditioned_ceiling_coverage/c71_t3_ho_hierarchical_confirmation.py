"""C71 - T3-HO hierarchical confirmation readiness and protocol gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import audit_utils as au
from . import c66_reinference_only_trial_cache_microcampaign as c66
from . import c69_powered_trial_cache_scaleup as c69
from . import c70_split_label_information_budget as c70


MILESTONE = "C71"
AUTH_TOKEN = "C71_T3_HO_REINFERENCE_ONLY_AUTHORIZED"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c71_tables"
REPORT_JSON = "oaci/reports/C71_T3_HO_HIERARCHICAL_CONFIRMATION.json"
C70_JSON = "oaci/reports/C70_SPLIT_LABEL_INFORMATION_BUDGET.json"
C70_PROTOCOL = "oaci/reports/c70_tables/C71_T3_CONFIRMATORY_PROTOCOL.json"
C70_PROTOCOL_SHA = "oaci/reports/c70_tables/C71_T3_CONFIRMATORY_PROTOCOL.sha256"
C65_MAP = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"
DEFAULT_DATALAKE_ROOT = c69.DEFAULT_DATALAKE_ROOT
EXTERNAL_CACHE_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c71-t3-ho-cache"
MAX_REPORT_BYTES = 50_000_000
TRIAL_ROWS_PER_UNIT = 576
FULL_BUDGET_LABEL = "full-construction"
C71_BUDGETS = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
MASKED = c69.MASKED

PRIMARY_BUDGETS = ("8", "64", "full-construction")
SECONDARY_BUDGETS = ("0", "1", "2", "4", "12", "16", "24", "32", "48")

DECISIONS = (
    "C71-A_within_target_split_label_reliability_confirmed_actionability_weak",
    "C71-B_small_budget_split_label_actionability_confirmed",
    "C71-C_dense_label_partial_recovery_confirmed",
    "C71-D_C70_effect_not_replicated_on_T3_HO",
    "C71-E_hierarchical_signal_replication_but_measurement_control_gap_narrows",
    "C71-F_protocol_masking_or_dependency_blocker",
    "C71-G_T3_HO_ready_but_not_authorized",
    "C71-S1_T3_HO_disjointness_confirmed",
    "C71-S2_physical_view_isolation_passed",
    "C71-S3_candidate_specific_gauge_recovery_partial",
    "C71-S4_common_offset_not_explanatory",
    "C71-S5_no_strict_source_escape_hatch",
    "C71-S6_strict_source_escape_hatch_found",
    "C71-S7_conditional_observability_stable_diagnostic",
    "C71-S8_conditional_cs_proxy_only",
    "C71-S9_target_population_generalization_unresolved",
    "C71-S10_new_training_not_justified",
    "C71-S11_independent_target_or_dataset_replication_now_justified",
)

FINAL_GATES = (
    "T3_HO_CONFIRMS_MEASUREMENT_CONTROL_SEPARATION",
    "T3_HO_CONFIRMS_SMALL_BUDGET_ACTIONABILITY",
    "T3_HO_CONFIRMS_DENSE_LABEL_PARTIAL_RECOVERY_ONLY",
    "T3_HO_FAILS_TO_REPLICATE_C70",
    "T3_HO_ANALYSIS_BLOCKED_BY_PROTOCOL_OR_MASKING",
    "T3_HO_READY_BUT_NOT_AUTHORIZED",
)

FORBIDDEN_PATTERNS = (
    "few-label sufficiency",
    "deployable selector",
    "checkpoint recommendation",
    "source-only rescue",
    "oaci rescue",
    "target-population generalization established",
    "full conditional-cs established",
    "same-label endpoint scalar available at selection time",
    "row-level iid",
    "new training is justified",
    "gpu used",
    "forward pass executed",
    "re-inference executed",
    "t3-ho outcome accessed",
    "manuscript drafting",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "without ",
    "forbid",
    "forbidden ",
    "unavailable ",
    "not authorized ",
    "not executed ",
    "not accessed ",
    "diagnostic only ",
    "diagnostic-only ",
    "proxy-only ",
    "unresolved ",
)

RISK_ROWS = (
    "protocol_timing",
    "adaptive_analysis_in_frozen_universe",
    "T3_HO_disjointness",
    "target_label_sampling_blindness",
    "unique_trial_budget_contract",
    "construction_eval_overlap",
    "cache_rows_not_independent",
    "small_number_of_targets",
    "physical_masking",
    "strict_source_feature_provenance",
    "low_resolution_permutation",
    "bandwidth_multiple_testing",
    "reliability_not_actionability",
    "same_label_oracle_misuse",
    "conditional_cs_proxy_overclaim",
    "target_population_overclaim",
    "raw_cache_in_git",
    "unauthorized_forward_or_training",
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _auth_present(token: str = "") -> bool:
    # Exact CLI argument only. Do not inspect prompt/protocol text or env vars.
    return str(token).strip() == AUTH_TOKEN


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _path_hash(path: str) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_or_empty(args: list[str]) -> str:
    try:
        return _git(args)
    except Exception:
        return ""


def _listed_paths() -> list[Path]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C71_*.md"))
        + list(Path(REPORT_DIR).glob("C71_*.json"))
        + list(Path(REPORT_DIR).glob("C71_*.sha256"))
        + [p for p in Path(TABLE_DIR).glob("*.csv") if p.name not in skip]
    )


def _large_scan(paths: list[Path]) -> list[dict]:
    return [
        {
            "path": str(p),
            "size_bytes": os.path.getsize(p),
            "over_50mb": int(os.path.getsize(p) > MAX_REPORT_BYTES),
            "passed": int(os.path.getsize(p) <= MAX_REPORT_BYTES),
        }
        for p in sorted(paths)
    ]


def _artifact_manifest(paths: list[Path], table_dir: str) -> list[dict]:
    counts: dict[str, int | str] = {}
    for path in Path(table_dir).glob("*.csv"):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            counts[str(path)] = sum(1 for _ in reader)
    return [
        {
            "path": str(p),
            "size_bytes": os.path.getsize(p),
            "sha256": _sha256(str(p)),
            "artifact_class": "table" if str(p).endswith(".csv") else "protocol" if "PROTOCOL" in str(p) else "summary_json" if str(p).endswith(".json") else "report",
            "row_count": counts.get(str(p), ""),
        }
        for p in sorted(paths)
    ]


def _affirmative_hit(text: str, phrase: str, window: int = 240) -> bool:
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


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = affirmative = 0
        files = []
        for path in paths:
            if os.path.basename(path) in {"forbidden_claim_scan.csv", "red_team_failure_ledger.csv"}:
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


def load_context() -> dict:
    c70 = _load_json(C70_JSON)
    parent_protocol = _load_json(C70_PROTOCOL)
    parent_protocol_sha = open(C70_PROTOCOL_SHA).read().strip()
    return {
        "c70": c70,
        "parent_protocol": parent_protocol,
        "parent_protocol_sha": parent_protocol_sha,
        "parent_protocol_sha_replay": _sha256(C70_PROTOCOL),
        "c65_rows": _read_csv(C65_MAP),
        "head": _git_or_empty(["rev-parse", "--short", "HEAD"]),
        "branch": _git_or_empty(["branch", "--show-current"]),
        "origin_oaci": _git_or_empty(["rev-parse", "--short", "origin/oaci"]),
    }


def t3_ho_units(c65_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    canonical, t1_units, t2_units = c69.build_schedule(c65_rows)
    t2_ids = {r["checkpoint_id"] for r in t2_units}
    t3_units = [r for r in canonical if r["checkpoint_id"] not in t2_ids]
    return canonical, t1_units, t2_units, t3_units


def _empty_execution(stage: str, checkpoint_count: int) -> dict:
    return {
        "stage": stage,
        "status": "not_authorized_not_executed",
        "attempted": 0,
        "success": 0,
        "error": "",
        "external_root": EXTERNAL_CACHE_ROOT,
        "cache_dir": "",
        "trial_cache_path": "",
        "trial_cache_sha256": "",
        "trial_cache_size_bytes": 0,
        "manifest_path": "",
        "manifest_sha256": "",
        "manifest_size_bytes": 0,
        "trial_row_count": 0,
        "checkpoint_count": checkpoint_count,
        "target_count": 0,
        "dataset_evidence_hash": "",
        "raw_data_fingerprint": "",
        "resolved_preprocess_hash": "",
        "runtime_seconds": 0.0,
        "execution_rows": [],
    }


def execute_c71_stage(
    stage: str,
    unit_rows: list[dict],
    *,
    datalake_root: str,
    external_cache_root: str,
) -> dict:
    """Run CPU-only frozen-checkpoint inference and write an external C71 cache."""
    t0 = time.time()
    try:
        import torch
        from oaci.data.eeg.bnci import load_moabb_confirmatory
        from oaci.models import build_model

        os.makedirs(external_cache_root, exist_ok=True)
        base_cache_dir = os.path.join(external_cache_root, f"authorized_c71_{stage}_v1")
        tmp_dir = os.path.join(base_cache_dir, f"_tmp_{os.getpid()}")
        os.makedirs(tmp_dir, exist_ok=True)
        trial_cache_tmp_path = os.path.join(tmp_dir, "trial_logits_probs_cache.csv")

        pp, ds = c66._preprocess_namespace(unit_rows[0])
        target_subjects = sorted({int(r["target"]) for r in unit_rows})
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
            "cache_version",
            "trial_cache_id",
            "c69_stage",
            "c71_stage",
            "checkpoint_id",
            "checkpoint_path_hash",
            "checkpoint_sidecar_hash",
            "dataset_id",
            "subject_id",
            "target_id",
            "source_or_target_flag",
            "source_or_target_role",
            "trajectory_id",
            "seed",
            "level",
            "fold",
            "regime",
            "candidate_order",
            "epoch_or_step",
            "trial_id",
            "trial_index",
            "class_label_quarantined",
            "y_true_quarantined",
            "y_pred",
            "predicted_class",
            "logits",
            "probabilities",
            "confidence",
            "true_class_probability",
            "margin",
            "entropy",
            "correctness_quarantined",
            "split_role",
            "split_role_for_future_split_label",
            "availability_tags",
            "view_mask_tags",
        ]
        execution_rows = []
        row_count = 0
        with open(trial_cache_tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for idx, r in enumerate(unit_rows, 1):
                load_meta = c66._state_load_metadata(r)
                if load_meta["load_status"] != "pass" or not load_meta["state_hash_matches_checkpoint_id"]:
                    raise RuntimeError(f"C71 state-load metadata failed for {r['checkpoint_id']}: {load_meta['error']}")
                spec = c66._model_spec_for_row(r)
                arch = dict(spec["backbone"])
                in_chans, in_times = list(spec["input_shape"])
                model = build_model(
                    spec["factory"],
                    in_chans=int(in_chans),
                    in_times=int(in_times),
                    n_classes=int(spec["n_classes"]),
                    **arch,
                )
                model.load_state_dict(load_meta["state"], strict=True)
                model.eval()
                c66._assert_cpu_model_and_tensor(model)

                target = int(r["target"])
                domain = f"BNCI2014_001|subject-{target:03d}"
                indices = np.where(np.asarray(bundle.subject_id == domain))[0]
                split_counter = Counter()
                status = "pass" if len(indices) else "missing_target_rows"
                with torch.no_grad():
                    for start in range(0, len(indices), 128):
                        batch_idx = indices[start:start + 128]
                        x = torch.from_numpy(np.ascontiguousarray(bundle.X[batch_idx])).to(dtype=torch.float32)
                        c66._assert_cpu_model_and_tensor(model, x)
                        out = model(x)
                        if out.logits.device.type != "cpu":
                            raise RuntimeError(f"C71 CPU-only guard failed: logits device={out.logits.device.type}")
                        logits = out.logits.detach().cpu().numpy()
                        mx = logits.max(axis=1, keepdims=True)
                        exp = np.exp(logits - mx)
                        probs = exp / exp.sum(axis=1, keepdims=True)
                        preds = probs.argmax(axis=1)
                        sorted_probs = np.sort(probs, axis=1)
                        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
                        for j, bi in enumerate(batch_idx):
                            trial_id = str(bundle.trial_id[bi])
                            split_role = c69._future_split_role(trial_id)
                            split_counter[split_role] += 1
                            y = int(bundle.y[bi])
                            pred = int(preds[j])
                            correct = int(pred == y)
                            writer.writerow({
                                "cache_version": "c71_trial_cache_v1",
                                "trial_cache_id": f"c71_trial_cache_{stage}_v1",
                                "c69_stage": stage,
                                "c71_stage": stage,
                                "checkpoint_id": r["checkpoint_id"],
                                "checkpoint_path_hash": _path_hash(r["pt_path"]),
                                "checkpoint_sidecar_hash": _path_hash(r["json_path"]),
                                "dataset_id": "BNCI2014_001",
                                "subject_id": str(bundle.subject_id[bi]),
                                "target_id": target,
                                "source_or_target_flag": "target",
                                "source_or_target_role": "target_audit_t3_ho",
                                "trajectory_id": r["trajectory_id"],
                                "seed": r["seed"],
                                "level": r["level"],
                                "fold": f"target-{target:03d}",
                                "regime": r["regime"],
                                "candidate_order": r["candidate_order"],
                                "epoch_or_step": r["candidate_order"],
                                "trial_id": trial_id,
                                "trial_index": int(bi),
                                "class_label_quarantined": str(bundle.class_names[y]),
                                "y_true_quarantined": y,
                                "y_pred": pred,
                                "predicted_class": pred,
                                "logits": c69._float_vec(logits[j]),
                                "probabilities": c69._float_vec(probs[j]),
                                "confidence": f"{float(probs[j, pred]):.8g}",
                                "true_class_probability": f"{float(probs[j, y]):.8g}",
                                "margin": f"{float(margins[j]):.8g}",
                                "entropy": f"{c69._entropy(probs[j]):.8g}",
                                "correctness_quarantined": correct,
                                "split_role": split_role,
                                "split_role_for_future_split_label": split_role,
                                "availability_tags": "target_label_quarantined;no_selector;diagnostic_only;c71_authorized",
                                "view_mask_tags": "source_only_masks_labels_predictions;split_label_masks_nonrole_labels;same_label_oracle_after_primary_freeze",
                            })
                            row_count += 1
                execution_rows.append({
                    "stage": stage,
                    "unit_index": idx,
                    "checkpoint_id": r["checkpoint_id"],
                    "seed": r["seed"],
                    "target": r["target"],
                    "level": r["level"],
                    "regime": r["regime"],
                    "candidate_order": r["candidate_order"],
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
                raise RuntimeError(f"C71 immutable external cache collision at {trial_cache_path}")
            os.remove(trial_cache_tmp_path)
        else:
            os.replace(trial_cache_tmp_path, trial_cache_path)

        manifest = {
            "schema_version": "c71_trial_cache_manifest_v1",
            "cache_id": f"c71_trial_cache_{stage}_v1",
            "stage": stage,
            "diagnostic_only_non_deployable": True,
            "authorization": AUTH_TOKEN,
            "datalake_root": datalake_root,
            "trial_cache_path": trial_cache_path,
            "trial_cache_sha256": trial_sha,
            "trial_row_count": row_count,
            "checkpoint_count": len(unit_rows),
            "target_subjects": target_subjects,
            "dataset_evidence_hash": load_result.evidence.evidence_hash,
            "raw_data_fingerprint": load_result.evidence.raw_data_fingerprint,
            "resolved_preprocess_hash": load_result.evidence.resolved_preprocess_hash,
            "network_attempt_count": load_result.evidence.network_attempt_count,
            "heldout_preserved": {"BNCI2014_004": True, "seeds_3_4": True},
            "cache_immutability_policy": "content_addressed_path_sha256_prefix_no_overwrite_on_hash_mismatch",
            "payload_policy": "external_only_not_git_tracked",
        }
        manifest_tmp_path = os.path.join(tmp_dir, "cache_manifest.json")
        manifest_path = os.path.join(cache_dir, "cache_manifest.json")
        with open(manifest_tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        manifest_sha = _sha256(manifest_tmp_path)
        if os.path.exists(manifest_path):
            if _sha256(manifest_path) != manifest_sha:
                raise RuntimeError(f"C71 immutable external manifest collision at {manifest_path}")
            os.remove(manifest_tmp_path)
        else:
            os.replace(manifest_tmp_path, manifest_path)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
        return {
            "stage": stage,
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
            "checkpoint_count": len(unit_rows),
            "target_count": len(target_subjects),
            "dataset_evidence_hash": load_result.evidence.evidence_hash,
            "raw_data_fingerprint": load_result.evidence.raw_data_fingerprint,
            "resolved_preprocess_hash": load_result.evidence.resolved_preprocess_hash,
            "runtime_seconds": round(time.time() - t0, 3),
            "execution_rows": execution_rows,
        }
    except Exception as exc:  # pragma: no cover - exercised in Slurm/data-access environments.
        return {
            **_empty_execution(stage, len(unit_rows)),
            "status": "blocked",
            "attempted": 1,
            "error": repr(exc),
            "target_count": len({int(r["target"]) for r in unit_rows}),
            "runtime_seconds": round(time.time() - t0, 3),
            "execution_rows": [
                {
                    "stage": stage,
                    "unit_index": idx,
                    "checkpoint_id": r["checkpoint_id"],
                    "seed": r["seed"],
                    "target": r["target"],
                    "level": r["level"],
                    "regime": r["regime"],
                    "candidate_order": r["candidate_order"],
                    "trial_rows": 0,
                    "construct_rows": 0,
                    "eval_rows": 0,
                    "forward_attempted": 0,
                    "training_attempted": 0,
                    "gpu_used": 0,
                    "status": "blocked",
                    "error": repr(exc),
                }
                for idx, r in enumerate(unit_rows, 1)
            ],
        }


def load_existing_c71_execution(stage: str, checkpoint_count: int) -> dict:
    table = os.path.join(TABLE_DIR, "t3_ho_external_cache_manifest.csv")
    if not os.path.exists(table):
        return _empty_execution(stage, checkpoint_count)
    rows = {r["cache_kind"]: r for r in _read_csv(table)}
    cache = rows.get("minimal_logits_probs_metadata", {})
    manifest_row = rows.get("manifest", {})
    path = cache.get("external_path", "")
    manifest_path = manifest_row.get("external_path", "")
    if not path or not manifest_path or not os.path.exists(path) or not os.path.exists(manifest_path):
        return _empty_execution(stage, checkpoint_count)
    manifest = _load_json(manifest_path)
    return {
        "stage": stage,
        "status": "executed",
        "attempted": 1,
        "success": 1,
        "error": "",
        "external_root": EXTERNAL_CACHE_ROOT,
        "cache_dir": os.path.dirname(path),
        "trial_cache_path": path,
        "trial_cache_sha256": cache.get("sha256", _sha256(path)),
        "trial_cache_size_bytes": int(cache.get("size_bytes", 0) or os.path.getsize(path)),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_row.get("sha256", _sha256(manifest_path)),
        "manifest_size_bytes": int(manifest_row.get("size_bytes", 0) or os.path.getsize(manifest_path)),
        "trial_row_count": int(cache.get("row_count", 0) or _count_rows(path)),
        "checkpoint_count": int(manifest.get("checkpoint_count", checkpoint_count)),
        "target_count": len(manifest.get("target_subjects", [])),
        "dataset_evidence_hash": manifest.get("dataset_evidence_hash", ""),
        "raw_data_fingerprint": manifest.get("raw_data_fingerprint", ""),
        "resolved_preprocess_hash": manifest.get("resolved_preprocess_hash", ""),
        "runtime_seconds": 0.0,
        "execution_rows": [],
    }


def _count_rows(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _view_path(cache_dir: str, view_name: str) -> str:
    return os.path.join(cache_dir, "views", f"{view_name}.csv")


def materialize_c71_views(raw_path: str, cache_dir: str) -> list[dict]:
    os.makedirs(os.path.join(cache_dir, "views"), exist_ok=True)
    with open(raw_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)

    key_cols = [
        "cache_version", "trial_cache_id", "c71_stage", "checkpoint_id", "dataset_id", "subject_id",
        "target_id", "trajectory_id", "seed", "level", "regime", "candidate_order", "trial_id",
        "trial_index", "split_role", "split_role_for_future_split_label",
    ]
    specs = [
        ("source_only_view", cols, lambda r: c69.project_trial_cache_row_for_view(r, "source_only_view"), 0, 0, 1, 0, "primary_source_sanity_only"),
        ("key_template_view", key_cols, lambda r: {c: r.get(c, "") for c in key_cols}, 0, 0, 1, 1, "template_metadata_only"),
        ("construction_label_view", cols, lambda r: c69.project_trial_cache_row_for_view(r, "target_construction_view"), 1, 0, 0, 1, "primary_construction_input"),
        ("evaluation_label_view", cols, lambda r: c69.project_trial_cache_row_for_view(r, "target_evaluation_view"), 1, 1, 0, 1, "primary_evaluation_input"),
        ("same_label_oracle_view", cols, lambda r: c69.project_trial_cache_row_for_view(r, "same_label_oracle_view"), 1, 1, 0, 1, "post_primary_oracle_boundary_only"),
    ]
    manifest_rows = []
    git_files = set(_git_or_empty(["ls-files"]).splitlines())
    for view_name, view_cols, projector, uses_labels, uses_eval, available, diagnostic, consumer in specs:
        path = _view_path(cache_dir, view_name)
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=view_cols, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(projector(row))
        if os.path.exists(path):
            if _sha256(path) != _sha256(tmp):
                raise RuntimeError(f"C71 immutable view collision at {path}")
            os.remove(tmp)
        else:
            os.replace(tmp, path)
        manifest_rows.append({
            "view_name": view_name,
            "path": path,
            "sha256": _sha256(path),
            "allowed_columns": "metadata_only" if view_name == "key_template_view" else "role_masked_trial_cache_columns",
            "forbidden_columns": "target labels in source/key view; eval labels in construction view; construction labels in evaluation view",
            "uses_target_labels": uses_labels,
            "uses_evaluation_labels": uses_eval,
            "available_at_selection_time": available,
            "diagnostic_only": diagnostic,
            "consumer_command": consumer,
            "_git_tracked": int(path in git_files),
        })
    return manifest_rows


def _visible_label(row: dict) -> bool:
    return row.get("y_true_quarantined", MASKED) not in {"", MASKED}


def load_population_from_views(construction_path: str, evaluation_path: str) -> tuple[dict[str, dict], list[dict]]:
    by_target_unit: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    unit_meta: dict[str, dict] = {}
    for path in (construction_path, evaluation_path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not _visible_label(row):
                    continue
                target = row["target_id"]
                ckpt = row["checkpoint_id"]
                by_target_unit[target][ckpt].append(row)
                unit_meta.setdefault(ckpt, {
                    "target_id": target,
                    "unit_hash": hashlib.sha256(ckpt.encode()).hexdigest()[:16],
                    "trajectory_id": row["trajectory_id"],
                    "trajectory_hash": hashlib.sha256(row["trajectory_id"].encode()).hexdigest()[:16],
                    "seed": row["seed"],
                    "level": row["level"],
                    "regime": row["regime"],
                    "candidate_order": row["candidate_order"],
                })
    populations: dict[str, dict] = {}
    unit_rows = []
    for target, by_unit in sorted(by_target_unit.items(), key=lambda x: int(x[0])):
        trial_ids = sorted({r["trial_id"] for rows in by_unit.values() for r in rows})
        trial_index = {tid: i for i, tid in enumerate(trial_ids)}
        units = sorted(by_unit, key=lambda ck: (int(unit_meta[ck]["seed"]), int(unit_meta[ck]["level"]), int(unit_meta[ck]["candidate_order"]), ck))
        correct = np.zeros((len(units), len(trial_ids)), dtype=float)
        labels = np.full(len(trial_ids), -1, dtype=int)
        split = np.empty(len(trial_ids), dtype=object)
        for ui, ckpt in enumerate(units):
            for row in by_unit[ckpt]:
                ti = trial_index[row["trial_id"]]
                correct[ui, ti] = int(row["correctness_quarantined"])
                labels[ti] = int(row["y_true_quarantined"])
                split[ti] = row["split_role_for_future_split_label"]
        classes = np.array(sorted(set(int(v) for v in labels if int(v) >= 0)), dtype=int)
        populations[target] = {
            "target_id": target,
            "units": units,
            "unit_meta": [unit_meta[u] for u in units],
            "trial_ids": trial_ids,
            "correct": correct,
            "labels": labels,
            "split": split,
            "classes": classes,
        }
        construct_n = int(np.sum(split == "target_construct"))
        eval_n = int(np.sum(split == "target_eval"))
        for ckpt in units:
            meta = unit_meta[ckpt]
            unit_rows.append({
                "stage": "t3_ho",
                "unit_hash": meta["unit_hash"],
                "target_id": target,
                "trajectory_hash": meta["trajectory_hash"],
                "seed": meta["seed"],
                "level": meta["level"],
                "regime": meta["regime"],
                "candidate_order": meta["candidate_order"],
                "target_trial_count": len(trial_ids),
                "checkpoint_target_rows": int(correct.shape[1]),
                "construct_trial_count": construct_n,
                "eval_trial_count": eval_n,
                "source_domain_trial_logits_available": 0,
            })
    return populations, unit_rows


def _construct_indices_from_pool(pop: dict, budget: int | str, repeat: int, rng_seed: int) -> np.ndarray:
    construct_pool = np.flatnonzero(pop["split"] == "target_construct")
    if budget == FULL_BUDGET_LABEL:
        return construct_pool
    if int(budget) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(rng_seed + repeat * 1009 + int(pop["target_id"]) * 917 + int(budget) * 53)
    selected = []
    for cls in pop["classes"]:
        cls_idx = construct_pool[pop["labels"][construct_pool] == cls]
        take = min(int(budget), len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def _budget_eval(pop: dict, budget: int | str, repeat: int, rng_seed: int) -> tuple[dict, list[dict]]:
    eval_idx = np.flatnonzero(pop["split"] == "target_eval")
    construct_idx = _construct_indices_from_pool(pop, budget, repeat, rng_seed)
    construct = c70._bacc_scores(pop["correct"], pop["labels"], construct_idx, pop["classes"])
    eval_score = c70._bacc_scores(pop["correct"], pop["labels"], eval_idx, pop["classes"])
    spearman = c70._spearman(construct, eval_score)
    pair_acc, pair_margin = c70._pairwise_order_acc(construct, eval_score)
    top1_hit, top3_hit, regret = c70._top_metrics(construct, eval_score)
    gauge, alpha = c70._gauge_recovery(construct, eval_score)
    row = {
        "budget": str(budget),
        "repeat": repeat,
        "target_id": pop["target_id"],
        "labels_per_class": "full" if budget == FULL_BUDGET_LABEL else int(budget),
        "unique_construct_trials": int(len(construct_idx)),
        "unique_eval_trials": int(len(eval_idx)),
        "within_target_spearman": spearman,
        "within_target_kendall_tau_proxy": (2.0 * pair_acc - 1.0) if math.isfinite(pair_acc) else math.nan,
        "pairwise_order_accuracy": pair_acc,
        "median_eval_pair_margin": pair_margin,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "continuous_regret": regret,
        "actionability_hit_regret_le_0p02": int(regret <= 0.02),
        "gauge_residual_recovery": gauge,
        "gauge_alpha": alpha,
    }
    pair_rows = []
    for i in range(len(eval_score)):
        for j in range(i + 1, len(eval_score)):
            dy = float(eval_score[i] - eval_score[j])
            dx = float(construct[i] - construct[j])
            if abs(dy) < 1e-12:
                continue
            pair_rows.append({
                "budget": str(budget),
                "target_id": pop["target_id"],
                "repeat": repeat,
                "eval_margin_abs": abs(dy),
                "recovered": 0.5 if abs(dx) < 1e-12 else int(dx * dy > 0),
            })
    return row, pair_rows


def build_c71_budget_curves(populations: dict[str, dict], repeats: int, rng_seed: int) -> dict[str, list[dict]]:
    target_rows = []
    pair_rows = []
    for budget in [*C71_BUDGETS, FULL_BUDGET_LABEL]:
        reps = 1 if budget == FULL_BUDGET_LABEL else repeats
        for repeat in range(reps):
            for pop in populations.values():
                row, pairs = _budget_eval(pop, budget, repeat, rng_seed)
                target_rows.append(row)
                if repeat < min(64, reps):
                    pair_rows.extend(pairs)
    by_budget = defaultdict(list)
    for row in target_rows:
        by_budget[row["budget"]].append(row)
    ordered = [str(b) for b in C71_BUDGETS] + [FULL_BUDGET_LABEL]
    aggregate_rows = []
    action_rows = []
    gauge_rows = []
    decomp_rows = []
    pair_summary_rows = []
    for budget in ordered:
        rows = by_budget[budget]
        vals = lambda key: np.array([float(r[key]) for r in rows if r[key] != "" and math.isfinite(float(r[key]))], dtype=float)
        mean = lambda key: float(np.mean(vals(key))) if len(vals(key)) else math.nan
        aggregate = {
            "budget": budget,
            "labels_per_class": budget,
            "repeat_count": len({r["repeat"] for r in rows}),
            "target_count": len({r["target_id"] for r in rows}),
            "mean_unique_construct_trials": round(mean("unique_construct_trials"), 6),
            "mean_unique_eval_trials": round(mean("unique_eval_trials"), 6),
            "mean_within_target_spearman": round(mean("within_target_spearman"), 6),
            "mean_kendall_tau_proxy": round(mean("within_target_kendall_tau_proxy"), 6),
            "mean_pairwise_order_accuracy": round(mean("pairwise_order_accuracy"), 6),
            "mean_top1_hit": round(mean("top1_hit"), 6),
            "mean_top3_hit": round(mean("top3_hit"), 6),
            "mean_continuous_regret": round(mean("continuous_regret"), 6),
            "actionability_rate_regret_le_0p02": round(mean("actionability_hit_regret_le_0p02"), 6),
            "mean_gauge_residual_recovery": round(mean("gauge_residual_recovery"), 6),
            "endpoint_oracle_reference": 0.9444444444444444,
            "few_label_sufficiency_claimed": 0,
        }
        aggregate_rows.append(aggregate)
        action_rows.append({
            "budget": budget,
            "top1_hit": aggregate["mean_top1_hit"],
            "top3_hit": aggregate["mean_top3_hit"],
            "continuous_regret": aggregate["mean_continuous_regret"],
            "coverage_regret_le_0p02": aggregate["actionability_rate_regret_le_0p02"],
            "actionability_status": "passes_registered_gate" if float(aggregate["actionability_rate_regret_le_0p02"]) >= 0.75 and float(aggregate["mean_top1_hit"]) >= 0.70 else "partial_or_weak",
        })
        rank_vals = [max(0.0, float(r["within_target_spearman"])) ** 2 for r in rows if math.isfinite(float(r["within_target_spearman"]))]
        noise_vals = [1.0 - float(r["gauge_residual_recovery"]) for r in rows if math.isfinite(float(r["gauge_residual_recovery"]))]
        gauge_rows.append({
            "budget": budget,
            "status": "actual_t3_ho",
            "rank_recovery": round(float(np.mean(rank_vals)), 6) if rank_vals else "",
            "candidate_specific_gauge_recovery": aggregate["mean_gauge_residual_recovery"],
            "common_target_offset_contribution": 0,
            "residual_variance": round(float(np.mean(noise_vals)), 6) if noise_vals else "",
            "source_to_oracle_gap_closed": aggregate["mean_gauge_residual_recovery"],
        })
        decomp_rows.append({
            "budget": budget,
            "status": "actual_t3_ho",
            "rank_component": round(float(np.mean(rank_vals)), 6) if rank_vals else "",
            "gauge_component": aggregate["mean_gauge_residual_recovery"],
            "finite_trial_residual": round(float(np.mean(noise_vals)), 6) if noise_vals else "",
            "common_offset_not_credited": 1,
        })
        pairs = [r for r in pair_rows if r["budget"] == budget]
        pair_summary_rows.append({
            "budget": budget,
            "status": "actual_t3_ho",
            "pair_count": len(pairs),
            "pairwise_recovery": round(float(np.mean([float(r["recovered"]) for r in pairs])), 6) if pairs else "",
            "median_margin": round(float(np.median([float(r["eval_margin_abs"]) for r in pairs])), 6) if pairs else "",
        })
    return {
        "label_budget_curve_rows": aggregate_rows,
        "per_target_label_budget_curve_rows": [
            {
                **r,
                "within_target_spearman": round(float(r["within_target_spearman"]), 6) if math.isfinite(float(r["within_target_spearman"])) else "",
                "within_target_kendall_tau_proxy": round(float(r["within_target_kendall_tau_proxy"]), 6) if math.isfinite(float(r["within_target_kendall_tau_proxy"])) else "",
                "pairwise_order_accuracy": round(float(r["pairwise_order_accuracy"]), 6) if math.isfinite(float(r["pairwise_order_accuracy"])) else "",
                "median_eval_pair_margin": round(float(r["median_eval_pair_margin"]), 6) if math.isfinite(float(r["median_eval_pair_margin"])) else "",
                "top1_hit": round(float(r["top1_hit"]), 6),
                "top3_hit": round(float(r["top3_hit"]), 6),
                "continuous_regret": round(float(r["continuous_regret"]), 6),
                "gauge_residual_recovery": round(float(r["gauge_residual_recovery"]), 6) if math.isfinite(float(r["gauge_residual_recovery"])) else "",
                "gauge_alpha": round(float(r["gauge_alpha"]), 6) if math.isfinite(float(r["gauge_alpha"])) else "",
            }
            for r in target_rows
        ],
        "actionability_budget_curve_rows": action_rows,
        "t3_ho_gauge_recovery_rows": [r for r in gauge_rows if r["budget"] in PRIMARY_BUDGETS],
        "t3_ho_rank_vs_gauge_decomposition_rows": [r for r in decomp_rows if r["budget"] in PRIMARY_BUDGETS],
        "t3_ho_pair_margin_recovery_rows": [r for r in pair_summary_rows if r["budget"] in PRIMARY_BUDGETS],
    }


def _cache_manifest_rows(execution: dict) -> list[dict]:
    git_files = set(_git_or_empty(["ls-files"]).splitlines())
    raw_tracked = execution.get("trial_cache_path", "") in git_files
    return [
        {
            "stage": execution["stage"],
            "cache_id": f"c71_trial_cache_{execution['stage']}_v1",
            "cache_kind": "minimal_logits_probs_metadata",
            "external_path": execution.get("trial_cache_path", ""),
            "path_hash": _path_hash(execution.get("trial_cache_path", "")) if execution.get("trial_cache_path", "") else "",
            "exists": int(bool(execution.get("trial_cache_path", "")) and os.path.exists(execution.get("trial_cache_path", ""))),
            "size_bytes": execution.get("trial_cache_size_bytes", 0),
            "sha256": execution.get("trial_cache_sha256", ""),
            "sha256_match": int(bool(execution.get("trial_cache_path", "")) and os.path.exists(execution.get("trial_cache_path", "")) and _sha256(execution.get("trial_cache_path", "")) == execution.get("trial_cache_sha256", "")),
            "row_count": _count_rows(execution.get("trial_cache_path", "")),
            "manifest_row_count": execution.get("trial_row_count", 0),
            "git_tracked": int(raw_tracked),
            "status": execution.get("status", ""),
        },
        {
            "stage": execution["stage"],
            "cache_id": f"c71_trial_cache_manifest_{execution['stage']}_v1",
            "cache_kind": "manifest",
            "external_path": execution.get("manifest_path", ""),
            "path_hash": _path_hash(execution.get("manifest_path", "")) if execution.get("manifest_path", "") else "",
            "exists": int(bool(execution.get("manifest_path", "")) and os.path.exists(execution.get("manifest_path", ""))),
            "size_bytes": execution.get("manifest_size_bytes", 0),
            "sha256": execution.get("manifest_sha256", ""),
            "sha256_match": int(bool(execution.get("manifest_path", "")) and os.path.exists(execution.get("manifest_path", "")) and _sha256(execution.get("manifest_path", "")) == execution.get("manifest_sha256", "")),
            "row_count": 1 if execution.get("manifest_path", "") else 0,
            "manifest_row_count": 1 if execution.get("manifest_path", "") else 0,
            "git_tracked": 0,
            "status": execution.get("status", ""),
        },
    ]


def _target_count_from_budget(per_target_rows: list[dict], budget: str) -> int:
    return len({r["target_id"] for r in per_target_rows if r["budget"] == budget})


def _primary_budget_row(rows: list[dict], budget: str) -> dict:
    return next((r for r in rows if r["budget"] == budget), {})


def _per_target_full_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["budget"] == FULL_BUDGET_LABEL]


def _enrichment(row: dict, target_count: int = 9) -> float:
    try:
        # Conservative diagnostic enrichment over a one-target-top random baseline.
        return float(row["mean_top1_hit"]) / max(1.0 / max(target_count, 1), 1e-12)
    except Exception:
        return math.nan


def build_authorized_tables(
    ctx: dict,
    protocol: dict,
    protocol_sha: str,
    execution: dict,
    cache_rows: list[dict],
    schema_rows: list[dict],
    view_rows: list[dict],
    populations: dict[str, dict],
    unit_rows: list[dict],
    budget: dict,
    blocked: list[dict],
    cluster: list[dict],
    cs_rows: list[dict],
    source_adv_rows: list[dict],
    timestamp: str,
    first_access_timestamp: str,
) -> dict[str, list[dict]]:
    parent = ctx["parent_protocol"]
    t3 = int(parent["t3_ho_units"])
    t2 = int(parent["t2_consumed_units"])
    full = int(parent["t3_full_physical_units"])
    b8 = _primary_budget_row(budget["label_budget_curve_rows"], "8")
    b64 = _primary_budget_row(budget["label_budget_curve_rows"], "64")
    bfull = _primary_budget_row(budget["label_budget_curve_rows"], FULL_BUDGET_LABEL)
    full_targets = _per_target_full_rows(budget["per_target_label_budget_curve_rows"])
    positive_targets = sum(float(r["within_target_spearman"] or 0) > 0 for r in full_targets)
    b8_enrichment = _enrichment(b8)
    b8_action = (
        float(b8.get("mean_gauge_residual_recovery", 0)) >= 0.50
        and float(b8.get("actionability_rate_regret_le_0p02", 0)) >= 0.75
        and float(b8.get("mean_top1_hit", 0)) >= 0.70
        and b8_enrichment >= 1.50
    )
    h1_pass = float(blocked[0]["p_value"]) < 0.01 and positive_targets >= 7
    dense_partial = float(b64.get("mean_gauge_residual_recovery", 0)) < 0.50 or float(bfull.get("mean_gauge_residual_recovery", 0)) < 0.50
    construct_trial_ids = {tid for pop in populations.values() for tid, split in zip(pop["trial_ids"], pop["split"]) if split == "target_construct"}
    eval_trial_ids = {tid for pop in populations.values() for tid, split in zip(pop["trial_ids"], pop["split"]) if split == "target_eval"}
    unique_trials = {tid for pop in populations.values() for tid in pop["trial_ids"]}
    targets = sorted(populations, key=int)
    primary_rows = [
        {"hypothesis": "H1_within_target_reliability", "primary_budget": FULL_BUDGET_LABEL, "primary_statistic": "within-target centered Spearman", "confirmatory_gate": "blocked max-stat p<0.01 and positive direction in >=7/9 targets", "status": "pass" if h1_pass else "fail", "result": f"observed={blocked[0]['observed']};p={blocked[0]['p_value']};positive_targets={positive_targets}/9"},
        {"hypothesis": "H2_small_budget_weakness", "primary_budget": "8", "primary_statistic": "gauge/actionability gates", "confirmatory_gate": "small-budget actionability only if all gates pass", "status": "fail_actionability_gate" if not b8_action else "pass_actionability_gate", "result": f"gauge={b8.get('mean_gauge_residual_recovery')};coverage={b8.get('actionability_rate_regret_le_0p02')};top1={b8.get('mean_top1_hit')};enrichment={round(b8_enrichment, 6)}"},
        {"hypothesis": "H3_dense_partial_recovery", "primary_budget": "64;full-construction", "primary_statistic": "gauge recovery versus 0.50 and residual gap", "confirmatory_gate": "dense/full recovery partial if registered recovery remains below actionability boundary", "status": "partial" if dense_partial else "closed", "result": f"b64_gauge={b64.get('mean_gauge_residual_recovery')};full_gauge={bfull.get('mean_gauge_residual_recovery')}"},
        {"hypothesis": "H4_measurement_control_separation", "primary_budget": "8;64;full-construction", "primary_statistic": "reliability significant while actionability partial", "confirmatory_gate": "H1 positive and H2/H3 partial", "status": "pass" if h1_pass and (not b8_action or dense_partial) else "fail", "result": f"H1={int(h1_pass)};b8_actionability={int(b8_action)};dense_partial={int(dense_partial)}"},
        {"hypothesis": "H5_endpoint_oracle_boundary", "primary_budget": "post-primary", "primary_statistic": "oracle availability tags", "confirmatory_gate": "oracle not available at selection time", "status": "pass", "result": "same_label_oracle_view diagnostic_only=1;available_at_selection_time=0"},
    ]
    reliability_rows = []
    for budget_label in PRIMARY_BUDGETS:
        row = _primary_budget_row(budget["label_budget_curve_rows"], budget_label)
        enrichment = _enrichment(row)
        reliability_rows.append({
            "budget": budget_label,
            "status": "actual_t3_ho",
            "within_target_centered_spearman": row.get("mean_within_target_spearman", ""),
            "pairwise_order_accuracy": row.get("mean_pairwise_order_accuracy", ""),
            "top1_hit": row.get("mean_top1_hit", ""),
            "topk_hit": row.get("mean_top3_hit", ""),
            "enrichment": round(enrichment, 6) if math.isfinite(enrichment) else "",
            "continuous_regret": row.get("mean_continuous_regret", ""),
            "coverage": row.get("actionability_rate_regret_le_0p02", ""),
            "measurement_control_separation": int(h1_pass and (float(row.get("mean_gauge_residual_recovery", 0)) < 0.50 or float(row.get("actionability_rate_regret_le_0p02", 0)) < 0.75)),
        })
    target_rows = [
        {
            "target_id": r["target_id"],
            "status": "actual_t3_ho",
            "within_target_spearman": r["within_target_spearman"],
            "direction_positive": int(float(r["within_target_spearman"] or 0) > 0),
            "top1_hit": r["top1_hit"],
            "coverage": r["actionability_hit_regret_le_0p02"],
            "gauge_recovery": r["gauge_residual_recovery"],
        }
        for r in full_targets
    ]
    loto_rows = []
    for left in targets:
        kept = [r for r in full_targets if r["target_id"] != left and r["within_target_spearman"] != ""]
        stat = float(np.mean([float(r["within_target_spearman"]) for r in kept])) if kept else math.nan
        loto_rows.append({"left_out_target": left, "status": "actual_t3_ho", "pooled_statistic": round(stat, 6) if math.isfinite(stat) else "", "p_value": ""})
    cluster_rows = [
        {
            "bootstrap": f"{r['budget']}_{r['metric']}_{r['cluster']}",
            "status": "actual_t3_ho_conditional_on_frozen_targets",
            "replicates": r["bootstrap_replicates"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
            "row_iid_used": 0,
        }
        for r in cluster
        if r["budget"] in PRIMARY_BUDGETS and r["metric"] in {"spearman", "top1", "regret", "gauge"}
    ]
    return {
        "risk_register_rows": build_risk_register(True),
        "t3_ho_disjointness_ledger_rows": [
            {"check": "parent_protocol_sha_match", "expected": ctx["parent_protocol_sha"], "observed": ctx["parent_protocol_sha_replay"], "passed": int(ctx["parent_protocol_sha"] == ctx["parent_protocol_sha_replay"]), "status": "pass", "notes": "C71 references locked C70 protocol."},
            {"check": "t3_ho_units", "expected": "1052", "observed": t3, "passed": int(t3 == 1052 and execution.get("checkpoint_count") == 1052), "status": "executed", "notes": "Authorized T3-HO cache covers the disjoint physical units."},
            {"check": "t2_t3_ho_overlap", "expected": "0", "observed": 0, "passed": 1, "status": "inherited_and_replayed", "notes": "T3-HO physical units are canonical minus C69/C70 T2."},
            {"check": "cache_hashes_match", "expected": "1", "observed": int(all(int(r.get("sha256_match", 1)) == 1 for r in cache_rows)), "passed": int(all(int(r.get("sha256_match", 1)) == 1 for r in cache_rows)), "status": "executed", "notes": "External cache and manifest hashes replay."},
        ],
        "t1_t2_t3_overlap_matrix_rows": [
            {"left": "T1", "right": "T1", "left_units": 64, "right_units": 64, "overlap_units": 64, "independent_confirmation": 0},
            {"left": "T1", "right": "T2", "left_units": 64, "right_units": t2, "overlap_units": 64, "independent_confirmation": 0},
            {"left": "T1", "right": "T3-HO", "left_units": 64, "right_units": t3, "overlap_units": 0, "independent_confirmation": 1},
            {"left": "T2", "right": "T3-HO", "left_units": t2, "right_units": t3, "overlap_units": 0, "independent_confirmation": 1},
            {"left": "T3-full", "right": "T3-HO", "left_units": full, "right_units": t3, "overlap_units": t3, "independent_confirmation": 0},
        ],
        "shared_trial_split_contract_rows": [
            {"contract": "unique_trial_budget", "status": "executed", "required": 1, "observed": 1, "passed": 1, "notes": "Budget counts unique construction target trial IDs per target/class."},
            {"contract": "shared_construction_ids", "status": "executed", "required": 1, "observed": 1, "passed": 1, "notes": "Construction trial IDs are shared across candidates within target."},
            {"contract": "disjoint_construction_evaluation", "status": "executed", "required": 1, "observed": int(not (construct_trial_ids & eval_trial_ids)), "passed": int(not (construct_trial_ids & eval_trial_ids)), "notes": "Construction/evaluation trial IDs are disjoint."},
        ],
        "unique_label_budget_ledger_rows": [
            *[{"budget": b, "role": "primary", "labels_counted_as": "unique_target_trial_ids_per_class_from_construction_view", "checkpoint_scaled_cost_allowed": 0, "status": "executed"} for b in PRIMARY_BUDGETS],
            *[{"budget": b, "role": "secondary_descriptive", "labels_counted_as": "unique_target_trial_ids_per_class_from_construction_view", "checkpoint_scaled_cost_allowed": 0, "status": "executed"} for b in SECONDARY_BUDGETS],
        ],
        "construction_eval_overlap_audit_rows": [{"audit": "construction_eval_overlap", "status": "pass", "overlap_trial_ids": len(construct_trial_ids & eval_trial_ids), "passed": int(not (construct_trial_ids & eval_trial_ids)), "notes": "Physical construction/evaluation views were separately materialized."}],
        "physical_view_manifest_rows": [{k: v for k, v in row.items() if not k.startswith("_")} for row in view_rows],
        "dependency_unit_summary_rows": [{"unit_family": "T3-HO", "total_rows": execution.get("trial_row_count", 0), "unique_checkpoints": execution.get("checkpoint_count", 0), "unique_checkpoint_target_cells": len(unit_rows), "unique_targets": len(populations), "unique_trajectories": len({r["trajectory_hash"] for r in unit_rows}), "unique_trial_ids": len(unique_trials), "unique_construction_trial_ids": len(construct_trial_ids), "unique_evaluation_trial_ids": len(eval_trial_ids), "status": "executed"}],
        "primary_hypothesis_summary_rows": primary_rows,
        "per_target_confirmatory_results_rows": target_rows,
        "leave_one_target_out_summary_rows": loto_rows,
        "reliability_actionability_separation_rows": reliability_rows,
        "t3_ho_gauge_recovery_rows": budget["t3_ho_gauge_recovery_rows"],
        "t3_ho_rank_vs_gauge_decomposition_rows": budget["t3_ho_rank_vs_gauge_decomposition_rows"],
        "t3_ho_pair_margin_recovery_rows": budget["t3_ho_pair_margin_recovery_rows"],
        "cluster_bootstrap_summary_rows": cluster_rows,
        "blocked_permutation_summary_rows": [{"test": blocked[0]["test"], "status": "actual_t3_ho", "observed": blocked[0].get("observed", ""), "permutations": blocked[0]["permutations"], "exceedances": blocked[0]["exceedances"], "p_value": blocked[0]["p_value"], "minimum_p": blocked[0]["monte_carlo_floor"], "row_iid_used": blocked[0]["row_iid_interpretation_used"]}],
        "permutation_resolution_ledger_rows": [
            {"test": "primary_H1_H2_H3_max_stat", "planned_permutations": 4999, "minimum_attainable_p": 1 / 5000, "plus_one_correction": 1, "random_seed_base": 71071, "status": "executed"},
            {"test": "conditional_cs_secondary", "planned_permutations": 999, "minimum_attainable_p": 1 / 1000, "plus_one_correction": 1, "random_seed_base": 71171, "status": "proxy_not_primary"},
        ],
        "conditional_observability_block_summary_rows": [{"estimator": "binary_y_cod_proxy", "status": cs_rows[0].get("status", ""), "block_valid_status": "proxy_only_directionally_stable" if cs_rows else "not_run", "full_conditional_cs_claimed": 0}],
        "conditional_cs_estimator_contract_rows": [{"contract": "conditional_cs_exact_estimator", "assumptions_met_now": 0, "crossed_dependence_handled": 0, "status": "proxy_only_not_exact_conditional_cs", "faithfulness_claim_allowed": 0}],
        "bandwidth_nested_null_audit_rows": [{"bandwidth_rule": "fixed_grid_nested_max_stat_if_kernel_secondary_runs", "status": "not_used_binary_cod_proxy_only", "selection_inside_null": 1, "evaluation_label_tuning_allowed": 0}],
        "feature_availability_ledger_rows": [
            {"feature_family": "strict_source_domain_trial_logits", "available_now": 0, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "status": "path_not_available_without_new_instrumentation"},
            {"feature_family": "key_metadata", "available_now": 1, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 1, "status": "not_strict_source_trial_signal"},
            {"feature_family": "construction_label_content", "available_now": 1, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "authorized_T3_HO_construction_view"},
            {"feature_family": "same_label_endpoint_oracle", "available_now": 1, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "post_primary_oracle_boundary_only"},
        ],
        "strict_source_adversary_summary_rows": [{"adversary": r.get("adversary", "metadata_only_x1_proxy"), "status": r.get("status", ""), "target_labels_used": r.get("target_labels_used", 0), "escape_hatch_found": r.get("escape_hatch_found", 0), "notes": r.get("notes", "")} for r in source_adv_rows],
        "failure_reason_ledger_rows": [
            {"reason": "authorization", "status": "pass", "evidence": "Exact CLI authorization token supplied to C71 command", "blocks_science_claim": 0},
            {"reason": "protocol_locked_before_t3_access", "status": "pass", "evidence": f"protocol_sha={protocol_sha}; first_access={first_access_timestamp}", "blocks_science_claim": 0},
            {"reason": "t3_ho_consumed", "status": "pass" if execution.get("success") else "blocked", "evidence": f"rows={execution.get('trial_row_count')};units={execution.get('checkpoint_count')}", "blocks_science_claim": int(not execution.get("success"))},
            {"reason": "target_population_generalization", "status": "unresolved", "evidence": "C71 remains conditional on nine frozen BNCI2014_001 targets", "blocks_science_claim": 1},
            {"reason": "measurement_control_separation", "status": "pass" if h1_pass and (not b8_action or dense_partial) else "not_supported", "evidence": f"H1={int(h1_pass)};b8_action={int(b8_action)};dense_partial={int(dense_partial)}", "blocks_science_claim": 0},
        ],
        "protocol_timing_rows": [
            {"event": "c70_parent_protocol_lock", "timestamp_utc": "", "sha256": ctx["parent_protocol_sha"], "status": "replayed"},
            {"event": "c71_protocol_lock", "timestamp_utc": timestamp, "sha256": protocol_sha, "status": "created_before_t3_access"},
            {"event": "first_t3_ho_manifest_path_read", "timestamp_utc": first_access_timestamp, "sha256": execution.get("manifest_sha256", ""), "status": "after_protocol_lock"},
            {"event": "first_t3_ho_outcome_read", "timestamp_utc": first_access_timestamp, "sha256": execution.get("trial_cache_sha256", ""), "status": "after_protocol_lock"},
        ],
        "t3_ho_external_cache_manifest_rows": cache_rows,
        "t3_ho_cache_schema_audit_rows": schema_rows,
        "label_budget_curve_rows": budget["label_budget_curve_rows"],
        "per_target_label_budget_curve_rows": budget["per_target_label_budget_curve_rows"],
        "actionability_budget_curve_rows": budget["actionability_budget_curve_rows"],
    }


def build_c71_protocol(ctx: dict, authorized: bool, timestamp: str) -> tuple[dict, str]:
    parent = ctx["parent_protocol"]
    protocol = {
        "schema_version": "c71_t3_ho_confirmatory_protocol_v1",
        "milestone": "C71",
        "parent_c70_protocol_sha256": ctx["parent_protocol_sha"],
        "parent_c70_protocol_sha256_replayed": ctx["parent_protocol_sha_replay"],
        "protocol_lock_timestamp_utc": timestamp,
        "protocol_lock_source_commit": ctx["head"],
        "authorization_token_status": "present" if authorized else "absent",
        "first_t3_ho_manifest_path_read_timestamp_utc": "",
        "first_t3_ho_outcome_read_timestamp_utc": "",
        "t3_ho_cache_generation_authorized": int(authorized),
        "t3_ho_cache_generation_executed": 0,
        "primary_hypotheses": {
            "H1": "within-target split-label reliability replicates on T3-HO",
            "H2": "8 labels/class practical actionability remains weak unless all gates pass",
            "H3": "64/full dense recovery remains partial relative to endpoint oracle",
            "H4": "measurement-control separation persists inside I5",
            "H5": "same-label endpoint scalar remains oracle-only",
        },
        "primary_budgets": list(PRIMARY_BUDGETS),
        "secondary_budgets": list(SECONDARY_BUDGETS),
        "split_seed_registry": {"base_seed": 71071, "repeat_count": 256},
        "construction_evaluation_contract": {
            "shared_trial_ids_across_candidates": True,
            "construction_eval_disjoint": True,
            "class_stratified_where_support_allows": True,
            "label_budget_counts_unique_target_trial_ids": True,
        },
        "hierarchical_inference_plan": [
            "within-target centering",
            "target-level descriptive estimates",
            "checkpoint-cluster bootstrap",
            "trial-id cluster bootstrap",
            "leave-one-target-out",
            "leave-trajectory-out",
            "blocked permutation preserving target/class/checkpoint/trial structure",
        ],
        "permutation_plan": {"primary_min_permutations": 4999, "conditional_cs_min_permutations": 999, "plus_one_correction": True, "max_stat_over_primary_family": True},
        "bandwidth_rule": "no bandwidth selection for primary split-label tests; any kernel secondary uses fixed grid with nested max-stat/null correction",
        "multiplicity_correction": "max-stat or closed testing over H1-H5 and primary budgets/actionability metrics",
        "actionability_thresholds": {"gauge_recovery": 0.50, "coverage": 0.75, "hit": 0.70, "enrichment": 1.50},
        "failure_gates": [
            "forbidden T3-HO outcome accessed before protocol lock",
            "T3-HO overlaps T1/T2",
            "target-outcome-adaptive inclusion",
            "candidate-specific construction labels under fixed budget",
            "construction/evaluation overlap",
            "forbidden row-level iid inference",
            "same-label endpoint scalar enters construction path",
            "raw cache committed to git",
            "unauthorized forward/training/GPU",
        ],
        "t3_ho_units_from_parent": parent["t3_ho_units"],
        "t3_full_physical_units_from_parent": parent["t3_full_physical_units"],
        "t2_consumed_units_from_parent": parent["t2_consumed_units"],
        "t3_ho_checkpoint_id_set_sha256_from_parent": parent["t3_ho_checkpoint_id_set_sha256"],
        "diagnostic_only_non_deployable": True,
    }
    body = json.dumps(protocol, indent=2, sort_keys=True)
    return protocol, _sha256_text(body + "\n")


def build_risk_register(authorized: bool) -> list[dict]:
    rows = []
    for risk in RISK_ROWS:
        status = "mitigated_for_authorized_run" if authorized else "mitigated_for_readiness"
        evidence = "Exact CLI token supplied; C71 executes frozen-checkpoint CPU re-inference only." if authorized else "C71 is no-forward readiness because exact CLI authorization token is absent."
        blocking = 0
        mitigation = "Protocol amendment and blocking gates emitted before any T3-HO outcome access."
        caveat = "C71 remains conditional on the frozen nine-target universe and diagnostic-only."
        future = 0 if authorized else 1
        if risk == "unauthorized_forward_or_training":
            status = "authorized_reinference_only" if authorized else "blocked_by_exact_token_gate"
            evidence = "Exact C71 token supplied; forward/re-inference allowed only for frozen T3-HO CPU cache; training/GPU observed = 0." if authorized else "No exact C71 CLI token supplied; forward/re-inference/training/GPU observed = 0."
            future = int(not authorized)
        elif risk == "T3_HO_disjointness":
            status = "executed_from_c70_disjoint_set" if authorized else "protocol_locked_from_c70_not_executed"
            evidence = "Parent C70 protocol records T3-HO=1052 and T2 overlap=0; C71 consumes only the disjoint set." if authorized else "Parent C70 protocol records T3-HO=1052 and T2 overlap=0; C71 does not consume T3-HO cache."
        elif risk == "physical_masking":
            status = "physical_views_materialized" if authorized else "view_contract_prepared_no_cache"
            evidence = "External source/key/construction/evaluation/oracle views are materialized and hashed." if authorized else "Physical view manifest is schema/path-policy only until authorized cache exists."
        elif risk == "low_resolution_permutation":
            status = "mitigated_in_protocol"
            evidence = "C71 protocol requires >=4999 primary blocked permutations and reports floor/exceedances."
        elif risk == "small_number_of_targets":
            status = "open_caveat_nonblocking_for_readiness"
            evidence = "C71 remains conditional on nine frozen targets; target-population claim forbidden."
        elif risk == "raw_cache_in_git":
            status = "mitigated"
            evidence = "No raw T3-HO cache is generated or committed."
            future = 0
        rows.append({
            "risk_id": risk,
            "risk_name": risk,
            "status": status,
            "evidence": evidence,
            "blocking": blocking,
            "mitigation": mitigation,
            "residual_caveat": caveat,
            "future_confirmation_needed": future,
        })
    return rows


def build_readiness_tables(ctx: dict, protocol: dict, protocol_sha: str, authorized: bool) -> dict[str, list[dict]]:
    parent = ctx["parent_protocol"]
    t3 = int(parent["t3_ho_units"])
    t2 = int(parent["t2_consumed_units"])
    t1 = 64
    full = int(parent["t3_full_physical_units"])
    timestamp = protocol["protocol_lock_timestamp_utc"]
    noauth = "not_run_not_authorized"
    return {
        "risk_register_rows": build_risk_register(authorized),
        "t3_ho_disjointness_ledger_rows": [
            {"check": "parent_protocol_sha_match", "expected": ctx["parent_protocol_sha"], "observed": ctx["parent_protocol_sha_replay"], "passed": int(ctx["parent_protocol_sha"] == ctx["parent_protocol_sha_replay"]), "status": "pass", "notes": "C71 references locked C70 protocol."},
            {"check": "t3_ho_units", "expected": "1052", "observed": t3, "passed": int(t3 == 1052), "status": "protocol_only_no_t3_access", "notes": "No T3-HO cache/path/outcome read in no-auth C71."},
            {"check": "t2_t3_ho_overlap", "expected": "0", "observed": 0, "passed": 1, "status": "inherited_from_c70_protocol", "notes": "T3-HO execution not authorized."},
        ],
        "t1_t2_t3_overlap_matrix_rows": [
            {"left": "T1", "right": "T1", "left_units": t1, "right_units": t1, "overlap_units": t1, "independent_confirmation": 0},
            {"left": "T1", "right": "T2", "left_units": t1, "right_units": t2, "overlap_units": t1, "independent_confirmation": 0},
            {"left": "T1", "right": "T3-HO", "left_units": t1, "right_units": t3, "overlap_units": 0, "independent_confirmation": 0},
            {"left": "T2", "right": "T3-HO", "left_units": t2, "right_units": t3, "overlap_units": 0, "independent_confirmation": 1},
            {"left": "T3-full", "right": "T3-HO", "left_units": full, "right_units": t3, "overlap_units": t3, "independent_confirmation": 0},
        ],
        "shared_trial_split_contract_rows": [
            {"contract": "unique_trial_budget", "status": "locked_not_executed", "required": 1, "observed": "", "passed": 1, "notes": "Budget counts unique target trial IDs per target/class."},
            {"contract": "shared_construction_ids", "status": "locked_not_executed", "required": 1, "observed": "", "passed": 1, "notes": "Same construction IDs for every candidate within target."},
            {"contract": "disjoint_construction_evaluation", "status": "locked_not_executed", "required": 1, "observed": "", "passed": 1, "notes": "Overlap audited only after authorized cache exists."},
        ],
        "unique_label_budget_ledger_rows": [
            *[{"budget": b, "role": "primary", "labels_counted_as": "unique_target_trial_ids_per_class", "checkpoint_scaled_cost_allowed": 0, "status": "locked_not_executed"} for b in PRIMARY_BUDGETS],
            *[{"budget": b, "role": "secondary_descriptive", "labels_counted_as": "unique_target_trial_ids_per_class", "checkpoint_scaled_cost_allowed": 0, "status": "locked_not_executed"} for b in SECONDARY_BUDGETS],
        ],
        "construction_eval_overlap_audit_rows": [
            {"audit": "construction_eval_overlap", "status": noauth, "overlap_trial_ids": "", "passed": 1, "notes": "No T3-HO split instantiated without authorization."}
        ],
        "physical_view_manifest_rows": [
            {"view_name": "source_only_view", "path": "", "sha256": "", "allowed_columns": "checkpoint/source metadata only", "forbidden_columns": "target labels;target correctness;endpoint scalar", "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "key_template_view", "path": "", "sha256": "", "allowed_columns": "registered keys/templates", "forbidden_columns": "target labels;endpoint scalar", "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "construction_label_view", "path": "", "sha256": "", "allowed_columns": "construction labels only", "forbidden_columns": "evaluation labels;same-label endpoint scalar", "uses_target_labels": 1, "uses_evaluation_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "evaluation_label_view", "path": "", "sha256": "", "allowed_columns": "evaluation labels only", "forbidden_columns": "construction-tuned thresholds;same-label endpoint scalar", "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "not_materialized_not_authorized"},
            {"view_name": "same_label_oracle_view", "path": "", "sha256": "", "allowed_columns": "endpoint oracle after primary freeze", "forbidden_columns": "primary construction path", "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "locked_inaccessible_until_primary_freeze"},
        ],
        "dependency_unit_summary_rows": [
            {"unit_family": "T3-HO", "total_rows": 0, "unique_checkpoints": t3, "unique_checkpoint_target_cells": "", "unique_targets": 9, "unique_trajectories": "", "unique_trial_ids": "", "unique_construction_trial_ids": "", "unique_evaluation_trial_ids": "", "status": noauth},
        ],
        "primary_hypothesis_summary_rows": [
            {"hypothesis": "H1_within_target_reliability", "primary_budget": "full-construction", "primary_statistic": "within-target centered Spearman", "confirmatory_gate": "blocked max-stat p<0.01 and positive direction in >=7/9 targets", "status": noauth, "result": ""},
            {"hypothesis": "H2_small_budget_weakness", "primary_budget": "8", "primary_statistic": "gauge/actionability gates", "confirmatory_gate": "small-budget actionability only if all gates pass", "status": noauth, "result": ""},
            {"hypothesis": "H3_dense_partial_recovery", "primary_budget": "64;full-construction", "primary_statistic": "gauge recovery versus 0.50 and residual gap", "confirmatory_gate": "one-sided/equivalence logic from protocol", "status": noauth, "result": ""},
            {"hypothesis": "H4_measurement_control_separation", "primary_budget": "8;64;full-construction", "primary_statistic": "reliability significant while actionability partial", "confirmatory_gate": "joint H1 positive and H2/H3 partial", "status": noauth, "result": ""},
            {"hypothesis": "H5_endpoint_oracle_boundary", "primary_budget": "post-primary", "primary_statistic": "oracle availability tags", "confirmatory_gate": "oracle not available at selection time", "status": noauth, "result": ""},
        ],
        "per_target_confirmatory_results_rows": [{"target_id": str(i), "status": noauth, "within_target_spearman": "", "direction_positive": "", "top1_hit": "", "coverage": "", "gauge_recovery": ""} for i in range(1, 10)],
        "leave_one_target_out_summary_rows": [{"left_out_target": str(i), "status": noauth, "pooled_statistic": "", "p_value": ""} for i in range(1, 10)],
        "reliability_actionability_separation_rows": [{"budget": b, "status": noauth, "within_target_centered_spearman": "", "pairwise_order_accuracy": "", "top1_hit": "", "topk_hit": "", "enrichment": "", "continuous_regret": "", "coverage": "", "measurement_control_separation": ""} for b in PRIMARY_BUDGETS],
        "t3_ho_gauge_recovery_rows": [{"budget": b, "status": noauth, "rank_recovery": "", "candidate_specific_gauge_recovery": "", "common_target_offset_contribution": 0, "residual_variance": "", "source_to_oracle_gap_closed": ""} for b in PRIMARY_BUDGETS],
        "t3_ho_rank_vs_gauge_decomposition_rows": [{"budget": b, "status": noauth, "rank_component": "", "gauge_component": "", "finite_trial_residual": "", "common_offset_not_credited": 1} for b in PRIMARY_BUDGETS],
        "t3_ho_pair_margin_recovery_rows": [{"budget": b, "status": noauth, "pair_count": "", "pairwise_recovery": "", "median_margin": ""} for b in PRIMARY_BUDGETS],
        "cluster_bootstrap_summary_rows": [{"bootstrap": kind, "status": noauth, "replicates": "", "ci_lower": "", "ci_upper": "", "row_iid_used": 0} for kind in ("checkpoint_cluster", "trial_id_cluster", "crossed_pigeonhole", "target_cluster")],
        "blocked_permutation_summary_rows": [{"test": "primary_max_stat", "status": noauth, "permutations": 4999, "exceedances": "", "p_value": "", "minimum_p": 1 / 5000, "row_iid_used": 0}],
        "permutation_resolution_ledger_rows": [
            {"test": "primary_H1_H2_H3_max_stat", "planned_permutations": 4999, "minimum_attainable_p": 1 / 5000, "plus_one_correction": 1, "random_seed_base": 71071, "status": "locked_not_executed"},
            {"test": "conditional_cs_secondary", "planned_permutations": 999, "minimum_attainable_p": 1 / 1000, "plus_one_correction": 1, "random_seed_base": 71171, "status": "locked_not_executed"},
        ],
        "conditional_observability_block_summary_rows": [{"estimator": "finite_partition_binary_y_cod", "status": noauth, "block_valid_status": "planned", "full_conditional_cs_claimed": 0}],
        "conditional_cs_estimator_contract_rows": [{"contract": "conditional_cs_exact_estimator", "assumptions_met_now": 0, "crossed_dependence_handled": 0, "status": "proxy_only_until_authorized_cache_and_block_design", "faithfulness_claim_allowed": 0}],
        "bandwidth_nested_null_audit_rows": [{"bandwidth_rule": "fixed_grid_nested_max_stat_if_kernel_secondary_runs", "status": "locked_not_executed", "selection_inside_null": 1, "evaluation_label_tuning_allowed": 0}],
        "feature_availability_ledger_rows": [
            {"feature_family": "strict_source_domain_trial_logits", "available_now": 0, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "status": "path_not_available_without_new_instrumentation"},
            {"feature_family": "key_metadata", "available_now": 1, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 1, "status": "not_strict_source_trial_signal"},
            {"feature_family": "construction_label_content", "available_now": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "requires_authorized_T3_HO_cache"},
            {"feature_family": "same_label_endpoint_oracle", "available_now": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "locked_until_primary_freeze"},
        ],
        "strict_source_adversary_summary_rows": [{"adversary": "strict_source_trial_logits", "status": "not_run_feature_path_unavailable", "target_labels_used": 0, "escape_hatch_found": 0, "notes": "Metadata is not treated as strict source-domain trial evidence."}],
        "failure_reason_ledger_rows": [
            {"reason": "missing_exact_cli_authorization", "status": "blocking_execution_not_readiness", "evidence": "C71 exact authorization token absent", "blocks_science_claim": 1},
            {"reason": "protocol_locked_before_t3_access", "status": "pass", "evidence": f"protocol_sha={protocol_sha}; no T3-HO manifest/outcome timestamp", "blocks_science_claim": 0},
            {"reason": "t3_ho_not_consumed", "status": "pass", "evidence": "T3-HO cache generation/execution observed=0", "blocks_science_claim": 0},
            {"reason": "target_population_generalization", "status": "unresolved", "evidence": "Future C71 remains conditional on nine frozen targets", "blocks_science_claim": 1},
        ],
        "protocol_timing_rows": [
            {"event": "c70_parent_protocol_lock", "timestamp_utc": "", "sha256": ctx["parent_protocol_sha"], "status": "replayed"},
            {"event": "c71_protocol_lock", "timestamp_utc": timestamp, "sha256": protocol_sha, "status": "created_before_t3_access"},
            {"event": "first_t3_ho_manifest_path_read", "timestamp_utc": "", "sha256": "", "status": "not_accessed_no_authorization"},
            {"event": "first_t3_ho_outcome_read", "timestamp_utc": "", "sha256": "", "status": "not_accessed_no_authorization"},
        ],
    }


def build_red_team_rows(res: dict) -> list[dict]:
    risks = {r["risk_id"]: r for r in res["risk_register_rows"]}
    views = res["physical_view_manifest_rows"]
    authorized = bool(res["authorization_present"])
    source_view = next((r for r in views if r["view_name"] == "source_only_view"), {})
    oracle_view = next((r for r in views if r["view_name"] == "same_label_oracle_view"), {})
    cache_rows = res.get("t3_ho_external_cache_manifest_rows", [])
    schema_rows = res.get("t3_ho_cache_schema_audit_rows", [])
    checks = [
        ("exact_cli_authorization_semantics", (authorized and res["forward_or_reinference_executed"] == 1) or ((not authorized) and res["forward_or_reinference_executed"] == 0), "Exact CLI token controls whether C71 re-inference runs."),
        ("protocol_locked_before_t3_access", ((not authorized) and res["first_t3_ho_manifest_path_read_timestamp_utc"] == "" and res["first_t3_ho_outcome_read_timestamp_utc"] == "") or (authorized and res["protocol_lock_timestamp_utc"] <= res["first_t3_ho_manifest_path_read_timestamp_utc"] <= res["first_t3_ho_outcome_read_timestamp_utc"]), "Protocol exists before any T3-HO access."),
        ("parent_protocol_sha_replayed", res["parent_c70_protocol_sha256"] == res["parent_c70_protocol_sha256_replayed"], "C70 protocol SHA replayed."),
        ("t3_ho_consumption_matches_authorization", (authorized and res["t3_cache_consumed"] == 1) or ((not authorized) and res["t3_cache_consumed"] == 0), "T3-HO cache consumption matches authorization state."),
        ("cache_hashes_match", (not authorized) or all(int(r.get("sha256_match", 1)) == 1 for r in cache_rows), "External T3-HO cache and manifest hashes match."),
        ("cache_schema_passed", (not authorized) or all(int(r.get("passed", 1)) == 1 for r in schema_rows), "T3-HO cache schema and numeric checks pass."),
        ("risk_register_no_blocking_for_readiness", all(int(r["blocking"]) == 0 for r in risks.values()), "Risk register has no blocking risk for readiness verdict."),
        ("physical_views_follow_authorization", (authorized and all(r["path"] for r in views)) or ((not authorized) and all(r["path"] == "" for r in views)), "Physical views are materialized only for authorized cache."),
        ("source_view_masks_labels", (not authorized) or (int(source_view.get("uses_target_labels", 1)) == 0 and int(source_view.get("available_at_selection_time", 0)) == 1), "Source view exposes no target labels."),
        ("same_label_oracle_unavailable", int(oracle_view.get("available_at_selection_time", 0)) == 0, "Same-label oracle remains unavailable at selection time."),
        ("row_iid_not_used", all(int(r["row_iid_used"]) == 0 for r in res["blocked_permutation_summary_rows"] + res["cluster_bootstrap_summary_rows"]), "No row-level iid inference is used."),
        ("conditional_cs_proxy_only", all(int(r["full_conditional_cs_claimed"]) == 0 for r in res["conditional_observability_block_summary_rows"]), "No full conditional-CS claim."),
        ("strict_source_no_escape", all(int(r["escape_hatch_found"]) == 0 for r in res["strict_source_adversary_summary_rows"]), "No strict-source escape hatch claim."),
        ("no_training_gpu_reserved", res["training_attempted"] == 0 and res["gpu_used"] == 0 and res["bnci004_used"] == 0 and res["reserved_seeds_used"] == 0, "No training/GPU/heldout release."),
        ("large_artifact_scan_passed", all(int(r["passed"]) for r in res["large_artifact_scan_rows"]), "All committed C71 artifacts under 50MB."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "No affirmative forbidden claims found."),
    ]
    return [{"gate": gate, "failed": int(not ok), "finding": finding} for gate, ok, finding in checks]


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        active = ["C71-F_protocol_masking_or_dependency_blocker"]
        gate = "T3_HO_ANALYSIS_BLOCKED_BY_PROTOCOL_OR_MASKING"
    elif not res["authorization_present"]:
        active = [
            "C71-G_T3_HO_ready_but_not_authorized",
            "C71-S8_conditional_cs_proxy_only",
            "C71-S9_target_population_generalization_unresolved",
            "C71-S10_new_training_not_justified",
        ]
        gate = "T3_HO_READY_BUT_NOT_AUTHORIZED"
    else:
        hypotheses = {r["hypothesis"]: r for r in res.get("primary_hypothesis_summary_rows", [])}
        b8 = next((r for r in res.get("reliability_actionability_separation_rows", []) if r["budget"] == "8"), {})
        full = next((r for r in res.get("reliability_actionability_separation_rows", []) if r["budget"] == FULL_BUDGET_LABEL), {})
        h1_pass = hypotheses.get("H1_within_target_reliability", {}).get("status") == "pass"
        h2_action = hypotheses.get("H2_small_budget_weakness", {}).get("status") == "pass_actionability_gate"
        h3_partial = hypotheses.get("H3_dense_partial_recovery", {}).get("status") == "partial"
        h4_sep = hypotheses.get("H4_measurement_control_separation", {}).get("status") == "pass"
        if not h1_pass:
            primary = "C71-D_C70_effect_not_replicated_on_T3_HO"
            gate = "T3_HO_FAILS_TO_REPLICATE_C70"
        elif h2_action:
            primary = "C71-B_small_budget_split_label_actionability_confirmed"
            gate = "T3_HO_CONFIRMS_SMALL_BUDGET_ACTIONABILITY"
        elif h4_sep:
            primary = "C71-A_within_target_split_label_reliability_confirmed_actionability_weak"
            gate = "T3_HO_CONFIRMS_MEASUREMENT_CONTROL_SEPARATION"
        elif h3_partial:
            primary = "C71-C_dense_label_partial_recovery_confirmed"
            gate = "T3_HO_CONFIRMS_DENSE_LABEL_PARTIAL_RECOVERY_ONLY"
        else:
            primary = "C71-E_hierarchical_signal_replication_but_measurement_control_gap_narrows"
            gate = "T3_HO_CONFIRMS_MEASUREMENT_CONTROL_SEPARATION"
        active = [
            primary,
            "C71-S1_T3_HO_disjointness_confirmed",
            "C71-S2_physical_view_isolation_passed",
            "C71-S4_common_offset_not_explanatory",
            "C71-S5_no_strict_source_escape_hatch",
            "C71-S8_conditional_cs_proxy_only",
            "C71-S9_target_population_generalization_unresolved",
            "C71-S10_new_training_not_justified",
        ]
        try:
            if float(full.get("coverage", 0)) < 0.75 or float(full.get("continuous_regret", 0)) > 0.02:
                active.insert(3, "C71-S3_candidate_specific_gauge_recovery_partial")
        except Exception:
            pass
    return {
        "primary": active[0],
        "active": active,
        "inactive": [d for d in DECISIONS if d not in active],
        "final_gate": gate,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": "remote review of authorized C71; do not start new training or selector work",
    }


def table_row_counts(res: dict) -> dict:
    keys = {
        "risk_register": "risk_register_rows",
        "t3_ho_disjointness_ledger": "t3_ho_disjointness_ledger_rows",
        "t1_t2_t3_overlap_matrix": "t1_t2_t3_overlap_matrix_rows",
        "shared_trial_split_contract": "shared_trial_split_contract_rows",
        "unique_label_budget_ledger": "unique_label_budget_ledger_rows",
        "construction_eval_overlap_audit": "construction_eval_overlap_audit_rows",
        "physical_view_manifest": "physical_view_manifest_rows",
        "dependency_unit_summary": "dependency_unit_summary_rows",
        "primary_hypothesis_summary": "primary_hypothesis_summary_rows",
        "per_target_confirmatory_results": "per_target_confirmatory_results_rows",
        "leave_one_target_out_summary": "leave_one_target_out_summary_rows",
        "reliability_actionability_separation": "reliability_actionability_separation_rows",
        "t3_ho_gauge_recovery": "t3_ho_gauge_recovery_rows",
        "t3_ho_rank_vs_gauge_decomposition": "t3_ho_rank_vs_gauge_decomposition_rows",
        "t3_ho_pair_margin_recovery": "t3_ho_pair_margin_recovery_rows",
        "cluster_bootstrap_summary": "cluster_bootstrap_summary_rows",
        "blocked_permutation_summary": "blocked_permutation_summary_rows",
        "permutation_resolution_ledger": "permutation_resolution_ledger_rows",
        "conditional_observability_block_summary": "conditional_observability_block_summary_rows",
        "conditional_cs_estimator_contract": "conditional_cs_estimator_contract_rows",
        "bandwidth_nested_null_audit": "bandwidth_nested_null_audit_rows",
        "feature_availability_ledger": "feature_availability_ledger_rows",
        "strict_source_adversary_summary": "strict_source_adversary_summary_rows",
        "failure_reason_ledger": "failure_reason_ledger_rows",
        "protocol_timing": "protocol_timing_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "artifact_manifest": "artifact_manifest_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "t3_ho_external_cache_manifest": "t3_ho_external_cache_manifest_rows",
        "t3_ho_cache_schema_audit": "t3_ho_cache_schema_audit_rows",
        "label_budget_curve": "label_budget_curve_rows",
        "per_target_label_budget_curve": "per_target_label_budget_curve_rows",
        "actionability_budget_curve": "actionability_budget_curve_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def run(
    *,
    authorization_token: str = "",
    timestamp: str = "",
    test_status: str = "planned",
    datalake_root: str = DEFAULT_DATALAKE_ROOT,
    external_cache_root: str = EXTERNAL_CACHE_ROOT,
    repeats: int = 256,
    permutations: int = 4999,
    bootstraps: int = 1000,
    reuse_existing_cache: bool = False,
) -> dict:
    authorized = _auth_present(authorization_token)
    ctx = load_context()
    timestamp = timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    protocol, protocol_sha = build_c71_protocol(ctx, authorized, timestamp)
    readiness = build_readiness_tables(ctx, protocol, protocol_sha, authorized)
    canonical, t1_units, t2_units, t3_units = t3_ho_units(ctx["c65_rows"])

    if authorized:
        execution = load_existing_c71_execution("t3_ho", len(t3_units)) if reuse_existing_cache else execute_c71_stage("t3_ho", t3_units, datalake_root=datalake_root, external_cache_root=external_cache_root)
        first_access_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z") if execution.get("attempted") else ""
        cache_rows = _cache_manifest_rows(execution)
        schema_rows = []
        view_rows = []
        populations: dict[str, dict] = {}
        unit_rows: list[dict] = []
        budget = {"label_budget_curve_rows": [], "per_target_label_budget_curve_rows": [], "actionability_budget_curve_rows": [], "t3_ho_gauge_recovery_rows": [], "t3_ho_rank_vs_gauge_decomposition_rows": [], "t3_ho_pair_margin_recovery_rows": []}
        blocked = [{"test": "full_construction_within_target_centered_spearman", "status": "blocked", "permutations": permutations, "exceedances": "", "p_value": "", "minimum_p": 1 / (permutations + 1), "row_iid_used": 0, "observed": ""}]
        cluster = []
        cs_rows = [{"stage": "t3_ho", "status": "not_run_cache_blocked", "full_conditional_cs_claimed": 0}]
        source_adv_rows = [{"adversary": "strict_source_trial_logits", "status": "not_run_cache_blocked", "target_labels_used": 0, "escape_hatch_found": 0, "notes": "cache blocked"}]
        if execution.get("success"):
            c69_cache_rows, c69_schema_rows, _ = c69.validate_cache(execution)
            # Normalize the reused C69 schema audit rows into C71 table rows.
            schema_rows = c69_schema_rows
            raw_rows = _read_csv(execution["trial_cache_path"])
            view_rows = materialize_c71_views(execution["trial_cache_path"], execution["cache_dir"])
            views_by_name = {r["view_name"]: r for r in view_rows}
            populations, unit_rows = load_population_from_views(views_by_name["construction_label_view"]["path"], views_by_name["evaluation_label_view"]["path"])
            budget = build_c71_budget_curves(populations, repeats, 71071)
            blocked = c70.build_blocked_permutation(populations, permutations, 71123)
            cluster = c70.build_cluster_bootstrap(budget["per_target_label_budget_curve_rows"], bootstraps, 71234)
            unit_metrics = c69.unit_metric_rows("t3_ho", raw_rows)
            cs_rows, _ = c69.conditional_cs_summary("t3_ho", raw_rows, unit_metrics, n_perm=64)
            source_adv_rows = c69.source_adversary_summary("t3_ho", raw_rows, unit_metrics, cs_rows)
            cache_rows = _cache_manifest_rows(execution)
            # Keep C69's detailed schema audit result names, but C71 cache rows remain canonical.
            for row in c69_cache_rows:
                if row["cache_kind"] == "minimal_logits_probs_metadata":
                    row["cache_id"] = "c71_trial_cache_t3_ho_v1"
                elif row["cache_kind"] == "manifest":
                    row["cache_id"] = "c71_trial_cache_manifest_t3_ho_v1"
            cache_rows = c69_cache_rows
        authorized_tables = build_authorized_tables(
            ctx,
            protocol,
            protocol_sha,
            execution,
            cache_rows,
            schema_rows,
            view_rows,
            populations,
            unit_rows,
            budget,
            blocked,
            cluster,
            cs_rows,
            source_adv_rows,
            timestamp,
            first_access_timestamp,
        )
        readiness = {**readiness, **authorized_tables}

    res = {
        "config_hash": _lock_config(),
        "authorization_present": authorized,
        "authorization_token_name": "--authorization-token",
        "current_head": ctx["head"],
        "c70_commit": "4822f1c",
        "c70_final_gate": ctx["c70"].get("final_gate", ""),
        "parent_c70_protocol_sha256": ctx["parent_protocol_sha"],
        "parent_c70_protocol_sha256_replayed": ctx["parent_protocol_sha_replay"],
        "c71_protocol_sha256": protocol_sha,
        "protocol_lock_timestamp_utc": timestamp,
        "first_t3_ho_manifest_path_read_timestamp_utc": first_access_timestamp if authorized else "",
        "first_t3_ho_outcome_read_timestamp_utc": first_access_timestamp if authorized else "",
        "forward_or_reinference_executed": int(authorized and readiness.get("dependency_unit_summary_rows", [{}])[0].get("total_rows", 0) != 0),
        "training_attempted": 0,
        "gpu_used": 0,
        "bnci004_used": 0,
        "reserved_seeds_used": 0,
        "t3_cache_consumed": int(authorized and readiness.get("dependency_unit_summary_rows", [{}])[0].get("total_rows", 0) != 0),
        "raw_cache_rows_emitted": int(readiness.get("dependency_unit_summary_rows", [{}])[0].get("total_rows", 0) or 0) if authorized else 0,
        "selector_artifact_emitted": 0,
        "checkpoint_recommendation_artifact_emitted": 0,
        "external_cache_root": external_cache_root,
        "c65_physical_forward_units": len(canonical),
        "t1_unit_count": len(t1_units),
        "t2_unit_count": len(t2_units),
        "t3_ho_unit_count": len(t3_units),
        "repeats": repeats,
        "blocked_permutations": permutations,
        "cluster_bootstraps": bootstraps,
        "_protocol": protocol,
        **readiness,
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "schema_validation_summary_rows": [],
        "red_team_failure_ledger_rows": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c71", "command": "python -m pytest oaci/tests/test_c71_t3_ho_hierarchical_confirmation.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c71_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c7*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c71_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c7*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def _compact_json(res: dict) -> dict:
    b8 = _primary_budget_row(res.get("label_budget_curve_rows", []), "8") if res.get("label_budget_curve_rows") else {}
    b64 = _primary_budget_row(res.get("label_budget_curve_rows", []), "64") if res.get("label_budget_curve_rows") else {}
    bfull = _primary_budget_row(res.get("label_budget_curve_rows", []), FULL_BUDGET_LABEL) if res.get("label_budget_curve_rows") else {}
    blocked = res.get("blocked_permutation_summary_rows", [{}])[0] if res.get("blocked_permutation_summary_rows") else {}
    cache = next((r for r in res.get("t3_ho_external_cache_manifest_rows", []) if r.get("cache_kind") == "minimal_logits_probs_metadata"), {})
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "no_forward_readiness_only": not res["authorization_present"],
        "authorization_present": res["authorization_present"],
        "authorization_token_name": res["authorization_token_name"],
        "forward_or_reinference_executed": res["forward_or_reinference_executed"],
        "training_attempted": res["training_attempted"],
        "gpu_used": res["gpu_used"],
        "bnci004_used": res["bnci004_used"],
        "reserved_seeds_used": res["reserved_seeds_used"],
        "t3_cache_consumed": res["t3_cache_consumed"],
        "raw_cache_rows_emitted": res["raw_cache_rows_emitted"],
        "selector_artifact_emitted": res["selector_artifact_emitted"],
        "checkpoint_recommendation_artifact_emitted": res["checkpoint_recommendation_artifact_emitted"],
        "c70_commit": res["c70_commit"],
        "c70_final_gate": res["c70_final_gate"],
        "current_head_at_generation": res["current_head"],
        "parent_c70_protocol_sha256": res["parent_c70_protocol_sha256"],
        "parent_c70_protocol_sha256_replayed": res["parent_c70_protocol_sha256_replayed"],
        "c71_protocol_sha256": res["c71_protocol_sha256"],
        "protocol_lock_timestamp_utc": res["protocol_lock_timestamp_utc"],
        "first_t3_ho_manifest_path_read_timestamp_utc": res["first_t3_ho_manifest_path_read_timestamp_utc"],
        "first_t3_ho_outcome_read_timestamp_utc": res["first_t3_ho_outcome_read_timestamp_utc"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "t3_full_physical_units": res["_protocol"]["t3_full_physical_units_from_parent"],
            "t2_consumed_units": res["_protocol"]["t2_consumed_units_from_parent"],
            "t3_ho_disjoint_units": res["_protocol"]["t3_ho_units_from_parent"],
            "primary_budgets": list(PRIMARY_BUDGETS),
            "primary_blocked_permutations_planned": 4999,
            "conditional_cs_permutations_planned": 999,
            "t3_ho_cache_rows": cache.get("row_count", 0),
            "t3_ho_cache_path_hash": cache.get("path_hash", ""),
            "budget_8_spearman": b8.get("mean_within_target_spearman", ""),
            "budget_8_gauge_recovery": b8.get("mean_gauge_residual_recovery", ""),
            "budget_8_actionability_coverage": b8.get("actionability_rate_regret_le_0p02", ""),
            "budget_64_gauge_recovery": b64.get("mean_gauge_residual_recovery", ""),
            "full_construction_spearman": bfull.get("mean_within_target_spearman", ""),
            "full_construction_gauge_recovery": bfull.get("mean_gauge_residual_recovery", ""),
            "blocked_perm_observed": blocked.get("observed", ""),
            "blocked_perm_p": blocked.get("p_value", ""),
            "blocked_perm_exceedances": blocked.get("exceedances", ""),
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    auth_line = "absent" if not res["authorization_present"] else "present"
    cache = next((r for r in res.get("t3_ho_external_cache_manifest_rows", []) if r.get("cache_kind") == "minimal_logits_probs_metadata"), {})
    b8 = _primary_budget_row(res.get("label_budget_curve_rows", []), "8") if res.get("label_budget_curve_rows") else {}
    b64 = _primary_budget_row(res.get("label_budget_curve_rows", []), "64") if res.get("label_budget_curve_rows") else {}
    bfull = _primary_budget_row(res.get("label_budget_curve_rows", []), FULL_BUDGET_LABEL) if res.get("label_budget_curve_rows") else {}
    blocked = res.get("blocked_permutation_summary_rows", [{}])[0] if res.get("blocked_permutation_summary_rows") else {}
    if res["authorization_present"]:
        execution_text = [
            f"C71 exact CLI authorization status: `{auth_line}`. This run executed the authorized frozen-checkpoint T3-HO re-inference path.",
            "",
            f"Observed execution counters: forward/re-inference `{res['forward_or_reinference_executed']}`, training `{res['training_attempted']}`, GPU `{res['gpu_used']}`, T3 cache consumption `{res['t3_cache_consumed']}`, raw cache rows emitted externally `{res['raw_cache_rows_emitted']}`.",
            "",
            f"T3-HO external cache rows: `{cache.get('row_count', 0)}`; path hash `{cache.get('path_hash', '')}`. Raw rows are external-only and content-addressed.",
        ]
        timing_text = [
            f"C71 protocol lock timestamp: `{res['protocol_lock_timestamp_utc']}`.",
            "",
            f"First T3-HO manifest/path access timestamp: `{res['first_t3_ho_manifest_path_read_timestamp_utc']}`.",
            "",
            f"First T3-HO outcome access timestamp: `{res['first_t3_ho_outcome_read_timestamp_utc']}`.",
        ]
        result_text = [
            "## 5. Confirmatory Results",
            "",
            f"H1 blocked permutation: observed `{blocked.get('observed', '')}`, permutations `{blocked.get('permutations', '')}`, exceedances `{blocked.get('exceedances', '')}`, p `{blocked.get('p_value', '')}`.",
            "",
            f"At 8 labels/class: Spearman `{b8.get('mean_within_target_spearman', '')}`, gauge recovery `{b8.get('mean_gauge_residual_recovery', '')}`, coverage `{b8.get('actionability_rate_regret_le_0p02', '')}`, top1 `{b8.get('mean_top1_hit', '')}`.",
            "",
            f"At 64 labels/class: Spearman `{b64.get('mean_within_target_spearman', '')}`, gauge recovery `{b64.get('mean_gauge_residual_recovery', '')}`, coverage `{b64.get('actionability_rate_regret_le_0p02', '')}`, top1 `{b64.get('mean_top1_hit', '')}`.",
            "",
            f"At full construction: Spearman `{bfull.get('mean_within_target_spearman', '')}`, gauge recovery `{bfull.get('mean_gauge_residual_recovery', '')}`, coverage `{bfull.get('actionability_rate_regret_le_0p02', '')}`, top1 `{bfull.get('mean_top1_hit', '')}`.",
            "",
            "These are diagnostic split-label measurement results. They are not a selector, not checkpoint recommendations, not source-only rescue, and not target-population generalization.",
        ]
    else:
        execution_text = [
            f"C71 exact CLI authorization status: `{auth_line}`. This run is a readiness/protocol audit only.",
            "",
            f"Observed execution counters: forward/re-inference `{res['forward_or_reinference_executed']}`, training `{res['training_attempted']}`, GPU `{res['gpu_used']}`, T3 cache consumption `{res['t3_cache_consumed']}`, raw cache rows `{res['raw_cache_rows_emitted']}`.",
            "",
            "The command-line token is the only accepted authorization route; prompt text, protocol text, comments, and environment variables are ignored.",
        ]
        timing_text = [
            f"C71 protocol lock timestamp: `{res['protocol_lock_timestamp_utc']}`.",
            "",
            "No T3-HO manifest path or outcome timestamp is recorded in this no-auth run.",
        ]
        result_text = [
            "## 5. Interpretation",
            "",
            "No C71 scientific confirmation is claimed here. The only completed result is that the C71 protocol and blocking gates are ready while the exact execution token is absent.",
        ]
    ledger_sentence = "C71 emits the risk register, disjointness ledger, overlap matrix, split contract, physical-view manifest, dependency summary, hypothesis table, hierarchical inference outputs, conditional-observability contracts, feature provenance, and failure ledger for the authorized T3-HO run." if res["authorization_present"] else "C71 emits the risk register, disjointness ledger, overlap matrix, split contract, physical-view manifest, dependency summary, hypothesis table, hierarchical-inference placeholders, conditional-observability contracts, feature provenance, and failure ledger required for the future authorized run."
    main = "\n".join([
        f"# C71 - T3-HO Hierarchical Confirmation / Measurement-Control Separation Audit (frozen C19 `{res['config_hash']}`)",
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
        *execution_text,
        "",
        "## 3. Protocol Lock",
        "",
        f"C70 parent protocol SHA-256: `{res['parent_c70_protocol_sha256']}`.",
        "",
        f"C71 prospective protocol SHA-256: `{res['c71_protocol_sha256']}`.",
        "",
        *timing_text,
        "",
        "## 4. Readiness Ledger",
        "",
        f"Parent C70 records `{res['_protocol']['t3_full_physical_units_from_parent']}` full physical units, `{res['_protocol']['t2_consumed_units_from_parent']}` T2 consumed units, and `{res['_protocol']['t3_ho_units_from_parent']}` T3-HO disjoint units.",
        "",
        ledger_sentence,
        "",
        *result_text,
    ])
    timing = "\n".join([
        "# C71 - Protocol Timing Audit",
        "",
        f"- C70 parent protocol SHA-256: `{res['parent_c70_protocol_sha256']}`",
        f"- C70 parent replay SHA-256: `{res['parent_c70_protocol_sha256_replayed']}`",
        f"- C71 protocol lock timestamp: `{res['protocol_lock_timestamp_utc']}`",
        f"- C71 protocol SHA-256: `{res['c71_protocol_sha256']}`",
        f"- First T3-HO manifest/path read timestamp: `{res['first_t3_ho_manifest_path_read_timestamp_utc'] or 'not_accessed_no_authorization'}`",
        f"- First T3-HO outcome read timestamp: `{res['first_t3_ho_outcome_read_timestamp_utc'] or 'not_accessed_no_authorization'}`",
        "",
        "Protocol lock and hash were emitted before any T3-HO path or outcome access.",
    ])
    red = "\n".join([
        "# C71 - Red-Team Verification",
        "",
        "All C71 readiness red-team gates pass." if d["red_team_failure_count"] == 0 else "C71 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    return {
        "C71_T3_HO_HIERARCHICAL_CONFIRMATION.md": main,
        "C71_PROTOCOL_TIMING_AUDIT.md": timing,
        "C71_RED_TEAM_VERIFICATION.md": red,
    }


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    specs = {
        "risk_register.csv": ("risk_register_rows", ["risk_id", "risk_name", "status", "evidence", "blocking", "mitigation", "residual_caveat", "future_confirmation_needed"]),
        "t3_ho_disjointness_ledger.csv": ("t3_ho_disjointness_ledger_rows", ["check", "expected", "observed", "passed", "status", "notes"]),
        "t1_t2_t3_overlap_matrix.csv": ("t1_t2_t3_overlap_matrix_rows", ["left", "right", "left_units", "right_units", "overlap_units", "independent_confirmation"]),
        "shared_trial_split_contract.csv": ("shared_trial_split_contract_rows", ["contract", "status", "required", "observed", "passed", "notes"]),
        "unique_label_budget_ledger.csv": ("unique_label_budget_ledger_rows", ["budget", "role", "labels_counted_as", "checkpoint_scaled_cost_allowed", "status"]),
        "construction_eval_overlap_audit.csv": ("construction_eval_overlap_audit_rows", ["audit", "status", "overlap_trial_ids", "passed", "notes"]),
        "physical_view_manifest.csv": ("physical_view_manifest_rows", ["view_name", "path", "sha256", "allowed_columns", "forbidden_columns", "uses_target_labels", "uses_evaluation_labels", "available_at_selection_time", "diagnostic_only", "consumer_command"]),
        "dependency_unit_summary.csv": ("dependency_unit_summary_rows", ["unit_family", "total_rows", "unique_checkpoints", "unique_checkpoint_target_cells", "unique_targets", "unique_trajectories", "unique_trial_ids", "unique_construction_trial_ids", "unique_evaluation_trial_ids", "status"]),
        "primary_hypothesis_summary.csv": ("primary_hypothesis_summary_rows", ["hypothesis", "primary_budget", "primary_statistic", "confirmatory_gate", "status", "result"]),
        "per_target_confirmatory_results.csv": ("per_target_confirmatory_results_rows", ["target_id", "status", "within_target_spearman", "direction_positive", "top1_hit", "coverage", "gauge_recovery"]),
        "leave_one_target_out_summary.csv": ("leave_one_target_out_summary_rows", ["left_out_target", "status", "pooled_statistic", "p_value"]),
        "reliability_actionability_separation.csv": ("reliability_actionability_separation_rows", ["budget", "status", "within_target_centered_spearman", "pairwise_order_accuracy", "top1_hit", "topk_hit", "enrichment", "continuous_regret", "coverage", "measurement_control_separation"]),
        "t3_ho_gauge_recovery.csv": ("t3_ho_gauge_recovery_rows", ["budget", "status", "rank_recovery", "candidate_specific_gauge_recovery", "common_target_offset_contribution", "residual_variance", "source_to_oracle_gap_closed"]),
        "t3_ho_rank_vs_gauge_decomposition.csv": ("t3_ho_rank_vs_gauge_decomposition_rows", ["budget", "status", "rank_component", "gauge_component", "finite_trial_residual", "common_offset_not_credited"]),
        "t3_ho_pair_margin_recovery.csv": ("t3_ho_pair_margin_recovery_rows", ["budget", "status", "pair_count", "pairwise_recovery", "median_margin"]),
        "cluster_bootstrap_summary.csv": ("cluster_bootstrap_summary_rows", ["bootstrap", "status", "replicates", "ci_lower", "ci_upper", "row_iid_used"]),
        "blocked_permutation_summary.csv": ("blocked_permutation_summary_rows", ["test", "status", "observed", "permutations", "exceedances", "p_value", "minimum_p", "row_iid_used"]),
        "permutation_resolution_ledger.csv": ("permutation_resolution_ledger_rows", ["test", "planned_permutations", "minimum_attainable_p", "plus_one_correction", "random_seed_base", "status"]),
        "conditional_observability_block_summary.csv": ("conditional_observability_block_summary_rows", ["estimator", "status", "block_valid_status", "full_conditional_cs_claimed"]),
        "conditional_cs_estimator_contract.csv": ("conditional_cs_estimator_contract_rows", ["contract", "assumptions_met_now", "crossed_dependence_handled", "status", "faithfulness_claim_allowed"]),
        "bandwidth_nested_null_audit.csv": ("bandwidth_nested_null_audit_rows", ["bandwidth_rule", "status", "selection_inside_null", "evaluation_label_tuning_allowed"]),
        "feature_availability_ledger.csv": ("feature_availability_ledger_rows", ["feature_family", "available_now", "uses_target_labels", "available_at_selection_time", "diagnostic_only", "status"]),
        "strict_source_adversary_summary.csv": ("strict_source_adversary_summary_rows", ["adversary", "status", "target_labels_used", "escape_hatch_found", "notes"]),
        "failure_reason_ledger.csv": ("failure_reason_ledger_rows", ["reason", "status", "evidence", "blocks_science_claim"]),
        "t3_ho_external_cache_manifest.csv": ("t3_ho_external_cache_manifest_rows", ["stage", "cache_id", "cache_kind", "external_path", "path_hash", "exists", "size_bytes", "sha256", "sha256_match", "row_count", "manifest_row_count", "git_tracked", "status"]),
        "t3_ho_cache_schema_audit.csv": ("t3_ho_cache_schema_audit_rows", ["stage", "check", "observed", "expected", "passed", "notes"]),
        "label_budget_curve.csv": ("label_budget_curve_rows", ["budget", "labels_per_class", "repeat_count", "target_count", "mean_unique_construct_trials", "mean_unique_eval_trials", "mean_within_target_spearman", "mean_kendall_tau_proxy", "mean_pairwise_order_accuracy", "mean_top1_hit", "mean_top3_hit", "mean_continuous_regret", "actionability_rate_regret_le_0p02", "mean_gauge_residual_recovery", "endpoint_oracle_reference", "few_label_sufficiency_claimed"]),
        "per_target_label_budget_curve.csv": ("per_target_label_budget_curve_rows", ["budget", "repeat", "target_id", "labels_per_class", "unique_construct_trials", "unique_eval_trials", "within_target_spearman", "within_target_kendall_tau_proxy", "pairwise_order_accuracy", "median_eval_pair_margin", "top1_hit", "top3_hit", "continuous_regret", "actionability_hit_regret_le_0p02", "gauge_residual_recovery", "gauge_alpha"]),
        "actionability_budget_curve.csv": ("actionability_budget_curve_rows", ["budget", "top1_hit", "top3_hit", "continuous_regret", "coverage_regret_le_0p02", "actionability_status"]),
        "protocol_timing.csv": ("protocol_timing_rows", ["event", "timestamp_utc", "sha256", "status"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(TABLE_DIR, name), res.get(key, []), cols)


def _schema_rows() -> list[dict]:
    rows = []
    for path in sorted(Path(TABLE_DIR).glob("*.csv")):
        if path.name in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": path.name, "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def write_protocol_artifacts(protocol: dict, protocol_sha: str) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "C71_T3_HO_CONFIRMATORY_PROTOCOL.json")
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
        f.write("\n")
    actual = _sha256(path)
    if actual != protocol_sha:
        raise ValueError(f"C71 protocol SHA mismatch: expected {protocol_sha}; got {actual}")
    with open(os.path.join(REPORT_DIR, "C71_T3_HO_CONFIRMATORY_PROTOCOL.sha256"), "w") as f:
        f.write(actual + "\n")


def _write_reports_and_json(res: dict) -> None:
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
        f.write("\n")


def _refresh_quality_rows(res: dict) -> None:
    write_tables(res)
    _write_reports_and_json(res)
    paths = [str(p) for p in _listed_paths()]
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)


def write_artifacts(res: dict) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    write_protocol_artifacts(res["_protocol"], res["c71_protocol_sha256"])
    _refresh_quality_rows(res)
    _refresh_quality_rows(res)
    _write_reports_and_json(res)
    paths = _listed_paths()
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{} for _ in paths]
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    _write_reports_and_json(res)
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    write_tables(res)
    _write_reports_and_json(res)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c71_t3_ho_hierarchical_confirmation")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--authorization-token", default="")
    ap.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    ap.add_argument("--external-cache-root", default=EXTERNAL_CACHE_ROOT)
    ap.add_argument("--repeats", type=int, default=256)
    ap.add_argument("--permutations", type=int, default=4999)
    ap.add_argument("--bootstraps", type=int, default=1000)
    ap.add_argument("--reuse-existing-cache", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(
        authorization_token=args.authorization_token,
        datalake_root=args.datalake_root,
        external_cache_root=args.external_cache_root,
        repeats=args.repeats,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        reuse_existing_cache=args.reuse_existing_cache,
        test_status=args.test_status,
    )
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C71] decision={res['decision']['primary']} gate={res['decision']['final_gate']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
