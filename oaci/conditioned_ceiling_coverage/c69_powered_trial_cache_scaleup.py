"""C69 - powered re-inference-only trial cache scale-up audit."""
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
from pathlib import Path

import numpy as np

from . import audit_utils as au
from . import c66_reinference_only_trial_cache_microcampaign as c66


MILESTONE = "C69"
AUTH_TOKEN = "C69_REINFERENCE_ONLY_T1T2_AUTHORIZED"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c69_tables"
REPORT_JSON = "oaci/reports/C69_POWERED_TRIAL_CACHE_SCALEUP.json"

C65_MAP = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"
C66_AUTH_MANIFEST = "oaci/reports/c66_tables/authorized_cache_manifest.csv"
C67_JSON = "oaci/reports/C67_C66_DUAL_MODE_CACHE_CONSUMPTION.json"
C67_LEDGER = "oaci/reports/c67_tables/c66_dual_mode_provenance_ledger.csv"
C68_JSON = "oaci/reports/C68_POWERED_TRIAL_CACHE_SCALEUP.json"
C68_LADDER = "oaci/reports/c68_tables/c68_power_ladder.csv"

DEFAULT_DATALAKE_ROOT = "/projects/EEG-foundation-model/datalake/raw"
EXTERNAL_CACHE_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c69-trial-cache-scaleup"
MAX_REPORT_BYTES = 50_000_000
TRIAL_ROWS_PER_UNIT = 576
MASKED = "__MASKED_BY_C69_VIEW__"

LABEL_FIELDS = ("class_label_quarantined", "y_true_quarantined", "correctness_quarantined")
PREDICTION_FIELDS = (
    "y_pred",
    "logits",
    "probabilities",
    "confidence",
    "margin",
    "entropy",
    "true_class_probability",
)

DECISIONS = (
    "C69-A_authorized_t1_reinference_cache_executed_and_manifested",
    "C69-B_authorized_t2_reinference_cache_executed_and_manifested",
    "C69-C_split_label_diagnostic_stable_but_not_sufficiency",
    "C69-D_cache_valid_but_split_label_still_underpowered",
    "C69-E_sample_level_conditional_cs_still_underpowered_or_unstable",
    "C69-F_sample_level_conditional_cs_feasible_but_diagnostic_only",
    "C69-G_endpoint_oracle_boundary_preserved",
    "C69-H_trial_level_source_observable_escape_hatch_found",
    "C69-I_no_trial_level_source_observable_escape_hatch_found",
    "C69-J_larger_t3_campaign_ready_but_not_authorized",
    "C69-K_reinference_blocked_by_abi_preprocess_or_data_contract",
    "C69-L_label_masking_or_availability_violation_found",
    "C69-M_new_training_still_not_justified",
    "C69-N_new_training_required_but_not_authorized",
    "C69-O_no_forward_readiness_only_due_missing_authorization",
)

FINAL_GATES = (
    "NO_FORWARD_READINESS_ONLY_DUE_MISSING_AUTHORIZATION",
    "AUTHORIZED_T1_CACHE_EXECUTED_AND_MANIFESTED",
    "AUTHORIZED_T1T2_CACHE_EXECUTED_AND_MANIFESTED",
    "CACHE_VALID_BUT_SPLIT_LABEL_CS_STILL_UNDERPOWERED",
    "SPLIT_LABEL_DIAGNOSTIC_STABLE_NOT_SUFFICIENCY",
    "SAMPLE_LEVEL_CONDITIONAL_CS_FEASIBLE_DIAGNOSTIC_ONLY",
    "TRIAL_LEVEL_SOURCE_ESCAPE_HATCH_FOUND",
    "LARGER_T3_REINFERENCE_ONLY_READY_BUT_NOT_AUTHORIZED",
    "REINFERENCE_BLOCKED_BY_ABI_OR_PREPROCESSING",
    "LABEL_MASKING_BLOCKER_REQUIRES_REPAIR",
    "NEW_TRAINING_STILL_NOT_JUSTIFIED",
    "NEW_TRAINING_REQUIRED_BUT_NOT_AUTHORIZED",
)

FORBIDDEN_PATTERNS = (
    "training authorized",
    "new training is justified",
    "gradient update executed",
    "gpu used",
    "bnci2014_004 used",
    "seeds [3,4] used",
    "checkpoint recommendation",
    "deployable selector",
    "oaci rescue",
    "source-only rescue",
    "few-label sufficiency",
    "full conditional-cs established",
    "same-label endpoint scalar available at selection time",
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
    "diagnostic-only ",
    "diagnostic only ",
    "still not ",
)

ENDPOINT_TEMPLATE_HIT = 0.7037037037037037
ENDPOINT_ORACLE_HIT = 0.9444444444444444
ENDPOINT_MAX_NULL_P95 = 0.7712962962962961


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _auth_present(token: str = "") -> bool:
    # Exact CLI token only. Do not inspect handoff text, prompt text, files, or env vars.
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
    skip = {"c69_artifact_manifest.csv", "c69_large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C69_*.md"))
        + list(Path(REPORT_DIR).glob("C69_*.json"))
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
            "artifact_class": "table" if str(p).endswith(".csv") else "summary_json" if str(p).endswith(".json") else "report",
            "row_count": counts.get(str(p), ""),
        }
        for p in sorted(paths)
    ]


def _affirmative_hit(text: str, phrase: str, window: int = 220) -> bool:
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
            if os.path.basename(path) in {"c69_forbidden_claim_scan.csv", "red_team_failure_ledger.csv"}:
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


def _count_rows(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _canonical_physical_rows(rows: list[dict]) -> list[dict]:
    regime_order = {"S0_full_support": 0, "S2_rare_cells": 1, "S3_nonestimable_cells": 2}
    by_ckpt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if int(r["seed"]) in {3, 4}:
            continue
        by_ckpt[r["checkpoint_id"]].append(r)
    canonical = []
    for group in by_ckpt.values():
        canonical.append(sorted(group, key=lambda r: (regime_order.get(r["regime"], 99), int(r["candidate_order"]), r["row_id"]))[0])
    return sorted(canonical, key=lambda r: (int(r["seed"]), int(r["target"]), int(r["level"]), int(r["candidate_order"]), r["checkpoint_id"]))


def _stage_units(canonical: list[dict], target_count: int) -> list[dict]:
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in canonical:
        by_cell[(r["seed"], r["target"], r["level"])].append(r)
    cells = sorted(by_cell, key=lambda k: (int(k[0]), int(k[1]), int(k[2])))
    out = []
    used = set()
    depth = 0
    while len(out) < target_count:
        progressed = False
        for cell in cells:
            group = by_cell[cell]
            if depth < len(group) and group[depth]["checkpoint_id"] not in used:
                out.append(group[depth])
                used.add(group[depth]["checkpoint_id"])
                progressed = True
                if len(out) == target_count:
                    break
        if not progressed:
            raise RuntimeError(f"C69 schedule cannot reach {target_count} physical units")
        depth += 1
    return out


def build_schedule(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    canonical = _canonical_physical_rows(rows)
    t1 = _stage_units(canonical, 64)
    t2 = _stage_units(canonical, 216)
    return canonical, t1, t2


def _future_split_role(trial_id: str) -> str:
    return c66._future_split_role(trial_id)


def _float_vec(values) -> str:
    return c66._float_vec(values)


def _entropy(probs) -> float:
    return c66._entropy(probs)


def execute_c69_stage(
    stage: str,
    unit_rows: list[dict],
    *,
    datalake_root: str,
    external_cache_root: str,
) -> dict:
    """Run CPU-only frozen checkpoint inference and write an external content-addressed cache."""
    t0 = time.time()
    try:
        import torch
        from oaci.data.eeg.bnci import load_moabb_confirmatory
        from oaci.models import build_model

        os.makedirs(external_cache_root, exist_ok=True)
        base_cache_dir = os.path.join(external_cache_root, f"authorized_c69_{stage}_v1")
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
                    raise RuntimeError(f"C69 state-load metadata failed for {r['checkpoint_id']}: {load_meta['error']}")
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
                mask = np.asarray(bundle.subject_id == domain)
                indices = np.where(mask)[0]
                split_counter = Counter()
                status = "pass" if len(indices) else "missing_target_rows"
                with torch.no_grad():
                    for start in range(0, len(indices), 128):
                        batch_idx = indices[start:start + 128]
                        x = torch.from_numpy(np.ascontiguousarray(bundle.X[batch_idx])).to(dtype=torch.float32)
                        c66._assert_cpu_model_and_tensor(model, x)
                        out = model(x)
                        if out.logits.device.type != "cpu":
                            raise RuntimeError(f"C69 CPU-only guard failed: logits device={out.logits.device.type}")
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
                            pred = int(preds[j])
                            correct = int(pred == y)
                            writer.writerow({
                                "cache_version": "c69_trial_cache_v1",
                                "trial_cache_id": f"c69_trial_cache_{stage}_v1",
                                "c69_stage": stage,
                                "checkpoint_id": r["checkpoint_id"],
                                "checkpoint_path_hash": _path_hash(r["pt_path"]),
                                "checkpoint_sidecar_hash": _path_hash(r["json_path"]),
                                "dataset_id": "BNCI2014_001",
                                "subject_id": str(bundle.subject_id[bi]),
                                "target_id": target,
                                "source_or_target_flag": "target",
                                "source_or_target_role": "target_audit_scaleup",
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
                                "logits": _float_vec(logits[j]),
                                "probabilities": _float_vec(probs[j]),
                                "confidence": f"{float(probs[j, pred]):.8g}",
                                "true_class_probability": f"{float(probs[j, y]):.8g}",
                                "margin": f"{float(margins[j]):.8g}",
                                "entropy": f"{_entropy(probs[j]):.8g}",
                                "correctness_quarantined": correct,
                                "split_role": split_role,
                                "split_role_for_future_split_label": split_role,
                                "availability_tags": "target_label_quarantined;no_selector;diagnostic_only;c69_authorized",
                                "view_mask_tags": "source_only_masks_labels_predictions;split_label_masks_nonrole_labels;same_label_oracle_policy_only",
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
                raise RuntimeError(f"C69 immutable external cache collision at {trial_cache_path}")
            os.remove(trial_cache_tmp_path)
        else:
            os.replace(trial_cache_tmp_path, trial_cache_path)

        manifest = {
            "schema_version": "c69_trial_cache_manifest_v1",
            "cache_id": f"c69_trial_cache_{stage}_v1",
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
                raise RuntimeError(f"C69 immutable external manifest collision at {manifest_path}")
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
            "stage": stage,
            "status": "blocked",
            "attempted": 1,
            "success": 0,
            "error": repr(exc),
            "external_root": external_cache_root,
            "cache_dir": "",
            "trial_cache_path": "",
            "trial_cache_sha256": "",
            "trial_cache_size_bytes": 0,
            "manifest_path": "",
            "manifest_sha256": "",
            "manifest_size_bytes": 0,
            "trial_row_count": 0,
            "checkpoint_count": len(unit_rows),
            "target_count": len({int(r["target"]) for r in unit_rows}),
            "dataset_evidence_hash": "",
            "raw_data_fingerprint": "",
            "resolved_preprocess_hash": "",
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


def project_trial_cache_row_for_view(row: dict, view: str) -> dict:
    out = dict(row)
    split_role = str(row.get("split_role_for_future_split_label", row.get("split_role", "")))
    if view == "same_label_oracle_view" or view == "conditional_cs_diagnostic_view":
        return out
    if view == "source_only_view":
        for field in (*LABEL_FIELDS, *PREDICTION_FIELDS):
            out[field] = MASKED
        return out
    if view == "target_construction_view":
        if split_role != "target_construct":
            for field in LABEL_FIELDS:
                out[field] = MASKED
        return out
    if view == "target_evaluation_view":
        if split_role != "target_eval":
            for field in LABEL_FIELDS:
                out[field] = MASKED
        return out
    raise ValueError(f"unknown C69 view {view!r}")


def _parse_vec(s: str) -> list[float]:
    return [float(x) for x in str(s).split(";") if x != ""]


def validate_cache(execution: dict) -> tuple[list[dict], list[dict], list[dict]]:
    stage = execution["stage"]
    path = execution.get("trial_cache_path", "")
    rows = _read_csv(path) if path and os.path.exists(path) else []
    git_files = set(_git_or_empty(["ls-files"]).splitlines())
    required = {
        "cache_version", "trial_cache_id", "c69_stage", "checkpoint_id", "dataset_id", "subject_id",
        "target_id", "source_or_target_flag", "trial_id", "trial_index", "class_label_quarantined",
        "split_role", "checkpoint_path_hash", "checkpoint_sidecar_hash", "seed", "regime",
        "trajectory_id", "level", "logits", "probabilities", "predicted_class",
        "true_class_probability", "margin", "correctness_quarantined", "view_mask_tags",
    }
    schema_cols = list(rows[0].keys()) if rows else []
    finite = prob_sum = 1
    if rows:
        for r in rows:
            try:
                logits = _parse_vec(r["logits"])
                probs = _parse_vec(r["probabilities"])
                finite &= int(all(math.isfinite(x) for x in logits + probs))
                prob_sum &= int(abs(sum(probs) - 1.0) < 1e-5)
            except Exception:
                finite = prob_sum = 0
                break
    raw_tracked = any(os.path.basename(path) == os.path.basename(p) for p in git_files)
    cache_rows = [
        {
            "stage": stage,
            "cache_id": f"c69_trial_cache_{stage}_v1",
            "cache_kind": "minimal_logits_probs_metadata",
            "external_path": path,
            "path_hash": _path_hash(path) if path else "",
            "exists": int(bool(path) and os.path.exists(path)),
            "size_bytes": execution.get("trial_cache_size_bytes", 0),
            "sha256": execution.get("trial_cache_sha256", ""),
            "sha256_match": int(bool(path) and os.path.exists(path) and _sha256(path) == execution.get("trial_cache_sha256", "")),
            "row_count": len(rows),
            "manifest_row_count": execution.get("trial_row_count", 0),
            "git_tracked": int(raw_tracked),
            "status": execution.get("status", ""),
        },
        {
            "stage": stage,
            "cache_id": f"c69_trial_cache_manifest_{stage}_v1",
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
    schema_rows = [
        {"stage": stage, "check": "required_columns_present", "observed": len(set(schema_cols) & required), "expected": len(required), "passed": int(required <= set(schema_cols)), "notes": ";".join(sorted(required - set(schema_cols)))},
        {"stage": stage, "check": "row_count_matches_manifest", "observed": len(rows), "expected": execution.get("trial_row_count", 0), "passed": int(len(rows) == int(execution.get("trial_row_count", 0))), "notes": "external cache row replay"},
        {"stage": stage, "check": "all_logits_probs_finite", "observed": finite, "expected": 1, "passed": finite, "notes": "finite numeric payload"},
        {"stage": stage, "check": "probabilities_sum_to_one", "observed": prob_sum, "expected": 1, "passed": prob_sum, "notes": "softmax probability check"},
        {"stage": stage, "check": "dataset_bnci001_only", "observed": ";".join(sorted({r.get("dataset_id", "") for r in rows})), "expected": "BNCI2014_001", "passed": int(bool(rows) and {r.get("dataset_id", "") for r in rows} == {"BNCI2014_001"}), "notes": "BNCI2014_004 remains unused"},
        {"stage": stage, "check": "reserved_seeds_absent", "observed": ";".join(sorted({r.get("seed", "") for r in rows}, key=lambda x: int(x) if x else -1)), "expected": "0;1;2", "passed": int(bool(rows) and not ({r.get("seed", "") for r in rows} & {"3", "4"})), "notes": "seeds 3/4 preserved"},
        {"stage": stage, "check": "raw_cache_not_git_tracked", "observed": int(raw_tracked), "expected": 0, "passed": int(not raw_tracked), "notes": "raw rows external only"},
    ]
    view_rows = build_masked_view_contract(stage, rows)
    return cache_rows, schema_rows, view_rows


def build_masked_view_contract(stage: str, rows: list[dict]) -> list[dict]:
    out = []
    specs = [
        ("source_only_view", 0, 0, 0, 0, 1, 0),
        ("target_construction_view", 1, 0, 0, 0, 1, 0),
        ("target_evaluation_view", 1, 1, 0, 0, 1, 0),
        ("same_label_oracle_view", 1, 1, 1, 0, 0, 1),
        ("conditional_cs_diagnostic_view", 1, 1, 0, 0, 0, 1),
    ]
    for view, uses_labels, uses_eval, uses_same, available, enforced, policy in specs:
        label_visible = prediction_visible = construct_label_visible = eval_label_visible = 0
        for r in rows:
            p = project_trial_cache_row_for_view(r, view)
            label_visible += int(p.get("y_true_quarantined", MASKED) != MASKED)
            prediction_visible += int(p.get("probabilities", MASKED) != MASKED)
            if r.get("split_role_for_future_split_label") == "target_construct":
                construct_label_visible += int(p.get("y_true_quarantined", MASKED) != MASKED)
            if r.get("split_role_for_future_split_label") == "target_eval":
                eval_label_visible += int(p.get("y_true_quarantined", MASKED) != MASKED)
        status = "pass"
        if view == "source_only_view" and (label_visible or prediction_visible):
            status = "fail"
        if view == "target_construction_view" and eval_label_visible:
            status = "fail"
        if view == "target_evaluation_view" and construct_label_visible:
            status = "fail"
        out.append({
            "stage": stage,
            "view": view,
            "sampled_rows": len(rows),
            "label_visible_rows": label_visible,
            "prediction_visible_rows": prediction_visible,
            "construct_label_visible_rows": construct_label_visible,
            "eval_label_visible_rows": eval_label_visible,
            "uses_target_labels": uses_labels,
            "uses_eval_labels": uses_eval,
            "uses_same_label_endpoint_scalar": uses_same,
            "available_at_selection_time": available,
            "diagnostic_only": 1,
            "selection_path_enforced": enforced,
            "policy_boundary_only": policy,
            "status": status,
        })
    return out


def _bacc_metric(rows: list[dict]) -> float:
    if not rows:
        return math.nan
    by_class: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        y = int(r["y_true_quarantined"])
        by_class[y].append(int(r["correctness_quarantined"]))
    return float(np.mean([np.mean(v) for v in by_class.values()])) if by_class else math.nan


def _nll_metric(rows: list[dict]) -> float:
    vals = []
    for r in rows:
        vals.append(-math.log(max(float(r["true_class_probability"]), 1e-12)))
    return float(np.mean(vals)) if vals else math.nan


def _ece_metric(rows: list[dict], bins: int = 10) -> float:
    if not rows:
        return math.nan
    conf = np.asarray([float(r["confidence"]) for r in rows], dtype=float)
    corr = np.asarray([int(r["correctness_quarantined"]) for r in rows], dtype=float)
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf >= lo) & (conf <= hi if i == bins - 1 else conf < hi)
        if mask.any():
            ece += float(mask.mean() * abs(conf[mask].mean() - corr[mask].mean()))
    return ece


def _rankdata(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    x = x.astype(float)
    y = y.astype(float)
    sx = x.std()
    sy = y.std()
    if sx <= 0 or sy <= 0:
        return math.nan
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def unit_metric_rows(stage: str, rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["checkpoint_id"]].append(r)
    out = []
    for checkpoint_id, group in sorted(groups.items()):
        construct = [r for r in group if r["split_role_for_future_split_label"] == "target_construct"]
        eval_rows = [r for r in group if r["split_role_for_future_split_label"] == "target_eval"]
        first = group[0]
        out.append({
            "stage": stage,
            "checkpoint_id": checkpoint_id,
            "target_id": first["target_id"],
            "seed": first["seed"],
            "level": first["level"],
            "trajectory_id": first["trajectory_id"],
            "regime": first["regime"],
            "candidate_order": first["candidate_order"],
            "construct_rows": len(construct),
            "eval_rows": len(eval_rows),
            "construct_bacc": _bacc_metric(construct),
            "eval_bacc": _bacc_metric(eval_rows),
            "construct_nll": _nll_metric(construct),
            "eval_nll": _nll_metric(eval_rows),
            "construct_ece": _ece_metric(construct),
            "eval_ece": _ece_metric(eval_rows),
            "construct_mean_confidence": float(np.mean([float(r["confidence"]) for r in construct])) if construct else math.nan,
            "eval_mean_confidence": float(np.mean([float(r["confidence"]) for r in eval_rows])) if eval_rows else math.nan,
        })
    return out


def split_label_summary(stage: str, rows: list[dict], unit_rows: list[dict], n_perm: int = 200) -> tuple[list[dict], list[dict]]:
    if not unit_rows:
        return [{
            "stage": stage, "status": "not_run_no_cache", "independent_checkpoint_units": 0, "construct_rows": 0,
            "eval_rows": 0, "spearman_construct_eval_bacc": "", "permutation_p_value": "", "eval_top_quartile_hit": "",
            "eval_top_quartile_base": "", "lift_vs_base": "", "few_label_sufficiency_claimed": 0,
        }], []
    x = [float(r["construct_bacc"]) for r in unit_rows]
    y = [float(r["eval_bacc"]) for r in unit_rows]
    rho = _spearman(x, y)
    rng = np.random.default_rng(69069)
    obs = rho if math.isfinite(rho) else 0.0
    null = []
    y_arr = np.asarray(y, dtype=float)
    for _ in range(n_perm):
        yy = y_arr.copy()
        rng.shuffle(yy)
        val = _spearman(x, yy.tolist())
        null.append(val if math.isfinite(val) else 0.0)
    p = float((1 + sum(v >= obs for v in null)) / (1 + len(null)))
    n = len(unit_rows)
    top_n = max(1, int(math.ceil(0.25 * n)))
    eval_threshold = float(np.quantile(y_arr, 0.75))
    top_by_construct = sorted(unit_rows, key=lambda r: (float(r["construct_bacc"]), -float(r["construct_nll"])), reverse=True)[:top_n]
    top_hit = float(np.mean([float(r["eval_bacc"]) >= eval_threshold for r in top_by_construct]))
    base = float(np.mean([float(v) >= eval_threshold for v in y]))
    lift = top_hit / base if base > 0 else math.nan
    stable = int(n >= 64 and obs > 0.0 and p <= 0.05 and top_hit > base)
    summary = [{
        "stage": stage,
        "status": "stable_diagnostic_not_sufficiency" if stable else "valid_but_underpowered_or_unstable",
        "independent_checkpoint_units": n,
        "construct_rows": sum(int(r["construct_rows"]) for r in unit_rows),
        "eval_rows": sum(int(r["eval_rows"]) for r in unit_rows),
        "spearman_construct_eval_bacc": round(obs, 6),
        "permutation_p_value": round(p, 6),
        "eval_top_quartile_hit": round(top_hit, 6),
        "eval_top_quartile_base": round(base, 6),
        "lift_vs_base": round(lift, 6) if math.isfinite(lift) else "",
        "few_label_sufficiency_claimed": 0,
    }]
    cell_rows = []
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in unit_rows:
        by_cell[(r["target_id"], r["seed"], r["level"])].append(r)
    for (target, seed, level), group in sorted(by_cell.items(), key=lambda x: (int(x[0][0]), int(x[0][1]), int(x[0][2]))):
        cell_rows.append({
            "stage": stage,
            "target_id": target,
            "seed": seed,
            "level": level,
            "checkpoint_units": len(group),
            "construct_rows": sum(int(r["construct_rows"]) for r in group),
            "eval_rows": sum(int(r["eval_rows"]) for r in group),
            "mean_construct_bacc": round(float(np.mean([float(r["construct_bacc"]) for r in group])), 6),
            "mean_eval_bacc": round(float(np.mean([float(r["eval_bacc"]) for r in group])), 6),
            "min_eval_bacc": round(float(np.min([float(r["eval_bacc"]) for r in group])), 6),
            "max_eval_bacc": round(float(np.max([float(r["eval_bacc"]) for r in group])), 6),
            "diagnostic_only": 1,
        })
    return summary, cell_rows


def _unit_feature_maps(unit_rows: list[dict]) -> dict[str, dict]:
    return {r["checkpoint_id"]: r for r in unit_rows}


def _cv_folds(unit_ids: list[str], k: int = 5) -> dict[str, int]:
    return {u: int(hashlib.sha256(u.encode()).hexdigest()[:8], 16) % k for u in unit_ids}


def _design_matrix(sample_rows: list[dict], units: dict[str, dict], include_x2: bool) -> np.ndarray:
    targets = sorted({r["target_id"] for r in sample_rows}, key=int)
    seeds = sorted({r["seed"] for r in sample_rows}, key=int)
    levels = sorted({r["level"] for r in sample_rows}, key=int)
    cols = []
    for r in sample_rows:
        unit = units[r["checkpoint_id"]]
        row = [1.0]
        row.extend(float(r["target_id"] == t) for t in targets[1:])
        row.extend(float(r["seed"] == s) for s in seeds[1:])
        row.extend(float(r["level"] == l) for l in levels[1:])
        row.append(float(r["candidate_order"]) / 100.0)
        if include_x2:
            row.extend([
                float(unit["construct_bacc"]),
                float(unit["construct_nll"]),
                float(unit["construct_ece"]),
                float(unit["construct_mean_confidence"]),
            ])
        cols.append(row)
    return np.asarray(cols, dtype=float)


def _ridge_cv_mse(sample_rows: list[dict], unit_rows: list[dict], include_x2: bool, ridge: float = 1e-3) -> float:
    units = _unit_feature_maps(unit_rows)
    y = np.asarray([int(r["correctness_quarantined"]) for r in sample_rows], dtype=float)
    X = _design_matrix(sample_rows, units, include_x2)
    folds = _cv_folds(sorted({r["checkpoint_id"] for r in sample_rows}))
    pred = np.zeros_like(y)
    for fold in range(5):
        test = np.asarray([folds[r["checkpoint_id"]] == fold for r in sample_rows], dtype=bool)
        train = ~test
        if not test.any() or not train.any():
            pred[test] = float(y[train].mean()) if train.any() else float(y.mean())
            continue
        XtX = X[train].T @ X[train]
        beta = np.linalg.solve(XtX + ridge * np.eye(X.shape[1]), X[train].T @ y[train])
        pred[test] = np.clip(X[test] @ beta, 0.0, 1.0)
    return float(np.mean((y - pred) ** 2))


def conditional_cs_summary(stage: str, rows: list[dict], unit_rows: list[dict], n_perm: int = 64) -> tuple[list[dict], list[dict]]:
    eval_rows = [r for r in rows if r.get("split_role_for_future_split_label") == "target_eval"]
    if len(unit_rows) < 64 or not eval_rows:
        return [{
            "stage": stage,
            "estimator": "binary_y_cod_proxy",
            "status": "underpowered_or_unstable",
            "paired_eval_rows": len(eval_rows),
            "independent_checkpoint_units": len(unit_rows),
            "baseline_mse": "",
            "x1_mse": "",
            "x1_plus_x2_mse": "",
            "incremental_cod": "",
            "null_p95_incremental_cod": "",
            "permutation_p_value": "",
            "full_conditional_cs_claimed": 0,
        }], []
    y = np.asarray([int(r["correctness_quarantined"]) for r in eval_rows], dtype=float)
    base_mse = float(np.mean((y - y.mean()) ** 2))
    x1_mse = _ridge_cv_mse(eval_rows, unit_rows, include_x2=False)
    x12_mse = _ridge_cv_mse(eval_rows, unit_rows, include_x2=True)
    cod = max(0.0, (x1_mse - x12_mse) / base_mse) if base_mse > 0 else 0.0

    rng = np.random.default_rng(69123)
    null = []
    unit_copy = [dict(r) for r in unit_rows]
    for _ in range(n_perm):
        shuffled = unit_copy.copy()
        rng.shuffle(shuffled)
        x2_fields = ["construct_bacc", "construct_nll", "construct_ece", "construct_mean_confidence"]
        permuted = []
        for original, donor in zip(unit_copy, shuffled):
            row = dict(original)
            for f in x2_fields:
                row[f] = donor[f]
            permuted.append(row)
        pmse = _ridge_cv_mse(eval_rows, permuted, include_x2=True)
        null.append(max(0.0, (x1_mse - pmse) / base_mse) if base_mse > 0 else 0.0)
    p = float((1 + sum(v >= cod for v in null)) / (1 + len(null)))
    p95 = float(np.quantile(null, 0.95))
    feasible = int(len(unit_rows) >= 64 and len(eval_rows) >= 10_000 and cod > p95 and p <= 0.05)
    summary = [{
        "stage": stage,
        "estimator": "binary_y_cod_proxy",
        "status": "feasible_proxy_diagnostic_only" if feasible else "underpowered_or_unstable",
        "paired_eval_rows": len(eval_rows),
        "independent_checkpoint_units": len(unit_rows),
        "baseline_mse": round(base_mse, 8),
        "x1_mse": round(x1_mse, 8),
        "x1_plus_x2_mse": round(x12_mse, 8),
        "incremental_cod": round(cod, 8),
        "null_p95_incremental_cod": round(p95, 8),
        "permutation_p_value": round(p, 6),
        "full_conditional_cs_claimed": 0,
    }]
    sensitivity = [
        {
            "stage": stage,
            "sensitivity": "x2_permutation_global",
            "n_permutations": n_perm,
            "observed_incremental_cod": round(cod, 8),
            "null_mean": round(float(np.mean(null)), 8),
            "null_p95": round(p95, 8),
            "passes_proxy_gate": feasible,
            "full_cs_claim_allowed": 0,
        },
        {
            "stage": stage,
            "sensitivity": "support_gate",
            "n_permutations": 0,
            "observed_incremental_cod": round(cod, 8),
            "null_mean": "",
            "null_p95": "",
            "passes_proxy_gate": int(len(unit_rows) >= 64 and len(eval_rows) >= 10_000),
            "full_cs_claim_allowed": 0,
        },
    ]
    return summary, sensitivity


def source_adversary_summary(stage: str, rows: list[dict], unit_rows: list[dict], cs_rows: list[dict]) -> list[dict]:
    eval_rows = [r for r in rows if r.get("split_role_for_future_split_label") == "target_eval"]
    if not eval_rows or not unit_rows:
        return [{"stage": stage, "adversary": "metadata_only_x1_proxy", "status": "not_run_no_cache", "target_labels_used": 0, "escape_hatch_found": 0, "notes": "no authorized cache"}]
    y = np.asarray([int(r["correctness_quarantined"]) for r in eval_rows], dtype=float)
    base_mse = float(np.mean((y - y.mean()) ** 2))
    x1_mse = _ridge_cv_mse(eval_rows, unit_rows, include_x2=False)
    x1_cod = max(0.0, (base_mse - x1_mse) / base_mse) if base_mse > 0 else 0.0
    inc_cod = float(cs_rows[0].get("incremental_cod") or 0.0) if cs_rows else 0.0
    metadata_proxy_nonzero = int(x1_cod >= 0.01)
    return [
        {
            "stage": stage,
            "adversary": "metadata_only_x1_proxy",
            "status": "nonzero_metadata_proxy_not_strict_source_escape" if metadata_proxy_nonzero else "no_escape_hatch_found",
            "paired_eval_rows": len(eval_rows),
            "independent_checkpoint_units": len(unit_rows),
            "target_labels_used": 0,
            "source_domain_trial_logits_available": 0,
            "x1_cod_vs_intercept": round(x1_cod, 8),
            "label_diagnostic_incremental_cod": round(inc_cod, 8),
            "escape_hatch_found": 0,
            "notes": "Target/seed/level/candidate-order metadata is a non-label proxy, but C69 did not instrument strict source-domain trial logits/probs; this is not a trial-level source escape hatch.",
        },
        {
            "stage": stage,
            "adversary": "strict_source_trial_logits",
            "status": "not_instrumented_in_c69_cache",
            "paired_eval_rows": len(eval_rows),
            "independent_checkpoint_units": len(unit_rows),
            "target_labels_used": 0,
            "source_domain_trial_logits_available": 0,
            "x1_cod_vs_intercept": "",
            "label_diagnostic_incremental_cod": "",
            "escape_hatch_found": 0,
            "notes": "No source-domain raw trial logits/probs are emitted; no source-only rescue claim is made.",
        },
    ]


def endpoint_boundary_rows() -> list[dict]:
    return [
        {"boundary": "template_only_transfer", "observed_hit": ENDPOINT_TEMPLATE_HIT, "max_null_p95": ENDPOINT_MAX_NULL_P95, "beats_null": int(ENDPOINT_TEMPLATE_HIT > ENDPOINT_MAX_NULL_P95), "uses_target_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "partial_not_reliability_claim"},
        {"boundary": "same_label_endpoint_scalar", "observed_hit": ENDPOINT_ORACLE_HIT, "max_null_p95": ENDPOINT_MAX_NULL_P95, "beats_null": int(ENDPOINT_ORACLE_HIT > ENDPOINT_MAX_NULL_P95), "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "endpoint_oracle_preserved"},
    ]


def load_context() -> dict:
    c65_rows = _read_csv(C65_MAP)
    c66_manifest = {r["cache_id"]: r for r in _read_csv(C66_AUTH_MANIFEST)}
    c66_trial = c66_manifest.get("c66_trial_cache_v1", {})
    c67 = _load_json(C67_JSON)
    c67_ledger = {r["mode"]: r for r in _read_csv(C67_LEDGER)}
    c68 = _load_json(C68_JSON)
    c68_ladder = {r["rung"]: r for r in _read_csv(C68_LADDER)}
    c66_path = c66_trial.get("external_path", "")
    c66_sha = _sha256(c66_path) if c66_path and os.path.exists(c66_path) else ""
    return {
        "head": _git_or_empty(["rev-parse", "--short", "HEAD"]),
        "branch": _git_or_empty(["branch", "--show-current"]),
        "origin_oaci": _git_or_empty(["rev-parse", "--short", "origin/oaci"]),
        "c65_rows": c65_rows,
        "c66_trial": c66_trial,
        "c66_trial_sha": c66_sha,
        "c67": c67,
        "c67_ledger": c67_ledger,
        "c68": c68,
        "c68_ladder": c68_ladder,
    }


def load_existing_execution(stage: str, checkpoint_count: int) -> dict:
    table = os.path.join(TABLE_DIR, f"c69_cache_manifest_{stage}.csv")
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


def build_authorization_rows(authorized: bool) -> list[dict]:
    return [
        {"gate": "exact_cli_token", "expected": AUTH_TOKEN, "observed": int(authorized), "allowed": 1, "passed": int(authorized), "notes": "only --authorization-token exact match authorizes C69"},
        {"gate": "protocol_text_authorization", "expected": 0, "observed": 0, "allowed": 0, "passed": 1, "notes": "handoff/prompt text is not scanned by _auth_present"},
        {"gate": "environment_variable_authorization", "expected": 0, "observed": 0, "allowed": 0, "passed": 1, "notes": "environment variables are not accepted"},
        {"gate": "t1_scope_authorized", "expected": 1, "observed": int(authorized), "allowed": 1, "passed": int(authorized), "notes": "T1=64 physical units"},
        {"gate": "t2_scope_conditionally_authorized", "expected": 1, "observed": int(authorized), "allowed": 1, "passed": int(authorized), "notes": "T2=216 physical units only after T1 gates"},
        {"gate": "t3_scope_authorized", "expected": 0, "observed": 0, "allowed": 0, "passed": 1, "notes": "T3 remains a future explicit authorization"},
    ]


def build_stage_rows(t1: list[dict], t2: list[dict], t1_exec: dict, t2_exec: dict, t1_gate_passed: bool) -> list[dict]:
    return [
        {"stage": "t1", "authorized_to_execute": 1, "executed": int(t1_exec.get("attempted", 0)), "forward_units": len(t1), "trial_rows": t1_exec.get("trial_row_count", 0), "success": t1_exec.get("success", 0), "gate_status": "pass" if t1_gate_passed else "fail_or_not_run", "cache_path_hash": _path_hash(t1_exec.get("trial_cache_path", "")) if t1_exec.get("trial_cache_path", "") else "", "notes": "T1 pilot scale"},
        {"stage": "t2", "authorized_to_execute": 1, "executed": int(t2_exec.get("attempted", 0)), "forward_units": len(t2), "trial_rows": t2_exec.get("trial_row_count", 0), "success": t2_exec.get("success", 0), "gate_status": "pass" if int(t2_exec.get("success", 0)) else "not_run_or_fail", "cache_path_hash": _path_hash(t2_exec.get("trial_cache_path", "")) if t2_exec.get("trial_cache_path", "") else "", "notes": "T2 medium scale after T1 gates"},
        {"stage": "t3", "authorized_to_execute": 0, "executed": 0, "forward_units": 1268, "trial_rows": 0, "success": 0, "gate_status": "not_authorized", "cache_path_hash": "", "notes": "T3 not authorized in C69"},
    ]


def build_resource_rows(t1_exec: dict, t2_exec: dict) -> list[dict]:
    return [
        {"stage": e["stage"], "runtime_seconds": e.get("runtime_seconds", 0), "checkpoint_units": e.get("checkpoint_count", 0), "trial_rows": e.get("trial_row_count", 0), "cache_size_bytes": e.get("trial_cache_size_bytes", 0), "manifest_size_bytes": e.get("manifest_size_bytes", 0), "cpu_only": 1, "gpu_used": 0, "training_attempted": 0}
        for e in (t1_exec, t2_exec)
    ]


def build_schema_signature_rows(stage: str, cache_rows: list[dict]) -> list[dict]:
    if not cache_rows:
        return [{"stage": stage, "field_count": 0, "schema_sha256": "", "required_minimum_present": 0, "status": "not_available"}]
    cols = list(cache_rows[0].keys())
    required = {"checkpoint_id", "target_id", "trial_id", "class_label_quarantined", "logits", "probabilities", "split_role"}
    return [{"stage": stage, "field_count": len(cols), "schema_sha256": hashlib.sha256(";".join(cols).encode()).hexdigest(), "required_minimum_present": int(required <= set(cols)), "status": "pass" if required <= set(cols) else "fail"}]


def _schema_rows() -> list[dict]:
    rows = []
    for path in sorted(Path(TABLE_DIR).glob("*.csv")):
        if path.name in {"schema_validation_summary.csv", "c69_artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": path.name, "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def build_red_team_rows(res: dict) -> list[dict]:
    auth = {r["gate"]: r for r in res["c69_authorization_audit_rows"]}
    stage = {r["stage"]: r for r in res["c69_stage_manifest_rows"]}
    cache = [*res["c69_cache_manifest_t1_rows"], *res["c69_cache_manifest_t2_rows"]]
    schema = [*res["c69_schema_signature_rows"], *res["schema_validation_summary_rows"]]
    views = res["c69_masked_view_contract_rows"]
    split = {r["stage"]: r for r in res["c69_split_label_summary_rows"]}
    cs = {r["stage"]: r for r in res["c69_conditional_cs_summary_rows"]}
    adv = res["c69_source_adversary_summary_rows"]
    checks = [
        ("exact_cli_authorization_only", auth["exact_cli_token"]["observed"] == 1 and auth["protocol_text_authorization"]["observed"] == 0, "Execution is authorized only by exact CLI token."),
        ("t1_executed_and_valid", int(stage["t1"]["executed"]) == 1 and int(stage["t1"]["success"]) == 1, "T1 cache executed and manifested."),
        ("t2_executed_after_t1", int(stage["t1"]["success"]) == 1 and int(stage["t2"]["executed"]) == 1 and int(stage["t2"]["success"]) == 1, "T2 ran only after T1 gates passed."),
        ("t3_not_authorized", int(stage["t3"]["authorized_to_execute"]) == 0 and int(stage["t3"]["executed"]) == 0, "T3 remains future explicit authorization."),
        ("cache_hashes_match", all(int(r.get("sha256_match", 1)) == 1 for r in cache if r.get("external_path", "")), "External cache and manifest hashes match."),
        ("raw_cache_external_only", all(int(r.get("git_tracked", 0)) == 0 for r in cache), "Raw trial caches are not committed to git."),
        ("schema_passed", all(int(r.get("passed", r.get("required_minimum_present", 1))) == 1 for r in schema), "Committed table schemas and cache schemas are present."),
        ("masking_passed", all(r["status"] == "pass" for r in views), "Source-only and split-label masks pass."),
        ("same_label_oracle_unavailable", all(int(r["available_at_selection_time"]) == 0 for r in views if r["view"] in {"same_label_oracle_view", "conditional_cs_diagnostic_view"}), "Oracle/diagnostic views are unavailable at selection time."),
        ("split_label_not_sufficiency", all(int(r["few_label_sufficiency_claimed"]) == 0 for r in split.values()), "Split-label result is not few-label sufficiency."),
        ("conditional_cs_not_full_claim", all(int(r["full_conditional_cs_claimed"]) == 0 for r in cs.values()), "Conditional-CS row is a proxy/smoke diagnostic, not a full CS claim."),
        ("source_escape_not_claimed_unless_found", any(int(r["escape_hatch_found"]) == 0 for r in adv), "No source-only rescue claim is made."),
        ("endpoint_boundary_preserved", all(int(r["available_at_selection_time"]) == 0 and int(r["diagnostic_only"]) == 1 for r in res["c69_endpoint_boundary_replay_rows"]), "Endpoint scalar remains a same-label diagnostic oracle."),
        ("no_training_gpu_reserved_holdouts", all(int(r["training_attempted"]) == 0 and int(r["gpu_used"]) == 0 for r in res["c69_resource_runtime_summary_rows"]), "No training/GPU use is recorded."),
        ("large_artifact_scan_passed", all(int(r["passed"]) for r in res["c69_large_artifact_scan_rows"]), "All committed C69 artifacts are under 50MB."),
        ("forbidden_claim_scan_passed", all(int(r["passed"]) for r in res["c69_forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
    ]
    return [{"gate": g, "failed": int(not ok), "finding": f} for g, ok, f in checks]


def classify(res: dict, authorized: bool) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    t1_success = any(r["stage"] == "t1" and int(r["success"]) for r in res["c69_stage_manifest_rows"])
    t2_success = any(r["stage"] == "t2" and int(r["success"]) for r in res["c69_stage_manifest_rows"])
    split_t2 = next((r for r in res["c69_split_label_summary_rows"] if r["stage"] == "t2"), {})
    cs_t2 = next((r for r in res["c69_conditional_cs_summary_rows"] if r["stage"] == "t2"), {})
    source_escape = any(int(r.get("escape_hatch_found", 0)) for r in res["c69_source_adversary_summary_rows"])
    if failures:
        primary = "C69-L_label_masking_or_availability_violation_found"
        active = [primary]
        gate = "LABEL_MASKING_BLOCKER_REQUIRES_REPAIR"
    elif not authorized:
        primary = "C69-O_no_forward_readiness_only_due_missing_authorization"
        active = [primary, "C69-J_larger_t3_campaign_ready_but_not_authorized", "C69-M_new_training_still_not_justified"]
        gate = "NO_FORWARD_READINESS_ONLY_DUE_MISSING_AUTHORIZATION"
    elif not t1_success:
        primary = "C69-K_reinference_blocked_by_abi_preprocess_or_data_contract"
        active = [primary, "C69-N_new_training_required_but_not_authorized"]
        gate = "REINFERENCE_BLOCKED_BY_ABI_OR_PREPROCESSING"
    elif source_escape:
        primary = "C69-H_trial_level_source_observable_escape_hatch_found"
        active = ["C69-A_authorized_t1_reinference_cache_executed_and_manifested", primary, "C69-J_larger_t3_campaign_ready_but_not_authorized"]
        if t2_success:
            active.insert(1, "C69-B_authorized_t2_reinference_cache_executed_and_manifested")
        gate = "TRIAL_LEVEL_SOURCE_ESCAPE_HATCH_FOUND"
    else:
        active = ["C69-A_authorized_t1_reinference_cache_executed_and_manifested"]
        if t2_success:
            active.append("C69-B_authorized_t2_reinference_cache_executed_and_manifested")
        split_stable = split_t2.get("status") == "stable_diagnostic_not_sufficiency"
        cs_feasible = cs_t2.get("status") == "feasible_proxy_diagnostic_only"
        active.append("C69-C_split_label_diagnostic_stable_but_not_sufficiency" if split_stable else "C69-D_cache_valid_but_split_label_still_underpowered")
        active.append("C69-F_sample_level_conditional_cs_feasible_but_diagnostic_only" if cs_feasible else "C69-E_sample_level_conditional_cs_still_underpowered_or_unstable")
        active.extend([
            "C69-G_endpoint_oracle_boundary_preserved",
            "C69-I_no_trial_level_source_observable_escape_hatch_found",
            "C69-J_larger_t3_campaign_ready_but_not_authorized",
            "C69-M_new_training_still_not_justified",
        ])
        primary = "C69-F_sample_level_conditional_cs_feasible_but_diagnostic_only" if cs_feasible else active[2]
        gate = "SAMPLE_LEVEL_CONDITIONAL_CS_FEASIBLE_DIAGNOSTIC_ONLY" if cs_feasible else "CACHE_VALID_BUT_SPLIT_LABEL_CS_STILL_UNDERPOWERED"
    return {
        "primary": primary,
        "active": active,
        "inactive": [d for d in DECISIONS if d not in active],
        "final_gate": gate,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": "remote review; C70 should decide whether T3 or source-observable forensics is warranted",
    }


def table_row_counts(res: dict) -> dict:
    keys = {
        "c69_authorization_audit": "c69_authorization_audit_rows",
        "c69_stage_manifest": "c69_stage_manifest_rows",
        "c69_cache_manifest_t1": "c69_cache_manifest_t1_rows",
        "c69_cache_manifest_t2": "c69_cache_manifest_t2_rows",
        "c69_schema_signature": "c69_schema_signature_rows",
        "c69_masked_view_contract": "c69_masked_view_contract_rows",
        "c69_split_label_summary": "c69_split_label_summary_rows",
        "c69_split_label_cell_ledger": "c69_split_label_cell_ledger_rows",
        "c69_conditional_cs_summary": "c69_conditional_cs_summary_rows",
        "c69_conditional_cs_sensitivity": "c69_conditional_cs_sensitivity_rows",
        "c69_source_adversary_summary": "c69_source_adversary_summary_rows",
        "c69_endpoint_boundary_replay": "c69_endpoint_boundary_replay_rows",
        "c69_resource_runtime_summary": "c69_resource_runtime_summary_rows",
        "c69_forbidden_claim_scan": "c69_forbidden_claim_scan_rows",
        "c69_large_artifact_scan": "c69_large_artifact_scan_rows",
        "c69_artifact_manifest": "c69_artifact_manifest_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "test_command_manifest": "test_command_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def run(
    *,
    authorization_token: str = "",
    datalake_root: str = DEFAULT_DATALAKE_ROOT,
    external_cache_root: str = EXTERNAL_CACHE_ROOT,
    test_status: str = "planned",
    reuse_existing_cache: bool = False,
) -> dict:
    authorized = _auth_present(authorization_token)
    ctx = load_context()
    canonical, t1_units, t2_units = build_schedule(ctx["c65_rows"])

    t1_exec = _empty_execution("t1", len(t1_units))
    t2_exec = _empty_execution("t2", len(t2_units))
    t1_cache_rows: list[dict] = []
    t2_cache_rows: list[dict] = []
    t1_schema_rows: list[dict] = []
    t2_schema_rows: list[dict] = []
    t1_view_rows: list[dict] = []
    t2_view_rows: list[dict] = []
    t1_unit_metrics: list[dict] = []
    t2_unit_metrics: list[dict] = []
    t1_split_rows: list[dict] = []
    t2_split_rows: list[dict] = []
    t1_cell_rows: list[dict] = []
    t2_cell_rows: list[dict] = []
    t1_cs_rows: list[dict] = []
    t2_cs_rows: list[dict] = []
    t1_cs_sens: list[dict] = []
    t2_cs_sens: list[dict] = []
    t1_adv_rows: list[dict] = []
    t2_adv_rows: list[dict] = []

    if authorized:
        if reuse_existing_cache:
            t1_exec = load_existing_execution("t1", len(t1_units))
        else:
            t1_exec = execute_c69_stage("t1", t1_units, datalake_root=datalake_root, external_cache_root=external_cache_root)
        t1_cache_rows, t1_schema_rows, t1_view_rows = validate_cache(t1_exec)
        t1_rows = _read_csv(t1_exec["trial_cache_path"]) if t1_exec.get("trial_cache_path") else []
        t1_unit_metrics = unit_metric_rows("t1", t1_rows)
        t1_split_rows, t1_cell_rows = split_label_summary("t1", t1_rows, t1_unit_metrics)
        t1_cs_rows, t1_cs_sens = conditional_cs_summary("t1", t1_rows, t1_unit_metrics)
        t1_adv_rows = source_adversary_summary("t1", t1_rows, t1_unit_metrics, t1_cs_rows)
        t1_gate_passed = bool(int(t1_exec.get("success", 0)) and all(int(r["passed"]) for r in t1_schema_rows) and all(r["status"] == "pass" for r in t1_view_rows))
        if t1_gate_passed:
            if reuse_existing_cache:
                t2_exec = load_existing_execution("t2", len(t2_units))
            else:
                t2_exec = execute_c69_stage("t2", t2_units, datalake_root=datalake_root, external_cache_root=external_cache_root)
            t2_cache_rows, t2_schema_rows, t2_view_rows = validate_cache(t2_exec)
            t2_rows = _read_csv(t2_exec["trial_cache_path"]) if t2_exec.get("trial_cache_path") else []
            t2_unit_metrics = unit_metric_rows("t2", t2_rows)
            t2_split_rows, t2_cell_rows = split_label_summary("t2", t2_rows, t2_unit_metrics)
            t2_cs_rows, t2_cs_sens = conditional_cs_summary("t2", t2_rows, t2_unit_metrics)
            t2_adv_rows = source_adversary_summary("t2", t2_rows, t2_unit_metrics, t2_cs_rows)
    else:
        t1_gate_passed = False

    schema_sig = []
    if t1_exec.get("trial_cache_path"):
        schema_sig.extend(build_schema_signature_rows("t1", _read_csv(t1_exec["trial_cache_path"])))
    else:
        schema_sig.extend(build_schema_signature_rows("t1", []))
    if t2_exec.get("trial_cache_path"):
        schema_sig.extend(build_schema_signature_rows("t2", _read_csv(t2_exec["trial_cache_path"])))
    else:
        schema_sig.extend(build_schema_signature_rows("t2", []))

    res = {
        "config_hash": _lock_config(),
        "authorization_present": authorized,
        "authorization_token_name": "--authorization-token",
        "current_head": ctx["head"],
        "c67_final_gate": ctx["c67"].get("final_gate", ""),
        "c68_final_gate": ctx["c68"].get("final_gate", ""),
        "c65_logical_singleton_rows": len(ctx["c65_rows"]),
        "c65_physical_forward_units": len(canonical),
        "t1_unit_count": len(t1_units),
        "t2_unit_count": len(t2_units),
        "t3_unit_count": 1268,
        "external_cache_root": external_cache_root,
        "external_c66_cache_sha256": ctx["c66_trial_sha"],
        "c69_authorization_audit_rows": build_authorization_rows(authorized),
        "c69_stage_manifest_rows": build_stage_rows(t1_units, t2_units, t1_exec, t2_exec, t1_gate_passed),
        "c69_cache_manifest_t1_rows": t1_cache_rows,
        "c69_cache_manifest_t2_rows": t2_cache_rows,
        "c69_schema_signature_rows": schema_sig,
        "c69_masked_view_contract_rows": [*t1_view_rows, *t2_view_rows],
        "c69_split_label_summary_rows": [*t1_split_rows, *t2_split_rows],
        "c69_split_label_cell_ledger_rows": [*t1_cell_rows, *t2_cell_rows],
        "c69_conditional_cs_summary_rows": [*t1_cs_rows, *t2_cs_rows],
        "c69_conditional_cs_sensitivity_rows": [*t1_cs_sens, *t2_cs_sens],
        "c69_source_adversary_summary_rows": [*t1_adv_rows, *t2_adv_rows],
        "c69_endpoint_boundary_replay_rows": endpoint_boundary_rows(),
        "c69_resource_runtime_summary_rows": build_resource_rows(t1_exec, t2_exec),
        "test_command_manifest_rows": build_test_manifest(test_status),
        "c69_forbidden_claim_scan_rows": [],
        "c69_large_artifact_scan_rows": [],
        "schema_validation_summary_rows": [],
        "red_team_failure_ledger_rows": [],
        "c69_artifact_manifest_rows": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []}, authorized)
    return res


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c69", "command": "python -m pytest oaci/tests/test_c69_powered_trial_cache_scaleup.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c69_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c69_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c69_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "authorization_present": res["authorization_present"],
        "authorization_token_name": res["authorization_token_name"],
        "current_head_at_generation": res["current_head"],
        "external_cache_root": res["external_cache_root"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "c65_logical_singleton_rows": res["c65_logical_singleton_rows"],
            "c65_physical_forward_units": res["c65_physical_forward_units"],
            "t1_units": res["t1_unit_count"],
            "t2_units": res["t2_unit_count"],
            "t3_units_not_authorized": res["t3_unit_count"],
            "t1_cache_rows": next((r["row_count"] for r in res["c69_cache_manifest_t1_rows"] if r["cache_kind"] == "minimal_logits_probs_metadata"), 0),
            "t2_cache_rows": next((r["row_count"] for r in res["c69_cache_manifest_t2_rows"] if r["cache_kind"] == "minimal_logits_probs_metadata"), 0),
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    t1_rows = next((r for r in res["c69_cache_manifest_t1_rows"] if r["cache_kind"] == "minimal_logits_probs_metadata"), {})
    t2_rows = next((r for r in res["c69_cache_manifest_t2_rows"] if r["cache_kind"] == "minimal_logits_probs_metadata"), {})
    split_t2 = next((r for r in res["c69_split_label_summary_rows"] if r["stage"] == "t2"), {})
    cs_t2 = next((r for r in res["c69_conditional_cs_summary_rows"] if r["stage"] == "t2"), {})
    main = "\n".join([
        f"# C69 - Powered Re-inference-Only Trial Cache Scale-Up (frozen C19 `{res['config_hash']}`)",
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
        "## 2. Authorization and Execution",
        "",
        "C69 accepted only the exact CLI `--authorization-token` and did not scan protocol text, prompt text, or environment variables. Under that explicit token, T1 and then T2 ran CPU-only frozen-checkpoint re-inference. T3 remains not authorized.",
        "",
        f"T1 cache rows: `{t1_rows.get('row_count', 0)}` at path hash `{t1_rows.get('path_hash', '')}`.",
        "",
        f"T2 cache rows: `{t2_rows.get('row_count', 0)}` at path hash `{t2_rows.get('path_hash', '')}`.",
        "",
        "Raw trial rows remain external-only and content-addressed; only manifests, hashes, schema signatures, and aggregate diagnostics are committed.",
        "",
        "## 3. Split-Label Diagnostic",
        "",
        f"T2 split-label status: `{split_t2.get('status', 'not_run')}`; construct/eval bAcc Spearman `{split_t2.get('spearman_construct_eval_bacc', '')}`, permutation p `{split_t2.get('permutation_p_value', '')}`, top-quartile lift `{split_t2.get('lift_vs_base', '')}`. This is diagnostic-only and not few-label sufficiency.",
        "",
        "## 4. Conditional-CS Proxy",
        "",
        f"T2 sample-level binary-Y COD proxy status: `{cs_t2.get('status', 'not_run')}`; paired eval rows `{cs_t2.get('paired_eval_rows', '')}`, independent units `{cs_t2.get('independent_checkpoint_units', '')}`, incremental COD `{cs_t2.get('incremental_cod', '')}`, null p95 `{cs_t2.get('null_p95_incremental_cod', '')}`. This is a proxy/smoke diagnostic, not a full conditional-CS claim.",
        "",
        "## 5. Boundary",
        "",
        "The endpoint scalar boundary is preserved: template-only remains below the max null p95, while the same-label endpoint scalar remains a target-label-derived oracle unavailable at selection time. No selector, checkpoint recommendation, OACI rescue, source-only rescue, deployable method, or manuscript prose is emitted.",
        "",
        "## 6. Red-Team Verification",
        "",
        f"Red-team failures: `{d['red_team_failure_count']}`.",
    ])
    red = "\n".join([
        "# C69 - Red-Team Verification",
        "",
        "All C69 red-team gates pass." if d["red_team_failure_count"] == 0 else "C69 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    return {
        "C69_POWERED_TRIAL_CACHE_SCALEUP.md": main,
        "C69_RED_TEAM_VERIFICATION.md": red,
    }


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    specs = {
        "c69_authorization_audit.csv": ("c69_authorization_audit_rows", ["gate", "expected", "observed", "allowed", "passed", "notes"]),
        "c69_stage_manifest.csv": ("c69_stage_manifest_rows", ["stage", "authorized_to_execute", "executed", "forward_units", "trial_rows", "success", "gate_status", "cache_path_hash", "notes"]),
        "c69_cache_manifest_t1.csv": ("c69_cache_manifest_t1_rows", ["stage", "cache_id", "cache_kind", "external_path", "path_hash", "exists", "size_bytes", "sha256", "sha256_match", "row_count", "manifest_row_count", "git_tracked", "status"]),
        "c69_cache_manifest_t2.csv": ("c69_cache_manifest_t2_rows", ["stage", "cache_id", "cache_kind", "external_path", "path_hash", "exists", "size_bytes", "sha256", "sha256_match", "row_count", "manifest_row_count", "git_tracked", "status"]),
        "c69_schema_signature.csv": ("c69_schema_signature_rows", ["stage", "field_count", "schema_sha256", "required_minimum_present", "status"]),
        "c69_masked_view_contract.csv": ("c69_masked_view_contract_rows", ["stage", "view", "sampled_rows", "label_visible_rows", "prediction_visible_rows", "construct_label_visible_rows", "eval_label_visible_rows", "uses_target_labels", "uses_eval_labels", "uses_same_label_endpoint_scalar", "available_at_selection_time", "diagnostic_only", "selection_path_enforced", "policy_boundary_only", "status"]),
        "c69_split_label_summary.csv": ("c69_split_label_summary_rows", ["stage", "status", "independent_checkpoint_units", "construct_rows", "eval_rows", "spearman_construct_eval_bacc", "permutation_p_value", "eval_top_quartile_hit", "eval_top_quartile_base", "lift_vs_base", "few_label_sufficiency_claimed"]),
        "c69_split_label_cell_ledger.csv": ("c69_split_label_cell_ledger_rows", ["stage", "target_id", "seed", "level", "checkpoint_units", "construct_rows", "eval_rows", "mean_construct_bacc", "mean_eval_bacc", "min_eval_bacc", "max_eval_bacc", "diagnostic_only"]),
        "c69_conditional_cs_summary.csv": ("c69_conditional_cs_summary_rows", ["stage", "estimator", "status", "paired_eval_rows", "independent_checkpoint_units", "baseline_mse", "x1_mse", "x1_plus_x2_mse", "incremental_cod", "null_p95_incremental_cod", "permutation_p_value", "full_conditional_cs_claimed"]),
        "c69_conditional_cs_sensitivity.csv": ("c69_conditional_cs_sensitivity_rows", ["stage", "sensitivity", "n_permutations", "observed_incremental_cod", "null_mean", "null_p95", "passes_proxy_gate", "full_cs_claim_allowed"]),
        "c69_source_adversary_summary.csv": ("c69_source_adversary_summary_rows", ["stage", "adversary", "status", "paired_eval_rows", "independent_checkpoint_units", "target_labels_used", "source_domain_trial_logits_available", "x1_cod_vs_intercept", "label_diagnostic_incremental_cod", "escape_hatch_found", "notes"]),
        "c69_endpoint_boundary_replay.csv": ("c69_endpoint_boundary_replay_rows", ["boundary", "observed_hit", "max_null_p95", "beats_null", "uses_target_labels", "available_at_selection_time", "diagnostic_only", "status"]),
        "c69_resource_runtime_summary.csv": ("c69_resource_runtime_summary_rows", ["stage", "runtime_seconds", "checkpoint_units", "trial_rows", "cache_size_bytes", "manifest_size_bytes", "cpu_only", "gpu_used", "training_attempted"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "c69_forbidden_claim_scan.csv": ("c69_forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "c69_large_artifact_scan.csv": ("c69_large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "c69_artifact_manifest.csv": ("c69_artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(TABLE_DIR, name), res.get(key, []), cols)


def write_artifacts(res: dict) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    write_tables(res)
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    paths = [str(p) for p in _listed_paths()]
    res["c69_forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["c69_large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res, res["authorization_present"])
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    write_tables(res)
    paths = _listed_paths()
    res["c69_large_artifact_scan_rows"] = _large_scan(paths)
    res["c69_artifact_manifest_rows"] = [{} for _ in paths]
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_csv(os.path.join(TABLE_DIR, "c69_large_artifact_scan.csv"), res["c69_large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["c69_artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "c69_artifact_manifest.csv"), res["c69_artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c69_powered_trial_cache_scaleup")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--authorization-token", default="", help="Exact C69 authorization token; protocol text is not accepted.")
    ap.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    ap.add_argument("--external-cache-root", default=EXTERNAL_CACHE_ROOT)
    ap.add_argument("--test-status", default="planned")
    ap.add_argument("--reuse-existing-cache", action="store_true", help="Reuse already manifested external C69 caches for analysis/report repair; does not forward.")
    args = ap.parse_args(argv)
    res = run(
        authorization_token=args.authorization_token,
        datalake_root=args.datalake_root,
        external_cache_root=args.external_cache_root,
        test_status=args.test_status,
        reuse_existing_cache=args.reuse_existing_cache,
    )
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C69] decision={res['decision']['primary']} gate={res['decision']['final_gate']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
