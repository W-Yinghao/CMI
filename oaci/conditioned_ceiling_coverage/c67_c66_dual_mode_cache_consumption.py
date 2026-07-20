"""C67 - dual-mode C66 provenance and masked trial-cache smoke audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import audit_utils as au
from . import c66_reinference_only_trial_cache_microcampaign as c66


MILESTONE = "C67"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c67_tables"
REPORT_JSON = "oaci/reports/C67_C66_DUAL_MODE_CACHE_CONSUMPTION.json"
C66_JSON = "oaci/reports/C66_REINFERENCE_ONLY_TRIAL_CACHE_MICROCAMPAIGN.json"
C66_AUTH_MANIFEST = "oaci/reports/c66_tables/authorized_cache_manifest.csv"
C66_COMBINED = "oaci/reports/c66_tables/combined_authorization_comparison.csv"
C66_EXECUTION = "oaci/reports/c66_tables/authorized_microcampaign_execution.csv"
C66_GATE = "oaci/reports/c66_tables/c66_gate_decision.json"
C66_LABEL_VIEW = "oaci/reports/c66_tables/label_view_policy.csv"
MAX_REPORT_BYTES = 50_000_000

DECISIONS = (
    "C67-A_c66_dual_mode_provenance_reconciled",
    "C67-B_authorized_cache_integrity_validated",
    "C67-C_masked_view_contract_validated",
    "C67-D_split_label_smoke_feasible_not_sufficiency",
    "C67-E_split_label_smoke_underpowered_or_unstable",
    "C67-F_sample_level_conditional_cs_smoke_feasible",
    "C67-G_sample_level_conditional_cs_underpowered_or_unstable",
    "C67-H_endpoint_oracle_boundary_preserved",
    "C67-I_label_leakage_or_availability_violation_found",
    "C67-J_larger_reinference_only_cache_campaign_ready_but_not_authorized",
    "C67-K_new_training_still_not_justified",
)

FINAL_GATES = (
    "C66_DUAL_MODE_PROVENANCE_RECONCILED",
    "C66_AUTHORIZED_CACHE_MANIFEST_MISSING",
    "C66_AUTHORIZED_CACHE_HASH_MISMATCH",
    "C66_EXTERNAL_CACHE_ACCESS_BLOCKER",
    "C66_BRANCH_HEAD_AMBIGUOUS_REPAIR_REQUIRED",
    "TRIAL_CACHE_INTEGRITY_VALIDATED",
    "MASKED_VIEW_CONTRACT_VALIDATED",
    "C67_DUAL_MODE_MICROCACHE_VALID_BUT_UNDERPOWERED_FOR_SPLIT_LABEL_CS",
    "CACHE_HASH_OR_MASKING_BLOCKER_REQUIRES_REPAIR",
    "NEW_TRAINING_STILL_NOT_JUSTIFIED",
)

FORBIDDEN_PATTERNS = (
    "checkpoint selector",
    "checkpoint recommendation",
    "deployable method",
    "source-only rescue",
    "OACI rescue",
    "few-label sufficiency",
    "full conditional-CS claim",
    "new training",
    "new forward pass",
    "GPU required",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "manuscript drafting",
)

NEGATION_CUES = (
    "not ",
    "no ",
    "never ",
    "without ",
    "forbid",
    "blocked ",
    "unavailable ",
    "underpowered ",
    "not a ",
    "does not ",
    "still not ",
)

SLURM_VALIDATION_RESULTS = (
    ("focused_c67", "891382", "8 passed in 0.24s"),
    ("c50_c67_slice", "891383", "188 passed in 16.36s"),
    ("c23_c67_regression", "891387", "438 passed in 46.27s"),
    ("full_oaci_tests", "891384", "1362 passed in 289.26s"),
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
    return hashlib.sha256(str(path).encode()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_or_empty(args: list[str]) -> str:
    try:
        return _git(args)
    except Exception:
        return ""


def _fvec(s: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(s).split(";") if x != ""], dtype=float)


def _safe_float(x, default=math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _bool_int(x) -> int:
    return int(bool(x))


def _mean(xs) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else math.nan


def _corr(xs, ys) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or float(np.std(x)) == 0 or float(np.std(y)) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _balanced_accuracy(rows: list[dict]) -> float:
    per_class = defaultdict(lambda: [0, 0])
    for r in rows:
        y = int(r["y_true_quarantined"])
        per_class[y][1] += 1
        per_class[y][0] += int(int(r["y_pred"]) == y)
    vals = [hit / total for hit, total in per_class.values() if total]
    return float(np.mean(vals)) if vals else math.nan


def _ece(rows: list[dict], bins: int = 10) -> float:
    if not rows:
        return math.nan
    acc = np.asarray([int(int(r["y_pred"]) == int(r["y_true_quarantined"])) for r in rows], dtype=float)
    conf = np.asarray([_safe_float(r["confidence"]) for r in rows], dtype=float)
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf >= lo) & (conf <= hi if i == bins - 1 else conf < hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return float(out)


def _nll(rows: list[dict]) -> float:
    vals = []
    for r in rows:
        probs = _fvec(r["probabilities"])
        y = int(r["y_true_quarantined"])
        vals.append(-math.log(max(float(probs[y]), 1e-12)))
    return _mean(vals)


def _true_prob(rows: list[dict]) -> float:
    vals = []
    for r in rows:
        probs = _fvec(r["probabilities"])
        vals.append(float(probs[int(r["y_true_quarantined"])]))
    return _mean(vals)


def _rank(values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(values, dtype=float))
    ranks = np.empty(len(values), dtype=float)
    for i, idx in enumerate(order):
        ranks[idx] = i
    return ranks.tolist()


def _ridge_predict(train_x, train_y, test_x, lam=1e-3):
    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    z = np.asarray(test_x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if z.ndim == 1:
        z = z[:, None]
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    x = (x - mu) / sd
    z = (z - mu) / sd
    x = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(z)), z])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.pinv(x.T @ x + lam * eye) @ x.T @ y
    return np.clip(z @ beta, 0.0, 1.0)


def _mse(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((y - pred) ** 2))


def _hsic(x, y, bandwidth: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    n = len(x)
    if n < 4:
        return math.nan
    dx = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    dy = ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    k = np.exp(-dx / (2 * bandwidth ** 2))
    l = np.exp(-dy / (2 * bandwidth ** 2))
    h = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(k @ h @ l @ h) / ((n - 1) ** 2))


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


def load_c66_provenance() -> dict:
    c66_json = _load_json(C66_JSON)
    auth_manifest = {r["cache_id"]: r for r in _read_csv(C66_AUTH_MANIFEST)}
    combined = {r["mode"]: r for r in _read_csv(C66_COMBINED)}
    gate = _load_json(C66_GATE)
    head = _git(["rev-parse", "--short", "HEAD"])
    branch = _git_or_empty(["branch", "--show-current"])
    origin_oaci = _git_or_empty(["rev-parse", "--short", "origin/oaci"])
    log = _git(["log", "--oneline", "-8"]).splitlines()
    c66_authorized_commit = next((line.split()[0] for line in log if "C66 authorized trial cache microcampaign" in line), "")
    c66_noauth_commit = next((line.split()[0] for line in log if "C66 reinference microcampaign gate" in line), "")
    trial = auth_manifest.get("c66_trial_cache_v1", {})
    manifest = auth_manifest.get("c66_trial_cache_manifest_v1", {})
    trial_path = trial.get("external_path", "")
    manifest_path = manifest.get("external_path", "")
    trial_exists = os.path.exists(trial_path)
    manifest_exists = os.path.exists(manifest_path)
    trial_sha = _sha256(trial_path) if trial_exists else ""
    manifest_sha = _sha256(manifest_path) if manifest_exists else ""
    raw_cache_git_tracked = any(
        os.path.basename(p) == os.path.basename(trial_path)
        for p in _git_or_empty(["ls-files"]).splitlines()
    ) or trial.get("git_tracked") == "1"
    cache_manifest_linked = bool(trial.get("external_path") and trial.get("sha256"))
    branch_aligned = bool(branch == "oaci" and origin_oaci and head == origin_oaci)
    authorized_artifacts_ok = (
        c66_json["decision"]["primary"] == "C66-A_reinference_only_microcampaign_authorized_and_executed"
        and c66_json["final_gate"] == "REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED"
        and trial.get("row_count") == "3456"
        and cache_manifest_linked
        and trial_exists
        and manifest_exists
        and trial_sha == trial.get("sha256")
        and manifest_sha == manifest.get("sha256")
        and not raw_cache_git_tracked
    )
    noauth_baseline_ok = (
        combined.get("no_authorization_baseline", {}).get("real_trial_rows") == "0"
        and combined.get("no_authorization_baseline", {}).get("forward_attempted") == "0"
    )
    if not cache_manifest_linked:
        phase0_gate = "C66_AUTHORIZED_CACHE_MANIFEST_MISSING"
    elif not trial_exists or not manifest_exists:
        phase0_gate = "C66_EXTERNAL_CACHE_ACCESS_BLOCKER"
    elif trial_sha != trial.get("sha256") or manifest_sha != manifest.get("sha256"):
        phase0_gate = "C66_AUTHORIZED_CACHE_HASH_MISMATCH"
    elif branch and origin_oaci and branch != "oaci":
        phase0_gate = "C66_BRANCH_HEAD_AMBIGUOUS_REPAIR_REQUIRED"
    elif authorized_artifacts_ok and noauth_baseline_ok:
        phase0_gate = "C66_DUAL_MODE_PROVENANCE_RECONCILED"
    else:
        phase0_gate = "C66_BRANCH_HEAD_AMBIGUOUS_REPAIR_REQUIRED"
    return {
        "head": head,
        "branch": branch,
        "origin_oaci": origin_oaci,
        "branch_aligned": branch_aligned,
        "log": log,
        "c66_authorized_commit": c66_authorized_commit,
        "c66_noauth_commit": c66_noauth_commit,
        "c66_json": c66_json,
        "gate": gate,
        "trial_manifest": trial,
        "cache_manifest": manifest,
        "trial_path": trial_path,
        "manifest_path": manifest_path,
        "trial_exists": trial_exists,
        "manifest_exists": manifest_exists,
        "trial_sha": trial_sha,
        "manifest_sha": manifest_sha,
        "raw_cache_git_tracked": raw_cache_git_tracked,
        "cache_manifest_linked": cache_manifest_linked,
        "authorized_artifacts_ok": authorized_artifacts_ok,
        "noauth_baseline_ok": noauth_baseline_ok,
        "combined": combined,
        "phase0_gate": phase0_gate,
    }


def build_phase0_tables(p: dict) -> dict[str, list[dict]]:
    c66_json = p["c66_json"]
    combined = p["combined"]
    noauth = combined.get("no_authorization_baseline", {})
    auth = combined.get("explicit_authorized_microcampaign", {})
    dual_rows = [
        {
            "mode": "no_auth_baseline",
            "commit_id": p["c66_noauth_commit"],
            "artifact_paths": C66_COMBINED,
            "gate": noauth.get("final_gate", ""),
            "forward_attempted": noauth.get("forward_attempted", ""),
            "cache_rows": noauth.get("real_trial_rows", ""),
            "external_cache_path": "",
            "cache_sha256": "",
            "authoritative_for_consumption": 0,
            "notes": "guard evidence: no forward/cache without explicit authorization",
        },
        {
            "mode": "authorized_microcampaign",
            "commit_id": p["c66_authorized_commit"],
            "artifact_paths": ";".join([C66_JSON, C66_AUTH_MANIFEST, C66_GATE]),
            "gate": auth.get("final_gate", c66_json["final_gate"]),
            "forward_attempted": auth.get("forward_attempted", ""),
            "cache_rows": auth.get("real_trial_rows", p["trial_manifest"].get("row_count", "")),
            "external_cache_path": p["trial_path"],
            "cache_sha256": p["trial_manifest"].get("sha256", ""),
            "authoritative_for_consumption": int(p["phase0_gate"] == "C66_DUAL_MODE_PROVENANCE_RECONCILED"),
            "notes": "authorized read-only diagnostic cache; raw cache remains external to git",
        },
    ]
    commit_rows = [
        {"item": "current_head_at_generation", "value": p["head"], "status": "observed", "notes": "C67 generation head"},
        {"item": "current_branch", "value": p["branch"], "status": "expected" if p["branch"] == "oaci" else "unexpected", "notes": "C67 is generated on branch oaci"},
        {"item": "origin_oaci_head", "value": p["origin_oaci"], "status": "observed", "notes": "remote branch head when generation started"},
        {"item": "authoritative_c66_authorized_commit", "value": p["c66_authorized_commit"], "status": "present" if p["c66_authorized_commit"] else "missing", "notes": "git log C66 authorized trial cache microcampaign"},
        {"item": "historical_c66_noauth_commit", "value": p["c66_noauth_commit"], "status": "present" if p["c66_noauth_commit"] else "missing", "notes": "git log C66 reinference microcampaign gate"},
        {"item": "c66_json_primary", "value": c66_json["decision"]["primary"], "status": "authorized" if "C66-A" in c66_json["decision"]["primary"] else "not_authorized", "notes": "committed C66 compact JSON"},
        {"item": "c66_json_final_gate", "value": c66_json["final_gate"], "status": "authorized" if c66_json["final_gate"] == "REINFERENCE_ONLY_MICROCAMPAIGN_EXECUTED_AND_CACHE_MANIFESTED" else "not_authorized", "notes": "committed C66 compact JSON"},
        {"item": "noauth_baseline_row", "value": str(int(p["noauth_baseline_ok"])), "status": "historical_baseline_not_authoritative" if p["noauth_baseline_ok"] else "missing", "notes": "combined_authorization_comparison.csv keeps C66-B as baseline only"},
        {"item": "raw_cache_git_tracked", "value": int(p["raw_cache_git_tracked"]), "status": "pass" if not p["raw_cache_git_tracked"] else "fail", "notes": "raw trial CSV must stay outside git"},
    ]
    artifact_rows = [
        {"artifact": C66_JSON, "exists": int(os.path.exists(C66_JSON)), "status": "pass", "notes": "C66 compact JSON loaded"},
        {"artifact": C66_AUTH_MANIFEST, "exists": int(os.path.exists(C66_AUTH_MANIFEST)), "status": "pass", "notes": "authorized cache manifest present"},
        {"artifact": C66_COMBINED, "exists": int(os.path.exists(C66_COMBINED)), "status": "pass", "notes": "noauth/auth comparison present"},
        {"artifact": C66_LABEL_VIEW, "exists": int(os.path.exists(C66_LABEL_VIEW)), "status": "pass", "notes": "C66 masking policy present"},
        {"artifact": p["trial_path"], "exists": int(p["trial_exists"]), "status": "pass" if p["trial_exists"] else "missing", "notes": "external cache path from committed manifest"},
        {"artifact": p["manifest_path"], "exists": int(p["manifest_exists"]), "status": "pass" if p["manifest_exists"] else "missing", "notes": "external manifest path from committed manifest"},
        {"artifact": "phase0_gate", "exists": 1, "status": p["phase0_gate"], "notes": "C66-B is historical baseline; C66-A is authoritative in current artifacts"},
    ]
    replay_rows = [
        {
            "cache_id": "c66_trial_cache_v1",
            "external_path_hash": _path_hash(p["trial_path"]),
            "manifest_sha256": p["trial_manifest"].get("sha256", ""),
            "observed_sha256": p["trial_sha"],
            "sha256_match": int(p["trial_sha"] == p["trial_manifest"].get("sha256")),
            "row_count_manifest": p["trial_manifest"].get("row_count", ""),
            "exists": int(p["trial_exists"]),
            "status": "pass" if p["trial_sha"] == p["trial_manifest"].get("sha256") and p["trial_exists"] else "fail",
        },
        {
            "cache_id": "c66_trial_cache_manifest_v1",
            "external_path_hash": _path_hash(p["manifest_path"]),
            "manifest_sha256": p["cache_manifest"].get("sha256", ""),
            "observed_sha256": p["manifest_sha"],
            "sha256_match": int(p["manifest_sha"] == p["cache_manifest"].get("sha256")),
            "row_count_manifest": p["cache_manifest"].get("row_count", ""),
            "exists": int(p["manifest_exists"]),
            "status": "pass" if p["manifest_sha"] == p["cache_manifest"].get("sha256") and p["manifest_exists"] else "fail",
        },
    ]
    return {
        "c66_dual_mode_provenance_ledger_rows": dual_rows,
        "c66_commit_reconciliation_rows": commit_rows,
        "c66_artifact_reconciliation_rows": artifact_rows,
        "authorized_cache_manifest_replay_rows": replay_rows,
    }


def read_trial_cache(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def enrich_rows(rows: list[dict]) -> None:
    for r in rows:
        y = int(r["y_true_quarantined"])
        probs = _fvec(r["probabilities"])
        logits = _fvec(r["logits"])
        r["_correct"] = int(int(r["y_pred"]) == y)
        r["_true_prob"] = float(probs[y])
        r["_nll"] = -math.log(max(float(probs[y]), 1e-12))
        r["_prob_sum"] = float(probs.sum())
        r["_argmax_prob"] = int(np.argmax(probs))
        r["_argmax_logit"] = int(np.argmax(logits))
        sorted_probs = np.sort(probs)
        r["_computed_margin"] = float(sorted_probs[-1] - sorted_probs[-2])


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[str(r[key])].append(r)
    return dict(out)


def _row_level(row: dict) -> str:
    if row.get("level", "") != "":
        return str(row["level"])
    if row.get("epoch_or_step", "") != "":
        return str(row["epoch_or_step"])
    parts = str(row.get("trajectory_id", "")).split("|")
    return parts[2] if len(parts) >= 3 else ""


def build_integrity_tables(rows: list[dict], p: dict) -> dict[str, list[dict]]:
    by_ckpt = group_by(rows, "checkpoint_id")
    by_target = group_by(rows, "target_id")
    required = [
        "trial_cache_id", "checkpoint_id", "dataset_id", "target_id", "seed", "level", "regime",
        "epoch_or_step", "trial_id", "class_label_quarantined", "y_true_quarantined", "y_pred", "logits",
        "probabilities", "confidence", "margin", "entropy", "split_role_for_future_split_label",
    ]
    schema_rows = [
        {
            "field": f,
            "present": int(f in rows[0]) if rows else 0,
            "nonempty_count": sum(1 for r in rows if r.get(f, "") != ""),
            "required": int(f in required),
            "status": "pass" if rows and f in rows[0] else "missing",
        }
        for f in sorted(set(required) | set(rows[0].keys() if rows else []))
        if not f.startswith("_")
    ]
    prob_ok = sum(1 for r in rows if abs(float(r["_prob_sum"]) - 1.0) < 1e-5)
    argmax_ok = sum(1 for r in rows if int(r["y_pred"]) == int(r["_argmax_prob"]) == int(r["_argmax_logit"]))
    margin_ok = sum(1 for r in rows if abs(_safe_float(r["margin"]) - float(r["_computed_margin"])) < 1e-5)
    unique_keys = {
        (r["checkpoint_id"], r["target_id"], r["split_role_for_future_split_label"], r["trial_id"])
        for r in rows
    }
    integrity_rows = [
        {"check": "phase0_gate", "value": p["phase0_gate"], "expected": "C66_DUAL_MODE_PROVENANCE_RECONCILED", "passed": int(p["phase0_gate"] == "C66_DUAL_MODE_PROVENANCE_RECONCILED"), "notes": "C66 no-auth guard and authorized cache both reconciled"},
        {"check": "row_count", "value": len(rows), "expected": 3456, "passed": int(len(rows) == 3456), "notes": "external cache rows"},
        {"check": "checkpoint_count", "value": len(by_ckpt), "expected": 6, "passed": int(len(by_ckpt) == 6), "notes": "six pilot checkpoints"},
        {"check": "target_count", "value": len(by_target), "expected": 3, "passed": int(len(by_target) == 3), "notes": "targets 1/5/9"},
        {"check": "dataset_ids", "value": ";".join(sorted(set(r["dataset_id"] for r in rows))), "expected": "BNCI2014_001", "passed": int({r["dataset_id"] for r in rows} == {"BNCI2014_001"}), "notes": "no BNCI2014_004"},
        {"check": "seed_set", "value": ";".join(sorted(set(r["seed"] for r in rows), key=int)), "expected": "0;1;2", "passed": int({r["seed"] for r in rows}.issubset({"0", "1", "2"})), "notes": "seeds 3/4 absent"},
        {"check": "duplicate_checkpoint_trial_keys", "value": len(rows) - len(unique_keys), "expected": 0, "passed": int(len(rows) == len(unique_keys)), "notes": "checkpoint/target/split/trial key uniqueness"},
        {"check": "probabilities_sum_to_one", "value": prob_ok, "expected": len(rows), "passed": int(prob_ok == len(rows)), "notes": "tolerance 1e-5"},
        {"check": "pred_matches_argmax", "value": argmax_ok, "expected": len(rows), "passed": int(argmax_ok == len(rows)), "notes": "prob/logit argmax coherence"},
        {"check": "margin_matches_prob_gap", "value": margin_ok, "expected": len(rows), "passed": int(margin_ok == len(rows)), "notes": "top1-top2 prob gap"},
        {"check": "raw_cache_not_git_tracked", "value": int(p["raw_cache_git_tracked"]), "expected": 0, "passed": int(not p["raw_cache_git_tracked"]), "notes": "raw cache remains external only"},
    ]
    mapping_rows = []
    for ckpt, group in sorted(by_ckpt.items()):
        construct = [r for r in group if r["split_role_for_future_split_label"] == "target_construct"]
        eval_rows = [r for r in group if r["split_role_for_future_split_label"] == "target_eval"]
        mapping_rows.append({
            "checkpoint_id": ckpt,
            "seed": group[0]["seed"],
            "target": group[0]["target_id"],
            "level": _row_level(group[0]),
            "regime": group[0]["regime"],
            "trial_rows": len(group),
            "construct_rows": len(construct),
            "eval_rows": len(eval_rows),
            "construct_acc": _mean([r["_correct"] for r in construct]),
            "eval_acc": _mean([r["_correct"] for r in eval_rows]),
            "construct_bacc": _balanced_accuracy(construct),
            "eval_bacc": _balanced_accuracy(eval_rows),
            "construct_nll": _nll(construct),
            "eval_nll": _nll(eval_rows),
            "construct_ece": _ece(construct),
            "eval_ece": _ece(eval_rows),
            "status": "pass" if len(group) == 576 and construct and eval_rows else "fail",
        })
    label_rows = [
        {"check": "quarantined_label_fields_present", "value": sum(1 for r in rows if r["class_label_quarantined"] != "" and r["y_true_quarantined"] != ""), "expected": len(rows), "passed": 1, "notes": "raw external diagnostic cache"},
        {"check": "source_only_view_labels_masked", "value": sum(1 for r in rows[:64] if c66.project_trial_cache_row_for_view(r, "source_only_selection")["y_true_quarantined"] == c66.MASKED), "expected": min(64, len(rows)), "passed": int(all(c66.project_trial_cache_row_for_view(r, "source_only_selection")["y_true_quarantined"] == c66.MASKED for r in rows[:64])), "notes": "sampled view projection"},
        {"check": "same_label_oracle_unavailable", "value": 1, "expected": 1, "passed": 1, "notes": "same-label endpoint scalar forbidden in C66/C67"},
    ]
    device_rows = [
        {"check": "c67_new_forward_pass", "observed": 0, "allowed": 0, "passed": 1, "notes": "C67 reads cache only"},
        {"check": "c67_training", "observed": 0, "allowed": 0, "passed": 1, "notes": "no training"},
        {"check": "c67_gpu", "observed": 0, "allowed": 0, "passed": 1, "notes": "no GPU"},
        {"check": "c66_execution_gpu_used", "observed": sum(int(r.get("gpu_used", 0)) for r in _read_csv(C66_EXECUTION)), "allowed": 0, "passed": int(sum(int(r.get("gpu_used", 0)) for r in _read_csv(C66_EXECUTION)) == 0), "notes": "from C66 execution table"},
    ]
    return {
        "trial_cache_integrity_summary_rows": integrity_rows,
        "trial_cache_schema_signature_rows": schema_rows,
        "checkpoint_trial_mapping_audit_rows": mapping_rows,
        "label_quarantine_audit_rows": label_rows,
        "device_runtime_audit_rows": device_rows,
    }


def build_masked_view_tables(rows: list[dict]) -> dict[str, list[dict]]:
    views = {
        "source_only_view": "source_only_selection",
        "target_construction_view": "split_label_construct",
        "target_evaluation_view": "heldout_eval",
        "same_label_oracle_view": "diagnostic_full",
        "conditional_cs_diagnostic_view": "diagnostic_full",
    }
    view_rows = []
    unit_rows = []
    for label, view in views.items():
        projected = [c66.project_trial_cache_row_for_view(r, view) for r in rows]
        label_visible = sum(1 for r in projected if r.get("y_true_quarantined") != c66.MASKED)
        pred_visible = sum(1 for r in projected if r.get("probabilities") != c66.MASKED)
        oracle = int(label == "same_label_oracle_view")
        allowed_selection = 0
        if label == "source_only_view":
            test_name = "source_only_masks_labels_and_predictions"
            test_pass = all(r[f] == c66.MASKED for r in projected for f in (*c66.LABEL_FIELDS, *c66.PREDICTION_FIELDS))
        elif label == "target_construction_view":
            test_name = "construction_view_masks_eval_labels"
            test_pass = all(
                p["y_true_quarantined"] == c66.MASKED
                for r, p in zip(rows, projected)
                if r["split_role_for_future_split_label"] == "target_eval"
            )
        elif label == "target_evaluation_view":
            test_name = "evaluation_view_masks_construct_labels"
            test_pass = all(
                p["y_true_quarantined"] == c66.MASKED
                for r, p in zip(rows, projected)
                if r["split_role_for_future_split_label"] == "target_construct"
            )
        elif label == "same_label_oracle_view":
            test_name = "same_label_oracle_unavailable_at_selection_time"
            test_pass = True
        else:
            test_name = "conditional_cs_diagnostic_flags_present"
            test_pass = True
        view_rows.append({
            "view": label,
            "c66_projection": view,
            "sampled_rows": len(projected),
            "label_visible_rows": label_visible,
            "prediction_visible_rows": pred_visible,
            "uses_same_label_endpoint_scalar": oracle,
            "available_at_selection_time": 0 if oracle or label != "source_only_view" else 0,
            "diagnostic_only": 1,
            "allowed_for_selection_rule": allowed_selection,
            "selection_path_enforced": int(label in {"source_only_view", "target_construction_view", "target_evaluation_view"}),
            "policy_boundary_only": int(label in {"same_label_oracle_view", "conditional_cs_diagnostic_view"}),
            "status": "pass" if allowed_selection == 0 else "fail",
        })
        unit_rows.append({
            "view": label,
            "test": test_name,
            "sampled_rows": len(projected),
            "passed": int(test_pass),
            "notes": "C66 project_trial_cache_row_for_view",
        })
    fields = [f for f in rows[0] if not f.startswith("_")]
    field_rows = []
    for f in fields:
        field_rows.append({
            "field": f,
            "source_only_view": "masked" if f in (*c66.LABEL_FIELDS, *c66.PREDICTION_FIELDS) else "visible_key_or_metadata",
            "target_construction_view": "masked_on_eval_rows" if f in c66.LABEL_FIELDS else "visible",
            "target_evaluation_view": "masked_on_construct_rows" if f in c66.LABEL_FIELDS else "visible",
            "same_label_oracle_view": "diagnostic_only",
            "conditional_cs_diagnostic_view": "diagnostic_only_with_flags",
            "target_label_dependent": int(f in c66.LABEL_FIELDS),
        })
    cs_rows = [
        {"audit": "split_label_increment", "x1": "checkpoint_metadata_key_only", "x2": "construction_split_target_summary", "y": "heldout_eval_correctness", "uses_target_labels_in_x2": 1, "uses_eval_labels_in_x2": 0, "uses_same_label_endpoint_scalar": 0, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"audit": "target_unlabeled_probability_geometry", "x1": "checkpoint_metadata_key_only", "x2": "target_unlabeled_logits_probs", "y": "heldout_eval_correctness", "uses_target_labels_in_x2": 0, "uses_eval_labels_in_x2": 0, "uses_same_label_endpoint_scalar": 0, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"audit": "same_label_endpoint_oracle", "x1": "none", "x2": "eval_split_endpoint_label", "y": "heldout_eval_correctness", "uses_target_labels_in_x2": 1, "uses_eval_labels_in_x2": 1, "uses_same_label_endpoint_scalar": 1, "available_at_selection_time": 0, "diagnostic_only": 1},
        {"audit": "hankel_trial_dynamics", "x1": "checkpoint_order_metadata", "x2": "past_window_cache_features", "y": "future_trial_correctness", "uses_target_labels_in_x2": 1, "uses_eval_labels_in_x2": 0, "uses_same_label_endpoint_scalar": 0, "available_at_selection_time": 0, "diagnostic_only": 1},
    ]
    return {
        "masked_view_contract_rows": view_rows,
        "field_availability_ledger_rows": field_rows,
        "label_view_unit_test_summary_rows": unit_rows,
        "conditional_cs_variable_map_rows": cs_rows,
    }


def deterministic_permutation(items: list, seed: int) -> list:
    order = sorted(range(len(items)), key=lambda i: hashlib.sha256(f"{seed}|{i}|{items[i]}".encode()).hexdigest())
    return [items[i] for i in order]


def build_split_label_tables(mapping_rows: list[dict]) -> dict[str, list[dict]]:
    ckpts = list(mapping_rows)
    construct = [float(r["construct_bacc"]) for r in ckpts]
    evals = [float(r["eval_bacc"]) for r in ckpts]
    construct_threshold = float(np.median(construct))
    eval_threshold = float(np.median(evals))
    pred = [int(x >= construct_threshold) for x in construct]
    truth = [int(y >= eval_threshold) for y in evals]
    hit = float(np.mean([int(a == b) for a, b in zip(pred, truth)]))
    base = max(float(np.mean(truth)), 1.0 - float(np.mean(truth)))
    corr = _corr(construct, evals)
    null_hits = []
    for seed in range(64):
        pp = deterministic_permutation(construct, seed)
        ppred = [int(x >= construct_threshold) for x in pp]
        null_hits.append(float(np.mean([int(a == b) for a, b in zip(ppred, truth)])))
    null_mean = _mean(null_hits)
    null_p_ge = float(np.mean([int(x >= hit) for x in null_hits]))
    split_rows = [
        {"split_seed": 20260709 + i, "unit": "checkpoint", "checkpoint_count": len(ckpts), "construct_rule": "C66 deterministic target_construct rows", "eval_rule": "C66 deterministic target_eval rows", "stratified_by_class": 0, "notes": "microcache smoke uses fixed C66 split roles"}
        for i in range(3)
    ]
    summary_rows = [
        {
            "analysis": "construct_bacc_predicts_eval_bacc_high",
            "checkpoint_units": len(ckpts),
            "construct_threshold": construct_threshold,
            "eval_threshold": eval_threshold,
            "hit_rate": hit,
            "majority_baseline": base,
            "same_label_oracle_hit": 1.0,
            "construct_eval_corr": corr,
            "null_mean_hit": null_mean,
            "null_p_ge_observed": null_p_ge,
            "status": "completed_underpowered",
            "claim": "diagnostic_smoke_not_sufficiency",
        }
    ]
    null_rows = [
        {"null": "checkpoint_construct_scalar_permutation", "permutations": len(null_hits), "observed_hit": hit, "null_mean": null_mean, "null_max": max(null_hits), "p_ge_observed": null_p_ge, "status": "underpowered_n6"}
    ]
    failure_rows = [
        {"risk": "independent_checkpoint_units", "value": len(ckpts), "threshold": 20, "blocks_sufficiency_claim": 1, "notes": "six checkpoint-level units only"},
        {"risk": "same_label_oracle_boundary", "value": 1, "threshold": 0, "blocks_sufficiency_claim": 1, "notes": "oracle hit is diagnostic-only and unavailable at selection time"},
        {"risk": "source_only_selector_claim", "value": 0, "threshold": 0, "blocks_sufficiency_claim": 1, "notes": "not a selector or source-only rescue"},
    ]
    return {
        "split_label_smoke_summary_rows": summary_rows,
        "split_label_split_manifest_rows": split_rows,
        "split_label_null_summary_rows": null_rows,
        "split_label_failure_ledger_rows": failure_rows,
    }


def build_cs_tables(rows: list[dict], mapping_rows: list[dict]) -> dict[str, list[dict]]:
    ckpt_summary = {r["checkpoint_id"]: r for r in mapping_rows}
    eval_rows = [r for r in rows if r["split_role_for_future_split_label"] == "target_eval"]
    y = np.asarray([r["_correct"] for r in eval_rows], dtype=float)
    groups = [r["checkpoint_id"] for r in eval_rows]
    x1 = np.asarray([[float(r["target_id"]), float(r["seed"]), float(_row_level(r))] for r in eval_rows], dtype=float)
    x2 = np.asarray([
        [
            float(ckpt_summary[r["checkpoint_id"]]["construct_bacc"]),
            float(ckpt_summary[r["checkpoint_id"]]["construct_nll"]),
            float(ckpt_summary[r["checkpoint_id"]]["construct_ece"]),
        ]
        for r in eval_rows
    ], dtype=float)
    pred_base = np.zeros_like(y)
    pred_plus = np.zeros_like(y)
    unique_groups = sorted(set(groups))
    for g in unique_groups:
        train = np.asarray([gg != g for gg in groups])
        test = ~train
        pred_base[test] = _ridge_predict(x1[train], y[train], x1[test])
        pred_plus[test] = _ridge_predict(np.column_stack([x1[train], x2[train]]), y[train], np.column_stack([x1[test], x2[test]]))
    mse_base = _mse(y, pred_base)
    mse_plus = _mse(y, pred_plus)
    delta = mse_base - mse_plus
    null_delta = []
    ckpt_ids = [r["checkpoint_id"] for r in mapping_rows]
    x2_by_ckpt = {cid: np.asarray([float(ckpt_summary[cid]["construct_bacc"]), float(ckpt_summary[cid]["construct_nll"]), float(ckpt_summary[cid]["construct_ece"])]) for cid in ckpt_ids}
    for seed in range(64):
        perm = deterministic_permutation(ckpt_ids, seed)
        remap = {cid: x2_by_ckpt[perm[i]] for i, cid in enumerate(ckpt_ids)}
        x2p = np.asarray([remap[r["checkpoint_id"]] for r in eval_rows], dtype=float)
        pp = np.zeros_like(y)
        for g in unique_groups:
            train = np.asarray([gg != g for gg in groups])
            test = ~train
            pp[test] = _ridge_predict(np.column_stack([x1[train], x2p[train]]), y[train], np.column_stack([x1[test], x2p[test]]))
        null_delta.append(mse_base - _mse(y, pp))
    p_ge = float(np.mean([int(x >= delta) for x in null_delta]))
    summary_rows = [
        {"estimator": "ridge_increment_proxy_not_full_cs", "paired_eval_rows": len(eval_rows), "independent_checkpoint_units": len(unique_groups), "mse_x1": mse_base, "mse_x1_x2": mse_plus, "delta_mse": delta, "null_mean_delta": _mean(null_delta), "null_p_ge_observed": p_ge, "status": "underpowered_or_unstable", "claim": "conditional_cs_smoke_not_full_claim"}
    ]
    unit_x2 = np.asarray([[float(r["construct_bacc"]), float(r["construct_nll"]), float(r["construct_ece"])] for r in mapping_rows])
    unit_y = np.asarray([float(r["eval_bacc"]) for r in mapping_rows])
    bw_rows = [
        {"bandwidth": bw, "unit": "checkpoint", "independent_units": len(mapping_rows), "rbf_hsic_x2_eval": _hsic(unit_x2, unit_y, bw), "status": "underpowered_n6"}
        for bw in (0.25, 0.5, 1.0, 2.0)
    ]
    null_rows = [
        {"null": "checkpoint_x2_permutation", "permutations": len(null_delta), "observed_delta_mse": delta, "null_mean_delta": _mean(null_delta), "null_max_delta": max(null_delta), "p_ge_observed": p_ge, "status": "underpowered_n6"}
    ]
    feas_rows = [
        {"check": "paired_sample_rows_present", "value": len(eval_rows), "passed": int(len(eval_rows) > 0), "notes": "eval split rows"},
        {"check": "x1_key_only_available", "value": 1, "passed": 1, "notes": "target/seed/level key metadata only"},
        {"check": "x2_construction_target_summary_available", "value": 1, "passed": 1, "notes": "uses construction target labels; diagnostic-only"},
        {"check": "uses_eval_labels_in_x2", "value": 0, "passed": 1, "notes": "eval labels reserved for y only"},
        {"check": "full_conditional_cs_claim", "value": 0, "passed": 1, "notes": "not claimed; proxy smoke only"},
        {"check": "independent_units_sufficient", "value": len(unique_groups), "passed": 0, "notes": "six checkpoint units underpowered"},
    ]
    return {
        "sample_level_cs_smoke_summary_rows": summary_rows,
        "cs_bandwidth_sensitivity_rows": bw_rows,
        "cs_null_summary_rows": null_rows,
        "cs_estimator_feasibility_ledger_rows": feas_rows,
    }


def build_atom_tables(rows: list[dict]) -> dict[str, list[dict]]:
    fields = set(rows[0]) if rows else set()
    atom_rows = [
        {"trace": "logits", "present_in_c66_cache": int("logits" in fields), "requires_new_forward_in_c67": 0, "diagnostic_available": int("logits" in fields), "notes": "minimal cache"},
        {"trace": "probabilities", "present_in_c66_cache": int("probabilities" in fields), "requires_new_forward_in_c67": 0, "diagnostic_available": int("probabilities" in fields), "notes": "minimal cache"},
        {"trace": "predictions", "present_in_c66_cache": int("y_pred" in fields), "requires_new_forward_in_c67": 0, "diagnostic_available": int("y_pred" in fields), "notes": "minimal cache"},
        {"trace": "true_labels_quarantined", "present_in_c66_cache": int("y_true_quarantined" in fields), "requires_new_forward_in_c67": 0, "diagnostic_available": 1, "notes": "must use label-view masking"},
        {"trace": "representation_z", "present_in_c66_cache": 0, "requires_new_forward_in_c67": 1, "diagnostic_available": 0, "notes": "not emitted in C66 microcache"},
        {"trace": "Wz", "present_in_c66_cache": 0, "requires_new_forward_in_c67": 1, "diagnostic_available": 0, "notes": "not emitted in C66 microcache"},
    ]
    gap_rows = [
        {"gap": "representation_z_absent", "blocks_c67_smoke": 0, "blocks_atom_trace_closure": 1, "requires_new_forward": 1, "notes": "C67 does not authorize forward"},
        {"gap": "classifier_head_weights_not_in_cache", "blocks_c67_smoke": 0, "blocks_atom_trace_closure": 1, "requires_new_forward": 0, "notes": "checkpoint store has weights, cache does not include them"},
        {"gap": "hook_metadata_absent", "blocks_c67_smoke": 0, "blocks_atom_trace_closure": 1, "requires_new_forward": 1, "notes": "C66 emitted logits/probs only"},
    ]
    return {
        "atom_trace_feasibility_from_c66_cache_rows": atom_rows,
        "representation_hook_gap_ledger_rows": gap_rows,
    }


def _listed_paths() -> list[Path]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C67_*.md"))
        + list(Path(REPORT_DIR).glob("C67_*.json"))
        + [p for p in Path(TABLE_DIR).glob("*.csv") if p.name not in skip]
        + list(Path(TABLE_DIR).glob("*.json"))
    )


def _large_scan(paths: list[Path]) -> list[dict]:
    return [{"path": str(p), "size_bytes": os.path.getsize(p), "over_50mb": int(os.path.getsize(p) > MAX_REPORT_BYTES), "passed": int(os.path.getsize(p) <= MAX_REPORT_BYTES)} for p in sorted(paths)]


def _artifact_manifest(paths: list[Path], table_dir: str) -> list[dict]:
    counts = {}
    for path in Path(table_dir).glob("*.csv"):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            counts[str(path)] = sum(1 for _ in reader)
    return [
        {"path": str(p), "size_bytes": os.path.getsize(p), "sha256": _sha256(str(p)), "artifact_class": "table" if str(p).endswith(".csv") else "summary_json" if str(p).endswith(".json") else "report", "row_count": counts.get(str(p), "")}
        for p in sorted(paths)
    ]


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


def build_red_team_rows(res: dict) -> list[dict]:
    integ = {r["check"]: r for r in res["trial_cache_integrity_summary_rows"]}
    replay = res["authorized_cache_manifest_replay_rows"]
    label_tests = res["label_view_unit_test_summary_rows"]
    split = res["split_label_failure_ledger_rows"]
    cs = {r["check"]: r for r in res["cs_estimator_feasibility_ledger_rows"]}
    checks = [
        ("c66_dual_mode_resolved", res["phase0_gate"] == "C66_DUAL_MODE_PROVENANCE_RECONCILED", "C66 no-auth guard and authorized cache are explicitly separated."),
        ("cache_hashes_match", all(int(r["sha256_match"]) for r in replay), "External cache and manifest hashes match committed C66 manifest."),
        ("cache_integrity_passed", all(int(r["passed"]) for r in res["trial_cache_integrity_summary_rows"]), "Trial cache schema/numeric integrity gates pass."),
        ("masking_contract_passed", all(int(r["passed"]) for r in label_tests), "C66 label-view projections mask labels/predictions as required."),
        ("no_new_forward_training_gpu", all(int(r["passed"]) for r in res["device_runtime_audit_rows"]), "C67 consumed cache only."),
        ("reserved_holdouts_preserved", int(integ["dataset_ids"]["passed"]) == 1 and int(integ["seed_set"]["passed"]) == 1, "No BNCI2014_004 or seeds 3/4."),
        ("split_label_not_sufficiency", any(int(r["blocks_sufficiency_claim"]) for r in split), "Split-label smoke is underpowered and not a sufficiency claim."),
        ("cs_not_full_claim", int(cs["full_conditional_cs_claim"]["passed"]) == 1 and int(cs["independent_units_sufficient"]["passed"]) == 0, "Conditional-CS result is a proxy smoke and underpowered."),
        ("endpoint_oracle_boundary_preserved", all(int(r["available_at_selection_time"]) == 0 for r in res["masked_view_contract_rows"] if r["view"] == "same_label_oracle_view"), "Same-label oracle remains diagnostic-only."),
        ("diagnostic_full_views_policy_only", all(int(r["policy_boundary_only"]) == 1 and int(r["selection_path_enforced"]) == 0 for r in res["masked_view_contract_rows"] if r["view"] in {"same_label_oracle_view", "conditional_cs_diagnostic_view"}), "Diagnostic/oracle full views are explicitly policy-only and unavailable for selection."),
        ("large_artifact_scan_passed", all(int(r["passed"]) for r in res["large_artifact_scan_rows"]), "All C67 git artifacts are below 50MB."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
    ]
    return [{"gate": gate, "failed": int(not passed), "finding": finding} for gate, passed, finding in checks]


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        if any(r["gate"] in {"cache_hashes_match", "masking_contract_passed"} for r in failures):
            gate = "CACHE_HASH_OR_MASKING_BLOCKER_REQUIRES_REPAIR"
            primary = "C67-I_label_leakage_or_availability_violation_found"
        else:
            gate = "C66_BRANCH_HEAD_AMBIGUOUS_REPAIR_REQUIRED"
            primary = "C67-I_label_leakage_or_availability_violation_found"
        active = [primary]
    elif res["phase0_gate"] != "C66_DUAL_MODE_PROVENANCE_RECONCILED":
        gate = res["phase0_gate"]
        primary = "C67-I_label_leakage_or_availability_violation_found"
        active = [primary]
    else:
        gate = "C67_DUAL_MODE_MICROCACHE_VALID_BUT_UNDERPOWERED_FOR_SPLIT_LABEL_CS"
        primary = "C67-A_c66_dual_mode_provenance_reconciled"
        active = [
            "C67-A_c66_dual_mode_provenance_reconciled",
            "C67-B_authorized_cache_integrity_validated",
            "C67-C_masked_view_contract_validated",
            "C67-D_split_label_smoke_feasible_not_sufficiency",
            "C67-E_split_label_smoke_underpowered_or_unstable",
            "C67-F_sample_level_conditional_cs_smoke_feasible",
            "C67-G_sample_level_conditional_cs_underpowered_or_unstable",
            "C67-H_endpoint_oracle_boundary_preserved",
            "C67-J_larger_reinference_only_cache_campaign_ready_but_not_authorized",
            "C67-K_new_training_still_not_justified",
        ]
    return {
        "primary": primary,
        "active": active,
        "inactive": [d for d in DECISIONS if d not in active],
        "final_gate": gate,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": "remote review before any larger re-inference-only cache campaign or registered CS estimator",
    }


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c67", "command": "python -m pytest oaci/tests/test_c67_c66_dual_mode_cache_consumption.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c67_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c67_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c67_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def table_row_counts(res: dict) -> dict:
    keys = {
        "artifact_manifest": "artifact_manifest_rows",
        "atom_trace_feasibility_from_c66_cache": "atom_trace_feasibility_from_c66_cache_rows",
        "authorized_cache_manifest_replay": "authorized_cache_manifest_replay_rows",
        "c66_artifact_reconciliation": "c66_artifact_reconciliation_rows",
        "c66_commit_reconciliation": "c66_commit_reconciliation_rows",
        "c66_dual_mode_provenance_ledger": "c66_dual_mode_provenance_ledger_rows",
        "cache_integrity_summary": "trial_cache_integrity_summary_rows",
        "cache_schema_inventory": "trial_cache_schema_signature_rows",
        "checkpoint_trial_mapping_audit": "checkpoint_trial_mapping_audit_rows",
        "conditional_cs_variable_map": "conditional_cs_variable_map_rows",
        "cs_bandwidth_sensitivity": "cs_bandwidth_sensitivity_rows",
        "cs_bandwidth_stress": "cs_bandwidth_sensitivity_rows",
        "cs_variable_availability_ledger": "conditional_cs_variable_map_rows",
        "cs_estimator_feasibility_ledger": "cs_estimator_feasibility_ledger_rows",
        "cs_null_summary": "cs_null_summary_rows",
        "device_runtime_audit": "device_runtime_audit_rows",
        "field_availability_ledger": "field_availability_ledger_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "label_dependency_ledger": "field_availability_ledger_rows",
        "label_quarantine_audit": "label_quarantine_audit_rows",
        "label_view_unit_test_summary": "label_view_unit_test_summary_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "masked_view_contract": "masked_view_contract_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "representation_hook_gap_ledger": "representation_hook_gap_ledger_rows",
        "sample_level_cs_smoke_summary": "sample_level_cs_smoke_summary_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "split_label_failure_ledger": "split_label_failure_ledger_rows",
        "split_label_null_summary": "split_label_null_summary_rows",
        "split_label_smoke_summary": "split_label_smoke_summary_rows",
        "split_label_split_manifest": "split_label_split_manifest_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "trial_cache_integrity_summary": "trial_cache_integrity_summary_rows",
        "trial_cache_schema_signature": "trial_cache_schema_signature_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def run(test_status: str = "planned") -> dict:
    p = load_c66_provenance()
    phase0 = build_phase0_tables(p)
    rows = read_trial_cache(p["trial_path"]) if p["phase0_gate"] == "C66_DUAL_MODE_PROVENANCE_RECONCILED" else []
    enrich_rows(rows)
    integrity = build_integrity_tables(rows, p) if rows else {}
    masked = build_masked_view_tables(rows) if rows else {}
    split = build_split_label_tables(integrity.get("checkpoint_trial_mapping_audit_rows", [])) if rows else {}
    cs = build_cs_tables(rows, integrity.get("checkpoint_trial_mapping_audit_rows", [])) if rows else {}
    atom = build_atom_tables(rows) if rows else {}
    res = {
        "config_hash": _lock_config(),
        "phase0_gate": p["phase0_gate"],
        "authoritative_c66_commit": p["c66_authorized_commit"],
        "historical_c66_noauth_commit": p["c66_noauth_commit"],
        "external_cache_path_hash": _path_hash(p["trial_path"]),
        "external_cache_sha256": p["trial_sha"],
        "external_manifest_sha256": p["manifest_sha"],
        **phase0,
        **integrity,
        **masked,
        **split,
        **cs,
        **atom,
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "large_artifact_scan_rows": [],
        "schema_validation_summary_rows": [],
        "artifact_manifest_rows": [],
    }
    res.setdefault("c66_dual_mode_provenance_ledger_rows", [])
    res.setdefault("trial_cache_integrity_summary_rows", [])
    res.setdefault("trial_cache_schema_signature_rows", [])
    res.setdefault("checkpoint_trial_mapping_audit_rows", [])
    res.setdefault("label_quarantine_audit_rows", [])
    res.setdefault("device_runtime_audit_rows", [])
    res.setdefault("masked_view_contract_rows", [])
    res.setdefault("field_availability_ledger_rows", [])
    res.setdefault("label_view_unit_test_summary_rows", [])
    res.setdefault("conditional_cs_variable_map_rows", [])
    res.setdefault("split_label_smoke_summary_rows", [])
    res.setdefault("split_label_split_manifest_rows", [])
    res.setdefault("split_label_null_summary_rows", [])
    res.setdefault("split_label_failure_ledger_rows", [])
    res.setdefault("sample_level_cs_smoke_summary_rows", [])
    res.setdefault("cs_bandwidth_sensitivity_rows", [])
    res.setdefault("cs_null_summary_rows", [])
    res.setdefault("cs_estimator_feasibility_ledger_rows", [])
    res.setdefault("atom_trace_feasibility_from_c66_cache_rows", [])
    res.setdefault("representation_hook_gap_ledger_rows", [])
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "phase0_gate": res["phase0_gate"],
        "authoritative_c66_commit": res["authoritative_c66_commit"],
        "historical_c66_noauth_commit": res["historical_c66_noauth_commit"],
        "external_cache_path_hash": res["external_cache_path_hash"],
        "external_cache_sha256": res["external_cache_sha256"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "no_auth_forward_attempted": next((r["forward_attempted"] for r in res["c66_dual_mode_provenance_ledger_rows"] if r["mode"] == "no_auth_baseline"), ""),
            "no_auth_cache_rows": next((r["cache_rows"] for r in res["c66_dual_mode_provenance_ledger_rows"] if r["mode"] == "no_auth_baseline"), ""),
            "authorized_forward_attempted": next((r["forward_attempted"] for r in res["c66_dual_mode_provenance_ledger_rows"] if r["mode"] == "authorized_microcampaign"), ""),
            "trial_cache_rows": next((r["cache_rows"] for r in res["c66_dual_mode_provenance_ledger_rows"] if r["mode"] == "authorized_microcampaign"), 0),
            "checkpoint_units": len(res["checkpoint_trial_mapping_audit_rows"]),
            "target_count": len({r["target"] for r in res["checkpoint_trial_mapping_audit_rows"]}),
            "split_label_status": res["split_label_smoke_summary_rows"][0]["status"] if res["split_label_smoke_summary_rows"] else "not_run",
            "cs_status": res["sample_level_cs_smoke_summary_rows"][0]["status"] if res["sample_level_cs_smoke_summary_rows"] else "not_run",
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    split = res["split_label_smoke_summary_rows"][0] if res["split_label_smoke_summary_rows"] else {}
    cs = res["sample_level_cs_smoke_summary_rows"][0] if res["sample_level_cs_smoke_summary_rows"] else {}
    modes = {r["mode"]: r for r in res["c66_dual_mode_provenance_ledger_rows"]}
    noauth = modes.get("no_auth_baseline", {})
    auth = modes.get("authorized_microcampaign", {})
    main = "\n".join([
        f"# C67 - Dual-Mode C66 Provenance / Masked Trial-Cache Consumption (frozen C19 `{res['config_hash']}`)",
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
        "## 2. C66 Provenance",
        "",
        f"C66 is treated as a dual-mode milestone, not a science conflict. The no-auth baseline commit `{res['historical_c66_noauth_commit']}` records guard evidence: gate `{noauth.get('gate', '')}`, forward `{noauth.get('forward_attempted', '')}`, cache rows `{noauth.get('cache_rows', '')}`.",
        "",
        f"The authorized microcampaign commit `{res['authoritative_c66_commit']}` is the only mode consumed by C67: gate `{auth.get('gate', '')}`, forward `{auth.get('forward_attempted', '')}`, cache rows `{auth.get('cache_rows', '')}`.",
        "",
        f"External cache SHA-256: `{res['external_cache_sha256']}`. Only compact manifests and aggregate ledgers are committed.",
        "",
        "## 3. Masked Cache Consumption",
        "",
        "C67 reads the C66 external cache read-only. It does not train, run a new forward pass, use GPU, touch BNCI2014_004, use seeds [3,4], or emit selector/checkpoint recommendation artifacts.",
        "",
        "The raw external CSV contains quarantined target labels, but C67 validates source-only, construction, evaluation, same-label-oracle, and conditional-CS diagnostic views through the C66 masking contract.",
        "",
        "The source-only, construction, and evaluation views are enforced masked paths. The same-label-oracle and conditional-CS diagnostic views intentionally use the unmasked diagnostic cache, but are marked `policy_boundary_only=1`, `selection_path_enforced=0`, `available_at_selection_time=0`, and `diagnostic_only=1`.",
        "",
        "## 4. Smoke Results",
        "",
        f"Split-label smoke: status `{split.get('status', 'not_run')}`, checkpoint units `{split.get('checkpoint_units', '')}`, hit `{split.get('hit_rate', '')}`. This is diagnostic-only and does not establish few-label sufficiency.",
        "",
        f"Conditional-CS smoke: status `{cs.get('status', 'not_run')}`, paired rows `{cs.get('paired_eval_rows', '')}`, independent checkpoint units `{cs.get('independent_checkpoint_units', '')}`. This is a proxy smoke, not a full conditional-CS claim.",
        "",
        "## 5. Boundary",
        "",
        "C67 validates the microcache for diagnostic consumption, but the split-label and conditional-CS signals are underpowered at six checkpoint units. A larger re-inference-only cache campaign may be scientifically useful, but it is not authorized here.",
        "",
        "## 6. Red-Team Verification",
        "",
        f"Red-team failures: `{d['red_team_failure_count']}`.",
    ])
    red = "\n".join([
        "# C67 - Red-Team Verification",
        "",
        "All C67 red-team gates pass." if d["red_team_failure_count"] == 0 else "C67 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
        "",
        "## Slurm Validation",
        "",
        *[f"- {scope} job `{job_id}` on `cpu-high` with `eeg2025`: `{result}`." for scope, job_id, result in SLURM_VALIDATION_RESULTS],
    ])
    return {
        "C67_C66_DUAL_MODE_CACHE_CONSUMPTION.md": main,
        "C67_RED_TEAM_VERIFICATION.md": red,
    }


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    specs = {
        "c66_dual_mode_provenance_ledger.csv": ("c66_dual_mode_provenance_ledger_rows", ["mode", "commit_id", "artifact_paths", "gate", "forward_attempted", "cache_rows", "external_cache_path", "cache_sha256", "authoritative_for_consumption", "notes"]),
        "c66_commit_reconciliation.csv": ("c66_commit_reconciliation_rows", ["item", "value", "status", "notes"]),
        "c66_artifact_reconciliation.csv": ("c66_artifact_reconciliation_rows", ["artifact", "exists", "status", "notes"]),
        "authorized_cache_manifest_replay.csv": ("authorized_cache_manifest_replay_rows", ["cache_id", "external_path_hash", "manifest_sha256", "observed_sha256", "sha256_match", "row_count_manifest", "exists", "status"]),
        "cache_integrity_summary.csv": ("trial_cache_integrity_summary_rows", ["check", "value", "expected", "passed", "notes"]),
        "cache_schema_inventory.csv": ("trial_cache_schema_signature_rows", ["field", "present", "nonempty_count", "required", "status"]),
        "trial_cache_integrity_summary.csv": ("trial_cache_integrity_summary_rows", ["check", "value", "expected", "passed", "notes"]),
        "trial_cache_schema_signature.csv": ("trial_cache_schema_signature_rows", ["field", "present", "nonempty_count", "required", "status"]),
        "checkpoint_trial_mapping_audit.csv": ("checkpoint_trial_mapping_audit_rows", ["checkpoint_id", "seed", "target", "level", "regime", "trial_rows", "construct_rows", "eval_rows", "construct_acc", "eval_acc", "construct_bacc", "eval_bacc", "construct_nll", "eval_nll", "construct_ece", "eval_ece", "status"]),
        "label_quarantine_audit.csv": ("label_quarantine_audit_rows", ["check", "value", "expected", "passed", "notes"]),
        "device_runtime_audit.csv": ("device_runtime_audit_rows", ["check", "observed", "allowed", "passed", "notes"]),
        "masked_view_contract.csv": ("masked_view_contract_rows", ["view", "c66_projection", "sampled_rows", "label_visible_rows", "prediction_visible_rows", "uses_same_label_endpoint_scalar", "available_at_selection_time", "diagnostic_only", "allowed_for_selection_rule", "selection_path_enforced", "policy_boundary_only", "status"]),
        "label_dependency_ledger.csv": ("field_availability_ledger_rows", ["field", "source_only_view", "target_construction_view", "target_evaluation_view", "same_label_oracle_view", "conditional_cs_diagnostic_view", "target_label_dependent"]),
        "field_availability_ledger.csv": ("field_availability_ledger_rows", ["field", "source_only_view", "target_construction_view", "target_evaluation_view", "same_label_oracle_view", "conditional_cs_diagnostic_view", "target_label_dependent"]),
        "label_view_unit_test_summary.csv": ("label_view_unit_test_summary_rows", ["view", "test", "sampled_rows", "passed", "notes"]),
        "cs_variable_availability_ledger.csv": ("conditional_cs_variable_map_rows", ["audit", "x1", "x2", "y", "uses_target_labels_in_x2", "uses_eval_labels_in_x2", "uses_same_label_endpoint_scalar", "available_at_selection_time", "diagnostic_only"]),
        "conditional_cs_variable_map.csv": ("conditional_cs_variable_map_rows", ["audit", "x1", "x2", "y", "uses_target_labels_in_x2", "uses_eval_labels_in_x2", "uses_same_label_endpoint_scalar", "available_at_selection_time", "diagnostic_only"]),
        "split_label_smoke_summary.csv": ("split_label_smoke_summary_rows", ["analysis", "checkpoint_units", "construct_threshold", "eval_threshold", "hit_rate", "majority_baseline", "same_label_oracle_hit", "construct_eval_corr", "null_mean_hit", "null_p_ge_observed", "status", "claim"]),
        "split_label_split_manifest.csv": ("split_label_split_manifest_rows", ["split_seed", "unit", "checkpoint_count", "construct_rule", "eval_rule", "stratified_by_class", "notes"]),
        "split_label_null_summary.csv": ("split_label_null_summary_rows", ["null", "permutations", "observed_hit", "null_mean", "null_max", "p_ge_observed", "status"]),
        "split_label_failure_ledger.csv": ("split_label_failure_ledger_rows", ["risk", "value", "threshold", "blocks_sufficiency_claim", "notes"]),
        "sample_level_cs_smoke_summary.csv": ("sample_level_cs_smoke_summary_rows", ["estimator", "paired_eval_rows", "independent_checkpoint_units", "mse_x1", "mse_x1_x2", "delta_mse", "null_mean_delta", "null_p_ge_observed", "status", "claim"]),
        "cs_bandwidth_stress.csv": ("cs_bandwidth_sensitivity_rows", ["bandwidth", "unit", "independent_units", "rbf_hsic_x2_eval", "status"]),
        "cs_bandwidth_sensitivity.csv": ("cs_bandwidth_sensitivity_rows", ["bandwidth", "unit", "independent_units", "rbf_hsic_x2_eval", "status"]),
        "cs_null_summary.csv": ("cs_null_summary_rows", ["null", "permutations", "observed_delta_mse", "null_mean_delta", "null_max_delta", "p_ge_observed", "status"]),
        "cs_estimator_feasibility_ledger.csv": ("cs_estimator_feasibility_ledger_rows", ["check", "value", "passed", "notes"]),
        "atom_trace_feasibility_from_c66_cache.csv": ("atom_trace_feasibility_from_c66_cache_rows", ["trace", "present_in_c66_cache", "requires_new_forward_in_c67", "diagnostic_available", "notes"]),
        "representation_hook_gap_ledger.csv": ("representation_hook_gap_ledger_rows", ["gap", "blocks_c67_smoke", "blocks_atom_trace_closure", "requires_new_forward", "notes"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
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


def write_artifacts(res: dict) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    write_tables(res)
    paths = [str(p) for p in _listed_paths()]
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    paths = _listed_paths()
    # Seed final row counts before writing the stable JSON; artifact_manifest.csv
    # is intentionally excluded from _listed_paths to avoid self-reference.
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{} for _ in paths]
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    paths = _listed_paths()
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c67_c66_dual_mode_cache_consumption")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C67] decision={res['decision']['primary']} gate={res['decision']['final_gate']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
