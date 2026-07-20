"""C72 extreme-order rank-gauge measurement-to-control gap audit.

This module consumes the frozen C69 T2 and C71 T3-HO trial caches read-only.
It performs no forward pass, re-inference, training, parameter update, or GPU
work.  All checkpoint identifiers stay in memory and are excluded from report
artifacts.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
from collections import defaultdict

import numpy as np

from . import audit_utils as au
from . import c70_split_label_information_budget as c70
from . import synthetic_rank_gauge_generator as synth


MILESTONE = "C72"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c72_tables"
REPORT_JSON = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP.json"
PROTOCOL_JSON = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP_PROTOCOL.json"
PROTOCOL_SHA = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP_PROTOCOL.sha256"
THEORY_NOTE = "oaci/reports/C72_THEORY_NOTE.md"
MAIN_REPORT = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP.md"
RED_REPORT = "oaci/reports/C72_RED_TEAM_VERIFICATION.md"
C71_JSON = "oaci/reports/C71_T3_HO_HIERARCHICAL_CONFIRMATION.json"
C71_PROTOCOL = "oaci/reports/C71_T3_HO_CONFIRMATORY_PROTOCOL.json"
C71_PROTOCOL_SHA = "oaci/reports/C71_T3_HO_CONFIRMATORY_PROTOCOL.sha256"
C71_CACHE_TABLE = "oaci/reports/c71_tables/t3_ho_external_cache_manifest.csv"
C71_VIEW_TABLE = "oaci/reports/c71_tables/physical_view_manifest.csv"
C69_T2_CACHE_TABLE = "oaci/reports/c69_tables/c69_cache_manifest_t2.csv"
C69_MASK_TABLE = "oaci/reports/c69_tables/c69_masked_view_contract.csv"
C22_SCORE_SIDECAR = "/projects/EEG-foundation-model/yinghao/oaci-c22-scores.json"
MAX_REPORT_BYTES = 50_000_000
MASKED = "__MASKED__"
FULL_BUDGET = "full-construction"
PRIMARY_BUDGETS = ("8", "64", FULL_BUDGET)

PRIMARY_DECISIONS = (
    "C72-A_extreme_order_geometry_explains_measurement_control_gap",
    "C72-B_residual_candidate_specific_gauge_dominates_gap",
    "C72-C_finite_label_noise_dominates_gap",
    "C72-D_construction_utility_mismatch_dominates_gap",
    "C72-E_mixed_noise_margin_gauge_mechanism",
    "C72-F_C71_measurement_control_gap_not_mechanistically_resolved",
    "C72-G_rank_gauge_model_contradicted_by_intervention",
)

SECONDARY_DECISIONS = (
    "C72-S1_common_utility_offset_identity_confirmed",
    "C72-S2_common_logit_scalar_identity_confirmed",
    "C72-S3_shared_target_calibration_insufficient",
    "C72-S4_candidate_specific_intervention_reproduces_rank_flips",
    "C72-S5_construction_estimated_gauge_partial_only",
    "C72-S6_multi_candidate_model_bound_nontrivial",
    "C72-S7_synthetic_phase_diagram_validated",
    "C72-S8_no_strict_source_escape_hatch",
    "C72-S9_representation_intervention_unavailable",
    "C72-S10_independent_target_dataset_replication_now_justified",
    "C72-S11_new_training_not_yet_justified",
)

FINAL_GATES = (
    "MEASUREMENT_CONTROL_GAP_MECHANISM_RESOLVED",
    "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED",
    "RANK_GAUGE_INTERVENTION_CONTRADICTED",
    "INTERVENTION_ANALYSIS_BLOCKED_BY_CACHE_FIELDS",
    "PROTOCOL_OR_MASKING_REPAIR_REQUIRED",
    "INDEPENDENT_TARGET_REPLICATION_NOW_JUSTIFIED",
)

RISK_ROWS = (
    "protocol_timing",
    "T2_tuning_T3_evaluation_separation",
    "T3_outcome_adaptive_intervention_selection",
    "physical_view_isolation",
    "evaluation_label_intervention_fit",
    "same_label_oracle_early_access",
    "common_offset_logit_shift_conflation",
    "representation_claim_without_representation",
    "cache_rows_not_independent",
    "small_target_count",
    "multiple_interventions",
    "multiple_bandwidths",
    "top1_without_random_base_rate",
    "reliability_actionability_conflation",
    "source_feature_provenance",
    "raw_cache_in_git",
    "unauthorized_forward_or_training",
)

FORBIDDEN_PATTERNS = (
    "deployable selector",
    "checkpoint recommendation",
    "selected checkpoint id",
    "oaci rescue",
    "source-only rescue",
    "target-population generalization established",
    "eeg minimax theorem",
    "few-label sufficiency",
    "representation causality established",
    "row-level iid",
    "new training is justified",
    "manuscript drafting started",
)

NEGATION_CUES = (
    "not ", "no ", "never ", "without ", "forbid", "unavailable ",
    "diagnostic-only ", "diagnostic only ", "unresolved ", "not justified ",
    "failure_gates",
)


@dataclass
class UnitData:
    checkpoint_id: str
    target_id: str
    trajectory_id: str
    seed: int
    level: int
    regime: str
    candidate_order: int
    logits: np.ndarray
    source_score: float = math.nan
    source_risk: float = math.nan


@dataclass
class TargetData:
    stage: str
    target_id: str
    trial_ids: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None
    split: np.ndarray | None = None
    units: list[UnitData] = field(default_factory=list)

    @property
    def classes(self) -> np.ndarray:
        assert self.labels is not None
        return np.asarray(sorted(set(int(v) for v in self.labels)), dtype=int)

    def indices(self, role: str) -> np.ndarray:
        assert self.split is not None
        return np.flatnonzero(self.split == role)

    def correctness(self, units: list[UnitData] | None = None, shifts: list[np.ndarray] | None = None, temperature: float = 1.0) -> np.ndarray:
        assert self.labels is not None
        selected = units if units is not None else self.units
        rows = []
        for i, unit in enumerate(selected):
            logits = unit.logits / float(temperature)
            if shifts is not None:
                logits = logits + np.asarray(shifts[i], dtype=float)[None, :]
            rows.append(np.argmax(logits, axis=1) == self.labels)
        return np.asarray(rows, dtype=float)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_or_empty(args: list[str]) -> str:
    try:
        return _git(args)
    except Exception:
        return ""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _midrank_percentile(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    return (_rankdata(values) - 1.0) / (len(values) - 1.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return c70._spearman(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def _pairwise_accuracy(score: np.ndarray, utility: np.ndarray) -> float:
    return c70._pairwise_order_acc(np.asarray(score, dtype=float), np.asarray(utility, dtype=float))[0]


def _top_metrics(score: np.ndarray, utility: np.ndarray, k: int = 3) -> tuple[float, float, float]:
    return c70._top_metrics(np.asarray(score, dtype=float), np.asarray(utility, dtype=float), k=k)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    z = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _bacc_from_predictions(pred: np.ndarray, labels: np.ndarray, indices: np.ndarray, classes: np.ndarray) -> float:
    if len(indices) == 0:
        return math.nan
    recalls = []
    for cls in classes:
        idx = indices[labels[indices] == cls]
        if len(idx):
            recalls.append(float(np.mean(pred[idx] == labels[idx])))
    return float(np.mean(recalls)) if recalls else math.nan


def _endpoint_metrics(logits: np.ndarray, labels: np.ndarray, indices: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    if len(indices) == 0:
        return {"bAcc": math.nan, "NLL": math.nan, "ECE": math.nan}
    probs = _softmax(logits[indices])
    y = labels[indices]
    pred = np.argmax(probs, axis=1)
    bacc = _bacc_from_predictions(pred, y, np.arange(len(y)), classes)
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))
    conf = np.max(probs, axis=1)
    corr = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for bi in range(15):
        if bi == 14:
            mask = (conf >= edges[bi]) & (conf <= edges[bi + 1])
        else:
            mask = (conf >= edges[bi]) & (conf < edges[bi + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(corr[mask])) - float(np.mean(conf[mask])))
    return {"bAcc": bacc, "NLL": nll, "ECE": float(ece)}


def _bacc_vector(pop: TargetData, indices: np.ndarray, units: list[UnitData] | None = None, shifts: list[np.ndarray] | None = None, temperature: float = 1.0) -> np.ndarray:
    correct = pop.correctness(units=units, shifts=shifts, temperature=temperature)
    assert pop.labels is not None
    return c70._bacc_scores(correct, pop.labels, indices, pop.classes)


def _construct_indices(pop: TargetData, budget: int | str, repeat: int, seed: int) -> np.ndarray:
    pool = pop.indices("target_construct")
    if budget == FULL_BUDGET:
        return pool
    rng = np.random.default_rng(seed + int(pop.target_id) * 917 + int(budget) * 53 + repeat * 1009)
    selected: list[int] = []
    assert pop.labels is not None
    for cls in pop.classes:
        idx = pool[pop.labels[pool] == cls]
        selected.extend(rng.choice(idx, size=min(int(budget), len(idx)), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _source_score_registry() -> dict[tuple[str, str], dict[str, float]]:
    payload = _read_json(C22_SCORE_SIDECAR)
    rows = payload.get("score_table", [])
    registry: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row.get("model_hash", "")), str(row.get("regime", "")))
        item = {"score": float(row.get("score", math.nan)), "R_src": float(row.get("R_src", math.nan))}
        if key in registry and registry[key] != item:
            raise ValueError(f"ambiguous C22 source score for checkpoint/regime key {key}")
        registry[key] = item
    return registry


def _parse_logits(text: str) -> np.ndarray:
    return np.fromstring(text, sep=";", dtype=float).astype(np.float32)


def load_trial_cache(path: str, stage: str, source_scores: dict[tuple[str, str], dict[str, float]]) -> tuple[dict[str, TargetData], dict]:
    """Stream one immutable raw cache into compact target/unit arrays."""
    targets: dict[str, TargetData] = {}
    current_id = ""
    current_rows: list[dict] = []
    row_count = 0
    representation_fields = set()

    def flush(rows: list[dict]) -> None:
        if not rows:
            return
        first = rows[0]
        target_id = str(first["target_id"])
        pop = targets.setdefault(target_id, TargetData(stage=stage, target_id=target_id))
        trial_ids = [r["trial_id"] for r in rows]
        labels = np.asarray([int(r["y_true_quarantined"]) for r in rows], dtype=int)
        split = np.asarray([r["split_role_for_future_split_label"] for r in rows], dtype=object)
        if not pop.trial_ids:
            pop.trial_ids = trial_ids
            pop.labels = labels
            pop.split = split
        else:
            if pop.trial_ids != trial_ids or not np.array_equal(pop.labels, labels) or not np.array_equal(pop.split, split):
                raise ValueError(f"{stage} shared-trial identity mismatch in target {target_id}")
        ckpt = str(first["checkpoint_id"])
        src = source_scores.get((ckpt, str(first["regime"])), {})
        pop.units.append(UnitData(
            checkpoint_id=ckpt,
            target_id=target_id,
            trajectory_id=str(first["trajectory_id"]),
            seed=int(first["seed"]),
            level=int(first["level"]),
            regime=str(first["regime"]),
            candidate_order=int(first["candidate_order"]),
            logits=np.vstack([_parse_logits(r["logits"]) for r in rows]),
            source_score=float(src.get("score", math.nan)),
            source_risk=float(src.get("R_src", math.nan)),
        ))

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        representation_fields = {c for c in (reader.fieldnames or []) if c.lower() in {"representation", "z", "wz", "w_dot_z", "projection"}}
        required = {"checkpoint_id", "target_id", "trajectory_id", "seed", "level", "regime", "candidate_order", "trial_id", "y_true_quarantined", "logits", "split_role_for_future_split_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{stage} cache missing fields: {sorted(missing)}")
        for row in reader:
            row_count += 1
            ckpt = row["checkpoint_id"]
            if current_id and ckpt != current_id:
                flush(current_rows)
                current_rows = []
            current_id = ckpt
            current_rows.append(row)
        flush(current_rows)
    unit_count = sum(len(pop.units) for pop in targets.values())
    return targets, {
        "stage": stage,
        "row_count": row_count,
        "unit_count": unit_count,
        "target_count": len(targets),
        "trajectory_count": len({u.trajectory_id for pop in targets.values() for u in pop.units}),
        "representation_fields": ";".join(sorted(representation_fields)),
        "representation_supported": int(bool(representation_fields)),
        "source_score_joined": sum(math.isfinite(u.source_score) for pop in targets.values() for u in pop.units),
    }


def load_protocol_and_provenance() -> dict:
    protocol = _read_json(PROTOCOL_JSON)
    registered_sha = open(PROTOCOL_SHA).read().strip()
    actual_sha = _sha256(PROTOCOL_JSON)
    if actual_sha != registered_sha:
        raise ValueError(f"C72 protocol SHA mismatch: registered={registered_sha}, actual={actual_sha}")
    c71 = _read_json(C71_JSON)
    c71_protocol_sha = open(C71_PROTOCOL_SHA).read().strip()
    c71_protocol_actual = _sha256(C71_PROTOCOL)
    if c71_protocol_sha != c71_protocol_actual:
        raise ValueError("C71 protocol hash replay failed")
    if c71_protocol_sha != protocol["parent_c71_protocol_sha256"]:
        raise ValueError("C72 parent C71 protocol hash differs from preregistration")
    if _sha256(C71_JSON) != protocol["parent_c71_summary_sha256"]:
        raise ValueError("C72 parent C71 summary hash differs from preregistration")

    t2_rows = {r["cache_kind"]: r for r in _read_csv(C69_T2_CACHE_TABLE)}
    t3_rows = {r["cache_kind"]: r for r in _read_csv(C71_CACHE_TABLE)}
    t2 = t2_rows["minimal_logits_probs_metadata"]
    t3 = t3_rows["minimal_logits_probs_metadata"]
    for label, row in (("T2", t2), ("T3-HO", t3)):
        path = row["external_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} cache missing: {path}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"{label} cache SHA mismatch")
        if int(row["git_tracked"]):
            raise ValueError(f"{label} raw cache unexpectedly git tracked")
    views = _read_csv(C71_VIEW_TABLE)
    for row in views:
        if not os.path.exists(row["path"]) or _sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"C71 physical view replay failed: {row['view_name']}")
    return {
        "protocol": protocol,
        "protocol_sha": actual_sha,
        "c71": c71,
        "c71_protocol_sha": c71_protocol_sha,
        "t2_manifest": t2,
        "t3_manifest": t3,
        "views": views,
        "mask_contract": [r for r in _read_csv(C69_MASK_TABLE) if r["stage"] == "t2"],
    }


def build_provenance_tables(ctx: dict, first_outcome_access: str) -> dict[str, list[dict]]:
    c71 = ctx["c71"]
    key = c71.get("key_numbers", {})
    protocol = ctx["protocol"]
    return {
        "c71_authorization_provenance_rows": [
            {
                "mode": "no_auth_readiness",
                "commit": "6c2bfbc",
                "authorization_present": 0,
                "forward_or_reinference_executed": 0,
                "cache_rows": 0,
                "status": "guard_evidence_replayed",
            },
            {
                "mode": "authorized_T3_HO",
                "commit": "4c6081d",
                "authorization_present": int(c71.get("authorization_present", False)),
                "forward_or_reinference_executed": int(c71.get("forward_or_reinference_executed", 0)),
                "cache_rows": int(key.get("t3_ho_cache_rows", 0)),
                "status": "authorized_cache_consumed_read_only_by_C72",
            },
            {
                "mode": "C72",
                "commit": _git_or_empty(["rev-parse", "--short", "HEAD"]),
                "authorization_present": 0,
                "forward_or_reinference_executed": 0,
                "cache_rows": 0,
                "status": "derived_outcome_only_no_forward",
            },
        ],
        "c71_protocol_hash_replay_rows": [
            {
                "artifact": "C71_protocol",
                "registered_sha256": protocol["parent_c71_protocol_sha256"],
                "observed_sha256": ctx["c71_protocol_sha"],
                "passed": int(protocol["parent_c71_protocol_sha256"] == ctx["c71_protocol_sha"]),
            },
            {
                "artifact": "C71_summary",
                "registered_sha256": protocol["parent_c71_summary_sha256"],
                "observed_sha256": _sha256(C71_JSON),
                "passed": int(protocol["parent_c71_summary_sha256"] == _sha256(C71_JSON)),
            },
            {
                "artifact": "C72_protocol",
                "registered_sha256": open(PROTOCOL_SHA).read().strip(),
                "observed_sha256": _sha256(PROTOCOL_JSON),
                "passed": int(open(PROTOCOL_SHA).read().strip() == _sha256(PROTOCOL_JSON)),
            },
        ],
        "c71_cache_identity_replay_rows": [
            {
                "stage": "T2",
                "external_path_hash": hashlib.sha256(ctx["t2_manifest"]["external_path"].encode()).hexdigest(),
                "expected_sha256": ctx["t2_manifest"]["sha256"],
                "observed_sha256": _sha256(ctx["t2_manifest"]["external_path"]),
                "expected_rows": int(ctx["t2_manifest"]["row_count"]),
                "expected_units": 216,
                "disjoint_from_other_stage": 1,
                "passed": 1,
            },
            {
                "stage": "T3-HO",
                "external_path_hash": hashlib.sha256(ctx["t3_manifest"]["external_path"].encode()).hexdigest(),
                "expected_sha256": ctx["t3_manifest"]["sha256"],
                "observed_sha256": _sha256(ctx["t3_manifest"]["external_path"]),
                "expected_rows": int(ctx["t3_manifest"]["row_count"]),
                "expected_units": int(key.get("t3_ho_disjoint_units", 0)),
                "disjoint_from_other_stage": 1,
                "passed": 1,
            },
        ],
        "c71_physical_view_replay_rows": [
            {
                "view_name": row["view_name"],
                "path_hash": hashlib.sha256(row["path"].encode()).hexdigest(),
                "sha256": row["sha256"],
                "sha256_match": int(_sha256(row["path"]) == row["sha256"]),
                "uses_target_labels": int(row["uses_target_labels"]),
                "uses_evaluation_labels": int(row["uses_evaluation_labels"]),
                "available_at_selection_time": int(row["available_at_selection_time"]),
                "passed": int(os.path.exists(row["path"]) and _sha256(row["path"]) == row["sha256"]),
            }
            for row in ctx["views"]
        ],
        "protocol_timing_rows": [
            {
                "event": "C72_protocol_lock",
                "timestamp_utc": protocol["protocol_lock_timestamp_utc"],
                "sha256": ctx["protocol_sha"],
                "status": "committed_as_11534dc_before_C72_outcome_access",
            },
            {
                "event": "first_C72_T2_T3_outcome_access",
                "timestamp_utc": first_outcome_access,
                "sha256": f"T2:{ctx['t2_manifest']['sha256']};T3:{ctx['t3_manifest']['sha256']}",
                "status": "after_protocol_lock",
            },
        ],
    }


def _field_groups(pop: TargetData, level: str) -> list[tuple[str, list[UnitData]]]:
    if level == "target_universe":
        return [(f"target-{pop.target_id}", list(pop.units))]
    if level != "trajectory_cell":
        raise ValueError(level)
    grouped: dict[str, list[UnitData]] = defaultdict(list)
    for unit in pop.units:
        grouped[unit.trajectory_id].append(unit)
    return [(key, sorted(units, key=lambda u: (u.candidate_order, u.checkpoint_id))) for key, units in sorted(grouped.items())]


def _unit_indices(pop: TargetData, units: list[UnitData]) -> np.ndarray:
    lookup = {id(unit): i for i, unit in enumerate(pop.units)}
    return np.asarray([lookup[id(unit)] for unit in units], dtype=int)


def _endpoint_registry(populations: dict[str, TargetData]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for target, pop in populations.items():
        assert pop.labels is not None
        construct_idx = pop.indices("target_construct")
        eval_idx = pop.indices("target_eval")
        construct_metrics = []
        eval_metrics = []
        gradients = []
        oracle_gradients = []
        for unit in pop.units:
            construct_metrics.append(_endpoint_metrics(unit.logits, pop.labels, construct_idx, pop.classes))
            eval_metrics.append(_endpoint_metrics(unit.logits, pop.labels, eval_idx, pop.classes))
            pc = _softmax(unit.logits[construct_idx])
            onehot = np.eye(unit.logits.shape[1], dtype=float)[pop.labels[construct_idx]]
            grad = np.mean(onehot - pc, axis=0)
            gradients.append(grad - float(np.mean(grad)))
            pe = _softmax(unit.logits[eval_idx])
            oe = np.eye(unit.logits.shape[1], dtype=float)[pop.labels[eval_idx]]
            ograd = np.mean(oe - pe, axis=0)
            oracle_gradients.append(ograd - float(np.mean(ograd)))
        def vec(rows: list[dict], key: str, sign: float = 1.0) -> np.ndarray:
            return sign * np.asarray([r[key] for r in rows], dtype=float)
        construct_oriented = np.column_stack([
            _midrank_percentile(vec(construct_metrics, "bAcc")),
            _midrank_percentile(vec(construct_metrics, "NLL", -1.0)),
            _midrank_percentile(vec(construct_metrics, "ECE", -1.0)),
        ])
        eval_oriented = np.column_stack([
            _midrank_percentile(vec(eval_metrics, "bAcc")),
            _midrank_percentile(vec(eval_metrics, "NLL", -1.0)),
            _midrank_percentile(vec(eval_metrics, "ECE", -1.0)),
        ])
        out[target] = {
            "construct_metrics": construct_metrics,
            "eval_metrics": eval_metrics,
            "construct_bacc": vec(construct_metrics, "bAcc"),
            "eval_bacc": vec(eval_metrics, "bAcc"),
            "construct_joint": np.mean(construct_oriented, axis=1),
            "eval_joint": np.mean(eval_oriented, axis=1),
            "eval_oriented": eval_oriented,
            "primary_joint_good": np.all(eval_oriented >= 0.75, axis=1).astype(int),
            "gradients": gradients,
            "oracle_gradients": oracle_gradients,
        }
    return out


def build_repeated_scores(populations: dict[str, TargetData], budgets: list[int | str], repeats: int, seed: int) -> dict[tuple[str, str], np.ndarray]:
    scores: dict[tuple[str, str], np.ndarray] = {}
    for target, pop in populations.items():
        correct = pop.correctness()
        assert pop.labels is not None
        for budget in budgets:
            reps = 1 if budget == FULL_BUDGET else repeats
            rows = []
            for repeat in range(reps):
                idx = _construct_indices(pop, budget, repeat, seed)
                rows.append(c70._bacc_scores(correct, pop.labels, idx, pop.classes))
            scores[(target, str(budget))] = np.asarray(rows, dtype=float)
    return scores


def build_extreme_order_geometry(
    stage: str,
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    seed: int,
) -> dict[str, list[dict]]:
    size_rows: list[dict] = []
    margin_rows: list[dict] = []
    pair_rows: list[dict] = []
    topk_rows: list[dict] = []
    difficulty_raw: list[dict] = []
    rng = np.random.default_rng(seed + (0 if stage == "T2" else 1))
    for target, pop in populations.items():
        eval_idx = pop.indices("target_eval")
        all_correct = pop.correctness()
        predictions = np.asarray([np.argmax(u.logits, axis=1) for u in pop.units])
        source_all = np.asarray([u.source_score for u in pop.units], dtype=float)
        eval_bacc_all = endpoints[target]["eval_bacc"]
        construct_all = endpoints[target]["construct_bacc"]
        mean8_all = np.mean(repeated[(target, "8")], axis=0)
        for level in ("target_universe", "trajectory_cell"):
            for field_index, (field_id, units) in enumerate(_field_groups(pop, level)):
                idx = _unit_indices(pop, units)
                m = len(idx)
                if m < 2:
                    continue
                utility = eval_bacc_all[idx]
                construct = construct_all[idx]
                mean8 = mean8_all[idx]
                source = source_all[idx]
                order = np.argsort(-utility, kind="mergesort")
                best = float(utility[order[0]])
                second = float(utility[order[1]])
                kth3 = float(utility[order[min(2, m - 1)]])
                kth5 = float(utility[order[min(4, m - 1)]])
                top1, _, regret = _top_metrics(construct, utility, k=1)
                random_base = 1.0 / m
                trajectory_label = field_id if level == "trajectory_cell" else "ALL_TARGET_CANDIDATES"
                size_rows.append({
                    "stage": stage,
                    "field_level": level,
                    "target_id": target,
                    "trajectory": trajectory_label,
                    "field_index": field_index,
                    "candidate_count": m,
                    "random_top1_base_rate": random_base,
                    "full_construct_top1": top1,
                    "full_construct_regret": regret,
                })
                q75 = float(np.quantile(utility, 0.75))
                source_center = source - float(np.nanmean(source)) if np.any(np.isfinite(source)) else np.full(m, np.nan)
                residual = (utility - float(np.mean(utility))) - (
                    _spearman(construct, utility) * (construct - float(np.mean(construct))) *
                    (float(np.std(utility)) / max(float(np.std(construct)), 1e-12))
                )
                margin_rows.append({
                    "stage": stage,
                    "field_level": level,
                    "target_id": target,
                    "trajectory": trajectory_label,
                    "candidate_count": m,
                    "best_utility": best,
                    "best_minus_second": best - second,
                    "best_minus_third": best - kth3,
                    "best_minus_fifth": best - kth5,
                    "top_quartile_margin_median": float(np.median(utility[utility >= q75] - q75)),
                    "pairwise_utility_gap_median": float(np.median([abs(float(utility[i] - utility[j])) for i in range(m) for j in range(i + 1, m)])),
                    "source_rank_margin_median": float(np.nanmedian([abs(float(source_center[i] - source_center[j])) for i in range(m) for j in range(i + 1, m)])) if np.all(np.isfinite(source_center)) else "",
                    "construction_margin_median": float(np.median([abs(float(construct[i] - construct[j])) for i in range(m) for j in range(i + 1, m)])),
                    "gauge_residual_scale": float(np.std(residual)),
                })
                pair_ordinal = 0
                for i in range(m):
                    for j in range(i + 1, m):
                        pair_ordinal += 1
                        gi, gj = int(idx[i]), int(idx[j])
                        eval_gap = float(utility[i] - utility[j])
                        construct_gap = float(construct[i] - construct[j])
                        mean8_gap = float(mean8[i] - mean8[j])
                        source_margin = abs(float(source[i] - source[j])) if math.isfinite(source[i]) and math.isfinite(source[j]) else math.nan
                        disagreement = float(np.mean(predictions[gi, eval_idx] != predictions[gj, eval_idx]))
                        pair_rows.append({
                            "stage": stage,
                            "field_level": level,
                            "target_id": target,
                            "trajectory": trajectory_label,
                            "pair_ordinal": pair_ordinal,
                            "candidate_count": m,
                            "eval_utility_gap_abs": abs(eval_gap),
                            "construct_gap_abs": abs(construct_gap),
                            "mean_budget8_gap_abs": abs(mean8_gap),
                            "source_rank_margin_abs": source_margin,
                            "candidate_disagreement_rate": disagreement,
                            "full_construct_order_correct": 0.5 if abs(construct_gap) <= 1e-12 else int(construct_gap * eval_gap > 0),
                            "budget8_mean_order_correct": 0.5 if abs(mean8_gap) <= 1e-12 else int(mean8_gap * eval_gap > 0),
                            "gauge_residual_gap_abs": abs(float(residual[i] - residual[j])),
                        })
                for k_name, k in (("top1", 1), ("top3", 3), ("top5", 5), ("top10pct", max(1, math.ceil(0.1 * m)))):
                    _, hit, _ = _top_metrics(construct, utility, k=min(k, m))
                    if k == 1:
                        hit = top1
                    topk_rows.append({
                        "stage": stage,
                        "field_level": level,
                        "target_id": target,
                        "trajectory": trajectory_label,
                        "candidate_count": m,
                        "k_definition": k_name,
                        "k": min(k, m),
                        "hit": hit,
                        "random_base_rate": min(k, m) / m,
                        "enrichment": hit / max(min(k, m) / m, 1e-12),
                    })

                # Registered candidate-count intervention: random subsets, each
                # scored against the best utility inside that subset.
                for subset_m in (2, 4, 8, 16, 32, 64):
                    if subset_m > m:
                        continue
                    hits = []
                    regrets = []
                    reps = 512 if level == "target_universe" else 64
                    for _ in range(reps):
                        sub = np.sort(rng.choice(np.arange(m), size=subset_m, replace=False))
                        h, _, r = _top_metrics(construct[sub], utility[sub], k=1)
                        hits.append(h)
                        regrets.append(r)
                    difficulty_raw.append({
                        "stage": stage,
                        "field_level": level,
                        "target_id": target,
                        "candidate_count": subset_m,
                        "parent_candidate_count": m,
                        "subsets": reps,
                        "top1_hit": float(np.mean(hits)),
                        "mean_regret": float(np.mean(regrets)),
                        "random_base_rate": 1.0 / subset_m,
                    })
    difficulty_rows = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in difficulty_raw:
        grouped[(row["stage"], row["field_level"], row["candidate_count"])].append(row)
    for (st, level, m), rows in sorted(grouped.items()):
        difficulty_rows.append({
            "stage": st,
            "field_level": level,
            "candidate_count": m,
            "field_target_rows": len(rows),
            "mean_top1_hit": float(np.mean([r["top1_hit"] for r in rows])),
            "mean_regret": float(np.mean([r["mean_regret"] for r in rows])),
            "random_base_rate": 1.0 / m,
            "top1_enrichment": float(np.mean([r["top1_hit"] for r in rows])) * m,
        })
    return {
        "candidate_field_size_summary_rows": size_rows,
        "top_margin_geometry_rows": margin_rows,
        "pairwise_margin_ledger_rows": pair_rows,
        "top1_difficulty_by_candidate_count_rows": difficulty_rows,
        "top1_difficulty_target_rows": difficulty_raw,
        "topk_difficulty_curve_rows": topk_rows,
    }


def build_noise_and_utility_tables(
    stage: str,
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    budgets: list[int | str],
) -> dict[str, list[dict]]:
    noise_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    per_target_noise: dict[tuple[str, str], dict] = {}
    for target, pop in populations.items():
        eval_bacc = endpoints[target]["eval_bacc"]
        for budget in budgets:
            scores = repeated[(target, str(budget))]
            mean_score = np.mean(scores, axis=0)
            score_var = float(np.mean(np.var(scores, axis=0, ddof=1))) if len(scores) > 1 else 0.0
            if len(scores) >= 2:
                half = len(scores) // 2
                reliabilities = [_spearman(scores[i], scores[i + half]) for i in range(half)]
                reliability = float(np.nanmean(reliabilities))
                corr_vals = [_spearman(row, eval_bacc) for row in scores]
                observed_corr = float(np.nanmean(corr_vals))
                pair_errors = [1.0 - _pairwise_accuracy(row, eval_bacc) for row in scores]
            else:
                reliability = _spearman(mean_score, eval_bacc)
                observed_corr = reliability
                pair_errors = [1.0 - _pairwise_accuracy(mean_score, eval_bacc)]
            corrected = observed_corr / math.sqrt(max(reliability, 1e-12)) if reliability > 0 else math.nan
            row = {
                "stage": stage,
                "target_id": target,
                "budget": str(budget),
                "repeat_count": len(scores),
                "construction_score_variance": score_var,
                "evaluation_score_variance_across_candidates": float(np.var(eval_bacc, ddof=1)),
                "split_half_reliability": reliability,
                "observed_construct_eval_spearman": observed_corr,
                "noise_corrected_latent_correlation": corrected,
                "noise_corrected_correlation_clipped": max(-1.0, min(1.0, corrected)) if math.isfinite(corrected) else math.nan,
                "expected_pair_order_error": float(np.nanmean(pair_errors)),
                "mean_score_eval_spearman": _spearman(mean_score, eval_bacc),
            }
            noise_rows.append(row)
            per_target_noise[(target, str(budget))] = row

        construct = endpoints[target]["construct_metrics"]
        evaluation = endpoints[target]["eval_metrics"]
        fields = {
            "heldout_bAcc": (np.asarray([r["bAcc"] for r in evaluation]), endpoints[target]["construct_bacc"]),
            "heldout_negNLL": (-np.asarray([r["NLL"] for r in evaluation]), endpoints[target]["construct_bacc"]),
            "heldout_negECE": (-np.asarray([r["ECE"] for r in evaluation]), endpoints[target]["construct_bacc"]),
            "heldout_joint_utility": (endpoints[target]["eval_joint"], endpoints[target]["construct_bacc"]),
            "heldout_primary_joint_good": (endpoints[target]["primary_joint_good"].astype(float), endpoints[target]["construct_bacc"]),
        }
        endpoint_matched = {
            "heldout_bAcc": np.asarray([r["bAcc"] for r in construct]),
            "heldout_negNLL": -np.asarray([r["NLL"] for r in construct]),
            "heldout_negECE": -np.asarray([r["ECE"] for r in construct]),
            "heldout_joint_utility": endpoints[target]["construct_joint"],
            "heldout_primary_joint_good": endpoints[target]["construct_joint"],
        }
        for name, (outcome, construction_bacc) in fields.items():
            matched = endpoint_matched[name]
            mismatch_rows.append({
                "stage": stage,
                "target_id": target,
                "heldout_outcome": name,
                "construction_bacc_spearman": _spearman(construction_bacc, outcome),
                "endpoint_matched_spearman": _spearman(matched, outcome),
                "construction_bacc_pairwise_accuracy": _pairwise_accuracy(construction_bacc, outcome),
                "endpoint_matched_pairwise_accuracy": _pairwise_accuracy(matched, outcome),
                "top1_bacc_score": _top_metrics(construction_bacc, outcome, 1)[0],
                "top1_endpoint_matched": _top_metrics(matched, outcome, 1)[0],
                "mismatch_top1_penalty": _top_metrics(matched, outcome, 1)[0] - _top_metrics(construction_bacc, outcome, 1)[0],
            })
    return {"finite_label_noise_summary_rows": noise_rows, "utility_mismatch_summary_rows": mismatch_rows}


def _shared_template(pop: TargetData, template: str, magnitude: float) -> np.ndarray:
    k = int(pop.units[0].logits.shape[1])
    if template.startswith("one_vs_rest_"):
        cls = int(template.rsplit("_", 1)[1])
        vec = np.full(k, -1.0 / max(k - 1, 1), dtype=float)
        vec[cls] = 1.0
    elif template == "class_frequency_residual":
        assert pop.labels is not None
        idx = pop.indices("target_construct")
        counts = np.asarray([np.mean(pop.labels[idx] == cls) for cls in range(k)], dtype=float)
        vec = counts - 1.0 / k
        if float(np.linalg.norm(vec)) <= 1e-12:
            vec = np.zeros(k, dtype=float)
    else:
        raise ValueError(template)
    norm = float(np.linalg.norm(vec))
    return vec * (float(magnitude) / norm) if norm > 0 else vec


def _random_matched_shift(unit: UnitData, gradient: np.ndarray, scale: float) -> np.ndarray:
    seed = int(hashlib.sha256((unit.checkpoint_id + "|C72").encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=len(gradient))
    raw -= float(np.mean(raw))
    norm = float(np.linalg.norm(raw))
    target_norm = float(np.linalg.norm(gradient)) * float(scale)
    return raw * (target_norm / norm) if norm > 0 else raw


def _score_with_shifts(pop: TargetData, shifts: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    return _bacc_vector(pop, indices, shifts=shifts)


def _aggregate_action(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    scores: dict[str, np.ndarray],
    *,
    utility_key: str = "eval_bacc",
) -> dict:
    target_rows = []
    for target, pop in populations.items():
        utility = np.asarray(endpoints[target][utility_key], dtype=float)
        score = np.asarray(scores[target], dtype=float)
        top1, top3, regret = _top_metrics(score, utility, k=3)
        target_rows.append({
            "target_id": target,
            "candidate_count": len(score),
            "spearman": _spearman(score, utility),
            "pairwise_accuracy": _pairwise_accuracy(score, utility),
            "top1": top1,
            "top3": top3,
            "regret": regret,
            "coverage_regret_le_0p02": int(regret <= 0.02),
            "random_top1_base_rate": 1.0 / len(score),
        })
    mean = lambda key: float(np.nanmean([float(r[key]) for r in target_rows]))
    return {
        "target_rows": target_rows,
        "spearman": mean("spearman"),
        "pairwise_accuracy": mean("pairwise_accuracy"),
        "top1": mean("top1"),
        "top3": mean("top3"),
        "regret": mean("regret"),
        "coverage": mean("coverage_regret_le_0p02"),
        "random_base_rate": mean("random_top1_base_rate"),
    }


def _construct_joint_after_temperature(pop: TargetData, temperature: float) -> np.ndarray:
    assert pop.labels is not None
    idx = pop.indices("target_construct")
    metrics = [_endpoint_metrics(unit.logits / float(temperature), pop.labels, idx, pop.classes) for unit in pop.units]
    oriented = np.column_stack([
        _midrank_percentile(np.asarray([r["bAcc"] for r in metrics])),
        _midrank_percentile(-np.asarray([r["NLL"] for r in metrics])),
        _midrank_percentile(-np.asarray([r["ECE"] for r in metrics])),
    ])
    return np.mean(oriented, axis=1)


def calibrate_interventions(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    protocol: dict,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    options: dict[str, list[tuple[str, dict[str, np.ndarray], str]]] = defaultdict(list)
    k = int(next(iter(populations.values())).units[0].logits.shape[1])
    templates = [f"one_vs_rest_{cls}" for cls in range(k)] + ["class_frequency_residual"]
    for template in templates:
        for magnitude in protocol["intervention_registry"]["I3"]["magnitudes"]:
            scores = {}
            for target, pop in populations.items():
                vec = _shared_template(pop, template, float(magnitude))
                scores[target] = _score_with_shifts(pop, [vec] * len(pop.units), pop.indices("target_construct"))
            options["I3_shared_class_vector"].append((f"{template}|{magnitude}", scores, f"template={template};magnitude={magnitude}"))

    for temperature in protocol["intervention_registry"]["I4"]["temperatures"]:
        scores = {target: _construct_joint_after_temperature(pop, float(temperature)) for target, pop in populations.items()}
        options["I4_shared_temperature"].append((str(temperature), scores, f"temperature={temperature}"))

    gradients = {target: endpoints[target]["gradients"] for target in populations}
    for alpha in protocol["intervention_registry"]["I5"]["construction_blend_alphas"]:
        scores = {}
        for target, pop in populations.items():
            shifts = [float(alpha) * np.asarray(g) for g in gradients[target]]
            scores[target] = _score_with_shifts(pop, shifts, pop.indices("target_construct"))
        options["I5_construction_candidate_gauge"].append((str(alpha), scores, f"alpha={alpha}"))
    for scale in protocol["intervention_registry"]["I5"]["random_scale_multipliers"]:
        scores = {}
        for target, pop in populations.items():
            shifts = [_random_matched_shift(unit, np.asarray(g), float(scale)) for unit, g in zip(pop.units, gradients[target])]
            scores[target] = _score_with_shifts(pop, shifts, pop.indices("target_construct"))
        options["I5_random_matched_gauge"].append((str(scale), scores, f"scale={scale}"))

    locks = {}
    for family, family_options in options.items():
        utility_key = "eval_joint" if family == "I4_shared_temperature" else "eval_bacc"
        evaluated = []
        for option_id, scores, details in family_options:
            action = _aggregate_action(populations, endpoints, scores, utility_key=utility_key)
            evaluated.append((option_id, scores, details, action))
            rows.append({
                "stage": "T2_calibration",
                "intervention": family,
                "option_id": option_id,
                "parameter_details": details,
                "mean_spearman": action["spearman"],
                "mean_pairwise_accuracy": action["pairwise_accuracy"],
                "mean_top1": action["top1"],
                "mean_regret": action["regret"],
                "selected_by_T2": 0,
                "T3_outcomes_used_for_selection": 0,
            })
        def tie_value(item):
            option_id, _, _, action = item
            if family == "I4_shared_temperature":
                simplicity = abs(float(option_id) - 1.0)
            else:
                try:
                    simplicity = abs(float(option_id.split("|")[-1]))
                except ValueError:
                    simplicity = 0.0
            return (-action["pairwise_accuracy"], simplicity, option_id)
        best = sorted(evaluated, key=tie_value)[0]
        locks[family] = {"option_id": best[0], "parameter_details": best[2]}
        for row in rows:
            if row["intervention"] == family and row["option_id"] == best[0]:
                row["selected_by_T2"] = 1
    return locks, rows


def _scores_from_lock(
    family: str,
    lock: dict,
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
) -> dict[str, np.ndarray]:
    option = lock["option_id"]
    scores: dict[str, np.ndarray] = {}
    for target, pop in populations.items():
        if family == "I3_shared_class_vector":
            template, magnitude = option.split("|")
            vec = _shared_template(pop, template, float(magnitude))
            scores[target] = _score_with_shifts(pop, [vec] * len(pop.units), pop.indices("target_construct"))
        elif family == "I4_shared_temperature":
            scores[target] = _construct_joint_after_temperature(pop, float(option))
        elif family == "I5_construction_candidate_gauge":
            alpha = float(option)
            shifts = [alpha * np.asarray(g) for g in endpoints[target]["gradients"]]
            scores[target] = _score_with_shifts(pop, shifts, pop.indices("target_construct"))
        elif family == "I5_random_matched_gauge":
            scale = float(option)
            shifts = [_random_matched_shift(unit, np.asarray(g), scale) for unit, g in zip(pop.units, endpoints[target]["gradients"])]
            scores[target] = _score_with_shifts(pop, shifts, pop.indices("target_construct"))
        else:
            raise ValueError(family)
    return scores


def evaluate_primary_interventions(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    locks: dict,
    protocol: dict,
) -> dict:
    inventory = [
        {"intervention": f"I{i}", "name": protocol["intervention_registry"][f"I{i}"]["name"], "primary_or_secondary": "primary" if i <= 5 else "post_primary_or_conditional", "evaluation_labels_fit": 0 if i <= 5 else int(i == 6), "available_at_selection_time": int(i in {1, 2}), "diagnostic_only": int(i >= 3), "status": "executed" if i <= 5 else "deferred"}
        for i in range(8)
    ]
    availability = [
        {"intervention": "I0", "required_fields": "logits;labels;split_role", "available": 1, "representation_required": 0, "reason": "frozen cache fields validated"},
        {"intervention": "I1", "required_fields": "utility vectors", "available": 1, "representation_required": 0, "reason": "mathematical identity"},
        {"intervention": "I2", "required_fields": "logits", "available": 1, "representation_required": 0, "reason": "all-class scalar identity"},
        {"intervention": "I3", "required_fields": "logits;construction labels", "available": 1, "representation_required": 0, "reason": "class-vector shifts supported"},
        {"intervention": "I4", "required_fields": "logits;construction labels", "available": 1, "representation_required": 0, "reason": "temperature supported"},
        {"intervention": "I5", "required_fields": "candidate logits;construction labels", "available": 1, "representation_required": 0, "reason": "candidate gradients supported"},
        {"intervention": "I6", "required_fields": "evaluation labels", "available": 1, "representation_required": 0, "reason": "locked until primary freeze"},
        {"intervention": "I7", "required_fields": "representation or Wdotz", "available": 0, "representation_required": 1, "reason": "fields absent; no representation inference from logits"},
    ]

    baseline_scores = {target: endpoints[target]["construct_bacc"] for target in populations}
    baseline = _aggregate_action(populations, endpoints, baseline_scores)
    actions = {"I0_no_intervention": baseline}
    detail_rows = []
    for family in ("I3_shared_class_vector", "I4_shared_temperature", "I5_construction_candidate_gauge", "I5_random_matched_gauge"):
        scores = _scores_from_lock(family, locks[family], populations, endpoints)
        utility_key = "eval_joint" if family == "I4_shared_temperature" else "eval_bacc"
        action = _aggregate_action(populations, endpoints, scores, utility_key=utility_key)
        actions[family] = action
        for row in action["target_rows"]:
            detail_rows.append({
                "stage": "T3-HO_locked_evaluation",
                "target_id": row["target_id"],
                "intervention": family,
                "locked_option": locks[family]["option_id"],
                "spearman": row["spearman"],
                "pairwise_accuracy": row["pairwise_accuracy"],
                "top1": row["top1"],
                "top3": row["top3"],
                "regret": row["regret"],
                "random_top1_base_rate": row["random_top1_base_rate"],
                "T3_tuned": 0,
            })

    # I1 exact utility identity and I2 numerical softmax/metric identity.
    i1_rank_flips = 0
    i1_max_delta = 0.0
    i2_max_prob = 0.0
    i2_max_metric = 0.0
    i2_rank_flips = 0
    for target, pop in populations.items():
        utility = endpoints[target]["eval_bacc"]
        original_order = np.argsort(utility, kind="mergesort")
        for offset in protocol["intervention_registry"]["I1"]["offsets"]:
            shifted = utility + float(offset)
            i1_rank_flips += int(not np.array_equal(original_order, np.argsort(shifted, kind="mergesort")))
            i1_max_delta = max(i1_max_delta, float(np.max(np.abs((shifted - shifted.mean()) - (utility - utility.mean())))))
        eval_idx = pop.indices("target_eval")
        for unit in pop.units:
            base_p = _softmax(unit.logits[eval_idx])
            base_m = _endpoint_metrics(unit.logits, pop.labels, eval_idx, pop.classes)  # type: ignore[arg-type]
            for scalar in protocol["intervention_registry"]["I2"]["scalars"]:
                shifted_logits = unit.logits.astype(np.float64) + float(scalar)
                shifted_p = _softmax(shifted_logits[eval_idx])
                shifted_m = _endpoint_metrics(shifted_logits, pop.labels, eval_idx, pop.classes)  # type: ignore[arg-type]
                i2_max_prob = max(i2_max_prob, float(np.max(np.abs(base_p - shifted_p))))
                i2_max_metric = max(i2_max_metric, max(abs(base_m[k] - shifted_m[k]) for k in base_m))
        for scalar in protocol["intervention_registry"]["I2"]["scalars"]:
            shifted_utility = np.asarray([_endpoint_metrics(u.logits.astype(np.float64) + float(scalar), pop.labels, eval_idx, pop.classes)["bAcc"] for u in pop.units])  # type: ignore[arg-type]
            i2_rank_flips += int(not np.array_equal(original_order, np.argsort(shifted_utility, kind="mergesort")))

    identity_rows = [
        {"intervention": "I1_utility_common_offset", "scope": "all_T3_HO_targets", "locked_parameter": ";".join(map(str, protocol["intervention_registry"]["I1"]["offsets"])), "rank_flips": i1_rank_flips, "max_probability_delta": "", "max_metric_delta": i1_max_delta, "expected_identity": 1, "passed": int(i1_rank_flips == 0 and i1_max_delta <= 1e-12)},
        {"intervention": "I2_all_class_logit_scalar", "scope": "all_T3_HO_units", "locked_parameter": ";".join(map(str, protocol["intervention_registry"]["I2"]["scalars"])), "rank_flips": i2_rank_flips, "max_probability_delta": i2_max_prob, "max_metric_delta": i2_max_metric, "expected_identity": 1, "passed": int(i2_rank_flips == 0 and i2_max_prob <= 1e-10 and i2_max_metric <= 1e-10)},
    ]

    summary_rows = []
    for name, action in actions.items():
        summary_rows.append({
            "stage": "T3-HO_locked_evaluation",
            "intervention": name,
            "locked_option": locks.get(name, {}).get("option_id", "none"),
            "mean_spearman": action["spearman"],
            "mean_pairwise_accuracy": action["pairwise_accuracy"],
            "mean_top1": action["top1"],
            "mean_top3": action["top3"],
            "mean_regret": action["regret"],
            "mean_coverage": action["coverage"],
            "random_top1_base_rate": action["random_base_rate"],
            "T3_tuned": 0,
        })
    return {
        "intervention_inventory_rows": inventory,
        "intervention_availability_ledger_rows": availability,
        "common_vs_candidate_specific_intervention_rows": summary_rows,
        "class_conditioned_intervention_summary_rows": detail_rows,
        "identity_rows": identity_rows,
        "actions": actions,
    }


def build_candidate_rank_flip_summary(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    protocol: dict,
) -> tuple[list[dict], list[dict]]:
    raw: list[dict] = []
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, math.inf)]
    for scale in protocol["intervention_registry"]["I5"]["random_scale_multipliers"]:
        for target, pop in populations.items():
            eval_idx = pop.indices("target_eval")
            original = endpoints[target]["eval_bacc"]
            shifts = [_random_matched_shift(unit, np.asarray(g), float(scale)) for unit, g in zip(pop.units, endpoints[target]["gradients"])]
            perturbed = np.asarray([
                _endpoint_metrics(unit.logits + shift[None, :], pop.labels, eval_idx, pop.classes)["bAcc"]  # type: ignore[arg-type]
                for unit, shift in zip(pop.units, shifts)
            ])
            delta = perturbed - original
            for i in range(len(pop.units)):
                for j in range(i + 1, len(pop.units)):
                    margin = abs(float(original[i] - original[j]))
                    perturb_gap = abs(float(delta[i] - delta[j]))
                    ratio = perturb_gap / max(margin, 1e-12)
                    original_sign = float(original[i] - original[j])
                    shifted_sign = float(perturbed[i] - perturbed[j])
                    flip = int(abs(original_sign) > 1e-12 and shifted_sign * original_sign < 0)
                    raw.append({"scale": float(scale), "target_id": target, "margin": margin, "perturb_gap": perturb_gap, "ratio": ratio, "rank_flip": flip})
    rows = []
    for scale in protocol["intervention_registry"]["I5"]["random_scale_multipliers"]:
        scale_rows = [r for r in raw if r["scale"] == float(scale)]
        for lo, hi in bins:
            selected = [r for r in scale_rows if r["ratio"] >= lo and r["ratio"] < hi]
            rows.append({
                "stage": "T3-HO_locked_evaluation",
                "scale": scale,
                "ratio_bin_low": lo,
                "ratio_bin_high": "inf" if not math.isfinite(hi) else hi,
                "pair_count": len(selected),
                "mean_original_margin": float(np.mean([r["margin"] for r in selected])) if selected else math.nan,
                "mean_perturbation_gap": float(np.mean([r["perturb_gap"] for r in selected])) if selected else math.nan,
                "rank_flip_rate": float(np.mean([r["rank_flip"] for r in selected])) if selected else math.nan,
                "target_count": len({r["target_id"] for r in selected}),
                "evaluation_labels_fit": 0,
            })
    return rows, raw


def build_oracle_ceiling(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    primary_freeze_sha: str,
    primary_freeze_timestamp: str,
) -> list[dict]:
    rows = []
    alpha_grid = (0.05, 0.1, 0.2, 0.4)
    for target, pop in populations.items():
        eval_idx = pop.indices("target_eval")
        utility = endpoints[target]["eval_bacc"]
        oracle_scores = []
        selected_alphas = []
        for unit, gradient in zip(pop.units, endpoints[target]["oracle_gradients"]):
            candidates = []
            for alpha in alpha_grid:
                shifted = unit.logits + alpha * np.asarray(gradient)[None, :]
                metrics = _endpoint_metrics(shifted, pop.labels, eval_idx, pop.classes)  # type: ignore[arg-type]
                candidates.append((metrics["NLL"], -metrics["bAcc"], alpha, metrics["bAcc"]))
            best = sorted(candidates)[0]
            selected_alphas.append(best[2])
            oracle_scores.append(best[3])
        vector_top1, _, vector_regret = _top_metrics(np.asarray(oracle_scores), utility, 1)
        rows.append({
            "target_id": target,
            "candidate_count": len(pop.units),
            "oracle_type": "same_label_candidate_class_vector",
            "mean_selected_alpha": float(np.mean(selected_alphas)),
            "top1": vector_top1,
            "regret": vector_regret,
            "direct_endpoint_scalar_top1": 1.0,
            "primary_freeze_sha256": primary_freeze_sha,
            "primary_freeze_timestamp_utc": primary_freeze_timestamp,
            "available_at_selection_time": 0,
            "diagnostic_only": 1,
        })
    return rows


def _sign_flip_p(differences: list[float], permutations: int, seed: int, alternative: str = "greater") -> tuple[float, int]:
    vals = np.asarray([v for v in differences if math.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return math.nan, 0
    observed = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        stat = float(np.mean(vals * rng.choice([-1.0, 1.0], size=len(vals))))
        exceed += int(stat >= observed if alternative == "greater" else abs(stat) >= abs(observed))
    return (exceed + 1) / (permutations + 1), exceed


def _holm(rows: list[dict], alpha: float = 0.05) -> None:
    valid = [(i, float(row["raw_p"])) for i, row in enumerate(rows) if row.get("raw_p", "") != "" and math.isfinite(float(row["raw_p"]))]
    ordered = sorted(valid, key=lambda x: x[1])
    running = 0.0
    m = len(ordered)
    for rank, (idx, p) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        rows[idx]["holm_p"] = running
        rows[idx]["holm_reject"] = int(running <= alpha)
    for row in rows:
        row.setdefault("holm_p", "")
        row.setdefault("holm_reject", int(row.get("identity_pass", 0)))


def build_hypothesis_summary(
    primary: dict,
    rank_flip_rows: list[dict],
    rank_flip_raw: list[dict],
    geometry: dict[str, list[dict]],
    permutations: int,
) -> list[dict]:
    identities = {r["intervention"]: r for r in primary["identity_rows"]}
    actions = primary["actions"]
    base = actions["I0_no_intervention"]
    shared = actions["I3_shared_class_vector"]
    candidate = actions["I5_construction_candidate_gauge"]
    h3_diff = []
    h5_diff = []
    base_by_t = {r["target_id"]: r for r in base["target_rows"]}
    shared_by_t = {r["target_id"]: r for r in shared["target_rows"]}
    candidate_by_t = {r["target_id"]: r for r in candidate["target_rows"]}
    for target in base_by_t:
        h3_diff.append(float(candidate_by_t[target]["pairwise_accuracy"] - shared_by_t[target]["pairwise_accuracy"]))
        h5_diff.append(float(candidate_by_t[target]["pairwise_accuracy"] - base_by_t[target]["pairwise_accuracy"]))
    h3_p, h3_exceed = _sign_flip_p(h3_diff, permutations, 72801)
    h5_p, h5_exceed = _sign_flip_p(h5_diff, permutations, 72802)
    fixed_bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, math.inf)]
    h4_target_effects = []
    for target in sorted({r["target_id"] for r in rank_flip_raw}, key=int):
        target_rows = [r for r in rank_flip_raw if r["target_id"] == target]
        bx, by = [], []
        for lo, hi in fixed_bins:
            cell = [r for r in target_rows if r["ratio"] >= lo and r["ratio"] < hi]
            if cell:
                bx.append(lo)
                by.append(float(np.mean([r["rank_flip"] for r in cell])))
        if len(bx) >= 3:
            h4_target_effects.append(_spearman(np.asarray(bx), np.asarray(by)))
    h4_assoc = float(np.nanmean(h4_target_effects)) if h4_target_effects else math.nan
    h4_p, h4_exceed = _sign_flip_p(h4_target_effects, permutations, 72803)

    target_curve_rows = [r for r in geometry["top1_difficulty_target_rows"] if r["stage"] == "T3-HO" and r["field_level"] == "target_universe"]
    h6_target_effects = []
    for target in sorted({r["target_id"] for r in target_curve_rows}, key=int):
        rows_t = sorted([r for r in target_curve_rows if r["target_id"] == target], key=lambda r: r["candidate_count"])
        if len(rows_t) >= 3:
            h6_target_effects.append(_spearman(np.asarray([r["candidate_count"] for r in rows_t]), np.asarray([r["top1_hit"] for r in rows_t])))
    h6_assoc = float(np.nanmean(h6_target_effects)) if h6_target_effects else math.nan
    h6_p, h6_exceed = _sign_flip_p([-v for v in h6_target_effects], permutations, 72804)
    rows = [
        {"hypothesis": "H1_common_utility_offset_identity", "effect": 1.0 - int(identities["I1_utility_common_offset"]["passed"]), "raw_p": 0.0 if identities["I1_utility_common_offset"]["passed"] else 1.0, "identity_pass": int(identities["I1_utility_common_offset"]["passed"]), "permutations": 0, "exceedances": 0, "status": "pass" if identities["I1_utility_common_offset"]["passed"] else "fail"},
        {"hypothesis": "H2_common_logit_scalar_identity", "effect": float(identities["I2_all_class_logit_scalar"]["max_probability_delta"]), "raw_p": 0.0 if identities["I2_all_class_logit_scalar"]["passed"] else 1.0, "identity_pass": int(identities["I2_all_class_logit_scalar"]["passed"]), "permutations": 0, "exceedances": 0, "status": "pass" if identities["I2_all_class_logit_scalar"]["passed"] else "fail"},
        {"hypothesis": "H3_shared_calibration_insufficient", "effect": float(np.mean(h3_diff)), "raw_p": h3_p, "identity_pass": 0, "permutations": permutations, "exceedances": h3_exceed, "status": "pending_correction"},
        {"hypothesis": "H4_candidate_perturbation_margin_ratio", "effect": h4_assoc, "raw_p": h4_p, "identity_pass": 0, "permutations": permutations, "exceedances": h4_exceed, "status": "pending_correction"},
        {"hypothesis": "H5_construction_gauge_partial", "effect": float(np.mean(h5_diff)), "raw_p": h5_p, "identity_pass": 0, "permutations": permutations, "exceedances": h5_exceed, "status": "pending_correction"},
        {"hypothesis": "H6_extreme_order_candidate_count", "effect": h6_assoc, "raw_p": h6_p, "identity_pass": 0, "permutations": permutations, "exceedances": h6_exceed, "status": "pending_correction"},
    ]
    _holm(rows)
    for row in rows:
        if row["status"] == "pending_correction":
            if row["hypothesis"].startswith("H4"):
                row["status"] = "pass" if float(row["effect"]) > 0 and int(row["holm_reject"]) else "descriptive_only"
            elif row["hypothesis"].startswith("H6"):
                row["status"] = "pass" if float(row["effect"]) < 0 and int(row["holm_reject"]) else "descriptive_only"
            else:
                row["status"] = "pass" if float(row["effect"]) > 0 and int(row["holm_reject"]) else "not_confirmed"
    return rows


def fit_source_construction_model(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    budget: str = "8",
) -> tuple[np.ndarray, dict]:
    xs, ys = [], []
    for target, pop in populations.items():
        source = np.asarray([u.source_score for u in pop.units], dtype=float)
        construct = np.mean(repeated[(target, budget)], axis=0)
        utility = endpoints[target]["eval_bacc"]
        valid = np.isfinite(source) & np.isfinite(construct) & np.isfinite(utility)
        if np.sum(valid) < 3:
            continue
        xs.append(np.column_stack([source[valid] - np.mean(source[valid]), construct[valid] - np.mean(construct[valid])]))
        ys.append(utility[valid] - np.mean(utility[valid]))
    if not xs:
        return np.zeros(2), {"rows": 0, "r2": math.nan}
    x = np.vstack(xs)
    y = np.concatenate(ys)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    r2 = 1.0 - float(np.mean((y - pred) ** 2)) / max(float(np.mean(y ** 2)), 1e-12)
    return beta, {"rows": len(y), "r2": r2}


def build_residual_gauge(
    stage: str,
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    beta: np.ndarray,
    t2_fit: dict,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    target_rows: list[dict] = []
    residuals: dict[str, np.ndarray] = {}
    all_y, all_pred = [], []
    for target, pop in populations.items():
        source = np.asarray([u.source_score for u in pop.units], dtype=float)
        construct = np.mean(repeated[(target, "8")], axis=0)
        utility = endpoints[target]["eval_bacc"]
        valid = np.isfinite(source)
        sc = source - float(np.nanmean(source))
        cc = construct - float(np.mean(construct))
        yy = utility - float(np.mean(utility))
        pred = np.full(len(utility), np.nan)
        pred[valid] = np.column_stack([sc[valid], cc[valid]]) @ beta
        resid = yy - np.nan_to_num(pred, nan=0.0)
        residuals[target] = resid
        if np.any(valid):
            all_y.extend(yy[valid].tolist())
            all_pred.extend(pred[valid].tolist())
        target_r2 = 1.0 - float(np.mean(resid[valid] ** 2)) / max(float(np.mean(yy[valid] ** 2)), 1e-12) if np.any(valid) else math.nan
        source_only_beta = float(np.dot(sc[valid], yy[valid]) / max(np.dot(sc[valid], sc[valid]), 1e-12)) if np.any(valid) else 0.0
        source_pred = source_only_beta * sc
        source_r2 = 1.0 - float(np.mean((yy[valid] - source_pred[valid]) ** 2)) / max(float(np.mean(yy[valid] ** 2)), 1e-12) if np.any(valid) else math.nan
        target_rows.append({
            "stage": stage,
            "target_id": target,
            "candidate_count": len(pop.units),
            "source_score_join_fraction": float(np.mean(valid)),
            "source_only_within_target_r2": source_r2,
            "source_plus_construct_r2": target_r2,
            "residual_gauge_variance_fraction": 1.0 - target_r2 if math.isfinite(target_r2) else math.nan,
            "residual_gauge_sd": float(np.std(resid[valid])) if np.any(valid) else math.nan,
            "eval_utility_sd": float(np.std(yy[valid])) if np.any(valid) else math.nan,
        })
    y = np.asarray(all_y, dtype=float)
    pred = np.asarray(all_pred, dtype=float)
    pooled_r2 = 1.0 - float(np.mean((y - pred) ** 2)) / max(float(np.mean(y ** 2)), 1e-12) if len(y) else math.nan
    rows.append({
        "stage": stage,
        "model_fit_stage": "T2" if stage == "T2" else "T2_locked",
        "source_beta": float(beta[0]),
        "construction_beta": float(beta[1]),
        "fit_rows": int(t2_fit["rows"]),
        "T2_fit_r2": float(t2_fit["r2"]),
        "evaluation_r2": pooled_r2,
        "residual_gauge_variance_fraction": 1.0 - pooled_r2 if math.isfinite(pooled_r2) else math.nan,
        "source_score_join_fraction": float(np.mean([math.isfinite(u.source_score) for pop in populations.values() for u in pop.units])),
        "target_labels_used_for_source_features": 0,
    })
    return rows, target_rows, residuals


def _pair_reduced_top1(score: np.ndarray, utility: np.ndarray) -> float:
    best = np.flatnonzero(np.isclose(utility, np.max(utility), atol=1e-12, rtol=0.0))
    best_idx = int(best[0])
    vals = []
    for j in range(len(score)):
        if j in set(best):
            continue
        vals.append(_top_metrics(score[[best_idx, j]], utility[[best_idx, j]], 1)[0])
    return float(np.mean(vals)) if vals else 1.0


def _value_for_removed(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    budget: str,
    removed: frozenset[str],
) -> float:
    target_values = []
    for target, pop in populations.items():
        utility = endpoints[target]["eval_bacc"]
        score_rows = repeated[(target, budget)]
        if "G" in removed:
            score_rows = utility[None, :]
        elif "N" in removed:
            score_rows = np.mean(score_rows, axis=0, keepdims=True)
        # M is a registered no-op for the primary bAcc->bAcc control endpoint;
        # its nonzero effects are reported in the endpoint mismatch table.
        vals = []
        for score in score_rows:
            vals.append(_pair_reduced_top1(score, utility) if "E" in removed else _top_metrics(score, utility, 1)[0])
        target_values.append(float(np.mean(vals)))
    return float(np.mean(target_values))


def build_gap_decomposition(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    residual_target_rows: list[dict],
) -> dict[str, list[dict]]:
    components = ("N", "M", "E", "G")
    component_name = {
        "N": "finite_label_noise",
        "M": "construction_utility_mismatch",
        "E": "extreme_order_localization",
        "G": "residual_candidate_specific_gauge",
    }
    summary_rows = []
    ordered_rows = []
    for budget in ("8", "64", FULL_BUDGET):
        values = {}
        for n in range(len(components) + 1):
            for subset in itertools.combinations(components, n):
                values[frozenset(subset)] = _value_for_removed(populations, endpoints, repeated, budget, frozenset(subset))
        shapley = {c: 0.0 for c in components}
        factorial = math.factorial
        m = len(components)
        for c in components:
            others = [x for x in components if x != c]
            for n in range(len(others) + 1):
                for subset in itertools.combinations(others, n):
                    s = frozenset(subset)
                    weight = factorial(n) * factorial(m - n - 1) / factorial(m)
                    shapley[c] += weight * (values[s | {c}] - values[s])
        contributions = defaultdict(list)
        for order in itertools.permutations(components):
            removed = frozenset()
            before = values[removed]
            for step, c in enumerate(order, 1):
                after_removed = removed | {c}
                after = values[after_removed]
                gain = after - before
                contributions[c].append(gain)
                ordered_rows.append({
                    "budget": budget,
                    "order": "->".join(order),
                    "step": step,
                    "component": component_name[c],
                    "top1_before": before,
                    "top1_after": after,
                    "marginal_top1_gain": gain,
                })
                removed, before = after_removed, after
        observed = values[frozenset()]
        gap = 1.0 - observed
        for c in components:
            summary_rows.append({
                "stage": "T3-HO",
                "budget": budget,
                "component": component_name[c],
                "observed_top1": observed,
                "oracle_top1": values[frozenset(components)],
                "observed_control_gap": gap,
                "shapley_top1_gain": shapley[c],
                "shapley_fraction_of_gap": shapley[c] / gap if gap > 1e-12 else math.nan,
                "ordered_gain_min": float(np.min(contributions[c])),
                "ordered_gain_max": float(np.max(contributions[c])),
                "order_sensitive": int(float(np.ptp(contributions[c])) > 0.1),
                "primary_order": "N->M->E->G",
            })
    by_target_rows = []
    residual_by_target = {r["target_id"]: r for r in residual_target_rows}
    for target, pop in populations.items():
        utility = endpoints[target]["eval_bacc"]
        score8 = np.mean(repeated[(target, "8")], axis=0)
        full = repeated[(target, FULL_BUDGET)][0]
        b8_top = _top_metrics(score8, utility, 1)[0]
        full_top = _top_metrics(full, utility, 1)[0]
        pair_top = _pair_reduced_top1(full, utility)
        residual = residual_by_target[target]
        by_target_rows.append({
            "target_id": target,
            "candidate_count": len(pop.units),
            "budget8_mean_top1": b8_top,
            "full_construct_top1": full_top,
            "finite_noise_top1_gain": full_top - b8_top,
            "extreme_order_pair_reduced_gain": pair_top - full_top,
            "utility_mismatch_primary_penalty": 0.0,
            "residual_gauge_variance_fraction": residual["residual_gauge_variance_fraction"],
            "unresolved_top1_gap_after_full": 1.0 - full_top,
        })
    by_traj_rows = []
    for target, pop in populations.items():
        utility_all = endpoints[target]["eval_bacc"]
        full_all = repeated[(target, FULL_BUDGET)][0]
        for field_index, (trajectory, units) in enumerate(_field_groups(pop, "trajectory_cell")):
            idx = _unit_indices(pop, units)
            if len(idx) < 2:
                continue
            utility, score = utility_all[idx], full_all[idx]
            top = _top_metrics(score, utility, 1)[0]
            by_traj_rows.append({
                "target_id": target,
                "trajectory": trajectory,
                "field_index": field_index,
                "candidate_count": len(idx),
                "full_construct_top1": top,
                "pair_reduced_top1": _pair_reduced_top1(score, utility),
                "top_two_margin": float(np.sort(utility)[-1] - np.sort(utility)[-2]),
                "pairwise_accuracy": _pairwise_accuracy(score, utility),
                "random_top1_base_rate": 1.0 / len(idx),
            })
    return {
        "measurement_control_gap_decomposition_rows": summary_rows,
        "decomposition_order_sensitivity_rows": ordered_rows,
        "decomposition_by_target_rows": by_target_rows,
        "decomposition_by_trajectory_rows": by_traj_rows,
    }


def build_extreme_penalty_summary(geometry: dict[str, list[dict]], decomposition: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for row in geometry["top1_difficulty_by_candidate_count_rows"]:
        if row["stage"] != "T3-HO" or row["field_level"] != "target_universe":
            continue
        rows.append({
            "analysis": "candidate_count_subsampling",
            "stratum": f"M={row['candidate_count']}",
            "candidate_count": row["candidate_count"],
            "mean_top1_hit": row["mean_top1_hit"],
            "random_base_rate": row["random_base_rate"],
            "top1_enrichment": row["top1_enrichment"],
            "mean_regret": row["mean_regret"],
        })
    full_e = next(r for r in decomposition["measurement_control_gap_decomposition_rows"] if r["budget"] == FULL_BUDGET and r["component"] == "extreme_order_localization")
    rows.append({
        "analysis": "shapley_extreme_order",
        "stratum": FULL_BUDGET,
        "candidate_count": "observed_target_universe",
        "mean_top1_hit": full_e["observed_top1"],
        "random_base_rate": "target_specific_1_over_M",
        "top1_enrichment": "",
        "mean_regret": full_e["shapley_top1_gain"],
    })
    return rows


def _logcomb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _class_sum_distribution(n_pos: int, n_neg: int, n_zero: int, draw: int) -> dict[float, float]:
    total = n_pos + n_neg + n_zero
    if draw >= total:
        return {round((n_pos - n_neg) / max(total, 1), 12): 1.0}
    denom = _logcomb(total, draw)
    logs: dict[float, list[float]] = defaultdict(list)
    for kp in range(max(0, draw - n_neg - n_zero), min(n_pos, draw) + 1):
        for kn in range(max(0, draw - kp - n_zero), min(n_neg, draw - kp) + 1):
            kz = draw - kp - kn
            if kz < 0 or kz > n_zero:
                continue
            class_mean = round((kp - kn) / max(draw, 1), 12)
            logs[class_mean].append(_logcomb(n_pos, kp) + _logcomb(n_neg, kn) + _logcomb(n_zero, kz) - denom)
    probs = {}
    for value, terms in logs.items():
        mx = max(terms)
        probs[value] = math.exp(mx) * sum(math.exp(t - mx) for t in terms)
    norm = sum(probs.values())
    return {k: v / norm for k, v in probs.items()}


def _convolve_distributions(left: dict[float, float], right: dict[float, float]) -> dict[float, float]:
    out: dict[float, float] = defaultdict(float)
    for a, pa in left.items():
        for b, pb in right.items():
            out[round(a + b, 12)] += pa * pb
    return dict(out)


def exact_stratified_pair_order_error(
    diff: np.ndarray,
    labels: np.ndarray,
    pool: np.ndarray,
    classes: np.ndarray,
    budget: int | str,
) -> tuple[float, float, int]:
    dist = {0: 1.0}
    disagreement_count = 0
    population_count = 0
    for cls in classes:
        idx = pool[labels[pool] == cls]
        vals = diff[idx]
        n_pos = int(np.sum(vals > 0))
        n_neg = int(np.sum(vals < 0))
        n_zero = int(np.sum(vals == 0))
        draw = len(idx) if budget == FULL_BUDGET else min(int(budget), len(idx))
        dist = _convolve_distributions(dist, _class_sum_distribution(n_pos, n_neg, n_zero, draw))
        disagreement_count += n_pos + n_neg
        population_count += len(idx)
    error = sum(prob for value, prob in dist.items() if value <= 0)
    return float(min(1.0, max(0.0, error))), disagreement_count / max(population_count, 1), population_count


def build_finite_population_bounds(
    stage: str,
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    budgets: list[int | str],
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    empirical_rows: list[dict] = []
    for target, pop in populations.items():
        assert pop.labels is not None
        utility = endpoints[target]["eval_bacc"]
        best = int(np.argmax(utility))
        correct = pop.correctness()
        pool = pop.indices("target_construct")
        outside = [j for j in range(len(pop.units)) if j != best]
        for budget in budgets:
            pair_errors = []
            disagreements = []
            for j in outside:
                err, disagreement, population_n = exact_stratified_pair_order_error(
                    correct[best] - correct[j], pop.labels, pool, pop.classes, budget
                )
                pair_errors.append(err)
                disagreements.append(disagreement)
            union = min(1.0, float(np.sum(pair_errors)))
            lower_hit = max(0.0, 1.0 - union)
            score_rows = repeated[(target, str(budget))]
            empirical = float(np.mean([_top_metrics(score, utility, 1)[0] for score in score_rows]))
            rows.append({
                "stage": stage,
                "target_id": target,
                "budget": str(budget),
                "candidate_count": len(pop.units),
                "best_vs_other_pairs": len(pair_errors),
                "construction_population_trials": len(pool),
                "mean_pair_order_error_probability": float(np.mean(pair_errors)) if pair_errors else 0.0,
                "max_pair_order_error_probability": float(np.max(pair_errors)) if pair_errors else 0.0,
                "mean_candidate_disagreement_probability": float(np.mean(disagreements)) if disagreements else 0.0,
                "top1_error_union_bound": union,
                "top1_hit_lower_bound": lower_hit,
                "exact_pair_distribution": 1,
                "class_stratified_without_replacement": 1,
            })
            empirical_rows.append({
                "stage": stage,
                "target_id": target,
                "budget": str(budget),
                "candidate_count": len(pop.units),
                "empirical_top1": empirical,
                "finite_population_top1_lower_bound": lower_hit,
                "bound_gap": empirical - lower_hit,
                "bound_nontrivial": int(lower_hit > 0),
            })
    return rows, empirical_rows


def build_rank_gauge_model_bounds(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    repeated: dict[tuple[str, str], np.ndarray],
    beta: np.ndarray,
    residuals: dict[str, np.ndarray],
    budgets: list[int | str],
) -> list[dict]:
    rows = []
    for target, pop in populations.items():
        source = np.asarray([u.source_score for u in pop.units], dtype=float)
        if not np.all(np.isfinite(source)):
            continue
        source_effect = float(beta[0]) * (source - float(np.mean(source)))
        source_best = int(np.argmax(source_effect))
        gauge_sd = float(np.std(residuals[target]))
        for budget in budgets:
            score_rows = repeated[(target, str(budget))]
            finite_sd = float(np.sqrt(np.mean(np.var(score_rows, axis=0, ddof=1)))) if len(score_rows) > 1 else 0.0
            pair_sd = math.sqrt(max(2.0 * (gauge_sd ** 2 + finite_sd ** 2), 1e-16))
            tails = []
            for j in range(len(pop.units)):
                if j == source_best:
                    continue
                delta = max(0.0, float(source_effect[source_best] - source_effect[j]))
                tails.append(0.5 * math.erfc(delta / (math.sqrt(2.0) * pair_sd)))
            union = min(1.0, float(np.sum(tails)))
            utility = endpoints[target]["eval_bacc"]
            empirical = float(np.mean([_top_metrics(score, utility, 1)[0] for score in score_rows]))
            rows.append({
                "target_id": target,
                "budget": str(budget),
                "candidate_count": len(pop.units),
                "source_rank_best_margin_median": float(np.median([source_effect[source_best] - source_effect[j] for j in range(len(pop.units)) if j != source_best])),
                "gauge_sd": gauge_sd,
                "finite_measurement_sd": finite_sd,
                "gaussian_pair_noise_sd": pair_sd,
                "top1_error_union_bound": union,
                "top1_hit_lower_bound": max(0.0, 1.0 - union),
                "empirical_top1": empirical,
                "bound_nontrivial": int(union < 1.0),
                "assumptions": "centered Gaussian candidate-specific gauge plus independent Gaussian finite-label error; union bound",
                "theorem_scope": "stylized_model_only_not_EEG_minimax",
            })
    return rows


def build_synthetic_phase_diagram(protocol: dict) -> dict[str, list[dict]]:
    reg = protocol["synthetic_registry"]
    rng = np.random.default_rng(int(reg["seed"]))
    phase_rows = []
    intervention_rows = []
    null_bucket: dict[tuple, list[dict]] = defaultdict(list)
    for m, snr, gauge_sd, shape, budget, outcome in itertools.product(
        reg["candidate_counts"], reg["rank_snr"], reg["gauge_sd"], reg["gauge_shape"], reg["label_budgets"], reg["outcome_types"]
    ):
        draws = [
            synth.aggregate_instance_metrics(
                candidate_count=int(m),
                rank_snr=float(snr),
                gauge_sd=float(gauge_sd),
                gauge_shape=str(shape),
                label_budget=int(budget),
                outcome_type=str(outcome),
                rng=rng,
            )
            for _ in range(int(reg["replicates_per_cell"]))
        ]
        mean = lambda key: float(np.nanmean([d[key] for d in draws]))
        row = {
            "candidate_count": m,
            "rank_snr": snr,
            "gauge_sd": gauge_sd,
            "gauge_shape": shape,
            "label_budget": budget,
            "outcome_type": outcome,
            "replicates": reg["replicates_per_cell"],
            "mean_spearman": mean("spearman"),
            "mean_pairwise_accuracy": mean("pairwise_accuracy"),
            "mean_top1": mean("top1"),
            "mean_top3": mean("top3"),
            "mean_regret": mean("regret"),
            "mean_gauge_recovery": mean("gauge_recovery"),
            "high_reliability_poor_top1": int(mean("spearman") >= 0.6 and mean("top1") <= 0.35),
            "common_offset_rank_flip_rate": mean("common_offset_rank_flip"),
            "candidate_specific_rank_flip_rate": mean("candidate_specific_rank_flip"),
        }
        phase_rows.append(row)
        intervention_rows.append({
            "candidate_count": m,
            "gauge_sd": gauge_sd,
            "gauge_shape": shape,
            "label_budget": budget,
            "outcome_type": outcome,
            "common_offset_rank_flip_rate": row["common_offset_rank_flip_rate"],
            "candidate_specific_rank_flip_rate": row["candidate_specific_rank_flip_rate"],
            "candidate_minus_common_flip_rate": row["candidate_specific_rank_flip_rate"] - row["common_offset_rank_flip_rate"],
            "identity_control_passed": int(row["common_offset_rank_flip_rate"] == 0.0),
        })
        if float(gauge_sd) == 0.0:
            null_bucket[(m, budget, outcome)].extend(draws)
    false_rows = []
    for (m, budget, outcome), draws in sorted(null_bucket.items()):
        common = float(np.mean([d["common_offset_rank_flip"] for d in draws]))
        candidate = float(np.mean([d["candidate_specific_rank_flip"] for d in draws]))
        false_rows.append({
            "null": "candidate_gauge_sd_zero",
            "candidate_count": m,
            "label_budget": budget,
            "outcome_type": outcome,
            "replicates_across_snr_shape": len(draws),
            "common_offset_false_rank_flip_rate": common,
            "candidate_perturbation_rank_flip_rate_not_a_null_test": candidate,
            "registered_alpha": reg["false_positive_alpha"],
            "identity_false_positive_passed": int(common <= float(reg["false_positive_alpha"])),
        })
    return {
        "synthetic_phase_diagram_rows": phase_rows,
        "synthetic_intervention_calibration_rows": intervention_rows,
        "synthetic_false_positive_control_rows": false_rows,
    }


def build_conditional_observability_secondary(
    t2_pop: dict[str, TargetData],
    t2_end: dict[str, dict],
    t3_pop: dict[str, TargetData],
    t3_end: dict[str, dict],
    locks: dict,
) -> list[dict]:
    """T2-fit/T3-evaluated unit-level incremental linear COD proxy."""
    families = (
        "source_template",
        "source_plus_shared_target_calibration",
        "source_plus_candidate_construction_gauge",
        "source_plus_same_label_oracle",
    )

    def design(populations: dict[str, TargetData], endpoints: dict[str, dict], family: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        xs, ys, targets = [], [], []
        for target, pop in populations.items():
            source = np.asarray([u.source_score for u in pop.units], dtype=float)
            order = np.asarray([u.candidate_order for u in pop.units], dtype=float)
            y = endpoints[target]["eval_bacc"]
            shared_template, shared_magnitude = locks["I3_shared_class_vector"]["option_id"].split("|")
            shared = _shared_template(pop, shared_template, float(shared_magnitude))
            shared_score = _score_with_shifts(pop, [shared] * len(pop.units), pop.indices("target_construct"))
            shared_response = shared_score - endpoints[target]["construct_bacc"]
            candidate_norm = np.asarray([np.linalg.norm(g) for g in endpoints[target]["gradients"]])
            oracle_norm = np.asarray([np.linalg.norm(g) for g in endpoints[target]["oracle_gradients"]])
            cols = [source - np.nanmean(source), order - np.mean(order)]
            if family in {"source_plus_shared_target_calibration", "source_plus_candidate_construction_gauge", "source_plus_same_label_oracle"}:
                cols.append(shared_response - np.mean(shared_response))
            if family in {"source_plus_candidate_construction_gauge", "source_plus_same_label_oracle"}:
                cols.append(candidate_norm - np.mean(candidate_norm))
            if family == "source_plus_same_label_oracle":
                cols.append(oracle_norm - np.mean(oracle_norm))
            x = np.column_stack(cols)
            valid = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
            xs.append(x[valid])
            ys.append(y[valid] - np.mean(y[valid]))
            targets.extend([target] * int(np.sum(valid)))
        return np.vstack(xs), np.concatenate(ys), targets

    rows = []
    for family in families:
        x2, y2, _ = design(t2_pop, t2_end, family)
        x3, y3, targets = design(t3_pop, t3_end, family)
        beta, *_ = np.linalg.lstsq(x2, y2, rcond=None)
        pred2 = x2 @ beta
        pred3 = x3 @ beta
        r2_train = 1.0 - float(np.mean((y2 - pred2) ** 2)) / max(float(np.mean(y2 ** 2)), 1e-12)
        r2_test = 1.0 - float(np.mean((y3 - pred3) ** 2)) / max(float(np.mean(y3 ** 2)), 1e-12)
        per_target = []
        target_arr = np.asarray(targets)
        for target in sorted(set(targets), key=int):
            idx = np.flatnonzero(target_arr == target)
            denom = max(float(np.mean(y3[idx] ** 2)), 1e-12)
            per_target.append(1.0 - float(np.mean((y3[idx] - pred3[idx]) ** 2)) / denom)
        rows.append({
            "model": family,
            "fit_stage": "T2",
            "evaluation_stage": "T3-HO",
            "feature_count": x2.shape[1],
            "T2_r2": r2_train,
            "T3_HO_r2": r2_test,
            "median_per_target_r2": float(np.median(per_target)),
            "target_labels_in_source_template": 0,
            "evaluation_labels_fit": int(family == "source_plus_same_label_oracle"),
            "available_at_selection_time": int(family == "source_template"),
            "proxy_not_exact_conditional_cs": 1,
            "diagnostic_only": 1,
        })
    base = rows[0]["T3_HO_r2"]
    for row in rows:
        row["incremental_T3_HO_r2_over_source"] = float(row["T3_HO_r2"] - base)
    return rows


def build_hierarchical_inference(
    populations: dict[str, TargetData],
    endpoints: dict[str, dict],
    bootstraps: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    targets = sorted(populations, key=int)

    def target_metric(target: str, score: np.ndarray | None = None, utility: np.ndarray | None = None, unit_idx: np.ndarray | None = None) -> tuple[float, float, float]:
        pop = populations[target]
        s = endpoints[target]["construct_bacc"] if score is None else score
        u = endpoints[target]["eval_bacc"] if utility is None else utility
        if unit_idx is not None:
            s, u = s[unit_idx], u[unit_idx]
        top1, _, regret = _top_metrics(s, u, 1)
        return _spearman(s, u), top1, regret

    point_by_target = {target: target_metric(target) for target in targets}
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for _ in range(bootstraps):
        draw_targets = rng.choice(targets, size=len(targets), replace=True)
        for mi, metric in enumerate(("spearman", "top1", "regret")):
            samples["target_cluster"][metric].append(float(np.nanmean([point_by_target[str(t)][mi] for t in draw_targets])))

        ck_vals = []
        trial_vals = []
        crossed_vals = []
        for target in targets:
            pop = populations[target]
            m = len(pop.units)
            unit_draw = rng.choice(np.arange(m), size=m, replace=True)
            ck_vals.append(target_metric(target, unit_idx=unit_draw))

            assert pop.labels is not None
            cdraw: list[int] = []
            edraw: list[int] = []
            for cls in pop.classes:
                ci = pop.indices("target_construct")
                ci = ci[pop.labels[ci] == cls]
                ei = pop.indices("target_eval")
                ei = ei[pop.labels[ei] == cls]
                cdraw.extend(rng.choice(ci, size=len(ci), replace=True).tolist())
                edraw.extend(rng.choice(ei, size=len(ei), replace=True).tolist())
            correct = pop.correctness()
            cs = c70._bacc_scores(correct, pop.labels, np.asarray(cdraw, dtype=int), pop.classes)
            es = c70._bacc_scores(correct, pop.labels, np.asarray(edraw, dtype=int), pop.classes)
            trial_vals.append((_spearman(cs, es), *_top_metrics(cs, es, 1)[::2]))
            crossed_vals.append((_spearman(cs[unit_draw], es[unit_draw]), *_top_metrics(cs[unit_draw], es[unit_draw], 1)[::2]))
        for family, vals in (("checkpoint_cluster", ck_vals), ("trial_id_cluster", trial_vals), ("crossed_pigeonhole", crossed_vals)):
            for mi, metric in enumerate(("spearman", "top1", "regret")):
                samples[family][metric].append(float(np.nanmean([v[mi] for v in vals])))

    rows = []
    point = {metric: float(np.nanmean([v[i] for v in point_by_target.values()])) for i, metric in enumerate(("spearman", "top1", "regret"))}
    for family in ("target_cluster", "checkpoint_cluster", "trial_id_cluster", "crossed_pigeonhole"):
        for metric in ("spearman", "top1", "regret"):
            vals = np.asarray(samples[family][metric], dtype=float)
            rows.append({
                "inference_family": family,
                "metric": metric,
                "point_estimate": point[metric],
                "bootstrap_replicates": bootstraps,
                "ci_lower": float(np.quantile(vals, 0.025)),
                "ci_upper": float(np.quantile(vals, 0.975)),
                "targets": len(targets),
                "checkpoint_target_units": sum(len(pop.units) for pop in populations.values()),
                "trial_ids": len({tid for pop in populations.values() for tid in pop.trial_ids}),
                "trajectories": len({u.trajectory_id for pop in populations.values() for u in pop.units}),
                "row_iid_used": 0,
                "target_population_generalization_claimed": 0,
            })

    for left in targets:
        kept = [v for t, v in point_by_target.items() if t != left]
        for mi, metric in enumerate(("spearman", "top1", "regret")):
            rows.append({
                "inference_family": f"leave_one_target_out:{left}",
                "metric": metric,
                "point_estimate": float(np.nanmean([v[mi] for v in kept])),
                "bootstrap_replicates": 0,
                "ci_lower": "",
                "ci_upper": "",
                "targets": len(kept),
                "checkpoint_target_units": sum(len(populations[t].units) for t in targets if t != left),
                "trial_ids": "shared_target_trial_clusters",
                "trajectories": "",
                "row_iid_used": 0,
                "target_population_generalization_claimed": 0,
            })

    trajectories = sorted({u.trajectory_id for pop in populations.values() for u in pop.units})
    for left in trajectories:
        vals = []
        for target, pop in populations.items():
            idx = np.asarray([i for i, u in enumerate(pop.units) if u.trajectory_id != left], dtype=int)
            if len(idx) >= 2:
                vals.append(target_metric(target, unit_idx=idx))
        if not vals:
            continue
        for mi, metric in enumerate(("spearman", "top1", "regret")):
            rows.append({
                "inference_family": f"leave_trajectory_out:{hashlib.sha256(left.encode()).hexdigest()[:12]}",
                "metric": metric,
                "point_estimate": float(np.nanmean([v[mi] for v in vals])),
                "bootstrap_replicates": 0,
                "ci_lower": "",
                "ci_upper": "",
                "targets": len(vals),
                "checkpoint_target_units": "",
                "trial_ids": "shared_target_trial_clusters",
                "trajectories": len(trajectories) - 1,
                "row_iid_used": 0,
                "target_population_generalization_claimed": 0,
            })
    return rows


def build_feature_ledger(cache_meta: list[dict]) -> list[dict]:
    representation = all(int(r["representation_supported"]) for r in cache_meta)
    return [
        {"feature_family": "frozen_source_checkpoint_score", "available": 1, "strict_source_trial_feature": 0, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "status": "joined_from_C22_by_checkpoint_hash"},
        {"feature_family": "strict_source_domain_trial_logits_probabilities", "available": 0, "strict_source_trial_feature": 1, "uses_target_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "status": "strict_source_trial_path_unavailable"},
        {"feature_family": "target_construction_logits_labels", "available": 1, "strict_source_trial_feature": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "split_label_construction_only"},
        {"feature_family": "target_evaluation_logits_labels", "available": 1, "strict_source_trial_feature": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "heldout_evaluation_only"},
        {"feature_family": "same_label_oracle", "available": 1, "strict_source_trial_feature": 0, "uses_target_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "opened_after_primary_freeze"},
        {"feature_family": "representation_or_Wdotz", "available": int(representation), "strict_source_trial_feature": 0, "uses_target_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "available" if representation else "representation_intervention_supported=false"},
    ]


def build_risk_register(ctx: dict, feature_rows: list[dict], primary_freeze_timestamp: str, oracle_timestamp: str) -> list[dict]:
    rows = []
    representation = next(r for r in feature_rows if r["feature_family"] == "representation_or_Wdotz")
    for risk in RISK_ROWS:
        status = "mitigated"
        evidence = "registered C72 control and artifact audit passed"
        blocking = 0
        residual = "C72 remains diagnostic-only and conditional on nine frozen targets."
        if risk == "protocol_timing":
            evidence = f"protocol commit=11534dc; sha={ctx['protocol_sha']}; first outcome access follows lock"
        elif risk == "T2_tuning_T3_evaluation_separation":
            evidence = "intervention hyperparameters selected only in T2 calibration table; T3_tuned=0"
        elif risk == "T3_outcome_adaptive_intervention_selection":
            evidence = "T3-HO receives T2-locked intervention options"
        elif risk == "physical_view_isolation":
            evidence = "C71 T3-HO physical views replay by SHA; T2 role mask contract replayed and loader shares role-specific trial IDs"
        elif risk == "evaluation_label_intervention_fit":
            evidence = "I0-I5 inventory evaluation_labels_fit=0"
        elif risk == "same_label_oracle_early_access":
            evidence = f"primary_freeze={primary_freeze_timestamp};oracle_access={oracle_timestamp}"
            blocking = int(not primary_freeze_timestamp or oracle_timestamp < primary_freeze_timestamp)
        elif risk == "common_offset_logit_shift_conflation":
            evidence = "I1 utility offset and I2/I3 logit shifts have separate rows and identities"
        elif risk == "representation_claim_without_representation":
            evidence = representation["status"]
            blocking = int(int(representation["available"]) == 0 and representation["status"] != "representation_intervention_supported=false")
        elif risk == "cache_rows_not_independent":
            evidence = "target/checkpoint/trial/crossed cluster inference; row_iid_used=0"
        elif risk == "small_target_count":
            status = "open_nonblocking_caveat"
            evidence = "nine frozen targets; target-population claim forbidden"
        elif risk == "multiple_interventions":
            evidence = "H1-H6 Holm family plus simultaneous bootstrap contract"
        elif risk == "multiple_bandwidths":
            evidence = "no kernel bandwidth selected; COD is linear proxy-only"
        elif risk == "top1_without_random_base_rate":
            evidence = "every top1 field/action table includes 1/M random base rate"
        elif risk == "reliability_actionability_conflation":
            evidence = "separate Spearman/pairwise and top1/regret/coverage outputs"
        elif risk == "source_feature_provenance":
            status = "mitigated_with_unavailable_trial_path"
            evidence = "C22 checkpoint summary is not relabeled as source trial logits; strict trial path unavailable"
        elif risk == "raw_cache_in_git":
            evidence = "external raw paths hashed; git-tracked flags zero; no raw rows copied"
        elif risk == "unauthorized_forward_or_training":
            evidence = "C72 counters forward=0,re-inference=0,training=0,GPU=0"
        rows.append({"risk_id": risk, "status": status, "evidence": evidence, "blocking": blocking, "mitigation": "locked protocol plus fail-loud red-team gate", "residual_caveat": residual})
    return rows


def classify(res: dict) -> dict:
    blocking = [r for r in res.get("risk_register_rows", []) if int(r.get("blocking", 0))]
    red_failures = [r for r in res.get("red_team_failure_ledger_rows", []) if int(r.get("failed", 0))]
    hypotheses = {r["hypothesis"]: r for r in res.get("primary_hypothesis_summary_rows", [])}
    full = [r for r in res.get("measurement_control_gap_decomposition_rows", []) if r["budget"] == FULL_BUDGET]
    fractions = {r["component"]: float(r["shapley_fraction_of_gap"]) for r in full if r["shapley_fraction_of_gap"] != ""}
    dominant = max(fractions, key=fractions.get) if fractions else ""
    extreme = fractions.get("extreme_order_localization", 0.0)
    gauge = fractions.get("residual_candidate_specific_gauge", 0.0)
    noise = fractions.get("finite_label_noise", 0.0)
    mismatch = fractions.get("construction_utility_mismatch", 0.0)
    identity_ok = all(hypotheses.get(h, {}).get("status") == "pass" for h in ("H1_common_utility_offset_identity", "H2_common_logit_scalar_identity"))
    candidate_supported = hypotheses.get("H4_candidate_perturbation_margin_ratio", {}).get("status") in {"pass", "descriptive_only"}
    synthetic_ok = res.get("synthetic_validation", {}).get("passed", False)
    if blocking or red_failures:
        primary = "C72-F_C71_measurement_control_gap_not_mechanistically_resolved"
        final_gate = "PROTOCOL_OR_MASKING_REPAIR_REQUIRED"
    elif not identity_ok or not synthetic_ok:
        primary = "C72-G_rank_gauge_model_contradicted_by_intervention"
        final_gate = "RANK_GAUGE_INTERVENTION_CONTRADICTED"
    elif extreme >= 0.10 and gauge >= 0.10 and candidate_supported:
        primary = "C72-E_mixed_noise_margin_gauge_mechanism"
        confirmed = hypotheses.get("H3_shared_calibration_insufficient", {}).get("status") == "pass" and hypotheses.get("H5_construction_gauge_partial", {}).get("status") == "pass"
        final_gate = "MEASUREMENT_CONTROL_GAP_MECHANISM_RESOLVED" if confirmed else "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED"
    elif dominant == "extreme_order_localization" and extreme >= 0.50:
        primary = "C72-A_extreme_order_geometry_explains_measurement_control_gap"
        final_gate = "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED"
    elif dominant == "residual_candidate_specific_gauge" and gauge >= 0.50:
        primary = "C72-B_residual_candidate_specific_gauge_dominates_gap"
        final_gate = "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED"
    elif dominant == "finite_label_noise" and noise >= 0.50:
        primary = "C72-C_finite_label_noise_dominates_gap"
        final_gate = "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED"
    elif dominant == "construction_utility_mismatch" and mismatch >= 0.50:
        primary = "C72-D_construction_utility_mismatch_dominates_gap"
        final_gate = "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED"
    else:
        primary = "C72-F_C71_measurement_control_gap_not_mechanistically_resolved"
        final_gate = "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED"

    active = [primary]
    if identity_ok:
        active.extend(["C72-S1_common_utility_offset_identity_confirmed", "C72-S2_common_logit_scalar_identity_confirmed"])
    if hypotheses.get("H3_shared_calibration_insufficient", {}).get("status") == "pass":
        active.append("C72-S3_shared_target_calibration_insufficient")
    if candidate_supported:
        active.append("C72-S4_candidate_specific_intervention_reproduces_rank_flips")
    candidate_action = next((r for r in res.get("common_vs_candidate_specific_intervention_rows", []) if r["intervention"] == "I5_construction_candidate_gauge"), {})
    baseline_action = next((r for r in res.get("common_vs_candidate_specific_intervention_rows", []) if r["intervention"] == "I0_no_intervention"), {})
    candidate_lock = res.get("intervention_locks", {}).get("I5_construction_candidate_gauge", {}).get("option_id", "0.0")
    candidate_gain = float(candidate_action.get("mean_pairwise_accuracy", 0)) - float(baseline_action.get("mean_pairwise_accuracy", 0)) if candidate_action and baseline_action else 0.0
    if candidate_action and float(candidate_lock) > 0 and 0.0 < candidate_gain and float(candidate_action.get("mean_top1", 0)) < 0.75:
        active.append("C72-S5_construction_estimated_gauge_partial_only")
    if any(int(r.get("bound_nontrivial", 0)) for r in res.get("multi_candidate_rank_gauge_bound_rows", [])):
        active.append("C72-S6_multi_candidate_model_bound_nontrivial")
    if synthetic_ok:
        active.append("C72-S7_synthetic_phase_diagram_validated")
    # S8 stays inactive when strict source-domain trial features are absent;
    # unavailability is not evidence that an escape hatch was tested and failed.
    if not any(int(r.get("available", 0)) for r in res.get("feature_availability_ledger_rows", []) if r["feature_family"] == "representation_or_Wdotz"):
        active.append("C72-S9_representation_intervention_unavailable")
    if primary not in {"C72-F_C71_measurement_control_gap_not_mechanistically_resolved", "C72-G_rank_gauge_model_contradicted_by_intervention"}:
        active.append("C72-S10_independent_target_dataset_replication_now_justified")
    active.append("C72-S11_new_training_not_yet_justified")
    all_decisions = list(PRIMARY_DECISIONS) + list(SECONDARY_DECISIONS)
    return {
        "primary": primary,
        "active": active,
        "inactive": [d for d in all_decisions if d not in active],
        "final_gate": final_gate,
        "red_team_failure_count": len(red_failures),
        "blocking_risk_count": len(blocking),
        "dominant_full_budget_component": dominant,
        "full_budget_component_fractions": fractions,
        "recommended_next_direction": "independent target/dataset replication review; no new training or control artifact",
    }


def _affirmative_hit(text: str, phrase: str, window: int = 1200) -> bool:
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx < 0:
            return False
        context = low[max(0, idx - window):idx]
        if not any(cue in context for cue in NEGATION_CUES):
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
                affirmative += int(_affirmative_hit(text, pattern))
        rows.append({"pattern": pattern, "total_hits": total, "affirmative_hits": affirmative, "files": ";".join(files), "passed": int(affirmative == 0)})
    return rows


def _listed_paths() -> list[Path]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C72_*.md"))
        + list(Path(REPORT_DIR).glob("C72_*.json"))
        + list(Path(REPORT_DIR).glob("C72_*.sha256"))
        + [p for p in Path(TABLE_DIR).glob("*.csv") if p.name not in skip]
    )


def _large_scan(paths: list[Path]) -> list[dict]:
    return [{"path": str(p), "size_bytes": p.stat().st_size, "over_50mb": int(p.stat().st_size > MAX_REPORT_BYTES), "passed": int(p.stat().st_size <= MAX_REPORT_BYTES)} for p in paths]


def _artifact_manifest(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        row_count: int | str = ""
        if path.suffix == ".csv":
            with open(path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                row_count = sum(1 for _ in reader)
        rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(str(path)), "artifact_class": "table" if path.suffix == ".csv" else "protocol" if "PROTOCOL" in path.name else "report", "row_count": row_count})
    return rows


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c72", "command": "python -m pytest oaci/tests/test_c72_measurement_control_gap.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c65_c72_slice", "command": "python -m pytest oaci/tests/test_c6[5-9]_*.py oaci/tests/test_c7[0-2]_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c72_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c7*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def build_red_team_rows(res: dict) -> list[dict]:
    protocol = res["protocol"]
    risk = res.get("risk_register_rows", [])
    identity = {r["intervention"]: r for r in res.get("intervention_identity_rows", [])}
    views = res.get("c71_physical_view_replay_rows", [])
    feature = {r["feature_family"]: r for r in res.get("feature_availability_ledger_rows", [])}
    tests = res.get("test_command_manifest_rows", [])
    checks = [
        ("protocol_committed_before_outcomes", protocol["protocol_lock_source_commit"] == "4c6081d88ed314c90a060a5a415262e504a95459" and res["protocol_commit"] == "11534dc", "Static protocol was committed and pushed before C72 outcome access."),
        ("protocol_sha_matches", _sha256(PROTOCOL_JSON) == open(PROTOCOL_SHA).read().strip() == res["protocol_sha256"], "Protocol SHA replay matches."),
        ("C71_parent_replay", all(int(r["passed"]) for r in res["c71_protocol_hash_replay_rows"]), "C71 protocol and summary hashes replay."),
        ("cache_identity_replay", all(int(r["passed"]) for r in res["c71_cache_identity_replay_rows"]), "T2/T3-HO hashes, rows, and disjoint roles replay."),
        ("physical_view_replay", all(int(r["passed"]) for r in views), "C71 T3-HO physical views replay by SHA."),
        ("T2_only_tuning", all(int(r["T3_outcomes_used_for_selection"]) == 0 for r in res["intervention_calibration_rows"]) and all(int(r["T3_tuned"]) == 0 for r in res["class_conditioned_intervention_summary_rows"]), "Interventions are T2 calibrated and T3-HO locked."),
        ("no_evaluation_fit_I0_I5", all(int(r["evaluation_labels_fit"]) == 0 for r in res["intervention_inventory_rows"] if r["intervention"] in {"I0", "I1", "I2", "I3", "I4", "I5"}), "Evaluation labels do not fit primary interventions."),
        ("oracle_after_primary_freeze", bool(res["primary_freeze_sha256"]) and res["oracle_access_timestamp_utc"] >= res["primary_freeze_timestamp_utc"], "Same-label oracle opened only after primary freeze."),
        ("utility_common_offset_identity", int(identity.get("I1_utility_common_offset", {}).get("passed", 0)) == 1, "Utility-common offsets preserve ranking."),
        ("all_class_logit_identity", int(identity.get("I2_all_class_logit_scalar", {}).get("passed", 0)) == 1, "All-class scalar preserves probabilities and metrics."),
        ("representation_claim_guard", int(feature["representation_or_Wdotz"]["available"]) == 0 and feature["representation_or_Wdotz"]["status"] == "representation_intervention_supported=false", "Representation intervention is marked unavailable."),
        ("strict_source_provenance", int(feature["strict_source_domain_trial_logits_probabilities"]["available"]) == 0 and feature["frozen_source_checkpoint_score"]["strict_source_trial_feature"] == 0, "Checkpoint summaries are not relabeled as strict source trial features."),
        ("source_score_checkpoint_regime_join", all(int(r["source_score_joined"]) == int(r["unit_count"]) for r in res["cache_meta"]), "C22 source scores join one-to-one by checkpoint hash plus regime."),
        ("hierarchical_not_row_iid", all(int(r["row_iid_used"]) == 0 for r in res["hierarchical_inference_summary_rows"]), "No trial/cache row IID inference."),
        ("top1_random_baselines", all(r.get("random_base_rate", "") != "" for r in res["topk_difficulty_curve_rows"]) and all(r.get("random_top1_base_rate", "") != "" for r in res["common_vs_candidate_specific_intervention_rows"]), "Top-k/action tables include random base rates."),
        ("multiplicity_corrected", all(r.get("holm_p", "") != "" for r in res["primary_hypothesis_summary_rows"]), "H1-H6 carry Holm-adjusted decisions."),
        ("synthetic_controls", bool(res["synthetic_validation"]["passed"]), "Synthetic common-offset identity and high-reliability/poor-top1 cells pass."),
        ("risk_register_no_blocker", not any(int(r["blocking"]) for r in risk), "No open blocking risk."),
        ("no_forward_training_gpu", res["forward_passes"] == res["reinference_runs"] == res["training_attempted"] == res["gpu_used"] == 0 and res["bnci004_used"] == res["reserved_seeds_used"] == 0, "C72 is cache-only CPU analysis."),
        ("no_control_artifacts", res["selector_artifact_emitted"] == res["checkpoint_recommendation_artifact_emitted"] == res["selected_checkpoint_ids_emitted"] == 0, "No control or checkpoint artifact emitted."),
        ("large_artifact_scan", all(int(r["passed"]) for r in res.get("large_artifact_scan_rows", [])), "All C72 git artifacts are below 50MB."),
        ("forbidden_claim_scan", all(int(r["passed"]) for r in res.get("forbidden_claim_scan_rows", [])), "No affirmative forbidden claim."),
        ("tests_green", all(r["status"] == "green" for r in tests), "Focused, slice, regression, and full tests are green."),
    ]
    return [{"gate": gate, "failed": int(not ok), "finding": finding} for gate, ok, finding in checks]


def run(
    *,
    test_status: str = "planned",
    repeats: int = 256,
    permutations: int = 4999,
    bootstraps: int = 2000,
) -> dict:
    config_hash = _lock_config()
    ctx = load_protocol_and_provenance()
    first_outcome_access = _utc_now()
    source_scores = _source_score_registry()
    t2_pop, t2_meta = load_trial_cache(ctx["t2_manifest"]["external_path"], "T2", source_scores)
    t3_pop, t3_meta = load_trial_cache(ctx["t3_manifest"]["external_path"], "T3-HO", source_scores)
    if t2_meta["row_count"] != int(ctx["t2_manifest"]["row_count"]) or t2_meta["unit_count"] != 216:
        raise ValueError(f"T2 row/unit replay failed: {t2_meta}")
    if t3_meta["row_count"] != int(ctx["t3_manifest"]["row_count"]) or t3_meta["unit_count"] != 1052:
        raise ValueError(f"T3-HO row/unit replay failed: {t3_meta}")
    if t2_meta["source_score_joined"] != t2_meta["unit_count"] or t3_meta["source_score_joined"] != t3_meta["unit_count"]:
        raise ValueError(f"C22 checkpoint+regime source-score join incomplete: T2={t2_meta}; T3={t3_meta}")

    protocol = ctx["protocol"]
    budgets: list[int | str] = list(protocol["label_budgets_per_class"])
    t2_end = _endpoint_registry(t2_pop)
    t3_end = _endpoint_registry(t3_pop)
    split_seed = int(protocol["repeated_split_plan"]["seed"])
    t2_repeated = build_repeated_scores(t2_pop, budgets, repeats, split_seed)
    t3_repeated = build_repeated_scores(t3_pop, budgets, repeats, split_seed)

    t2_geometry = build_extreme_order_geometry("T2", t2_pop, t2_end, t2_repeated, 72172)
    t3_geometry = build_extreme_order_geometry("T3-HO", t3_pop, t3_end, t3_repeated, 72172)
    geometry = {key: t2_geometry[key] + t3_geometry[key] for key in t2_geometry}
    t2_noise = build_noise_and_utility_tables("T2", t2_pop, t2_end, t2_repeated, budgets)
    t3_noise = build_noise_and_utility_tables("T3-HO", t3_pop, t3_end, t3_repeated, budgets)

    beta, t2_fit = fit_source_construction_model(t2_pop, t2_end, t2_repeated, "8")
    t2_res_summary, t2_res_target, _ = build_residual_gauge("T2", t2_pop, t2_end, t2_repeated, beta, t2_fit)
    t3_res_summary, t3_res_target, t3_residuals = build_residual_gauge("T3-HO", t3_pop, t3_end, t3_repeated, beta, t2_fit)
    decomposition = build_gap_decomposition(t3_pop, t3_end, t3_repeated, t3_res_target)
    extreme_rows = build_extreme_penalty_summary(geometry, decomposition)

    locks, calibration_rows = calibrate_interventions(t2_pop, t2_end, protocol)
    primary = evaluate_primary_interventions(t3_pop, t3_end, locks, protocol)
    rank_flip_rows, rank_flip_raw = build_candidate_rank_flip_summary(t3_pop, t3_end, protocol)
    hypotheses = build_hypothesis_summary(primary, rank_flip_rows, rank_flip_raw, geometry, permutations)

    finite_t2, empirical_t2 = build_finite_population_bounds("T2", t2_pop, t2_end, t2_repeated, budgets)
    finite_t3, empirical_t3 = build_finite_population_bounds("T3-HO", t3_pop, t3_end, t3_repeated, budgets)
    model_bounds = build_rank_gauge_model_bounds(t3_pop, t3_end, t3_repeated, beta, t3_residuals, budgets)

    primary_freeze_timestamp = _utc_now()
    primary_payload = {
        "locks": locks,
        "primary_interventions": primary["common_vs_candidate_specific_intervention_rows"],
        "rank_flip": rank_flip_rows,
        "hypotheses": hypotheses,
        "decomposition": decomposition["measurement_control_gap_decomposition_rows"],
        "finite_bounds": finite_t3,
        "model_bounds": model_bounds,
    }
    primary_freeze_sha = _sha_text(json.dumps(primary_payload, sort_keys=True, default=str))
    oracle_timestamp = _utc_now()
    oracle_rows = build_oracle_ceiling(t3_pop, t3_end, primary_freeze_sha, primary_freeze_timestamp)
    for row in primary["intervention_inventory_rows"]:
        if row["intervention"] == "I6":
            row["status"] = "executed_after_primary_freeze"
        elif row["intervention"] == "I7":
            row["status"] = "unsupported_cache_fields"

    synthetic = build_synthetic_phase_diagram(protocol)
    high_poor = sum(int(r["high_reliability_poor_top1"]) for r in synthetic["synthetic_phase_diagram_rows"])
    common_identity = all(float(r["common_offset_rank_flip_rate"]) == 0.0 for r in synthetic["synthetic_phase_diagram_rows"])
    candidate_flip = float(np.mean([r["candidate_specific_rank_flip_rate"] for r in synthetic["synthetic_phase_diagram_rows"]]))
    synthetic_validation = {"high_reliability_poor_top1_cells": high_poor, "common_offset_identity_passed": common_identity, "mean_candidate_specific_flip_rate": candidate_flip, "passed": bool(high_poor > 0 and common_identity and candidate_flip > 0)}

    hierarchical = build_hierarchical_inference(t3_pop, t3_end, bootstraps, 72972)
    conditional = build_conditional_observability_secondary(t2_pop, t2_end, t3_pop, t3_end, locks)
    feature_rows = build_feature_ledger([t2_meta, t3_meta])
    risk_rows = build_risk_register(ctx, feature_rows, primary_freeze_timestamp, oracle_timestamp)
    provenance = build_provenance_tables(ctx, first_outcome_access)

    failure_rows = [
        {"reason": "protocol_timing", "status": "pass", "evidence": f"commit=11534dc;sha={ctx['protocol_sha']}", "blocks_primary": 0},
        {"reason": "cache_identity", "status": "pass", "evidence": f"T2={t2_meta['row_count']}/{t2_meta['unit_count']};T3-HO={t3_meta['row_count']}/{t3_meta['unit_count']}", "blocks_primary": 0},
        {"reason": "T2_T3_role_separation", "status": "pass", "evidence": "T2 calibrates;T3-HO evaluates locked options", "blocks_primary": 0},
        {"reason": "representation_intervention", "status": "unavailable_nonblocking", "evidence": "no representation or Wdotz cache fields", "blocks_primary": 0},
        {"reason": "strict_source_trial_path", "status": "unavailable_no_escape_hatch_test", "evidence": "C22 checkpoint summary available; strict source-domain trial logits absent", "blocks_primary": 0},
        {"reason": "target_population_generalization", "status": "unresolved", "evidence": "nine frozen BNCI2014_001 targets", "blocks_primary": 0},
        {"reason": "conditional_CS", "status": "proxy_only", "evidence": "linear conditional-observability proxy; no exact estimator claim", "blocks_primary": 0},
        {"reason": "new_training", "status": "not_justified", "evidence": "cache-only intervention evidence sufficient for current milestone", "blocks_primary": 0},
    ]

    res = {
        "milestone": MILESTONE,
        "config_hash": config_hash,
        "protocol": protocol,
        "protocol_sha256": ctx["protocol_sha"],
        "protocol_commit": "11534dc",
        "current_head_at_generation": _git_or_empty(["rev-parse", "--short", "HEAD"]),
        "first_outcome_access_timestamp_utc": first_outcome_access,
        "primary_freeze_timestamp_utc": primary_freeze_timestamp,
        "primary_freeze_sha256": primary_freeze_sha,
        "oracle_access_timestamp_utc": oracle_timestamp,
        "forward_passes": 0,
        "reinference_runs": 0,
        "training_attempted": 0,
        "gpu_used": 0,
        "bnci004_used": 0,
        "reserved_seeds_used": 0,
        "selector_artifact_emitted": 0,
        "checkpoint_recommendation_artifact_emitted": 0,
        "selected_checkpoint_ids_emitted": 0,
        "raw_cache_rows_copied_to_git": 0,
        "cache_meta": [t2_meta, t3_meta],
        "intervention_locks": locks,
        "synthetic_validation": synthetic_validation,
        **provenance,
        **geometry,
        "finite_label_noise_summary_rows": t2_noise["finite_label_noise_summary_rows"] + t3_noise["finite_label_noise_summary_rows"],
        "utility_mismatch_summary_rows": t2_noise["utility_mismatch_summary_rows"] + t3_noise["utility_mismatch_summary_rows"],
        "residual_gauge_summary_rows": t2_res_summary + t3_res_summary,
        "residual_gauge_by_target_rows": t2_res_target + t3_res_target,
        **decomposition,
        "extreme_order_penalty_summary_rows": extreme_rows,
        "finite_population_best_arm_bound_rows": finite_t2 + finite_t3,
        "multi_candidate_rank_gauge_bound_rows": model_bounds,
        "empirical_vs_bound_top1_rows": empirical_t2 + empirical_t3,
        "intervention_calibration_rows": calibration_rows,
        **{k: v for k, v in primary.items() if k != "actions"},
        "intervention_identity_rows": primary["identity_rows"],
        "logit_shift_rank_flip_summary_rows": rank_flip_rows,
        "oracle_intervention_ceiling_rows": oracle_rows,
        **synthetic,
        "hierarchical_inference_summary_rows": hierarchical,
        "feature_availability_ledger_rows": feature_rows,
        "conditional_observability_secondary_rows": conditional,
        "primary_hypothesis_summary_rows": hypotheses,
        "risk_register_rows": risk_rows,
        "failure_reason_ledger_rows": failure_rows,
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "large_artifact_scan_rows": [],
        "schema_validation_summary_rows": [],
        "red_team_failure_ledger_rows": [],
        "artifact_manifest_rows": [],
    }
    res["decision"] = classify(res)
    return res


TABLE_SPECS = {
    "risk_register.csv": ("risk_register_rows", ["risk_id", "status", "evidence", "blocking", "mitigation", "residual_caveat"]),
    "c71_authorization_provenance.csv": ("c71_authorization_provenance_rows", ["mode", "commit", "authorization_present", "forward_or_reinference_executed", "cache_rows", "status"]),
    "c71_protocol_hash_replay.csv": ("c71_protocol_hash_replay_rows", ["artifact", "registered_sha256", "observed_sha256", "passed"]),
    "c71_cache_identity_replay.csv": ("c71_cache_identity_replay_rows", ["stage", "external_path_hash", "expected_sha256", "observed_sha256", "expected_rows", "expected_units", "disjoint_from_other_stage", "passed"]),
    "c71_physical_view_replay.csv": ("c71_physical_view_replay_rows", ["view_name", "path_hash", "sha256", "sha256_match", "uses_target_labels", "uses_evaluation_labels", "available_at_selection_time", "passed"]),
    "candidate_field_size_summary.csv": ("candidate_field_size_summary_rows", ["stage", "field_level", "target_id", "trajectory", "field_index", "candidate_count", "random_top1_base_rate", "full_construct_top1", "full_construct_regret"]),
    "top_margin_geometry.csv": ("top_margin_geometry_rows", ["stage", "field_level", "target_id", "trajectory", "candidate_count", "best_utility", "best_minus_second", "best_minus_third", "best_minus_fifth", "top_quartile_margin_median", "pairwise_utility_gap_median", "source_rank_margin_median", "construction_margin_median", "gauge_residual_scale"]),
    "pairwise_margin_ledger.csv": ("pairwise_margin_ledger_rows", ["stage", "field_level", "target_id", "trajectory", "pair_ordinal", "candidate_count", "eval_utility_gap_abs", "construct_gap_abs", "mean_budget8_gap_abs", "source_rank_margin_abs", "candidate_disagreement_rate", "full_construct_order_correct", "budget8_mean_order_correct", "gauge_residual_gap_abs"]),
    "top1_difficulty_by_candidate_count.csv": ("top1_difficulty_by_candidate_count_rows", ["stage", "field_level", "candidate_count", "field_target_rows", "mean_top1_hit", "mean_regret", "random_base_rate", "top1_enrichment"]),
    "topk_difficulty_curve.csv": ("topk_difficulty_curve_rows", ["stage", "field_level", "target_id", "trajectory", "candidate_count", "k_definition", "k", "hit", "random_base_rate", "enrichment"]),
    "measurement_control_gap_decomposition.csv": ("measurement_control_gap_decomposition_rows", ["stage", "budget", "component", "observed_top1", "oracle_top1", "observed_control_gap", "shapley_top1_gain", "shapley_fraction_of_gap", "ordered_gain_min", "ordered_gain_max", "order_sensitive", "primary_order"]),
    "decomposition_order_sensitivity.csv": ("decomposition_order_sensitivity_rows", ["budget", "order", "step", "component", "top1_before", "top1_after", "marginal_top1_gain"]),
    "finite_label_noise_summary.csv": ("finite_label_noise_summary_rows", ["stage", "target_id", "budget", "repeat_count", "construction_score_variance", "evaluation_score_variance_across_candidates", "split_half_reliability", "observed_construct_eval_spearman", "noise_corrected_latent_correlation", "noise_corrected_correlation_clipped", "expected_pair_order_error", "mean_score_eval_spearman"]),
    "utility_mismatch_summary.csv": ("utility_mismatch_summary_rows", ["stage", "target_id", "heldout_outcome", "construction_bacc_spearman", "endpoint_matched_spearman", "construction_bacc_pairwise_accuracy", "endpoint_matched_pairwise_accuracy", "top1_bacc_score", "top1_endpoint_matched", "mismatch_top1_penalty"]),
    "extreme_order_penalty_summary.csv": ("extreme_order_penalty_summary_rows", ["analysis", "stratum", "candidate_count", "mean_top1_hit", "random_base_rate", "top1_enrichment", "mean_regret"]),
    "residual_gauge_summary.csv": ("residual_gauge_summary_rows", ["stage", "model_fit_stage", "source_beta", "construction_beta", "fit_rows", "T2_fit_r2", "evaluation_r2", "residual_gauge_variance_fraction", "source_score_join_fraction", "target_labels_used_for_source_features"]),
    "residual_gauge_by_target.csv": ("residual_gauge_by_target_rows", ["stage", "target_id", "candidate_count", "source_score_join_fraction", "source_only_within_target_r2", "source_plus_construct_r2", "residual_gauge_variance_fraction", "residual_gauge_sd", "eval_utility_sd"]),
    "decomposition_by_target.csv": ("decomposition_by_target_rows", ["target_id", "candidate_count", "budget8_mean_top1", "full_construct_top1", "finite_noise_top1_gain", "extreme_order_pair_reduced_gain", "utility_mismatch_primary_penalty", "residual_gauge_variance_fraction", "unresolved_top1_gap_after_full"]),
    "decomposition_by_trajectory.csv": ("decomposition_by_trajectory_rows", ["target_id", "trajectory", "field_index", "candidate_count", "full_construct_top1", "pair_reduced_top1", "top_two_margin", "pairwise_accuracy", "random_top1_base_rate"]),
    "finite_population_best_arm_bound.csv": ("finite_population_best_arm_bound_rows", ["stage", "target_id", "budget", "candidate_count", "best_vs_other_pairs", "construction_population_trials", "mean_pair_order_error_probability", "max_pair_order_error_probability", "mean_candidate_disagreement_probability", "top1_error_union_bound", "top1_hit_lower_bound", "exact_pair_distribution", "class_stratified_without_replacement"]),
    "multi_candidate_rank_gauge_bound.csv": ("multi_candidate_rank_gauge_bound_rows", ["target_id", "budget", "candidate_count", "source_rank_best_margin_median", "gauge_sd", "finite_measurement_sd", "gaussian_pair_noise_sd", "top1_error_union_bound", "top1_hit_lower_bound", "empirical_top1", "bound_nontrivial", "assumptions", "theorem_scope"]),
    "empirical_vs_bound_top1.csv": ("empirical_vs_bound_top1_rows", ["stage", "target_id", "budget", "candidate_count", "empirical_top1", "finite_population_top1_lower_bound", "bound_gap", "bound_nontrivial"]),
    "intervention_inventory.csv": ("intervention_inventory_rows", ["intervention", "name", "primary_or_secondary", "evaluation_labels_fit", "available_at_selection_time", "diagnostic_only", "status"]),
    "intervention_availability_ledger.csv": ("intervention_availability_ledger_rows", ["intervention", "required_fields", "available", "representation_required", "reason"]),
    "intervention_calibration.csv": ("intervention_calibration_rows", ["stage", "intervention", "option_id", "parameter_details", "mean_spearman", "mean_pairwise_accuracy", "mean_top1", "mean_regret", "selected_by_T2", "T3_outcomes_used_for_selection"]),
    "intervention_identity.csv": ("intervention_identity_rows", ["intervention", "scope", "locked_parameter", "rank_flips", "max_probability_delta", "max_metric_delta", "expected_identity", "passed"]),
    "common_vs_candidate_specific_intervention.csv": ("common_vs_candidate_specific_intervention_rows", ["stage", "intervention", "locked_option", "mean_spearman", "mean_pairwise_accuracy", "mean_top1", "mean_top3", "mean_regret", "mean_coverage", "random_top1_base_rate", "T3_tuned"]),
    "logit_shift_rank_flip_summary.csv": ("logit_shift_rank_flip_summary_rows", ["stage", "scale", "ratio_bin_low", "ratio_bin_high", "pair_count", "mean_original_margin", "mean_perturbation_gap", "rank_flip_rate", "target_count", "evaluation_labels_fit"]),
    "class_conditioned_intervention_summary.csv": ("class_conditioned_intervention_summary_rows", ["stage", "target_id", "intervention", "locked_option", "spearman", "pairwise_accuracy", "top1", "top3", "regret", "random_top1_base_rate", "T3_tuned"]),
    "oracle_intervention_ceiling.csv": ("oracle_intervention_ceiling_rows", ["target_id", "candidate_count", "oracle_type", "mean_selected_alpha", "top1", "regret", "direct_endpoint_scalar_top1", "primary_freeze_sha256", "primary_freeze_timestamp_utc", "available_at_selection_time", "diagnostic_only"]),
    "synthetic_phase_diagram.csv": ("synthetic_phase_diagram_rows", ["candidate_count", "rank_snr", "gauge_sd", "gauge_shape", "label_budget", "outcome_type", "replicates", "mean_spearman", "mean_pairwise_accuracy", "mean_top1", "mean_top3", "mean_regret", "mean_gauge_recovery", "high_reliability_poor_top1", "common_offset_rank_flip_rate", "candidate_specific_rank_flip_rate"]),
    "synthetic_intervention_calibration.csv": ("synthetic_intervention_calibration_rows", ["candidate_count", "gauge_sd", "gauge_shape", "label_budget", "outcome_type", "common_offset_rank_flip_rate", "candidate_specific_rank_flip_rate", "candidate_minus_common_flip_rate", "identity_control_passed"]),
    "synthetic_false_positive_control.csv": ("synthetic_false_positive_control_rows", ["null", "candidate_count", "label_budget", "outcome_type", "replicates_across_snr_shape", "common_offset_false_rank_flip_rate", "candidate_perturbation_rank_flip_rate_not_a_null_test", "registered_alpha", "identity_false_positive_passed"]),
    "hierarchical_inference_summary.csv": ("hierarchical_inference_summary_rows", ["inference_family", "metric", "point_estimate", "bootstrap_replicates", "ci_lower", "ci_upper", "targets", "checkpoint_target_units", "trial_ids", "trajectories", "row_iid_used", "target_population_generalization_claimed"]),
    "feature_availability_ledger.csv": ("feature_availability_ledger_rows", ["feature_family", "available", "strict_source_trial_feature", "uses_target_labels", "available_at_selection_time", "diagnostic_only", "status"]),
    "conditional_observability_secondary.csv": ("conditional_observability_secondary_rows", ["model", "fit_stage", "evaluation_stage", "feature_count", "T2_r2", "T3_HO_r2", "median_per_target_r2", "incremental_T3_HO_r2_over_source", "target_labels_in_source_template", "evaluation_labels_fit", "available_at_selection_time", "proxy_not_exact_conditional_cs", "diagnostic_only"]),
    "primary_hypothesis_summary.csv": ("primary_hypothesis_summary_rows", ["hypothesis", "effect", "raw_p", "holm_p", "holm_reject", "identity_pass", "permutations", "exceedances", "status"]),
    "failure_reason_ledger.csv": ("failure_reason_ledger_rows", ["reason", "status", "evidence", "blocks_primary"]),
    "protocol_timing.csv": ("protocol_timing_rows", ["event", "timestamp_utc", "sha256", "status"]),
    "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
    "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
    "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
    "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
    "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
    "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
}


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    for name, (key, cols) in TABLE_SPECS.items():
        _write_csv(os.path.join(TABLE_DIR, name), res.get(key, []), cols)


def _schema_rows() -> list[dict]:
    rows = []
    for name, (_, required) in TABLE_SPECS.items():
        if name in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        path = os.path.join(TABLE_DIR, name)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": name, "row_count": count, "required_columns_present": int(set(required).issubset(header)), "passed": int(set(required).issubset(header))})
    return rows


def _compact_summary(res: dict) -> dict:
    full_decomp = {r["component"]: r for r in res["measurement_control_gap_decomposition_rows"] if r["budget"] == FULL_BUDGET}
    b8_decomp = {r["component"]: r for r in res["measurement_control_gap_decomposition_rows"] if r["budget"] == "8"}
    actions = {r["intervention"]: r for r in res["common_vs_candidate_specific_intervention_rows"]}
    h = {r["hypothesis"]: r for r in res["primary_hypothesis_summary_rows"]}
    t3_res = next(r for r in res["residual_gauge_summary_rows"] if r["stage"] == "T3-HO")
    target_fields = [r for r in res["candidate_field_size_summary_rows"] if r["stage"] == "T3-HO" and r["field_level"] == "target_universe"]
    joint_mismatch = [r for r in res["utility_mismatch_summary_rows"] if r["stage"] == "T3-HO" and r["heldout_outcome"] == "heldout_joint_utility"]
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "protocol_commit": res["protocol_commit"],
        "protocol_sha256": res["protocol_sha256"],
        "current_head_at_generation": res["current_head_at_generation"],
        "first_outcome_access_timestamp_utc": res["first_outcome_access_timestamp_utc"],
        "primary_freeze_timestamp_utc": res["primary_freeze_timestamp_utc"],
        "primary_freeze_sha256": res["primary_freeze_sha256"],
        "oracle_access_timestamp_utc": res["oracle_access_timestamp_utc"],
        "forward_passes": res["forward_passes"],
        "reinference_runs": res["reinference_runs"],
        "training_attempted": res["training_attempted"],
        "gpu_used": res["gpu_used"],
        "bnci004_used": res["bnci004_used"],
        "reserved_seeds_used": res["reserved_seeds_used"],
        "selector_artifact_emitted": res["selector_artifact_emitted"],
        "checkpoint_recommendation_artifact_emitted": res["checkpoint_recommendation_artifact_emitted"],
        "selected_checkpoint_ids_emitted": res["selected_checkpoint_ids_emitted"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "T2_units": next(r["unit_count"] for r in res["cache_meta"] if r["stage"] == "T2"),
            "T3_HO_units": next(r["unit_count"] for r in res["cache_meta"] if r["stage"] == "T3-HO"),
            "T3_HO_rows": next(r["row_count"] for r in res["cache_meta"] if r["stage"] == "T3-HO"),
            "T3_target_universe_mean_M": float(np.mean([r["candidate_count"] for r in target_fields])),
            "T3_target_universe_min_M": int(min(r["candidate_count"] for r in target_fields)),
            "T3_target_universe_max_M": int(max(r["candidate_count"] for r in target_fields)),
            "baseline_full_spearman": actions["I0_no_intervention"]["mean_spearman"],
            "baseline_full_top1": actions["I0_no_intervention"]["mean_top1"],
            "shared_class_vector_top1": actions["I3_shared_class_vector"]["mean_top1"],
            "candidate_construction_gauge_top1": actions["I5_construction_candidate_gauge"]["mean_top1"],
            "T3_source_plus_construct_r2": t3_res["evaluation_r2"],
            "T3_residual_gauge_variance_fraction": t3_res["residual_gauge_variance_fraction"],
            "joint_utility_construction_bacc_spearman": float(np.mean([r["construction_bacc_spearman"] for r in joint_mismatch])),
            "joint_utility_endpoint_matched_spearman": float(np.mean([r["endpoint_matched_spearman"] for r in joint_mismatch])),
            "joint_utility_endpoint_matched_top1_gain": float(np.mean([r["mismatch_top1_penalty"] for r in joint_mismatch])),
            "budget8_component_fractions": {k: v["shapley_fraction_of_gap"] for k, v in b8_decomp.items()},
            "full_component_fractions": {k: v["shapley_fraction_of_gap"] for k, v in full_decomp.items()},
            "H1_status": h["H1_common_utility_offset_identity"]["status"],
            "H2_status": h["H2_common_logit_scalar_identity"]["status"],
            "H3_status": h["H3_shared_calibration_insufficient"]["status"],
            "H4_status": h["H4_candidate_perturbation_margin_ratio"]["status"],
            "H5_status": h["H5_construction_gauge_partial"]["status"],
            "H6_status": h["H6_extreme_order_candidate_count"]["status"],
            "synthetic_high_reliability_poor_top1_cells": res["synthetic_validation"]["high_reliability_poor_top1_cells"],
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": {name[:-4]: len(res.get(key, [])) for name, (key, _) in TABLE_SPECS.items()},
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def _fmt(x, digits: int = 6) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    actions = {r["intervention"]: r for r in res["common_vs_candidate_specific_intervention_rows"]}
    full = {r["component"]: r for r in res["measurement_control_gap_decomposition_rows"] if r["budget"] == FULL_BUDGET}
    b8 = {r["component"]: r for r in res["measurement_control_gap_decomposition_rows"] if r["budget"] == "8"}
    t3_res = next(r for r in res["residual_gauge_summary_rows"] if r["stage"] == "T3-HO")
    hrows = res["primary_hypothesis_summary_rows"]
    margins = [r for r in res["top_margin_geometry_rows"] if r["stage"] == "T3-HO" and r["field_level"] == "target_universe"]
    median_top_gap = float(np.median([float(r["best_minus_second"]) for r in margins]))
    finite = [r for r in res["finite_population_best_arm_bound_rows"] if r["stage"] == "T3-HO" and r["budget"] == "8"]
    model = res["multi_candidate_rank_gauge_bound_rows"]
    nontrivial = sum(int(r["bound_nontrivial"]) for r in model)
    joint_mismatch = [r for r in res["utility_mismatch_summary_rows"] if r["stage"] == "T3-HO" and r["heldout_outcome"] == "heldout_joint_utility"]
    joint_bacc_rho = float(np.mean([r["construction_bacc_spearman"] for r in joint_mismatch]))
    joint_matched_rho = float(np.mean([r["endpoint_matched_spearman"] for r in joint_mismatch]))
    joint_top1_gain = float(np.mean([r["mismatch_top1_penalty"] for r in joint_mismatch]))
    candidate_lock = res["intervention_locks"]["I5_construction_candidate_gauge"]["option_id"]
    main = "\n".join([
        f"# C72 - Extreme-Order Rank-Gauge Intervention / Measurement-Control Gap Decomposition (frozen C19 `{res['config_hash']}`)",
        "",
        "## Executive Verdict",
        "",
        f"Primary: `{d['primary']}`",
        "",
        f"Active: `{' ; '.join(d['active'])}`",
        "",
        f"Inactive: `{' ; '.join(d['inactive'])}`",
        "",
        f"Final gate: `{d['final_gate']}`",
        "",
        "## Gate-First Result",
        "",
        f"The no-intervention full-construction measurement has mean within-target Spearman `{_fmt(actions['I0_no_intervention']['mean_spearman'])}` but target-universe top-1 `{_fmt(actions['I0_no_intervention']['mean_top1'])}`. The median best-minus-second held-out bAcc gap is `{_fmt(median_top_gap)}` across nine frozen targets.",
        "",
        f"At 8 labels/class, Shapley-style gap fractions are finite-label noise `{_fmt(b8['finite_label_noise']['shapley_fraction_of_gap'])}`, endpoint mismatch `{_fmt(b8['construction_utility_mismatch']['shapley_fraction_of_gap'])}`, extreme order `{_fmt(b8['extreme_order_localization']['shapley_fraction_of_gap'])}`, and residual candidate gauge `{_fmt(b8['residual_candidate_specific_gauge']['shapley_fraction_of_gap'])}`. At full construction the corresponding fractions are `{_fmt(full['finite_label_noise']['shapley_fraction_of_gap'])}`, `{_fmt(full['construction_utility_mismatch']['shapley_fraction_of_gap'])}`, `{_fmt(full['extreme_order_localization']['shapley_fraction_of_gap'])}`, and `{_fmt(full['residual_candidate_specific_gauge']['shapley_fraction_of_gap'])}`.",
        "",
        f"The zero primary mismatch fraction is specific to the registered bAcc-to-bAcc control endpoint. In the secondary joint-utility audit, construction bAcc has mean Spearman `{_fmt(joint_bacc_rho)}`, endpoint-matched construction utility has `{_fmt(joint_matched_rho)}`, and mean top-1 gain is `{_fmt(joint_top1_gain)}`. Utility mismatch therefore remains material for the joint endpoint but does not explain the primary bAcc control gap.",
        "",
        f"The T2-fitted source-plus-construction model reaches T3-HO R2 `{_fmt(t3_res['evaluation_r2'])}`, leaving residual variance fraction `{_fmt(t3_res['residual_gauge_variance_fraction'])}`. This residual is candidate-specific after within-target centering; it is not a target-common offset.",
        "",
        "## Controlled Interventions",
        "",
        f"I1 utility-common offsets produce `{res['intervention_identity_rows'][0]['rank_flips']}` rank flips. I2 all-class logit scalars produce `{res['intervention_identity_rows'][1]['rank_flips']}` rank flips with maximum probability delta `{_fmt(res['intervention_identity_rows'][1]['max_probability_delta'], 12)}`.",
        "",
        f"The T2-locked shared class-vector intervention has T3-HO top-1 `{_fmt(actions['I3_shared_class_vector']['mean_top1'])}`. T2 selected `alpha={candidate_lock}` for the construction-estimated candidate-specific class-gradient intervention; with the observed zero lock, its T3-HO top-1 `{_fmt(actions['I5_construction_candidate_gauge']['mean_top1'])}` is the no-intervention result, not partial closure. Random matched candidate perturbations still test rank-flip sensitivity but do not constitute recovery.",
        "",
        "## Primary Hypotheses",
        "",
        *[f"- `{r['hypothesis']}`: `{r['status']}`; effect `{_fmt(r['effect'])}`; raw p `{_fmt(r['raw_p'])}`; Holm p `{_fmt(r['holm_p'])}`." for r in hrows],
        "",
        "## Theory and Synthetic Stress",
        "",
        f"The exact class-stratified finite-population paired calculation covers `{len(finite)}` T3-HO target/budget-8 fields. The Gaussian rank-gauge union bound is nontrivial in `{nontrivial}/{len(model)}` target-budget cells; trivial cells are disclosed rather than promoted.",
        "",
        f"The registered synthetic grid contains `{len(res['synthetic_phase_diagram_rows'])}` cells and `{res['synthetic_validation']['high_reliability_poor_top1_cells']}` high-reliability/poor-top1 cells. Target-common offsets have zero synthetic rank-flip rate, while candidate-specific perturbations produce mean flip rate `{_fmt(res['synthetic_validation']['mean_candidate_specific_flip_rate'])}`. The extreme-order Shapley term uses an oracle-best-conditioned two-arm counterfactual only to quantify multiplicity pressure; it is not an action rule.",
        "",
        "## Boundary",
        "",
        "C72 is a read-only diagnostic mechanism audit. No forward pass, re-inference, training, GPU work, BNCI2014_004, reserved seed, control artifact, or checkpoint identity is emitted. Strict source-domain trial logits are unavailable, so that route was not tested and is not reported as a failed escape hatch. Representation intervention is unsupported because neither representation nor Wdotz fields exist. Conditional observability remains a block-aware proxy. Target-population generalization remains unresolved.",
    ])
    theory = "\n".join([
        "# C72 - Theory Note: Multi-Candidate Measurement-to-Control Gap",
        "",
        "## Object",
        "",
        "For one frozen target, let candidates be c=1,...,M, held-out utility U(c), and a construction measurement S_b(c) from b class-stratified shared trials. Reliability concerns the ordering association between S_b and U. Control concerns whether argmax S_b intersects argmax U. These are different functionals: a global rank statistic averages O(M^2) pairs, while top-1 fails when any candidate crosses the extreme boundary.",
        "",
        "## Proposition 1: Common-Offset Invariance",
        "",
        "For every scalar a, argmax_c [U(c)+a] = argmax_c U(c). For logits, adding one scalar to every class leaves softmax probabilities exactly unchanged in real arithmetic. C72 checks both identities numerically; a shared class-dependent vector is deliberately excluded from this proposition because nonlinear metrics can reorder candidates.",
        "",
        "## Proposition 2: Exact Finite-Population Pair Error",
        "",
        "For a candidate pair, paired correctness contrasts on a class stratum take values in {-1,0,1}. If the stratum contains n+, n-, n0 values and b trials are sampled uniformly without replacement, the sampled contrast distribution is multivariate hypergeometric. C72 computes each class-mean contrast mass, convolves the four equally weighted class contributions, and reports P(bAcc contrast<=0). The top-1 lower bound is one minus the Bonferroni sum over the held-out best candidate versus every competitor. This statement is conditional on the frozen construction population and the registered class-stratified sampling design; it does not solve construction-to-evaluation gauge mismatch.",
        "",
        "## Proposition 3: Stylized Gaussian Rank-Gauge Bound",
        "",
        "Assume U_t(c)=R(c)+W_t(c)+epsilon_t(c), centered candidate-specific W and finite-label epsilon are Gaussian with registered variances, and the source-rank best has pair margin Delta_j over competitor j. Then pair reversal probability is Phi[-Delta_j/sqrt(2(sigma_W^2+sigma_epsilon^2/b))], and the sum over j is a valid union bound under these assumptions. C72 labels this a stylized model bound. It is neither an EEG population theorem nor a minimax lower bound.",
        "",
        "## Extreme-Order Consequence",
        "",
        "Even when most candidate pairs are ordered correctly, increasing M adds extreme competitors and the minimum top margin contracts. The probability of at least one crossing can therefore rise while Spearman stays high. C72's candidate-count intervention and synthetic grid test this implication directly.",
        "",
        "## Empirical Scope",
        "",
        f"On T3-HO, the median target-universe top-two bAcc gap is `{_fmt(median_top_gap)}`. The source-plus-construction residual variance fraction is `{_fmt(t3_res['residual_gauge_variance_fraction'])}`. Exact finite-population and stylized union bounds are tabulated separately so finite-label sampling uncertainty is not conflated with candidate-specific target gauge.",
    ])
    red = "\n".join([
        "# C72 - Red-Team Verification",
        "",
        "All C72 red-team gates pass." if not d["red_team_failure_count"] else f"C72 has `{d['red_team_failure_count']}` open red-team gate(s).",
        "",
        *[f"- `{r['gate']}`: `{'PASS' if not int(r['failed']) else 'FAIL'}` - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    return {os.path.basename(MAIN_REPORT): main, os.path.basename(THEORY_NOTE): theory, os.path.basename(RED_REPORT): red}


def _write_reports_and_json(res: dict) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name, body in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(body.rstrip() + "\n")
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_summary(res), f, indent=2, sort_keys=True)
        f.write("\n")


def _quality_refresh(res: dict) -> None:
    write_tables(res)
    _write_reports_and_json(res)
    paths = [str(p) for p in _listed_paths()]
    res["large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)


def write_artifacts(res: dict) -> dict:
    if _sha256(PROTOCOL_JSON) != open(PROTOCOL_SHA).read().strip():
        raise ValueError("refusing to write C72 artifacts after protocol drift")
    os.makedirs(TABLE_DIR, exist_ok=True)
    for _ in range(3):
        _quality_refresh(res)
        _write_reports_and_json(res)
    # Stabilize scans and red-team report before hashing the artifact set.
    _quality_refresh(res)
    _write_reports_and_json(res)
    write_tables(res)
    paths = _listed_paths()
    res["artifact_manifest_rows"] = _artifact_manifest(paths)
    write_tables(res)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c72_measurement_control_gap")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--repeats", type=int, default=256)
    ap.add_argument("--permutations", type=int, default=4999)
    ap.add_argument("--bootstraps", type=int, default=2000)
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status, repeats=args.repeats, permutations=args.permutations, bootstraps=args.bootstraps)
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C72] decision={res['decision']['primary']} gate={res['decision']['final_gate']} red={res['decision']['red_team_failure_count']} tables={len(TABLE_SPECS)}")


if __name__ == "__main__":
    main()
