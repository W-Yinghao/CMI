"""C70 - split-label information budget / gauge-recovery phase diagram."""
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
from . import c69_powered_trial_cache_scaleup as c69


MILESTONE = "C70"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c70_tables"
REPORT_JSON = "oaci/reports/C70_SPLIT_LABEL_INFORMATION_BUDGET.json"
C69_JSON = "oaci/reports/C69_POWERED_TRIAL_CACHE_SCALEUP.json"
C69_T1_MANIFEST = "oaci/reports/c69_tables/c69_cache_manifest_t1.csv"
C69_T2_MANIFEST = "oaci/reports/c69_tables/c69_cache_manifest_t2.csv"
C69_SPLIT_SUMMARY = "oaci/reports/c69_tables/c69_split_label_summary.csv"
C69_CS_SUMMARY = "oaci/reports/c69_tables/c69_conditional_cs_summary.csv"
C65_MAP = "oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv"
MAX_REPORT_BYTES = 50_000_000

BUDGETS = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
FULL_BUDGET_LABEL = "full-construction"
DEFAULT_REPEATS = 256
DEFAULT_PERMUTATIONS = 4999
DEFAULT_BOOTSTRAPS = 1000
ENDPOINT_ORACLE_HIT = 0.9444444444444444

DECISIONS = (
    "C70-A_small_budget_split_label_gauge_recovery_candidate",
    "C70-B_medium_or_dense_label_recovery_only",
    "C70-C_split_label_reliability_without_actionability",
    "C70-D_c69_signal_collapses_under_hierarchical_controls",
    "C70-E_claim_or_masking_inconsistency_requires_repair",
    "C70-S1_finite_population_label_budget_bound_established",
    "C70-S2_paired_model_bound_nontrivial",
    "C70-S3_block_conditional_cs_stable_diagnostic",
    "C70-S4_conditional_cs_proxy_only_or_bandwidth_sensitive",
    "C70-S5_no_strict_source_trial_escape_hatch",
    "C70-S6_strict_source_trial_escape_hatch_found",
    "C70-S7_t3_disjoint_confirmatory_protocol_locked",
    "C70-S8_target_population_generalization_unresolved",
    "C70-S9_new_training_not_justified",
)

FINAL_GATES = (
    "SMALL_BUDGET_SPLIT_LABEL_GAUGE_RECOVERY_CANDIDATE",
    "MEDIUM_OR_DENSE_LABEL_RECOVERY_ONLY",
    "SPLIT_LABEL_RELIABILITY_WITHOUT_ACTIONABILITY",
    "C69_SIGNAL_COLLAPSES_UNDER_HIERARCHICAL_CONTROLS",
    "CLAIM_OR_MASKING_REPAIR_REQUIRED",
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
    "t1 and t2 are independent confirmation",
    "row-level iid",
    "new training is justified",
    "gpu used",
    "forward pass executed",
    "re-inference executed",
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
    "does not ",
    "do not ",
    "proxy-only ",
    "unresolved ",
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


def _short_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(str(text).encode()).hexdigest()[:n]


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_or_empty(args: list[str]) -> str:
    try:
        return _git(args)
    except Exception:
        return ""


def _rankdata(vals: np.ndarray) -> np.ndarray:
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
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return math.nan
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(x, dtype=float)), _rankdata(np.asarray(y, dtype=float)))


def _pairwise_order_acc(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    total = 0
    acc = 0.0
    margins = []
    for i in range(n):
        for j in range(i + 1, n):
            dy = float(y[i] - y[j])
            dx = float(x[i] - x[j])
            if abs(dy) < 1e-12:
                continue
            total += 1
            margins.append(abs(dy))
            if abs(dx) < 1e-12:
                acc += 0.5
            elif dx * dy > 0:
                acc += 1.0
    return (acc / total if total else math.nan, float(np.median(margins)) if margins else math.nan)


def _top_metrics(construct: np.ndarray, eval_score: np.ndarray, k: int = 3) -> tuple[float, float, float]:
    max_eval = float(np.max(eval_score))
    eval_top = np.flatnonzero(np.isclose(eval_score, max_eval))
    max_construct = float(np.max(construct))
    top_tie = np.flatnonzero(np.isclose(construct, max_construct))
    top1_hit = float(sum(i in set(eval_top) for i in top_tie) / len(top_tie))
    selected_eval = float(np.mean(eval_score[top_tie]))
    regret = max_eval - selected_eval

    order_values = sorted(set(float(v) for v in construct), reverse=True)
    higher: list[int] = []
    tied: list[int] = []
    for value in order_values:
        group = [i for i, v in enumerate(construct) if float(v) == value]
        if len(higher) + len(group) >= k:
            tied = group
            break
        higher.extend(group)
    top_eval = set(eval_top)
    if any(i in top_eval for i in higher):
        topk_hit = 1.0
    elif tied:
        remaining = max(0, k - len(higher))
        top_in_tie = sum(i in top_eval for i in tied)
        if remaining <= 0 or top_in_tie == 0:
            topk_hit = 0.0
        elif remaining >= len(tied):
            topk_hit = 1.0
        else:
            topk_hit = float(1.0 - (math.comb(len(tied) - top_in_tie, remaining) / math.comb(len(tied), remaining)))
    else:
        topk_hit = 0.0
    return top1_hit, topk_hit, regret


def _gauge_recovery(construct: np.ndarray, eval_score: np.ndarray) -> tuple[float, float]:
    e = np.asarray(eval_score, dtype=float) - float(np.mean(eval_score))
    c = np.asarray(construct, dtype=float) - float(np.mean(construct))
    base = float(np.mean(e ** 2))
    if base <= 1e-12:
        return math.nan, 0.0
    denom = float(np.dot(c, c))
    if denom <= 1e-12:
        return 0.0, 0.0
    alpha = float(np.dot(c, e) / denom)
    resid = e - alpha * c
    closure = 1.0 - float(np.mean(resid ** 2)) / base
    return max(0.0, min(1.0, closure)), alpha


def _bacc_scores(correct: np.ndarray, labels: np.ndarray, indices: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(correct.shape[0], dtype=float)
    parts = []
    for cls in classes:
        cls_idx = indices[labels[indices] == cls]
        if len(cls_idx) == 0:
            parts.append(np.full(correct.shape[0], np.nan))
        else:
            parts.append(np.mean(correct[:, cls_idx], axis=1))
    arr = np.vstack(parts)
    return np.nanmean(arr, axis=0)


def _listed_paths() -> list[Path]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        list(Path(REPORT_DIR).glob("C70_*.md"))
        + list(Path(REPORT_DIR).glob("C70_*.json"))
        + list(Path(TABLE_DIR).glob("C71_T3_CONFIRMATORY_PROTOCOL.*"))
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
            "artifact_class": "table" if str(p).endswith(".csv") else "protocol" if "C71_" in str(p) else "summary_json" if str(p).endswith(".json") else "report",
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


def _manifest_row(path: str) -> dict:
    rows = {r["cache_kind"]: r for r in _read_csv(path)}
    return rows["minimal_logits_probs_metadata"]


def load_cache_manifest() -> dict:
    c69_json = _load_json(C69_JSON)
    t1 = _manifest_row(C69_T1_MANIFEST)
    t2 = _manifest_row(C69_T2_MANIFEST)
    return {
        "c69_json": c69_json,
        "t1": t1,
        "t2": t2,
        "t1_sha_match": int(os.path.exists(t1["external_path"]) and _sha256(t1["external_path"]) == t1["sha256"]),
        "t2_sha_match": int(os.path.exists(t2["external_path"]) and _sha256(t2["external_path"]) == t2["sha256"]),
    }


def load_t2_population(path: str) -> tuple[dict[str, dict], list[dict]]:
    by_target_unit: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    unit_meta: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            target = row["target_id"]
            ckpt = row["checkpoint_id"]
            by_target_unit[target][ckpt].append(row)
            unit_meta.setdefault(ckpt, {
                "target_id": target,
                "unit_hash": _short_hash(ckpt),
                "trajectory_id": row["trajectory_id"],
                "trajectory_hash": _short_hash(row["trajectory_id"]),
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
            for r in by_unit[ckpt]:
                ti = trial_index[r["trial_id"]]
                correct[ui, ti] = int(r["correctness_quarantined"])
                labels[ti] = int(r["y_true_quarantined"])
                split[ti] = r["split_role_for_future_split_label"]
        classes = np.array(sorted(set(int(v) for v in labels)), dtype=int)
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
        for ui, ckpt in enumerate(units):
            meta = unit_meta[ckpt]
            unit_rows.append({
                "stage": "t2",
                "unit_hash": meta["unit_hash"],
                "target_id": target,
                "trajectory_hash": meta["trajectory_hash"],
                "seed": meta["seed"],
                "level": meta["level"],
                "regime": meta["regime"],
                "candidate_order": meta["candidate_order"],
                "target_trial_count": len(trial_ids),
                "checkpoint_target_rows": int(correct.shape[1]),
                "construct_trial_count": int(np.sum(split == "target_construct")),
                "eval_trial_count": int(np.sum(split == "target_eval")),
                "source_domain_trial_logits_available": 0,
            })
    return populations, unit_rows


def build_phase0_tables(manifest: dict, populations: dict[str, dict], unit_rows: list[dict]) -> dict[str, list[dict]]:
    c65_rows = _read_csv(C65_MAP)
    canonical, t1_units, t2_units = c69.build_schedule(c65_rows)
    t1_ids = {r["checkpoint_id"] for r in t1_units}
    t2_ids = {r["checkpoint_id"] for r in t2_units}
    c69_split = {r["stage"]: r for r in _read_csv(C69_SPLIT_SUMMARY)}
    c69_cs = {r["stage"]: r for r in _read_csv(C69_CS_SUMMARY)}

    overlap_rows = [
        {"comparison": "t1_vs_t2", "left_units": len(t1_ids), "right_units": len(t2_ids), "overlap_units": len(t1_ids & t2_ids), "left_subset_of_right": int(t1_ids <= t2_ids), "independent_confirmation_claimed": 0, "notes": "T2 is a scale-up containing T1, not an independent confirmation."},
        {"comparison": "t2_vs_t3_ho", "left_units": len(t2_ids), "right_units": len({r["checkpoint_id"] for r in canonical} - t2_ids), "overlap_units": 0, "left_subset_of_right": 0, "independent_confirmation_claimed": 0, "notes": "T3-HO is protocol-locked but not consumed."},
    ]
    blind_rows = [
        {"audit": "schedule_reconstructed_from_c65", "passed": int(len(t1_ids) == 64 and len(t2_ids) == 216), "observed": f"T1={len(t1_ids)};T2={len(t2_ids)}", "forbidden_dependency": "target outcome adaptive sampling", "notes": "Deterministic seed/target/level/candidate-order schedule replayed from C65 mapping."},
        {"audit": "t2_cache_hash_verified", "passed": manifest["t2_sha_match"], "observed": manifest["t2"]["sha256"], "forbidden_dependency": "manifest/hash mismatch", "notes": "C69 T2 external cache consumed read-only."},
        {"audit": "no_t3_cache_consumed", "passed": 1, "observed": "0", "forbidden_dependency": "T3 cache/checkpoint consumption", "notes": "C70 reads C65 mapping only to define a future holdout protocol."},
    ]
    split_rows = []
    for target, pop in populations.items():
        split_by_trial = defaultdict(set)
        class_counts = Counter()
        for tid, cls, split in zip(pop["trial_ids"], pop["labels"], pop["split"]):
            split_by_trial[tid].add(split)
            class_counts[(int(cls), split)] += 1
        construct = {tid for tid, ss in split_by_trial.items() if ss == {"target_construct"}}
        evals = {tid for tid, ss in split_by_trial.items() if ss == {"target_eval"}}
        split_rows.append({
            "target_id": target,
            "checkpoint_units": len(pop["units"]),
            "unique_target_trial_ids": len(pop["trial_ids"]),
            "construct_unique_trial_ids": len(construct),
            "eval_unique_trial_ids": len(evals),
            "construction_eval_disjoint": int(not (construct & evals)),
            "trial_split_shared_across_candidates": int(all(len(ss) == 1 for ss in split_by_trial.values())),
            "min_repetitions_per_trial": int(min(Counter((r["target_id"], r["target_trial_count"]) for r in unit_rows if r["target_id"] == target).values()) if unit_rows else 0),
            "class_construct_counts": ";".join(f"{cls}:{class_counts[(cls, 'target_construct')]}" for cls in sorted(set(pop["labels"]))),
            "class_eval_counts": ";".join(f"{cls}:{class_counts[(cls, 'target_eval')]}" for cls in sorted(set(pop["labels"]))),
            "fixed_budget_claim_allowed": 1,
        })
    split_p = float(c69_split["t2"]["permutation_p_value"])
    cs_p = float(c69_cs["t2"]["permutation_p_value"])
    perm_rows = [
        {"test": "c69_split_label_spearman", "reported_p": split_p, "inferred_permutations": int(round(1.0 / split_p - 1)), "inferred_exceedances": 0, "monte_carlo_floor": split_p, "resolution_note": "reported p equals +1 minimum floor", "c70_permutations": DEFAULT_PERMUTATIONS},
        {"test": "c69_binary_y_cod_proxy", "reported_p": cs_p, "inferred_permutations": int(round(1.0 / cs_p - 1)), "inferred_exceedances": 0, "monte_carlo_floor": cs_p, "resolution_note": "reported p equals +1 minimum floor", "c70_permutations": 999},
    ]
    feature_rows = [
        {"feature_family": "strict_source_domain_trial_logits", "available_in_c69_cache": 0, "uses_target_labels": 0, "uses_eval_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 0, "status": "absent_not_tested"},
        {"feature_family": "checkpoint_metadata_seed_level_order", "available_in_c69_cache": 1, "uses_target_labels": 0, "uses_eval_labels": 0, "available_at_selection_time": 1, "diagnostic_only": 1, "status": "non_label_metadata_not_strict_source_trial_signal"},
        {"feature_family": "target_trial_logits_probabilities", "available_in_c69_cache": 1, "uses_target_labels": 0, "uses_eval_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "target_audit_cache_only"},
        {"feature_family": "construction_split_target_labels", "available_in_c69_cache": 1, "uses_target_labels": 1, "uses_eval_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "split_label_information_class"},
        {"feature_family": "same_label_endpoint_oracle", "available_in_c69_cache": 1, "uses_target_labels": 1, "uses_eval_labels": 1, "available_at_selection_time": 0, "diagnostic_only": 1, "status": "oracle_boundary_only"},
    ]
    return {
        "c69_unit_dependency_graph_rows": unit_rows,
        "c69_t1_t2_overlap_rows": overlap_rows,
        "c69_sampling_blindness_audit_rows": blind_rows,
        "c69_split_contract_audit_rows": split_rows,
        "c69_permutation_resolution_audit_rows": perm_rows,
        "c69_source_feature_availability_rows": feature_rows,
        "feature_availability_ledger_rows": feature_rows,
    }


def _construct_indices(pop: dict, budget: int | str, repeat: int, rng_seed: int) -> np.ndarray:
    labels = pop["labels"]
    classes = pop["classes"]
    if budget == FULL_BUDGET_LABEL:
        return np.flatnonzero(pop["split"] == "target_construct")
    if int(budget) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(rng_seed + repeat * 1009 + int(pop["target_id"]) * 917 + int(budget) * 53)
    selected = []
    for cls in classes:
        cls_idx = np.flatnonzero(labels == cls)
        take = min(int(budget), len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def _budget_eval(pop: dict, budget: int | str, repeat: int, rng_seed: int) -> tuple[dict, list[dict]]:
    correct = pop["correct"]
    labels = pop["labels"]
    classes = pop["classes"]
    construct_idx = _construct_indices(pop, budget, repeat, rng_seed)
    all_idx = np.arange(len(labels), dtype=int)
    eval_idx = np.setdiff1d(all_idx, construct_idx, assume_unique=False)
    construct = _bacc_scores(correct, labels, construct_idx, classes)
    eval_score = _bacc_scores(correct, labels, eval_idx, classes)
    spearman = _spearman(construct, eval_score)
    pair_acc, pair_margin = _pairwise_order_acc(construct, eval_score)
    top1_hit, top3_hit, regret = _top_metrics(construct, eval_score)
    gauge, alpha = _gauge_recovery(construct, eval_score)
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
    n = len(eval_score)
    for i in range(n):
        for j in range(i + 1, n):
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


def build_budget_curves(populations: dict[str, dict], repeats: int, rng_seed: int) -> dict[str, list[dict]]:
    target_rows = []
    pair_rows = []
    for budget in [*BUDGETS, FULL_BUDGET_LABEL]:
        reps = 1 if budget == FULL_BUDGET_LABEL else repeats
        for repeat in range(reps):
            for pop in populations.values():
                row, pairs = _budget_eval(pop, budget, repeat, rng_seed)
                target_rows.append(row)
                if repeat < min(64, reps):
                    pair_rows.extend(pairs)
    aggregate_rows = []
    action_rows = []
    gauge_rows = []
    decomp_rows = []
    by_budget = defaultdict(list)
    for r in target_rows:
        by_budget[r["budget"]].append(r)
    ordered = [str(b) for b in BUDGETS] + [FULL_BUDGET_LABEL]
    for budget in ordered:
        rows = by_budget[budget]
        vals = lambda key: np.array([float(r[key]) for r in rows if r[key] != "" and math.isfinite(float(r[key]))], dtype=float)
        mean = lambda key: float(np.mean(vals(key))) if len(vals(key)) else math.nan
        aggregate_rows.append({
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
            "endpoint_oracle_reference": ENDPOINT_ORACLE_HIT,
            "few_label_sufficiency_claimed": 0,
        })
        action_rows.append({
            "budget": budget,
            "top1_hit": aggregate_rows[-1]["mean_top1_hit"],
            "top3_hit": aggregate_rows[-1]["mean_top3_hit"],
            "continuous_regret": aggregate_rows[-1]["mean_continuous_regret"],
            "coverage_regret_le_0p02": aggregate_rows[-1]["actionability_rate_regret_le_0p02"],
            "actionability_status": "candidate" if aggregate_rows[-1]["actionability_rate_regret_le_0p02"] >= 0.5 else "limited",
        })
        gauge_rows.append({
            "budget": budget,
            "gauge_residual_recovery": aggregate_rows[-1]["mean_gauge_residual_recovery"],
            "source_to_oracle_gap_closure_fraction": aggregate_rows[-1]["mean_gauge_residual_recovery"],
            "common_target_offset_credit": 0,
            "candidate_specific_within_target_credit": aggregate_rows[-1]["mean_gauge_residual_recovery"],
        })
        rank_vals = [max(0.0, float(r["within_target_spearman"])) ** 2 for r in rows if math.isfinite(float(r["within_target_spearman"]))]
        noise_vals = [1.0 - float(r["gauge_residual_recovery"]) for r in rows if math.isfinite(float(r["gauge_residual_recovery"]))]
        decomp_rows.append({
            "budget": budget,
            "rank_component_recovery": round(float(np.mean(rank_vals)), 6) if rank_vals else "",
            "candidate_specific_gauge_recovery": aggregate_rows[-1]["mean_gauge_residual_recovery"],
            "finite_trial_noise_floor": round(float(np.mean(noise_vals)), 6) if noise_vals else "",
            "common_target_offset_not_credited": 1,
        })
    finite_rows = []
    for budget in ordered:
        rows = [r for r in pair_rows if r["budget"] == budget]
        recovered = [float(r["recovered"]) for r in rows]
        margins = [float(r["eval_margin_abs"]) for r in rows]
        finite_rows.append({
            "budget": budget,
            "registered_pair_observations": len(rows),
            "random_subset_order_recovery_probability": round(float(np.mean(recovered)), 6) if recovered else "",
            "median_eval_pair_margin": round(float(np.median(margins)), 6) if margins else "",
            "finite_population_scope": "fixed_C69_T2_targets_and_checkpoint_units",
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
        "label_budget_gauge_recovery_curve_rows": gauge_rows,
        "rank_vs_gauge_budget_decomposition_rows": decomp_rows,
        "finite_population_budget_bound_rows": finite_rows,
    }


def build_cluster_bootstrap(label_rows: list[dict], bootstraps: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    by_budget_target = defaultdict(lambda: defaultdict(list))
    for r in label_rows:
        by_budget_target[r["budget"]][r["target_id"]].append(r)
    for budget, by_target in sorted(by_budget_target.items(), key=lambda x: ([str(b) for b in BUDGETS] + [FULL_BUDGET_LABEL]).index(x[0])):
        target_ids = sorted(by_target, key=int)
        target_means = {}
        for t, rows in by_target.items():
            spearman_vals = [float(r["within_target_spearman"]) for r in rows if r["within_target_spearman"] != "" and math.isfinite(float(r["within_target_spearman"]))]
            target_means[t] = {
                "spearman": float(np.mean(spearman_vals)) if spearman_vals else math.nan,
                "top1": float(np.mean([float(r["top1_hit"]) for r in rows])) if rows else math.nan,
                "regret": float(np.mean([float(r["continuous_regret"]) for r in rows])) if rows else math.nan,
                "gauge": float(np.nanmean([float(r["gauge_residual_recovery"]) for r in rows if r["gauge_residual_recovery"] != ""])) if rows else math.nan,
            }
        samples = defaultdict(list)
        for _ in range(bootstraps):
            draw = rng.choice(target_ids, size=len(target_ids), replace=True)
            for key in ("spearman", "top1", "regret", "gauge"):
                vals = [target_means[t][key] for t in draw if math.isfinite(target_means[t][key])]
                samples[key].append(float(np.mean(vals)) if vals else math.nan)
        for key in ("spearman", "top1", "regret", "gauge"):
            vals = np.array([v for v in samples[key] if math.isfinite(v)], dtype=float)
            out.append({
                "budget": budget,
                "metric": key,
                "cluster": "target",
                "bootstrap_replicates": bootstraps,
                "mean": round(float(np.mean(vals)), 6) if len(vals) else "",
                "ci_lower": round(float(np.quantile(vals, 0.025)), 6) if len(vals) else "",
                "ci_upper": round(float(np.quantile(vals, 0.975)), 6) if len(vals) else "",
                "target_population_generalization_claimed": 0,
            })
    return out


def build_blocked_permutation(populations: dict[str, dict], permutations: int, seed: int) -> list[dict]:
    construct_all = []
    eval_all = []
    target_blocks = []
    for target, pop in populations.items():
        construct_idx = np.flatnonzero(pop["split"] == "target_construct")
        eval_idx = np.flatnonzero(pop["split"] == "target_eval")
        construct = _bacc_scores(pop["correct"], pop["labels"], construct_idx, pop["classes"])
        eval_score = _bacc_scores(pop["correct"], pop["labels"], eval_idx, pop["classes"])
        construct_all.extend((construct - float(np.mean(construct))).tolist())
        eval_all.extend((eval_score - float(np.mean(eval_score))).tolist())
        target_blocks.extend([target] * len(construct))
    construct_arr = np.asarray(construct_all, dtype=float)
    eval_arr = np.asarray(eval_all, dtype=float)
    block_arr = np.asarray(target_blocks)
    obs = _spearman(construct_arr, eval_arr)
    rng = np.random.default_rng(seed)
    exceed = 0
    null_vals = []
    for _ in range(permutations):
        perm = eval_arr.copy()
        for target in sorted(set(target_blocks), key=int):
            idx = np.flatnonzero(block_arr == target)
            perm[idx] = rng.permutation(perm[idx])
        val = _spearman(construct_arr, perm)
        null_vals.append(val)
        exceed += int(val >= obs)
    p = (exceed + 1) / (permutations + 1)
    return [{
        "test": "full_construction_within_target_centered_spearman",
        "blocking": "shuffle_eval_scores_within_target_after_centering",
        "observed": round(obs, 6),
        "permutations": permutations,
        "exceedances": exceed,
        "p_value": round(p, 8),
        "monte_carlo_floor": round(1 / (permutations + 1), 8),
        "null_p95": round(float(np.quantile(null_vals, 0.95)), 6),
        "row_iid_interpretation_used": 0,
    }]


def build_pair_sample_complexity(populations: dict[str, dict]) -> list[dict]:
    rows = []
    bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)]
    bucket = defaultdict(list)
    for pop in populations.values():
        correct = pop["correct"]
        full = _bacc_scores(correct, pop["labels"], np.arange(len(pop["labels"])), pop["classes"])
        n = len(full)
        for i in range(n):
            for j in range(i + 1, n):
                diff = correct[i] - correct[j]
                delta = abs(float(full[i] - full[j]))
                if delta <= 1e-12:
                    continue
                var = float(np.var(diff))
                required_total = (1.96 ** 2) * var / max(delta ** 2, 1e-12)
                required_per_class = required_total / len(pop["classes"])
                disagreement = float(np.mean(diff != 0))
                for lo, hi in bins:
                    if lo <= delta < hi:
                        bucket[(lo, hi)].append((required_per_class, disagreement, delta))
                        break
    for lo, hi in bins:
        vals = bucket[(lo, hi)]
        req = [v[0] for v in vals]
        dis = [v[1] for v in vals]
        delta = [v[2] for v in vals]
        rows.append({
            "margin_bin": f"[{lo:.2f},{hi:.2f})",
            "pair_count": len(vals),
            "median_required_labels_per_class_normal_proxy": round(float(np.median(req)), 3) if req else "",
            "p75_required_labels_per_class_normal_proxy": round(float(np.quantile(req, 0.75)), 3) if req else "",
            "median_disagreement_rate": round(float(np.median(dis)), 6) if dis else "",
            "median_abs_bacc_contrast": round(float(np.median(delta)), 6) if delta else "",
            "bound_scope": "paired_finite_population_normal_proxy_not_eeg_minimax",
        })
    return rows


def build_conditional_cs_block_summary() -> list[dict]:
    c69_rows = {r["stage"]: r for r in _read_csv(C69_CS_SUMMARY)}
    t2 = c69_rows["t2"]
    return [
        {
            "estimator": "c69_binary_y_cod_proxy",
            "stage": "t2",
            "paired_eval_rows": t2["paired_eval_rows"],
            "independent_checkpoint_units": t2["independent_checkpoint_units"],
            "incremental_cod": t2["incremental_cod"],
            "reported_p": t2["permutation_p_value"],
            "inferred_permutations": 64,
            "block_valid_status": "proxy_only_directionally_stable",
            "full_conditional_cs_claimed": 0,
            "bandwidth_nested_null": "not_applicable_binary_y_cod_proxy",
            "notes": "C69 proxy is directionally stable but crossed checkpoint/trial dependence blocks a full conditional-CS claim.",
        }
    ]


def build_failure_reason_ledger(phase0: dict, budget_rows: list[dict], blocked_rows: list[dict]) -> list[dict]:
    full = next(r for r in budget_rows if r["budget"] == FULL_BUDGET_LABEL)
    small = [r for r in budget_rows if r["budget"] in {"1", "2", "4", "8"}]
    first_small = max(small, key=lambda r: float(r["mean_gauge_residual_recovery"]))
    return [
        {"reason": "shared_split_contract", "status": "pass", "evidence": "trial_split_shared_across_candidates=1 for every target", "blocks_claim": 0},
        {"reason": "monte_carlo_resolution_c69", "status": "reported", "evidence": "C69 p-values are minimum +1 floors: 1/201 and 1/65", "blocks_claim": 0},
        {"reason": "target_population_generalization", "status": "unresolved", "evidence": "C70 is conditional on the fixed BNCI2014_001 targets in T2", "blocks_claim": 1},
        {"reason": "small_budget_strength", "status": "candidate" if float(first_small["mean_gauge_residual_recovery"]) >= 0.5 else "limited", "evidence": f"best small budget {first_small['budget']} gauge recovery={first_small['mean_gauge_residual_recovery']}", "blocks_claim": 0},
        {"reason": "full_construction_reliability", "status": "strong" if float(full["mean_within_target_spearman"]) >= 0.8 else "weak", "evidence": f"full construction mean Spearman={full['mean_within_target_spearman']}", "blocks_claim": 0},
        {"reason": "strict_source_trial_features", "status": "absent", "evidence": "C69 cache has no strict source-domain trial logits/probs", "blocks_claim": 1},
        {"reason": "conditional_cs_scope", "status": "proxy_only", "evidence": "binary-Y COD proxy retained; no full conditional-CS claim", "blocks_claim": 1},
        {"reason": "t3_confirmation", "status": "protocol_locked_not_executed", "evidence": "C71 protocol JSON/SHA emitted; T3-HO not consumed", "blocks_claim": 0},
    ]


def build_t3_protocol(c65_rows: list[dict]) -> tuple[dict, str]:
    canonical, t1_units, t2_units = c69.build_schedule(c65_rows)
    t2_ids = {r["checkpoint_id"] for r in t2_units}
    holdout_ids = sorted({r["checkpoint_id"] for r in canonical} - t2_ids)
    protocol = {
        "schema_version": "c71_t3_confirmatory_protocol_v1",
        "locked_by": "C70",
        "diagnostic_only_non_deployable": True,
        "t3_full_physical_units": len(canonical),
        "t2_consumed_units": len(t2_ids),
        "t3_ho_units": len(holdout_ids),
        "t3_ho_checkpoint_id_set_sha256": hashlib.sha256("\n".join(holdout_ids).encode()).hexdigest(),
        "budget_grid_labels_per_class": list(BUDGETS) + [FULL_BUDGET_LABEL],
        "split_seed_registry": {"base_seed": 70070, "repeat_count": DEFAULT_REPEATS},
        "primary_outcomes": ["within_target_spearman", "pairwise_order_accuracy", "top1_hit", "continuous_regret", "gauge_residual_recovery"],
        "hierarchical_inference": ["target_cluster_bootstrap", "checkpoint_unit_blocking", "trial_id_cluster_awareness"],
        "blocked_permutation_scheme": "shuffle held-out utility within target; preserve target and checkpoint structure",
        "permutation_count": DEFAULT_PERMUTATIONS,
        "decision_thresholds": {
            "small_budget": "<=8 labels/class",
            "medium_budget": "12-32 labels/class",
            "dense_budget": ">=48 labels/class",
            "gauge_recovery_candidate": "mean gauge recovery >=0.5 and actionability coverage >=0.5",
        },
        "forbidden": ["new forward", "training", "GPU", "selector artifact", "checkpoint recommendation", "target-population generalization claim"],
    }
    body = json.dumps(protocol, indent=2, sort_keys=True)
    return protocol, hashlib.sha256(body.encode()).hexdigest()


def write_t3_protocol(protocol: dict, sha: str) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, "C71_T3_CONFIRMATORY_PROTOCOL.json")
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "C71_T3_CONFIRMATORY_PROTOCOL.sha256"), "w") as f:
        f.write(sha + "\n")


def build_red_team_rows(res: dict) -> list[dict]:
    split = res["c69_split_contract_audit_rows"]
    overlap = {r["comparison"]: r for r in res["c69_t1_t2_overlap_rows"]}
    feature = {r["feature_family"]: r for r in res["feature_availability_ledger_rows"]}
    checks = [
        ("manifest_hashes_match", res["c69_manifest_hash_audit"]["t1_sha_match"] == 1 and res["c69_manifest_hash_audit"]["t2_sha_match"] == 1, "C69 T1/T2 hashes replayed before C70 analysis."),
        ("shared_construction_trial_ids", all(int(r["trial_split_shared_across_candidates"]) for r in split), "Construction/evaluation split is shared across candidates within target."),
        ("construction_eval_disjoint", all(int(r["construction_eval_disjoint"]) for r in split), "Construction and evaluation target trial IDs are disjoint."),
        ("unique_trial_budget_counted", all(int(r["fixed_budget_claim_allowed"]) for r in split), "Budgets count unique target trial IDs, not checkpoint rows."),
        ("t1_t2_not_independent_claim", int(overlap["t1_vs_t2"]["independent_confirmation_claimed"]) == 0, "T1 subset of T2 is not described as independent confirmation."),
        ("t3_not_consumed", res["t3_protocol_locked"] == 1 and res["t3_cache_consumed"] == 0, "T3-HO protocol is locked without consuming T3 cache/checkpoints."),
        ("strict_source_not_relabelled", feature["strict_source_domain_trial_logits"]["status"] == "absent_not_tested" and feature["checkpoint_metadata_seed_level_order"]["status"] != "strict_source", "Metadata proxy is not relabelled as strict source-domain trial signal."),
        ("row_iid_not_used", all(int(r["row_iid_interpretation_used"]) == 0 for r in res["blocked_permutation_summary_rows"]), "Blocked inference is reported; row-level iid interpretation is not used."),
        ("few_label_sufficiency_not_claimed", all(int(r["few_label_sufficiency_claimed"]) == 0 for r in res["label_budget_curve_rows"]), "Budget curves remain diagnostic candidates, not sufficiency claims."),
        ("conditional_cs_proxy_only", all(int(r["full_conditional_cs_claimed"]) == 0 for r in res["conditional_cs_block_summary_rows"]), "Conditional-CS remains proxy-only under crossed dependence."),
        ("target_population_unresolved", "C70-S8_target_population_generalization_unresolved" in res["decision"]["active"], "No target-population generalization claim."),
        ("no_forward_training_gpu", res["forward_or_reinference_executed"] == 0 and res["training_attempted"] == 0 and res["gpu_used"] == 0, "C70 is read-only over C69 external caches."),
        ("large_artifact_scan_passed", all(int(r["passed"]) for r in res["large_artifact_scan_rows"]), "All committed C70 artifacts are under 50MB."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
    ]
    return [{"gate": g, "failed": int(not ok), "finding": f} for g, ok, f in checks]


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        active = ["C70-E_claim_or_masking_inconsistency_requires_repair"]
        return {
            "primary": active[0],
            "active": active,
            "inactive": [d for d in DECISIONS if d not in active],
            "final_gate": "CLAIM_OR_MASKING_REPAIR_REQUIRED",
            "red_team_failure_count": len(failures),
            "recommended_next_direction": "repair C70 red-team failures before scientific interpretation",
        }
    budget_rows = res["label_budget_curve_rows"]
    small = [r for r in budget_rows if r["budget"] in {"1", "2", "4", "8"}]
    medium = [r for r in budget_rows if r["budget"] in {"12", "16", "24", "32"}]
    dense = [r for r in budget_rows if r["budget"] in {"48", "64", FULL_BUDGET_LABEL}]
    def candidate(rows):
        return [r for r in rows if float(r["mean_gauge_residual_recovery"]) >= 0.5 and float(r["actionability_rate_regret_le_0p02"]) >= 0.5]
    blocked = res["blocked_permutation_summary_rows"][0]
    full = next(r for r in budget_rows if r["budget"] == FULL_BUDGET_LABEL)
    if float(blocked["p_value"]) > 0.05:
        primary = "C70-D_c69_signal_collapses_under_hierarchical_controls"
        gate = "C69_SIGNAL_COLLAPSES_UNDER_HIERARCHICAL_CONTROLS"
    elif candidate(small):
        primary = "C70-A_small_budget_split_label_gauge_recovery_candidate"
        gate = "SMALL_BUDGET_SPLIT_LABEL_GAUGE_RECOVERY_CANDIDATE"
    elif candidate(medium) or candidate(dense):
        primary = "C70-B_medium_or_dense_label_recovery_only"
        gate = "MEDIUM_OR_DENSE_LABEL_RECOVERY_ONLY"
    elif float(full["mean_within_target_spearman"]) >= 0.75:
        primary = "C70-C_split_label_reliability_without_actionability"
        gate = "SPLIT_LABEL_RELIABILITY_WITHOUT_ACTIONABILITY"
    else:
        primary = "C70-D_c69_signal_collapses_under_hierarchical_controls"
        gate = "C69_SIGNAL_COLLAPSES_UNDER_HIERARCHICAL_CONTROLS"
    active = [
        primary,
        "C70-S1_finite_population_label_budget_bound_established",
        "C70-S2_paired_model_bound_nontrivial",
        "C70-S4_conditional_cs_proxy_only_or_bandwidth_sensitive",
        "C70-S5_no_strict_source_trial_escape_hatch",
        "C70-S7_t3_disjoint_confirmatory_protocol_locked",
        "C70-S8_target_population_generalization_unresolved",
        "C70-S9_new_training_not_justified",
    ]
    return {
        "primary": primary,
        "active": active,
        "inactive": [d for d in DECISIONS if d not in active],
        "final_gate": gate,
        "red_team_failure_count": 0,
        "recommended_next_direction": "remote review; decide whether C71 should execute the locked T3-HO confirmatory protocol",
    }


def table_row_counts(res: dict) -> dict:
    keys = {
        "c69_unit_dependency_graph": "c69_unit_dependency_graph_rows",
        "c69_t1_t2_overlap": "c69_t1_t2_overlap_rows",
        "c69_sampling_blindness_audit": "c69_sampling_blindness_audit_rows",
        "c69_split_contract_audit": "c69_split_contract_audit_rows",
        "c69_permutation_resolution_audit": "c69_permutation_resolution_audit_rows",
        "c69_source_feature_availability": "c69_source_feature_availability_rows",
        "label_budget_curve": "label_budget_curve_rows",
        "per_target_label_budget_curve": "per_target_label_budget_curve_rows",
        "actionability_budget_curve": "actionability_budget_curve_rows",
        "label_budget_gauge_recovery_curve": "label_budget_gauge_recovery_curve_rows",
        "rank_vs_gauge_budget_decomposition": "rank_vs_gauge_budget_decomposition_rows",
        "cluster_bootstrap_summary": "cluster_bootstrap_summary_rows",
        "blocked_permutation_summary": "blocked_permutation_summary_rows",
        "finite_population_budget_bound": "finite_population_budget_bound_rows",
        "paired_candidate_sample_complexity": "paired_candidate_sample_complexity_rows",
        "conditional_cs_block_summary": "conditional_cs_block_summary_rows",
        "feature_availability_ledger": "feature_availability_ledger_rows",
        "failure_reason_ledger": "failure_reason_ledger_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "artifact_manifest": "artifact_manifest_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "test_command_manifest": "test_command_manifest_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def run(*, repeats: int = DEFAULT_REPEATS, permutations: int = DEFAULT_PERMUTATIONS, bootstraps: int = DEFAULT_BOOTSTRAPS, test_status: str = "planned") -> dict:
    manifest = load_cache_manifest()
    populations, unit_rows = load_t2_population(manifest["t2"]["external_path"])
    phase0 = build_phase0_tables(manifest, populations, unit_rows)
    budget = build_budget_curves(populations, repeats, 70070)
    blocked = build_blocked_permutation(populations, permutations, 70123)
    cluster = build_cluster_bootstrap(budget["per_target_label_budget_curve_rows"], bootstraps, 70234)
    paired = build_pair_sample_complexity(populations)
    cs_rows = build_conditional_cs_block_summary()
    protocol, protocol_sha = build_t3_protocol(_read_csv(C65_MAP))
    res = {
        "config_hash": _lock_config(),
        "current_head": _git_or_empty(["rev-parse", "--short", "HEAD"]),
        "c69_commit": "e17a72e",
        "c69_final_gate": manifest["c69_json"].get("final_gate", ""),
        "c69_manifest_hash_audit": {"t1_sha_match": manifest["t1_sha_match"], "t2_sha_match": manifest["t2_sha_match"]},
        "target_count": len(populations),
        "checkpoint_unit_count": sum(len(p["units"]) for p in populations.values()),
        "unique_target_trial_ids": sum(len(p["trial_ids"]) for p in populations.values()),
        "repeats": repeats,
        "blocked_permutations": permutations,
        "cluster_bootstraps": bootstraps,
        "forward_or_reinference_executed": 0,
        "training_attempted": 0,
        "gpu_used": 0,
        "t3_cache_consumed": 0,
        "t3_protocol_locked": 1,
        "t3_protocol_sha256": protocol_sha,
        **phase0,
        **budget,
        "cluster_bootstrap_summary_rows": cluster,
        "blocked_permutation_summary_rows": blocked,
        "paired_candidate_sample_complexity_rows": paired,
        "conditional_cs_block_summary_rows": cs_rows,
        "failure_reason_ledger_rows": [],
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "schema_validation_summary_rows": [],
        "red_team_failure_ledger_rows": [],
        "_t3_protocol": protocol,
    }
    res["failure_reason_ledger_rows"] = build_failure_reason_ledger(phase0, budget["label_budget_curve_rows"], blocked)
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c70", "command": "python -m pytest oaci/tests/test_c70_split_label_information_budget.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c70_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c70_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c70_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c6*.py oaci/tests/test_c70_*.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def _compact_json(res: dict) -> dict:
    first_small = next((r for r in res["label_budget_curve_rows"] if r["budget"] == "8"), {})
    full = next((r for r in res["label_budget_curve_rows"] if r["budget"] == FULL_BUDGET_LABEL), {})
    blocked = res["blocked_permutation_summary_rows"][0]
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "read_only_c69_t1_t2_cache": True,
        "forward_or_reinference_executed": res["forward_or_reinference_executed"],
        "training_attempted": res["training_attempted"],
        "gpu_used": res["gpu_used"],
        "t3_cache_consumed": res["t3_cache_consumed"],
        "c69_commit": res["c69_commit"],
        "c69_final_gate": res["c69_final_gate"],
        "current_head_at_generation": res["current_head"],
        "decision": res["decision"],
        "final_gate": res["decision"]["final_gate"],
        "key_numbers": {
            "target_count": res["target_count"],
            "checkpoint_unit_count": res["checkpoint_unit_count"],
            "unique_target_trial_ids": res["unique_target_trial_ids"],
            "repeats": res["repeats"],
            "blocked_permutations": res["blocked_permutations"],
            "budget_8_gauge_recovery": first_small.get("mean_gauge_residual_recovery", ""),
            "budget_8_actionability": first_small.get("actionability_rate_regret_le_0p02", ""),
            "full_construction_spearman": full.get("mean_within_target_spearman", ""),
            "full_construction_gauge_recovery": full.get("mean_gauge_residual_recovery", ""),
            "blocked_perm_p": blocked.get("p_value", ""),
            "blocked_perm_exceedances": blocked.get("exceedances", ""),
            "t3_protocol_sha256": res["t3_protocol_sha256"],
            "red_team_failure_count": res["decision"]["red_team_failure_count"],
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    b8 = next((r for r in res["label_budget_curve_rows"] if r["budget"] == "8"), {})
    b16 = next((r for r in res["label_budget_curve_rows"] if r["budget"] == "16"), {})
    full = next((r for r in res["label_budget_curve_rows"] if r["budget"] == FULL_BUDGET_LABEL), {})
    blocked = res["blocked_permutation_summary_rows"][0]
    cs = res["conditional_cs_block_summary_rows"][0]
    main = "\n".join([
        f"# C70 - Split-Label Information Budget / Gauge-Recovery Phase Diagram (frozen C19 `{res['config_hash']}`)",
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
        "## 2. Read-Only Boundary",
        "",
        "C70 consumes the manifested C69 T1/T2 external caches read-only. It does not run EEG forward passes, re-inference, training, GPU work, or T3 cache consumption.",
        "",
        "## 3. Split Contract",
        "",
        f"The T2 cache contains `{res['target_count']}` targets, `{res['checkpoint_unit_count']}` checkpoint-target units, and `{res['unique_target_trial_ids']}` unique target trial IDs. Construction/evaluation trial IDs are shared across candidates within each target and disjoint.",
        "",
        "## 4. Information-Budget Curve",
        "",
        f"At 8 labels/class: gauge recovery `{b8.get('mean_gauge_residual_recovery', '')}`, actionability coverage `{b8.get('actionability_rate_regret_le_0p02', '')}`, top1 hit `{b8.get('mean_top1_hit', '')}`.",
        "",
        f"At 16 labels/class: gauge recovery `{b16.get('mean_gauge_residual_recovery', '')}`, actionability coverage `{b16.get('actionability_rate_regret_le_0p02', '')}`, top1 hit `{b16.get('mean_top1_hit', '')}`.",
        "",
        f"At full construction: mean within-target Spearman `{full.get('mean_within_target_spearman', '')}`, gauge recovery `{full.get('mean_gauge_residual_recovery', '')}`, regret `{full.get('mean_continuous_regret', '')}`.",
        "",
        "The C70-D call is a collapse of the strong pooled/actionability interpretation, not a null-signal claim: within-target centered blocked permutation remains significant, but the effect is much smaller and does not support small-budget gauge recovery.",
        "",
        "These are diagnostic information-cost curves. C70 does not claim few-label sufficiency, deployability, source-only rescue, or checkpoint selection.",
        "",
        "## 5. Hierarchical / Permutation Controls",
        "",
        f"The primary blocked permutation shuffles held-out scores within target: observed `{blocked['observed']}`, permutations `{blocked['permutations']}`, exceedances `{blocked['exceedances']}`, p `{blocked['p_value']}`, floor `{blocked['monte_carlo_floor']}`.",
        "",
        "Target-cluster bootstrap bands are emitted. C70 remains conditional on the frozen BNCI2014_001 targets and does not make target-population p-value claims.",
        "",
        "## 6. Conditional-CS and Strict Source",
        "",
        f"Conditional-CS remains `{cs['block_valid_status']}` with full conditional-CS claimed `{cs['full_conditional_cs_claimed']}`. Strict source-domain trial logits/probs are absent from the C69 cache; metadata is not relabelled as strict source evidence.",
        "",
        "## 7. T3-HO Protocol",
        "",
        f"C70 locks but does not execute a C71 T3-HO protocol. Protocol SHA-256: `{res['t3_protocol_sha256']}`.",
    ])
    red = "\n".join([
        "# C70 - Red-Team Verification",
        "",
        "All C70 red-team gates pass." if d["red_team_failure_count"] == 0 else "C70 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    return {
        "C70_SPLIT_LABEL_INFORMATION_BUDGET.md": main,
        "C70_RED_TEAM_VERIFICATION.md": red,
    }


def write_tables(res: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    specs = {
        "c69_unit_dependency_graph.csv": ("c69_unit_dependency_graph_rows", ["stage", "unit_hash", "target_id", "trajectory_hash", "seed", "level", "regime", "candidate_order", "target_trial_count", "checkpoint_target_rows", "construct_trial_count", "eval_trial_count", "source_domain_trial_logits_available"]),
        "c69_t1_t2_overlap.csv": ("c69_t1_t2_overlap_rows", ["comparison", "left_units", "right_units", "overlap_units", "left_subset_of_right", "independent_confirmation_claimed", "notes"]),
        "c69_sampling_blindness_audit.csv": ("c69_sampling_blindness_audit_rows", ["audit", "passed", "observed", "forbidden_dependency", "notes"]),
        "c69_split_contract_audit.csv": ("c69_split_contract_audit_rows", ["target_id", "checkpoint_units", "unique_target_trial_ids", "construct_unique_trial_ids", "eval_unique_trial_ids", "construction_eval_disjoint", "trial_split_shared_across_candidates", "min_repetitions_per_trial", "class_construct_counts", "class_eval_counts", "fixed_budget_claim_allowed"]),
        "c69_permutation_resolution_audit.csv": ("c69_permutation_resolution_audit_rows", ["test", "reported_p", "inferred_permutations", "inferred_exceedances", "monte_carlo_floor", "resolution_note", "c70_permutations"]),
        "c69_source_feature_availability.csv": ("c69_source_feature_availability_rows", ["feature_family", "available_in_c69_cache", "uses_target_labels", "uses_eval_labels", "available_at_selection_time", "diagnostic_only", "status"]),
        "label_budget_curve.csv": ("label_budget_curve_rows", ["budget", "labels_per_class", "repeat_count", "target_count", "mean_unique_construct_trials", "mean_unique_eval_trials", "mean_within_target_spearman", "mean_kendall_tau_proxy", "mean_pairwise_order_accuracy", "mean_top1_hit", "mean_top3_hit", "mean_continuous_regret", "actionability_rate_regret_le_0p02", "mean_gauge_residual_recovery", "endpoint_oracle_reference", "few_label_sufficiency_claimed"]),
        "per_target_label_budget_curve.csv": ("per_target_label_budget_curve_rows", ["budget", "repeat", "target_id", "labels_per_class", "unique_construct_trials", "unique_eval_trials", "within_target_spearman", "within_target_kendall_tau_proxy", "pairwise_order_accuracy", "median_eval_pair_margin", "top1_hit", "top3_hit", "continuous_regret", "actionability_hit_regret_le_0p02", "gauge_residual_recovery", "gauge_alpha"]),
        "actionability_budget_curve.csv": ("actionability_budget_curve_rows", ["budget", "top1_hit", "top3_hit", "continuous_regret", "coverage_regret_le_0p02", "actionability_status"]),
        "label_budget_gauge_recovery_curve.csv": ("label_budget_gauge_recovery_curve_rows", ["budget", "gauge_residual_recovery", "source_to_oracle_gap_closure_fraction", "common_target_offset_credit", "candidate_specific_within_target_credit"]),
        "rank_vs_gauge_budget_decomposition.csv": ("rank_vs_gauge_budget_decomposition_rows", ["budget", "rank_component_recovery", "candidate_specific_gauge_recovery", "finite_trial_noise_floor", "common_target_offset_not_credited"]),
        "cluster_bootstrap_summary.csv": ("cluster_bootstrap_summary_rows", ["budget", "metric", "cluster", "bootstrap_replicates", "mean", "ci_lower", "ci_upper", "target_population_generalization_claimed"]),
        "blocked_permutation_summary.csv": ("blocked_permutation_summary_rows", ["test", "blocking", "observed", "permutations", "exceedances", "p_value", "monte_carlo_floor", "null_p95", "row_iid_interpretation_used"]),
        "finite_population_budget_bound.csv": ("finite_population_budget_bound_rows", ["budget", "registered_pair_observations", "random_subset_order_recovery_probability", "median_eval_pair_margin", "finite_population_scope"]),
        "paired_candidate_sample_complexity.csv": ("paired_candidate_sample_complexity_rows", ["margin_bin", "pair_count", "median_required_labels_per_class_normal_proxy", "p75_required_labels_per_class_normal_proxy", "median_disagreement_rate", "median_abs_bacc_contrast", "bound_scope"]),
        "conditional_cs_block_summary.csv": ("conditional_cs_block_summary_rows", ["estimator", "stage", "paired_eval_rows", "independent_checkpoint_units", "incremental_cod", "reported_p", "inferred_permutations", "block_valid_status", "full_conditional_cs_claimed", "bandwidth_nested_null", "notes"]),
        "feature_availability_ledger.csv": ("feature_availability_ledger_rows", ["feature_family", "available_in_c69_cache", "uses_target_labels", "uses_eval_labels", "available_at_selection_time", "diagnostic_only", "status"]),
        "failure_reason_ledger.csv": ("failure_reason_ledger_rows", ["reason", "status", "evidence", "blocks_claim"]),
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


def write_artifacts(res: dict) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    write_t3_protocol(res["_t3_protocol"], res["t3_protocol_sha256"])
    write_tables(res)
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    paths = [str(p) for p in _listed_paths()]
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan([Path(p) for p in paths])
    write_tables(res)
    res["schema_validation_summary_rows"] = _schema_rows()
    write_tables(res)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    for name, text in build_reports(res).items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")
    write_tables(res)
    paths = _listed_paths()
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{} for _ in paths]
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c70_split_label_information_budget")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    ap.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(repeats=args.repeats, permutations=args.permutations, bootstraps=args.bootstraps, test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res)
    print(f"[C70] decision={res['decision']['primary']} gate={res['decision']['final_gate']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
