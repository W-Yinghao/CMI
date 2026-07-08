"""C36 read-only artifact loading for C35 robust pairs and C10 selector traces."""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from . import schema


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _body(path):
    d = json.load(open(path))
    return d.get("body", d)


def as_float(v, default=np.nan):
    try:
        if v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def as_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def finite(v) -> bool:
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def pair_id(seed, target, level, regime, comparison, selected_order, candidate_order) -> str:
    return "|".join(map(str, (seed, target, level, regime, comparison, selected_order, candidate_order)))


def load_preference_robust_pairs(c35_table_dir=None, c34_table_dir=None):
    c35_table_dir = c35_table_dir or schema.C35_TABLE_DIR
    c34_table_dir = c34_table_dir or schema.C34_TABLE_DIR
    simplex = read_csv(os.path.join(c35_table_dir, "utility_simplex_regret_by_pair.csv"))
    pareto = {r["pair_id"]: r for r in read_csv(os.path.join(c35_table_dir, "pareto_status_selected_pairs.csv"))}
    c34 = {}
    for r in read_csv(os.path.join(c34_table_dir, "selected_vs_continuous_better_pairs.csv")):
        pid = pair_id(r["seed"], r["target"], r["level"], r.get("regime", ""), r["comparison"],
                      r["selected_order"], r["candidate_order"])
        c34[pid] = r

    out = []
    for r in simplex:
        if (r["comparison"] != schema.ROBUST_COMPARISON or
                r["scaling"] != "raw" or
                r["utility_cone_category"] != schema.ROBUST_CATEGORY):
            continue
        pid = r["pair_id"]
        if pid not in c34:
            raise KeyError(f"C36 robust pair missing C34 row: {pid}")
        pieces = pid.split("|")
        row = {**c34[pid], **r}
        row["pair_id"] = pid
        row["seed"] = str(as_int(pieces[0]))
        row["target"] = str(as_int(pieces[1]))
        row["level"] = str(as_int(pieces[2]))
        row["regime"] = pieces[3]
        row["selected_order"] = str(as_int(pieces[-2]))
        row["candidate_order"] = str(as_int(pieces[-1]))
        row["pareto_status"] = pareto.get(pid, {}).get("pareto_status", "")
        row["endpoint_tradeoff_c35"] = pareto.get(pid, {}).get("endpoint_tradeoff", "")
        out.append(row)
    return out


def _method_body_for_level(artifact_dir, level, method="OACI"):
    path = os.path.join(artifact_dir, f"levels/level-{int(level):03d}", "methods", method, "method.json")
    return _body(path)


def _candidate_id(seed, target, level, regime, order):
    label = "erm" if int(order) < 0 else f"o{int(order):03d}"
    return f"s{int(seed)}_t{int(target):03d}_l{int(level):03d}_{regime}_{label}"


def load_c10_selector_trace(c10_dir=None, regimes=None):
    c10_dir = c10_dir or schema.C10_REPLAY_DIR
    regimes = tuple(sorted(regimes or ("S0_full_support",)))
    paths = sorted(glob.glob(os.path.join(c10_dir, "seed-*-target-*.json")))
    if not paths:
        raise FileNotFoundError(f"no C10 replay json files under {c10_dir}")

    registry = []
    unit_rows = defaultdict(list)
    selected_meta_by_unit = {}
    for path in paths:
        d = json.load(open(path))
        seed, target = str(int(d["seed"])), str(int(d["target"]))
        for level, lv in sorted(d["levels"].items(), key=lambda kv: int(kv[0])):
            method = _method_body_for_level(d["artifact_dir"], level, "OACI")
            sel = method["selection"]
            tau = as_float(method.get("shared_tau"))
            selected_hash = lv["selected"]["OACI"]
            selected_score = as_float(sel.get("selection_score"))
            selected_score_name = sel.get("score_name")
            selected_reason = sel.get("selection_reason")
            selected_status = sel.get("selection_status")
            n_feasible = as_int(sel.get("n_feasible"))
            order = 0
            per_regime_rows = {regime: [] for regime in regimes}
            for raw_index, c in enumerate(lv["candidates"]):
                is_erm = bool(c.get("is_erm"))
                candidate_order = -1 if is_erm else order
                if not is_erm:
                    order += 1
                selected = bool(c.get("model_hash") == selected_hash)
                base = {
                    "seed": seed,
                    "target": target,
                    "level": str(int(level)),
                    "candidate_order": candidate_order,
                    "raw_replay_index": raw_index,
                    "candidate_role": "ERM" if is_erm else "OACI",
                    "origin": c.get("origin"),
                    "is_erm": int(is_erm),
                    "selected_oaci": int(selected),
                    "epoch": c.get("epoch"),
                    "lambda": c.get("lambda"),
                    "R_src": c.get("R_src"),
                    "shared_tau": tau,
                    "risk_slack_to_tau": (tau - as_float(c.get("R_src")) if finite(tau) and finite(c.get("R_src")) else ""),
                    "feasible": int(bool(c.get("feasible"))),
                    "balanced_err": c.get("balanced_err"),
                    "train_surrogate": c.get("train_surrogate"),
                    "selection_leakage_point": c.get("selection_leakage_point"),
                    "audit_leakage_point": c.get("audit_leakage_point"),
                    "source_guard_worst_bacc": c.get("source_guard_worst_bacc"),
                    "source_guard_worst_nll": c.get("source_guard_worst_nll"),
                    "source_guard_worst_ece": c.get("source_guard_worst_ece"),
                    "source_audit_worst_bacc": c.get("source_audit_worst_bacc"),
                    "source_audit_worst_nll": c.get("source_audit_worst_nll"),
                    "source_audit_worst_ece": c.get("source_audit_worst_ece"),
                    "target_worst_bacc": c.get("target_worst_bacc"),
                    "target_worst_nll": c.get("target_worst_nll"),
                    "target_worst_ece": c.get("target_worst_ece"),
                    "actual_selector_score_name": selected_score_name,
                    "actual_selector_score_ucl": selected_score if selected else "",
                    "actual_selector_score_available": int(selected and finite(selected_score)),
                    "per_candidate_selector_ucl_available": 0,
                    "actual_selector_rank_known": int(selected),
                    "actual_selector_rank": 1 if selected else "",
                    "actual_selector_relation": "selected_by_actual_rule" if selected else "not_selected_by_actual_rule",
                    "selection_reason": selected_reason if selected else "",
                    "selection_status": selected_status,
                    "n_feasible": n_feasible,
                    "checkpoint_hash_available": 1,
                    "checkpoint_hash_emitted": 0,
                    "tie_break_metadata_available": 0,
                }
                for regime in regimes:
                    row = dict(base)
                    row["regime"] = regime
                    row["candidate_id"] = _candidate_id(seed, target, level, regime, candidate_order)
                    registry.append(row)
                    per_regime_rows[regime].append(row)
            for regime in regimes:
                key = (seed, target, str(int(level)), regime)
                unit_rows[key].extend(per_regime_rows[regime])
                selected_meta_by_unit[key] = {
                    "actual_selector_score_name": selected_score_name,
                    "selected_actual_selector_ucl": selected_score,
                    "selection_reason": selected_reason,
                    "selection_status": selected_status,
                    "n_feasible": n_feasible,
                    "shared_tau": tau,
                    "per_candidate_selector_ucl_available": False,
                    "tie_break_metadata_available": False,
                }

    by_key = {}
    for row in registry:
        by_key[(row["seed"], row["target"], row["level"], row["regime"], str(row["candidate_order"]))] = row
    for key in unit_rows:
        unit_rows[key] = sorted(unit_rows[key], key=lambda r: (int(r["candidate_order"]), as_float(r["epoch"])))
    return {"registry": registry, "by_key": by_key, "unit_rows": dict(unit_rows),
            "selected_meta_by_unit": selected_meta_by_unit, "c10_files": paths}


def regimes_from_pairs(pairs):
    return tuple(sorted({p["regime"] for p in pairs}))

