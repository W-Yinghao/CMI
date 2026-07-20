"""C37 read-only loaders over C36/C35/C10 and Phase-A replay stores."""
from __future__ import annotations

import csv
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from ..artifacts import plan_codec as P
from ..artifacts import support_codec as S
from ..artifacts.reader import read_artifact
from ..runner.replay_store import ReplayStore
from . import schema


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def body(path):
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


def pair_id(seed, target, level, regime, comparison, selected_order, candidate_order):
    return "|".join(map(str, (seed, target, level, regime, comparison, selected_order, candidate_order)))


def _candidate_id(seed, target, level, regime, order):
    return f"s{int(seed)}_t{int(target):03d}_l{int(level):03d}_{regime}_o{int(order):03d}"


def load_robust_pairs():
    simplex = read_csv(os.path.join(schema.C35_TABLE_DIR, "utility_simplex_regret_by_pair.csv"))
    c34 = {}
    for r in read_csv(os.path.join(schema.C34_TABLE_DIR, "selected_vs_continuous_better_pairs.csv")):
        pid = pair_id(r["seed"], r["target"], r["level"], r.get("regime", ""), r["comparison"],
                      r["selected_order"], r["candidate_order"])
        c34[pid] = r
    c36_trace = {r["pair_id"]: r for r in read_csv(os.path.join(schema.C36_TABLE_DIR,
                                                                "selected_vs_better_selector_trace.csv"))}
    out = []
    for r in simplex:
        if (r["comparison"] != schema.ROBUST_COMPARISON or r["scaling"] != "raw" or
                r["utility_cone_category"] != schema.ROBUST_CATEGORY):
            continue
        pid = r["pair_id"]
        pieces = pid.split("|")
        row = {**c34[pid], **r, **c36_trace.get(pid, {})}
        row["pair_id"] = pid
        row["seed"], row["target"], row["level"], row["regime"] = pieces[0], pieces[1], pieces[2], pieces[3]
        row["selected_order"], row["candidate_order"] = pieces[-2], pieces[-1]
        out.append(row)
    return out


def load_c10_trace(regimes):
    paths = sorted(glob.glob(os.path.join(schema.C10_REPLAY_DIR, "seed-*-target-*.json")))
    if not paths:
        raise FileNotFoundError(schema.C10_REPLAY_DIR)
    by_key = {}
    units = {}
    artifact_dirs = {}
    for path in paths:
        d = json.load(open(path))
        seed, target = str(int(d["seed"])), str(int(d["target"]))
        for level, lv in sorted(d["levels"].items(), key=lambda kv: int(kv[0])):
            method = body(os.path.join(d["artifact_dir"], f"levels/level-{int(level):03d}/methods/OACI/method.json"))
            selected_hash = lv["selected"]["OACI"]
            selected_leak = body(os.path.join(d["artifact_dir"],
                                              f"levels/level-{int(level):03d}/methods/OACI/selection_leakage.json"))
            order = 0
            raw_rows = []
            for c in lv["candidates"]:
                if c.get("is_erm"):
                    continue
                row = dict(c)
                row["order"] = order
                row["selected_oaci"] = int(c["model_hash"] == selected_hash)
                row["persisted_selected_ucl"] = selected_leak["bootstrap_ucl"] if row["selected_oaci"] else ""
                row["persisted_selected_point"] = selected_leak["extractable_LQ_ov"] if row["selected_oaci"] else ""
                row["persisted_selected_percentile_ucl"] = selected_leak.get("percentile_ucl") if row["selected_oaci"] else ""
                row["artifact_dir"] = d["artifact_dir"]
                row["selection_score"] = method["selection"]["selection_score"] if row["selected_oaci"] else ""
                raw_rows.append(row)
                order += 1
            for regime in regimes:
                key = (seed, target, str(int(level)), regime)
                artifact_dirs[key] = d["artifact_dir"]
                copied = []
                for r in raw_rows:
                    rr = dict(r)
                    rr["regime"] = regime
                    rr["candidate_id"] = _candidate_id(seed, target, level, regime, rr["order"])
                    by_key[(seed, target, str(int(level)), regime, str(rr["order"]))] = rr
                    copied.append(rr)
                units[key] = copied
    return {"by_key": by_key, "units": units, "artifact_dirs": artifact_dirs}


def unique_pair_keys(pairs):
    seen = {}
    for p in pairs:
        k = (p["seed"], p["target"], p["level"], p["selected_order"], p["candidate_order"])
        seen.setdefault(k, p)
    return list(seen.values())


def p0_slice(pairs):
    unique = sorted({(p["seed"], p["target"], p["level"], p["regime"], p["selected_order"]) for p in pairs},
                    key=lambda x: (int(x[0]), int(x[1]), int(x[2]), x[3], int(x[4])))
    if len(unique) <= schema.P0_SLICE_SIZE:
        chosen = unique
    else:
        idx = np.linspace(0, len(unique) - 1, schema.P0_SLICE_SIZE, dtype=int).tolist()
        chosen = [unique[i] for i in idx]
    return chosen


class UnitContext:
    def __init__(self, artifact_dir, level):
        self.artifact_dir = artifact_dir
        self.level = int(level)
        self.staging_dir = artifact_dir.split("/artifacts/")[0] + "/staging"
        self.store_path = os.path.join(self.staging_dir, f"level-{self.level}-store.pkl")
        self.fold_path = os.path.join(self.staging_dir, "fold.pkl")
        self.store = None
        self.fold = None
        self._feature_keys = None
        self.support_graph = None
        self.fold_plan = None
        self.bootstrap_plan = None

    def load(self):
        if self.store is None:
            self.store = ReplayStore.load(self.store_path)
        if self.fold is None:
            with open(self.fold_path, "rb") as f:
                self.fold = pickle.load(f)
        if self.support_graph is None:
            base = os.path.join(self.artifact_dir, f"levels/level-{self.level:03d}")
            _, b, a = read_artifact(os.path.join(base, "support.json"), S.SUPPORT_KIND)
            self.support_graph = S.decode_support_graph(b, a)
            _, b, a = read_artifact(os.path.join(base, "plans/selection_fold_plan.json"), P.FOLD_KIND)
            self.fold_plan = P.decode_fold_plan(b, a)
            _, b, a = read_artifact(os.path.join(base, "plans/selection_bootstrap_plan.json"), P.BOOTSTRAP_KIND)
            self.bootstrap_plan = P.decode_bootstrap_plan(b, a)
        return self

    def feature_by_hash(self, model_hash):
        self.load()
        if self._feature_keys is None:
            self._feature_keys = {getattr(k, "model_hash", None): k
                                  for kind, k in self.store._d if kind == "feat:source_train"}
        key = self._feature_keys.get(model_hash)
        if key is None:
            raise KeyError("source-train feature missing for candidate")
        return self.store.lookup("feat:source_train", key)


class ContextCache:
    def __init__(self, trace):
        self.trace = trace
        self._cache = {}

    def get(self, seed, target, level, regime):
        key = (str(seed), str(target), str(level), str(regime))
        ck = (str(seed), str(target), str(level))
        if ck not in self._cache:
            self._cache[ck] = UnitContext(self.trace["artifact_dirs"][key], level)
        return self._cache[ck]


def recovery_manifest(pairs, trace, feature_status=None):
    rows = []
    ctx_cache = ContextCache(trace)
    feature_status = feature_status or {}
    for p in unique_pair_keys(pairs):
        ctx = ctx_cache.get(p["seed"], p["target"], p["level"], p["regime"])
        store_exists = os.path.exists(ctx.store_path)
        selected_key = ("selected", p["seed"], p["target"], p["level"], p["regime"], p["selected_order"])
        better_key = ("better", p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])
        selected_feature = feature_status.get(selected_key, "")
        better_feature = feature_status.get(better_key, "")
        feature_checked = int(selected_key in feature_status or better_key in feature_status)
        rows.append({
            "unit_id": f"s{p['seed']}_t{int(p['target']):03d}_l{int(p['level']):03d}",
            "seed": p["seed"],
            "target": p["target"],
            "level": p["level"],
            "regime": p["regime"],
            "selected_order": p["selected_order"],
            "better_order": p["candidate_order"],
            "store_exists": int(store_exists),
            "selection_design_available": int(os.path.exists(os.path.join(ctx.artifact_dir,
                                           f"levels/level-{int(p['level']):03d}/plans/selection_design.json"))),
            "selection_fold_plan_available": int(os.path.exists(os.path.join(ctx.artifact_dir,
                                               f"levels/level-{int(p['level']):03d}/plans/selection_fold_plan.json"))),
            "selection_bootstrap_plan_available": int(os.path.exists(os.path.join(ctx.artifact_dir,
                                                    f"levels/level-{int(p['level']):03d}/plans/selection_bootstrap_plan.json"))),
            "support_graph_available": int(os.path.exists(os.path.join(ctx.artifact_dir,
                                         f"levels/level-{int(p['level']):03d}/support.json"))),
            "selected_source_train_feature_available": selected_feature,
            "better_source_train_feature_available": better_feature,
            "feature_availability_checked_by_worker": feature_checked,
            "target_labels_loaded_for_replay": 0,
        })
    return rows
