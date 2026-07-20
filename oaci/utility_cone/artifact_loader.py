"""C35 loader for C34S compact JSON and row-level CSV tables."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from ..continuous_regret import artifact_hygiene
from . import schema


REQUIRED_C34_TABLES = artifact_hygiene.TABLES


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _as_float(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _resolve_manifest(compact, report_root):
    rows = []
    for entry in compact["table_manifest"]:
        path = os.path.join(report_root, entry["path"])
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        got = {
            "table": entry["table"],
            "path": entry["path"],
            "bytes": os.path.getsize(path),
            "rows": artifact_hygiene.csv_row_count(path),
            "sha256": artifact_hygiene.sha256_file(path),
        }
        rows.append(got)
    return rows


def _manifest_ok(compact, resolved):
    expected = {r["table"]: r for r in compact["table_manifest"]}
    return all(expected[r["table"]]["bytes"] == r["bytes"] and
               expected[r["table"]]["rows"] == r["rows"] and
               expected[r["table"]]["sha256"] == r["sha256"] for r in resolved)


def _selected_key(r):
    return (r["seed"], r["target"], r["level"], r.get("regime", ""), r.get("selected_order"))


def _reconstruct_headlines(tables, compact):
    pairs = tables["selected_vs_continuous_better_pairs.csv"]
    cont = [r for r in pairs if r["comparison"] == "nearest_continuous_better"]
    binary = [r for r in pairs if r["comparison"] == "nearest_binary_joint_good" and r["selected_joint_good"] == "0"]
    mean = lambda key: float(np.mean([_as_float(r[key]) for r in cont])) if cont else None
    out = {
        "taxonomy_cases": tables["c34_case_taxonomy.csv"][0]["cases"].split(";"),
        "real_endpoint_regret_fraction": float(np.mean([_as_int(r["meaningful_continuous_regret"]) for r in cont])),
        "threshold_only_fraction": float(np.mean([_as_int(r["threshold_artifact"]) for r in binary])) if binary else None,
        "mean_target_bacc_delta": mean("target_bacc_delta"),
        "mean_target_nll_delta": mean("target_nll_delta"),
        "mean_target_ece_delta": mean("target_ece_delta"),
        "continuous_raw_pareto_nonworse_count": sum(
            _as_float(r["target_bacc_delta"]) >= 0 and _as_float(r["target_nll_delta"]) >= 0 and
            _as_float(r["target_ece_delta"]) >= 0 for r in cont),
        "continuous_raw_endpoint_backward_count": sum(
            _as_float(r["target_bacc_delta"]) < 0 or _as_float(r["target_nll_delta"]) < 0 or
            _as_float(r["target_ece_delta"]) < 0 for r in cont),
        "continuous_joint_min_negative_count": sum(_as_float(r["joint_min_margin_delta"]) < 0 for r in cont),
        "n_selected_continuous_better_pairs": len(cont),
    }
    agg = compact["key_aggregates"]["selected_pair_regret"]
    matches = (
        out["taxonomy_cases"] == compact["verdict"]["primary_cases"] and
        abs(out["real_endpoint_regret_fraction"] - agg["real_endpoint_regret_fraction"]) < 1e-12 and
        abs(out["threshold_only_fraction"] - agg["threshold_only_fraction_among_binary_misses"]) < 1e-12 and
        out["continuous_raw_pareto_nonworse_count"] == agg["continuous_raw_pareto_nonworse_count"] and
        out["continuous_raw_endpoint_backward_count"] == agg["continuous_raw_endpoint_backward_count"] and
        out["continuous_joint_min_negative_count"] == agg["continuous_joint_min_negative_count"]
    )
    out["matches_c34s_compact"] = bool(matches)
    return out


def load_c34s(json_path=None, table_dir=None):
    json_path = json_path or schema.C34_COMPACT_JSON
    table_dir = table_dir or schema.C34_TABLE_DIR
    report_root = os.path.dirname(table_dir)
    with open(json_path) as f:
        compact = json.load(f)
    resolved = _resolve_manifest(compact, report_root)
    manifest_tables = {r["table"] for r in resolved}
    tables = {name: _read_csv(os.path.join(table_dir, name)) for name in REQUIRED_C34_TABLES}
    headlines = _reconstruct_headlines(tables, compact)
    heavy_keys = {"endpoint_registry", "selected_pairs", "source_direction", "source_objective_components",
                  "gauge_locality"}
    gates = {
        "G0_manifest_resolves": manifest_tables == set(REQUIRED_C34_TABLES),
        "G1_table_hashes_match": _manifest_ok(compact, resolved),
        "G2_key_numbers_reconstruct": headlines["matches_c34s_compact"],
        "G3_no_legacy_monolithic_dependency": (
            compact.get("schema_version") == "C34S-compact-v1" and
            compact.get("payload_policy") == "row_level_payload_in_c34_tables_csv" and
            os.path.getsize(json_path) < 200_000 and
            not any(k in compact for k in heavy_keys)
        ),
    }
    return {"compact": compact, "manifest": resolved, "tables": tables, "c34_reconstruction": headlines,
            "c34s_gates": gates}


def endpoint_registry_map(rows):
    out = {}
    for r in rows:
        key = (r["seed"], r["target"], r["level"], r.get("regime", ""), r.get("order"))
        out[key] = r
    return out


def primary_pairs(tables):
    return [r for r in tables["selected_vs_continuous_better_pairs.csv"]
            if r["comparison"] == "nearest_continuous_better"]


def all_pairs(tables):
    return tables["selected_vs_continuous_better_pairs.csv"]
