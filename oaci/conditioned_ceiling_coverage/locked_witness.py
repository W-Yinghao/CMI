"""Locked C49/C50 broad-witness registry."""
from __future__ import annotations

import json
import os

from . import audit_utils as au
from . import schema


WITNESS_SCOPE = "within_target"
WITNESS_SOURCE_SPACE = "all_source_objectives"
WITNESS_NEIGHBORHOOD = "eps_q20"
WITNESS_MIN_NEIGHBOR_COUNT = 1
WITNESS_LABEL = "primary_joint_good"
WITNESS_EPS_QUANTILE = "q20"
C49_COMMIT = "b0d7831"


def c49_locked_witness():
    c49 = json.load(open("oaci/reports/C49_SPARSE_LOCAL_BAYES_COVERAGE_AUDIT.json"))
    metrics = c49["taxonomy"]["primary_metrics"]
    required = {
        "coverage50_best_scope": WITNESS_SCOPE,
        "coverage50_best_source_space": WITNESS_SOURCE_SPACE,
        "coverage50_best_neighborhood": WITNESS_NEIGHBORHOOD,
        "coverage50_best_min_neighbor_count": WITNESS_MIN_NEIGHBOR_COUNT,
    }
    for key, expected in required.items():
        if metrics.get(key) != expected:
            raise ValueError(f"locked witness mismatch for {key}: {metrics.get(key)} != {expected}")
    rows = [
        r for r in au.read_csv(os.path.join(schema.C49_TABLE_DIR, "coverage_accuracy_curve.csv"))
        if r["group_scope"] == WITNESS_SCOPE and
        r["source_space"] == WITNESS_SOURCE_SPACE and
        r["neighborhood"] == WITNESS_NEIGHBORHOOD and
        int(float(r["min_neighbor_count"])) == WITNESS_MIN_NEIGHBOR_COUNT and
        r["label"] == WITNESS_LABEL
    ]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one locked witness row, got {len(rows)}")
    row = rows[0]
    return {
        "condition_scope": WITNESS_SCOPE,
        "source_space": WITNESS_SOURCE_SPACE,
        "neighborhood": WITNESS_NEIGHBORHOOD,
        "neighborhood_kind": row["neighborhood_kind"],
        "epsilon_radius": au.as_float(row["neighborhood_value"]),
        "min_neighbor_count": WITNESS_MIN_NEIGHBOR_COUNT,
        "label": WITNESS_LABEL,
        "c49_hit": au.as_float(row["mean_local_bayes_top1_hit"]),
        "c49_coverage": au.as_float(row["mean_coverage"]),
        "c49_enrichment": au.as_float(row["mean_local_bayes_enrichment"]),
        "c49_mean_neighbor_count": au.as_float(row["mean_neighbor_count"]),
        "c49_covered_base_rate": au.as_float(row["mean_covered_base_rate"]),
        "inherited_from_c49_commit": C49_COMMIT,
    }


def compact_witness():
    witness = c49_locked_witness()
    return {
        "condition": WITNESS_SCOPE,
        "source_objectives": WITNESS_SOURCE_SPACE,
        "eps_quantile": WITNESS_EPS_QUANTILE,
        "min_n": WITNESS_MIN_NEIGHBOR_COUNT,
        "epsilon": witness["epsilon_radius"],
    }
