"""C30 — FROZEN factor registry. Source-only factor families (risk / calibration / leakage / logit-geometry)
declared before analysis, plus the gauge/rank axis definitions. NOT feature selection: all family features are
evaluated; nothing is selected."""
from __future__ import annotations

from . import schema


def families() -> dict:
    return {k: list(v) for k, v in schema.SOURCE_FAMILIES.items()}


def all_source_features() -> list:
    return [f for feats in schema.SOURCE_FAMILIES.values() for f in feats]


def feature_family_rows() -> list:
    return [{"feature": f, "family": fam} for fam, feats in schema.SOURCE_FAMILIES.items() for f in feats] + [
        {"feature": schema.SCORE_KEY, "family": "frozen_probe_competence_score"},
        {"feature": "per_target_mean(score)", "family": "target_gauge_intercept"}]
