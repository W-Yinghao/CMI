"""Deterministic C34 margin-free taxonomy."""
from __future__ import annotations

import numpy as np

from . import schema


def _v(d, k, default=0.0):
    v = d.get(k)
    return default if v is None else v


def binary_vs_continuous_status(primary_pairs, robust_pairs) -> list:
    robust = {(r["seed"], r["target"], r["level"], r["regime"]): r for r in robust_pairs["per_unit"]}
    out = []
    for r in primary_pairs["per_unit"]:
        key = (r["seed"], r["target"], r["level"], r["regime"])
        rr = robust.get(key, {})
        if r.get("threshold_artifact"):
            status = "threshold_only"
        elif r.get("has_continuous_better") and (
                (r.get("selected_to_continuous_better_joint_min_delta") or 0.0) >=
                schema.STANDARDIZED_MEANINGFUL_REGRET or
                (r.get("selected_to_continuous_better_norm_regret_reduction") or 0.0) >=
                schema.STANDARDIZED_MEANINGFUL_REGRET):
            status = "real_endpoint_regret"
        elif r.get("has_continuous_better"):
            status = "continuous_tiny_or_weak"
        else:
            status = "no_near_continuous_better"
        out.append({
            "seed": r["seed"], "target": r["target"], "level": r["level"], "regime": r["regime"],
            "primary_selected_joint_good": r.get("selected_joint_good"),
            "robust_selected_joint_good": rr.get("selected_joint_good"),
            "has_binary_joint_neighbor": r.get("has_binary_joint_neighbor"),
            "has_continuous_better": r.get("has_continuous_better"),
            "continuous_regret_status": status,
            "threshold_artifact": r.get("threshold_artifact"),
            "primary_continuous_pair_case": r.get("continuous_pair_case"),
            "robust_continuous_pair_case": rr.get("continuous_pair_case"),
        })
    return out


def classify(endpoint_summary, selected_pairs, direction, components, gauge) -> dict:
    cases, evidence = [], {}
    ps = selected_pairs["summary"]
    ds = direction["summary"]
    cs = components["summary"]
    gs = gauge["summary"]

    m1 = bool((_v(ps, "threshold_only_fraction_among_binary_misses") >= 0.20) and
              (_v(ps, "real_endpoint_regret_fraction") < 0.70))
    evidence["M1"] = {"threshold_only_fraction_among_binary_misses":
                      ps.get("threshold_only_fraction_among_binary_misses"),
                      "real_endpoint_regret_fraction": ps.get("real_endpoint_regret_fraction")}
    if m1:
        cases.append(schema.M1)

    m2 = bool(_v(ps, "source_active_misranking_fraction") >= 0.25 or
              _v(cs, "source_score_wrong_direction_fraction") >= 0.25)
    evidence["M2"] = {"source_active_misranking_fraction": ps.get("source_active_misranking_fraction"),
                      "source_score_wrong_direction_fraction": cs.get("source_score_wrong_direction_fraction")}
    if m2:
        cases.append(schema.M2)

    m3 = bool(_v(cs, "leakage_mean_wrong_direction_fraction") >= 0.30)
    evidence["M3"] = {"leakage_mean_wrong_direction_fraction": cs.get("leakage_mean_wrong_direction_fraction")}
    if m3:
        cases.append(schema.M3)

    m4 = bool(_v(cs, "risk_mean_wrong_direction_fraction") >= 0.30)
    evidence["M4"] = {"risk_mean_wrong_direction_fraction": cs.get("risk_mean_wrong_direction_fraction")}
    if m4:
        cases.append(schema.M4)

    m5 = bool((_v(ds, "source_pairwise_auc", 0.5) >= 0.50) and
              (_v(ds, "source_flat_fraction") >= 0.30) and
              (_v(ds, "source_wrong_direction_fraction") < 0.30))
    evidence["M5"] = {"source_pairwise_auc": ds.get("source_pairwise_auc"),
                      "source_flat_fraction": ds.get("source_flat_fraction"),
                      "source_wrong_direction_fraction": ds.get("source_wrong_direction_fraction")}
    if m5:
        cases.append(schema.M5)

    m6 = bool(_v(gs, "meaningful_regret_gauge_unseen_fraction") >= 0.30)
    evidence["M6"] = {"meaningful_regret_gauge_unseen_fraction":
                      gs.get("meaningful_regret_gauge_unseen_fraction"),
                      "mean_joint_min_gain_with_gauge_jump": gs.get("mean_joint_min_gain_with_gauge_jump"),
                      "mean_joint_min_gain_without_gauge_jump": gs.get("mean_joint_min_gain_without_gauge_jump")}
    if m6:
        cases.append(schema.M6)

    tu_delta = gs.get("target_unlabeled_pm1_regret_delta_vs_source")
    m7 = bool(tu_delta is not None and tu_delta >= -0.005)
    evidence["M7"] = {"target_unlabeled_pm1_regret_delta_vs_source": tu_delta,
                      "source_pm1_mean_regret": gs.get("source_pm1_mean_regret"),
                      "target_unlabeled_pm1_mean_regret": gs.get("target_unlabeled_pm1_mean_regret")}
    if m7:
        cases.append(schema.M7)

    m8 = bool(_v(ps, "endpoint_tradeoff_fraction") >= 0.25)
    evidence["M8"] = {"endpoint_tradeoff_fraction": ps.get("endpoint_tradeoff_fraction")}
    if m8:
        cases.append(schema.M8)

    unexplained = bool((_v(ps, "real_endpoint_regret_fraction") >= 0.20) and
                       not any(c in cases for c in (schema.M2, schema.M3, schema.M4, schema.M6, schema.M8)))
    evidence["M9"] = {"real_endpoint_regret_fraction": ps.get("real_endpoint_regret_fraction"),
                      "explained_cases": [c for c in cases if c != schema.M1]}
    if unexplained:
        cases.append(schema.M9)

    if not cases:
        cases.append(schema.M9)
    return {"cases": cases, "evidence": evidence,
            "endpoint_summary_note": "Raw endpoint vectors are reported before scalar summaries."}


def case_rows(taxonomy) -> list:
    return [{"cases": ";".join(taxonomy["cases"])}]


def status_summary(rows) -> dict:
    counts = {}
    for r in rows:
        counts[r["continuous_regret_status"]] = counts.get(r["continuous_regret_status"], 0) + 1
    n = len(rows)
    return {"n_units": n, "status_counts": counts,
            "threshold_artifact_fraction": (
                float(np.mean([r["threshold_artifact"] for r in rows if r["threshold_artifact"] is not None]))
                if rows else None)}
