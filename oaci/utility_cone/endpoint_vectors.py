"""Endpoint vector registry and frozen normalization views for C35."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def _f(r, key):
    return artifact_loader._as_float(r.get(key))


def _vec_from_prefix(r, prefix):
    return np.array([_f(r, f"{prefix}_target_bacc_delta"),
                     _f(r, f"{prefix}_target_nll_delta"),
                     _f(r, f"{prefix}_target_ece_delta")], dtype=float)


def _pair_key(r):
    return (r["seed"], r["target"], r["level"], r.get("regime", ""), r["comparison"],
            r.get("selected_order"), r.get("candidate_order"))


def _unit_key(r):
    return (r["seed"], r["target"], r["level"], r.get("regime", ""))


def _z_stats(vals):
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return 0.0, 1.0
    sd = float(vals.std())
    return float(vals.mean()), sd if sd > 1e-12 else 1.0


def _within_stats(endpoint_rows):
    by = {}
    for r in endpoint_rows:
        by.setdefault(_unit_key(r), []).append(r)
    stats = {}
    ranks = {}
    for key, rows in by.items():
        for e in schema.ENDPOINT_KEYS:
            vals = [_f(r, e) for r in rows]
            stats[(key, e)] = _z_stats(vals)
            order = sorted([(i, _f(r, e)) for i, r in enumerate(rows)], key=lambda x: x[1])
            denom = max(len(order) - 1, 1)
            for rank, (i, _) in enumerate(order):
                ranks[(key, rows[i]["order"], e)] = rank / denom
    return stats, ranks


def build_endpoint_vectors(tables):
    endpoint_rows = tables["endpoint_utility_registry.csv"]
    endpoint_map = artifact_loader.endpoint_registry_map(endpoint_rows)
    within, ranks = _within_stats(endpoint_rows)
    rows = []
    for p in artifact_loader.all_pairs(tables):
        key = _unit_key(p)
        sel = endpoint_map[(p["seed"], p["target"], p["level"], p.get("regime", ""), p["selected_order"])]
        cand = endpoint_map[(p["seed"], p["target"], p["level"], p.get("regime", ""), p["candidate_order"])]
        selected_raw = _vec_from_prefix(p, "selected")
        candidate_raw = _vec_from_prefix(p, "candidate")
        raw_delta = candidate_raw - selected_raw
        selected_global_z = np.array([_f(sel, "target_bacc_z"), _f(sel, "target_nll_z"), _f(sel, "target_ece_z")])
        candidate_global_z = np.array([_f(cand, "target_bacc_z"), _f(cand, "target_nll_z"), _f(cand, "target_ece_z")])
        selected_within, candidate_within, selected_rank, candidate_rank = [], [], [], []
        for e in schema.ENDPOINT_KEYS:
            mu, sd = within[(key, e)]
            selected_within.append((_f(sel, e) - mu) / sd)
            candidate_within.append((_f(cand, e) - mu) / sd)
            selected_rank.append(ranks[(key, sel["order"], e)])
            candidate_rank.append(ranks[(key, cand["order"], e)])
        selected_within = np.array(selected_within)
        candidate_within = np.array(candidate_within)
        selected_rank = np.array(selected_rank)
        candidate_rank = np.array(candidate_rank)
        rows.append({
            "pair_id": "|".join(map(str, _pair_key(p))),
            "seed": p["seed"], "target": p["target"], "level": p["level"], "regime": p.get("regime", ""),
            "comparison": p["comparison"], "selected_order": p["selected_order"], "candidate_order": p["candidate_order"],
            "raw_selected_bacc": selected_raw[0], "raw_selected_nll_improve": selected_raw[1],
            "raw_selected_ece_improve": selected_raw[2], "raw_candidate_bacc": candidate_raw[0],
            "raw_candidate_nll_improve": candidate_raw[1], "raw_candidate_ece_improve": candidate_raw[2],
            "raw_delta_bacc": raw_delta[0], "raw_delta_nll_improve": raw_delta[1],
            "raw_delta_ece_improve": raw_delta[2],
            "global_z_delta_bacc": (candidate_global_z - selected_global_z)[0],
            "global_z_delta_nll_improve": (candidate_global_z - selected_global_z)[1],
            "global_z_delta_ece_improve": (candidate_global_z - selected_global_z)[2],
            "within_z_delta_bacc": (candidate_within - selected_within)[0],
            "within_z_delta_nll_improve": (candidate_within - selected_within)[1],
            "within_z_delta_ece_improve": (candidate_within - selected_within)[2],
            "rank_delta_bacc": (candidate_rank - selected_rank)[0],
            "rank_delta_nll_improve": (candidate_rank - selected_rank)[1],
            "rank_delta_ece_improve": (candidate_rank - selected_rank)[2],
        })
    return rows


def vector_for(row, scaling="raw"):
    if scaling == "raw":
        keys = ("raw_delta_bacc", "raw_delta_nll_improve", "raw_delta_ece_improve")
    elif scaling == "global_z":
        keys = ("global_z_delta_bacc", "global_z_delta_nll_improve", "global_z_delta_ece_improve")
    elif scaling == "within_z":
        keys = ("within_z_delta_bacc", "within_z_delta_nll_improve", "within_z_delta_ece_improve")
    elif scaling == "rank":
        keys = ("rank_delta_bacc", "rank_delta_nll_improve", "rank_delta_ece_improve")
    else:
        raise ValueError(scaling)
    return np.array([float(row[k]) for k in keys], dtype=float)
