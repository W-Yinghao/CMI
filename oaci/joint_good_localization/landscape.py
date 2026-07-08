"""C32 joint-good landscape and trajectory grouping utilities."""
from __future__ import annotations

import numpy as np


def by_trajectory(rows) -> dict:
    out = {}
    for r in rows:
        out.setdefault((r["seed"], r["target"], r["level"], r.get("regime", "")), []).append(r)
    return out


def joint_good_landscape(rows) -> dict:
    by = by_trajectory(rows)
    counts, rates, sizes, selected_counts = [], [], [], []
    for cs in by.values():
        n = len(cs)
        j = sum(int(c["joint_good"]) for c in cs)
        counts.append(j)
        rates.append(j / n if n else 0.0)
        sizes.append(n)
        selected_counts.append(sum(int(c.get("selected_oaci", 0)) for c in cs))
    vals = [int(r["joint_good"]) for r in rows]
    return {
        "n_candidates": len(rows),
        "n_trajectories": len(by),
        "joint_good_count": int(sum(vals)),
        "joint_good_rate": float(np.mean(vals)) if vals else None,
        "trajectory_any_joint_fraction": float(np.mean([c > 0 for c in counts])) if counts else None,
        "mean_joint_good_per_trajectory": float(np.mean(counts)) if counts else None,
        "median_joint_good_per_trajectory": float(np.median(counts)) if counts else None,
        "min_joint_good_per_trajectory": int(min(counts)) if counts else None,
        "max_joint_good_per_trajectory": int(max(counts)) if counts else None,
        "mean_joint_good_rate_per_trajectory": float(np.mean(rates)) if rates else None,
        "median_joint_good_rate_per_trajectory": float(np.median(rates)) if rates else None,
        "mean_candidates_per_trajectory": float(np.mean(sizes)) if sizes else None,
        "selected_oaci_trajectories": int(sum(c == 1 for c in selected_counts)),
        "selected_oaci_missing_or_duplicate": int(sum(c != 1 for c in selected_counts)),
    }
