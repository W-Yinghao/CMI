"""Empirical ambiguity lower-bound diagnostics for C45."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import source_space


def _ambiguity_for_rows(rows):
    labels = [int(r["primary_joint_good"]) for r in rows]
    if not labels:
        return 0.0, 0
    good = sum(labels)
    bad = len(labels) - good
    return min(good, bad) / len(labels), int(good > 0 and bad > 0)


def audit(ctx, space=None, radii=None):
    space = space or source_space.build_space(ctx)
    radii = radii or source_space.epsilon_radii(ctx, space)
    out = []
    for q, radius in radii.items():
        ambiguous = []
        lower = []
        discord_pairs = []
        all_pairs = []
        for _, rows in sorted(ctx["by_traj"].items()):
            idx = [int(r["source_idx"]) for r in rows]
            mat = space["z"][idx, :]
            dist = np.sqrt(((mat[:, None, :] - mat[None, :, :]) ** 2).sum(axis=2))
            for i in range(len(rows)):
                nidx = [j for j in range(len(rows)) if dist[i, j] <= radius]
                lb, amb = _ambiguity_for_rows([rows[j] for j in nidx])
                lower.append(lb)
                ambiguous.append(amb)
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    disagree = int(int(rows[i]["primary_joint_good"]) != int(rows[j]["primary_joint_good"]))
                    all_pairs.append(disagree)
                    if dist[i, j] <= radius:
                        discord_pairs.append(disagree)
        baseline = al.finite_mean(all_pairs)
        discord = al.finite_mean(discord_pairs)
        out.append({
            "epsilon_quantile": q,
            "epsilon_radius": radius,
            "ambiguous_neighborhood_fraction": al.finite_mean(ambiguous),
            "discordant_pair_fraction": discord,
            "trajectory_conditioned_pair_baseline": baseline,
            "discordant_pair_fraction_over_baseline": discord / baseline if baseline else None,
            "minimum_unavoidable_ambiguity_rate": al.finite_mean(lower),
            "target_good_non_good_cohabitation_rate": al.finite_mean(ambiguous),
            "n_source_equivalent_pairs": len(discord_pairs),
            "target_labels_diagnostic_only": 1,
        })
    return {"rows": out, "summary": {f"q{int(r['epsilon_quantile'] * 100):02d}": r for r in out}}
