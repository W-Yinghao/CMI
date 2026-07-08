"""Exact point-leakage atom replay for C39."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..leakage.critic import DomainProbe
from ..leakage.estimate import reference_conditional_entropy
from . import schema


def _fold_vector(feat, fold_plan):
    return np.array([fold_plan.fold_of_group[str(g)] for g in feat.group], dtype=int)


def _cell_nll_stats(feat, support_graph, fold_plan, capacity, cfg):
    fold = _fold_vector(feat, fold_plan)
    b_all = feat.sample_mass
    out = {}
    for y in support_graph.comparable_classes:
        S = [int(d) for d in support_graph.support_of_class[y]]
        dmap = {int(d): i for i, d in enumerate(S)}
        sel = (feat.y == int(y)) & np.isin(feat.d, S)
        idx = np.where(sel)[0]
        if idx.size == 0:
            raise ValueError(f"comparable class {y} has no rows")
        Z = feat.Z[idx]
        labels = np.array([dmap[int(d)] for d in feat.d[idx]], dtype=int)
        b = b_all[idx]
        f = fold[idx]
        raw_d = feat.d[idx]
        num_by_d = defaultdict(float)
        mass_by_d = defaultdict(float)
        rows_by_d = defaultdict(int)
        scored_folds = 0
        for k in range(fold_plan.n_folds):
            te = f == k
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            probe = DomainProbe(capacity, len(S), cfg).fit(Z[tr], labels[tr], sample_weight=b[tr])
            nll = probe.nll(Z[te], labels[te])
            d_te = raw_d[te]
            b_te = b[te]
            scored_folds += 1
            for d in S:
                m = d_te == int(d)
                if m.any():
                    num_by_d[int(d)] += float((b_te[m] * nll[m]).sum())
                    mass_by_d[int(d)] += float(b_te[m].sum())
                    rows_by_d[int(d)] += int(m.sum())
        out[int(y)] = {
            int(d): {
                "weighted_nll_num": float(num_by_d[int(d)]),
                "oof_mass": float(mass_by_d[int(d)]),
                "n_rows": int(rows_by_d[int(d)]),
                "scored_folds": int(scored_folds),
            }
            for d in S
        }
    return out


def _point_from_cells(cell_stats, support_graph):
    H = reference_conditional_entropy(support_graph)
    p_ref = support_graph.reference_prior
    l_y = {}
    class_nll = {}
    class_mass_diff = {}
    atoms = []
    for y in support_graph.comparable_classes:
        S = [int(d) for d in support_graph.support_of_class[y]]
        M_y = float(support_graph.cell_mass[S, int(y)].sum())
        if M_y <= 0:
            raise ValueError(f"class {y} has non-positive overlap mass")
        num_y = float(sum(cell_stats[int(y)][int(d)]["weighted_nll_num"] for d in S))
        mass_y = float(sum(cell_stats[int(y)][int(d)]["oof_mass"] for d in S))
        nll_y = num_y / M_y
        class_nll[int(y)] = nll_y
        class_mass_diff[int(y)] = abs(mass_y - M_y)
        l_y[int(y)] = float(H[int(y)] - nll_y)
        for d in S:
            d = int(d)
            cell_mass = float(support_graph.cell_mass[d, int(y)])
            p_d_y = cell_mass / M_y if M_y > 0 else 0.0
            entropy_cell = float(-p_d_y * np.log(p_d_y)) if p_d_y > 0 else 0.0
            nll_cell = float(cell_stats[int(y)][d]["weighted_nll_num"] / M_y)
            atom = float(p_ref[int(y)] * (entropy_cell - nll_cell))
            atoms.append({
                "atom_id": f"y{int(y)}:d{d}",
                "class_id": int(y),
                "class_name": support_graph.class_names[int(y)],
                "domain_id": d,
                "domain_name": support_graph.domain_names[d],
                "support_count": int(support_graph.counts[d, int(y)]),
                "support_m": int(support_graph.m),
                "cell_mass": cell_mass,
                "class_overlap_mass": M_y,
                "p_ref_y": float(p_ref[int(y)]),
                "p_d_given_y": p_d_y,
                "entropy_cell_weighted": float(p_ref[int(y)] * entropy_cell),
                "nll_cell_weighted": float(p_ref[int(y)] * nll_cell),
                "atom_value": atom,
                "oof_mass": float(cell_stats[int(y)][d]["oof_mass"]),
                "n_rows": int(cell_stats[int(y)][d]["n_rows"]),
                "scored_folds": int(cell_stats[int(y)][d]["scored_folds"]),
                "eligible": int(bool(support_graph.eligible[d, int(y)])),
                "present": int(bool(support_graph.present[d, int(y)])),
                "support_edge": int(int(support_graph.counts[d, int(y)]) <= int(support_graph.m)),
            })
    L_abs = float(sum(float(p_ref[int(y)]) * l_y[int(y)] for y in support_graph.comparable_classes))
    return {
        "L_abs": L_abs,
        "atoms": atoms,
        "L_y": l_y,
        "nll_by_class": class_nll,
        "reference_entropy": H,
        "max_class_mass_abs_diff": max(class_mass_diff.values()) if class_mass_diff else 0.0,
    }


def replay_point_atoms(feat, support_graph, fold_plan, cfg, *, expected_point=None):
    """Replay exact point leakage and return an additive domain x class atom decomposition.

    The selected capacity is the same aggregate ``max_c L_abs,c`` rule used by
    :func:`estimate_extractable_leakage`; no per-class capacity argmax is introduced.
    """
    by_capacity = {}
    for c in cfg.capacities:
        stats = _cell_nll_stats(feat, support_graph, fold_plan, int(c), cfg)
        by_capacity[int(c)] = _point_from_cells(stats, support_graph)
    selected_capacity = max([int(c) for c in cfg.capacities], key=lambda c: by_capacity[c]["L_abs"])
    selected = by_capacity[selected_capacity]
    atom_sum = float(sum(r["atom_value"] for r in selected["atoms"]))
    point_abs_diff = (abs(float(expected_point) - selected["L_abs"])
                      if expected_point is not None and np.isfinite(float(expected_point)) else np.nan)
    additive_abs_diff = abs(atom_sum - selected["L_abs"])
    identity_pass = (
        additive_abs_diff <= schema.ATOM_ADDITIVE_TOL and
        (not np.isfinite(point_abs_diff) or point_abs_diff <= schema.POINT_IDENTITY_TOL)
    )
    return {
        "aggregate_point": float(selected["L_abs"]),
        "selected_capacity": int(selected_capacity),
        "point_by_capacity": {int(c): float(v["L_abs"]) for c, v in by_capacity.items()},
        "atom_sum": atom_sum,
        "point_abs_diff": float(point_abs_diff) if np.isfinite(point_abs_diff) else np.nan,
        "additive_abs_diff": float(additive_abs_diff),
        "max_class_mass_abs_diff": float(selected["max_class_mass_abs_diff"]),
        "identity_pass": int(bool(identity_pass)),
        "atoms": selected["atoms"],
        "L_y": selected["L_y"],
        "nll_by_class": selected["nll_by_class"],
        "reference_entropy": selected["reference_entropy"],
        "target_labels_loaded_for_replay": 0,
    }
