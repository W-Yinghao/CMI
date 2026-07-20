"""C44 Pareto-front null audit."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import objective_registry, schema


def oriented_matrix(rows, specs):
    return np.asarray([[objective_registry.oriented(r, s) for s in specs] for r in rows], dtype=float)


def dominance_stats(mat):
    n = mat.shape[0]
    ge = mat[None, :, :] >= mat[:, None, :] - 1e-12
    gt = mat[None, :, :] > mat[:, None, :] + 1e-12
    dominates = ge.all(axis=2) & gt.any(axis=2)
    np.fill_diagonal(dominates, False)
    dominators = dominates.sum(axis=1).astype(int)
    dominated = dominates.sum(axis=0).astype(int)
    front = dominators == 0
    return front, dominators, dominated


def pareto_layers(mat):
    remaining = list(range(mat.shape[0]))
    layers = np.zeros(mat.shape[0], dtype=int)
    depth = 0
    while remaining:
        sub = mat[remaining]
        front, _, _ = dominance_stats(sub)
        front_idx = [remaining[i] for i, flag in enumerate(front) if flag]
        if not front_idx:
            break
        for idx in front_idx:
            layers[idx] = depth
        remaining = [idx for idx in remaining if idx not in set(front_idx)]
        depth += 1
    return layers


def _front_fraction_by_traj(ctx, specs, null_type, rep=0, *, compute_depth=False):
    null_offsets = {
        "observed": 0,
        "objective_shuffled": 11,
        "gaussian_iid_same_dimension": 23,
        "family_shuffled": 37,
        "sign_flipped_orientation": 41,
    }
    rng = np.random.RandomState(schema.NULL_SEED + rep * 101 + null_offsets[null_type])
    vals = []
    depth_vals = []
    family_to_cols = {}
    for j, s in enumerate(specs):
        family_to_cols.setdefault(s["family"], []).append(j)
    for _, rows in sorted(ctx["by_traj"].items()):
        mat = oriented_matrix(rows, specs)
        if null_type == "observed":
            nm = mat
        elif null_type == "objective_shuffled":
            nm = mat.copy()
            for j in range(nm.shape[1]):
                rng.shuffle(nm[:, j])
        elif null_type == "gaussian_iid_same_dimension":
            nm = rng.normal(size=mat.shape)
        elif null_type == "family_shuffled":
            nm = mat.copy()
            for cols in family_to_cols.values():
                perm = rng.permutation(nm.shape[0])
                nm[:, cols] = nm[perm][:, cols]
        elif null_type == "sign_flipped_orientation":
            signs = np.asarray([1 if ((j + rep) % 2 == 0) else -1 for j in range(mat.shape[1])])
            nm = mat * signs
        else:
            raise ValueError(null_type)
        front, _, _ = dominance_stats(nm)
        vals.append(float(front.mean()))
        if compute_depth:
            layers = pareto_layers(nm)
            depth_vals.append(float(np.mean(layers)))
    return float(np.mean(vals)), (float(np.mean(depth_vals)) if depth_vals else None)


def audit(ctx):
    specs = objective_registry.source_pareto_specs(ctx)
    null_types = ("observed", "objective_shuffled", "gaussian_iid_same_dimension", "family_shuffled",
                  "sign_flipped_orientation")
    observed_front, observed_depth = _front_fraction_by_traj(ctx, specs, "observed", 0, compute_depth=True)
    rows = []
    for nt in null_types:
        reps = 1 if nt == "observed" else schema.NULL_REPS
        front_vals = []
        depth_vals = []
        for rep in range(reps):
            ff, dd = _front_fraction_by_traj(ctx, specs, nt, rep, compute_depth=(nt == "observed"))
            front_vals.append(ff)
            if dd is not None:
                depth_vals.append(dd)
        rows.append({
            "null_type": nt,
            "n_reps": reps,
            "n_objectives": len(specs),
            "mean_front_fraction": al.finite_mean(front_vals),
            "median_front_fraction": al.finite_median(front_vals),
            "mean_layer_depth": al.finite_mean(depth_vals),
            "observed_front_fraction": observed_front,
            "observed_minus_null_front_fraction": observed_front - al.finite_mean(front_vals),
            "observed_mean_layer_depth": observed_depth,
            "pareto_nulls_frozen": 1,
        })
    summary = {
        "observed_front_fraction": observed_front,
        "observed_mean_layer_depth": observed_depth,
        "n_objectives": len(specs),
        "gaussian_null_front_fraction": next(r for r in rows if r["null_type"] == "gaussian_iid_same_dimension")[
            "mean_front_fraction"],
        "objective_shuffled_front_fraction": next(r for r in rows if r["null_type"] == "objective_shuffled")[
            "mean_front_fraction"],
        "family_shuffled_front_fraction": next(r for r in rows if r["null_type"] == "family_shuffled")[
            "mean_front_fraction"],
    }
    return {"rows": rows, "summary": summary}
