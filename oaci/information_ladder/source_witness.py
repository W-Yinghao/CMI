"""C24-A — source-only NON-IDENTIFIABILITY witnesses. Upgrades the C23 negative from 'the ridge did not learn'
to 'the source-only representation itself lacks offset identifiability'. Two complementary read-outs over
(target, regime) units:

  (1) Mantel-style test: does SOURCE-summary distance PREDICT the true score-OFFSET distance across unit pairs?
      A weak / insignificant correlation => source summaries are NON-IDENTIFYING for the offset.
  (2) Concrete witnesses: pairs whose source summaries are near-identical yet whose offsets diverge -- existence
      proof that source summaries COLLAPSE together units requiring different offsets.

Read-only; uses no target inputs / labels. A permutation null (shuffle the offset<->unit pairing) calibrates
whether source distance predicts offset distance ABOVE chance."""
from __future__ import annotations

import numpy as np

from ..score_gauge import gauge_feature_registry as gfr
from . import schema


def _units(rows, mode="in_regime"):
    names = gfr.gauge_feature_names()
    groups = {}
    for r in rows:
        if r["mode"] == mode:
            groups.setdefault((r["target"], r["regime"]), []).append(r)
    units = {}
    for u, cands in groups.items():
        v = gfr._vector(cands)
        units[u] = {"source_vec": np.array([v[n] for n in names], float),
                    "offset": float(np.mean([c["score"] for c in cands])), "n": len(cands)}
    return units


def witness_audit(rows, mode="in_regime") -> dict:
    units = _units(rows, mode)
    if len(units) < 4:
        return {"n_units": len(units), "insufficient_units": True, "source_nonidentifying": None}
    keys = list(units)
    X = np.stack([units[k]["source_vec"] for k in keys])
    off = np.array([units[k]["offset"] for k in keys])
    tgt = np.array([k[0] for k in keys])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    n = len(keys)
    ii, jj = np.triu_indices(n, k=1)
    sdist = np.sqrt(((Z[ii] - Z[jj]) ** 2).sum(1))
    odist = np.abs(off[ii] - off[jj])
    cross = tgt[ii] != tgt[jj]                               # CROSS-TARGET pairs only (removes same-target block confound)

    def _corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _mantel(mask):
        s, d = sdist[mask], odist[mask]
        obs = _corr(s, d)
        # permute the offset<->unit assignment, recompute the offset distances on the SAME mask
        rng = np.random.RandomState(schema.PERM_SEED); null = np.empty(schema.N_PERM)
        for p in range(schema.N_PERM):
            po = off[rng.permutation(n)]
            null[p] = _corr(s, np.abs(po[ii] - po[jj])[mask])
        return obs, float((np.sum(null >= obs) + 1) / (schema.N_PERM + 1))

    mantel, mantel_p = _mantel(np.ones(len(ii), bool))
    mantel_cross, mantel_cross_p = _mantel(cross)            # the HONEST identifiability read (cross-target)
    # source identifies the offset only if the CROSS-TARGET association is real (not a within-target block artifact)
    source_predicts_offset = bool(mantel_cross_p < 0.05 and mantel_cross >= schema.MANTEL_IDENTIFY_CORR)
    # (2) concrete witnesses: near source, far offset
    near_thr = float(np.quantile(sdist, schema.WITNESS_NEAR_QUANTILE))
    far_thr = float(np.quantile(odist, schema.WITNESS_FAR_OFFSET_QUANTILE))
    wmask = (sdist <= near_thr) & (odist >= far_thr)
    strength = odist / (sdist + 1e-9)
    order = np.argsort(-strength[wmask])
    wi = ii[wmask][order][:schema.WITNESS_TOP_K]; wj = jj[wmask][order][:schema.WITNESS_TOP_K]
    top = [{"unit_a": f"t{keys[a][0]}:{keys[a][1]}", "unit_b": f"t{keys[b][0]}:{keys[b][1]}",
            "source_distance": round(float(np.sqrt(((Z[a] - Z[b]) ** 2).sum())), 4),
            "offset_difference": round(float(abs(off[a] - off[b])), 4),
            "witness_strength": round(float(abs(off[a] - off[b]) / (np.sqrt(((Z[a] - Z[b]) ** 2).sum()) + 1e-9)), 4)}
           for a, b in zip(wi, wj)]
    source_nonidentifying = bool(not source_predicts_offset and int(wmask.sum()) > 0)
    within_block_confound = bool(mantel >= schema.MANTEL_IDENTIFY_CORR and mantel_cross < schema.MANTEL_IDENTIFY_CORR)
    return {"n_units": n, "n_pairs": int(len(ii)), "n_cross_target_pairs": int(cross.sum()),
            "mantel_corr_all_pairs": mantel, "mantel_perm_p_all_pairs": mantel_p,
            "mantel_corr_cross_target": mantel_cross, "mantel_perm_p_cross_target": mantel_cross_p,
            "mantel_corr_source_offset": mantel_cross,      # headline = cross-target (block-confound removed)
            "mantel_perm_p": mantel_cross_p,
            "source_predicts_offset": source_predicts_offset, "source_nonidentifying": source_nonidentifying,
            "within_target_block_confound_detected": within_block_confound,
            "near_distance_threshold": near_thr, "far_offset_threshold": far_thr,
            "n_strong_witnesses": int(wmask.sum()), "top_witnesses": top,
            "interpretation": (("CROSS-TARGET source-summary distance predicts offset distance (Mantel %.3f, p %.3f; "
                                "all-pairs %.3f) -> source is partially identifying" % (mantel_cross, mantel_cross_p, mantel))
                               if source_predicts_offset else
                               ("CROSS-TARGET source distance does NOT predict offset distance (Mantel %.3f, p %.3f)%s; "
                                "%d near-source/divergent-offset collisions -> source is non-identifying for the offset"
                                % (mantel_cross, mantel_cross_p,
                                   (" — all-pairs %.3f was a within-target block artifact" % mantel) if within_block_confound else "",
                                   int(wmask.sum()))))}
