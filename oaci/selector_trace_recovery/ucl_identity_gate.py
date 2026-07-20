"""C37 P0 selected-UCL identity gate."""
from __future__ import annotations

from . import artifact_loader as al
from . import leakage_ucl_replay
from . import schema


def run_p0_identity(pairs, trace, *, n_jobs=1, recompute=True):
    ctx_cache = al.ContextCache(trace)
    rows = []
    for seed, target, level, regime, selected_order in al.p0_slice(pairs):
        ctx = ctx_cache.get(seed, target, level, regime)
        cand = trace["by_key"][(seed, target, level, regime, selected_order)]
        persisted = leakage_ucl_replay.persisted_selected_leakage(ctx)
        replay = None
        if recompute:
            replay = leakage_ucl_replay.replay_ucl(ctx, cand["model_hash"], n_jobs=n_jobs)
        row = {
            "unit_id": f"s{seed}_t{int(target):03d}_l{int(level):03d}",
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "selected_order": selected_order,
            "p0_recomputed": int(bool(replay)),
            "persisted_selected_point": persisted["extractable_LQ_ov"],
            "recomputed_selected_point": None if replay is None else replay["extractable_LQ_ov"],
            "point_abs_diff": None if replay is None else abs(replay["extractable_LQ_ov"] -
                                                              persisted["extractable_LQ_ov"]),
            "persisted_selected_ucl": persisted["bootstrap_ucl"],
            "recomputed_selected_ucl": None if replay is None else replay["bootstrap_ucl"],
            "ucl_abs_diff": None if replay is None else abs(replay["bootstrap_ucl"] - persisted["bootstrap_ucl"]),
            "fold_plan_hash_matches": int(replay is not None and replay["fold_plan_hash"] ==
                                          persisted["fold_plan_hash"]),
            "bootstrap_plan_hash_matches": int(replay is not None and replay["bootstrap_plan_hash"] ==
                                               persisted["bootstrap_plan_hash"]),
            "n_bootstrap_matches": int(replay is not None and replay["n_bootstrap"] ==
                                       persisted["n_bootstrap"]),
            "runtime_seconds": None if replay is None else replay["runtime_seconds"],
            "target_labels_loaded_for_replay": 0,
        }
        row["p0_identity_pass"] = int(
            bool(replay) and
            row["point_abs_diff"] <= schema.POINT_IDENTITY_TOL and
            row["ucl_abs_diff"] <= schema.UCL_IDENTITY_TOL and
            row["fold_plan_hash_matches"] and
            row["bootstrap_plan_hash_matches"] and
            row["n_bootstrap_matches"]
        )
        rows.append(row)
    summary = {"n_p0": len(rows), "n_pass": sum(r["p0_identity_pass"] for r in rows),
               "p0_pass": bool(rows and all(r["p0_identity_pass"] for r in rows))}
    return {"rows": rows, "summary": summary}

