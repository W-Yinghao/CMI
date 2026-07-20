"""C37 exact UCL recovery for C35 preference-robust better candidates."""
from __future__ import annotations

from . import artifact_loader as al
from . import leakage_ucl_replay


def recover_better_ucls(pairs, trace, *, n_jobs=1, p0_pass=True):
    ctx_cache = al.ContextCache(trace)
    rows = []
    for p in al.unique_pair_keys(pairs):
        b = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])]
        row = {
            "pair_key": "|".join([p["seed"], p["target"], p["level"], p["selected_order"], p["candidate_order"]]),
            "seed": p["seed"],
            "target": p["target"],
            "level": p["level"],
            "selected_order": p["selected_order"],
            "better_order": p["candidate_order"],
            "better_candidate_id": b["candidate_id"],
            "p0_pass_required": int(bool(p0_pass)),
            "better_ucl_recovered": 0,
            "recovery_status": "blocked_p0_failed" if not p0_pass else "pending",
        }
        if p0_pass:
            ctx = ctx_cache.get(p["seed"], p["target"], p["level"], p["regime"])
            replay = leakage_ucl_replay.replay_ucl(ctx, b["model_hash"], n_jobs=n_jobs)
            row.update({
                "better_point": replay["extractable_LQ_ov"],
                "better_ucl": replay["bootstrap_ucl"],
                "better_percentile_ucl": replay["percentile_ucl"],
                "better_n_bootstrap": replay["n_bootstrap"],
                "better_candidate_draw_count": replay["candidate_draw_count"],
                "better_invalid_draw_rate": replay["invalid_draw_rate"],
                "runtime_seconds": replay["runtime_seconds"],
                "better_ucl_recovered": 1,
                "recovery_status": "exact_replay_from_phase_a_store",
                "target_labels_loaded_for_replay": 0,
            })
        rows.append(row)
    summary = {"n_unique_better": len(rows), "n_recovered": sum(r["better_ucl_recovered"] for r in rows),
               "all_recovered": bool(rows and all(r["better_ucl_recovered"] for r in rows))}
    return {"rows": rows, "summary": summary}

