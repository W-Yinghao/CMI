"""Exact leakage-UCL replay from Phase-A source-train feature stores."""
from __future__ import annotations

import time

from ..runner.scoring import compute_leakage_score
from . import artifact_loader as al


def replay_ucl(ctx, model_hash, *, n_jobs=1):
    """Replay exact selection bootstrap scoring. The feature is served from the Phase-A store."""
    if int(n_jobs) > 1:
        from ..leakage.parallel import set_leakage_parallel
        set_leakage_parallel(int(n_jobs), "process")
    feat = ctx.feature_by_hash(model_hash)
    t0 = time.time()
    res = compute_leakage_score(feat.features, ctx.support_graph, ctx.fold_plan, ctx.bootstrap_plan,
                                ctx.fold.execution_config.critic)
    return {
        "extractable_LQ_ov": float(res["extractable_LQ_ov"]),
        "bootstrap_ucl": float(res["bootstrap_ucl"]),
        "percentile_ucl": float(res["percentile_ucl"]),
        "n_bootstrap": int(res["n_bootstrap"]),
        "bootstrap_plan_hash": res["bootstrap_plan_hash"],
        "fold_plan_hash": res["fold_plan_hash"],
        "candidate_draw_count": int(res["candidate_draw_count"]),
        "invalid_draw_rate": float(res["invalid_draw_rate"]),
        "runtime_seconds": float(time.time() - t0),
    }


def persisted_selected_leakage(ctx):
    path = (f"{ctx.artifact_dir}/levels/level-{ctx.level:03d}/methods/OACI/selection_leakage.json")
    b = al.body(path)
    return {
        "extractable_LQ_ov": al.as_float(b.get("extractable_LQ_ov")),
        "bootstrap_ucl": al.as_float(b.get("bootstrap_ucl")),
        "percentile_ucl": al.as_float(b.get("percentile_ucl")),
        "n_bootstrap": al.as_int(b.get("n_bootstrap")),
        "bootstrap_plan_hash": b.get("bootstrap_plan_hash"),
        "fold_plan_hash": b.get("fold_plan_hash"),
        "candidate_draw_count": al.as_int(b.get("candidate_draw_count")),
        "invalid_draw_rate": al.as_float(b.get("invalid_draw_rate")),
    }

