"""C42 source-rank gauge sensitivity audit."""
from __future__ import annotations

from collections import defaultdict

from . import artifact_loader as al


def _top1_metrics(ctx, values_by_key, *, non_source_only=False, diagnostic_ceiling=False):
    chosen = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        vals = [(values_by_key[(r["trajectory_id"], r["candidate_order"])], r) for r in rs]
        vals.sort(key=lambda x: (x[0], -int(x[1]["candidate_order"])), reverse=True)
        c = vals[0][1]
        oracle = max(float(r["target_utility_score"]) for r in rs)
        chosen.append((c, oracle))
    return {
        "top1_joint_good_rate": al.finite_mean([c["primary_joint_good"] for c, _ in chosen]),
        "top1_pareto_good_rate": al.finite_mean([c["pareto_good"] for c, _ in chosen]),
        "top1_regret_vs_target_oracle": al.finite_mean(
            [oracle - float(c["target_utility_score"]) for c, oracle in chosen]),
        "non_source_only": int(non_source_only),
        "diagnostic_ceiling": int(diagnostic_ceiling),
    }


def audit(ctx):
    raw = {}
    for r in ctx["registry"]:
        raw[(r["trajectory_id"], r["candidate_order"])] = float(r["source_rank_score"])
    regime_mean = defaultdict(list)
    target_mean = defaultdict(list)
    target_regime_mean = defaultdict(list)
    for r in ctx["registry"]:
        regime_mean[r["regime"]].append(float(r["source_rank_score"]))
        target_mean[r["target"]].append(float(r["source_rank_score"]))
        target_regime_mean[(r["target"], r["regime"])].append(float(r["source_rank_score"]))
    regime_mu = {k: sum(v) / len(v) for k, v in regime_mean.items()}
    target_mu = {k: sum(v) / len(v) for k, v in target_mean.items()}
    target_regime_mu = {k: sum(v) / len(v) for k, v in target_regime_mean.items()}
    variants = {
        "raw_source_rank": raw,
        "regime_centered_source_rank": {
            (r["trajectory_id"], r["candidate_order"]): float(r["source_rank_score"]) - regime_mu[r["regime"]]
            for r in ctx["registry"]},
        "target_centered_diagnostic_source_rank": {
            (r["trajectory_id"], r["candidate_order"]): float(r["source_rank_score"]) - target_mu[r["target"]]
            for r in ctx["registry"]},
        "target_regime_centered_diagnostic_source_rank": {
            (r["trajectory_id"], r["candidate_order"]): (
                float(r["source_rank_score"]) - target_regime_mu[(r["target"], r["regime"])])
            for r in ctx["registry"]},
        "target_rank_oracle_diagnostic_ceiling": {
            (r["trajectory_id"], r["candidate_order"]): float(r["target_utility_score"])
            for r in ctx["registry"]},
    }
    rows = []
    raw_metrics = None
    for name, values in variants.items():
        m = _top1_metrics(ctx, values, non_source_only=name.startswith("target_"),
                          diagnostic_ceiling=name == "target_rank_oracle_diagnostic_ceiling")
        if name == "raw_source_rank":
            raw_metrics = m
        rows.append({
            "normalization": name,
            **m,
            "top1_joint_good_gain_vs_raw": (
                "" if raw_metrics is None else m["top1_joint_good_rate"] - raw_metrics["top1_joint_good_rate"]),
            "top1_regret_gain_vs_raw": (
                "" if raw_metrics is None else raw_metrics["top1_regret_vs_target_oracle"] -
                m["top1_regret_vs_target_oracle"]),
        })
    source_centered = [r for r in rows if r["normalization"] in (
        "regime_centered_source_rank", "target_centered_diagnostic_source_rank",
        "target_regime_centered_diagnostic_source_rank")]
    max_center_gain = max(float(r["top1_joint_good_gain_vs_raw"]) for r in source_centered)
    summary = {
        "raw_top1_joint_good_rate": raw_metrics["top1_joint_good_rate"],
        "max_centered_top1_joint_good_gain_vs_raw": max_center_gain,
        "target_rank_oracle_top1_joint_good_rate": next(
            r for r in rows if r["normalization"] == "target_rank_oracle_diagnostic_ceiling")["top1_joint_good_rate"],
        "gauge_breaks_source_rank_actionability": False,
    }
    return {"rows": rows, "summary": summary}
