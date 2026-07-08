"""C43 multiplicity and stability audit."""
from __future__ import annotations

from collections import defaultdict

from . import artifact_loader as al, schema


def poisson_binomial_upper_tail(ps, k):
    pmf = [1.0]
    for p in ps:
        nxt = [0.0] * (len(pmf) + 1)
        for i, val in enumerate(pmf):
            nxt[i] += val * (1.0 - p)
            nxt[i + 1] += val * p
        pmf = nxt
    return sum(pmf[k:])


def _holm(rows):
    ordered = sorted(rows, key=lambda r: r["p_value_vs_trajectory_random"])
    m = len(ordered)
    running = 0.0
    for i, r in enumerate(ordered):
        adj = min(1.0, (m - i) * r["p_value_vs_trajectory_random"])
        running = max(running, adj)
        r["holm_p_value"] = running


def _bh(rows):
    ordered = sorted(rows, key=lambda r: r["p_value_vs_trajectory_random"], reverse=True)
    m = len(ordered)
    running = 1.0
    for rank_from_high, r in enumerate(ordered):
        rank = m - rank_from_high
        q = min(running, r["p_value_vs_trajectory_random"] * m / rank)
        running = q
        r["bh_q_value"] = min(1.0, q)


def audit(ctx, grid, actionability):
    by_scalar = defaultdict(list)
    for r in actionability["trajectory_rows"]:
        by_scalar[r["scalarization_id"]].append(r)
    target_rows = actionability["per_target_rows"]
    target_by_scalar = defaultdict(list)
    for r in target_rows:
        target_by_scalar[r["scalarization_id"]].append(r)
    rows = []
    for scalar in grid["rows"]:
        sid = scalar["scalarization_id"]
        trs = by_scalar[sid]
        k = sum(int(r["top1_joint_good"]) for r in trs)
        ps = [float(r["random_joint_baseline"]) for r in trs]
        target_sign = target_by_scalar[sid]
        sign_consistency = max(
            sum(int(r["auc_positive_side"]) for r in target_sign),
            sum(1 - int(r["auc_positive_side"]) for r in target_sign),
        ) / len(target_sign)
        rows.append({
            "scalarization_id": sid,
            "grid_family": scalar["grid_family"],
            "n_trajectories": len(trs),
            "observed_top1_joint_good_rate": k / len(trs),
            "expected_random_top1_joint_good_rate": sum(ps) / len(ps),
            "top1_joint_good_gain_vs_random": k / len(trs) - sum(ps) / len(ps),
            "p_value_vs_trajectory_random": poisson_binomial_upper_tail(ps, k),
            "per_target_auc_sign_consistency": sign_consistency,
            "hindsight_diagnostic_only": 1,
        })
    _holm(rows)
    _bh(rows)
    for r in rows:
        r["passes_bh_0_05"] = int(r["bh_q_value"] < 0.05)
        r["positive_scalarization_claim_allowed"] = int(
            r["passes_bh_0_05"] and
            r["per_target_auc_sign_consistency"] >= schema.TARGET_SIGN_CONSISTENCY_GATE and
            r["observed_top1_joint_good_rate"] >= schema.RELIABLE_TOP1_JOINT_GATE)
    best = max(rows, key=lambda r: (
        r["observed_top1_joint_good_rate"],
        r["top1_joint_good_gain_vs_random"],
        -r["p_value_vs_trajectory_random"]))
    summary = {
        "best_scalarization_id": best["scalarization_id"],
        "best_top1_joint_good_rate": best["observed_top1_joint_good_rate"],
        "best_expected_random_top1_joint_good_rate": best["expected_random_top1_joint_good_rate"],
        "best_top1_gain_vs_random": best["top1_joint_good_gain_vs_random"],
        "best_p_value": best["p_value_vs_trajectory_random"],
        "best_holm_p_value": best["holm_p_value"],
        "best_bh_q_value": best["bh_q_value"],
        "best_per_target_sign_consistency": best["per_target_auc_sign_consistency"],
        "any_positive_scalarization_claim_allowed": any(r["positive_scalarization_claim_allowed"] for r in rows),
    }
    top = sorted(rows, key=lambda r: (-r["observed_top1_joint_good_rate"], r["p_value_vs_trajectory_random"]))[:10]
    return {"rows": rows, "summary": summary, "best_rows": top}
