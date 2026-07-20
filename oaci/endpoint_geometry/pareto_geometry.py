"""C31 Q5 — Pareto trajectory geometry. For each (seed,target,level) training trajectory: the Pareto front (non-
dominated in bAcc↑/NLL↓/ECE↓), the dominated fraction, the accuracy-oracle (max bAcc) and its calibration
status, the calibration-oracle (min NLL) and its accuracy status, and whether joint-good Pareto points exist.
Tests whether the C16 barrier is a Pareto trade-off: the accuracy-oracle systematically sacrifices calibration,
and joint-good Pareto points exist but the source rank does not identify them. Diagnostic-only; oracles are
non-deployable."""
from __future__ import annotations

import numpy as np

from . import artifact_loader


def pareto_geometry(rows) -> dict:
    by_traj = {}
    for r in rows:
        by_traj.setdefault((r["seed"], r["target"], r["level"]), []).append(r)
    front_sizes, dominated_fracs, acc_oracle_calib_bad, joint_pareto_exists = [], [], [], []
    acc_oracle_strict_calib_bad, nll_oracle_acc_bad, ece_oracle_acc_bad = [], [], []
    per_traj = []
    for traj, cs in by_traj.items():
        valid = [c for c in cs if None not in (c["bacc"], c["nll"], c["ece"])]
        if len(valid) < 3:
            continue
        front = [c for c in valid if c["pareto_good"] == 1]
        front_sizes.append(len(front)); dominated_fracs.append(np.mean([c["dominated"] for c in valid]))
        acc_oracle = max(valid, key=lambda c: c["bacc"])              # highest target bAcc (non-deployable oracle)
        acc_oracle_calib_bad.append(int(acc_oracle["calibration_good"] == 0))                    # OR-calibration
        acc_oracle_strict_calib_bad.append(int(not (acc_oracle["nll_good"] and acc_oracle["ece_good"])))  # strict: both
        # symmetry: does the calibration-oracle sacrifice accuracy? (base-rate check on the trade-off)
        nll_oracle = min(valid, key=lambda c: c["nll"]); ece_oracle = min(valid, key=lambda c: c["ece"])
        nll_oracle_acc_bad.append(int(nll_oracle["accuracy_good"] == 0)); ece_oracle_acc_bad.append(int(ece_oracle["accuracy_good"] == 0))
        joint_pareto = [c for c in front if c["joint_good"] == 1]
        joint_pareto_exists.append(int(len(joint_pareto) > 0))
        per_traj.append({"seed": traj[0], "target": traj[1], "level": traj[2], "n": len(valid),
                         "pareto_front_size": len(front), "dominated_fraction": float(np.mean([c["dominated"] for c in valid])),
                         "accuracy_oracle_calibration_good": int(acc_oracle["calibration_good"]),
                         "joint_good_pareto_points": len(joint_pareto)})
    # does the source rank identify the pareto/joint-good checkpoints? (within-target AUC of score for pareto/joint)
    score_ranks_pareto = artifact_loader.within_target_auc(rows, "score", "pareto_good")
    score_ranks_joint = artifact_loader.within_target_auc(rows, "score", "joint_good")
    _m = lambda a: float(np.mean(a)) if a else None
    return {"n_trajectories": len(front_sizes), "mean_pareto_front_size": float(np.mean(front_sizes)) if front_sizes else None,
            "mean_dominated_fraction": float(np.mean(dominated_fracs)) if dominated_fracs else None,
            "accuracy_oracle_calibration_bad_fraction": _m(acc_oracle_calib_bad),
            "accuracy_oracle_strict_calibration_bad_fraction": _m(acc_oracle_strict_calib_bad),   # both-NLL-and-ECE
            "nll_oracle_accuracy_bad_fraction": _m(nll_oracle_acc_bad),                           # symmetry (calib-oracle)
            "ece_oracle_accuracy_bad_fraction": _m(ece_oracle_acc_bad),
            "joint_good_pareto_exists_fraction": _m(joint_pareto_exists),
            "source_score_ranks_pareto_auc": score_ranks_pareto, "source_score_ranks_joint_auc": score_ranks_joint,
            "per_trajectory": per_traj,
            "note": ("mean Pareto front %.1f/traj, dominated fraction %.2f; accuracy-oracle is calibration-BAD in "
                     "%.0f%% of trajectories; joint-good Pareto points exist in %.0f%% of trajectories; source score "
                     "ranks pareto/joint within-target AUC %.3f/%.3f"
                     % (np.mean(front_sizes) if front_sizes else 0, np.mean(dominated_fracs) if dominated_fracs else 0,
                        100 * (np.mean(acc_oracle_calib_bad) if acc_oracle_calib_bad else 0),
                        100 * (np.mean(joint_pareto_exists) if joint_pareto_exists else 0),
                        score_ranks_pareto or 0, score_ranks_joint or 0))}
