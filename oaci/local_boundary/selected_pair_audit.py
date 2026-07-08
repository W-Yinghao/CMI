"""C33 selected-vs-nearest-joint-good pair audit."""
from __future__ import annotations

import numpy as np

from . import schema
from .artifact_loader import units


def _nearest_joint(cs, selected_idx):
    joints = [i for i, c in enumerate(cs) if c["joint_good"]]
    if not joints:
        return None
    return min(joints, key=lambda i: abs(i - selected_idx))


def selected_pair_audit(rows):
    pairs = []
    for key, cs in units(rows).items():
        selected = [i for i, c in enumerate(cs) if c.get("selected_oaci")]
        if len(selected) != 1:
            continue
        si = selected[0]
        ji = _nearest_joint(cs, si)
        s = cs[si]
        if ji is None:
            pairs.append({"seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
                          "pair_status": "no_joint_good_in_unit", "selected_joint_good": int(s["joint_good"])})
            continue
        j = cs[ji]
        source_delta = float(j["score"] - s["score"])
        rank_delta = float(j["c30_source_rank"] - s["c30_source_rank"])
        gauge_delta = float((j.get("joint_margin") or 0.0) - (s.get("joint_margin") or 0.0))
        tu_delta = float(j.get("target_unlabeled_r3_score", np.nan) - s.get("target_unlabeled_r3_score", np.nan))
        if s["joint_good"]:
            pair_case = "selected_already_joint_good"
        elif abs(source_delta) <= schema.SOURCE_FLAT_EPS:
            pair_case = "source_flat"
        elif source_delta < -schema.SOURCE_FLAT_EPS:
            pair_case = "source_wrong"
        elif rank_delta > 0 and abs(source_delta) <= 2 * schema.SOURCE_FLAT_EPS:
            pair_case = "rank_correct_but_weak"
        else:
            pair_case = "source_correct_or_selected_already_good"
        gauge_jump = bool(gauge_delta >= schema.GAUGE_JUMP_EPS and source_delta <= schema.SOURCE_FLAT_EPS)
        pairs.append({
            "seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
            "pair_status": "ok", "selected_joint_good": int(s["joint_good"]),
            "order_delta": abs(ji - si), "epoch_delta": abs(float(j["epoch"]) - float(s["epoch"])),
            "bacc_gain_to_nearest_joint": float(j["bacc"] - s["bacc"]),
            "nll_gain_to_nearest_joint": float(s["nll"] - j["nll"]),
            "ece_gain_to_nearest_joint": float(s["ece"] - j["ece"]),
            "source_score_delta_to_nearest_joint": source_delta,
            "source_rank_delta_to_nearest_joint": rank_delta,
            "R_src_delta_to_nearest_joint": float(j["R_src"] - s["R_src"]),
            "target_gauge_margin_delta_to_nearest_joint": gauge_delta,
            "target_unlabeled_R3_delta_to_nearest_joint": tu_delta,
            "pair_case": pair_case,
            "gauge_jump_unseen_by_source": int(gauge_jump),
        })
    ok = [p for p in pairs if p.get("pair_status") == "ok"]
    cases = {}
    for p in ok:
        cases[p["pair_case"]] = cases.get(p["pair_case"], 0) + 1
    misses = [p for p in ok if not p["selected_joint_good"]]
    miss_cases = {}
    for p in misses:
        miss_cases[p["pair_case"]] = miss_cases.get(p["pair_case"], 0) + 1
    summary = {
        "n_pairs": len(ok),
        "n_miss_pairs": len(misses),
        "selected_joint_hit_rate": float(np.mean([p["selected_joint_good"] for p in ok])) if ok else None,
        "median_order_delta": float(np.median([p["order_delta"] for p in ok])) if ok else None,
        "median_epoch_delta": float(np.median([p["epoch_delta"] for p in ok])) if ok else None,
        "source_flat_fraction": miss_cases.get("source_flat", 0) / len(misses) if misses else None,
        "source_wrong_fraction": miss_cases.get("source_wrong", 0) / len(misses) if misses else None,
        "rank_weak_fraction": miss_cases.get("rank_correct_but_weak", 0) / len(misses) if misses else None,
        "gauge_jump_unseen_fraction": float(np.mean([p["gauge_jump_unseen_by_source"] for p in misses])) if misses else None,
        "all_pair_source_flat_fraction": cases.get("source_flat", 0) / len(ok) if ok else None,
        "all_pair_source_wrong_fraction": cases.get("source_wrong", 0) / len(ok) if ok else None,
        "pair_case_counts": cases,
        "miss_pair_case_counts": miss_cases,
    }
    return {"summary": summary, "pairs": pairs}
