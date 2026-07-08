"""C33 adjacent local score-gradient alignment."""
from __future__ import annotations

import numpy as np

from . import schema
from .artifact_loader import units


def _transition_type(a, b):
    if a == 0 and b == 1:
        return "bad_to_good"
    if a == 1 and b == 0:
        return "good_to_bad"
    if a == 1 and b == 1:
        return "good_to_good"
    return "bad_to_bad"


def _agree(delta, typ):
    if typ == "bad_to_good":
        return int(delta > schema.SOURCE_FLAT_EPS)
    if typ == "good_to_bad":
        return int(delta < -schema.SOURCE_FLAT_EPS)
    return None


def local_gradient_alignment(rows):
    grads = []
    for key, cs in units(rows).items():
        for i in range(len(cs) - 1):
            a, b = cs[i], cs[i + 1]
            typ = _transition_type(int(a["joint_good"]), int(b["joint_good"]))
            sd = float(b["score"] - a["score"])
            gd = float((b.get("joint_margin") or 0.0) - (a.get("joint_margin") or 0.0))
            grads.append({
                "seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
                "order_a": a.get("order"), "order_b": b.get("order"), "transition_type": typ,
                "source_score_gradient": sd,
                "R_src_gradient": float(b["R_src"] - a["R_src"]),
                "rank_gradient": float(b["c30_source_rank"] - a["c30_source_rank"]),
                "target_gauge_margin_gradient": gd,
                "target_unlabeled_R3_gradient": float(b.get("target_unlabeled_r3_score", np.nan) -
                                                       a.get("target_unlabeled_r3_score", np.nan)),
                "source_sign_agrees_with_transition": _agree(sd, typ),
                "rank_sign_agrees_with_transition": _agree(float(b["c30_source_rank"] - a["c30_source_rank"]), typ),
                "target_gauge_jump": int(abs(gd) >= schema.GAUGE_JUMP_EPS),
            })
    transition = [g for g in grads if g["source_sign_agrees_with_transition"] is not None]
    summary = {
        "n_adjacent_pairs": len(grads),
        "n_transition_pairs": len(transition),
        "transition_fraction": len(transition) / len(grads) if grads else None,
        "source_gradient_agreement": float(np.mean([g["source_sign_agrees_with_transition"] for g in transition])) if transition else None,
        "rank_gradient_agreement": float(np.mean([g["rank_sign_agrees_with_transition"] for g in transition])) if transition else None,
        "transition_gauge_jump_fraction": float(np.mean([g["target_gauge_jump"] for g in transition])) if transition else None,
        "mean_abs_source_transition_gradient": float(np.mean([abs(g["source_score_gradient"]) for g in transition])) if transition else None,
    }
    return {"summary": summary, "gradients": grads}
