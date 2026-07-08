"""C32 trajectory-conditioned random baseline and top-k enrichment."""
from __future__ import annotations

import numpy as np

from .landscape import by_trajectory


def random_hit_prob(n, n_good, k) -> float:
    """Probability that k draws without replacement from one trajectory include at least one joint-good."""
    if n <= 0 or n_good <= 0:
        return 0.0
    if k >= n:
        return 1.0
    p_miss = 1.0
    for i in range(k):
        p_miss *= max(n - n_good - i, 0) / max(n - i, 1)
    return float(1.0 - p_miss)


def random_baseline(rows, top_ks) -> dict:
    by = by_trajectory(rows)
    out = []
    for k in top_ks:
        probs, precisions = [], []
        for cs in by.values():
            n = len(cs)
            g = sum(int(c["joint_good"]) for c in cs)
            probs.append(random_hit_prob(n, g, k))
            precisions.append(g / n if n else 0.0)
        out.append({"k": k, "hit_rate": float(np.mean(probs)) if probs else None,
                    "expected_precision_at_k": float(np.mean(precisions)) if precisions else None})
    return {"topk": out}


def topk_enrichment(rows, score_getter, top_ks, label_key="joint_good") -> dict:
    by = by_trajectory(rows)
    rand = {r["k"]: r for r in random_baseline(rows, top_ks)["topk"]}
    out, per_traj = [], []
    for k in top_ks:
        hits, precisions = [], []
        for traj, cs in by.items():
            top = sorted(cs, key=score_getter, reverse=True)[:k]
            hit = int(any(int(c[label_key]) for c in top))
            prec = float(np.mean([int(c[label_key]) for c in top])) if top else 0.0
            hits.append(hit)
            precisions.append(prec)
            per_traj.append({"seed": traj[0], "target": traj[1], "level": traj[2], "regime": traj[3], "k": k,
                             "topk_hit": hit, "precision_at_k": prec})
        hit_rate = float(np.mean(hits)) if hits else None
        precision = float(np.mean(precisions)) if precisions else None
        rh = rand[k]["hit_rate"]
        rp = rand[k]["expected_precision_at_k"]
        out.append({"k": k, "hit_rate": hit_rate, "random_hit_rate": rh,
                    "hit_enrichment": (hit_rate / rh if rh and hit_rate is not None else None),
                    "precision_at_k": precision, "random_precision_at_k": rp,
                    "precision_enrichment": (precision / rp if rp and precision is not None else None)})
    return {"topk": out, "per_trajectory": per_traj}
