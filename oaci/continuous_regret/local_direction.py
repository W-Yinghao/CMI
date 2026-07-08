"""Local source-objective direction versus continuous target utility."""
from __future__ import annotations

import numpy as np

from ..local_boundary.artifact_loader import units
from . import endpoint_utility, schema


LOCAL_STRATEGIES = (
    ("source_score", "source_only", "score", 1.0),
    ("R_src", "source_only_R_src", "R_src", -1.0),
    ("robust_core_score", "C19_robust_core_scalar", "robust_core_score", 1.0),
    ("c30_source_rank", "C30_source_rank", "c30_source_rank", 1.0),
    ("target_unlabeled_r3_score", "target_unlabeled_R3_non_source_only", "target_unlabeled_r3_score", 1.0),
    ("target_grouped_centered_score", "target_grouped_non_deployable", "target_grouped_centered_score", 1.0),
    ("target_label_oracle_score", "target_label_oracle_non_deployable", "target_label_oracle_score", 1.0),
)


def _score(r, key, orientation=1.0):
    v = r.get(key)
    return orientation * float(v) if endpoint_utility.finite(v) else np.nan


def component_value(r, spec):
    return _score(r, spec["key"], spec.get("orientation", 1.0))


def _selected_index(cs):
    idx = [i for i, c in enumerate(cs) if c.get("selected_oaci")]
    return idx[0] if len(idx) == 1 else None


def _nearest_joint_index(cs, si):
    idx = [i for i, c in enumerate(cs) if c.get("joint_good")]
    return min(idx, key=lambda i: abs(i - si)) if idx else None


def _joint_cluster(cs, ji, si):
    if ji is None:
        return [cs[si]]
    lo = hi = ji
    while lo > 0 and cs[lo - 1].get("joint_good"):
        lo -= 1
    while hi + 1 < len(cs) and cs[hi + 1].get("joint_good"):
        hi += 1
    idx = set(range(lo, hi + 1))
    idx.add(si)
    return [cs[i] for i in sorted(idx)]


def _neighborhoods(cs, si):
    out = {}
    for k in schema.ORDER_NEIGHBORHOODS:
        out[f"pm{k}"] = cs[max(0, si - k): min(len(cs), si + k + 1)]
    out["same_cluster"] = _joint_cluster(cs, _nearest_joint_index(cs, si), si)
    se = float(cs[si]["epoch"])
    out[f"epoch_pm{schema.EPOCH_WINDOW}"] = [c for c in cs if abs(float(c["epoch"]) - se) <= schema.EPOCH_WINDOW]
    return out


def _pair_row(key, a, b, pair_scope, neighborhood, order_a, order_b, spec):
    td = endpoint_utility.endpoint_delta(a, b)
    target_delta = float(b["continuous_joint_min_margin"] - a["continuous_joint_min_margin"])
    comp_delta = component_value(b, spec) - component_value(a, spec)
    valid = endpoint_utility.finite(comp_delta) and abs(target_delta) > schema.STANDARDIZED_TINY_REGRET
    if not valid:
        agree = None
        wrong = None
        flat = None
    elif abs(comp_delta) <= schema.COMPONENT_FLAT_EPS:
        agree = 0.5
        wrong = 0
        flat = 1
    else:
        agree = 1.0 if comp_delta * target_delta > 0 else 0.0
        wrong = int(comp_delta * target_delta < 0)
        flat = 0
    return {
        "seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
        "pair_scope": pair_scope, "neighborhood": neighborhood,
        "order_a": order_a, "order_b": order_b,
        "component": spec["component"], "component_family": spec["family"],
        "target_utility_delta": target_delta,
        "component_delta": comp_delta,
        "sign_agreement": agree,
        "wrong_direction": wrong,
        "flat_component": flat,
        "target_bacc_delta": td["target_bacc_delta"],
        "target_nll_delta": td["target_nll_delta"],
        "target_ece_delta": td["target_ece_delta"],
        "joint_min_margin_delta": target_delta,
        "pareto_distance_delta": float(b.get("pareto_distance", 0.0) - a.get("pareto_distance", 0.0)),
        "source_score_delta": float(b["score"] - a["score"]),
        "target_gauge_delta": float((b.get("joint_margin") or 0.0) - (a.get("joint_margin") or 0.0)),
        "target_unlabeled_R3_delta": float(b.get("target_unlabeled_r3_score", np.nan) -
                                           a.get("target_unlabeled_r3_score", np.nan)),
    }


def _aggregate_pair_rows(pair_rows):
    out = []
    targets = sorted({r["target"] for r in pair_rows})
    for spec in schema.SOURCE_COMPONENTS:
        rows = [r for r in pair_rows if r["component"] == spec["component"] and r["sign_agreement"] is not None]
        if not rows:
            out.append({"component": spec["component"], "component_family": spec["family"],
                        "available": False, "n_pairs": 0, "pairwise_auc": None,
                        "gradient_correlation": None, "random_pairwise_auc": 0.5,
                        "wrong_direction_fraction": None, "flat_fraction": None,
                        "per_target_sign_consistency": None})
            continue
        x = np.array([r["component_delta"] for r in rows], dtype=float)
        y = np.array([r["target_utility_delta"] for r in rows], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and x[ok].std() > 1e-12 and y[ok].std() > 1e-12 else None
        per_t = []
        for t in targets:
            tr = [r for r in rows if r["target"] == t]
            if tr:
                per_t.append(float(np.mean([r["sign_agreement"] for r in tr])))
        out.append({"component": spec["component"], "component_family": spec["family"], "available": True,
                    "n_pairs": len(rows), "pairwise_auc": float(np.mean([r["sign_agreement"] for r in rows])),
                    "gradient_correlation": corr, "random_pairwise_auc": 0.5,
                    "wrong_direction_fraction": float(np.mean([r["wrong_direction"] for r in rows])),
                    "flat_fraction": float(np.mean([r["flat_component"] for r in rows])),
                    "per_target_sign_consistency": (
                        float(np.mean([int(v > 0.5) for v in per_t])) if per_t else None)})
    return out


def local_random_baseline(rows):
    out = []
    for key, cs in units(rows).items():
        si = _selected_index(cs)
        if si is None:
            continue
        for nname, cands in _neighborhoods(cs, si).items():
            vals = np.array([float(c["continuous_joint_min_margin"]) for c in cands], dtype=float)
            if not len(vals):
                continue
            oracle = float(vals.max())
            random_regret = float(np.mean(oracle - vals))
            for strategy, info_class, key_name, orientation in LOCAL_STRATEGIES:
                scored = [(c, _score(c, key_name, orientation)) for c in cands]
                scored = [(c, s) for c, s in scored if np.isfinite(s)]
                if not scored:
                    continue
                chosen = max(scored, key=lambda x: x[1])[0]
                regret = float(oracle - float(chosen["continuous_joint_min_margin"]))
                out.append({"seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
                            "neighborhood": nname, "strategy": strategy, "info_class": info_class,
                            "n": len(cands), "local_oracle_utility": oracle,
                            "strategy_top1_regret": regret, "local_random_mean_regret": random_regret,
                            "regret_improvement_vs_random": random_regret - regret})
    return out


def _aggregate_random(rows):
    out = []
    for strategy in sorted({r["strategy"] for r in rows}):
        for nname in sorted({r["neighborhood"] for r in rows}):
            rs = [r for r in rows if r["strategy"] == strategy and r["neighborhood"] == nname]
            if not rs:
                continue
            src = [r for r in rows if r["strategy"] == "source_score" and r["neighborhood"] == nname]
            src_mean = float(np.mean([r["strategy_top1_regret"] for r in src])) if src else None
            mean_reg = float(np.mean([r["strategy_top1_regret"] for r in rs]))
            out.append({"strategy": strategy, "info_class": rs[0]["info_class"], "neighborhood": nname,
                        "mean_strategy_top1_regret": mean_reg,
                        "mean_local_random_regret": float(np.mean([r["local_random_mean_regret"] for r in rs])),
                        "mean_regret_improvement_vs_random": float(np.mean([r["regret_improvement_vs_random"] for r in rs])),
                        "regret_delta_vs_source": (mean_reg - src_mean if src_mean is not None else None)})
    return out


def local_source_direction(rows) -> dict:
    pair_rows = []
    for key, cs in units(rows).items():
        for i in range(len(cs) - 1):
            for spec in schema.SOURCE_COMPONENTS:
                if spec.get("available", True):
                    pair_rows.append(_pair_row(key, cs[i], cs[i + 1], "adjacent", "adjacent_pm1",
                                               cs[i].get("order"), cs[i + 1].get("order"), spec))
        si = _selected_index(cs)
        if si is None:
            continue
        s = cs[si]
        for nname, cands in _neighborhoods(cs, si).items():
            for c in cands:
                if c is s:
                    continue
                for spec in schema.SOURCE_COMPONENTS:
                    if spec.get("available", True):
                        pair_rows.append(_pair_row(key, s, c, "selected_neighborhood", nname,
                                                   s.get("order"), c.get("order"), spec))
    random_rows = local_random_baseline(rows)
    aggregate = _aggregate_pair_rows(pair_rows)
    random_aggregate = _aggregate_random(random_rows)
    source = next((r for r in aggregate if r["component"] == "source_score"), {})
    return {"summary": {
        "n_pair_component_rows": len(pair_rows),
        "source_pairwise_auc": source.get("pairwise_auc"),
        "source_gradient_correlation": source.get("gradient_correlation"),
        "source_wrong_direction_fraction": source.get("wrong_direction_fraction"),
        "source_flat_fraction": source.get("flat_fraction"),
        "random_pairwise_auc": 0.5,
    }, "pair_rows": pair_rows, "aggregate": aggregate,
        "random_rows": random_rows, "random_aggregate": random_aggregate}
