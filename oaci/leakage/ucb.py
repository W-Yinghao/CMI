"""Upper confidence bound on ``L_Q^ov`` via a recording-clustered bootstrap.

Discipline (the hard constraints):

* resample **whole recording groups** with replacement, **stratified within domain** (each
  domain keeps its group count) — the design ``p(D)`` is held fixed; we estimate the
  variability of the conditional from recording-level resampling;
* the ``SupportGraph``, ``S_y``, ``p_ref`` and the ``fold_of_group`` map are **fixed outside**
  the bootstrap; a replicate whose per-cell count drops below ``m`` does NOT redefine the
  estimand (the label space stays ``S_y``);
* duplicate copies of a resampled group inherit the original group's fold (guaranteed: the
  rebuilt rows keep the original group id and ``oof_nll_by_class`` looks fold up by it);
* **all capacities are retrained and the capacity sup is re-selected inside every replicate**
  (``estimate_extractable_leakage`` does the argmax) — selecting capacity once on the full
  sample would make the bound optimistic;
* negative estimates are kept (no clipping).

Main bound = one-sided **basic** cluster bootstrap upper limit
``bootstrap_ucl = 2·L̂_Q − Q_α(L̂*_Q)``; the percentile endpoint ``Q_{1-α}(L̂*_Q)`` is
reported separately as a sensitivity analysis (the two are never mixed). Output keys are
``extractable_LQ_ov`` / ``L_abs`` / ``L_cond`` / ``bootstrap_ucl`` — never "CMI" or
"true-I_ov upper bound".
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..support_graph import SupportGraph
from .critic import CriticConfig
from .crossfit import FoldPlan, FrozenFeatures, feat_population_hash
from .design import LeakageDesign, population_hash
from .estimate import estimate_extractable_leakage
from .plan import make_leakage_bootstrap_plan


def _group_to_rows(feat: FrozenFeatures) -> dict:
    rows: dict = defaultdict(list)
    for i, g in enumerate(feat.group):
        rows[g].append(i)           # STRING group key — never int-cast
    return {g: np.array(ix, dtype=int) for g, ix in rows.items()}


def within_domain_group_bootstrap(fold_plan: FoldPlan, rng) -> list[int]:
    """Resample recording groups with replacement, holding each domain's group count fixed.
    Returns a list of ORIGINAL group ids (with duplicates)."""
    by_dom: dict[int, list[int]] = defaultdict(list)
    for g, dom in fold_plan.domain_of_group.items():
        by_dom[dom].append(g)
    resampled: list[int] = []
    for dom in sorted(by_dom):
        gs = np.array(sorted(by_dom[dom]))
        resampled.extend(gs[rng.integers(0, gs.size, size=gs.size)].tolist())
    return resampled


def _rebuild(feat: FrozenFeatures, group_rows: dict, resampled_groups: list) -> FrozenFeatures:
    # a group drawn r times contributes its rows r times, EACH at its original base mass (the
    # replicate mass is r x the base mass; never renormalised back to a single copy). A replicate
    # gets fresh synthetic sample ids (duplicate rows need distinct ids).
    take = np.concatenate([group_rows[g] for g in resampled_groups])
    return FrozenFeatures(Z=feat.Z[take], y=feat.y[take], d=feat.d[take], group=feat.group[take],
                          sample_mass=feat.sample_mass[take],
                          sample_id=tuple(f"b{i}" for i in range(take.size)))


def _design_from_feat(feat: FrozenFeatures, support_graph: SupportGraph) -> LeakageDesign:
    """A LeakageDesign for the legacy entry point — reuses the feature's OWN string sample/group ids
    (no int round-trip); skips the cell-mass cross-check the formal ``make_leakage_design`` does."""
    sid = tuple(feat.sample_id)
    grp = tuple(str(g) for g in feat.group.tolist())
    y = np.asarray(feat.y, dtype=np.int64); d = np.asarray(feat.d, dtype=np.int64)
    mass = np.asarray(feat.sample_mass, dtype=np.float64)
    return LeakageDesign(sid, y, d, grp, mass, population_hash(sid, y, d, grp, mass),
                         support_graph.support_hash())


def bootstrap_ucb(
    feat: FrozenFeatures,
    support_graph: SupportGraph,
    fold_plan: FoldPlan,
    cfg: CriticConfig,
    alpha: float = 0.1,
    n_bootstrap: int = 200,
    seed: int = 0,
    *,
    bootstrap_plan=None,
) -> dict:
    """Point estimate + one-sided basic bootstrap UCL replaying an EXPLICIT
    :class:`LeakageBootstrapPlan`. With no plan supplied, a plan is built here (legacy entry point,
    ``max_invalid_draw_rate=1.0``) and then replayed through the SAME path — so the bootstrap is
    always plan-driven (no in-loop RNG, no per-method redraw)."""
    if bootstrap_plan is None:
        bootstrap_plan = make_leakage_bootstrap_plan(
            _design_from_feat(feat, support_graph), support_graph, fold_plan, alpha=alpha,
            requested_replicates=n_bootstrap, seed=seed, max_candidate_multiplier=5,
            max_invalid_draw_rate=1.0)

    # validate the plan matches THIS feature set / support / fold map / population / alpha
    feat_pop = feat_population_hash(feat)
    if feat_pop != fold_plan.population_hash:
        raise ValueError("features do not match the fold plan's population (sample_id-bound)")
    if feat_pop != bootstrap_plan.population_hash:
        raise ValueError("features do not match the bootstrap plan's population")
    if support_graph.support_hash() != bootstrap_plan.support_hash:
        raise ValueError("support graph does not match the bootstrap plan")
    if fold_plan.plan_hash != bootstrap_plan.fold_plan_hash:
        raise ValueError("fold plan does not match the bootstrap plan")

    point = estimate_extractable_leakage(feat, support_graph, fold_plan, cfg)
    Lhat = point["extractable_LQ_ov"]
    group_rows = _group_to_rows(feat)
    by_id = {dr.candidate_id: dr for dr in bootstrap_plan.candidate_draws}

    reps, rep_caps = [], []
    for cid in bootstrap_plan.accepted_candidate_ids:
        resampled = [g for g, m in by_id[cid].group_multiplicities for _ in range(int(m))]   # str groups
        feat_b = _rebuild(feat, group_rows, resampled)
        try:
            est_b = estimate_extractable_leakage(feat_b, support_graph, fold_plan, cfg)
        except ValueError as e:                     # an accepted draw must NOT fail; report it
            raise ValueError(f"accepted bootstrap candidate {cid} failed during scoring: {e}")
        reps.append(est_b["extractable_LQ_ov"]); rep_caps.append(est_b["selected_capacity"])

    reps = np.asarray(reps, dtype=np.float64)
    q_low = float(np.quantile(reps, bootstrap_plan.alpha))
    return {
        "extractable_LQ_ov": float(Lhat),
        "L_abs": float(point["L_abs"]),
        "L_cond": float(point["L_cond"]),
        "selected_capacity": point["selected_capacity"],
        "bootstrap_ucl": float(2.0 * Lhat - q_low),
        "percentile_ucl": float(np.quantile(reps, 1.0 - bootstrap_plan.alpha)),
        "alpha": float(bootstrap_plan.alpha),
        "n_bootstrap": int(len(reps)),
        "replicates": reps,
        "replicate_capacities": rep_caps,
        "bootstrap_plan_hash": bootstrap_plan.plan_hash,
        "fold_plan_hash": fold_plan.plan_hash,
        "accepted_candidate_ids": tuple(bootstrap_plan.accepted_candidate_ids),
        "candidate_draw_count": int(len(bootstrap_plan.candidate_draws)),
        "invalid_draw_rate": float(bootstrap_plan.invalid_draw_rate),
        "reference_entropy": point["reference_entropy"],
        "fold_notes": fold_plan.notes,
    }
