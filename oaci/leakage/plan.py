"""Explicit leakage bootstrap PLAN.

The accepted recording-group resamples are drawn and validated ONCE here, from labels / domains /
groups / mass / support / fold map only — never from ``Z`` or a method. ``bootstrap_ucb`` then
replays the accepted draws byte-for-byte (no RNG, no per-method redraw). A candidate is structurally
valid iff every comparable class's replicated eligible mass lands in ≥2 cross-fit folds (so each
fold with test mass has training mass elsewhere) — checked WITHOUT rebuilding the support graph.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .errors import BootstrapPlanNonEstimable


@dataclass(frozen=True)
class BootstrapDraw:
    candidate_id: int
    group_multiplicities: tuple        # ((group_str, mult), ...) sorted, ALL groups incl mult 0


@dataclass(frozen=True)
class LeakageBootstrapPlan:
    population_hash: str
    support_hash: str
    fold_plan_hash: str
    alpha: float
    requested_replicates: int
    candidate_draws: tuple             # (BootstrapDraw, ...) — full generated sequence
    accepted_candidate_ids: tuple
    invalid_draw_rate: float
    plan_hash: str


def _plan_hash(pop, sup, fold, alpha, requested, candidate_draws, accepted, invalid_rate) -> str:
    h = hashlib.sha256()
    h.update(pop.encode()); h.update(sup.encode()); h.update(fold.encode())
    h.update(f"{float(alpha)!r}|{int(requested)}".encode())
    for dr in candidate_draws:
        h.update(f"C{dr.candidate_id}:".encode())
        for g, m in dr.group_multiplicities:
            h.update(f"{g}={m};".encode())
    h.update(("A" + ",".join(str(i) for i in accepted)).encode())
    h.update(f"|inv={invalid_rate!r}".encode())
    return h.hexdigest()


def bootstrap_plan_hash(plan: "LeakageBootstrapPlan") -> str:
    return _plan_hash(plan.population_hash, plan.support_hash, plan.fold_plan_hash, plan.alpha,
                      plan.requested_replicates, plan.candidate_draws, plan.accepted_candidate_ids,
                      plan.invalid_draw_rate)


def validate_bootstrap_plan(plan: "LeakageBootstrapPlan") -> None:
    if bootstrap_plan_hash(plan) != plan.plan_hash:
        raise ValueError("bootstrap plan hash does not recompute")


def make_leakage_bootstrap_plan(design, support_graph, fold_plan, *, alpha, requested_replicates,
                                seed, max_candidate_multiplier=8, max_invalid_draw_rate=0.5):
    if support_graph.support_hash() != design.support_hash:
        raise ValueError("support graph and design disagree on support_hash")
    if fold_plan.support_hash != design.support_hash:
        raise ValueError("fold plan and design disagree on support_hash")
    fold_of = {str(g): int(f) for g, f in fold_plan.fold_of_group.items()}
    dom_of = {str(g): int(dv) for g, dv in fold_plan.domain_of_group.items()}
    groups = sorted(set(design.group_id))
    if set(groups) != set(fold_of):
        raise ValueError("design groups do not match the fold plan's groups")

    comparable = list(support_graph.comparable_classes)
    sup_of = {y: set(int(d) for d in support_graph.support_of_class[y]) for y in comparable}
    # per group: eligible mass per comparable class
    elig = {g: defaultdict(float) for g in groups}
    for g, yi, di, mi in zip(design.group_id, design.y.tolist(), design.d.tolist(), design.sample_mass.tolist()):
        for y in comparable:
            if int(yi) == y and int(di) in sup_of[y]:
                elig[str(g)][y] += float(mi)
    by_dom = defaultdict(list)
    for g in groups:
        by_dom[dom_of[g]].append(g)
    for dom in by_dom:
        by_dom[dom].sort()

    def _valid(mult):
        for y in comparable:
            fold_mass = defaultdict(float)
            for g, m in mult.items():
                if m and elig[g].get(y, 0.0) > 0:
                    fold_mass[fold_of[g]] += m * elig[g][y]
            if sum(1 for v in fold_mass.values() if v > 0) < 2:
                return False
        return True

    rng = np.random.default_rng(seed)
    budget = int(requested_replicates) * int(max_candidate_multiplier)
    candidates, accepted = [], []
    cid = 0
    while len(accepted) < requested_replicates and cid < budget:
        mult = {g: 0 for g in groups}
        for dom in sorted(by_dom):
            gs = by_dom[dom]
            for j in rng.integers(0, len(gs), size=len(gs)):
                mult[gs[int(j)]] += 1
        draw = BootstrapDraw(candidate_id=cid, group_multiplicities=tuple(sorted(mult.items())))
        candidates.append(draw)
        if _valid(mult):
            accepted.append(cid)
        cid += 1

    n_cand = len(candidates)
    invalid_rate = 0.0 if n_cand == 0 else (n_cand - len(accepted)) / n_cand
    if len(accepted) < requested_replicates:
        raise BootstrapPlanNonEstimable(f"only {len(accepted)}/{requested_replicates} structurally-valid "
                                        f"bootstrap draws within {budget} candidates; too few clusters")
    if invalid_rate > max_invalid_draw_rate:
        raise BootstrapPlanNonEstimable(f"invalid_draw_rate {invalid_rate:.3f} > {max_invalid_draw_rate}")
    candidate_draws = tuple(candidates); accepted = tuple(accepted)
    return LeakageBootstrapPlan(
        population_hash=design.population_hash, support_hash=design.support_hash,
        fold_plan_hash=fold_plan.plan_hash, alpha=float(alpha),
        requested_replicates=int(requested_replicates), candidate_draws=candidate_draws,
        accepted_candidate_ids=accepted, invalid_draw_rate=float(invalid_rate),
        plan_hash=_plan_hash(design.population_hash, design.support_hash, fold_plan.plan_hash,
                             alpha, requested_replicates, candidate_draws, accepted, invalid_rate))
