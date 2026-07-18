"""Bounded shadow pilot: pre-registered boundary taxonomy + discriminative demo.

Runs entirely on synthetic KNOWN-TRUTH scenarios through the Semantics-B query
server. Two jobs:

1. Pre-register the boundary decision rules (``classify`` + the taxonomy in
   ``constants.BOUNDARY_TAXONOMY``) BEFORE any real query.
2. Demonstrate the instrument + criteria correctly classify five ground-truth
   scenarios — a discriminative-validity check for the active-testing regime (the
   analogue of the positive control the 0-label battery never had).

No real data is opened and no claim is made about the real C84 field. The active
policy here is a static acquisition-score surrogate ("query the highest-score
trials"); real C86D would use LURE / Hara adaptive scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as K
from .field import DevelopmentField
from .query_server import QueryServer

CVAR_ALPHA = 0.25          # tail = mean of worst 25% context regrets
NEAR_OPT_EPS = 0.05        # a context is "near-optimal" if regret <= this
NEAROPT_MARGIN = 0.05      # near-opt probability improvement must beat this
TEST_BUDGET = 4            # small-budget operating point
N_PHYS = 32                # physical trials per target (FULL == 32)
N_CTX = 8                  # contexts per physical trial (production-equivalent)
K_CAND = 24                # candidates per context
PASSIVE_SEEDS = 8          # average passive uniform over this many draws

COHORTS = {"cohortA": ("T0", "T1", "T2"), "cohortB": ("T3", "T4", "T5")}
_ALL_TARGETS = tuple(t for ts in COHORTS.values() for t in ts)


@dataclass(frozen=True)
class PolicyMetrics:
    mean_regret: float
    tail_regret: float          # CVaR over context regrets
    near_opt_prob: float
    top1_rate: float
    mean_regret_by_cohort: dict[str, float]


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    passive: PolicyMetrics
    registered: PolicyMetrics
    oracle: PolicyMetrics
    ceiling: PolicyMetrics
    classification: str
    expected: str


# --------------------------------------------------------------------------- #
# Pre-registered classifier (frozen decision rules)
# --------------------------------------------------------------------------- #
def classify(passive: PolicyMetrics, registered: PolicyMetrics, oracle: PolicyMetrics,
             ceiling: PolicyMetrics) -> str:
    """Map measured policy metrics to a pre-registered boundary taxonomy label."""
    m = K.TAU_REGRET_MARGIN
    ceil_ok = ceiling.mean_regret <= K.TAU_CEILING_REGRET
    reg_gain = (passive.mean_regret - registered.mean_regret) >= m
    oracle_gain = (passive.mean_regret - oracle.mean_regret) >= m
    tail_gain = (passive.tail_regret - registered.tail_regret) >= m
    nearopt_gain = (registered.near_opt_prob - passive.near_opt_prob) >= NEAROPT_MARGIN
    cross_cohort = all(
        (passive.mean_regret_by_cohort[c] - registered.mean_regret_by_cohort[c]) >= m
        for c in passive.mean_regret_by_cohort
    )

    if not ceil_ok:
        return "ACQUISITION_VIEW_NONTRANSPORTABLE"
    if not reg_gain:
        return "POLICY_LIMITED" if oracle_gain else "NO_REGISTERED_ACTIVE_GAIN"
    if tail_gain and nearopt_gain and cross_cohort:
        return "BOUNDARY_OPERATIONALLY_CROSSED"
    return "BOUNDARY_WEAKENED_NOT_ROBUST"


# --------------------------------------------------------------------------- #
# Ground-truth scenario construction
# --------------------------------------------------------------------------- #
@dataclass
class _Truth:
    field: DevelopmentField
    availability: dict
    acc: dict                       # context -> np.ndarray[K] true accuracy
    best: dict                      # context -> int
    contexts_of: dict               # target -> list[context]
    reg_score: dict                 # trial_id -> float (registered acquisition score)
    oracle_score: dict              # trial_id -> float (true informativeness)
    structured: bool                # False when trials are exchangeable (no acquisition can help)


def _make_probs(correct: np.ndarray, label: int) -> np.ndarray:
    """[K,2] probs whose argmax == label iff correct[k]."""
    probs = np.empty((correct.shape[0], 2))
    hi, lo = 0.8, 0.2
    probs[:, label] = np.where(correct == 1, hi, lo)
    probs[:, 1 - label] = np.where(correct == 1, lo, hi)
    return probs


def _build(scenario: str, seed: int) -> _Truth:
    rng = np.random.default_rng(seed)
    construction_ids: list[str] = []
    plan: list[tuple] = []          # (target, trial_id, label, {ctx: probs})
    acc: dict = {}
    best: dict = {}
    contexts_of: dict = {}
    reg_score: dict = {}
    oracle_score: dict = {}

    for target in _ALL_TARGETS:
        contexts = [f"{target}/c{ci}" for ci in range(N_CTX)]
        contexts_of[target] = contexts
        tail_ctx = set(contexts[-2:]) if scenario == "S4" else set()

        # per-context true accuracies + best index
        for c in contexts:
            a = np.full(K_CAND, 0.6)
            b = int(rng.integers(0, K_CAND))
            if scenario == "S5":                       # dense near-optimal cluster
                cluster = rng.choice(K_CAND, size=5, replace=False)
                a[cluster] = [0.86, 0.87, 0.88, 0.89, 0.90]
                b = int(cluster[np.argmax(a[cluster])])
            elif scenario == "S2":                     # one best, tier-2 crowd (sample-size limited)
                a[:] = 0.5
                tier2 = rng.choice(K_CAND, size=6, replace=False)
                a[tier2] = 0.60
                a[b] = 0.95
            else:                                      # S1/S3/S4 big margin
                a[b] = 0.90
                if c in tail_ctx:                      # tail: only type-B separates
                    pass
            acc[c] = a
            best[c] = int(np.argmax(a))

        # physical trials
        for j in range(N_PHYS):
            trial_id = f"{target}-t{j}"
            construction_ids.append(trial_id)
            label = int(rng.integers(0, 2))
            is_disc = j < 4                              # first 4 trials are informative
            is_typeB = 4 <= j < 8                        # next 4 = type-B (tail-informative)
            ctx_probs = {}
            for c in contexts:
                a = acc[c]
                if scenario == "S2":                     # iid sample-size-limited
                    correct = (rng.random(K_CAND) < a).astype(np.int64)
                else:                                    # deterministic threshold
                    in_tail = c in tail_ctx
                    if scenario == "S4" and in_tail:
                        d = 0.75 if is_typeB else 0.05   # tail resolved only by type-B
                    else:
                        d = 0.75 if is_disc else 0.05    # non-tail resolved by type-A
                    correct = (a >= d).astype(np.int64)
                ctx_probs[c] = _make_probs(correct, label)
            plan.append((target, trial_id, label, ctx_probs))

            # acquisition scores
            if scenario == "S2":
                reg_score[trial_id] = float(rng.random())      # flat/uninformative
            elif scenario == "S3":
                reg_score[trial_id] = 1.0 if not is_disc else 0.0   # anti-aligned
            elif scenario == "S4":
                reg_score[trial_id] = 1.0 if is_disc else 0.0       # prefers type-A (fails tail)
            else:
                reg_score[trial_id] = 1.0 if is_disc else 0.0       # aligned

    field = DevelopmentField(
        declared_contexts=len(_ALL_TARGETS) * N_CTX,
        construction_trial_ids=frozenset(construction_ids),
        evaluation_trial_ids=frozenset(),
    )
    for target, trial_id, label, ctx_probs in plan:
        field.add_physical_trial(target, trial_id, label, ctx_probs)

    # oracle informativeness: contexts where the trial separates best from runner-up
    for target, trial_id, label, ctx_probs in plan:
        score = 0.0
        for c in contexts_of[target]:
            row = field._contrib[trial_id][c]
            a = acc[c]
            order = np.argsort(-a)
            b0, b1 = order[0], order[1]
            if row.correct[b0] == 1 and row.correct[b1] == 0:
                score += 1.0
        oracle_score[trial_id] = score

    availability = {(t, bud): True for t in _ALL_TARGETS for bud in K.BUDGET_GRID}
    # S2's trials are exchangeable (iid): no acquisition order can beat uniform.
    return _Truth(field, availability, acc, best, contexts_of, reg_score, oracle_score,
                  structured=(scenario != "S2"))


# --------------------------------------------------------------------------- #
# Policies + evaluation (through the Semantics-B server)
# --------------------------------------------------------------------------- #
def _select(truth: _Truth, target: str, budget: int, kind: str, seed: int) -> list[str]:
    trials = [t for t in truth.field.construction_trial_ids if truth.field._target_of[t] == target]
    trials.sort()
    if budget >= len(trials):
        return trials
    # Exchangeable trials (S2): every acquisition policy is uniform.
    if kind == "passive" or not truth.structured:
        rng = np.random.default_rng(seed)
        return list(rng.choice(trials, size=budget, replace=False))
    score = truth.reg_score if kind == "registered" else truth.oracle_score
    return sorted(trials, key=lambda t: (-score[t], t))[:budget]


def _regrets(truth: _Truth, target: str, labeled: list[str]) -> dict[str, float]:
    """Per-context regret after labeling ``labeled`` physical trials (via the server)."""
    server = QueryServer(truth.field, truth.availability)
    server.open_attempt("a", target, "FULL")
    est: dict[str, np.ndarray] = {c: np.zeros(K_CAND) for c in truth.contexts_of[target]}
    n = 0
    for t in labeled:
        resp = server.query("a", t)
        for c, row in resp.contributions.items():
            est[c] = est[c] + row.correct
        n += 1
    out = {}
    for c in truth.contexts_of[target]:
        acc = truth.acc[c]
        if n == 0:
            sel = 0
        else:
            sel = int(np.argmax(est[c] / n))           # first-index tie rule
        out[c] = float(acc[truth.best[c]] - acc[sel])
    return out


def _metrics(truth: _Truth, budget: int, kind: str) -> PolicyMetrics:
    per_cohort_regrets: dict[str, list[float]] = {ck: [] for ck in COHORTS}
    all_regrets: list[float] = []
    seeds = range(PASSIVE_SEEDS) if (kind == "passive" or not truth.structured) else (0,)
    for ck, targets in COHORTS.items():
        for target in targets:
            acc_over_seeds: list[dict[str, float]] = [
                _regrets(truth, target, _select(truth, target, budget, kind, s)) for s in seeds
            ]
            contexts = truth.contexts_of[target]
            for c in contexts:
                r = float(np.mean([a[c] for a in acc_over_seeds]))
                per_cohort_regrets[ck].append(r)
                all_regrets.append(r)
    arr = np.array(all_regrets)
    k_tail = max(1, int(np.ceil(CVAR_ALPHA * arr.size)))
    tail = float(np.mean(np.sort(arr)[-k_tail:]))
    return PolicyMetrics(
        mean_regret=float(arr.mean()),
        tail_regret=tail,
        near_opt_prob=float((arr <= NEAR_OPT_EPS).mean()),
        top1_rate=float((arr <= 1e-9).mean()),
        mean_regret_by_cohort={ck: float(np.mean(v)) for ck, v in per_cohort_regrets.items()},
    )


_EXPECTED = {
    "S1": "BOUNDARY_OPERATIONALLY_CROSSED",
    "S2": "NO_REGISTERED_ACTIVE_GAIN",
    "S3": "POLICY_LIMITED",
    "S4": "BOUNDARY_WEAKENED_NOT_ROBUST",
    "S5": "BOUNDARY_OPERATIONALLY_CROSSED",
}


def run_scenario(scenario: str, seed: int = 0) -> ScenarioResult:
    truth = _build(scenario, seed)
    passive = _metrics(truth, TEST_BUDGET, "passive")
    registered = _metrics(truth, TEST_BUDGET, "registered")
    oracle = _metrics(truth, TEST_BUDGET, "oracle")
    ceiling = _metrics(truth, N_PHYS, "registered")     # FULL budget
    return ScenarioResult(
        scenario=scenario, passive=passive, registered=registered, oracle=oracle,
        ceiling=ceiling, classification=classify(passive, registered, oracle, ceiling),
        expected=_EXPECTED[scenario],
    )


def run_pilot(seed: int = 0) -> list[ScenarioResult]:
    return [run_scenario(s, seed) for s in ("S1", "S2", "S3", "S4", "S5")]
