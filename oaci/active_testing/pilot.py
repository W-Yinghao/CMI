"""Bounded shadow pilot: pre-registered boundary taxonomy + discriminative demo.

Runs entirely on synthetic KNOWN-TRUTH scenarios through the Semantics-B query
server. Two jobs:

1. Pre-register the boundary decision rules (``classify`` + the taxonomy in
   ``constants.BOUNDARY_TAXONOMY``) BEFORE any real query.
2. Demonstrate the instrument + criteria correctly classify five ground-truth
   scenarios -- a discriminative-validity check for the active-testing regime.

Scientific units and gates (per PM review):
* The unit is the TARGET SUBJECT. Every metric is TARGET-FIRST: each target's 8
  contexts are averaged into one target regret before any cohort statistic, so the
  8 repeated contexts are never counted as 8 independent tail observations.
* Cohort tail is the upper-``TAIL_FRACTION`` CVaR over that cohort's TARGET
  regrets.
* BOUNDARY_OPERATIONALLY_CROSSED requires, in EVERY cohort, simultaneous
  improvement in mean AND tail AND near-optimal probability. Any cohort failing any
  gate -> BOUNDARY_WEAKENED_NOT_ROBUST.
* The FULL-budget ceiling gate is multi-endpoint per cohort (mean, tail, near-opt),
  not mean alone.

No real data is opened. The active policy here is a static acquisition-score
surrogate; real C86D would use LURE / Hara adaptive scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as K
from .field import DevelopmentField
from .query_server import QueryServer

TAIL_FRACTION = 0.25       # tail = mean of the worst 25% of TARGET regrets (upper-loss)
NEAR_OPT_EPS = 0.05        # a target is "near-optimal" if its regret <= this
# PRECISE endpoint definition (carried forward to C86D/C86H): target_near_opt_prob is
#   P( a target's 8-context MEAN regret <= NEAR_OPT_EPS ).
# It is NOT the per-context P( selected context-action in A_eps ). If that per-context
# quantity is ever reported it must be a SEPARATELY pre-defined endpoint, never conflated.
NEAR_OPT_DEFINITION = "P(target 8-context mean regret <= eps)"
NEAROPT_MARGIN = 0.05      # near-opt probability improvement must beat this
CEILING_NEAROPT_MIN = 0.9  # FULL ceiling must reach at least this near-opt per cohort
TEST_BUDGET = 4            # small-budget operating point
N_PHYS = 32                # physical trials per target (FULL == 32)
N_CTX = 8                  # contexts per physical trial (production-equivalent)
K_CAND = 16                # candidates per context
PASSIVE_SEEDS = 6          # average passive uniform over this many draws

TARGETS_PER_COHORT = 8
COHORTS = {
    "cohortA": tuple(f"T{i}" for i in range(TARGETS_PER_COHORT)),
    "cohortB": tuple(f"T{i}" for i in range(TARGETS_PER_COHORT, 2 * TARGETS_PER_COHORT)),
}
_ALL_TARGETS = tuple(t for ts in COHORTS.values() for t in ts)
_TAIL_TARGETS = frozenset(t for ts in COHORTS.values() for t in ts[-2:])   # 2 tail targets / cohort


@dataclass(frozen=True)
class PolicyMetrics:
    mean_regret: float                       # over all targets
    tail_regret: float                       # CVaR over all target regrets
    target_near_opt_prob: float
    top1_rate: float
    mean_regret_by_cohort: dict[str, float]
    tail_regret_by_cohort: dict[str, float]
    target_near_opt_prob_by_cohort: dict[str, float]


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    passive: PolicyMetrics
    registered: PolicyMetrics
    oracle: PolicyMetrics
    ceiling: PolicyMetrics
    classification: str
    expected: str


def _cvar(regrets: np.ndarray, frac: float = TAIL_FRACTION) -> float:
    """Mean of the worst ``frac`` fraction of regrets (upper-loss tail)."""
    if regrets.size == 0:
        return 0.0
    k = max(1, int(np.ceil(frac * regrets.size)))
    return float(np.mean(np.sort(regrets)[-k:]))


# --------------------------------------------------------------------------- #
# Pre-registered classifier (frozen decision rules)
# --------------------------------------------------------------------------- #
def classify(passive: PolicyMetrics, registered: PolicyMetrics, oracle: PolicyMetrics,
             ceiling: PolicyMetrics) -> str:
    """Map measured policy metrics to a pre-registered boundary taxonomy label."""
    m = K.TAU_REGRET_MARGIN
    cohorts = passive.mean_regret_by_cohort.keys()

    # FULL-budget acquisition view must transport in EVERY cohort on every endpoint.
    ceil_ok = all(
        ceiling.mean_regret_by_cohort[c] <= K.TAU_CEILING_REGRET
        and ceiling.tail_regret_by_cohort[c] <= K.TAU_CEILING_REGRET
        and ceiling.target_near_opt_prob_by_cohort[c] >= CEILING_NEAROPT_MIN
        for c in cohorts
    )
    reg_gain = (passive.mean_regret - registered.mean_regret) >= m          # overall (target-pooled)
    oracle_gain = (passive.mean_regret - oracle.mean_regret) >= m

    if not ceil_ok:
        return "ACQUISITION_VIEW_NONTRANSPORTABLE"
    if not reg_gain:
        return "POLICY_LIMITED" if oracle_gain else "NO_REGISTERED_ACTIVE_GAIN"

    crossed_every_cohort = all(
        (passive.mean_regret_by_cohort[c] - registered.mean_regret_by_cohort[c]) >= m
        and (passive.tail_regret_by_cohort[c] - registered.tail_regret_by_cohort[c]) >= m
        and (registered.target_near_opt_prob_by_cohort[c] - passive.target_near_opt_prob_by_cohort[c]) >= NEAROPT_MARGIN
        for c in cohorts
    )
    return "BOUNDARY_OPERATIONALLY_CROSSED" if crossed_every_cohort else "BOUNDARY_WEAKENED_NOT_ROBUST"


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
    plan: list[tuple] = []
    acc: dict = {}
    best: dict = {}
    contexts_of: dict = {}
    reg_score: dict = {}
    oracle_score: dict = {}

    for target in _ALL_TARGETS:
        contexts = [f"{target}/c{ci}" for ci in range(N_CTX)]
        contexts_of[target] = contexts
        is_tail_target = target in _TAIL_TARGETS

        for c in contexts:
            a = np.full(K_CAND, 0.6)
            b = int(rng.integers(0, K_CAND))
            if scenario == "S5":                       # dense near-optimal cluster
                cluster = rng.choice(K_CAND, size=5, replace=False)
                a[cluster] = [0.86, 0.87, 0.88, 0.89, 0.90]
                b = int(cluster[np.argmax(a[cluster])])
            elif scenario == "S2":                     # one clear best in a crowd (sample-size limited)
                a[:] = 0.5                              # at small budget crowd candidates tie best by luck;
                a[b] = 1.0                              # FULL isolates the best deterministically (no acquisition helps)
            else:                                      # S1/S3/S4 big margin
                a[b] = 0.90
            acc[c] = a
            best[c] = int(np.argmax(a))

        for j in range(N_PHYS):
            trial_id = f"{target}-t{j}"
            construction_ids.append(trial_id)
            label = int(rng.integers(0, 2))
            is_disc = j < 4                              # first 4 physical trials are informative
            ctx_probs = {}
            for c in contexts:
                a = acc[c]
                if scenario == "S2":                     # iid, sample-size-limited
                    correct = (rng.random(K_CAND) < a).astype(np.int64)
                else:                                    # deterministic threshold
                    d = 0.75 if is_disc else 0.05
                    correct = (a >= d).astype(np.int64)
                ctx_probs[c] = _make_probs(correct, label)
            plan.append((target, trial_id, label, ctx_probs))

            # registered acquisition score (aligned / anti-aligned / flat)
            if scenario == "S2":
                reg_score[trial_id] = float(rng.random())
            elif scenario == "S3":
                reg_score[trial_id] = 0.0 if is_disc else 1.0             # anti-aligned everywhere
            elif scenario == "S4":                                        # tail TARGETS misdirected
                aligned = not is_tail_target
                reg_score[trial_id] = (1.0 if is_disc else 0.0) if aligned else (0.0 if is_disc else 1.0)
            else:
                reg_score[trial_id] = 1.0 if is_disc else 0.0            # aligned

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
            order = np.argsort(-acc[c])
            if row.correct[order[0]] == 1 and row.correct[order[1]] == 0:
                score += 1.0
        oracle_score[trial_id] = score

    availability = {(t, bud): True for t in _ALL_TARGETS for bud in K.BUDGET_GRID}
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
    if kind == "passive" or not truth.structured:       # exchangeable trials => uniform
        # seed PER (target, seed) so passive draws are independent across targets
        # (otherwise every target picks the same positions and target-first averaging degenerates).
        rng = np.random.default_rng(seed * 100_003 + _ALL_TARGETS.index(target))
        return list(rng.choice(trials, size=budget, replace=False))
    score = truth.reg_score if kind == "registered" else truth.oracle_score
    return sorted(trials, key=lambda t: (-score[t], t))[:budget]


def _context_regrets(truth: _Truth, target: str, labeled: list[str]) -> list[float]:
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
    out = []
    for c in truth.contexts_of[target]:
        sel = 0 if n == 0 else int(np.argmax(est[c] / n))     # first-index tie rule
        out.append(float(truth.acc[c][truth.best[c]] - truth.acc[c][sel]))
    return out


def _target_regret(truth: _Truth, target: str, budget: int, kind: str, seed: int) -> float:
    """One target's regret = mean over its 8 contexts (target-first aggregation)."""
    return float(np.mean(_context_regrets(truth, target, _select(truth, target, budget, kind, seed))))


def _metrics(truth: _Truth, budget: int, kind: str) -> PolicyMetrics:
    seeds = range(PASSIVE_SEEDS) if (kind == "passive" or not truth.structured) else (0,)
    by_cohort: dict[str, list[float]] = {ck: [] for ck in COHORTS}
    for ck, targets in COHORTS.items():
        for target in targets:
            r = float(np.mean([_target_regret(truth, target, budget, kind, s) for s in seeds]))
            by_cohort[ck].append(r)                       # one regret per TARGET
    all_r = np.array([r for v in by_cohort.values() for r in v])
    cohort_arr = {ck: np.array(v) for ck, v in by_cohort.items()}
    return PolicyMetrics(
        mean_regret=float(all_r.mean()),
        tail_regret=_cvar(all_r),
        target_near_opt_prob=float((all_r <= NEAR_OPT_EPS).mean()),
        top1_rate=float((all_r <= 1e-9).mean()),
        mean_regret_by_cohort={ck: float(a.mean()) for ck, a in cohort_arr.items()},
        tail_regret_by_cohort={ck: _cvar(a) for ck, a in cohort_arr.items()},
        target_near_opt_prob_by_cohort={ck: float((a <= NEAR_OPT_EPS).mean()) for ck, a in cohort_arr.items()},
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
