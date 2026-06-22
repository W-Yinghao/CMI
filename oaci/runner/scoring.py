"""Selection scoring session — score each unique checkpoint ONCE, then hand the selector a pure
dict lookup.

The selector calls its ``score_fn`` several times (ERM baseline, ``min`` over feasibles, the tie
comparison, the final write-back). If that re-triggered the leakage estimator the ERM cache counts
would be unstable. A session pre-fetches the ERM score once (one cache request per session) and
memoises every Stage-2 checkpoint by ``model_hash``, so per deletion level the ERM key sees exactly
``request=4`` (one level-prefetch + three sessions), ``compute=1``, ``hit=3``.
"""
from __future__ import annotations

from ..leakage.ucb import bootstrap_ucb


class LeakageNonEstimableError(Exception):
    """A selection leakage score is structurally non-estimable (no comparable class / no valid
    grouped folds / too few accepted bootstrap draws). NOT for numerical/probe failures, which must
    propagate."""


def compute_leakage_score(feat, support_graph, fold_plan, bootstrap_plan, cfg) -> dict:
    """Replay the explicit bootstrap plan into a leakage score (point + ``bootstrap_ucl``)."""
    return bootstrap_ucb(feat, support_graph, fold_plan, cfg, bootstrap_plan=bootstrap_plan)


class SelectionScoringSession:
    """One per Stage-2 method. ``erm_scorer``/``checkpoint_scorer`` return a leakage dict (with a
    ``bootstrap_ucl`` key). The ERM score is requested ONCE at construction (a cache hit if the
    level already computed it); ``score`` never re-triggers the estimator for a repeated record."""

    def __init__(self, method_name, cache, erm_key, erm_model_hash, erm_scorer, checkpoint_scorer):
        self.method_name = method_name
        self.cache = cache
        self.erm_key = erm_key
        self.erm_model_hash = erm_model_hash
        self._checkpoint_scorer = checkpoint_scorer
        self._erm_score = cache.get_or_compute(erm_key, erm_scorer)   # the ONE ERM request per session
        self._memo: dict = {}

    def erm_ucl(self) -> float:
        return float(self._erm_score["bootstrap_ucl"])

    def score(self, record) -> float:
        if record.model_hash == self.erm_model_hash:
            return self.erm_ucl()
        if record.model_hash not in self._memo:
            self._memo[record.model_hash] = float(self._checkpoint_scorer(record)["bootstrap_ucl"])
        return self._memo[record.model_hash]

    def unique_scored_count(self) -> int:
        return len(self._memo)
