"""Selection scoring session — score each unique checkpoint ONCE, then hand the selector a pure
dict lookup.

The selector calls its ``score_fn`` several times (ERM baseline, ``min`` over feasibles, the tie
comparison, the final write-back). If that re-triggered the leakage estimator the ERM cache counts
would be unstable. A session pre-fetches the ERM score once (one cache request per session) and
memoises every Stage-2 checkpoint by ``model_hash``, so per deletion level the ERM key sees exactly
``request=4`` (one level-prefetch + three sessions), ``compute=1``, ``hit=3``.
"""
from __future__ import annotations

from ..leakage.cache import _deep_freeze
from ..leakage.errors import LeakageNonEstimableError  # re-export (canonical definition in leakage.errors)
from ..leakage.ucb import bootstrap_ucb

__all__ = ["LeakageNonEstimableError", "SelectionScoringSession", "compute_leakage_score",
           "overlap_probe_sample_ids"]


def compute_leakage_score(feat, support_graph, fold_plan, bootstrap_plan, cfg) -> dict:
    """Replay the explicit bootstrap plan into a leakage score (point + ``bootstrap_ucl``)."""
    return bootstrap_ucb(feat, support_graph, fold_plan, cfg, bootstrap_plan=bootstrap_plan)


def overlap_probe_sample_ids(design, support_graph) -> tuple:
    """Canonical-sorted stable ids of the rows that actually enter the overlap leakage probe:
    ``{i : y_i ∈ C_cmp, d_i ∈ S_{y_i}}`` — shared by selection AND audit. Empty when there is no
    comparable class; never redefined by row count / fold / bootstrap replicate."""
    sup = {y: set(int(d) for d in support_graph.support_of_class[y]) for y in support_graph.comparable_classes}
    out = sorted(sid for sid, yy, dd in zip(design.sample_id, design.y.tolist(), design.d.tolist())
                 if int(yy) in sup and int(dd) in sup[int(yy)])
    if len(set(out)) != len(out):
        raise ValueError("duplicate overlap-probe sample id")
    return tuple(out)


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
        self._memo: dict = {}                                          # model_hash -> FULL frozen result

    def erm_result(self):
        return self._erm_score

    def erm_ucl(self) -> float:
        return float(self._erm_score["bootstrap_ucl"])

    def result(self, model_hash):
        """The full deep-frozen leakage result for a previously-scored checkpoint (point estimate,
        capacity sequence, fold/bootstrap plan hashes) — not just the scalar UCL."""
        if model_hash == self.erm_model_hash:
            return self._erm_score
        return self._memo[model_hash]

    def score(self, record) -> float:
        if record.model_hash == self.erm_model_hash:
            return self.erm_ucl()
        if record.model_hash not in self._memo:                       # estimator runs ONCE per model_hash
            self._memo[record.model_hash] = _deep_freeze(self._checkpoint_scorer(record))
        return float(self._memo[record.model_hash]["bootstrap_ucl"])

    def unique_scored_count(self) -> int:
        return len(self._memo)
