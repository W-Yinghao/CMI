"""Development-field objects (Semantics B: per-target physical labels).

A physical C84 construction trial appears in several contexts (in the real field,
2 panels x 2 seeds x 2 levels = 8), each with a *different* 81-candidate
probability matrix and therefore a different contribution vector.  The label,
however, is a property of the physical trial: one physical label query reveals one
label and derives one contribution row *per context* the trial belongs to.  The
query budget is the number of physical labels per target — so a single physical
label is never double-billed across its repeated contexts.

Isolation here is LOGICAL / API only: the three stores live in one Python object
and the server reaches them via name mangling.  Real C86L would use separate
processes and filesystem roots (see ``constants.ISOLATION_LEVEL``).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Mapping

import numpy as np

from . import constants as K
from .contribution import ContributionRow, compute_contribution


class C86LPFieldError(RuntimeError):
    """Raised when a development-field isolation or coverage invariant is violated."""


# Fields an unlabeled-pool row is allowed to carry.  Anything outside this set is
# an isolation breach (labels / responses / metrics / selections / utilities).
_ALLOWED_POOL_FIELDS = frozenset({
    "dataset", "target", "context", "trial_id",
    "candidate_ids", "candidate_probs", "hard_preds", "confidence",
    "session_run", "input_digest",
})
_FORBIDDEN_POOL_FIELDS = frozenset({
    "true_label", "label", "queried_response", "construction_metric",
    "selected_action", "c85u_utility", "held_utility", "outcome",
})


@dataclass(frozen=True)
class UnlabeledPoolRow:
    """One (context, physical-trial) row visible to the active client — no label."""

    dataset: str
    target: str
    context: str
    trial_id: str
    candidate_ids: tuple[str, ...]
    candidate_probs: np.ndarray             # [K, 2]
    hard_preds: np.ndarray                  # [K]
    confidence: np.ndarray                  # [K]
    session_run: str
    input_digest: str

    def __post_init__(self) -> None:
        seen = set(self.__dataclass_fields__)
        leak = seen & _FORBIDDEN_POOL_FIELDS
        if leak:
            raise C86LPFieldError(f"unlabeled pool row leaks forbidden fields: {sorted(leak)}")
        if not seen <= _ALLOWED_POOL_FIELDS:
            raise C86LPFieldError(f"unlabeled pool row has unexpected fields: {sorted(seen - _ALLOWED_POOL_FIELDS)}")


@dataclass(frozen=True)
class LabelOracleRow:
    dataset: str
    target: str
    trial_id: str
    label: int
    construction_view_identity: str


@dataclass
class DevelopmentField:
    """Client-visible pool + server-private per-context contribution store.

    Keys everything by the *physical* trial id; ``_contrib[trial_id]`` maps each
    context of that trial to its own contribution row.
    """

    declared_contexts: int
    construction_trial_ids: frozenset[str]
    evaluation_trial_ids: frozenset[str]
    pool: list[UnlabeledPoolRow] = dc_field(default_factory=list)
    _labels: dict[str, int] = dc_field(default_factory=dict, repr=False)
    _contrib: dict[str, dict[str, ContributionRow]] = dc_field(default_factory=dict, repr=False)
    _target_of: dict[str, str] = dc_field(default_factory=dict, repr=False)
    c85u_identity: str = K.FROZEN_INPUT_SHA["c85u_acceptance_manifest"]

    def __post_init__(self) -> None:
        overlap = self.construction_trial_ids & self.evaluation_trial_ids
        if overlap:
            raise C86LPFieldError(
                f"construction/evaluation overlap must be zero; got {len(overlap)} shared IDs"
            )

    @property
    def observed_contexts(self) -> int:
        return len({r.context for r in self.pool})

    def contexts_of(self, trial_id: str) -> tuple[str, ...]:
        return tuple(self._contrib.get(trial_id, {}).keys())

    def add_physical_trial(
        self, target: str, trial_id: str, label: int,
        context_probs: Mapping[str, np.ndarray], *,
        construction_view_identity: str = "shadow-construction-view",
        dataset: str = "ShadowDS",
    ) -> None:
        """Register one physical construction trial across all its contexts.

        ``context_probs`` maps context -> [K, 2] candidate probability matrix.  The
        single ``label`` is shared across contexts; each context gets its own
        contribution row.
        """
        if trial_id not in self.construction_trial_ids:
            raise C86LPFieldError(f"{trial_id!r} is not a construction (acquisition-pool) trial")
        if trial_id in self.evaluation_trial_ids:
            raise C86LPFieldError(f"{trial_id!r} is an evaluation trial; construction pool only")
        if trial_id in self._labels:
            raise C86LPFieldError(f"{trial_id!r} already registered")
        rows: dict[str, ContributionRow] = {}
        for context, probs in context_probs.items():
            probs = np.asarray(probs, dtype=np.float64)
            candidate_ids = tuple(f"cand{k}" for k in range(probs.shape[0]))
            self.pool.append(UnlabeledPoolRow(
                dataset=dataset, target=target, context=context, trial_id=trial_id,
                candidate_ids=candidate_ids, candidate_probs=probs,
                hard_preds=np.argmax(probs, axis=1), confidence=np.max(probs, axis=1),
                session_run="shadow-s0-r0", input_digest="shadow",
            ))
            rows[context] = compute_contribution(trial_id, label, probs)
        self._labels[trial_id] = int(label)
        self._contrib[trial_id] = rows
        self._target_of[trial_id] = target

    # exposed for the server (name-mangled access) and metrics; not a bulk dump
    def _oracle_label(self, trial_id: str) -> int:
        return self._labels[trial_id]


@dataclass(frozen=True)
class DevelopmentFieldManifest:
    declared_contexts: int
    observed_contexts: int
    n_construction_trials: int
    isolation_level: str
    frozen_input_sha: dict
    frozen_gates: dict
    c85u_identity: str
    c85u_utility_values_accessed: int
    coverage_complete: bool

    def publish(self) -> str:
        """Return the instrument gate, or raise the reconciliation gate on incomplete coverage."""
        if not self.coverage_complete:
            raise C86LPFieldError(
                f"{K.GATE_RECONCILIATION}: partial field cannot publish "
                f"({self.observed_contexts}/{self.declared_contexts} contexts)"
            )
        if self.c85u_utility_values_accessed != 0:
            raise C86LPFieldError("held C85U utility values were accessed; isolation breach")
        if self.frozen_gates != K.FROZEN_GATES:
            raise C86LPFieldError("frozen C84/C85 gates were mutated; C86L must not change them")
        return K.GATE_INSTRUMENT


def build_manifest(fieldobj: DevelopmentField) -> DevelopmentFieldManifest:
    complete = fieldobj.observed_contexts == fieldobj.declared_contexts
    return DevelopmentFieldManifest(
        declared_contexts=fieldobj.declared_contexts,
        observed_contexts=fieldobj.observed_contexts,
        n_construction_trials=len(fieldobj._labels),
        isolation_level=K.ISOLATION_LEVEL,
        frozen_input_sha=dict(K.FROZEN_INPUT_SHA),
        frozen_gates=dict(K.FROZEN_GATES),
        c85u_identity=fieldobj.c85u_identity,
        c85u_utility_values_accessed=0,  # this package has no code path to read them
        coverage_complete=complete,
    )
