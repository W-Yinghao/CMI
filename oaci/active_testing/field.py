"""Physically-separate development-field objects and the field manifest.

Three stores with a strict information gradient (mirrors the C86L development-view
contract):

* ``UnlabeledPool``  — client-visible.  Trial identity + candidate probabilities /
  hard predictions / confidence only.  Carries NO label, queried response,
  construction metric, selected action, or C85U utility.
* ``LabelOracle``    — server-private.  Only {dataset, target, trial_id, label}.
  Never exposed to the active client as a bulk file.
* held development outcome (C85U utility field) — bound by identity only; this
  package never copies, opens, or summarizes its values.

Construction (acquisition) trial IDs and evaluation trial IDs are disjoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

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
    """A built (shadow) development field: client-visible pool + sealed label/contribution stores.

    ``declared_contexts`` is the number of contexts the field claims to cover; the
    manifest refuses to publish unless every declared context is observed (a partial
    field cannot publish).
    """

    declared_contexts: int
    construction_trial_ids: frozenset[str]
    evaluation_trial_ids: frozenset[str]
    pool: list[UnlabeledPoolRow] = dc_field(default_factory=list)
    # server-private stores (never handed to the client directly)
    _oracle: dict[str, LabelOracleRow] = dc_field(default_factory=dict, repr=False)
    _contrib: dict[str, ContributionRow] = dc_field(default_factory=dict, repr=False)
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

    def add_trial(self, row: UnlabeledPoolRow, *, true_label: int,
                  construction_view_identity: str) -> None:
        """Register one construction trial into all three stores at once."""
        if row.trial_id not in self.construction_trial_ids:
            raise C86LPFieldError(f"{row.trial_id!r} is not a construction (acquisition-pool) trial")
        if row.trial_id in self.evaluation_trial_ids:
            raise C86LPFieldError(f"{row.trial_id!r} is an evaluation trial; construction pool only")
        self.pool.append(row)
        self._oracle[row.trial_id] = LabelOracleRow(
            dataset=row.dataset, target=row.target, trial_id=row.trial_id,
            label=int(true_label), construction_view_identity=construction_view_identity,
        )
        self._contrib[row.trial_id] = compute_contribution(row.trial_id, true_label, row.candidate_probs)


@dataclass(frozen=True)
class DevelopmentFieldManifest:
    declared_contexts: int
    observed_contexts: int
    n_construction_trials: int
    frozen_input_sha: dict
    frozen_gates: dict
    c85u_identity: str
    c85u_utility_values_accessed: int
    coverage_complete: bool

    def publish(self) -> str:
        """Return the success gate, or raise the reconciliation gate on incomplete coverage."""
        if not self.coverage_complete:
            raise C86LPFieldError(
                "C86L_CONSTRUCTION_VIEW_PREDICTION_ALIGNMENT_QUERY_INTERFACE_OR_"
                "PROVENANCE_RECONCILIATION_REQUIRED: partial field cannot publish "
                f"({self.observed_contexts}/{self.declared_contexts} contexts)"
            )
        if self.c85u_utility_values_accessed != 0:
            raise C86LPFieldError("held C85U utility values were accessed; isolation breach")
        if self.frozen_gates != K.FROZEN_GATES:
            raise C86LPFieldError("frozen C84/C85 gates were mutated; C86L must not change them")
        return (
            "C86L_C84_CONSTRUCTION_POOL_TRIAL_CONTRIBUTION_FIELD_"
            "IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION"
        )


def build_manifest(fieldobj: DevelopmentField) -> DevelopmentFieldManifest:
    complete = fieldobj.observed_contexts == fieldobj.declared_contexts
    return DevelopmentFieldManifest(
        declared_contexts=fieldobj.declared_contexts,
        observed_contexts=fieldobj.observed_contexts,
        n_construction_trials=len(fieldobj._oracle),
        frozen_input_sha=dict(K.FROZEN_INPUT_SHA),
        frozen_gates=dict(K.FROZEN_GATES),
        c85u_identity=fieldobj.c85u_identity,
        c85u_utility_values_accessed=0,  # this package has no code path to read them
        coverage_complete=complete,
    )
