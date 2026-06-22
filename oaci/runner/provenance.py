"""Runner phase machine + fit-id provenance with deterministic (non-wall-clock) events.

The legal phase order is PREPARED -> TRAINING -> SELECTION -> SELECTION_LOCKED -> AUDIT -> COMPLETE.
Every fit (preprocess / optimization / selection / audit-estimator) is recorded as an ordered event
with the actual id set; final assertions prove the role-subset constraints and that no audit fit
happened before the selection lock.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from .keys import feed_string


class RunnerPhase(Enum):
    PREPARED = "prepared"
    TRAINING = "training"
    SELECTION = "selection"
    SELECTION_LOCKED = "selection_locked"
    AUDIT = "audit"
    COMPLETE = "complete"


_ORDER = [RunnerPhase.PREPARED, RunnerPhase.TRAINING, RunnerPhase.SELECTION,
          RunnerPhase.SELECTION_LOCKED, RunnerPhase.AUDIT, RunnerPhase.COMPLETE]


def _ids_hash(ids) -> str:
    h = hashlib.sha256()
    for s in sorted(str(i) for i in ids):
        feed_string(h, s)
    return h.hexdigest()


@dataclass(frozen=True)
class ProvenanceEvent:
    index: int
    phase: str
    kind: str
    ids_hash: str
    n_ids: int


class IllegalPhaseTransition(RuntimeError):
    pass


class RunProvenance:
    FIT_CATEGORIES = ("preprocess", "optimization", "selection", "audit_estimator", "target")

    def __init__(self):
        self.phase = RunnerPhase.PREPARED
        self.preprocess_fit_ids: set = set()
        self.optimization_fit_ids: set = set()
        self.selection_fit_ids: set = set()
        self.audit_estimator_fit_ids: set = set()
        self.target_fit_ids: set = set()
        self.ordered_events: list = []
        self.selection_locked_event_index: int | None = None
        self._counter = 0

    # ---- phase ----
    def transition(self, to_phase: RunnerPhase) -> None:
        if _ORDER.index(to_phase) != _ORDER.index(self.phase) + 1:
            raise IllegalPhaseTransition(f"{self.phase.value} -> {to_phase.value} is not allowed")
        self.phase = to_phase
        self._event(to_phase, "phase_transition", [])

    def lock_selection(self) -> None:
        self.transition(RunnerPhase.SELECTION_LOCKED)
        self.selection_locked_event_index = self.ordered_events[-1].index

    # ---- events / fits ----
    def _event(self, phase, kind, ids) -> ProvenanceEvent:
        ids = [str(i) for i in ids]
        ev = ProvenanceEvent(self._counter, phase.value, kind, _ids_hash(ids), len(set(ids)))
        self.ordered_events.append(ev)
        self._counter += 1
        return ev

    def record_fit(self, category: str, ids) -> ProvenanceEvent:
        if category not in self.FIT_CATEGORIES:
            raise ValueError(f"unknown fit category {category!r}")
        getattr(self, f"{category}_fit_ids").update(str(i) for i in ids)
        return self._event(self.phase, f"fit:{category}", ids)

    # ---- final assertions ----
    def assert_invariants(self, level0_source_train_ids, current_source_train_ids, source_audit_ids) -> None:
        l0 = set(map(str, level0_source_train_ids))
        cur = set(map(str, current_source_train_ids))
        aud = set(map(str, source_audit_ids))
        if not self.preprocess_fit_ids <= l0:
            raise ValueError("preprocess_fit_ids not subset of level-0 source_train")
        if not self.optimization_fit_ids <= cur:
            raise ValueError("optimization_fit_ids not subset of current source_train")
        if not self.selection_fit_ids <= cur:
            raise ValueError("selection_fit_ids not subset of current source_train")
        if not self.audit_estimator_fit_ids <= aud:
            raise ValueError("audit_estimator_fit_ids not subset of source_audit")
        if self.optimization_fit_ids & aud or self.selection_fit_ids & aud:
            raise ValueError("source_audit ids appear in optimization/selection fits")
        if self.target_fit_ids:
            raise ValueError("target_fit_ids must be empty")
        # every audit-estimator fit event must come AFTER the selection lock
        if self.audit_estimator_fit_ids:
            if self.selection_locked_event_index is None:
                raise ValueError("audit fits recorded but selection was never locked")
            for ev in self.ordered_events:
                if ev.kind == "fit:audit_estimator" and ev.index <= self.selection_locked_event_index:
                    raise ValueError("an audit-estimator fit happened before the selection lock")
