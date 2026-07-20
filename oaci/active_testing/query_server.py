"""The production query-server contract (Semantics B).

One query names one *physical* construction trial.  It reveals that trial's single
label and its per-context linear contribution rows (one per context the trial
belongs to) — nothing about any other trial.  The budget counts physical labels
per target, so a label is billed once regardless of how many contexts it informs.
The bulk label / contribution store is never exposed.  This is the interface a
future C86D active dispatcher must use; C86LP implements and shadow-tests it but
runs no registered active policy in production.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import constants as K
from .contribution import ContributionRow
from .field import DevelopmentField


class C86LPQueryError(RuntimeError):
    """Raised on an invalid query (duplicate / unknown / budget-exhausted / cross-target)."""


class C86LPInputUnavailable(RuntimeError):
    """Raised for an unsupported target-budget cell — no substitution is permitted."""


@dataclass(frozen=True)
class QueryReceipt:
    attempt_id: str
    target: str
    trial_id: str
    budget: int | str
    query_index: int                         # 1-based physical-label index within the attempt


@dataclass(frozen=True)
class QueryResponse:
    trial_id: str
    true_label: int
    contributions: dict[str, ContributionRow]  # context -> linear contribution row


class QueryServer:
    """Serves one physical-label query at a time against a sealed development field.

    ``budget_availability`` maps (target, budget) -> available bool.  An
    unsupported cell raises ``C86LPInputUnavailable`` at ``open_attempt`` time;
    there is no replacement sampling, budget substitution, or target deletion.
    ``budget`` is a per-target PHYSICAL-LABEL count (``"FULL"`` = every construction
    trial of the target).
    """

    def __init__(self, field: DevelopmentField, budget_availability: dict[tuple[str, object], bool]):
        # Name-mangled so the client handle cannot reach the bulk stores.  (This is
        # logical/API isolation, not a process/filesystem boundary.)
        self.__field = field
        self.__availability = dict(budget_availability)
        self.__attempts: dict[str, dict] = {}
        self.__receipts: list[QueryReceipt] = []

    def _target_trials(self, target: str) -> list[str]:
        return [t for t in self.__field.construction_trial_ids
                if self.__field._target_of.get(t) == target]

    # --- attempt lifecycle ----------------------------------------------------
    def open_attempt(self, attempt_id: str, target: str, budget: int | str) -> str:
        if budget not in K.BUDGET_GRID:
            raise C86LPQueryError(f"budget {budget!r} not in the locked grid {K.BUDGET_GRID}")
        if not self.__availability.get((target, budget), False):
            raise C86LPInputUnavailable(
                f"{K.UNSUPPORTED_BUDGET_DISPOSITION}: target {target!r} budget {budget!r} "
                "is unavailable; no replacement / substitution / deletion"
            )
        if attempt_id in self.__attempts:
            raise C86LPQueryError(f"attempt {attempt_id!r} already open")
        cap = len(self._target_trials(target)) if budget == "FULL" else int(budget)
        self.__attempts[attempt_id] = {
            "target": target, "budget": budget, "cap": cap, "queried": set(), "n": 0,
        }
        return attempt_id

    # --- the single physical-label query --------------------------------------
    def query(self, attempt_id: str, trial_id: str) -> QueryResponse:
        attempt = self.__attempts.get(attempt_id)
        if attempt is None:
            raise C86LPQueryError(f"unknown attempt {attempt_id!r}")
        if trial_id not in self.__field._labels:
            raise C86LPQueryError(f"unknown trial {trial_id!r}")
        if self.__field._target_of.get(trial_id) != attempt["target"]:
            raise C86LPQueryError(f"trial {trial_id!r} is not in target {attempt['target']!r}")
        if trial_id in attempt["queried"]:
            raise C86LPQueryError(f"duplicate query for trial {trial_id!r}")
        if attempt["n"] >= attempt["cap"]:
            raise C86LPQueryError(
                f"budget exhausted for attempt {attempt_id!r} (cap {attempt['cap']})"
            )
        attempt["queried"].add(trial_id)
        attempt["n"] += 1
        self.__receipts.append(QueryReceipt(
            attempt_id=attempt_id, target=attempt["target"], trial_id=trial_id,
            budget=attempt["budget"], query_index=attempt["n"],
        ))
        return QueryResponse(
            trial_id=trial_id,
            true_label=self.__field._oracle_label(trial_id),
            contributions=dict(self.__field._contrib[trial_id]),
        )

    # --- read-only introspection (no bulk data escapes) -----------------------
    def remaining(self, attempt_id: str) -> int:
        attempt = self.__attempts[attempt_id]
        return attempt["cap"] - attempt["n"]

    @property
    def receipts(self) -> tuple[QueryReceipt, ...]:
        return tuple(self.__receipts)
