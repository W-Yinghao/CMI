"""Pure fixed-panel source-support deletion for C84 level 1."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

from . import c84l1_protocols as protocol


class C84L1InterventionError(RuntimeError):
    """Raised when the registered deletion cannot be applied exactly."""


def _canonical_label(value: Any) -> int:
    if isinstance(value, str):
        mapping = {"left_hand": 0, "right_hand": 1}
        if value not in mapping:
            raise C84L1InterventionError(f"unregistered source class: {value!r}")
        return mapping[value]
    label = int(value)
    if label not in (0, 1):
        raise C84L1InterventionError(f"unregistered source class ID: {value!r}")
    return label


def _digest(values: Any) -> str:
    return hashlib.sha256(protocol.canonical_bytes(values)).hexdigest()


def _exact_subject_order(dataset: str, panel: str) -> tuple[int, ...]:
    rows = {
        (row["dataset"], row["panel"]): row
        for row in protocol.source_contract_rows()
    }
    try:
        row = rows[(dataset, panel)]
    except KeyError as exc:
        raise C84L1InterventionError(f"dataset/panel outside the locked registry: {dataset}/{panel}") from exc
    return tuple(int(value) for value in row["source_training_subjects"].split("|"))


def registered_deleted_cell(dataset: str, panel: str) -> tuple[int, int]:
    try:
        subject = protocol.DELETED_SUBJECTS[(dataset, panel)]
    except KeyError as exc:
        raise C84L1InterventionError(f"no registered deletion cell for {dataset}/{panel}") from exc
    order = _exact_subject_order(dataset, panel)
    if not order or order[0] != subject:
        raise C84L1InterventionError("registered deletion subject is not first in locked source order")
    return int(subject), protocol.DELETED_CLASS_ID


@dataclass(frozen=True)
class InterventionApplication:
    dataset: str
    panel: str
    level: int
    level_intervention_id: str
    deleted_source_subject: int | None
    deleted_class: str | None
    keep_indices: tuple[int, ...]
    deleted_indices: tuple[int, ...]
    pre_trial_count: int
    post_trial_count: int
    pre_cell_counts: tuple[tuple[int, int, int], ...]
    post_cell_counts: tuple[tuple[int, int, int], ...]
    deleted_trial_id_sha256: str
    population_signature_sha256: str
    support_graph_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "panel": self.panel,
            "level": self.level,
            "level_intervention_id": self.level_intervention_id,
            "deleted_source_subject": self.deleted_source_subject,
            "deleted_class": self.deleted_class,
            "keep_indices": list(self.keep_indices),
            "deleted_indices": list(self.deleted_indices),
            "pre_trial_count": self.pre_trial_count,
            "post_trial_count": self.post_trial_count,
            "pre_cell_counts": [list(row) for row in self.pre_cell_counts],
            "post_cell_counts": [list(row) for row in self.post_cell_counts],
            "deleted_trial_id_sha256": self.deleted_trial_id_sha256,
            "population_signature_sha256": self.population_signature_sha256,
            "support_graph_sha256": self.support_graph_sha256,
        }


def _counts(subjects: Sequence[int], labels: Sequence[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(subjects, labels))


def _count_rows(counts: Mapping[tuple[int, int], int]) -> tuple[tuple[int, int, int], ...]:
    return tuple((subject, label, int(count)) for (subject, label), count in sorted(counts.items()))


def _assert_protected_ids_unchanged(
    name: str,
    before: Iterable[Any] | None,
    after: Iterable[Any] | None,
) -> None:
    if before is None and after is None:
        return
    if before is None or after is None:
        raise C84L1InterventionError(f"{name} before/after identity must be supplied together")
    if tuple(map(str, before)) != tuple(map(str, after)):
        raise C84L1InterventionError(f"{name} was changed by the source-training intervention")


def apply_level_intervention(
    *,
    dataset: str,
    panel: str,
    level: int,
    source_subjects: Sequence[Any],
    source_labels: Sequence[Any],
    source_trial_ids: Sequence[Any],
    requested_deleted_subject: int | None = None,
    requested_deleted_class: str | None = None,
    target_dependent_choice: bool = False,
    outcome_dependent_choice: bool = False,
    source_audit_trial_ids_before: Iterable[Any] | None = None,
    source_audit_trial_ids_after: Iterable[Any] | None = None,
    target_trial_ids_before: Iterable[Any] | None = None,
    target_trial_ids_after: Iterable[Any] | None = None,
) -> InterventionApplication:
    """Validate and apply the unique registered level intervention.

    The function returns source-training row indices only. It cannot mutate or
    inspect source-audit rows, target arrays, or target labels.
    """
    if level not in (0, 1):
        raise C84L1InterventionError(f"unsupported C84 level: {level}")
    if target_dependent_choice or outcome_dependent_choice:
        raise C84L1InterventionError("target- or outcome-dependent deletion is forbidden")
    subjects = tuple(int(value) for value in source_subjects)
    labels = tuple(_canonical_label(value) for value in source_labels)
    trial_ids = tuple(str(value) for value in source_trial_ids)
    if not subjects or len(subjects) != len(labels) or len(labels) != len(trial_ids):
        raise C84L1InterventionError("source subjects, labels and trial IDs must be nonempty and aligned")
    if len(set(trial_ids)) != len(trial_ids):
        raise C84L1InterventionError("source-training trial IDs are not unique")
    expected_order = _exact_subject_order(dataset, panel)
    if set(subjects) != set(expected_order):
        raise C84L1InterventionError("source-training subject set differs from the locked 12-subject panel")
    pre = _counts(subjects, labels)
    expected_cells = {(subject, label) for subject in expected_order for label in (0, 1)}
    if set(pre) != expected_cells:
        raise C84L1InterventionError("pre-deletion support does not contain all 24 domain-by-class cells")
    if any(pre[cell] < protocol.MIN_CELL_SUPPORT for cell in expected_cells):
        raise C84L1InterventionError("a pre-deletion support cell is below the locked minimum of 8")
    _assert_protected_ids_unchanged(
        "source-audit trial IDs", source_audit_trial_ids_before, source_audit_trial_ids_after,
    )
    _assert_protected_ids_unchanged("target trial IDs", target_trial_ids_before, target_trial_ids_after)

    if level == 0:
        if requested_deleted_subject is not None or requested_deleted_class is not None:
            raise C84L1InterventionError("level 0 cannot request a deletion cell")
        keep = tuple(range(len(trial_ids)))
        deleted: tuple[int, ...] = ()
        post = pre
        intervention_id = protocol.LEVEL0_ID
        deleted_subject = None
        deleted_class = None
    else:
        deleted_subject, deleted_label = registered_deleted_cell(dataset, panel)
        if requested_deleted_subject is not None and int(requested_deleted_subject) != deleted_subject:
            raise C84L1InterventionError("runtime attempted to substitute the registered deleted subject")
        if requested_deleted_class is not None and requested_deleted_class != protocol.DELETED_CLASS:
            raise C84L1InterventionError("runtime attempted to substitute the registered deleted class")
        deleted = tuple(
            index for index, (subject, label) in enumerate(zip(subjects, labels))
            if subject == deleted_subject and label == deleted_label
        )
        if len(deleted) < protocol.MIN_CELL_SUPPORT:
            raise C84L1InterventionError("registered deleted cell is absent or below support minimum")
        deleted_set = set(deleted)
        keep = tuple(index for index in range(len(trial_ids)) if index not in deleted_set)
        post = _counts(tuple(subjects[index] for index in keep), tuple(labels[index] for index in keep))
        expected_post = expected_cells - {(deleted_subject, deleted_label)}
        if set(post) != expected_post or len(post) != 23:
            raise C84L1InterventionError("post-deletion support is not exactly the registered 23-cell graph")
        if any(post[cell] < protocol.MIN_CELL_SUPPORT for cell in expected_post):
            raise C84L1InterventionError("a retained support cell is below the locked minimum of 8")
        if post[(deleted_subject, 1)] < protocol.MIN_CELL_SUPPORT:
            raise C84L1InterventionError("deleted subject does not retain minimum right-hand support")
        intervention_id = protocol.LEVEL1_ID
        deleted_class = protocol.DELETED_CLASS

    kept_trial_ids = tuple(trial_ids[index] for index in keep)
    expected_kept = tuple(value for index, value in enumerate(trial_ids) if index not in set(deleted))
    if kept_trial_ids != expected_kept or len(set(kept_trial_ids)) != len(kept_trial_ids):
        raise C84L1InterventionError("source-training trial identity changed beyond the registered deletion")
    deleted_trial_ids = tuple(trial_ids[index] for index in deleted)
    population = {
        "dataset": dataset,
        "panel": panel,
        "level": level,
        "intervention": intervention_id,
        "rows": [
            [trial_ids[index], subjects[index], labels[index]]
            for index in keep
        ],
    }
    support = {
        "dataset": dataset,
        "panel": panel,
        "level": level,
        "cells": [list(row) for row in _count_rows(post)],
    }
    return InterventionApplication(
        dataset=dataset,
        panel=panel,
        level=level,
        level_intervention_id=intervention_id,
        deleted_source_subject=deleted_subject,
        deleted_class=deleted_class,
        keep_indices=keep,
        deleted_indices=deleted,
        pre_trial_count=len(trial_ids),
        post_trial_count=len(keep),
        pre_cell_counts=_count_rows(pre),
        post_cell_counts=_count_rows(post),
        deleted_trial_id_sha256=_digest(list(deleted_trial_ids)),
        population_signature_sha256=_digest(population),
        support_graph_sha256=_digest(support),
    )


def synthetic_source_panel(dataset: str, panel: str, rows_per_cell: int = 8) -> dict[str, tuple[Any, ...]]:
    """Create a deterministic schema-only source panel for tests."""
    if rows_per_cell < protocol.MIN_CELL_SUPPORT:
        raise ValueError("synthetic panel cannot undercut the registered minimum")
    subjects: list[int] = []
    labels: list[int] = []
    trial_ids: list[str] = []
    for subject in _exact_subject_order(dataset, panel):
        for label in (0, 1):
            for row in range(rows_per_cell):
                subjects.append(subject)
                labels.append(label)
                trial_ids.append(f"synthetic|{dataset}|{panel}|s{subject}|c{label}|r{row}")
    return {"subjects": tuple(subjects), "labels": tuple(labels), "trial_ids": tuple(trial_ids)}
