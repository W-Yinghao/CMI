"""Read-only, zoo-scoped access to frozen C84F source and target arrays."""
from __future__ import annotations

from typing import Any

import numpy as np

from .c84s_common import require
from .c84sr1_context_enumerator import ContextDescriptor


def zoo_key(context: ContextDescriptor) -> tuple[str, str, int, int]:
    return context.dataset, context.panel, context.training_seed, context.level


def zoo_then_target_key(context: ContextDescriptor) -> tuple[Any, ...]:
    return (*zoo_key(context), int(context.target_subject_id))


class FrozenZooReader:
    """Cache only the current 81-candidate zoo and expose target slices."""

    def __init__(self, *, include_source: bool):
        self.include_source = bool(include_source)
        self._key: tuple[str, str, int, int] | None = None
        self._data: dict[str, Any] | None = None
        self.files_opened = 0

    def _load_zoo(self, context: ContextDescriptor) -> None:
        candidates = context.candidates
        candidate_ids = [row.unit_id for row in candidates]
        regimes = [row.regime for row in candidates]
        trajectory_orders = [row.trajectory_order for row in candidates]
        all_subjects: np.ndarray | None = None
        all_trial_ids: np.ndarray | None = None
        target_logits: list[np.ndarray] = []
        source_probabilities: list[np.ndarray] = []
        source_labels = source_domains = source_trial_ids = None
        for candidate in candidates:
            with np.load(candidate.target_artifact_path, allow_pickle=False) as archive:
                self.files_opened += 1
                require(str(archive["unit_id"].item()) == candidate.unit_id,
                        "target artifact unit identity drift")
                subjects = np.asarray(archive["target_subject_id"]).astype(str)
                trial_ids = np.asarray(archive["target_trial_id"], dtype=str)
                if all_subjects is None:
                    all_subjects, all_trial_ids = subjects, trial_ids
                else:
                    require(np.array_equal(subjects, all_subjects),
                            "target subject order differs across zoo candidates")
                    require(np.array_equal(trial_ids, all_trial_ids),
                            "target trial order differs across zoo candidates")
                target_logits.append(np.asarray(archive["logits"], dtype=float))
            if self.include_source:
                with np.load(candidate.source_audit_path, allow_pickle=False) as archive:
                    self.files_opened += 1
                    require(str(archive["unit_id"].item()) == candidate.unit_id,
                            "source artifact unit identity drift")
                    labels = np.asarray(archive["source_class_label"], dtype=int)
                    domains = np.asarray(archive["source_domain_id"])
                    trial_ids = np.asarray(archive["source_trial_id"], dtype=str)
                    if source_labels is None:
                        source_labels, source_domains, source_trial_ids = labels, domains, trial_ids
                    else:
                        require(np.array_equal(labels, source_labels),
                                "source labels differ across zoo candidates")
                        require(np.array_equal(domains, source_domains),
                                "source domains differ across zoo candidates")
                        require(np.array_equal(trial_ids, source_trial_ids),
                                "source trial order differs across zoo candidates")
                    source_probabilities.append(np.asarray(archive["probabilities"], dtype=float))
        require(all_subjects is not None and all_trial_ids is not None,
                "target zoo arrays were not loaded")
        require(len(set(candidate_ids)) == 81, "zoo candidate identity coverage drift")
        data: dict[str, Any] = {
            "candidate_ids": candidate_ids,
            "regimes": regimes,
            "trajectory_orders": trajectory_orders,
            "all_target_subjects": all_subjects,
            "all_target_trial_ids": all_trial_ids,
            "all_target_logits": np.stack(target_logits),
        }
        if self.include_source:
            data.update({
                "source_probabilities": np.stack(source_probabilities),
                "source_labels": np.asarray(source_labels),
                "source_domains": np.asarray(source_domains),
                "source_trial_ids": np.asarray(source_trial_ids),
            })
        self._key, self._data = zoo_key(context), data

    def __call__(self, context: ContextDescriptor) -> dict[str, Any]:
        if self._key != zoo_key(context):
            self._load_zoo(context)
        require(self._data is not None, "zoo reader cache absent")
        subjects = self._data["all_target_subjects"]
        mask = subjects == str(context.target_subject_id)
        require(np.any(mask), "target subject absent from frozen zoo")
        output = {
            "candidate_ids": list(self._data["candidate_ids"]),
            "regimes": list(self._data["regimes"]),
            "trajectory_orders": list(self._data["trajectory_orders"]),
            "target_logits": self._data["all_target_logits"][:, mask],
            "target_trial_ids": self._data["all_target_trial_ids"][mask],
        }
        if self.include_source:
            output.update({
                "source_probabilities": self._data["source_probabilities"],
                "source_labels": self._data["source_labels"],
                "source_domains": self._data["source_domains"],
                "source_trial_ids": self._data["source_trial_ids"],
            })
        return output
