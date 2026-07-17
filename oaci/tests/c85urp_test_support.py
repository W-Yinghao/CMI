"""Synthetic-only helpers for C85URP readiness tests."""
from __future__ import annotations

import numpy as np

from oaci.multidataset.c84sr1_context_enumerator import (
    CandidateDescriptor,
    ContextDescriptor,
)

from oaci.theory.c85u_utility_builder import compute_context_utility_payload


def shadow_candidate(index: int, *, dataset: str = "Lee2019_MI") -> CandidateDescriptor:
    regime = "ERM" if index == 0 else ("OACI" if index <= 40 else "SRC")
    order = 0 if index == 0 else (index if index <= 40 else index - 40)
    return CandidateDescriptor(
        dataset=dataset, panel="A", training_seed=5, level=0,
        regime=regime, epoch=order, trajectory_order=order,
        unit_id=f"shadow_unit_{index:02d}",
        level_intervention_id="SHADOW_LEVEL",
        source_audit_path="SHADOW", source_audit_sha256="a" * 64,
        target_artifact_path="SHADOW", target_artifact_sha256=f"{index:064x}",
        training_sidecar_path="SHADOW", training_sidecar_sha256="b" * 64,
        target_sidecar_path="SHADOW", target_sidecar_sha256="c" * 64,
    )


def shadow_context(*, dataset: str = "Lee2019_MI") -> ContextDescriptor:
    return ContextDescriptor(
        context_id="shadow_context", dataset=dataset, target_subject_id="1",
        panel="A", training_seed=5, level=0,
        candidates=tuple(shadow_candidate(index, dataset=dataset) for index in range(81)),
    )


def shadow_rows(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "dataset": "Lee2019_MI", "target_subject_id": "1",
            "target_trial_id": f"trial_{index:03d}",
            "canonical_class_label": index % 2,
            "session": "0", "run": "0", "split_identity": "evaluation",
        }
        for index in range(count)
    ]


def shadow_payload(*, seed: int = 19, zero_spread: bool = False):
    context = shadow_context()
    rows = shadow_rows()
    generator = np.random.default_rng(seed)
    base = generator.normal(size=(1, len(rows), 2))
    logits = np.repeat(base, 81, axis=0) if zero_spread else generator.normal(
        size=(81, len(rows), 2),
    )
    return compute_context_utility_payload(
        context=context,
        candidate_data={
            "candidate_ids": [candidate.unit_id for candidate in context.candidates],
            "target_trial_ids": np.asarray(
                [row["target_trial_id"] for row in rows], dtype=str,
            ),
            "target_logits": logits,
        },
        evaluation_rows=rows,
        evaluation_label_view_manifest_sha256="d" * 64,
    )


__all__ = [
    "shadow_candidate", "shadow_context", "shadow_payload", "shadow_rows",
]
