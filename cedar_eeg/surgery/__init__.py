"""Surgery utilities for CEDAR-EEG."""

from .latent_mask import (
    apply_diagonal_mask,
    candidate_drop_sets,
    effective_rank,
    latent_dimension_scores,
    mask_from_drop_dims,
    rank_latent_dimensions,
)
from .selection import SurgeryCandidate, SurgeryDecision, decide_p0, score_candidate, target_eval_warnings

__all__ = [
    "SurgeryCandidate",
    "SurgeryDecision",
    "apply_diagonal_mask",
    "candidate_drop_sets",
    "decide_p0",
    "effective_rank",
    "latent_dimension_scores",
    "mask_from_drop_dims",
    "rank_latent_dimensions",
    "score_candidate",
    "target_eval_warnings",
]
