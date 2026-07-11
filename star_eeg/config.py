"""Frozen STAR_00A and future STAR_01 control-plane constants."""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


DEPENDENCY_BRANCH = "origin/project/s2p-subject-scaling"
DEPENDENCY_COMMIT = "a9134eb5eb7f8486a5e1ee41831823dab39381ed"
STAR_BRANCH = "project/star-task-anchor"
PROJECT_NAME = "STAR-EEG"
PROJECT_EXPANSION = "Source-Task Anchored Representation Reorganization for EEG"


@dataclass(frozen=True)
class Star01Protocol:
    start_tags: Tuple[str, ...] = ("H200_s0", "H200_s1")
    variants: Tuple[str, ...] = (
        "H200_BASE",
        "H200_SSL_CONT",
        "H200_STAR_TRUE",
        "H200_STAR_SHUFFLED",
    )
    continuation_optimizer_steps: int = 3750
    original_h200_optimizer_steps: int = 18750
    ssl_steps_per_cycle: int = 4
    anchor_steps_per_cycle: int = 1
    anchor_step_fraction: float = 0.20
    batch_size: int = 64
    optimizer_family: str = "AdamW"
    base_learning_rate: float = 5e-4
    weight_decay: float = 5e-2
    scheduler: str = "CosineAnnealingLR_by_optimizer_step"
    scheduler_eta_min: float = 1e-5
    checkpoint_save_steps: Tuple[int, ...] = (750, 1500, 2250, 3000, 3750)
    primary_checkpoint_step: int = 3750
    task_head: str = "temporary_linear_6400_to_9"
    task_loss: str = "source_train_cross_entropy_weight_1"
    task_loss_weight: float = 1.0
    permutation_seed: int = 20260711
    model_seeds: Tuple[int, ...] = (0, 1)
    head_seed_offset: int = 12000
    ssl_stream_seed_offset: int = 13000
    ssl_objective_seed_offset: int = 14000
    anchor_stream_seed_offset: int = 15000

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


STAR01 = Star01Protocol()

# The guard examines these enabled identifiers and imports, not prose or
# forbidden-method documentation. Only native SSL and a temporary source-task
# head are active in the approved scaffold.
ACTIVE_METHOD_REGISTRY = {
    "ssl_objective": "native_cbramod_ssl",
    "task_objective": "source_task_cross_entropy",
    "task_head": "temporary_linear_head",
    "target_data_access": "none",
}

ACTIVE_IMPORT_PATHS = (
    "star_eeg.objectives.alternating_schedule",
    "star_eeg.objectives.task_anchor",
    "s2p.scripts.route_b_33ch_loader",
)
