"""Deterministic optimizer-step schedule for the frozen STAR_01 variants."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from star_eeg.config import STAR01
from star_eeg.data.faced_split_contract import canonical_hash


BASE = "H200_BASE"
SSL_CONT = "H200_SSL_CONT"
STAR_TRUE = "H200_STAR_TRUE"
STAR_SHUFFLED = "H200_STAR_SHUFFLED"
TRAINED_VARIANTS = (SSL_CONT, STAR_TRUE, STAR_SHUFFLED)


@dataclass(frozen=True)
class ScheduledStep:
    optimizer_step: int
    semantic_slot: str
    update_kind: str
    ssl_stream: Optional[str]
    ssl_stream_index: Optional[int]
    anchor_stream_index: Optional[int]


def build_schedule(variant: str, total_steps: int = STAR01.continuation_optimizer_steps) -> List[ScheduledStep]:
    if variant == BASE:
        if total_steps not in (0, STAR01.continuation_optimizer_steps):
            raise ValueError("H200_BASE has no continuation schedule")
        return []
    if variant not in TRAINED_VARIANTS:
        raise ValueError(f"variant outside frozen universe: {variant}")
    cycle = STAR01.ssl_steps_per_cycle + STAR01.anchor_steps_per_cycle
    if total_steps <= 0 or total_steps % cycle:
        raise ValueError(f"total_steps must be a positive multiple of {cycle}")

    common_ssl = 0
    replacement_ssl = 0
    anchor = 0
    schedule = []
    for step in range(1, total_steps + 1):
        anchor_slot = step % cycle == 0
        if anchor_slot and variant == SSL_CONT:
            replacement_ssl += 1
            schedule.append(ScheduledStep(
                optimizer_step=step,
                semantic_slot="anchor_slot",
                update_kind="ssl",
                ssl_stream="replacement",
                ssl_stream_index=replacement_ssl,
                anchor_stream_index=None,
            ))
        elif anchor_slot:
            anchor += 1
            schedule.append(ScheduledStep(
                optimizer_step=step,
                semantic_slot="anchor_slot",
                update_kind="source_task_anchor",
                ssl_stream=None,
                ssl_stream_index=None,
                anchor_stream_index=anchor,
            ))
        else:
            common_ssl += 1
            schedule.append(ScheduledStep(
                optimizer_step=step,
                semantic_slot="common_ssl_slot",
                update_kind="ssl",
                ssl_stream="common",
                ssl_stream_index=common_ssl,
                anchor_stream_index=None,
            ))
    return schedule


def schedule_payload(variant: str, total_steps: int = STAR01.continuation_optimizer_steps) -> Dict[str, object]:
    steps = [asdict(step) for step in build_schedule(variant, total_steps=total_steps)]
    core = {"variant": variant, "total_steps": len(steps), "steps": steps}
    schedule_identity = {"total_steps": len(steps), "steps": steps}
    return {**core, "schedule_hash": canonical_hash(schedule_identity)}


def build_compute_match_contract(total_steps: int = STAR01.continuation_optimizer_steps) -> Dict[str, object]:
    payloads = {variant: schedule_payload(variant, total_steps) for variant in TRAINED_VARIANTS}
    summaries = {}
    for variant, payload in payloads.items():
        steps = payload["steps"]
        summaries[variant] = {
            "optimizer_steps": len(steps),
            "total_batches": len(steps),
            "common_ssl_steps": sum(row["ssl_stream"] == "common" for row in steps),
            "replacement_ssl_steps": sum(row["ssl_stream"] == "replacement" for row in steps),
            "anchor_steps": sum(row["update_kind"] == "source_task_anchor" for row in steps),
            "schedule_hash": payload["schedule_hash"],
        }
    semantic_c = [row["semantic_slot"] for row in payloads[STAR_TRUE]["steps"]]
    semantic_d = [row["semantic_slot"] for row in payloads[STAR_SHUFFLED]["steps"]]
    update_c = [row["update_kind"] for row in payloads[STAR_TRUE]["steps"]]
    update_d = [row["update_kind"] for row in payloads[STAR_SHUFFLED]["steps"]]
    core = {
        "schema_version": 1,
        "unit": "optimizer_step",
        "total_optimizer_steps": total_steps,
        "original_h200_optimizer_steps": STAR01.original_h200_optimizer_steps,
        "continuation_fraction_of_original_h200_updates": total_steps / STAR01.original_h200_optimizer_steps,
        "cycle": ["ssl", "ssl", "ssl", "ssl", "anchor_slot"],
        "anchor_step_fraction": STAR01.anchor_step_fraction,
        "batch_size": STAR01.batch_size,
        "optimizer_family": STAR01.optimizer_family,
        "base_learning_rate": STAR01.base_learning_rate,
        "weight_decay": STAR01.weight_decay,
        "scheduler": STAR01.scheduler,
        "scheduler_eta_min": STAR01.scheduler_eta_min,
        "checkpoint_save_steps": list(STAR01.checkpoint_save_steps),
        "primary_checkpoint_step": STAR01.primary_checkpoint_step,
        "primary_checkpoint_selection": "fixed_final_optimizer_step",
        "diagnostic_best_pretrain_val_selects_primary": False,
        "same_start_checkpoint_within_seed": True,
        "same_optimizer_parameter_registry": True,
        "same_base_schedule": True,
        "same_tueg_h200_route_b_source_pool_within_seed": True,
        "shared_ssl_slots_use_identical_common_stream_indices": True,
        "ssl_cont_anchor_slots_use_separate_replacement_ssl_stream": True,
        "summaries": summaries,
        "checks": {
            "equal_total_steps": len({row["optimizer_steps"] for row in summaries.values()}) == 1,
            "equal_total_batches": len({row["total_batches"] for row in summaries.values()}) == 1,
            "ssl_cont_replaces_every_anchor_slot": summaries[SSL_CONT]["replacement_ssl_steps"] == summaries[STAR_TRUE]["anchor_steps"],
            "true_shuffled_semantic_schedule_identical": semantic_c == semantic_d,
            "true_shuffled_update_schedule_identical": update_c == update_d,
            "anchor_fraction_exact": summaries[STAR_TRUE]["anchor_steps"] / total_steps == STAR01.anchor_step_fraction,
        },
    }
    return {**core, "compute_match_hash": canonical_hash(core)}
