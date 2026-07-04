"""ACAR V5 Stage-1B execution context (pure/stdlib). A gate-issued PERMIT: created ONLY after require_stage1b_full_build_ready
passes, and handed to the reader/trainer FACTORIES so the real reader/trainer are bound to the approved run contract (run_id,
protocol/impl SHAs, output_root, the exact 30 approved fold refs, and the per-disease approved source paths). Real reader/trainer
require this context.
"""
from __future__ import annotations
from dataclasses import dataclass
from acar.v5.substrate import stage1b_authorization as SA


class Stage1bContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stage1BExecutionContext:
    run_id: str
    protocol_tag: str
    protocol_tag_target_sha: str
    implementation_base_sha: str
    output_root: str
    approved_fold_refs: frozenset
    approved_source_paths_by_disease: tuple   # tuple of (disease, ((cohort, path), ...)) — hashable/immutable
    repair_staging_root: str = ""             # Stage-1B15: per-run EPHEMERAL scratch for BrainVision header repair (NOT an artifact)

    def is_approved_ref(self, ref):
        return ref in self.approved_fold_refs

    def source_paths(self, disease):
        for d, pairs in self.approved_source_paths_by_disease:
            if d == disease:
                return dict(pairs)
        raise Stage1bContextError(f"no approved source paths for disease {disease}")


def build_execution_context(authorization, runtime_lock, plan, *, output_root, repair_staging_root=""):
    """Build the context AFTER the gate has validated (authorization/lock/plan). Pure. `repair_staging_root` (Stage-1B15) is the
    validated per-run EPHEMERAL scratch dir the real reader uses for BrainVision header repair — empty for the synthetic path."""
    if not isinstance(output_root, str) or not output_root:
        raise Stage1bContextError("output_root must be a non-empty path")
    spb = {}
    for e in plan["fold_contained_refs"]:
        spb.setdefault(e["disease"], dict(e["source_paths_by_cohort"]))
    frozen_spb = tuple((d, tuple(sorted(spb[d].items()))) for d in sorted(spb))
    return Stage1BExecutionContext(
        run_id=authorization["run_id"], protocol_tag=SA.PROTOCOL_TAG,
        protocol_tag_target_sha=str(authorization["protocol_tag_target_sha"]),
        implementation_base_sha=str(authorization["implementation_base_sha"]),
        output_root=output_root, approved_fold_refs=frozenset(SA.CANONICAL_FOLD_REFS),
        approved_source_paths_by_disease=frozen_spb, repair_staging_root=str(repair_staging_root or ""))
