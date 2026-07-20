"""C86L production entrypoint — prepared but INERT.

PM decision: GO to prepare one real C86L production stage, but this is *not* an
execution authorization. The only valid execution trigger is a separate direct
"授权 C86L". Until then ``execute`` refuses.

This module encodes the short execution contract as executable guards, without a
heavyweight governance apparatus. It opens no real C84 construction labels, target
predictions, Q0 shards, or C85U utility values: readiness validation is metadata
only (roots, declared identities, topology, arithmetic). Payload access happens
only inside an authorized ``execute`` — which does not run now.

Bindings required before any real production (contract §):
  real input identities; construction ⟂ evaluation; Semantics-B physical-trial → 8
  contexts; separate process/filesystem roots; one query = one physical label;
  full contribution-field arithmetic; failure-stop; result manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from . import constants as K


class C86LNotAuthorized(RuntimeError):
    """Raised when execution is attempted without a valid direct '授权 C86L'."""


class C86LContractError(RuntimeError):
    """Raised when a readiness binding violates the execution contract."""


# The exact phrase that authorizes execution. No prior authorization carries forward.
AUTHORIZATION_PHRASE = "授权 C86L"


@dataclass(frozen=True)
class C86LInputBinding:
    """Declared identities + physical roots for a real C86L run (metadata only)."""

    # frozen upstream identities (SHA-256), verified against payloads only under authorization
    input_sha: dict = dc_field(default_factory=lambda: dict(K.FROZEN_INPUT_SHA))
    construction_registry_id: str = "c84f_construction_trial_ids"
    evaluation_registry_id: str = "c84f_evaluation_trial_ids"
    # physically separate roots (real C86L uses distinct processes/filesystem roots)
    acquisition_unlabeled_root: str = ""
    label_oracle_root: str = ""
    contribution_store_root: str = ""
    held_outcome_identity: str = dc_field(
        default_factory=lambda: K.FROZEN_INPUT_SHA["c85u_acceptance_manifest"])
    # declared construction/evaluation disjointness (proven on real IDs under authorization)
    construction_evaluation_disjoint_declared: bool = False


@dataclass(frozen=True)
class C86LResultManifest:
    """Schema of the eventual authorized-run output (no values here)."""

    gate: str
    n_construction_trials: int
    contexts_per_trial: int
    candidate_trial_context_rows: int
    endpoint_definition: str
    isolation_level: str
    failure_stop: bool


REQUIRED_ROOTS = ("acquisition_unlabeled_root", "label_oracle_root", "contribution_store_root")


def validate_readiness(binding: C86LInputBinding) -> dict:
    """Metadata-only readiness check. Opens no protected payload; never authorizes."""
    # all frozen input identities present
    missing = [k for k in K.FROZEN_INPUT_SHA if k not in binding.input_sha]
    if missing:
        raise C86LContractError(f"missing bound input identities: {missing}")
    # separate, non-empty, distinct roots (logical->physical separation for real C86L)
    roots = [getattr(binding, r) for r in REQUIRED_ROOTS]
    if any(not r for r in roots):
        raise C86LContractError("all three physical roots must be declared")
    if len(set(roots)) != len(roots):
        raise C86LContractError("acquisition / label-oracle / contribution roots must be distinct")
    if binding.held_outcome_identity in roots:
        raise C86LContractError("held C85U outcome must not share a root with C86L stores")
    # construction vs evaluation registries are distinct and declared disjoint
    if binding.construction_registry_id == binding.evaluation_registry_id:
        raise C86LContractError("construction and evaluation registries must differ")
    if not binding.construction_evaluation_disjoint_declared:
        raise C86LContractError("construction ⟂ evaluation disjointness must be declared")
    # contribution-field arithmetic is internally consistent
    if K.CONSTRUCTION_ROWS + K.HELD_EVAL_ROWS != K.TOTAL_TARGET_TRIALS:
        raise C86LContractError("construction+held-eval != total trials")
    if K.CONSTRUCTION_ROWS * K.CONTEXTS_PER_CONSTRUCTION_TRIAL != K.CONTEXT_TRIAL_ROWS:
        raise C86LContractError("Semantics-B context-trial arithmetic mismatch")
    if K.CONTEXT_TRIAL_ROWS * K.CANDIDATES_PER_CONTEXT != K.CANDIDATE_TRIAL_CONTEXT_ROWS:
        raise C86LContractError("candidate-trial-context arithmetic mismatch")
    return {
        "ready": True,
        "authorized": False,
        "gate": K.GATE_INSTRUMENT,
        "contexts_per_trial": K.CONTEXTS_PER_CONSTRUCTION_TRIAL,
        "one_query_one_physical_label": True,
        "isolation_level_target": "separate_process_and_filesystem_roots",
        "trigger": AUTHORIZATION_PHRASE,
    }


def execute(binding: C86LInputBinding, authorization: str | None = None,
            output_root: str = ""):
    """Run the real C86L production stage — REFUSES without a valid direct authorization.

    This never runs today: no ``authorization`` equal to the exact trigger phrase
    exists, and C86LP does not create one.
    """
    validate_readiness(binding)
    if authorization != AUTHORIZATION_PHRASE:
        raise C86LNotAuthorized(
            "C86L execution requires a separate direct '授权 C86L'; C86LP does not authorize it"
        )
    if not output_root:
        raise C86LContractError("authorized C86L execution requires an output_root")
    # Authorized real build against the verified inputs; fail-closed inside.
    from .c86l_build import build
    return build(output_root)
