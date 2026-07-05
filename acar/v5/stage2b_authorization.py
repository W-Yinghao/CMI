"""ACAR V5 Stage-2B authorization GATE (pure/stdlib; NO I/O). Structured + fail-closed — mirrors the Stage-1B gate.

Binding real-DEV candidate selection (Stage-2B) requires an EXPLICIT authorization that pins: the protocol tag + full target
sha; the admitted Stage-1B package (run_id + registry_sha256); EXACTLY the 10 canonical selection refs; EXACTLY the 22 frozen
candidate ids; and the three forbid flags (S1 refs for selection / external read / lockbox). There is NO global boolean flag —
the selection engine cannot run without a valid authorization bound to the admitted package. Stage-2B0 implements + tests this
gate but issues NO real authorization (no real DEV selection run).
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN

STAGE = "Stage-2B"
PROTOCOL_TAG = "acar-v5-protocol"
PROTOCOL_TAG_TARGET_SHA_FULL = "4278435975a72b1127803dd2cffab420c083e430"
REQUIRED_STAGE2B_STATEMENT = (
    "Authorize ACAR-V5 Stage-2B real DEV candidate selection exactly under ACAR_V5_ENDPOINTS.md on the admitted Stage-1B "
    "package, the 10 canonical selection refs, and the 22 frozen candidates")

CANONICAL_SELECTION_REFS = tuple(sorted(r["ref"] for r in PLAN.selection_refs()))   # the 10 (seed 20260711)
CANONICAL_CANDIDATE_IDS = tuple(P.CANDIDATE_IDS)                                     # the 22 frozen ids
STAGE2B_AUTH_FIELDS = (
    "stage", "protocol_tag", "protocol_tag_target_sha", "implementation_base_sha", "stage1b_run_id",
    "stage1b_registry_sha256", "allowed_selection_refs", "allowed_candidate_ids",
    "forbid_s1_refs_for_selection", "forbid_external_read", "forbid_lockbox", "run_id", "statement")

_HEX = "0123456789abcdef"


class Stage2bAuthorizationError(RuntimeError):
    """Raised when a Stage-2B authorization is malformed / not bound to the admitted package (fail-closed)."""


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in _HEX for c in s.lower())


def validate_stage2b_authorization(auth):
    """Fail-closed structural validation of a Stage-2B authorization (pure). Returns True or raises."""
    if not isinstance(auth, dict):
        raise Stage2bAuthorizationError("authorization must be a dict")
    extra = sorted(set(auth) - set(STAGE2B_AUTH_FIELDS))
    missing = sorted(set(STAGE2B_AUTH_FIELDS) - set(auth))
    if extra or missing:
        raise Stage2bAuthorizationError(f"auth field mismatch (missing {missing}, extra {extra})")
    if auth["stage"] != STAGE:
        raise Stage2bAuthorizationError(f"stage must be {STAGE!r}")
    if auth["protocol_tag"] != PROTOCOL_TAG:
        raise Stage2bAuthorizationError("protocol_tag mismatch")
    if auth["protocol_tag_target_sha"] != PROTOCOL_TAG_TARGET_SHA_FULL:
        raise Stage2bAuthorizationError("protocol_tag_target_sha must be the full 40-hex target sha")
    if not _is_hex(auth["implementation_base_sha"], 40):
        raise Stage2bAuthorizationError("implementation_base_sha must be 40-hex")
    if not (isinstance(auth["stage1b_run_id"], str) and auth["stage1b_run_id"]):
        raise Stage2bAuthorizationError("stage1b_run_id must be a non-empty string")
    if not _is_hex(auth["stage1b_registry_sha256"], 64):
        raise Stage2bAuthorizationError("stage1b_registry_sha256 must be 64-hex")
    if tuple(sorted(auth["allowed_selection_refs"])) != CANONICAL_SELECTION_REFS:
        raise Stage2bAuthorizationError("allowed_selection_refs must be exactly the 10 canonical selection refs")
    if tuple(sorted(auth["allowed_candidate_ids"])) != tuple(sorted(CANONICAL_CANDIDATE_IDS)):
        raise Stage2bAuthorizationError("allowed_candidate_ids must be exactly the 22 frozen candidate ids")
    for f in ("forbid_s1_refs_for_selection", "forbid_external_read", "forbid_lockbox"):
        if auth[f] is not True:
            raise Stage2bAuthorizationError(f"{f} must be True")
    if auth["statement"] != REQUIRED_STAGE2B_STATEMENT:
        raise Stage2bAuthorizationError("statement mismatch")
    if not (isinstance(auth["run_id"], str) and auth["run_id"]):
        raise Stage2bAuthorizationError("run_id must be a non-empty string")
    return True


def require_stage2b_ready(auth, *, stage1b_run_id, stage1b_registry_sha256):
    """Validate the auth AND bind it to the ACTUAL admitted package (run_id + recomputed registry_sha256). Fail-closed."""
    validate_stage2b_authorization(auth)
    if auth["stage1b_run_id"] != stage1b_run_id:
        raise Stage2bAuthorizationError(f"auth stage1b_run_id {auth['stage1b_run_id']!r} != admitted run {stage1b_run_id!r}")
    if auth["stage1b_registry_sha256"] != stage1b_registry_sha256:
        raise Stage2bAuthorizationError("auth stage1b_registry_sha256 does not match the admitted package registry hash")
    return True
