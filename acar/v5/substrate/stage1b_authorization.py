"""ACAR V5 Stage-1B authorization contract (pure/stdlib; no I/O, no data). A Stage-1B authorization is NOT a magic statement — it
is a structured, auditable run contract that BINDS the protocol tag/commit, the EXACT 30 fold-contained refs, the seed set, and
explicit forbid-flags. Any real DEV read must first pass this contract (+ a runtime lock, see stage1_runtime_lock.py).
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN

PROTOCOL_TAG = "acar-v5-protocol"
PROTOCOL_TAG_TARGET_SHA_FULL = "4278435975a72b1127803dd2cffab420c083e430"
REQUIRED_STAGE1B_STATEMENT = (
    "Authorize ACAR-V5 Stage-1B fold-contained substrate build (real DEV read) exactly under ACAR_V5_STAGE1B_AUTHORIZATION.md"
)
STAGE1B_AUTH_FIELDS = (
    "stage", "protocol_tag", "protocol_tag_target_sha", "implementation_base_sha",
    "allowed_ref_type", "allowed_refs", "allowed_seeds", "selection_seed",
    "forbid_final_external_refs", "forbid_external_sites", "forbid_candidate_selection", "forbid_external_read",
    "run_id", "statement",
)
CANONICAL_FOLD_REFS = frozenset(r["ref"] for r in PLAN.fold_refs())   # the exact 30

_HEX = "0123456789abcdef"


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in _HEX for c in s.lower())


class Stage1BuildNotAuthorizedError(RuntimeError):
    """Base error: a Stage-1B real-data build is not authorized (missing / malformed / non-conforming contract)."""


def validate_stage1b_authorization(auth):
    """Fail-closed structural check of a Stage-1B authorization contract. Returns the auth on success; raises
    Stage1BuildNotAuthorizedError otherwise. Pure — validates the contract only; performs NO read/build."""
    if not isinstance(auth, dict):
        raise Stage1BuildNotAuthorizedError("Stage-1B authorization missing or not a dict")
    missing = [f for f in STAGE1B_AUTH_FIELDS if f not in auth]
    if missing:
        raise Stage1BuildNotAuthorizedError(f"authorization missing fields: {missing}")
    extra = [f for f in auth if f not in STAGE1B_AUTH_FIELDS]
    if extra:
        raise Stage1BuildNotAuthorizedError(f"authorization has unknown fields: {extra}")
    if auth["stage"] != "Stage-1B":
        raise Stage1BuildNotAuthorizedError("authorization.stage must be 'Stage-1B'")
    if auth["protocol_tag"] != PROTOCOL_TAG:
        raise Stage1BuildNotAuthorizedError("authorization.protocol_tag must be acar-v5-protocol")
    tsha = str(auth["protocol_tag_target_sha"]).lower()
    if len(tsha) < 7 or not PROTOCOL_TAG_TARGET_SHA_FULL.startswith(tsha):
        raise Stage1BuildNotAuthorizedError("protocol_tag_target_sha must be a hex prefix of 4278435… (the tagged commit)")
    if not _is_hex(auth["implementation_base_sha"], 40):
        raise Stage1BuildNotAuthorizedError("implementation_base_sha must be a full 40-hex commit")
    if auth["allowed_ref_type"] != "fold_contained_only":
        raise Stage1BuildNotAuthorizedError("allowed_ref_type must be 'fold_contained_only'")
    try:
        allowed = set(auth["allowed_refs"])
    except TypeError:
        raise Stage1BuildNotAuthorizedError("allowed_refs must be a collection")
    if allowed != set(CANONICAL_FOLD_REFS):
        raise Stage1BuildNotAuthorizedError("allowed_refs must be EXACTLY the 30 fold-contained refs")
    if set(auth["allowed_seeds"]) != set(P.S1_SEEDS):
        raise Stage1BuildNotAuthorizedError(f"allowed_seeds must be exactly {sorted(P.S1_SEEDS)}")
    if auth["selection_seed"] != P.SELECTION_SEED:
        raise Stage1BuildNotAuthorizedError(f"selection_seed must be {P.SELECTION_SEED}")
    for flag in ("forbid_final_external_refs", "forbid_external_sites", "forbid_candidate_selection", "forbid_external_read"):
        if auth[flag] is not True:
            raise Stage1BuildNotAuthorizedError(f"authorization.{flag} must be True")
    if not isinstance(auth["run_id"], str) or not auth["run_id"]:
        raise Stage1BuildNotAuthorizedError("run_id must be a non-empty string")
    if auth["statement"] != REQUIRED_STAGE1B_STATEMENT:
        raise Stage1BuildNotAuthorizedError("authorization.statement mismatch")
    return auth
