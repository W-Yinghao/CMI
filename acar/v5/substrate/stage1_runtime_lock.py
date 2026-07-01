"""ACAR V5 Stage-1B runtime lock + the combined build-readiness gate (pure/stdlib; no I/O, no data). Real-data Stage-1B code may
run ONLY if BOTH (a) the structured authorization contract AND (b) an explicit, matching runtime lock validate, AND the build
manifest passes the DEV whitelist + final-external schema-only. In Stage-1B0 no real lock is captured, so this is wiring+guards
exercised on synthetic contracts only.
"""
from __future__ import annotations
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_manifest as MAN

RUNTIME_LOCK_FIELDS = ("stage", "protocol_tag", "protocol_tag_target_sha", "run_id", "device_kind", "status")
VERIFIED_STATUS = "CAPTURED_AND_VERIFIED"


class Stage1RuntimeLockError(RuntimeError):
    """Raised when the Stage-1B runtime lock is missing / malformed / not bound to the authorization."""


def validate_runtime_lock(lock, auth):
    """Fail-closed runtime-lock check, cross-bound to the (already-validated) authorization. Pure structural check."""
    if not isinstance(lock, dict):
        raise Stage1RuntimeLockError("runtime lock missing or not a dict")
    missing = [f for f in RUNTIME_LOCK_FIELDS if f not in lock]
    if missing:
        raise Stage1RuntimeLockError(f"runtime lock missing fields: {missing}")
    if lock["stage"] != "Stage-1B":
        raise Stage1RuntimeLockError("runtime lock.stage must be 'Stage-1B'")
    if lock["protocol_tag"] != SA.PROTOCOL_TAG:
        raise Stage1RuntimeLockError("runtime lock.protocol_tag must be acar-v5-protocol")
    if lock["status"] != VERIFIED_STATUS:
        raise Stage1RuntimeLockError(f"runtime lock.status must be {VERIFIED_STATUS}")
    if lock["device_kind"] not in ("cpu", "cuda"):
        raise Stage1RuntimeLockError("runtime lock.device_kind must be cpu|cuda")
    if str(lock["protocol_tag_target_sha"]).lower() != str(auth["protocol_tag_target_sha"]).lower():
        raise Stage1RuntimeLockError("runtime lock target sha must match the authorization")
    if lock["run_id"] != auth["run_id"]:
        raise Stage1RuntimeLockError("runtime lock.run_id must match the authorization.run_id")
    return lock


def require_stage1b_ready(plan, authorization, runtime_lock):
    """THE gate real Stage-1B build code must pass BEFORE any DEV read/train. Requires ALL of: (1) structured authorization
    contract, (2) matching runtime lock, (3) build manifest DEV whitelist (fold-refs-only, disease-matched) + final-external
    schema-only. Returns a readiness report; raises on any failure. Pure — validates contracts/strings; opens/reads NOTHING."""
    auth = SA.validate_stage1b_authorization(authorization)   # (1)
    validate_runtime_lock(runtime_lock, auth)                 # (2)
    admitted = MAN.validate_stage1b_build_manifest(plan, auth)  # (3) whitelist + final-external schema-only
    return {"status": "STAGE1B_READY", "run_id": auth["run_id"], "admitted_fold_refs": admitted,
            "device_kind": runtime_lock["device_kind"],
            "note": "contract+lock+whitelist validated; the actual DEV read/build is separate code (not invoked here)"}
