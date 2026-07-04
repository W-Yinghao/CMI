"""ACAR V5 Stage-1B15 repair staging root (pure/stdlib; nothing heavy at import). The production real build materializes EPHEMERAL
BrainVision repaired headers/markers (Stage-1B12/1B13/1B14 read-repair) under a per-run 'repair staging root'. This root is SCRATCH
ONLY: it is never a registered artifact, never appears in registry.json / FINALIZED.json, and lives outside both the raw DEV tree and
the run's hash-bound output package. This module VALIDATES it fail-closed and CREATES it — ONLY after the full-build gate passes and
BEFORE any factory/read.

Fail-closed rules: a non-empty absolute path; not a symlink; not overlapping (inside/equal/containing) any approved raw cohort source
path; not overlapping the run artifact root output_root/run_id; absent OR an empty real directory at launch.
"""
from __future__ import annotations
import os


class RepairStagingError(RuntimeError):
    pass


def _overlaps(a, b):
    """True iff realpath(a) is equal to, inside, or contains realpath(b)."""
    ra, rb = os.path.realpath(a), os.path.realpath(b)
    return ra == rb or ra.startswith(rb + os.sep) or rb.startswith(ra + os.sep)


def validate_repair_staging_root(repair_staging_root, *, output_root, run_id, approved_source_paths):
    """Fail-closed validation of the repair staging root (does NOT create it). `approved_source_paths` = the approved raw cohort dirs
    for this run. `output_root`/`run_id` identify the hash-bound artifact package the staging root must stay out of."""
    r = repair_staging_root
    if not (isinstance(r, str) and r and os.path.isabs(r)):
        raise RepairStagingError(f"repair_staging_root must be a non-empty absolute path (got {r!r})")
    if os.path.islink(r):
        raise RepairStagingError(f"repair_staging_root must not be a symlink: {r}")
    for sp in approved_source_paths:
        if sp and _overlaps(r, sp):
            raise RepairStagingError(f"repair_staging_root {r} overlaps an approved raw source path {sp} "
                                     "(must never touch the raw DEV tree)")
    if not (isinstance(output_root, str) and output_root and isinstance(run_id, str) and run_id):
        raise RepairStagingError("output_root and run_id are required to validate the repair staging root")
    run_root = os.path.join(output_root, run_id)
    if _overlaps(r, run_root):
        raise RepairStagingError(f"repair_staging_root {r} overlaps the run artifact root {run_root} "
                                 "(scratch must never be part of the hash-bound package)")
    if os.path.exists(r):
        if os.path.islink(r) or not os.path.isdir(r):
            raise RepairStagingError(f"repair_staging_root exists and is not a real directory: {r}")
        if os.listdir(r):
            raise RepairStagingError(f"repair_staging_root must be absent or an EMPTY directory at launch: {r}")
    return True


def create_repair_staging_root(repair_staging_root, *, output_root, run_id, approved_source_paths):
    """Validate (fail-closed) THEN create the empty repair staging root. Call ONLY after the gate passes. Returns its abspath."""
    validate_repair_staging_root(repair_staging_root, output_root=output_root, run_id=run_id,
                                 approved_source_paths=approved_source_paths)
    os.makedirs(repair_staging_root, exist_ok=True)            # absent-or-empty was just asserted
    if os.path.islink(repair_staging_root) or not os.path.isdir(repair_staging_root):
        raise RepairStagingError(f"repair_staging_root is not a real directory after creation: {repair_staging_root}")
    return os.path.abspath(repair_staging_root)
