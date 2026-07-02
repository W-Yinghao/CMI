"""ACAR V5 Stage-1B launch guard (pure/stdlib). Before ANY factory instantiation / read / train, the run root
output_root/run_id must be ABSENT or EMPTY — a Stage-1B real run never resumes, never overwrites, and never mixes artifacts with a
prior (possibly partial or finalized) run. Any existing content (FINALIZED.json, registry.json, per-ref dirs, or stray files) is a
fail-closed abort, so the first real run's artifact lineage is monotone.
"""
from __future__ import annotations
import os
from acar.v5.substrate import stage1b_output_layout as LO


class Stage1bLaunchError(RuntimeError):
    pass


def assert_fresh_run_root(output_root, run_id):
    """Fail-closed unless output_root/run_id is absent or an empty directory. No resume / no overwrite in Stage-1B."""
    run_root = LO.run_root(output_root, run_id)               # validates output_root/run_id tokens
    if os.path.islink(run_root):                              # a symlinked run root would let artifacts escape (realpath collapses
        raise Stage1bLaunchError(f"run root is a symlink (rejected): {run_root}")   # both file+base) → reject before it's used
    if not os.path.exists(run_root):
        return True                                           # absent → fresh (will be created by the run)
    if not os.path.isdir(run_root):
        raise Stage1bLaunchError(f"run root exists and is not a directory: {run_root}")
    contents = sorted(os.listdir(run_root))
    if contents:
        raise Stage1bLaunchError(f"run root {run_root} is not fresh (found {contents[:5]}); Stage-1B does not resume/overwrite")
    return True
