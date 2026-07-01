"""ACAR V5 Stage-1B output layout (pure/stdlib). Deterministic, injective run-scoped directory layout for a fold ref's artifact
files, plus containment + global-uniqueness helpers used by the file-artifact writer and the finalize barrier. A ref's artifacts
MUST live under  output_root / run_id / safe_ref_slug(ref) /  and no path may be reused within or across refs.
"""
from __future__ import annotations
import os
from acar.v5.substrate import stage1b_authorization as SA


class Stage1bLayoutError(RuntimeError):
    pass


def safe_ref_slug(ref):
    """Injective slug for a canonical fold ref (e.g. 'PD/fold0/seed20260711' -> 'PD_fold0_seed20260711'). Rejects a non-canonical
    ref and any ref whose characters aren't the expected [A-Za-z0-9_] after slashes->underscores (so the slug can't escape a dir)."""
    if ref not in SA.CANONICAL_FOLD_REFS:
        raise Stage1bLayoutError(f"ref {ref!r} is not a canonical fold ref")
    slug = ref.replace("/", "_")
    if not slug or not all(c.isalnum() or c == "_" for c in slug):
        raise Stage1bLayoutError(f"ref {ref!r} produced an unsafe slug {slug!r}")
    return slug


def run_root(output_root, run_id):
    if not output_root or not isinstance(output_root, str):
        raise Stage1bLayoutError("output_root must be a non-empty path")
    if not run_id or not isinstance(run_id, str) or "/" in run_id or run_id in (".", ".."):
        raise Stage1bLayoutError(f"run_id must be a safe non-empty token, got {run_id!r}")
    return os.path.join(output_root, run_id)


def ref_output_dir(output_root, run_id, ref):
    """The single directory a ref's 6 artifact files must live under: output_root/run_id/safe_ref_slug(ref)."""
    return os.path.join(run_root(output_root, run_id), safe_ref_slug(ref))


def assert_path_under(path, base):
    """Fail-closed containment: realpath(path) must be `base` itself or strictly inside `base` (no symlink escape, no ..)."""
    if not isinstance(path, str) or not path:
        raise Stage1bLayoutError("path must be a non-empty string")
    real = os.path.realpath(path)
    base_real = os.path.realpath(base)
    if real != base_real and not (real + os.sep).startswith(base_real + os.sep):
        raise Stage1bLayoutError(f"path {path!r} escapes required base {base!r}")
    return real


def assert_ref_file_contained(path, output_root, run_id, ref):
    """A single artifact file for `ref` must be a non-symlink regular file under that ref's output dir."""
    if os.path.islink(path):
        raise Stage1bLayoutError(f"{ref}: artifact path is a symlink (rejected): {path}")
    base = ref_output_dir(output_root, run_id, ref)
    return assert_path_under(path, base)


def assert_global_artifact_paths_unique(paths_by_ref):
    """`paths_by_ref` = {ref: [file paths...]}. No realpath may be shared within a ref OR across refs (so one ref's artifacts can
    never sit in another ref's directory or alias another ref's file). Fail-closed."""
    seen = {}
    for ref in sorted(paths_by_ref):
        for p in paths_by_ref[ref]:
            real = os.path.realpath(p)
            if real in seen:
                raise Stage1bLayoutError(f"artifact path {p!r} (ref {ref}) collides with ref {seen[real]}")
            seen[real] = ref
    return len(seen)
