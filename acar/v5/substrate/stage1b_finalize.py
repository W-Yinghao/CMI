"""ACAR V5 Stage-1B finalize BARRIER (pure/stdlib). Between "all 30 fold substrates built" and "registry populated" there is a
single fail-closed barrier that must pass IN FULL before ANY registration or finalized marker:
    1. exactly the 30 canonical fold refs are present (all-or-none — checked before any register);
    2. GLOBAL artifact-path uniqueness (no file shared within or across refs), for file-backed builds;
    3. config sidecars are CANONICAL (preprocessing_config file == preprocessing_config.canonical_json() and its hash ==
       config_sha256(); training_config sidecar file == training_config.canonical_json()), for file-backed builds;
    4. only then populate the registry (all-or-none) and write the FINALIZED marker.
On ANY failure before step 4 completes, the registry is left EMPTY and NO finalized marker is written.
"""
from __future__ import annotations
import hashlib
import json
import os
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_output_layout as LO
from acar.v5.substrate import stage1b_registry_populate as RP
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import training_config as TC
from acar.v5.substrate import stage1b_feature_dump_writer as FDW   # pure at import (numpy lazy inside)
from acar.v5.substrate import stage1b_registry_io as RIO

FINALIZED_MARKER = "FINALIZED.json"


class Stage1bFinalizeError(RuntimeError):
    pass


def _sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        raise Stage1bFinalizeError(f"cannot hash config file {path}: {e}")
    return h.hexdigest()


def _read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise Stage1bFinalizeError(f"cannot read config file {path}: {e}")


def marker_path(output_root, run_id):
    return os.path.join(LO.run_root(output_root, run_id), FINALIZED_MARKER)


def _contained(path, output_root, run_id, ref, what):
    """The config file must be a non-symlink regular file under output_root/run_id/safe_ref_slug(ref) (per-ref containment)."""
    if output_root and run_id:
        try:
            LO.assert_ref_file_contained(path, output_root, run_id, ref)
        except LO.Stage1bLayoutError as e:
            raise Stage1bFinalizeError(f"{ref}: {what} not contained under the per-ref output dir: {e}")


def _validate_config_sidecars(artifacts, paths_by_ref, sidecars_by_ref, output_root, run_id):
    """File-backed only: preprocessing_config + training_config sidecar files must be CONTAINED (per-ref) AND canonical. Returns
    extra_meta_by_ref with training_config_sha256. Fail-closed."""
    extra_meta = {}
    exp_preproc, exp_train = PC.canonical_json(), TC.canonical_json()
    for ref in sorted(paths_by_ref):
        paths = paths_by_ref[ref]
        pc_path = paths.get("preprocessing_config_path")
        if not pc_path:
            raise Stage1bFinalizeError(f"{ref}: missing preprocessing_config_path for finalize")
        _contained(pc_path, output_root, run_id, ref, "preprocessing_config")
        if _read_text(pc_path) != exp_preproc:
            raise Stage1bFinalizeError(f"{ref}: preprocessing_config file content != canonical preprocessing config")
        if artifacts[ref].get("preprocessing_config_sha256") != PC.config_sha256():
            raise Stage1bFinalizeError(f"{ref}: preprocessing_config_sha256 != pinned config_sha256()")
        tc_path = (sidecars_by_ref or {}).get(ref, {}).get("training_config_path")
        if not tc_path:
            raise Stage1bFinalizeError(f"{ref}: missing training_config sidecar for finalize")
        _contained(tc_path, output_root, run_id, ref, "training_config sidecar")   # the sidecar is NOT a registry file → check here
        if _read_text(tc_path) != exp_train:
            raise Stage1bFinalizeError(f"{ref}: training_config sidecar content != canonical training config")
        tc_sha = _sha256_file(tc_path)
        if tc_sha != TC.config_sha256():
            raise Stage1bFinalizeError(f"{ref}: training_config_sha256 {tc_sha[:12]} != pinned {TC.config_sha256()[:12]}")
        extra_meta[ref] = {"training_config_sha256": tc_sha}
    return extra_meta


def _validate_feature_dumps(paths_by_ref, expected_by_ref):
    """Each ref's feat_dump.npz must parse as the pinned, label-free schema (barrier-level, dumper-agnostic) AND, when an expected
    manifest is supplied, be COMPLETE + ref-consistent: dump ref/disease/fold/seed match the ref; the subject set equals the expected
    fold subjects; each subject's split_role matches; and each subject's window_ids are exactly 0..n-1 (contiguous, unique).
    Fail-closed."""
    for ref in sorted(paths_by_ref):
        fp = paths_by_ref[ref].get("feat_dump_path")
        if not fp:
            raise Stage1bFinalizeError(f"{ref}: missing feat_dump_path for finalize")
        try:
            dump = FDW.load_feature_dump(fp)
        except Exception as e:                                # unparseable / non-conforming / pickle → fail closed
            raise Stage1bFinalizeError(f"{ref}: feature dump failed schema validation: {e}")
        summ = dump["summary"]
        disease = ref.split("/")[0]
        fold = int(ref.split("fold")[1].split("/")[0])
        seed = int(ref.split("seed")[1])
        if not (summ.get("ref") == ref and summ.get("disease") == disease and summ.get("fold") == fold
                and summ.get("seed") == seed):
            raise Stage1bFinalizeError(f"{ref}: feature dump provenance (ref/disease/fold/seed) inconsistent with the ref")
        exp = (expected_by_ref or {}).get(ref)
        if exp is None:
            continue                                          # schema/provenance-only (direct finalize without an expected manifest)
        role_by = exp["role_by_subject"]
        subs = set(dump["subject_key"])
        if subs != set(role_by):
            missing = sorted(set(role_by) - subs)[:3]
            extra = sorted(subs - set(role_by))[:3]
            raise Stage1bFinalizeError(f"{ref}: feature dump subject set incomplete (missing {missing}, extra {extra})")
        wids_by_sub = {}
        for sk, role, wid in zip(dump["subject_key"], dump["split_role"], dump["window_id"]):
            if role != role_by.get(sk):
                raise Stage1bFinalizeError(f"{ref}: subject {sk} split_role {role!r} != expected {role_by.get(sk)!r}")
            wids_by_sub.setdefault(sk, []).append(int(wid))
        n_windows_by_subject = exp.get("n_windows_by_subject") or {}
        for sk, wids in wids_by_sub.items():
            if sorted(wids) != list(range(len(wids))):
                raise Stage1bFinalizeError(f"{ref}: subject {sk} window_ids not contiguous 0..n-1 (got {sorted(wids)[:5]}…)")
            expected_n = n_windows_by_subject.get(sk)         # authoritative window count (from the reader, via the dumper)
            if expected_n is not None and len(wids) != int(expected_n):
                raise Stage1bFinalizeError(f"{ref}: subject {sk} has {len(wids)} windows in the dump != expected {expected_n}")


def finalize_and_populate(registry, artifacts, *, git_commit, env_lock_sha256, channel_montage, sampling_rate,
                          windowing_config, paths_by_ref=None, sidecars_by_ref=None, expected_by_ref=None,
                          output_root=None, run_id=None):
    """Run the finalize barrier then populate. Registry untouched + no marker on any pre-populate failure. Returns n_registered."""
    # 1. all-or-none count check FIRST (before any register, before any marker)
    if set(artifacts) != set(SA.CANONICAL_FOLD_REFS):
        raise Stage1bFinalizeError(f"finalize requires exactly the 30 canonical fold refs (have {len(artifacts)})")
    extra_meta = None
    # 2/3. file-backed builds: global path uniqueness + canonical config sidecars
    if paths_by_ref:
        try:
            LO.assert_global_artifact_paths_unique({r: list(paths_by_ref[r].values()) for r in paths_by_ref})
        except LO.Stage1bLayoutError as e:
            raise Stage1bFinalizeError(f"global artifact-path uniqueness failed: {e}")
        extra_meta = _validate_config_sidecars(artifacts, paths_by_ref, sidecars_by_ref, output_root, run_id)
        _validate_feature_dumps(paths_by_ref, expected_by_ref)   # schema + (when expected supplied) completeness/ref-consistency
    # 4. barrier passed → populate (all-or-none). If a marker is due (file-backed), populate THEN write it ATOMICALLY; if the marker
    #    write fails, roll the registry back so the invariant holds: a FINALIZED marker exists IFF the registry is fully populated.
    n = RP.populate_registry(registry, artifacts, git_commit=git_commit, env_lock_sha256=env_lock_sha256,
                             channel_montage=channel_montage, sampling_rate=sampling_rate,
                             windowing_config=windowing_config, extra_meta_by_ref=extra_meta)
    if output_root and run_id and paths_by_ref:               # marker + registry file only for file-backed real builds
        os.makedirs(LO.run_root(output_root, run_id), exist_ok=True)
        reg_path = os.path.join(LO.run_root(output_root, run_id), RIO.REGISTRY_FILE)
        try:                                                  # persist the registry as a canonical, hash-bound FILE artifact
            reg_sha = RIO.write_registry(registry, reg_path)
        except Exception as e:
            registry._rollback(sorted(artifacts))
            raise Stage1bFinalizeError(f"registry.json write failed (rolled back): {e}")
        payload = {"status": "FINALIZED", "n_registered": n, "n_refs": len(artifacts), "registry_sha256": reg_sha,
                   "git_commit": git_commit, "env_lock_sha256": env_lock_sha256}
        try:                                                  # marker LAST + atomic → marker exists IFF registry fully populated+persisted
            write_finalized_marker(output_root, run_id, payload)
        except Exception as e:                                # marker failed → undo population + registry file (no half state)
            registry._rollback(sorted(artifacts))
            try:
                os.remove(reg_path)
                cleanup = "registry.json removed"
            except OSError as rm:                             # surface (do NOT silently swallow) a failed cleanup
                cleanup = f"WARNING registry.json may remain at {reg_path} (removal failed: {rm})"
            raise Stage1bFinalizeError(f"registry populated but FINALIZED marker write failed (rolled back; {cleanup}): {e}")
    return n


def write_finalized_marker(output_root, run_id, payload):
    """Write the FINALIZED marker ATOMICALLY: serialize to <run_root>/FINALIZED.json.tmp then os.replace → FINALIZED.json, so a
    reader never sees a partial marker (and the marker's presence means the write completed). Cleans up the temp on failure."""
    mp = marker_path(output_root, run_id)
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    tmp = mp + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, mp)                                   # atomic rename within the same directory
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return mp
