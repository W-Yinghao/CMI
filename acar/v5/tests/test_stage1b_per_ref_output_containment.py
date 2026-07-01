"""Guard (Stage-1B6, req6): with a run_id, the file writer enforces PER-REF containment — every artifact file for ref R must be a
non-symlink regular file under output_root/run_id/safe_ref_slug(R). A file in another ref's dir, outside the run root, or a symlink
is rejected. Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_file_artifact_writer as FW
from acar.v5.substrate import stage1b_output_layout as LO
from acar.v5.tests._util import expect_raises, ok

RUN = "run-syn-0001"
REF = "PD/fold0/seed20260711"
OTHER = "PD/fold1/seed20260711"


def _mk(root, ref, exclude=None):
    d = LO.ref_output_dir(root, RUN, ref)
    os.makedirs(d, exist_ok=True)
    raw = {"ref": ref, "disease": "PD", "fold": 0, "seed": 20260711}
    for pk in sorted(set(FW.FILE_SOURCE.values())):
        if pk == exclude:
            continue
        p = os.path.join(d, pk + ".bin")
        with open(p, "wb") as f:
            f.write((ref + pk).encode())
        raw[pk] = p
    return raw


def _write(root, ref, name, data=b"x"):
    d = LO.ref_output_dir(root, RUN, ref)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def test_contained_per_ref_ok():
    with tempfile.TemporaryDirectory() as root:
        art = FW.write_artifact_from_files(_mk(root, REF), expected_ref=REF, disease="PD", fold=0, seed=20260711,
                                           output_root=root, run_id=RUN)
        assert all(len(art[h]) == 64 for h in FW.FILE_SOURCE) and "_paths" in art
    ok("6 files under output_root/run_id/safe_ref_slug(ref) → written + hashed")


def test_file_in_other_refs_dir_rejected():
    with tempfile.TemporaryDirectory() as root:
        raw = _mk(root, REF, exclude="feat_dump_path")
        raw["feat_dump_path"] = _write(root, OTHER, "feat_dump.bin")   # lives under a DIFFERENT ref's dir
        expect_raises(FW.Stage1bFileArtifactError,
                      lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711,
                                                           output_root=root, run_id=RUN))
    ok("an artifact file under ANOTHER ref's per-ref dir → Stage1bFileArtifactError")


def test_file_outside_run_root_rejected():
    with tempfile.TemporaryDirectory() as root:
        raw = _mk(root, REF, exclude="feat_dump_path")
        stray = os.path.join(root, "feat_dump.bin")           # directly under output_root, not under run_id/slug
        with open(stray, "wb") as f:
            f.write(b"x")
        raw["feat_dump_path"] = stray
        expect_raises(FW.Stage1bFileArtifactError,
                      lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711,
                                                           output_root=root, run_id=RUN))
    ok("an artifact file outside the per-ref dir (in the run/output root) → Stage1bFileArtifactError")


def test_symlink_rejected():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
        raw = _mk(root, REF, exclude="feat_dump_path")
        target = os.path.join(outside, "secret.bin")
        with open(target, "wb") as f:
            f.write(b"held-out")
        link = os.path.join(LO.ref_output_dir(root, RUN, REF), "feat_dump.bin")
        os.symlink(target, link)
        raw["feat_dump_path"] = link
        expect_raises(FW.Stage1bFileArtifactError,
                      lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711,
                                                           output_root=root, run_id=RUN))
    ok("a symlink artifact path (even inside the per-ref dir) → Stage1bFileArtifactError")


def main():
    print("ACAR v5 Stage-1B6 guard: per-ref output containment")
    test_contained_per_ref_ok()
    test_file_in_other_refs_dir_rejected()
    test_file_outside_run_root_rejected()
    test_symlink_rejected()
    print("ALL V5 STAGE1B-PER-REF-CONTAINMENT GUARDS PASS")


if __name__ == "__main__":
    main()
