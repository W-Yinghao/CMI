"""Guard (Stage-1B5): the file artifact writer rejects any artifact path that is a symlink — even one that resolves back INSIDE
output_root — so a symlink can never be used to smuggle bytes past the containment check. Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_file_artifact_writer as FW
from acar.v5.tests._util import expect_raises, ok

REF = "PD/fold0/seed20260711"


def _write(d, name, data):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _full_raw(root):
    raw = {"ref": REF, "disease": "PD", "fold": 0, "seed": 20260711}
    for pk in sorted(set(FW.FILE_SOURCE.values())):
        raw[pk] = _write(root, pk + ".bin", (REF + pk).encode())
    return raw


def test_symlink_pointing_outside_rejected():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
        raw = _full_raw(root)
        target = _write(outside, "secret.bin", b"held-out-bytes")
        link = os.path.join(root, "feat_dump_path.bin")
        os.remove(link)                                   # replace the real file with a symlink to outside
        os.symlink(target, link)
        raw["feat_dump_path"] = link
        expect_raises(FW.Stage1bFileArtifactError,
                      lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711, output_root=root))
    ok("artifact path that is a symlink to OUTSIDE output_root → Stage1bFileArtifactError")


def test_symlink_even_when_resolving_inside_rejected():
    with tempfile.TemporaryDirectory() as root:
        raw = _full_raw(root)
        target = _write(root, "real_target.bin", b"inside-bytes")   # symlink target IS inside output_root
        link = os.path.join(root, "feat_dump_path.bin")
        os.remove(link)
        os.symlink(target, link)
        raw["feat_dump_path"] = link
        # islink is checked BEFORE realpath containment → rejected regardless of where it resolves
        expect_raises(FW.Stage1bFileArtifactError,
                      lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711, output_root=root))
    ok("artifact path that is a symlink is rejected EVEN when it resolves inside output_root (islink before containment)")


def main():
    print("ACAR v5 Stage-1B5 guard: file writer rejects symlink escape")
    test_symlink_pointing_outside_rejected()
    test_symlink_even_when_resolving_inside_rejected()
    print("ALL V5 STAGE1B-FILE-WRITER-SYMLINK GUARDS PASS")


if __name__ == "__main__":
    main()
