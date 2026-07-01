"""Guard (Stage-1B5): the file artifact writer enforces output_root containment + rejects duplicate artifact paths. Synthetic
temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_file_artifact_writer as FW
from acar.v5.tests._util import expect_raises, ok

REF = "PD/fold0/seed20260711"


def _write(d, name, data=b"x"):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def test_contained_paths_ok():
    with tempfile.TemporaryDirectory() as root:
        raw = {"ref": REF, "disease": "PD", "fold": 0, "seed": 20260711}
        for i, pk in enumerate(sorted(set(FW.FILE_SOURCE.values()))):
            raw[pk] = _write(root, pk + ".bin", f"{REF}:{pk}".encode())
        art = FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711, output_root=root)
        assert all(len(art[h]) == 64 for h in FW.FILE_SOURCE)
    ok("all artifact files inside output_root → written + hashed")


def test_escape_rejected():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
        raw = {"ref": REF, "disease": "PD", "fold": 0, "seed": 20260711}
        for pk in sorted(set(FW.FILE_SOURCE.values())):
            raw[pk] = _write(root, pk + ".bin")
        raw["feat_dump_path"] = _write(outside, "leak.bin")    # OUTSIDE output_root
        expect_raises(FW.Stage1bFileArtifactError,
                      lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711, output_root=root))
    ok("an artifact file OUTSIDE output_root → Stage1bFileArtifactError (containment)")


def test_duplicate_path_rejected():
    with tempfile.TemporaryDirectory() as root:
        shared = _write(root, "shared.bin")
        raw = {"ref": REF, "disease": "PD", "fold": 0, "seed": 20260711}
        for pk in sorted(set(FW.FILE_SOURCE.values())):
            raw[pk] = shared                                   # same file reused for all 6 → duplicate
        expect_raises(FW.Stage1bFileArtifactError,
                      lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711, output_root=root))
    ok("the 6 artifact files must be distinct (a reused path → Stage1bFileArtifactError)")


def main():
    print("ACAR v5 Stage-1B5 guard: file writer output-root containment")
    test_contained_paths_ok()
    test_escape_rejected()
    test_duplicate_path_rejected()
    print("ALL V5 STAGE1B-FILE-WRITER-CONTAINMENT GUARDS PASS")


if __name__ == "__main__":
    main()
