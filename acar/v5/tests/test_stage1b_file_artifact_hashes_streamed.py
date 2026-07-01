"""Guard (Stage-1B4): the file-backed artifact writer streams sha256 from real output files, rejects missing/empty files, and
ignores trainer-reported hashes. Synthetic only (temp files)."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_file_artifact_writer as FW
from acar.v5.tests._util import expect_raises, ok

REF = "PD/fold0/seed20260711"


def _raw_with_files(d, contents=None):
    raw = {"ref": REF, "disease": "PD", "fold": 0, "seed": 20260711}
    payloads = {}
    for hf, pk in FW.FILE_SOURCE.items():
        p = os.path.join(d, pk + ".bin")
        data = (contents if contents is not None else f"{REF}:{pk}").encode()
        with open(p, "wb") as f:
            f.write(data)
        raw[pk] = p
        payloads[hf] = data
    return raw, payloads


def test_streams_sha_from_files():
    with tempfile.TemporaryDirectory() as d:
        raw, payloads = _raw_with_files(d)
        art = FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711)
        for hf in P.REGISTRY_HASH_FIELDS:
            assert art[hf] == hashlib.sha256(payloads[hf]).hexdigest()
    ok("write_artifact_from_files streams sha256 from each output file (matches direct hash)")


def test_missing_and_empty_files_rejected():
    with tempfile.TemporaryDirectory() as d:
        raw, _ = _raw_with_files(d)
        raw["feat_dump_path"] = os.path.join(d, "nope.bin")
        expect_raises(FW.Stage1bFileArtifactError, lambda: FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711))
        empty = os.path.join(d, "empty.bin")
        open(empty, "wb").close()
        raw2, _ = _raw_with_files(d)
        raw2["encoder_checkpoint_file_path"] = empty
        expect_raises(FW.Stage1bFileArtifactError, lambda: FW.write_artifact_from_files(raw2, expected_ref=REF, disease="PD", fold=0, seed=20260711))
    ok("missing file / empty file → Stage1bFileArtifactError")


def test_reported_hash_ignored():
    with tempfile.TemporaryDirectory() as d:
        raw, payloads = _raw_with_files(d)
        raw["encoder_state_dict_sha256"] = "deadbeef" * 8      # bogus reported hash
        art = FW.write_artifact_from_files(raw, expected_ref=REF, disease="PD", fold=0, seed=20260711)
        assert art["encoder_state_dict_sha256"] == hashlib.sha256(payloads["encoder_state_dict_sha256"]).hexdigest()
    ok("a trainer-reported hash string is ignored — the writer streams from the file (computed, not trusted)")


def main():
    print("ACAR v5 Stage-1B4 guard: file artifact hashes streamed")
    test_streams_sha_from_files()
    test_missing_and_empty_files_rejected()
    test_reported_hash_ignored()
    print("ALL V5 STAGE1B-FILE-ARTIFACT-HASHES GUARDS PASS")


if __name__ == "__main__":
    main()
