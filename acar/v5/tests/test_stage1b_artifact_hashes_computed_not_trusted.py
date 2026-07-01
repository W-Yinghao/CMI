"""Guard (Stage-1B3): the artifact writer COMPUTES the registry hashes from the trainer's output bytes and IGNORES any
trainer-reported hash strings. Synthetic only."""
from __future__ import annotations
import hashlib
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_artifact_writer as AW
from acar.v5.tests._util import expect_raises, ok


def _raw(ref="PD/fold0/seed20260711"):
    raw = {"ref": ref, "disease": "PD", "fold": 0, "seed": 20260711}
    for bytes_key in set(AW.HASH_SOURCE.values()):
        raw[bytes_key] = f"{ref}:{bytes_key}".encode()
    return raw


def test_hashes_are_computed_from_bytes():
    art = AW.write_artifact(_raw(), expected_ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711)
    for hf, bk in AW.HASH_SOURCE.items():
        assert art[hf] == hashlib.sha256(f"PD/fold0/seed20260711:{bk}".encode()).hexdigest()
    ok("every registry hash == sha256 of its bytes payload (computed by the writer)")


def test_trainer_reported_hash_is_ignored():
    raw = _raw()
    raw["encoder_state_dict_sha256"] = "deadbeef" * 8         # a bogus trainer-reported hash
    art = AW.write_artifact(raw, expected_ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711)
    assert art["encoder_state_dict_sha256"] != "deadbeef" * 8
    assert art["encoder_state_dict_sha256"] == hashlib.sha256(raw["encoder_state_dict_bytes"]).hexdigest()
    ok("a trainer-reported hash string is IGNORED — the writer computes from bytes (not trusted)")


def test_non_bytes_payload_rejected():
    raw = _raw()
    raw["feat_dump_bytes"] = "not-bytes"
    expect_raises(AW.Stage1bArtifactWriteError,
                  lambda: AW.write_artifact(raw, expected_ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711))
    ok("a non-bytes payload → Stage1bArtifactWriteError")


def test_all_six_registry_hashes_covered():
    assert set(AW.HASH_SOURCE) == set(P.REGISTRY_HASH_FIELDS)
    ok("the writer covers exactly the 6 registry hash fields")


def main():
    print("ACAR v5 Stage-1B3 guard: artifact hashes computed not trusted")
    test_hashes_are_computed_from_bytes()
    test_trainer_reported_hash_is_ignored()
    test_non_bytes_payload_rejected()
    test_all_six_registry_hashes_covered()
    print("ALL V5 STAGE1B-ARTIFACT-HASHES-COMPUTED GUARDS PASS")


if __name__ == "__main__":
    main()
