"""Guard (Stage-1B2): a built artifact manifest must carry the COMPLETE V5 registry hash set + matching (ref,disease,fold,seed).
Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_artifacts as ART
from acar.v5.tests._util import expect_raises, ok


def _art():
    a = {"ref": "PD/fold0/seed20260711", "disease": "PD", "fold": 0, "seed": 20260711}
    for h in P.REGISTRY_HASH_FIELDS:
        a[h] = "a" * 64
    return a


def test_complete_ok():
    assert ART.validate_artifact_manifest(_art(), expected_ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711)
    ok("a complete artifact manifest (all 6 registry hashes + matching keys) validates")


def test_missing_hash_rejected():
    a = _art()
    del a["feat_dump_sha256"]
    expect_raises(ART.Stage1bArtifactError, lambda: ART.validate_artifact_manifest(a, expected_ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711))
    ok("a missing registry hash field → Stage1bArtifactError")


def test_non_hex_rejected():
    a = _art()
    a["encoder_state_dict_sha256"] = "notahash"
    expect_raises(ART.Stage1bArtifactError, lambda: ART.validate_artifact_manifest(a, expected_ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711))
    ok("a non-64-hex registry hash → Stage1bArtifactError")


def test_ref_or_keys_mismatch_rejected():
    expect_raises(ART.Stage1bArtifactError, lambda: ART.validate_artifact_manifest(_art(), expected_ref="PD/fold1/seed20260711", disease="PD", fold=1, seed=20260711))
    expect_raises(ART.Stage1bArtifactError, lambda: ART.validate_artifact_manifest(_art(), expected_ref="PD/fold0/seed20260711", disease="SCZ", fold=0, seed=20260711))
    ok("artifact ref / (disease,fold,seed) mismatch → Stage1bArtifactError")


def main():
    print("ACAR v5 Stage-1B2 guard: artifact manifest hash set complete")
    test_complete_ok()
    test_missing_hash_rejected()
    test_non_hex_rejected()
    test_ref_or_keys_mismatch_rejected()
    print("ALL V5 STAGE1B-ARTIFACT-HASH-SET GUARDS PASS")


if __name__ == "__main__":
    main()
