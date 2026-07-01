"""Guard (Stage-1B2): an executed build (synthetic reader+trainer) produces EXACTLY the 30 fold-contained substrates, each with a
valid artifact manifest. Synthetic only — no real data, no torch."""
from __future__ import annotations
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_artifacts as ART
from acar.v5.tests._util import ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_build_produces_exactly_30():
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(), trainer=FakeTrainer())
    assert rep["status"] == "STAGE1B_BUILT" and rep["n_artifacts"] == 30
    assert set(rep["artifacts"]) == set(SA.CANONICAL_FOLD_REFS)
    for ref, art in rep["artifacts"].items():
        d, rest = ref.split("/", 1)
        fold = int(rest.split("/")[0][4:])
        seed = int(rest.split("seed")[1])
        ART.validate_artifact_manifest(art, expected_ref=ref, disease=d, fold=fold, seed=seed)
    ok("executed build → STAGE1B_BUILT, exactly 30 artifacts == canonical fold refs, each artifact valid")


def main():
    print("ACAR v5 Stage-1B2 guard: outputs exact 30 refs")
    test_build_produces_exactly_30()
    print("ALL V5 STAGE1B-OUTPUTS-30-REFS GUARDS PASS")


if __name__ == "__main__":
    main()
