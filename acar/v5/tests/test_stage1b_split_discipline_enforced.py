"""Guard (Stage-1B2): the build hands the trainer EXACTLY the fold's FIT (train/val) subjects, computed by the subject-disjoint
split. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, stage1b_fake_subjects

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def _disease_subjects(subs_by, disease):
    s = set()
    for c in P.DEV_COHORTS[disease]:
        s.update(subs_by[(disease, c)])
    return sorted(s)


def test_trainer_receives_exactly_fit_split():
    subs_by = stage1b_fake_subjects()
    trainer = FakeTrainer()
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(subs_by), trainer=trainer)
    assert rep["status"] == "STAGE1B_BUILT"
    for ref, got in trainer.received.items():
        disease = ref.split("/", 1)[0]
        fold = int(ref.split("fold")[1].split("/")[0])
        split = SPL.make_fold(_disease_subjects(subs_by, disease), fold)
        assert got["train"] == set(split["train"]), (ref, "train")
        assert got["val"] == set(split["val"]), (ref, "val")
        assert (got["train"] | got["val"]) == set(split["fit"]), (ref, "train∪val==fit")
    ok("for every fold ref the trainer received EXACTLY split[train]/split[val] (== fit); subject-disjoint split enforced")


def main():
    print("ACAR v5 Stage-1B2 guard: split discipline enforced")
    test_trainer_receives_exactly_fit_split()
    print("ALL V5 STAGE1B-SPLIT-DISCIPLINE GUARDS PASS")


if __name__ == "__main__":
    main()
