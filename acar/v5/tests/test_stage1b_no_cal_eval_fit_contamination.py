"""Guard (Stage-1B2): CAL and EVAL subjects are NEVER handed to the trainer (no split contamination). Synthetic only."""
from __future__ import annotations
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, FakeDumper, stage1b_fake_subjects, stage1b_subject_index

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_no_cal_or_eval_reaches_trainer():
    subs_by = stage1b_fake_subjects()
    trainer = FakeTrainer()
    B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                        stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                        dev_reader=FakeDevReader(subs_by), trainer=trainer, dumper=FakeDumper())
    total_checked = 0
    for ref, got in trainer.received.items():
        disease = ref.split("/", 1)[0]
        fold = int(ref.split("fold")[1].split("/")[0])
        split = SPL.make_fold(stage1b_subject_index(subs_by, disease).subject_keys, fold)
        forbidden = set(split["cal"]) | set(split["eval"])
        seen = got["train"] | got["val"]
        assert seen & forbidden == set(), (ref, "CAL/EVAL passed to trainer", sorted(seen & forbidden)[:3])
        assert seen <= set(split["fit"]), (ref, "trainer received a non-FIT subject")
        assert set(trainer.reads[ref]) & forbidden == set(), (ref, "CAL/EVAL READ via view")   # view-level isolation
        total_checked += 1
    assert total_checked == 30
    ok(f"all {total_checked} folds: trainer received + read ONLY FIT subjects; CAL/EVAL never passed nor readable")


def main():
    print("ACAR v5 Stage-1B2 guard: no CAL/EVAL/FIT contamination")
    test_no_cal_or_eval_reaches_trainer()
    print("ALL V5 STAGE1B-NO-CONTAMINATION GUARDS PASS")


if __name__ == "__main__":
    main()
