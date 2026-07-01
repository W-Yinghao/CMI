"""Guard (Stage-1B6, req1): across a full build, the label-free feature dump covers EVERY fold subject (train∪val∪cal∪eval) for
all 30 refs, while the trainer only ever sees FIT — so CAL/EVAL are reachable ONLY through the label-free embedding view.
Synthetic only."""
from __future__ import annotations
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_embedding_orchestrator as ORC
from acar.v5.tests._util import (ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, FakeDumper,
                                 stage1b_fake_subjects, stage1b_subject_index)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_dump_covers_all_fold_subjects_trainer_only_fit():
    subs_by = stage1b_fake_subjects()
    trainer, dumper = FakeTrainer(), FakeDumper()
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(subs_by), trainer=trainer, dumper=dumper)
    assert rep["status"] == "STAGE1B_BUILT" and rep["n_registered"] == 30
    checked = 0
    for ref in SA.CANONICAL_FOLD_REFS:
        disease = ref.split("/", 1)[0]
        fold = int(ref.split("fold")[1].split("/")[0])
        split = SPL.make_fold(stage1b_subject_index(subs_by, disease).subject_keys, fold)
        fit = set(split["train"]) | set(split["val"])
        allk = set(ORC.all_fold_subject_keys(split))
        assert set(dumper.reads[ref]) == allk, (ref, "dumper must cover ALL fold subjects")
        assert set(trainer.reads[ref]) == fit, (ref, "trainer must read FIT only")
        cal_eval = set(split["cal"]) | set(split["eval"])
        assert cal_eval <= set(dumper.reads[ref]) and cal_eval & set(trainer.reads[ref]) == set()
        checked += 1
    assert checked == 30
    ok("all 30 refs: dumper reads train∪val∪cal∪eval; trainer reads FIT only; CAL/EVAL only via the embedding view")


def main():
    print("ACAR v5 Stage-1B6 guard: embedding dump all fold subjects")
    test_dump_covers_all_fold_subjects_trainer_only_fit()
    print("ALL V5 STAGE1B-EMBEDDING-DUMP-ALL-FOLD GUARDS PASS")


if __name__ == "__main__":
    main()
