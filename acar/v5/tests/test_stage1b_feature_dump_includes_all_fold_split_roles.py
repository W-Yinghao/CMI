"""Guard (Stage-1B7): across a full file-backed build (real trainer/dumper + fake numeric backend), every ref's feature dump is a
valid schema dump covering EVERY fold subject with all four split roles (train/val/cal/eval) present. Synthetic only."""
from __future__ import annotations
import tempfile
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1b_build as B
from acar.v5.tests._util import stage1b_repair_staging_root as _RSR
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_output_layout as LO
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.substrate import stage1b_embedding_orchestrator as ORC
from acar.v5.substrate import real_trainer as RT
from acar.v5.substrate import real_eegnet_trainer as RET
from acar.v5.tests._util import (ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeWindowsDevReader, FakeEegnetBackend,
                                 stage1b_fake_subjects, stage1b_subject_index)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
RUN = "run-syn-0001"


def test_every_ref_feat_dump_has_all_split_roles():
    subs_by = stage1b_fake_subjects()
    with tempfile.TemporaryDirectory() as d:
        rep = B.run_stage1b_real_build(
            stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL), stage1b_lock(protocol_tag_target_sha=FULL),
            output_root=d, repair_staging_root=_RSR(), dev_reader_factory=lambda ctx: FakeWindowsDevReader(subs_by),
            trainer_factory=lambda ctx: RT.RealSubstrateTrainer(ctx, backend=FakeEegnetBackend()),
            dumper_factory=lambda ctx: RET.RealEmbeddingDumper(ctx, backend=FakeEegnetBackend()))
        assert rep["n_registered"] == 30
        checked = 0
        for ref in SA.CANONICAL_FOLD_REFS:
            disease = ref.split("/", 1)[0]
            fold = int(ref.split("fold")[1].split("/")[0])
            split = SPL.make_fold(stage1b_subject_index(subs_by, disease).subject_keys, fold)
            n_subj = len(ORC.all_fold_subject_keys(split))
            feat = LO.ref_output_dir(d, RUN, ref) + "/feat_dump.npz"
            summ = FDW.parse_feature_dump(feat)
            assert summ["ref"] == ref and summ["n_records"] == n_subj, (ref, summ["n_records"], n_subj)
            assert set(summ["split_roles_present"]) == {"train", "val", "cal", "eval"}, (ref, summ["split_roles_present"])
            checked += 1
        assert checked == 30
    ok("all 30 refs: feat_dump.npz is a valid schema dump covering every fold subject with all 4 split roles present")


def main():
    print("ACAR v5 Stage-1B7 guard: feature dump includes all fold split roles")
    test_every_ref_feat_dump_has_all_split_roles()
    print("ALL V5 STAGE1B-FEATURE-DUMP-SPLIT-ROLES GUARDS PASS")


if __name__ == "__main__":
    main()
