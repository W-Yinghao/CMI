"""Guard (Stage-1B7): the embedding dump is driven by the FROZEN artifacts of the SAME ref (a FrozenSubstrateHandle built from the
trainer's output files, ref/disease/fold/seed-matched) — not by an incidentally shared backend object. A mismatched or
missing-artifact train_result fails closed. Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5 import splits as SPL
from acar.v5.substrate import real_eegnet_trainer as RET
from acar.v5.substrate import embedding_dataset_view as EV
from acar.v5.substrate import stage1b_embedding_orchestrator as ORC
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import (ok, expect_raises, FakeWindowsDevReader, FakeEegnetBackend, make_subject_windows,
                                 stage1b_fake_subjects, stage1b_subject_index)

SEED = 20260711
REF = f"PD/fold0/seed{SEED}"


def _train_result(out_dir, fold=0):
    train = [("PD/ds002778/sub-001", make_subject_windows("PD/ds002778/sub-001"), 0),
             ("PD/ds002778/sub-002", make_subject_windows("PD/ds002778/sub-002"), 1)]
    val = [("PD/ds002778/sub-003", make_subject_windows("PD/ds002778/sub-003"), 0)]
    return RET.train_encoder_and_source_state("PD", fold, SEED, train, val, output_dir=out_dir, backend=FakeEegnetBackend())


def test_frozen_handle_binds_and_matches():
    with tempfile.TemporaryDirectory() as d:
        tr = _train_result(d)
        h = RET.FrozenSubstrateHandle.from_train_result(tr)
        h.assert_matches("PD", 0, SEED)                       # matches
        expect_raises(RET.RealEegnetError, lambda: h.assert_matches("SCZ", 0, SEED))
        expect_raises(RET.RealEegnetError, lambda: h.assert_matches("PD", 1, SEED))
        for miss in ("encoder_checkpoint_file_path", "source_state_file_path", "training_config_path"):
            bad = dict(tr)
            del bad[miss]
            expect_raises(RET.RealEegnetError, lambda bad=bad: RET.FrozenSubstrateHandle.from_train_result(bad))
        bad = dict(tr, source_state_file_path="/no/such/file")
        expect_raises(RET.RealEegnetError, lambda: RET.FrozenSubstrateHandle.from_train_result(bad))
    ok("FrozenSubstrateHandle binds the trainer's frozen files + matches ref/disease/fold/seed; missing/absent artifact → fail-closed")


def _emb_setup():
    subs = stage1b_fake_subjects(n_per_cohort=4)
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    all_keys = ORC.all_fold_subject_keys(split)
    role = ORC.split_role_by_subject(split)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    emb = EV.AuthorizedEmbeddingDatasetView(idx, set(all_keys), FakeWindowsDevReader(subs).windows_only(), cps)
    return emb, all_keys, role


def test_dump_driven_by_frozen_artifacts():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
        tr = _train_result(d)
        emb, all_keys, role = _emb_setup()
        be = FakeEegnetBackend()
        raw = RET.dump_fold_embeddings("PD", 0, SEED, emb, all_keys, tr, role, output_dir=o, backend=be)
        assert be.embed_frozen_refs == [REF]                  # the dump was driven by THIS ref's frozen substrate
        summ = FDW.parse_feature_dump(raw["feat_dump_path"])
        assert summ["n_records"] == len(all_keys) and summ["ref"] == REF
        assert set(summ["split_roles_present"]) == {"train", "val", "cal", "eval"}
    ok("dump_fold_embeddings loads the frozen artifacts (embed_from_artifacts) → schema feat dump covering all fold subjects/roles")


def test_dump_rejects_mismatched_frozen_ref():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
        tr_fold1 = _train_result(d, fold=1)                   # frozen substrate for fold1
        emb, all_keys, role = _emb_setup()
        expect_raises(RET.RealEegnetError,                    # dumping it as fold0 must fail closed
                      lambda: RET.dump_fold_embeddings("PD", 0, SEED, emb, all_keys, tr_fold1, role,
                                                       output_dir=o, backend=FakeEegnetBackend()))
    ok("a train_result for a DIFFERENT ref → dump fails closed (frozen substrate must match the dump target)")


def main():
    print("ACAR v5 Stage-1B7 guard: backend uses frozen artifacts for embedding")
    test_frozen_handle_binds_and_matches()
    test_dump_driven_by_frozen_artifacts()
    test_dump_rejects_mismatched_frozen_ref()
    print("ALL V5 STAGE1B-FROZEN-ARTIFACT-EMBEDDING GUARDS PASS")


if __name__ == "__main__":
    main()
