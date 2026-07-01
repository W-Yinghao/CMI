"""Guard (Stage-1B6, req1+5): the per-fold build is TWO-PHASE — FIT-only training FIRST (no feat_dump), THEN a label-free feature
dump over ALL fold subjects. The trainer physically cannot emit feat_dump; the dumper is driven by a view with NO read_label. Also
exercises the REAL seam (RealSubstrateTrainer + RealEmbeddingDumper) end-to-end with a fake numeric backend + temp files.
Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1b_embedding_orchestrator as ORC
from acar.v5.substrate import stage1b_execution_context as EC
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import real_trainer as RT
from acar.v5.substrate import real_eegnet_trainer as RET
from acar.v5.tests._util import (ok, expect_raises, FakeDevReader, FakeTrainer, FakeDumper, FakeEegnetBackend,
                                 stage1b_fake_subjects, stage1b_subject_index, stage1b_auth, stage1b_full_plan)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
SEED = 20260711
REF = f"PD/fold0/seed{SEED}"


def _setup(fold=0):
    subs = stage1b_fake_subjects()
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, fold)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    return subs, idx, split, cps


def test_two_phase_merge_and_read_scopes():
    subs, idx, split, cps = _setup()
    tr, du = FakeTrainer(), FakeDumper()
    raw, sidecars = ORC.build_fold_raw("PD", 0, SEED, REF, idx, split, FakeDevReader(subs), tr, du, cps)
    for k in ("encoder_state_dict_bytes", "encoder_checkpoint_bytes", "source_state_artifact_bytes",
              "source_state_file_bytes", "preprocessing_config_bytes", "feat_dump_bytes"):
        assert k in raw, k
    fit = set(split["train"]) | set(split["val"])
    allk = ORC.all_fold_subject_keys(split)
    assert set(tr.reads[REF]) == fit                          # trainer read ONLY FIT
    assert set(du.reads[REF]) == set(allk)                    # dumper read EVERY fold subject
    assert (set(split["cal"]) | set(split["eval"])) & set(tr.reads[REF]) == set()   # CAL/EVAL never reached the trainer
    ok("two-phase merge: trainer read FIT-only + 5 model artifacts; dumper read ALL fold subjects + feat_dump")


def test_train_fold_cannot_emit_feat_dump():
    class _BadTrainer:
        def train_fold(self, disease, fold, seed, tk, vk, view):
            for k in list(tk) + list(vk):
                view.read_windows(k)
            return {"ref": f"{disease}/fold{fold}/seed{seed}", "disease": disease, "fold": fold, "seed": seed,
                    "feat_dump_bytes": b"leaked"}
    subs, idx, split, cps = _setup()
    expect_raises(ORC.Stage1bOrchestratorError,
                  lambda: ORC.build_fold_raw("PD", 0, SEED, REF, idx, split, FakeDevReader(subs), _BadTrainer(), FakeDumper(), cps))
    ok("a trainer that emits feat_dump during FIT training → Stage1bOrchestratorError (only the dumper may produce it)")


def test_embedding_view_handed_to_dumper_has_no_read_label():
    class _SpyDumper(FakeDumper):
        def dump_embeddings(self, disease, fold, seed, embedding_view, all_keys, train_result):
            self.view_has_read_label = hasattr(embedding_view, "read_label")
            return super().dump_embeddings(disease, fold, seed, embedding_view, all_keys, train_result)
    subs, idx, split, cps = _setup()
    du = _SpyDumper()
    ORC.build_fold_raw("PD", 0, SEED, REF, idx, split, FakeDevReader(subs), FakeTrainer(), du, cps)
    assert du.view_has_read_label is False
    ok("the view handed to the dumper has NO read_label (label-free by construction)")


def test_real_seam_two_phase_file_backed():
    subs, idx, split, cps = _setup()
    with tempfile.TemporaryDirectory() as d:
        ctx = EC.build_execution_context(stage1b_auth(protocol_tag_target_sha=FULL), {}, stage1b_full_plan(), output_root=d)
        be_t, be_d = FakeEegnetBackend(), FakeEegnetBackend()
        trainer, dumper = RT.RealSubstrateTrainer(ctx, backend=be_t), RET.RealEmbeddingDumper(ctx, backend=be_d)
        raw, sidecars = ORC.build_fold_raw("PD", 0, SEED, REF, idx, split, FakeDevReader(subs), trainer, dumper, cps)
        for pk in ("encoder_state_dict_path", "encoder_checkpoint_file_path", "source_state_artifact_path",
                   "source_state_file_path", "preprocessing_config_path", "feat_dump_path"):
            assert os.path.isfile(raw[pk]), pk
        assert "feat_dump_bytes" not in raw and os.path.isfile(sidecars["training_config_path"])
        assert be_t.seeds == [SEED] and be_t.fit_calls == [(len(split["train"]), len(split["val"]))]
        assert be_d.embed_calls == [len(ORC.all_fold_subject_keys(split))] and be_d.fit_calls == []
    ok("real seam: RealSubstrateTrainer fits FIT-only (seeded) + emits 5 files; RealEmbeddingDumper embeds ALL fold subjects → feat_dump")


def main():
    print("ACAR v5 Stage-1B6 guard: train then label-free dump order")
    test_two_phase_merge_and_read_scopes()
    test_train_fold_cannot_emit_feat_dump()
    test_embedding_view_handed_to_dumper_has_no_read_label()
    test_real_seam_two_phase_file_backed()
    print("ALL V5 STAGE1B-TRAIN-THEN-DUMP-ORDER GUARDS PASS")


if __name__ == "__main__":
    main()
