"""Guard (Stage-1B6, req7): finalize validates the emitted config FILES are canonical — the preprocessing_config file must equal
preprocessing_config.canonical_json() (and its hash == config_sha256()), and the training_config sidecar must equal
training_config.canonical_json(); the training_config_sha256 is recorded in registry meta (NOT among the six registry hash fields).
A tampered config file fails the finalize barrier. Synthetic temp files only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_finalize as FIN
from acar.v5.substrate import training_config as TC
from acar.v5 import protocol as P
from acar.v5.tests._util import (expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader,
                                 FakeFileTrainer, FakeFileDumper)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
RUN = "run-syn-0001"


def _run(d, trainer_cls):
    return B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                    stage1b_lock(protocol_tag_target_sha=FULL), output_root=d,
                                    dev_reader_factory=lambda ctx: FakeDevReader(),
                                    trainer_factory=lambda ctx: trainer_cls(ctx.output_root, ctx.run_id),
                                    dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id))


def test_canonical_configs_ok_and_meta_recorded():
    with tempfile.TemporaryDirectory() as d:
        rep = _run(d, FakeFileTrainer)
        assert rep["n_registered"] == 30
        reg = rep["registry"]
        for ref in SA.CANONICAL_FOLD_REFS:
            meta = reg._entries[ref]["meta"]
            assert meta["training_config_sha256"] == TC.config_sha256()
            assert "training_config_sha256" not in P.REGISTRY_HASH_FIELDS   # it is META, not a registry hash field
        assert os.path.exists(FIN.marker_path(d, RUN))
    ok("canonical config sidecars → build finalizes; training_config_sha256 recorded in meta (not a registry hash field)")


class _BadPreproc(FakeFileTrainer):
    def train_fold(self, *a, **k):
        raw = super().train_fold(*a, **k)
        with open(raw["preprocessing_config_path"], "w") as f:
            f.write("NOT_CANONICAL_PREPROCESSING")
        return raw


class _BadTrainCfg(FakeFileTrainer):
    def train_fold(self, *a, **k):
        raw = super().train_fold(*a, **k)
        with open(raw["training_config_path"], "w") as f:
            f.write("NOT_CANONICAL_TRAINING")
        return raw


def test_tampered_preprocessing_config_rejected():
    with tempfile.TemporaryDirectory() as d:
        expect_raises(FIN.Stage1bFinalizeError, lambda: _run(d, _BadPreproc))
        assert not os.path.exists(FIN.marker_path(d, RUN))
    ok("a non-canonical preprocessing_config file → finalize barrier fails; no FINALIZED marker")


def test_tampered_training_config_rejected():
    with tempfile.TemporaryDirectory() as d:
        expect_raises(FIN.Stage1bFinalizeError, lambda: _run(d, _BadTrainCfg))
        assert not os.path.exists(FIN.marker_path(d, RUN))
    ok("a non-canonical training_config sidecar → finalize barrier fails; no FINALIZED marker")


def main():
    print("ACAR v5 Stage-1B6 guard: config files canonical")
    test_canonical_configs_ok_and_meta_recorded()
    test_tampered_preprocessing_config_rejected()
    test_tampered_training_config_rejected()
    print("ALL V5 STAGE1B-CONFIG-FILES-CANONICAL GUARDS PASS")


if __name__ == "__main__":
    main()
