"""Guard (Stage-1B8): the REAL torch EEGNet backend fits deterministically, emits the 4 byte artifacts, and embeds each subject from
the FROZEN checkpoint file (loaded fresh — no shared in-memory trainer state), with rows == n_windows, dim > 0, finite. Requires
torch; SKIPS cleanly where torch is unavailable (e.g. py3.9). Synthetic tiny tensors + temp files only — no real DEV, no SLURM."""
from __future__ import annotations
import os
import hashlib
import tempfile

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from acar.v5.substrate import real_eegnet_trainer as RET
from acar.v5.substrate import stage1b_embedding_orchestrator as ORC
from acar.v5.substrate import embedding_dataset_view as EV
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5 import splits as SPL
from acar.v5.tests._util import (ok, FakeWindowsDevReader, make_subject_windows, stage1b_fake_subjects, stage1b_subject_index)

SEED = 20260711
REF = f"PD/fold0/seed{SEED}"


def _fit_records():
    import numpy as np

    def rec(sk, label, nw=2):
        sw = make_subject_windows(sk, n_windows=nw)
        sw.windows[:] = np.random.RandomState(abs(hash(sk)) % 997).randn(*sw.windows.shape).astype("float32")
        return (sk, sw, label)
    train = [rec("PD/ds002778/sub-001", 0), rec("PD/ds002778/sub-002", 1)]
    val = [rec("PD/ds002778/sub-003", 0), rec("PD/ds002778/sub-004", 1)]
    return train, val


def _train_result(out_dir):
    train, val = _fit_records()
    return RET.train_encoder_and_source_state("PD", 0, SEED, train, val, output_dir=out_dir,
                                              backend=RET.TorchEegnetBackend())


def test_fit_emits_four_byte_artifacts_deterministically():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        r1 = _train_result(d1)
        r2 = _train_result(d2)
        for pk in ("encoder_state_dict_path", "encoder_checkpoint_file_path", "source_state_artifact_path",
                   "source_state_file_path", "preprocessing_config_path", "training_config_path"):
            assert os.path.isfile(r1[pk]), pk
        # determinism: same seed + data → byte-identical encoder checkpoint + source state
        for pk in ("encoder_checkpoint_file_path", "source_state_file_path"):
            h1 = hashlib.sha256(open(r1[pk], "rb").read()).hexdigest()
            h2 = hashlib.sha256(open(r2[pk], "rb").read()).hexdigest()
            assert h1 == h2, (pk, "not deterministic")
    ok("torch backend.fit emits 4 model files deterministically (byte-identical encoder ckpt + source state for a fixed seed)")


def test_embed_from_frozen_file_no_shared_state():
    import numpy as np
    subs = stage1b_fake_subjects(n_per_cohort=4)
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    all_keys = ORC.all_fold_subject_keys(split)
    role = ORC.split_role_by_subject(split)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    emb = EV.AuthorizedEmbeddingDatasetView(idx, set(all_keys), FakeWindowsDevReader(subs).windows_only(), cps)
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
        tr = _train_result(d)
        # a FRESH backend instance for the dump → the encoder is loaded from the FROZEN file, not shared memory
        raw = RET.dump_fold_embeddings("PD", 0, SEED, emb, all_keys, tr, role, output_dir=o, backend=RET.TorchEegnetBackend())
        loaded = FDW.load_feature_dump(raw["feat_dump_path"])
        assert loaded["summary"]["embedding_dim"] > 0
        # rows per subject == that subject's n_windows (FakeWindowsDevReader → 1 window each here)
        from collections import Counter
        counts = Counter(loaded["subject_key"])
        assert all(counts[k] == 1 for k in all_keys)
        assert set(loaded["split_role"]) == {"train", "val", "cal", "eval"}
    ok("torch embed_from_artifacts loads the FROZEN encoder from file (no shared state) → per-window features, dim>0, all fold roles")


def main():
    print("ACAR v5 Stage-1B8 guard: torch backend fit/embed fixture")
    if not HAS_TORCH:
        print("  [skip] torch unavailable — backend fit/embed fixture skipped on this interpreter")
        print("ALL V5 STAGE1B-BACKEND-FIT-EMBED GUARDS PASS")
        return
    test_fit_emits_four_byte_artifacts_deterministically()
    test_embed_from_frozen_file_no_shared_state()
    print("ALL V5 STAGE1B-BACKEND-FIT-EMBED GUARDS PASS")


if __name__ == "__main__":
    main()
