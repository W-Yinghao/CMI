"""Guard (Stage-1B8): the dumper fail-closes unless each subject's embedding matrix has EXACTLY one row per input window
(rows == SubjectWindows.n_windows), is 2-D, dim>0, floating, finite. A backend that drops/adds rows is rejected. Non-torch
(FakeEegnetBackend + a deliberately-wrong backend). Synthetic temp files only."""
from __future__ import annotations
import tempfile
import numpy as np
from acar.v5 import splits as SPL
from acar.v5.substrate import real_eegnet_trainer as RET
from acar.v5.substrate import embedding_dataset_view as EV
from acar.v5.substrate import stage1b_embedding_orchestrator as ORC
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.tests._util import (ok, expect_raises, FakeWindowsDevReader, FakeEegnetBackend, make_subject_windows,
                                 stage1b_fake_subjects, stage1b_subject_index)

SEED = 20260711


def _train_result(out_dir):
    train = [("PD/ds002778/sub-001", make_subject_windows("PD/ds002778/sub-001"), 0),
             ("PD/ds002778/sub-002", make_subject_windows("PD/ds002778/sub-002"), 1)]
    val = [("PD/ds002778/sub-003", make_subject_windows("PD/ds002778/sub-003"), 0)]
    return RET.train_encoder_and_source_state("PD", 0, SEED, train, val, output_dir=out_dir, backend=FakeEegnetBackend())


def _emb_setup(n_windows=1):
    subs = stage1b_fake_subjects(n_per_cohort=4)
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    all_keys = ORC.all_fold_subject_keys(split)
    role = ORC.split_role_by_subject(split)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    # a reader whose SubjectWindows carry n_windows windows
    reader = FakeWindowsDevReader(subs)
    reader._nw = n_windows  # not used; make_subject_windows default 1

    class _NWReader:
        def __init__(self, subs, nw):
            self._subs, self._nw, self.read_calls = subs, nw, []

        def read_subject_windows(self, d, c, s, p):
            self.read_calls.append((d, c, s, p))
            return make_subject_windows(f"{d}/{c}/{s}", n_windows=self._nw)
    emb = EV.AuthorizedEmbeddingDatasetView(idx, set(all_keys), _NWReader(subs, n_windows), cps)
    return emb, all_keys, role


class _BadRowsBackend:
    """Returns 1 row per subject regardless of n_windows (drops windows)."""
    def embed_from_artifacts(self, windows_by_subject, frozen, training_config):
        return {sk: np.zeros((1, 4), dtype=np.float32) for sk in windows_by_subject}


class _BadDimBackend:
    def embed_from_artifacts(self, windows_by_subject, frozen, training_config):
        return {sk: np.zeros((getattr(windows_by_subject[sk], "n_windows", 1), 0), dtype=np.float32) for sk in windows_by_subject}


def test_rows_must_match_n_windows():
    emb, all_keys, role = _emb_setup(n_windows=3)             # 3 windows/subject; good backend returns 3 rows
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
        tr = _train_result(d)
        raw = RET.dump_fold_embeddings("PD", 0, SEED, emb, all_keys, tr, role, output_dir=o, backend=FakeEegnetBackend())
        loaded = FDW.load_feature_dump(raw["feat_dump_path"])
        from collections import Counter
        assert all(v == 3 for v in Counter(loaded["subject_key"]).values())   # 3 rows per subject
    ok("a backend emitting rows == n_windows (3) → dump has 3 records/subject")


def test_wrong_rows_and_dim_rejected():
    emb, all_keys, role = _emb_setup(n_windows=3)
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
        tr = _train_result(d)
        expect_raises(RET.RealEegnetError,
                      lambda: RET.dump_fold_embeddings("PD", 0, SEED, emb, all_keys, tr, role, output_dir=o, backend=_BadRowsBackend()))
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
        tr = _train_result(d)
        expect_raises(RET.RealEegnetError,
                      lambda: RET.dump_fold_embeddings("PD", 0, SEED, emb, all_keys, tr, role, output_dir=o, backend=_BadDimBackend()))
    ok("a backend with rows != n_windows, or embedding_dim == 0 → RealEegnetError (fail-closed before writing the dump)")


def main():
    print("ACAR v5 Stage-1B8 guard: embedding rows match windows")
    test_rows_must_match_n_windows()
    test_wrong_rows_and_dim_rejected()
    print("ALL V5 STAGE1B-EMBEDDING-ROWS-MATCH-WINDOWS GUARDS PASS")


if __name__ == "__main__":
    main()
