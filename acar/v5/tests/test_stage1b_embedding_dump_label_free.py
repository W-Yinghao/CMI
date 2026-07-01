"""Guard (Stage-1B5): the fold feature/embedding dump is LABEL-FREE — its view exposes no read_label and its records carry no
label field; the FIT training view (which CAN read labels) is a distinct type. Synthetic only."""
from __future__ import annotations
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1b_embedding_dump as ED
from acar.v5.substrate.embedding_dataset_view import AuthorizedEmbeddingDatasetView
from acar.v5.substrate.fit_dataset_view import AuthorizedFitDatasetView, DatasetViewAccessError
from acar.v5.tests._util import expect_raises, ok, FakeDevReader, stage1b_fake_subjects, stage1b_subject_index


def _views():
    subs = stage1b_fake_subjects()
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    fit = AuthorizedFitDatasetView(idx, set(split["train"]) | set(split["val"]), FakeDevReader(subs), cps)
    emb = AuthorizedEmbeddingDatasetView(idx, set(idx.subject_keys), FakeDevReader(subs), cps)   # ALL fold subjects
    return fit, emb, split


def test_dump_records_must_be_label_free():
    assert ED.validate_embedding_dump_label_free([{"subject_key": "PD/ds002778/sub-1", "z": [0.1, 0.2]}])
    for bad in ({"label": 1, "z": [0]}, {"y": 0}, {"diagnosis": "PD"}, {"target": 1}, {"case_control": 0}):
        expect_raises(ED.Stage1bEmbeddingDumpError, lambda bad=bad: ED.validate_embedding_dump_label_free(bad))
    ok("embedding dump records with any label-like field (label/y/diagnosis/target/case_control) → rejected")


def test_embedding_view_has_no_read_label():
    fit, emb, split = _views()
    assert hasattr(fit, "read_label"), "FIT training view CAN read labels"
    assert not hasattr(emb, "read_label"), "embedding view must NOT expose read_label"
    ED.assert_view_is_label_free(emb)
    expect_raises(ED.Stage1bEmbeddingDumpError, lambda: ED.assert_view_is_label_free(fit))   # FIT view rejected for dumping
    ok("embedding view exposes NO read_label; FIT view (has read_label) is rejected as a dump driver")


def test_embedding_view_reads_all_fold_subjects_no_label():
    fit, emb, split = _views()
    # the embedding view can read CAL/EVAL windows (label-free), which the FIT view refuses
    cal_key = list(split["cal"])[0]
    assert emb.read_windows(cal_key)["marker"].startswith("PD/")
    expect_raises(DatasetViewAccessError, lambda: fit.read_windows(cal_key))   # FIT view refuses CAL
    ok("embedding view reads ALL fold subjects (incl CAL/EVAL) label-free; FIT view still refuses CAL/EVAL")


def main():
    print("ACAR v5 Stage-1B5 guard: embedding dump label-free")
    test_dump_records_must_be_label_free()
    test_embedding_view_has_no_read_label()
    test_embedding_view_reads_all_fold_subjects_no_label()
    print("ALL V5 STAGE1B-EMBEDDING-DUMP-LABEL-FREE GUARDS PASS")


if __name__ == "__main__":
    main()
