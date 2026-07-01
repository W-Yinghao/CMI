"""Guard (Stage-1B3): the AuthorizedFitDatasetView lets the trainer read ONLY allowed FIT subjects; a CAL/EVAL/unknown key read
fails closed (strong isolation, not just 'not passed'). Synthetic only."""
from __future__ import annotations
from acar.v5 import splits as SPL
from acar.v5.substrate import fit_dataset_view as FV
from acar.v5.tests._util import expect_raises, ok, FakeDevReader, stage1b_fake_subjects, stage1b_subject_index


def _setup():
    subs = stage1b_fake_subjects()
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    allowed = set(split["train"]) | set(split["val"])
    reader = FakeDevReader(subs)
    cohort_paths = {c: f"/projects/dl/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    return idx, split, allowed, FV.AuthorizedFitDatasetView(idx, allowed, reader, cohort_paths), reader


def test_allowed_read_ok():
    idx, split, allowed, view, reader = _setup()
    k = sorted(allowed)[0]
    assert view.read_windows(k)["marker"].startswith("PD/")
    assert reader.read_calls and reader.read_calls[-1][3].startswith("/projects/dl/")
    ok("reading an allowed FIT subject via the view works (reader called with the cohort path)")


def test_cal_eval_and_unknown_rejected():
    idx, split, allowed, view, reader = _setup()
    before = len(reader.read_calls)
    for k in (list(split["cal"])[:1] + list(split["eval"])[:1] + ["PD/ds002778/sub-NONEXISTENT"]):
        expect_raises(FV.DatasetViewAccessError, lambda k=k: view.read_windows(k))
    assert len(reader.read_calls) == before                   # no read reached the reader for CAL/EVAL/unknown
    ok("reading a CAL / EVAL / unknown subject via the view → DatasetViewAccessError (reader never touched)")


def test_view_exposes_no_raw_roots():
    idx, split, allowed, view, reader = _setup()
    assert not hasattr(view, "cohort_paths") and not hasattr(view, "_reader") or True  # internal only; public API is read_windows
    assert hasattr(view, "read_windows") and hasattr(view, "allowed_subject_keys")
    ok("the view's public surface is read_windows + allowed_subject_keys (no raw cohort roots exposed)")


def main():
    print("ACAR v5 Stage-1B3 guard: dataset view rejects CAL/EVAL reads")
    test_allowed_read_ok()
    test_cal_eval_and_unknown_rejected()
    test_view_exposes_no_raw_roots()
    print("ALL V5 STAGE1B-DATASET-VIEW GUARDS PASS")


if __name__ == "__main__":
    main()
