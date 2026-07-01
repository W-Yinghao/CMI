"""Guard (Stage-1B4): the FIT dataset view exposes NO raw reader / cohort-root attributes (they live in a closure). Its public
surface is read_windows + allowed_subject_keys + reads. Synthetic only."""
from __future__ import annotations
from acar.v5 import splits as SPL
from acar.v5.substrate import fit_dataset_view as FV
from acar.v5.tests._util import ok, FakeDevReader, stage1b_fake_subjects, stage1b_subject_index


def _view():
    subs = stage1b_fake_subjects()
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    allowed = set(split["train"]) | set(split["val"])
    cps = {c: f"/projects/dl/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    return FV.AuthorizedFitDatasetView(idx, allowed, FakeDevReader(subs), cps)


def test_no_raw_root_attributes():
    v = _view()
    for attr in ("_reader", "reader", "_cohort_paths", "cohort_paths", "_index", "index"):
        assert not hasattr(v, attr), f"view must not expose {attr!r} (raw roots live in the closure)"
    ok("view exposes NO _reader / _cohort_paths / index attributes (no raw-root escape)")


def test_public_surface_is_minimal():
    v = _view()
    public = {a for a in dir(v) if not a.startswith("__")}
    # only these public names (plus the private _reads audit backing the reads property)
    assert public <= {"read_windows", "allowed_subject_keys", "reads", "_reads"}, sorted(public)
    assert callable(v.read_windows) and isinstance(v.reads, list)
    ok("view public surface = read_windows + allowed_subject_keys + reads")


def main():
    print("ACAR v5 Stage-1B4 guard: dataset view public surface")
    test_no_raw_root_attributes()
    test_public_surface_is_minimal()
    print("ALL V5 STAGE1B-DATASET-VIEW-SURFACE GUARDS PASS")


if __name__ == "__main__":
    main()
