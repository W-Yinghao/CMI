"""Guard (Stage-1B6): the label source maps ONLY to the pinned {control:0, case:1} classes, fail-closed on unknown/missing/
ambiguous; the participants.tsv reader is deterministic + fail-closed; and labels are reachable ONLY through the FIT training view
(the embedding view has no label path). Synthetic tsv + synthetic views only."""
from __future__ import annotations
import os
import tempfile
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1b_label_source as LS
from acar.v5.substrate.fit_dataset_view import AuthorizedFitDatasetView
from acar.v5.substrate.embedding_dataset_view import AuthorizedEmbeddingDatasetView
from acar.v5.tests._util import expect_raises, ok, FakeDevReader, stage1b_fake_subjects, stage1b_subject_index


def test_mapping_control_case():
    assert LS.resolve_label("control") == 0 and LS.resolve_label("HC") == 0 and LS.resolve_label("healthy_control") == 0
    assert LS.resolve_label("case") == 1 and LS.resolve_label("patient") == 1 and LS.resolve_label("PD") == 1
    ok("control/HC/healthy_control → 0 ; case/patient/PD → 1")


def test_mapping_fail_closed():
    for bad in (None, "", "unknown", "maybe", 5, "n/a"):
        expect_raises(LS.LabelSourceError, lambda bad=bad: LS.resolve_label(bad))
    ok("None / empty / unknown / non-string group → LabelSourceError (fail-closed, never defaulted)")


def _tsv(dir_, rows, header="participant_id\tgroup"):
    p = os.path.join(dir_, "participants.tsv")
    with open(p, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return p


def test_participants_tsv_reader():
    with tempfile.TemporaryDirectory() as d:
        p = _tsv(d, [("sub-001", "control"), ("sub-002", "case")])
        assert LS.resolve_subject_label(p, "sub-001") == 0 and LS.resolve_subject_label(p, "sub-002") == 1
        expect_raises(LS.LabelSourceError, lambda: LS.resolve_subject_label(p, "sub-999"))          # missing subject
        expect_raises(LS.LabelSourceError, lambda: LS.resolve_subject_label(os.path.join(d, "nope.tsv"), "sub-001"))
    with tempfile.TemporaryDirectory() as d:
        p = _tsv(d, [("sub-001", "x")], header="participant_id\tage")                                # no group column
        expect_raises(LS.LabelSourceError, lambda: LS.resolve_subject_label(p, "sub-001"))
    with tempfile.TemporaryDirectory() as d:
        p = _tsv(d, [("sub-001", "control"), ("sub-001", "case")])                                   # duplicate subject
        expect_raises(LS.LabelSourceError, lambda: LS.resolve_subject_label(p, "sub-001"))
    ok("participants.tsv reader: correct map; missing file/subject, no group column, duplicate subject → fail-closed")


def test_labels_reachable_only_via_fit_view():
    subs = stage1b_fake_subjects()
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    fit = AuthorizedFitDatasetView(idx, set(split["train"]) | set(split["val"]), FakeDevReader(subs), cps)
    emb = AuthorizedEmbeddingDatasetView(idx, set(idx.subject_keys), FakeDevReader(subs).windows_only(), cps)
    assert hasattr(fit, "read_label") and not hasattr(emb, "read_label")
    fit.read_label(list(split["train"])[0])                   # a FIT label read works (via the reader's label seam)
    ok("labels reachable ONLY via AuthorizedFitDatasetView.read_label; the embedding view has no label path")


def main():
    print("ACAR v5 Stage-1B6 guard: label loading FIT-only")
    test_mapping_control_case()
    test_mapping_fail_closed()
    test_participants_tsv_reader()
    test_labels_reachable_only_via_fit_view()
    print("ALL V5 STAGE1B-LABEL-LOADING-FIT-ONLY GUARDS PASS")


if __name__ == "__main__":
    main()
