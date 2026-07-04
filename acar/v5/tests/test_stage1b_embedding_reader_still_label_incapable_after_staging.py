"""Guard (Stage-1B15): adding the repair staging root did NOT open a label path. The WindowsOnlyReader facade still carries ONLY the
(label-free) execution context — no read_subject_label, no back-reference to the label-capable reader — and the embedding view built
from it reaches no label through its closure."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import real_dev_reader as RDR
from acar.v5.substrate import subject_index as SI
from acar.v5.substrate.embedding_dataset_view import AuthorizedEmbeddingDatasetView
from acar.v5.tests._util import ok, stage1b_reader_ctx


def test_windows_only_reader_is_label_incapable_after_staging():
    root = tempfile.mkdtemp()
    reader = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", "/p", root))
    wor = reader.windows_only()
    assert isinstance(wor, RDR.WindowsOnlyReader)
    assert not hasattr(wor, "read_subject_label")
    assert set(wor.__dict__) == {"_ctx"}                        # ONLY the context — no back-ref to the label-capable reader
    assert not hasattr(wor._ctx, "read_subject_label")          # the context itself carries no label capability
    ok("WindowsOnlyReader after Stage-1B15 staging still holds only the label-free context (no read_subject_label / no back-ref)")


def test_embedding_view_from_staged_windows_only_reader_reaches_no_label():
    root = tempfile.mkdtemp()
    wor = RDR.make_real_dev_reader(stage1b_reader_ctx("PD", "ds002778", "/p", root)).windows_only()
    idx = SI.build_subject_index("PD", {"ds002778": ["sub-1"], "ds003490": ["sub-2"], "ds004584": ["sub-3"]})
    view = AuthorizedEmbeddingDatasetView(idx, idx.subject_keys, wor,
                                          {"ds002778": "/p", "ds003490": "/q", "ds004584": "/r"})   # accepts a label-free reader
    assert not hasattr(view, "read_subject_label") and not hasattr(view, "read_label")
    for cell in (view.read_windows.__closure__ or ()):         # the captured reader (+ its ctx) exposes no label read
        assert not hasattr(cell.cell_contents, "read_subject_label")
    ok("the embedding view built from the staged windows-only reader reaches no read_subject_label via its closure")


def main():
    print("ACAR v5 Stage-1B15 guard: embedding reader still label-incapable after staging")
    test_windows_only_reader_is_label_incapable_after_staging()
    test_embedding_view_from_staged_windows_only_reader_reaches_no_label()
    print("ALL V5 STAGE1B15-EMBEDDING-READER-LABEL-INCAPABLE GUARDS PASS")


if __name__ == "__main__":
    main()
