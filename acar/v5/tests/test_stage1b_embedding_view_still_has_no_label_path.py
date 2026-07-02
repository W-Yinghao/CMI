"""Guard (Stage-1B10): the new cohort-label mechanism does NOT open a label path in the embedding view — the label VALUE is still
reachable ONLY via AuthorizedFitDatasetView.read_label; the label-free embedding view (windows-only reader) has neither read_label
nor any reader with a label method; the eligibility resolver returns a boolean. Synthetic only."""
from __future__ import annotations
from acar.v5 import splits as SPL
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.substrate.fit_dataset_view import AuthorizedFitDatasetView
from acar.v5.substrate.embedding_dataset_view import AuthorizedEmbeddingDatasetView
from acar.v5.tests._util import ok, FakeDevReader, stage1b_fake_subjects, stage1b_subject_index


def test_label_value_only_via_fit_view():
    subs = stage1b_fake_subjects()
    idx = stage1b_subject_index(subs, "PD")
    split = SPL.make_fold(idx.subject_keys, 0)
    cps = {c: f"/p/{c}" for c in {idx.cohort_of(k) for k in idx.subject_keys}}
    fit = AuthorizedFitDatasetView(idx, set(split["train"]) | set(split["val"]), FakeDevReader(subs), cps)
    emb = AuthorizedEmbeddingDatasetView(idx, set(idx.subject_keys), FakeDevReader(subs).windows_only(), cps)
    assert hasattr(fit, "read_label") and not hasattr(emb, "read_label")
    # the embedding view's closure holds no object exposing a label method (read_label / read_subject_label), incl via __self__
    def _reaches_label(objs):
        for o in objs:
            if hasattr(o, "read_label") or hasattr(o, "read_subject_label"):
                return True
            s = getattr(o, "__self__", None)
            if s is not None and (hasattr(s, "read_label") or hasattr(s, "read_subject_label")):
                return True
        return False
    cells = [c.cell_contents for c in (emb.read_windows.__closure__ or ())]
    assert not _reaches_label(cells)
    ok("cohort labels change the value SOURCE only; embedding view still exposes NO label path (closure has none either)")


def test_eligibility_resolver_is_boolean():
    assert CLS.label_resolvable("PD", "ds002778", "sub-hc1", None) in (True, False)
    assert CLS.label_resolvable("PD", "ds002778", "sub-bogus", None) is False
    ok("cohort_label_spec.label_resolvable returns a boolean only (no label value leaks at eligibility)")


def main():
    print("ACAR v5 Stage-1B10 guard: embedding view still has no label path")
    test_label_value_only_via_fit_view()
    test_eligibility_resolver_is_boolean()
    print("ALL V5 STAGE1B-EMBEDDING-VIEW-NO-LABEL-PATH GUARDS PASS")


if __name__ == "__main__":
    main()
