"""Synthetic guards for acar/v3/data.py (structured keys + label-firewall types + batching). TOY ONLY — no DEV
cohort is read. Run: python -m acar.v3.tests.test_data
"""
from dataclasses import FrozenInstanceError, fields
import numpy as np

from acar.config import MIN_BATCH, B
from acar.v3.set_features import WindowKey, NON_IDENTITY
from acar.v3.data import (SubjectKey, RecordingKey, DeploymentBatch, LabeledRiskRecord, canon_subject, subject_of,
                          build_deployment_batches, deployment_batch_digest, make_synthetic)


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_subjectkey_disambiguation():
    batches = make_synthetic(n_datasets=2, subj_per=3, rec_per=1, win_per=10)
    subs = {b.subject for b in batches}
    # same local id 'sub-000' in both datasets must be DISTINCT SubjectKeys
    assert SubjectKey("ds000", "sub-000") in subs and SubjectKey("ds001", "sub-000") in subs
    assert canon_subject(SubjectKey("ds000", "sub-000")) != canon_subject(SubjectKey("ds001", "sub-000"))
    print("  [ok] SubjectKey disambiguates identical local ids across datasets (no cross-dataset merge)")


def test_deployment_batch_no_label_and_immutable():
    fnames = {f.name for f in fields(DeploymentBatch)}
    assert not (fnames & {"y", "label", "labels", "delta_r", "target"}), "DeploymentBatch carries a label field"
    b = make_synthetic(n_datasets=1, subj_per=1, rec_per=1, win_per=10)[0]
    _expect(FrozenInstanceError, lambda: setattr(b, "fallback", True))
    assert not b.z.flags.writeable
    _expect(ValueError, lambda: DeploymentBatch(                              # window key subject != batch subject
        SubjectKey("dsX", "s1"), RecordingKey("dsX", "s1", "r1"),
        (WindowKey("dsX", "OTHER", "r1", 0),), np.zeros((1, 4)), True, "src::x"))
    print("  [ok] DeploymentBatch has NO label field; frozen + z read-only; key/subject mismatch rejected")


def test_labeled_risk_record_validation():
    good = tuple((a, 0.1) for a in NON_IDENTITY)
    LabeledRiskRecord("a" * 64, good)
    _expect(ValueError, lambda: LabeledRiskRecord("a" * 64, good[:-1]))      # missing action
    _expect(ValueError, lambda: LabeledRiskRecord("short", good))           # digest not full sha256
    _expect(ValueError, lambda: LabeledRiskRecord("a" * 64, tuple((a, float("nan")) for a in NON_IDENTITY)))
    print("  [ok] LabeledRiskRecord requires exact non-identity actions, finite ΔR, full-64 digest")


def test_batching_window_order_and_fallback():
    rows = [("sub-1", "rec-1", w, np.full(4, float(w))) for w in [3, 1, 2, 0]]      # out-of-order windows
    rows += [("sub-1", "rec-2", w, np.zeros(4)) for w in range(MIN_BATCH - 1)]      # tiny recording -> fallback
    batches = build_deployment_batches("dsZ", rows, "src::z", batch_size=B)
    big = [b for b in batches if b.recording.recording_id == "rec-1"][0]
    assert [wk.window_index for wk in big.window_keys] == [0, 1, 2, 3], "windows not sorted"
    small = [b for b in batches if b.recording.recording_id == "rec-2"][0]
    assert small.fallback and len(small.window_keys) == MIN_BATCH - 1
    print("  [ok] build_deployment_batches: window-ordered, B-chunked, <MIN_BATCH retained as fallback")


def test_digest_order_insensitive_value_sensitive():
    r1 = [("s", "r", w, np.full(4, float(w))) for w in range(10)]
    r2 = list(reversed(r1))
    d1 = deployment_batch_digest(build_deployment_batches("ds", r1, "src::a")[0])
    d2 = deployment_batch_digest(build_deployment_batches("ds", r2, "src::a")[0])
    assert d1 == d2 and len(d1) == 64                                        # row input order irrelevant
    r3 = [("s", "r", w, np.full(4, float(w) + (0.001 if w == 5 else 0.0))) for w in range(10)]
    assert deployment_batch_digest(build_deployment_batches("ds", r3, "src::a")[0]) != d1
    assert deployment_batch_digest(build_deployment_batches("ds", r1, "src::OTHER")[0]) != d1   # source_state_ref in digest
    print("  [ok] deployment_batch_digest: full-64, input-order-insensitive, z/source-sensitive")


def main():
    print("ACAR v3 data-layer synthetic guards:")
    for t in (test_subjectkey_disambiguation, test_deployment_batch_no_label_and_immutable,
              test_labeled_risk_record_validation, test_batching_window_order_and_fallback,
              test_digest_order_insensitive_value_sensitive):
        t()
    print("ALL V3 DATA-LAYER GUARDS PASS")


if __name__ == "__main__":
    main()
