"""Synthetic guards for acar/v3/data.py (structured keys + label firewall + batching-protocol validation). TOY ONLY.
Run: python -m acar.v3.tests.test_data
"""
import hashlib
from dataclasses import FrozenInstanceError, fields
import numpy as np

from acar.config import MIN_BATCH, B
from acar.v3.set_features import WindowKey, NON_IDENTITY
from acar.v3.data import (SubjectKey, RecordingKey, DeploymentBatch, LabeledRiskRecord, canon_subject,
                          build_deployment_batches, deployment_batch_digest, make_synthetic)

HEX = hashlib.sha256(b"src").hexdigest()


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _batch(n=MIN_BATCH, fallback=None, src=HEX, disease="PD", neg=False, empty_id=False):
    keys = tuple(WindowKey("ds", "" if empty_id else "s1", "r1", (-1 if (neg and i == 0) else i)) for i in range(n))
    z = np.zeros((n, 4))
    fb = (n < MIN_BATCH) if fallback is None else fallback
    return DeploymentBatch(disease, SubjectKey("ds", "s1"), RecordingKey("ds", "s1", "r1"), keys, z, fb, src)


def test_subjectkey_disambiguation():
    batches = make_synthetic(n_datasets=2, subj_per=3, rec_per=1, win_per=10)
    subs = {b.subject for b in batches}
    assert SubjectKey("ds000", "sub-000") in subs and SubjectKey("ds001", "sub-000") in subs
    assert canon_subject(SubjectKey("ds000", "sub-000")) != canon_subject(SubjectKey("ds001", "sub-000"))
    print("  [ok] SubjectKey disambiguates identical local ids across datasets")


def test_no_label_field_and_immutable():
    fnames = {f.name for f in fields(DeploymentBatch)}
    assert not (fnames & {"y", "label", "labels", "delta_r", "target"})
    assert "disease" in fnames
    b = _batch(); _expect(FrozenInstanceError, lambda: setattr(b, "fallback", True)); assert not b.z.flags.writeable
    def _reenable():
        b.z.flags.writeable = True
    _expect(ValueError, _reenable)                                            # bytes-backed: cannot re-enable
    print("  [ok] DeploymentBatch has disease + NO label field; frozen + z strongly read-only")


def test_batching_protocol_validation():
    _expect(ValueError, lambda: _batch(n=MIN_BATCH - 1, fallback=False))        # tiny but not fallback
    _expect(ValueError, lambda: _batch(n=MIN_BATCH, fallback=True))             # big but fallback
    _expect(ValueError, lambda: _batch(src="z" * 64))                          # source not hex
    _expect(ValueError, lambda: _batch(neg=True))                              # negative window index
    _expect(ValueError, lambda: _batch(empty_id=True))                        # empty id
    _expect(ValueError, lambda: DeploymentBatch("PD", SubjectKey("ds", "s"), RecordingKey("ds", "s", "r"),
                                                tuple(WindowKey("ds", "s", "r", i) for i in range(B + 1)),
                                                np.zeros((B + 1, 4)), False, HEX))   # n>B
    _expect(ValueError, lambda: DeploymentBatch("FLU", SubjectKey("ds", "s"), RecordingKey("ds", "s", "r"),
                                                (WindowKey("ds", "s", "r", 0),), np.zeros((1, 4)), True, HEX))  # bad disease
    _expect(ValueError, lambda: DeploymentBatch("PD", SubjectKey("ds", "s"), RecordingKey("ds", "s", "r"),
            tuple(WindowKey("ds", "s", "r", i) for i in range(MIN_BATCH)), np.zeros((MIN_BATCH, 0)), False, HEX))  # d==0
    print("  [ok] DeploymentBatch validates fallback<=>n<MIN_BATCH, n in [1,B], d>=1, hex source, keys, disease")


def test_labeled_risk_record():
    good = tuple((a, 0.1) for a in NON_IDENTITY)
    LabeledRiskRecord(HEX, good)
    _expect(ValueError, lambda: LabeledRiskRecord(HEX, good[:-1]))                              # missing action
    _expect(ValueError, lambda: LabeledRiskRecord(HEX, tuple(reversed(good))))                  # non-canonical order
    _expect(ValueError, lambda: LabeledRiskRecord("z" * 64, good))                             # non-hex digest
    _expect(ValueError, lambda: LabeledRiskRecord("short", good))
    print("  [ok] LabeledRiskRecord requires canonical action order, full-hex digest, finite ΔR")


def test_build_window_order_fallback_and_dup_index():
    rows = [("s1", "r1", w, np.full(4, float(w))) for w in [3, 1, 2, 0]]
    rows += [("s1", "r2", w, np.zeros(4)) for w in range(MIN_BATCH - 1)]
    batches = build_deployment_batches("dsZ", "PD", rows, HEX)
    big = [b for b in batches if b.recording.recording_id == "r1"][0]
    assert [wk.window_index for wk in big.window_keys] == [0, 1, 2, 3]
    small = [b for b in batches if b.recording.recording_id == "r2"][0]; assert small.fallback
    # duplicate window index within a recording (would span a chunk boundary) -> raise before chunking
    dup = [("s1", "r1", w, np.zeros(4)) for w in list(range(B)) + [0]]
    _expect(ValueError, lambda: build_deployment_batches("dsZ", "PD", dup, HEX))
    print("  [ok] build: window-ordered, fallback retained, duplicate window-index rejected (no chunk-boundary escape)")


def test_structured_identity_and_build_locks():
    rows = [("s1", "r1", w, np.zeros(4)) for w in range(10)]
    _expect(TypeError, lambda: build_deployment_batches("ds", "PD", rows, HEX, batch_size=4))   # batching is frozen B
    _expect(ValueError, lambda: build_deployment_batches("ds", "PD", [(1, "r", 0, np.zeros(4))], HEX))   # int id, no coercion
    _expect(ValueError, lambda: build_deployment_batches("ds", "PD", [("s", "r", True, np.zeros(4))], HEX))  # bool index
    _expect(ValueError, lambda: build_deployment_batches("ds", "PD",
            [("s", "r", 0, np.zeros(4)), ("s", "r", 1, np.zeros(5))], HEX))                     # mixed embedding dim
    _expect(ValueError, lambda: build_deployment_batches("ds", "PD", [], HEX))                  # empty
    # plain tuples rejected as keys
    _expect(TypeError, lambda: DeploymentBatch("PD", ("ds", "s"), RecordingKey("ds", "s", "r"),
            tuple(WindowKey("ds", "s", "r", i) for i in range(MIN_BATCH)), np.zeros((MIN_BATCH, 4)), False, HEX))
    print("  [ok] frozen-B batching (no batch_size); no id coercion; consistent dim; plain-tuple keys rejected")


def test_digest_order_insensitive_value_sensitive():
    r1 = [("s", "r", w, np.full(4, float(w))) for w in range(10)]
    d1 = deployment_batch_digest(build_deployment_batches("ds", "PD", r1, HEX)[0])
    d2 = deployment_batch_digest(build_deployment_batches("ds", "PD", list(reversed(r1)), HEX)[0])
    assert d1 == d2 and len(d1) == 64
    r3 = [("s", "r", w, np.full(4, float(w) + (0.001 if w == 5 else 0.0))) for w in range(10)]
    assert deployment_batch_digest(build_deployment_batches("ds", "PD", r3, HEX)[0]) != d1
    assert deployment_batch_digest(build_deployment_batches("ds", "SCZ", r1, HEX)[0]) != d1   # disease in digest
    assert deployment_batch_digest(build_deployment_batches("ds", "PD", r1, hashlib.sha256(b"o").hexdigest())[0]) != d1
    print("  [ok] deployment_batch_digest: full-64, input-order-insensitive, z/disease/source-sensitive")


def main():
    print("ACAR v3 data-layer guards:")
    for t in (test_subjectkey_disambiguation, test_no_label_field_and_immutable, test_batching_protocol_validation,
              test_labeled_risk_record, test_build_window_order_fallback_and_dup_index,
              test_structured_identity_and_build_locks, test_digest_order_insensitive_value_sensitive):
        t()
    print("ALL V3 DATA-LAYER GUARDS PASS")


if __name__ == "__main__":
    main()
