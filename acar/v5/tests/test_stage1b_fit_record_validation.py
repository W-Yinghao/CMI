"""Guard (Stage-1B7): the real trainer VALIDATES FIT records before the numeric backend sees anything — each windows is a validated
SubjectWindows, each label ∈ {0,1}, train/val are subject-disjoint, no duplicate keys, val non-empty, keys canonical. Synthetic only."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import real_eegnet_trainer as RET
from acar.v5.tests._util import expect_raises, ok, FakeEegnetBackend, make_subject_windows

SEED = 20260711


def _t(sk, label=0):
    return (sk, make_subject_windows(sk), label)


def _run(train, val):
    with tempfile.TemporaryDirectory() as d:
        return RET.train_encoder_and_source_state("PD", 0, SEED, train, val, output_dir=d, backend=FakeEegnetBackend())


def test_valid_fit_records_ok():
    raw = _run([_t("PD/ds002778/sub-001", 0), _t("PD/ds002778/sub-002", 1)], [_t("PD/ds002778/sub-003", 0)])
    assert raw["ref"] == f"PD/fold0/seed{SEED}" and "feat_dump_path" not in raw
    ok("valid FIT records (SubjectWindows + label 0/1 + disjoint + val non-empty) → training proceeds, no feat_dump")


def test_bad_records_rejected():
    good_t = [_t("PD/ds002778/sub-001", 0)]
    good_v = [_t("PD/ds002778/sub-002", 1)]
    # non-SubjectWindows windows
    expect_raises(RET.RealEegnetError, lambda: _run([("PD/ds002778/sub-001", {"marker": "x"}, 0)], good_v))
    # label not in {0,1} (incl bool)
    expect_raises(RET.RealEegnetError, lambda: _run([_t("PD/ds002778/sub-001", 2)], good_v))
    expect_raises(RET.RealEegnetError, lambda: _run([("PD/ds002778/sub-001", make_subject_windows("PD/ds002778/sub-001"), True)], good_v))
    # train/val overlap
    expect_raises(RET.RealEegnetError, lambda: _run([_t("PD/ds002778/sub-001", 0)], [_t("PD/ds002778/sub-001", 1)]))
    # duplicate subject within train
    expect_raises(RET.RealEegnetError, lambda: _run([_t("PD/ds002778/sub-001", 0), _t("PD/ds002778/sub-001", 1)], good_v))
    # empty val
    expect_raises(RET.RealEegnetError, lambda: _run(good_t, []))
    # non-canonical subject key (wrong disease prefix)
    expect_raises(RET.RealEegnetError, lambda: _run([_t("SCZ/ds003944/sub-001", 0)], good_v))
    ok("bad FIT records (non-SubjectWindows / label∉{0,1} / overlap / dup / empty val / non-canonical key) → RealEegnetError")


def main():
    print("ACAR v5 Stage-1B7 guard: FIT record validation")
    test_valid_fit_records_ok()
    test_bad_records_rejected()
    print("ALL V5 STAGE1B-FIT-RECORD-VALIDATION GUARDS PASS")


if __name__ == "__main__":
    main()
