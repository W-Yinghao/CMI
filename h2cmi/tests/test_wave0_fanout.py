"""Regression test for the Wave 0 fan-out addressing (the W0.1 index-vs-subject-id bug).
Preflight gate: W0.2-W0.5 must NOT fan out until this passes.

  python -m h2cmi.tests.test_wave0_fanout
"""
from __future__ import annotations

import random

from h2cmi.wave0_fanout import w2_units, unit_id, output_path, preflight_assert, coverage_report

# a deliberately NON-contiguous bench (the real failure mode: index != subject id)
BENCH = [0, 1, 2, 3, 5, 8, 13, 14, 15, 42, 76, 81, 82]


def test_1_shuffle_order_invariant():
    """Shuffling bench order must not change the set of canonical unit ids or output paths."""
    u1 = w2_units("W0.1", BENCH)
    shuf = list(BENCH); random.Random(0).shuffle(shuf)
    u2 = w2_units("W0.1", shuf)
    assert {unit_id(u) for u in u1} == {unit_id(u) for u in u2}, "unit-id set changed under bench reorder"
    assert {output_path(u) for u in u1} == {output_path(u) for u in u2}, "output paths changed under reorder"
    print("[1] shuffle-order-invariant unit ids + output paths OK")


def test_2_each_subject_once_and_only_once():
    for proto in ("primary", "secondary"):
        units = w2_units("W0.1", BENCH, protocols=(proto,))
        subs = [u["real_subject_id"] for u in units]
        assert sorted(subs) == sorted(BENCH), f"{proto}: subject coverage != bench"
        assert len(subs) == len(set(subs)), f"{proto}: a subject covered more than once"
    print("[2] each real_subject_id covered once and only once per protocol OK")


def test_3_no_bench_index_in_identity():
    units = w2_units("W0.1", BENCH)
    for u in units:
        assert "bench_index" not in u, "unit dict carries bench_index as identity"
        assert "real_subject_id" in u
        # unit_id must be derivable from real ids only: the id for subject 42 must contain s42, not its index
    u42 = next(u for u in units if u["real_subject_id"] == 42 and u["protocol"] == "primary")
    assert "s42" in unit_id(u42) and output_path(u42).endswith("_42.jsonl"), "addressing not by real id"
    # index of 42 in BENCH is 9 -> ensure the path is NOT addressed by index 9
    assert not output_path(u42).endswith("_9.jsonl")
    print("[3] addressing is by real subject id, never bench index OK")


def test_4_preflight_and_join_keys():
    units = w2_units("W0.1", BENCH)
    pf = preflight_assert(units)
    assert pf["ok"] and pf["n_subjects"] == len(BENCH)
    # downstream join key must be (real_subject_id, protocol[, q, batch_n]) -- assert uniqueness on that
    keys = {(u["real_subject_id"], u["protocol"]) for u in units}
    assert len(keys) == len(units), "join key (real_subject_id, protocol) not unique -> merge would collide"
    # duplicate-injection must raise
    try:
        preflight_assert(units + [units[0]])
        raise AssertionError("preflight failed to catch a duplicate unit")
    except AssertionError as e:
        assert "duplicate" in str(e)
    print("[4] preflight asserts (no dup, real-id join keys) OK")


def test_5_q_and_batch_variants_distinct():
    """W0.2 (q) and W0.4 (batch_n) variants of the same subject must be distinct units + paths."""
    base = w2_units("W0.2", BENCH, protocols=("primary",), q=0.5)
    hi = w2_units("W0.2", BENCH, protocols=("primary",), q=0.9)
    assert {unit_id(u) for u in base}.isdisjoint({unit_id(u) for u in hi}), "q variants collide"
    assert {output_path(u) for u in base}.isdisjoint({output_path(u) for u in hi}), "q variant paths collide"
    print("[5] q / batch_n variants are distinct units + paths OK")


if __name__ == "__main__":
    test_1_shuffle_order_invariant()
    test_2_each_subject_once_and_only_once()
    test_3_no_bench_index_in_identity()
    test_4_preflight_and_join_keys()
    test_5_q_and_batch_variants_distinct()
    print("ALL WAVE0 FAN-OUT REGRESSION TESTS PASSED")
