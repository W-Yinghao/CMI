"""scientific_value_hash identity guarantees (A2b-2a residual)."""
from __future__ import annotations

import numpy as np

from oaci.runner.scientific_hash import leakage_result_hash, normalize_keys, scientific_value_hash


def test_scientific_hash_rejects_object_array():
    try:
        scientific_value_hash(np.array(["a", "b"], dtype=object))
    except TypeError:
        pass
    else:
        raise AssertionError("an object-dtype ndarray must be rejected")


def test_scientific_hash_rejects_nonstring_mapping_key():
    try:
        scientific_value_hash({1: 2.0})
    except TypeError:
        pass
    else:
        raise AssertionError("a non-string mapping key must be rejected")


def test_scientific_hash_tags_nonfinite_floats():
    # nan / +inf / -inf are distinct and stable (not via NaN payload bits)
    hn = scientific_value_hash(float("nan"))
    hp = scientific_value_hash(float("inf"))
    hm = scientific_value_hash(float("-inf"))
    assert len({hn, hp, hm}) == 3
    assert scientific_value_hash(float("nan")) == hn                # deterministic
    a = scientific_value_hash(np.array([1.0, np.nan, np.inf]))
    b = scientific_value_hash(np.array([1.0, np.nan, np.inf]))
    assert a == b and a != scientific_value_hash(np.array([1.0, 2.0, np.inf]))


def test_scientific_hash_is_value_sensitive():
    assert scientific_value_hash({"a": 1, "b": [1, 2]}) != scientific_value_hash({"a": 1, "b": [1, 3]})
    assert scientific_value_hash({"a": 1}) == scientific_value_hash({"a": 1})


def test_leakage_result_hash_normalizes_int_keys():
    # a per-class int-keyed sub-map is hashable via leakage_result_hash, and int:0 never collides with "0"
    h = leakage_result_hash({"reference_entropy": {0: 0.5, 1: 0.7}, "ucl": 0.3})
    assert len(h) == 64
    assert normalize_keys({0: 1})["int:0"] == 1
    assert leakage_result_hash({"x": {0: 1.0}}) != leakage_result_hash({"x": {"0": 1.0}})


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} scientific-hash tests")


if __name__ == "__main__":
    _run_all()
