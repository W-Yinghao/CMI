"""C4b foundation: the replay store captures the selection/audit/prediction GPU forwards bit-exactly.

A RECORD run stores every frozen-feature and row-prediction artifact under its deterministic key; a later
REPLAY run serves them all from the store (no forward) and produces an IDENTICAL FoldRunResult. The store
is persistable (pickled), so a GPU record stage and a CPU replay stage can be separate jobs. Here both
runs are on the CPU fixture, which proves the MECHANISM (record -> persist -> replay -> bit-exact); the
cross-device guarantee is checked by the GPU run.

Standalone (``python -m oaci.tests.test_replay_store``) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile

import oaci.protocol
import oaci.runner.replay_store as RS
from oaci.runner.fake import DEFAULT_METHOD_ORDER, run_fake_two_level_in_memory
from oaci.runner.fake_data import build_fake_fold

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")


def _run():
    return run_fake_two_level_in_memory(build_fake_fold(_MAN), model_seed=0, method_order=DEFAULT_METHOD_ORDER)


def _record():
    RS.set_replay_store(RS.ReplayStore(), "record")
    try:
        fr = _run()
        return fr, RS.get_replay_store()
    finally:
        RS.set_replay_store(None, "off")


def _replay(store):
    RS.set_replay_store(store, "replay")
    try:
        return _run()
    finally:
        RS.set_replay_store(None, "off")


def test_record_captures_feature_and_prediction_artifacts():
    _fr, store = _record()
    kinds = store.kinds()
    assert kinds.get("feature", 0) > 0 and kinds.get("prediction", 0) > 0
    assert len(store) == sum(kinds.values())


def test_replay_reproduces_fold_result_bit_exactly():
    a, store = _record()
    b = _replay(store)                                          # served entirely from the store, no forward
    assert a.fold_result_hash == b.fold_result_hash
    # every level / method scientific identity matches
    for (la, lra), (lb, lrb) in zip(a.level_items, b.level_items):
        assert lra.level_result_hash == lrb.level_result_hash
        assert lra.erm_stage.checkpoint.model_hash == lrb.erm_stage.checkpoint.model_hash
        for (na, ma), (nb, mb) in zip(lra.method_items, lrb.method_items):
            assert ma.selection.model_hash == mb.selection.model_hash
            assert ma.target_predictions.prediction_content_hash() == mb.target_predictions.prediction_content_hash()


def test_replay_survives_pickle_round_trip():
    a, store = _record()
    p = os.path.join(tempfile.mkdtemp(), "store.pkl")
    store.save(p)
    b = _replay(RS.ReplayStore.load(p))                         # persisted store (separate-job analogue)
    assert a.fold_result_hash == b.fold_result_hash


def test_replay_requires_every_key_present():
    a, store = _record()
    # drop one recorded feature artifact -> replay must fail (proving it truly serves from the store)
    a_feature_key = next(k for (kind, k) in store._d if kind == "feature")
    store.drop("feature", a_feature_key)
    try:
        _replay(store)
    except KeyError:
        return
    finally:
        RS.set_replay_store(None, "off")
    raise AssertionError("replay must fail when a recorded GPU artifact is missing")


def test_off_mode_is_the_default_and_unchanged():
    assert RS.replay_mode() == "off" and RS.get_replay_store() is None
    fr = _run()                                                # no store interaction
    assert fr.fold_result_hash and RS.replay_mode() == "off"


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runner.replay_store  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} replay-store tests")


if __name__ == "__main__":
    _run_all()
