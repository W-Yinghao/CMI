"""Confirmatory schema + materialize + the manifest_v2 `pilot` block (CPU; no data, no GPU).

Standalone (`python -m oaci.tests.test_confirmatory_adapter`) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile

import oaci.protocol
from oaci.confirmatory.materialize import BNCI2014_001_SUBJECTS, materialize_pilot_manifest, split_subjects
from oaci.confirmatory.schema import load_confirmatory
from oaci.protocol.manifest_v2 import load_v2

_PROTO = os.path.join(os.path.dirname(oaci.protocol.__file__), "confirmatory_v2.yaml")
_SMOKE = os.path.join(os.path.dirname(oaci.protocol.__file__), "smoke_v1.yaml")


def _materialize(target=1, seeds=(0, 1, 2), out=None):
    proto = load_confirmatory(_PROTO)
    out = out or os.path.join(tempfile.mkdtemp(), "man.yaml")
    return materialize_pilot_manifest(proto, "BNCI2014_001", target_subject=target, out_path=out,
                                      model_seeds=list(seeds))


# ============ schema ============
def test_schema_parses_confirmatory_protocol():
    proto = load_confirmatory(_PROTO)
    assert proto.protocol_id == "oaci-confirmatory-v2"
    assert "BNCI2014_001" in proto.enabled_datasets()


def test_schema_rejects_missing_block():
    import yaml
    proto = load_confirmatory(_PROTO)
    bad = {k: v for k, v in proto.raw.items() if k != "training"}      # drop a required block
    p = os.path.join(tempfile.mkdtemp(), "bad.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(bad, f)
    try:
        load_confirmatory(p)
    except ValueError:
        return
    raise AssertionError("a protocol missing 'training' must be rejected")


# ============ split ============
def test_ordered_split_holds_out_target_and_uses_all_subjects():
    s = split_subjects(BNCI2014_001_SUBJECTS, 1)
    assert s["target_subjects"] == [1] and s["source_audit_subjects"] == [2, 3]
    assert s["source_train_subjects"] == [4, 5, 6, 7, 8, 9]
    assert sorted(s["subjects"]) == list(range(1, 10))
    s5 = split_subjects(BNCI2014_001_SUBJECTS, 5)
    assert s5["target_subjects"] == [5] and 5 not in s5["source_audit_subjects"] + s5["source_train_subjects"]


# ============ materialize ============
def test_materialize_produces_runnable_full_budget_pilot():
    _, m = _materialize()
    m.validate_complete()                                              # must be runnable
    assert m.status == "pilot"
    assert m.pilot.target_subjects == [1] and m.pilot.source_audit_subjects == [2, 3]
    assert m.pilot.source_train_subjects == [4, 5, 6, 7, 8, 9]
    assert m.training.stage1_epochs == 200 and m.training.stage2_epochs == 200     # FULL budget, not smoke
    assert m.probe.folds == 5 and m.evaluation.paired_bootstrap == 2000
    assert m.seeds.model == [0, 1, 2]
    ds = m.enabled_datasets()["BNCI2014_001"]
    assert ds.expected_n_times == 385 and ds.expected_sfreq == 128.0          # the (22,385,4) geometry holds
    assert ds.preprocessing.kind == "moabb_motor_imagery" and ds.preprocessing.channel_interpolation is False


def test_materialize_is_deterministic():
    _, a = _materialize()
    _, b = _materialize()
    assert a.freeze()["sha256"] == b.freeze()["sha256"]


def test_bootstrap_override_shrinks_only_leakage_eval_keeps_training():
    from oaci.confirmatory.materialize import VALIDATION_BOOTSTRAP
    proto = load_confirmatory(_PROTO)
    d = tempfile.mkdtemp()
    _, full = materialize_pilot_manifest(proto, "BNCI2014_001", target_subject=1,
                                         out_path=os.path.join(d, "full.yaml"), model_seeds=[0])
    _, red = materialize_pilot_manifest(proto, "BNCI2014_001", target_subject=1,
                                        out_path=os.path.join(d, "red.yaml"), model_seeds=[0],
                                        bootstrap_override=VALIDATION_BOOTSTRAP)
    red.validate_complete()
    assert red.probe.selection_bootstrap == 64 and red.probe.audit_bootstrap == 256
    assert red.evaluation.paired_bootstrap == 256
    assert red.training.stage1_epochs == full.training.stage1_epochs == 200      # TRAINING budget untouched
    assert red.probe.folds == full.probe.folds and red.probe.capacities == full.probe.capacities
    assert "validredbootstrap" in red.protocol_id                               # recorded in id + hash
    assert red.freeze()["sha256"] != full.freeze()["sha256"]


def test_materialize_rejects_target_not_in_subjects():
    proto = load_confirmatory(_PROTO)
    try:
        materialize_pilot_manifest(proto, "BNCI2014_001", target_subject=99,
                                   out_path=os.path.join(tempfile.mkdtemp(), "m.yaml"), model_seeds=[0])
    except ValueError:
        return
    raise AssertionError("target subject not in the subject set must be rejected")


# ============ manifest_v2 pilot block validation ============
def test_pilot_block_rejects_overlapping_roles():
    _, m = _materialize()
    m.pilot.source_audit_subjects = [1, 2]                            # 1 also the target -> overlap
    try:
        m.validate_complete()
    except ValueError:
        return
    raise AssertionError("overlapping pilot roles must be rejected")


def test_pilot_block_rejects_union_mismatch_and_multi_target():
    _, m = _materialize()
    m.pilot.subjects = [1, 2, 3, 4, 5, 6, 7, 8]                       # 9 missing from union
    try:
        m.validate_complete()
    except ValueError:
        pass
    else:
        raise AssertionError("union != subjects must be rejected")
    _, m2 = _materialize()
    m2.pilot.target_subjects = [1, 2]
    m2.pilot.source_audit_subjects = [3]
    try:
        m2.validate_complete()
    except ValueError:
        return
    raise AssertionError("more than one target must be rejected")


def test_smoke_manifest_still_validates_unchanged():
    m = load_v2(_SMOKE)
    m.validate_complete()                                             # additive pilot support must not break smoke
    assert m.status == "smoke" and m.pilot is None
    assert m.smoke.target_subjects == [1] and m.smoke.source_train_subjects == [4, 5, 6]


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} confirmatory-adapter tests")


if __name__ == "__main__":
    _run_all()
