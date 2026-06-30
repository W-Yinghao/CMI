"""C4b-2: the staged (GPU-record / CPU-replay) level executor is BIT-IDENTICAL to the monolithic
run_level_complete, over-extracts every feasible candidate (no capping), and replays without forwards.

Both runs are on the CPU fixture (proving the record -> over-extract -> replay -> bit-exact mechanism);
the cross-device guarantee is checked by the GPU validation run.

Standalone (``python -m oaci.tests.test_staged_executor``) and pytest-compatible.
"""
from __future__ import annotations

import os

import torch

import oaci.protocol
from oaci.runner.fake_data import build_fake_fold
from oaci.runner.fold import run_level_complete
from oaci.runner.keys import RunKey
from oaci.runner.plans import build_level_plans
from oaci.runner.replay_store import ReplayStore
from oaci.runner.scope import build_level_population
from oaci.runner.staged import (feasible_candidate_states, prefetch_level_gpu_artifacts,
                                resume_level_from_store, run_level_staged, train_level)
from oaci.runner.selection import unique_feasible_records
from oaci.runner.support import build_level_support, level0_reference_prior

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")
_CPU = torch.device("cpu")


def _level_inputs(level=0, seed=0):
    fake = build_fake_fold(_MAN)
    fd, maps, schedule, fs = fake.fold_data, fake.maps, fake.deletion_schedule, fake.fold_scope
    support_m = int(fake.manifest.enabled_datasets()["FAKE_TWO_LEVEL"].support_m)
    ref = level0_reference_prior(fd, maps)
    ss = build_level_support(fd, maps, level, schedule, ref, support_m=support_m)
    lp = build_level_population(fd, maps, ss)
    plans = build_level_plans(fs, level, ss, lp, fake.scope_config, model_seed=seed)
    rk = RunKey(fs.fold_key, level, seed)
    return dict(fake=fake, rk=rk, fd=fd, ss=ss, lp=lp, fs=fs, plans=plans,
                exec_cfg=fake.execution_config, model_spec=fake.model_spec)


def _monolithic(level=0):
    g = _level_inputs(level)
    return run_level_complete(g["rk"], g["fd"], g["ss"], g["lp"], g["fs"], g["plans"], g["exec_cfg"],
                              g["model_spec"], g["fake"].model_factory(), _CPU)


def _staged(level=0):
    g = _level_inputs(level)
    lr, store = run_level_staged(g["rk"], g["fd"], g["ss"], g["lp"], g["fs"], g["plans"], g["exec_cfg"],
                                 g["model_spec"], g["fake"].model_factory(), _CPU)
    return lr, store


# ============ candidate set ============
def test_stage_a_extracts_all_feasible_unique_candidate_hashes_no_capping():
    g = _level_inputs(0)
    s1, tr = train_level(g["rk"], g["lp"], g["plans"], g["ss"], g["fs"], g["exec_cfg"], g["model_spec"],
                         g["fake"].model_factory(), _CPU)
    tol = g["exec_cfg"].engine_template.numerical_tol
    cands = feasible_candidate_states(s1, tr, tol)
    expected = {s1.erm_stage.checkpoint.model_hash}
    for name in ("OACI", "global_lpc", "uniform"):
        expected |= {r.model_hash for r in unique_feasible_records(tr[name].train_result, numerical_tol=tol)}
    assert set(cands) == expected                              # exactly ERM + every feasible unique, no cap


def test_stage_a_includes_erm_once():
    g = _level_inputs(0)
    s1, tr = train_level(g["rk"], g["lp"], g["plans"], g["ss"], g["fs"], g["exec_cfg"], g["model_spec"],
                         g["fake"].model_factory(), _CPU)
    cands = feasible_candidate_states(s1, tr, g["exec_cfg"].engine_template.numerical_tol)
    assert s1.erm_stage.checkpoint.model_hash in cands


# ============ bit-exact vs monolithic ============
def test_staged_level_result_hash_matches_monolithic():
    for level in (0, 1):
        assert _monolithic(level).level_result_hash == _staged(level)[0].level_result_hash


def test_staged_selection_audit_prediction_metrics_match_monolithic():
    mono = _monolithic(0)
    staged, _ = _staged(0)
    assert mono.erm_stage.checkpoint.model_hash == staged.erm_stage.checkpoint.model_hash
    for (na, ma), (nb, mb) in zip(mono.method_items, staged.method_items):
        assert ma.selection.model_hash == mb.selection.model_hash                       # selected checkpoint
        assert _leak(ma.selection_leakage) == _leak(mb.selection_leakage)               # selection leakage
        assert _leak(ma.audit_leakage) == _leak(mb.audit_leakage)                       # audit leakage
        for role in ("source_guard_predictions", "source_audit_predictions", "target_predictions"):
            assert getattr(ma, role).prediction_content_hash() == getattr(mb, role).prediction_content_hash()
        for role in ("source_guard_metrics", "source_audit_metrics", "target_metrics"):
            assert getattr(ma, role).metrics_hash == getattr(mb, role).metrics_hash


def test_target_fit_ids_remain_empty_in_staged():
    staged, _ = _staged(0)
    assert not staged.provenance.target_fit_ids


# ============ replay really serves from the store ============
def test_stage_b_replay_fails_without_the_store():
    g = _level_inputs(0)
    s1, tr = train_level(g["rk"], g["lp"], g["plans"], g["ss"], g["fs"], g["exec_cfg"], g["model_spec"],
                         g["fake"].model_factory(), _CPU)
    empty = ReplayStore()                                      # nothing recorded -> replay must fail loudly
    try:
        resume_level_from_store(g["rk"], g["fd"], g["ss"], g["lp"], g["fs"], g["plans"], g["exec_cfg"],
                                g["model_spec"], g["fake"].model_factory(), s1, tr, empty, device=_CPU)
    except KeyError:
        return
    raise AssertionError("Stage B must fail loudly when a GPU-extracted artifact is missing")


def test_prefetch_records_every_role_for_every_candidate():
    g = _level_inputs(0)
    s1, tr = train_level(g["rk"], g["lp"], g["plans"], g["ss"], g["fs"], g["exec_cfg"], g["model_spec"],
                         g["fake"].model_factory(), _CPU)
    store = ReplayStore()
    cands = prefetch_level_gpu_artifacts(g["rk"], s1, tr, g["fd"], g["ss"], g["lp"], g["fs"], g["plans"],
                                         g["exec_cfg"], g["model_spec"], g["fake"].model_factory(), _CPU, store)
    kinds = store.kinds()
    n = len(cands)
    # every candidate has a source_train + source_audit feature and all three prediction-logit roles
    assert kinds.get("feat:source_train") == n and kinds.get("feat:source_audit") == n
    for role in ("source_guard", "source_audit", "target_audit"):
        assert kinds.get(f"logits:{role}") == n


# ============ two-phase persistence boundary (the 2-job analogue) ============
def test_staged_two_phase_fold_matches_monolithic():
    import tempfile

    from oaci.runner.fake import run_fake_two_level_in_memory
    from oaci.runner.staged_fold import staged_phase_a, staged_phase_b
    mono = run_fake_two_level_in_memory(build_fake_fold(_MAN), model_seed=0)
    d = tempfile.mkdtemp()
    staged_phase_a(build_fake_fold(_MAN), dataset_id="FAKE_TWO_LEVEL", model_seed=0, gpu_device=_CPU, out_dir=d)
    staged = staged_phase_b(build_fake_fold(_MAN), d)              # rebuilt fold + persisted trained/store
    assert mono.fold_result_hash == staged.fold_result_hash
    assert mono.fold_scope.fold_scope_hash == staged.fold_scope.fold_scope_hash
    for (lva, lra), (lvb, lrb) in zip(mono.level_items, staged.level_items):
        assert lra.level_result_hash == lrb.level_result_hash


def test_phase_b_rejects_a_mismatched_fold():
    import tempfile

    from oaci.runner.staged_fold import staged_phase_a, staged_phase_b
    d = tempfile.mkdtemp()
    staged_phase_a(build_fake_fold(_MAN), dataset_id="FAKE_TWO_LEVEL", model_seed=0, gpu_device=_CPU, out_dir=d)
    # tamper the persisted Phase-A manifest hash -> Phase B must refuse
    import json
    mp = os.path.join(d, "phase_a.json")
    meta = json.load(open(mp)); meta["manifest_hash"] = "BAD"; json.dump(meta, open(mp, "w"))
    try:
        staged_phase_b(build_fake_fold(_MAN), d)
    except ValueError:
        return
    raise AssertionError("Phase B must reject a fold whose manifest hash disagrees with Phase A")


def _leak(v):
    from oaci.runner.scientific_hash import leakage_result_hash
    return None if v is None else leakage_result_hash(v)


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} staged-executor tests")


if __name__ == "__main__":
    _run_all()
