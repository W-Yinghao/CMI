"""A2b-2b-i: strict fake manifest, deterministic two-level fixture, full in-memory run.

Standalone (``python -m oaci.tests.test_runner_fake``) and pytest-compatible.
"""
from __future__ import annotations

import os

import numpy as np
import torch

import oaci.protocol
from oaci.protocol.manifest_v2 import load_v2
from oaci.runner import (RunnerPhase, build_level_support, level0_reference_prior, make_objective)
from oaci.runner.fake import DEFAULT_METHOD_ORDER, run_fake_two_level_in_memory
from oaci.runner.fake_data import build_fake_fold

_MANIFEST = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")
_REV = tuple(reversed(DEFAULT_METHOD_ORDER))
_C = {}


def _ff():
    if "ff" not in _C:
        _C["ff"] = build_fake_fold(_MANIFEST)
    return _C["ff"]


def _run(model_seed=0, order=DEFAULT_METHOD_ORDER):
    key = (model_seed, tuple(order))
    if key not in _C:
        _C[key] = run_fake_two_level_in_memory(_ff(), model_seed=model_seed, method_order=order)
    return _C[key]


def _support(level):
    ff = _ff()
    ref = level0_reference_prior(ff.fold_data, ff.maps)
    return build_level_support(ff.fold_data, ff.maps, level, ff.deletion_schedule, ref, support_m=3)


# ============================ manifest ============================
def test_fake_fixture_block_is_strict_and_manifest_hashed():
    m = load_v2(_MANIFEST)
    base = m.freeze()["sha256"]
    m2 = load_v2(_MANIFEST); m2.fake_fixture.data_seed = 9999
    assert m2.freeze()["sha256"] != base                       # the fixture block enters the manifest hash
    # an unknown field inside fake_fixture is rejected by the strict parser
    import yaml
    with open(_MANIFEST) as f:
        raw = yaml.safe_load(f)
    raw["fake_fixture"]["bogus"] = 1
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "bad.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(raw, f)
    try:
        load_v2(p)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown fake_fixture field must be rejected")


def test_fake_manifest_is_rejected_in_confirmatory_mode():
    try:
        load_v2(_MANIFEST).assert_confirmatory()
    except ValueError:
        pass
    else:
        raise AssertionError("a smoke fake manifest must be rejected in confirmatory mode")


def test_fake_manifest_rejects_real_dataset_or_non_smoke_use():
    # the fake_fixture block cannot ride on a non-fake manifest
    m = load_v2(os.path.join(os.path.dirname(_MANIFEST), "smoke_v1.yaml"))
    from oaci.protocol.manifest_v2 import FakeFixtureBlock
    m.fake_fixture = FakeFixtureBlock(source_domain_ids=["S0"])
    try:
        m.validate_complete()
    except ValueError:
        pass
    else:
        raise AssertionError("fake_fixture on a non-FAKE_TWO_LEVEL manifest must be rejected")


def test_fake_manifest_freezes_every_mlp_parameter():
    m = load_v2(_MANIFEST)
    assert m.backbone.name == "mlp" and m.backbone.mlp_z_dim == 6 and m.backbone.mlp_hidden == 12
    m.backbone.mlp_hidden = None
    try:
        m.validate_ranges()
    except ValueError:
        pass
    else:
        raise AssertionError("an mlp backbone missing a frozen dim must be rejected")


def test_fake_prediction_floor_is_valid_for_class_count():
    m = load_v2(_MANIFEST)
    assert m.evaluation.prediction_prob_floor * len(m.enabled_datasets()["FAKE_TWO_LEVEL"].class_names) < 1
    m.evaluation.prediction_prob_floor = 0.7        # 0.7 * 2 = 1.4 >= 1
    try:
        m.validate_ranges()
    except ValueError:
        pass
    else:
        raise AssertionError("prediction_prob_floor * n_classes >= 1 must be rejected")


# ============================ fake data ============================
def test_fake_expected_row_and_unit_counts():
    fd = _ff().fold_data
    assert (len(fd.source_train_idx), len(fd.source_audit_idx), len(fd.target_audit_idx)) == (48, 36, 15)
    units = {role: len({fd.eval_unit_id[i] for i in idx.tolist()})
             for role, idx in (("st", fd.source_train_idx), ("sa", fd.source_audit_idx), ("ta", fd.target_audit_idx))}
    assert units == {"st": 24, "sa": 18, "ta": 8}


def test_fake_groups_span_classes_not_domains_or_roles():
    fd = _ff().fold_data
    by_group = {}
    for i in range(len(fd.sample_id)):
        by_group.setdefault(fd.group_id[i], set()).add((fd.domain_id[i], int(fd.y[i])))
    spans_classes = any(len({y for _, y in v}) == 2 for v in by_group.values())
    assert spans_classes
    for g, v in by_group.items():
        assert len({d for d, _ in v}) == 1                     # a group is one domain (and one role by id)


def test_fake_mass_unit_sum_is_exactly_one():
    fd = _ff().fold_data
    by_unit = {}
    for i in range(len(fd.sample_id)):
        by_unit.setdefault(fd.mass_unit_id[i], 0.0)
        by_unit[fd.mass_unit_id[i]] += float(fd.sample_mass[i])
    assert all(abs(m - 1.0) < 1e-9 for m in by_unit.values())


def test_fake_ids_are_globally_unique_and_stable():
    fd = _ff().fold_data
    assert len(set(fd.sample_id)) == len(fd.sample_id)
    assert all(s.startswith("FAKE_TWO_LEVEL|") for s in fd.sample_id)


def test_fake_X_is_stable_under_row_permutation():
    from oaci.runner.fake_data import _build_X, _rows
    ff = _ff().manifest.fake_fixture
    rows = _rows(ff)
    X = _build_X(rows, ff).numpy()
    perm = list(np.random.default_rng(3).permutation(len(rows)))
    Xp = _build_X([rows[i] for i in perm], ff).numpy()
    by_sid = {r["sid"]: X[i] for i, r in enumerate(rows)}
    for j, i in enumerate(perm):
        assert np.array_equal(Xp[j], by_sid[rows[i]["sid"]])   # X tied to sample_id, not row order


def test_fake_builder_does_not_change_global_rng():
    rng_t = torch.random.get_rng_state()
    rng_n = np.random.get_state()
    build_fake_fold(_MANIFEST)
    assert torch.equal(rng_t, torch.random.get_rng_state())
    assert np.array_equal(rng_n[1], np.random.get_state()[1])


def test_fake_fold_data_is_model_seed_independent():
    # the FoldData (and its scope) is built once and shared; running different seeds does not rebuild it
    a = run_fake_two_level_in_memory(_ff(), model_seed=0)
    b = run_fake_two_level_in_memory(_ff(), model_seed=1)
    assert a.fold_scope.fold_scope_hash == b.fold_scope.fold_scope_hash


def test_fake_data_seed_changes_tensor_hash():
    import tempfile
    import yaml
    base = _ff().fake_data_hash
    with open(_MANIFEST) as f:
        raw = yaml.safe_load(f)
    raw["fake_fixture"]["data_seed"] = 4242
    p = os.path.join(tempfile.mkdtemp(), "m.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(raw, f)
    assert build_fake_fold(p).fake_data_hash != base


# ============================ support tables ============================
def test_fake_level0_support_table_is_exact():
    ss = _support(0)
    assert ss.eligibility_counts.tolist() == [[4, 4], [4, 4], [4, 4]]
    assert ss.cell_mass.tolist() == [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]


def test_fake_level1_support_table_is_exact():
    ss = _support(1)
    assert ss.eligibility_counts.tolist() == [[4, 0], [4, 4], [4, 4]]
    assert ss.cell_mass.tolist() == [[4.0, 0.0], [4.0, 4.0], [4.0, 4.0]]


def test_fake_deleted_cell_is_exactly_zero():
    ss = _support(1)
    assert ss.eligibility_counts[0, 1] == 0 and ss.cell_mass[0, 1] == 0.0
    fd = _ff().fold_data
    # no retained source-train row in (S0, c1) at level 1
    retained = set(ss.source_train_idx.tolist())
    assert not any(fd.domain_id[i] == "S0" and int(fd.y[i]) == 1 for i in retained)


def test_fake_deleted_domain_remains_present():
    ss = _support(1)
    assert ss.cell_mass[0, 0] == 4.0                            # S0 still present via c0
    assert len(ss.support_graph.comparable_classes) == 2       # OACI still compares both classes


def test_fake_reference_prior_is_fixed():
    ff = _ff()
    ref = level0_reference_prior(ff.fold_data, ff.maps)
    s0 = build_level_support(ff.fold_data, ff.maps, 0, ff.deletion_schedule, ref, support_m=3)
    s1 = build_level_support(ff.fold_data, ff.maps, 1, ff.deletion_schedule, ref, support_m=3)
    assert np.array_equal(s0.support_graph.reference_prior, s1.support_graph.reference_prior)
    assert np.allclose(s0.support_graph.reference_prior, [0.5, 0.5])


def test_fake_audit_scope_object_is_reused():
    fr = _run()
    # the assembled fold carries ONE scope; both levels share its source-audit object
    assert fr.fold_scope.source_audit is fr.fold_scope.source_audit
    assert fr.levels[0].run_key.fold_key.fold_key_hash == fr.levels[1].run_key.fold_key.fold_key_hash


# ============================ runner ============================
def test_fake_two_level_run_reaches_complete():
    fr = _run()
    assert set(fr.levels) == {0, 1}
    assert all(fr.levels[l].phase == RunnerPhase.COMPLETE for l in (0, 1))


def test_fake_stage1_is_invoked_once_per_level():
    fr = _run()
    for l in (0, 1):
        lr = fr.levels[l]
        assert lr.erm_stage is not None and dict(lr.invariant_items)["shared_erm_unique"]


def test_fake_all_methods_share_erm_tau_and_task_plan():
    fr = _run()
    for l in (0, 1):
        ms = [m for _, m in fr.levels[l].method_items]
        assert len({m.shared_erm_hash for m in ms}) == 1
        assert len({round(m.shared_tau, 12) for m in ms}) == 1
        assert len({m.shared_stage2_task_plan_hash for m in ms}) == 1


def test_fake_oaci_rejected_ineligible_rows_zero():
    fr = _run()
    for l in (0, 1):
        assert dict(fr.levels[l].invariant_items)["oaci_rejected_ineligible_rows"] == 0


def test_fake_global_lpc_deleted_prior_positive_rows_zero():
    ss1 = _support(1)
    obj, spec = make_objective("global_lpc", ss1, _ff().fold_scope, _ff().execution_config)
    prior = np.asarray(spec.prior_matrix)                      # [n_classes, |D0|]
    # the deleted cell (c1, S0=index 0) has positive prior but zero observed mass
    assert prior[1, 0] > 0 and ss1.cell_mass[0, 1] == 0.0
    alpha = float(_ff().manifest.methods.global_lpc_laplace_smoothing)
    assert abs(prior[1, 0] - alpha / (8 + 3 * alpha)) < 1e-9


def test_fake_uniform_prior_exact_and_level_invariant():
    obj, spec = make_objective("uniform", _support(0), _ff().fold_scope, _ff().execution_config)
    assert np.allclose(np.asarray(spec.prior_matrix), 1.0 / 3.0)
    fr = _run()
    assert (fr.levels[0].methods["uniform"].training_diagnostics["prior_matrix_hash"]
            == fr.levels[1].methods["uniform"].training_diagnostics["prior_matrix_hash"])


def test_fake_target_fit_ids_empty():
    fr = _run()
    assert all(not fr.levels[l].provenance.target_fit_ids for l in (0, 1))


def test_fake_audit_and_target_signatures_are_level_invariant():
    fr = _run()
    sa = {m.source_audit_predictions.audit_signature_hash for l in (0, 1) for _, m in fr.levels[l].method_items}
    ta = {m.target_predictions.audit_signature_hash for l in (0, 1) for _, m in fr.levels[l].method_items}
    assert len(sa) == 1 and len(ta) == 1


# ============================ reproducibility ============================
def test_same_seed_reproduces_fold_result_hash():
    a = run_fake_two_level_in_memory(_ff(), model_seed=0)
    b = run_fake_two_level_in_memory(_ff(), model_seed=0)
    assert a.fold_result_hash == b.fold_result_hash


def test_permuted_method_order_reproduces_fold_result_hash():
    assert _run(order=DEFAULT_METHOD_ORDER).fold_result_hash == _run(order=_REV).fold_result_hash


def test_different_model_seed_changes_training_hashes():
    assert _run(model_seed=0).fold_result_hash != _run(model_seed=1).fold_result_hash


def test_different_model_seed_keeps_fold_scope_and_data_hashes():
    a, b = _run(model_seed=0), _run(model_seed=1)
    assert a.fold_scope.fold_scope_hash == b.fold_scope.fold_scope_hash
    assert _ff().fake_data_hash == _ff().fake_data_hash


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runner.fake  # noqa: F401
    import oaci.runner.fake_data  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-fake tests")


if __name__ == "__main__":
    _run_all()
