"""A2b-1a runner contracts: keys (model-seed separation), frozen maps, phase/provenance, canonical
feature extraction, and the TrainingData population-hash group fix.

Standalone (``python -m oaci.tests.test_runner_contracts``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np
import torch

from oaci.leakage import make_leakage_design
from oaci.models import build_model
from oaci.runner import (FoldKey, FrozenMaps, RunKey, RunnerPhase, RunProvenance, build_frozen_maps,
                         extract_frozen_features)
from oaci.runner.provenance import IllegalPhaseTransition
from oaci.support_graph import build_support_graph, empirical_class_prior
from oaci.train.data import TrainingData, population_signature_hash


# ---------------- fix: TrainingData population hash group length-prefix ----------------
def _td(group, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 6
    return TrainingData(X=torch.randn(n, 4, generator=g), y=torch.tensor([0, 1, 0, 1, 0, 1]),
                        sample_id=tuple(f"s{i}" for i in range(n)), sample_mass=torch.ones(n),
                        n_classes=2, d=torch.tensor([0, 0, 1, 1, 2, 2]), group=group).validate()


def test_training_population_hash_length_prefixes_string_group():
    # ("a","bc") vs ("ab","c") concatenate identically without a length prefix -> must differ now
    h1 = population_signature_hash(_td(("a", "bc", "x", "y", "z", "w")))
    h2 = population_signature_hash(_td(("ab", "c", "x", "y", "z", "w")))
    assert h1 != h2


# ---------------- keys: model-seed separation ----------------
def _fk(**over):
    base = dict(manifest_hash="m", dataset_id="ds", outer_fold="f0", split_seed=1, deletion_seed=2)
    base.update(over)
    return FoldKey(**base)


def test_fold_key_excludes_model_seed_and_run_key_includes_it():
    import dataclasses
    assert "model_seed" not in {f.name for f in dataclasses.fields(FoldKey)}
    assert "model_seed" in {f.name for f in dataclasses.fields(RunKey)}


def test_fold_scope_hash_is_model_seed_invariant():
    a = RunKey(_fk(), deletion_level=0, model_seed=0)
    b = RunKey(_fk(), deletion_level=0, model_seed=7)
    assert a.fold_key.fold_key_hash == b.fold_key.fold_key_hash       # fold identity ignores model seed
    assert a.run_key_hash != b.run_key_hash                            # run key does not


def test_run_key_hash_changes_with_model_seed_or_level():
    base = RunKey(_fk(), 0, 0).run_key_hash
    assert RunKey(_fk(), 0, 1).run_key_hash != base
    assert RunKey(_fk(), 1, 0).run_key_hash != base
    assert RunKey(_fk(split_seed=9), 0, 0).run_key_hash != base


# ---------------- frozen maps ----------------
def test_frozen_source_domain_map_is_contiguous():
    m = build_frozen_maps(["neg", "pos"], ["subB", "subA", "subC"], ["t0", "t1"])
    assert sorted(m.source_domain_to_index.values()) == [0, 1, 2]
    assert m.source_domain_ids == ("subA", "subB", "subC")            # sorted -> 0..|D0|-1


def test_maps_preserve_manifest_class_order():
    m = build_frozen_maps(["pos", "neg"], ["d0", "d1"], ["d0"])       # NOT sorted
    assert m.class_names == ("pos", "neg") and m.class_to_index == {"pos": 0, "neg": 1}


def test_maps_hash_changes_with_domain_or_class_order():
    base = build_frozen_maps(["a", "b"], ["d0", "d1"], ["d0"]).maps_hash
    assert build_frozen_maps(["b", "a"], ["d0", "d1"], ["d0"]).maps_hash != base
    try:
        build_frozen_maps(["a", "a"], ["d0"], ["d0"])                 # duplicate class
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate id must be rejected")


# ---------------- phase / provenance ----------------
def test_illegal_phase_transition_fails():
    p = RunProvenance()
    try:
        p.transition(RunnerPhase.SELECTION)                          # skips TRAINING
    except IllegalPhaseTransition:
        pass
    else:
        raise AssertionError("a skipped phase transition must fail")


def _walk_to(p, phase):
    order = [RunnerPhase.TRAINING, RunnerPhase.SELECTION, RunnerPhase.SELECTION_LOCKED,
             RunnerPhase.AUDIT, RunnerPhase.COMPLETE]
    for ph in order:
        if ph == RunnerPhase.SELECTION_LOCKED:
            p.lock_selection()
        else:
            p.transition(ph)
        if ph == phase:
            return


def test_provenance_rejects_source_audit_in_selection():
    p = RunProvenance(); _walk_to(p, RunnerPhase.SELECTION)
    p.record_fit("selection", ["s0", "AUDIT0"])                       # AUDIT0 is a source-audit id
    try:
        p.assert_invariants(["s0", "s1"], ["s0", "s1"], ["AUDIT0"])
    except ValueError:
        pass
    else:
        raise AssertionError("a source-audit id in selection fits must be rejected")


def test_provenance_rejects_audit_before_selection_lock():
    p = RunProvenance(); p.transition(RunnerPhase.TRAINING); p.transition(RunnerPhase.SELECTION)
    p.record_fit("audit_estimator", ["a0"])                          # BEFORE lock
    p.lock_selection()
    try:
        p.assert_invariants(["s0"], ["s0"], ["a0"])
    except ValueError:
        pass
    else:
        raise AssertionError("an audit fit before the selection lock must be rejected")


def test_provenance_rejects_any_target_fit_id():
    p = RunProvenance(); p.transition(RunnerPhase.TRAINING)
    p.record_fit("target", ["t0"])
    try:
        p.assert_invariants(["s0"], ["s0"], [])
    except ValueError:
        pass
    else:
        raise AssertionError("any target fit id must be rejected")


# ---------------- canonical feature extraction ----------------
def _feat_setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    n, nd, nc = 18, 3, 2
    y = np.array([c for _ in range(n // nc) for c in range(nc)])
    d = np.array([i % nd for i in range(n)])
    sid = tuple(f"r{i}" for i in range(n))
    grp = tuple(f"rec{d[i]}-{i}" for i in range(n))
    X = torch.randn(n, 5, generator=g)
    data = TrainingData(X=X, y=torch.tensor(y), sample_id=sid, sample_mass=torch.ones(n),
                        n_classes=nc, d=torch.tensor(d), group=grp).validate()
    counts = np.zeros((nd, nc), int)
    for i in range(n):
        counts[d[i], y[i]] += 1
    sg = build_support_graph(counts, m=2, cell_mass=counts.astype(float),
                             reference_prior=empirical_class_prior(counts))
    design = make_leakage_design(sid, y, d, grp, np.ones(n), sg)
    return data, design


def _model_factory():
    return build_model("mlp", in_dim=5, n_classes=2)


def test_feature_extraction_is_canonical_and_population_matches_design():
    from oaci.train.checkpoint import model_state_hash
    data, design = _feat_setup()
    model = _model_factory(); mh = model_state_hash(model); state = model.state_dict()
    art = extract_frozen_features(state, mh, _model_factory, data, design,
                                  factory_seed=1, chunk_size=4, device=torch.device("cpu"))
    assert art.features.sample_id == tuple(sorted(data.sample_id))     # canonical order
    assert art.population_hash == design.population_hash               # matches the design
    assert art.model_hash == mh


def test_feature_extraction_restores_rng_and_model_state():
    from oaci.train.checkpoint import model_state_hash
    data, design = _feat_setup()
    model = _model_factory(); mh = model_state_hash(model); state = model.state_dict()
    rng = torch.random.get_rng_state()
    cpu = torch.device("cpu")
    a = extract_frozen_features(state, mh, _model_factory, data, design, factory_seed=1, chunk_size=None, device=cpu)
    assert torch.equal(torch.random.get_rng_state(), rng)            # caller RNG unchanged
    a2 = extract_frozen_features(state, mh, _model_factory, data, design, factory_seed=1, chunk_size=None, device=cpu)
    assert a.feature_hash == a2.feature_hash                         # SAME chunk size -> byte-identical
    # across chunk sizes the float matmul is batch-size dependent (BLAS), so compare within tolerance
    b = extract_frozen_features(state, mh, _model_factory, data, design, factory_seed=1, chunk_size=3, device=cpu)
    assert np.allclose(a.features.Z, b.features.Z, atol=1e-5)


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-contract tests")


if __name__ == "__main__":
    _run_all()
