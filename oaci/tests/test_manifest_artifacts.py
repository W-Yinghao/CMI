"""A2a-part2: strict manifest blocks (risk/evaluation/k1/k2 + value ranges) and strict
PredictionBundle validation + content hashing.

Standalone (``python -m oaci.tests.test_manifest_artifacts``) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from oaci.eval.artifacts import PredictionBundle, population_hash
from oaci.protocol.freeze import default_confirmatory_path
from oaci.protocol.manifest_v2 import (EvaluationBlock, K1Block, K2Block, RiskBlock, load_v2)


def _dir():
    return os.path.dirname(default_confirmatory_path())


def _smoke():
    return load_v2(os.path.join(_dir(), "smoke_v1.yaml"))


# ---------------- manifest strictness ----------------
def test_manifest_risk_evaluation_k1_k2_are_strict():
    m = _smoke()
    assert isinstance(m.risk, RiskBlock) and isinstance(m.evaluation, EvaluationBlock)
    assert isinstance(m.k1, K1Block) and isinstance(m.k2, K2Block)
    assert m.risk.metric == "balanced_ce" and m.evaluation.paired_bootstrap == 100


def test_manifest_rejects_unknown_nested_key():
    p = os.path.join(tempfile.mkdtemp(), "bad.yaml")
    open(p, "w").write("protocol_id: t\nstatus: smoke\nrisk: {metric: ce, epsilon: 0.0, bogus: 1}\n")
    try:
        load_v2(p)
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("an unknown nested (risk) key must be rejected")


def test_manifest_value_ranges_are_enforced():
    base = _smoke().freeze()["sha256"]                    # valid
    for mutate in (
        lambda m: setattr(m.evaluation, "alpha", 1.5),
        lambda m: setattr(m.evaluation, "ece_bins", 1),
        lambda m: setattr(m.probe, "l2_C", 0.0),
        lambda m: setattr(m.probe, "prob_floor", 1.0),
        lambda m: setattr(m.risk, "epsilon", -0.1),
        lambda m: setattr(m.optimizer, "lambda_floor", 0.1),
        lambda m: setattr(m.methods, "global_lpc_laplace_smoothing", 0.0),
    ):
        m = _smoke(); mutate(m)
        try:
            m.freeze()
        except ValueError:
            pass
        else:
            raise AssertionError("an out-of-range manifest value must be rejected")
    assert len(base) == 64


def test_manifest_hash_changes_with_probe_hyperparameter():
    base = _smoke().freeze()["sha256"]
    for mutate in (lambda m: setattr(m.probe, "l2_C", m.probe.l2_C * 2),
                   lambda m: setattr(m.probe, "max_candidate_multiplier", m.probe.max_candidate_multiplier + 1),
                   lambda m: setattr(m.evaluation, "paired_bootstrap", m.evaluation.paired_bootstrap + 1)):
        m = _smoke(); mutate(m)
        assert m.freeze()["sha256"] != base


def test_smoke_and_confirmatory_manifests_roundtrip_all_blocks():
    for name in ("smoke_v1.yaml", "confirmatory_v2.yaml"):
        m = load_v2(os.path.join(_dir(), name))
        for blk in ("seeds", "backbone", "optimizer", "training", "sampler", "probe", "methods",
                    "risk", "evaluation", "k1", "k2"):
            assert getattr(m, blk) is not None, f"{name} dropped block {blk}"
        assert len(m.freeze()["sha256"]) == 64
        canon = m.to_canonical_json()
        for key in ("critic_gradient_clip", "stage2_steps_per_epoch", "max_invalid_draw_rate",
                    "paired_bootstrap", "n_permutations"):
            assert key in canon


# ---------------- PredictionBundle ----------------
def _pb(**over):
    kw = dict(sample_id=["a", "b", "c", "d"],
              logits=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
              y=[0, 1, 0, 1], domain=[0, 0, 1, 1], group=[0, 0, 1, 1], method="ERM", seed=0,
              split_id="s0", split_role="source_audit", deletion_level=0, class_names=["c0", "c1"])
    kw.update(over)
    return PredictionBundle(**kw)


def _rejects(**over):
    try:
        _pb(**over)
    except ValueError:
        return
    raise AssertionError(f"PredictionBundle must reject {list(over)}")


def test_prediction_bundle_rejects_duplicate_ids():
    _rejects(sample_id=["a", "a", "c", "d"])


def test_prediction_bundle_rejects_nonfinite_or_wrong_shape_logits():
    _rejects(logits=np.array([[1.0, np.inf], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    _rejects(logits=np.array([1.0, 0.0, 1.0, 0.0]))                       # 1-D, not [N,C]
    _rejects(logits=np.zeros((4, 3)))                                     # C != len(class_names)


def test_prediction_bundle_rejects_invalid_label_range():
    _rejects(y=[0, 1, 0, 2])                                              # 2 >= C=2
    _rejects(split_role="nope")
    _rejects(deletion_level=-1)


def test_prediction_bundle_rejects_group_spanning_domains():
    _rejects(group=[0, 0, 0, 1], domain=[0, 0, 1, 1])                     # group 0 in domains 0 and 1


def test_population_hash_is_stable_for_unicode_string_ids():
    ids = ["α-1", "β-2", "γ-3", "δ-4"]
    a = _pb(sample_id=ids)
    order = [3, 1, 0, 2]
    b = _pb(sample_id=[ids[i] for i in order], logits=a.logits[order], y=a.y[order],
            domain=a.domain[order], group=a.group[order])
    assert a.eval_population_hash == b.eval_population_hash               # order-invariant + unicode-stable
    assert population_hash(ids, [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]) == a.eval_population_hash


def test_prediction_bundle_preserves_string_group_ids():
    pb = _pb(group=["recA", "recA", "recB", "recB"], domain=[0, 0, 1, 1])
    assert pb.group.dtype == object and pb.group[0] == "recA"        # stable string recording id
    assert pb.eval_population_hash == _pb(group=["recA", "recA", "recB", "recB"]).eval_population_hash


def test_eval_bootstrap_accepts_noninteger_group_ids():
    from oaci.eval.bootstrap import is_whole_group_resample, make_bootstrap_plan
    dom = [0, 0, 0, 0, 1, 1, 1, 1]
    group = ["A", "A", "B", "B", "C", "C", "D", "D"]                  # non-integer recording ids
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    plan = make_bootstrap_plan(dom, group, y, reference_classes=[0, 1], n_boot=8, seed=0, min_clusters=2)
    assert plan.estimable and len(plan.replicates) == 8
    assert is_whole_group_resample(plan.replicates[0], group, plan)  # whole-group check on string keys


def test_eval_population_hash_binds_string_group_mapping():
    from oaci.eval.artifacts import population_hash
    h = population_hash(["a", "b"], [0, 1], [0, 0], ["g0", "g1"])
    assert population_hash(["a", "b"], [0, 1], [0, 0], ["g1", "g0"]) != h     # group->id mapping bound
    assert population_hash(["b", "a"], [1, 0], [0, 0], ["g1", "g0"]) == h     # row-order invariant


def test_prediction_content_hash_changes_with_logits():
    a = _pb()
    b = _pb(logits=a.logits + 0.5)
    assert a.prediction_content_hash() != b.prediction_content_hash()
    assert a.prediction_content_hash() == _pb().prediction_content_hash()  # stable for identical content


def test_prediction_hashes_are_full_sha256():
    pb = _pb()
    for h in (pb.eval_population_hash, pb.prediction_content_hash()):
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_prediction_content_hash_binds_all_scientific_metadata():
    base = _pb().prediction_content_hash()
    for over in (dict(seed=99), dict(checkpoint_hash="ck"), dict(support_mask_hash="sm"),
                 dict(audit_tensor_hash="tt"), dict(split_manifest_hash="sm2"), dict(preprocess_hash="pp")):
        assert _pb(**over).prediction_content_hash() != base
    pb = _pb()
    assert pb.input_tensor_hash == pb.audit_tensor_hash             # semantic alias
    try:
        pb.logits[0, 0] = 9.0                                       # arrays are read-only
    except ValueError:
        pass
    else:
        raise AssertionError("PredictionBundle arrays must be read-only")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} manifest/artifact tests")


if __name__ == "__main__":
    _run_all()
