"""C4a: the optimization (training) RNG is decoupled from the inference/bootstrap config.

Changing a pure inference/reporting field (selection/audit bootstrap, probe folds/capacities, ECE bins,
paired bootstrap) must NOT change the trained model; changing a training/optimizer field must. The full
manifest is still bound by the fold/artifact identity (fold_key_hash differs), only the optimization
identity (optimization_fold_hash) is bootstrap-independent.

Standalone (``python -m oaci.tests.test_optimization_seed_decoupling``) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile

import oaci.protocol
from oaci.protocol.manifest_v2 import load_v2, optimization_manifest_hash
from oaci.runner.fake import DEFAULT_METHOD_ORDER, run_fake_two_level_in_memory
from oaci.runner.fake_data import build_fake_fold

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")


def _patched_path(patch: dict) -> str:
    """Write a copy of the fake manifest with dotted-path values patched."""
    import yaml
    with open(_MAN) as f:
        d = yaml.safe_load(f)
    for dotted, val in patch.items():
        keys = dotted.split("."); o = d
        for k in keys[:-1]:
            o = o[k]
        o[keys[-1]] = val
    p = os.path.join(tempfile.mkdtemp(), "patched.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(d, f)
    return p


def _run(path):
    return run_fake_two_level_in_memory(build_fake_fold(path), model_seed=0, method_order=DEFAULT_METHOD_ORDER)


def _training_fingerprint(fr):
    """Everything the TRAINING produced: per-level stage1/stage2 plan hashes, the ERM checkpoint, and every
    method's full trajectory of checkpoints. NOT the selected checkpoint -- selection is leakage-based
    inference (it legitimately depends on the bootstrap), and the spec only requires the trained artefacts
    (plans / ERM / trajectory) to be bootstrap-independent."""
    out = []
    for lvl, lr in fr.level_items:
        out.append((int(lvl), lr.plans.stage1_task.plan_hash, lr.plans.stage2_task.plan_hash,
                    lr.erm_stage.checkpoint.model_hash))
        for name, m in lr.method_items:
            out.append((int(lvl), name, tuple(c.model_hash for c in m.train_result.trajectory)))
    return tuple(out)


# ============ optimization_manifest_hash: bootstrap/eval independence + training sensitivity ============
def test_opt_hash_ignores_inference_and_reporting_fields():
    m = load_v2(_MAN)
    h0 = optimization_manifest_hash(m)
    m.probe.audit_bootstrap = 99999          # leakage bootstrap
    m.probe.selection_bootstrap = 12345
    m.probe.folds = 9
    m.probe.capacities = [0, 7]
    m.evaluation.paired_bootstrap = 4242
    m.evaluation.ece_bins = 99
    m.evaluation.alpha = 0.42
    m.seeds.selection_bootstrap = 7
    m.seeds.audit_bootstrap = 7
    assert optimization_manifest_hash(m) == h0       # none of these touch the training identity


def test_opt_hash_changes_with_training_optimizer_model_objective():
    base = optimization_manifest_hash(load_v2(_MAN))
    for dotted, val in (("training.stage1_epochs", 5), ("optimizer.lr_stage1", 9.9e-3),
                        ("backbone.dropout", 0.123), ("methods.global_lpc_laplace_smoothing", 2.0),
                        ("risk.epsilon", 0.111), ("sampler.adv_microbatch_size", 32),
                        ("seeds.split", 999)):
        m = load_v2(_MAN)
        keys = dotted.split(".")
        blk = getattr(m, keys[0])
        setattr(blk, keys[1], val)
        assert optimization_manifest_hash(m) != base, f"opt hash must change with {dotted}"


# ============ end-to-end: same training, different bootstrap ============
def test_inference_change_preserves_training_but_changes_result_identity():
    a = _run(_MAN)
    b = _run(_patched_path({"probe.audit_bootstrap": 16}))   # inference-only change
    ka = a.fold_scope.fold_key
    kb = b.fold_scope.fold_key
    # the optimization identity is identical; the FULL fold identity differs (binds the whole manifest)
    assert ka.optimization_fold_hash == kb.optimization_fold_hash
    assert ka.fold_key_hash != kb.fold_key_hash
    # the trained models (plans, ERM, full trajectories) are bit-identical
    assert _training_fingerprint(a) == _training_fingerprint(b)
    # per-level ERM checkpoint specifically (the R_ERM_hat / tau confound that motivated C4a)
    for (_, la), (_, lb) in zip(a.level_items, b.level_items):
        assert la.erm_stage.checkpoint.model_hash == lb.erm_stage.checkpoint.model_hash
        assert la.erm_stage.R_ERM_hat == lb.erm_stage.R_ERM_hat and la.erm_stage.tau == lb.erm_stage.tau
    # but the overall scientific result differs (the audit leakage used a different bootstrap)
    assert a.fold_result_hash != b.fold_result_hash


def test_training_change_changes_the_trained_model():
    a = _run(_MAN)
    for patch in ({"optimizer.lr_stage1": 9.9e-3}, {"training.stage1_steps_per_epoch": 3}):
        c = _run(_patched_path(patch))
        assert _training_fingerprint(a) != _training_fingerprint(c), f"training must change with {patch}"
        # ERM checkpoint specifically must differ
        ea = dict((int(l), lr.erm_stage.checkpoint.model_hash) for l, lr in a.level_items)
        ec = dict((int(l), lr.erm_stage.checkpoint.model_hash) for l, lr in c.level_items)
        assert ea != ec


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} optimization-seed-decoupling tests")


if __name__ == "__main__":
    _run_all()
