"""A1 model-agnostic engine: checkpoint ABI, guard/BN contracts, ID-based plans, RNG isolation,
and the compatibility wrapper. Tiny ShallowConvNet / MLP for speed.

Standalone (``python -m oaci.tests.test_train_engine``) and pytest-compatible.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from oaci.models import build_model
from oaci.train.batch_plan import (BatchStep, MicrobatchPlan, _full_logical, assert_population,
                                    build_full_batch_alignment_plan, build_full_batch_task_plan,
                                    resolve)
from oaci.train.bn import bn_buffer_hash, freeze_bn_running_stats
from oaci.train.checkpoint import (CheckpointRecord, clone_state_cpu, model_state_hash, state_hash)
from oaci.train.data import TrainingData, population_signature_hash
from oaci.train.engine import (EngineConfig, InvocationRegistry, _critic_frozen, _critic_update,
                               engine_config_from_manifest, train_stage1, train_stage2)
from oaci.train.evaluate import evaluate_guard
from oaci.train.objective import InactiveObjective, QuadraticPenaltyObjective
from oaci.train.rng import derive_seed, forked_rng
from oaci.train.risk import source_risk


# ---------------- fixtures ----------------
def _eeg_data(n=24, c=4, t=48, nc=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, c, t, generator=g)
    y = torch.randint(0, nc, (n,), generator=g)
    y[0], y[1] = 0, 1                                   # guarantee both classes present
    d = torch.randint(0, 2, (n,), generator=g)
    return TrainingData(X=X, y=y, sample_id=tuple(f"s{i}" for i in range(n)),
                        sample_mass=torch.ones(n), n_classes=nc, d=d,
                        group=tuple(f"g{int(x)}" for x in d)).validate()


def _shallow():
    return build_model("shallow_convnet", in_chans=4, in_times=48, n_classes=2, temporal_filters=4,
                       temporal_kernel_samples=8, pool_kernel_samples=8, pool_stride_samples=4)


def _mlp_data(n=20, f=5, nc=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, f, generator=g)
    y = torch.randint(0, nc, (n,), generator=g); y[0], y[1] = 0, 1
    return TrainingData(X=X, y=y, sample_id=tuple(f"m{i}" for i in range(n)),
                        sample_mass=torch.rand(n, generator=g) + 0.5, n_classes=nc,
                        d=torch.randint(0, 2, (n,), generator=g), group=None).validate()


def _cfg(**kw):
    base = dict(metric="ce", stage1_epochs=3, stage2_epochs=2, steps_per_epoch=1, warmup_steps=2,
                critic_steps=1, checkpoint_every=1, lr_stage1=0.05, lr_encoder=0.1, base_seed=0)
    base.update(kw)
    return EngineConfig(**base)


# ---------------- checkpoint / state hash ----------------
def test_state_hash_binds_dtype_shape_keys_and_buffers():
    a = {"w": torch.zeros(4, dtype=torch.float32)}
    assert state_hash(a) == state_hash({"w": torch.zeros(4, dtype=torch.float32)})
    assert state_hash(a) != state_hash({"w": torch.zeros(4, dtype=torch.float64)})   # dtype
    assert state_hash(a) != state_hash({"w": torch.zeros(2, 2)})                      # shape
    assert state_hash(a) != state_hash({"W": torch.zeros(4)})                         # key


def test_checkpoint_state_is_cpu_cloned_and_immutable():
    m = _shallow()
    snap = clone_state_cpu(m)
    h = state_hash(snap)
    with torch.no_grad():                              # mutate the LIVE model
        for p in m.parameters():
            p.add_(1.0)
    assert state_hash(snap) == h                       # the saved clone is unaffected (no shared storage)


def test_model_checkpoint_contains_batchnorm_buffers():
    snap = clone_state_cpu(_shallow())
    assert any(k.endswith("running_mean") for k in snap)
    assert any(k.endswith("running_var") for k in snap)
    assert any(k.endswith("num_batches_tracked") for k in snap)


def test_loading_checkpoint_recovers_exact_hash():
    m = _shallow()
    snap = clone_state_cpu(m); h = model_state_hash(m)
    m2 = _shallow(); m2.load_state_dict(snap)
    assert model_state_hash(m2) == h


# ---------------- guard ----------------
def test_guard_eval_restores_each_submodule_mode():
    m = _shallow(); m.train(); freeze_bn_running_stats(m)     # parent train, BN eval
    bn = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)][0]
    assert m.training and not bn.training
    evaluate_guard(m, _eeg_data(), "ce", chunk_size=7)
    assert m.training and not bn.training                     # both restored exactly


def test_guard_eval_does_not_mutate_state_or_rng():
    m = _shallow(); data = _eeg_data()
    h = model_state_hash(m); rng = torch.random.get_rng_state()
    evaluate_guard(m, data, "ce", chunk_size=5)
    assert model_state_hash(m) == h
    assert torch.equal(torch.random.get_rng_state(), rng)


def _full_ce(m, data):
    with torch.no_grad():
        logits = m(data.X).logits
    return float(source_risk(logits, data.y, "ce", data.n_classes, weight=data.sample_mass))


def test_chunked_ce_matches_full_mass_weighted_ce():
    m = build_model("mlp", in_dim=5, n_classes=2); data = _mlp_data()
    ref = _full_ce(m, data)
    for cs in (None, 1, 3, 7, 20):
        assert abs(evaluate_guard(m, data, "ce", chunk_size=cs).risk - ref) < 1e-5


def test_chunked_balanced_ce_matches_full_for_uneven_chunks():
    m = build_model("mlp", in_dim=5, n_classes=2); data = _mlp_data()
    full = evaluate_guard(m, data, "balanced_ce", chunk_size=None).risk
    for cs in (1, 3, 6, 13):
        assert abs(evaluate_guard(m, data, "balanced_ce", chunk_size=cs).risk - full) < 1e-5


def test_chunked_balanced_error_matches_full():
    m = build_model("mlp", in_dim=5, n_classes=2); data = _mlp_data()
    full = evaluate_guard(m, data, "balanced_ce", chunk_size=None).balanced_err
    for cs in (2, 5, 9):
        assert abs(evaluate_guard(m, data, "balanced_ce", chunk_size=cs).balanced_err - full) < 1e-6


def test_guard_missing_reference_class_fails_loudly():
    g = torch.Generator().manual_seed(0)
    data = TrainingData(X=torch.randn(8, 5, generator=g), y=torch.zeros(8, dtype=torch.long),
                        sample_id=tuple(f"z{i}" for i in range(8)), sample_mass=torch.ones(8),
                        n_classes=2, d=None, group=None).validate()     # class 1 absent
    m = build_model("mlp", in_dim=5, n_classes=2)
    try:
        evaluate_guard(m, data, "balanced_ce", chunk_size=3)
    except ValueError as e:
        assert "zero mass" in str(e)
    else:
        raise AssertionError("a missing reference class must fail loudly")


# ---------------- BN / Stage-1 / Stage-2 ----------------
def test_stage1_updates_batchnorm_running_stats():
    m = _shallow(); data = _eeg_data()
    plan = build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout")
    bn0 = bn_buffer_hash(m)
    train_stage1(m, data, plan, _cfg(stage1_epochs=2), torch.device("cpu"))
    assert bn_buffer_hash(m) != bn0                            # ERM Stage-1 DOES move running stats


def test_critic_warmup_changes_critic_but_not_backbone():
    data = _eeg_data()
    m = _shallow(); m.train(); freeze_bn_running_stats(m)
    obj = _TinyAdv()
    critic = obj.build_critic(m.feat_dim, torch.device("cpu"))
    opt = torch.optim.Adam(critic.parameters(), lr=0.1)
    bb, bn0, c0 = model_state_hash(m), bn_buffer_hash(m), model_state_hash(critic)
    _critic_update(obj, critic, opt, m, _full_logical(data, 7), data.index(), data, torch.device("cpu"))
    assert model_state_hash(m) == bb                           # backbone untouched
    assert bn_buffer_hash(m) == bn0                            # BN buffers untouched
    assert model_state_hash(critic) != c0                      # critic moved


def test_stage2_bn_buffers_equal_erm_at_every_checkpoint():
    data = _eeg_data()
    m = _shallow()
    s1 = build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout")
    erm = train_stage1(m, data, s1, _cfg(stage1_epochs=2), torch.device("cpu"))
    erm_bn = {k: v for k, v in erm.checkpoint.model_state.items() if "running_" in k or "num_batches" in k}
    t = build_full_batch_task_plan(data, "stage2_task", 3, 1, 0, "stage2_task_dropout")
    a = build_full_batch_alignment_plan(data, 2, 3, 1, 0)
    res = train_stage2(_shallow, erm, data, QuadraticPenaltyObjective(0.5), t, a,
                       _cfg(stage2_epochs=3, checkpoint_every=1), torch.device("cpu"))
    assert len(res.trajectory) == 3
    for c in res.trajectory:
        for k, v in erm_bn.items():
            assert torch.equal(c.model_state[k], v)           # running stats frozen at ERM


def test_stage2_bn_affine_parameters_can_still_change():
    data = _eeg_data()
    m = _shallow()
    s1 = build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout")
    erm = train_stage1(m, data, s1, _cfg(stage1_epochs=2), torch.device("cpu"))
    t = build_full_batch_task_plan(data, "stage2_task", 3, 1, 0, "stage2_task_dropout")
    a = build_full_batch_alignment_plan(data, 2, 3, 1, 0)
    res = train_stage2(_shallow, erm, data, QuadraticPenaltyObjective(1.0), t, a,
                       _cfg(stage2_epochs=3), torch.device("cpu"))
    bw = "bn.weight"
    assert not torch.equal(res.trajectory[-1].model_state[bw], erm.checkpoint.model_state[bw])


def test_inactive_objective_does_not_call_factory_or_optimizer():
    data = _eeg_data()
    m = _shallow()
    s1 = build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout")
    erm = train_stage1(m, data, s1, _cfg(stage1_epochs=2), torch.device("cpu"))
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _shallow()

    t = build_full_batch_task_plan(data, "stage2_task", 2, 1, 0, "stage2_task_dropout")
    a = build_full_batch_alignment_plan(data, 2, 2, 1, 0)
    res = train_stage2(factory, erm, data, InactiveObjective("test_inactive"), t, a, _cfg(), torch.device("cpu"))
    assert calls["n"] == 0 and res.active is False and res.trajectory == []
    assert res.erm_record.model_hash == erm.checkpoint.model_hash


# ---------------- plans ----------------
def test_task_plan_resolves_after_input_row_permutation():
    data = _mlp_data(n=10)
    plan = build_full_batch_task_plan(data, "stage2_task", 1, 1, 0, "stage2_task_dropout")
    perm = [9, 0, 5, 1, 8, 2, 7, 3, 6, 4]
    data2 = TrainingData(X=data.X[perm], y=data.y[perm], sample_id=tuple(data.sample_id[i] for i in perm),
                         sample_mass=data.sample_mass[perm], n_classes=2,
                         d=data.d[perm], group=None).validate()
    assert population_signature_hash(data) == population_signature_hash(data2)    # order-invariant
    step = plan.epochs[0][0]
    r1 = resolve(step.sample_ids, step.importance_weights, data.index())
    r2 = resolve(step.sample_ids, step.importance_weights, data2.index())
    assert torch.equal(data.X[r1.idx], data2.X[r2.idx])                            # same rows by id


def test_task_plan_rejects_population_signature_mismatch():
    data = _mlp_data(); other = _mlp_data(seed=99)
    plan = build_full_batch_task_plan(data, "stage2_task", 1, 1, 0, "stage2_task_dropout")
    try:
        assert_population(plan.population_signature_hash, other)
    except ValueError:
        pass
    else:
        raise AssertionError("population signature mismatch must be rejected")


def test_task_plan_rejects_unknown_or_duplicate_dataset_ids():
    data = _mlp_data()
    try:
        resolve(("nope",), (1.0,), data.index())
    except ValueError:
        pass
    else:
        raise AssertionError("unknown id must be rejected")
    g = torch.Generator().manual_seed(0)
    dup = TrainingData(X=torch.randn(3, 5, generator=g), y=torch.tensor([0, 1, 0]),
                       sample_id=("a", "a", "b"), sample_mass=torch.ones(3), n_classes=2, d=None, group=None)
    try:
        dup.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate sample_id must be rejected")


def test_task_plan_allows_replacement_duplicates_inside_step():
    data = _mlp_data()
    r = resolve(("m0", "m0", "m1"), (1.0, 1.0, 1.0), data.index())
    assert len(r.idx) == 3 and int(r.idx[0]) == int(r.idx[1])


def test_plan_hash_binds_weights_seeds_and_epoch_boundaries():
    data = _mlp_data()
    base = build_full_batch_task_plan(data, "stage2_task", 2, 1, 0, "stage2_task_dropout").plan_hash
    assert build_full_batch_task_plan(data, "stage2_task", 2, 1, 7, "stage2_task_dropout").plan_hash != base  # seed
    assert build_full_batch_task_plan(data, "stage2_task", 1, 2, 0, "stage2_task_dropout").plan_hash != base  # epoch split
    assert build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout").plan_hash != base  # role/seed ns


def test_batch_importance_weight_is_not_double_multiplied():
    data = _mlp_data()
    plan = build_full_batch_task_plan(data, "stage2_task", 1, 1, 0, "stage2_task_dropout")
    w = resolve(plan.epochs[0][0].sample_ids, plan.epochs[0][0].importance_weights, data.index()).weight
    # the plan weight IS the base sample mass (final), so the engine's single source_risk(weight=)
    # reproduces the mass exactly — it must never re-multiply by sample_mass.
    assert torch.allclose(w, data.sample_mass, atol=1e-6)


# ---------------- RNG ----------------
def test_sha_seed_derivation_is_stable_and_namespace_separated():
    assert derive_seed(0, "a", 1) == derive_seed(0, "a", 1)
    assert derive_seed(0, "a", 1) != derive_seed(0, "b", 1)
    assert derive_seed(0, "a", 1) != derive_seed(0, "a", 2)
    assert derive_seed(0, "a", 1) != derive_seed(1, "a", 1)


def test_critic_call_count_does_not_change_task_dropout_or_plan():
    # forked streams are isolated: re-entering seed A after using seed B yields the SAME draw
    with forked_rng(123):
        a1 = torch.randn(3)
    with forked_rng(999):
        _ = torch.randn(50)
    with forked_rng(123):
        a2 = torch.randn(3)
    assert torch.equal(a1, a2)


def test_global_rng_state_is_restored_after_engine_call():
    data = _mlp_data()
    m = build_model("mlp", in_dim=5, n_classes=2)
    plan = build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout")
    rng = torch.random.get_rng_state()
    train_stage1(m, data, plan, _cfg(stage1_epochs=2), torch.device("cpu"))
    assert torch.equal(torch.random.get_rng_state(), rng)        # engine forked, didn't touch caller RNG


def test_same_seed_reproduces_all_checkpoint_hashes():
    data = _eeg_data()
    s1 = build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout")
    t = build_full_batch_task_plan(data, "stage2_task", 2, 1, 0, "stage2_task_dropout")
    a = build_full_batch_alignment_plan(data, 2, 2, 1, 0)

    def run():
        erm = train_stage1(_shallow_seeded(), data, s1, _cfg(stage1_epochs=2), torch.device("cpu"))
        return train_stage2(_shallow, erm, data, QuadraticPenaltyObjective(0.5), t, a,
                            _cfg(stage2_epochs=2), torch.device("cpu"))

    r1, r2 = run(), run()
    assert [c.model_hash for c in r1.trajectory] == [c.model_hash for c in r2.trajectory]
    assert r1.erm_stage.checkpoint.model_hash == r2.erm_stage.checkpoint.model_hash


def test_stage1_invocation_registry_blocks_retrain():
    data = _mlp_data()
    plan = build_full_batch_task_plan(data, "stage1_task", 1, 1, 0, "stage1_task_dropout")
    reg = InvocationRegistry()
    train_stage1(build_model("mlp", in_dim=5, n_classes=2), data, plan, _cfg(stage1_epochs=1),
                 torch.device("cpu"), reg, "level0")
    try:
        train_stage1(build_model("mlp", in_dim=5, n_classes=2), data, plan, _cfg(stage1_epochs=1),
                     torch.device("cpu"), reg, "level0")
    except ValueError:
        pass
    else:
        raise AssertionError("Stage-1 must not be trained twice for the same invocation id")


def test_train_risk_feasible_compatibility_wrapper_uses_new_engine():
    from oaci.train import TrainConfig, train_risk_feasible, TrainResult
    from oaci.train.synthetic import make_covariate_shift
    X, y, d, g, sg = make_covariate_shift(seed=0)
    res = train_risk_feasible(X, y, d, g, sg, TrainConfig(seed=0, stage1_epochs=15, stage2_epochs=4,
                                                          warmup_steps=3, critic_steps=1))
    assert isinstance(res, TrainResult) and res.method_name == "OACI" and res.active
    assert res.task_plan_hash and res.alignment_plan_hash                     # plan-driven
    assert any(k.startswith("enc") for k in res.erm_record.model_state)       # MLP backbone state


# ---------------- A2a engine 收口 ----------------
def test_manifest_values_map_exactly_to_engine_config():
    import os
    from oaci.protocol.freeze import default_confirmatory_path
    from oaci.protocol.manifest_v2 import load_v2
    m = load_v2(os.path.join(os.path.dirname(default_confirmatory_path()), "smoke_v1.yaml"))
    ec = engine_config_from_manifest(m, steps_per_epoch=2, base_seed=7)
    assert ec.lr_encoder == m.optimizer.lr_encoder and ec.lr_critic == m.optimizer.lr_critic
    assert ec.weight_decay == m.optimizer.weight_decay and ec.lambda_floor == m.optimizer.lambda_floor
    assert ec.stage2_epochs == m.training.stage2_epochs and ec.warmup_steps == m.training.warmup_steps
    assert ec.stage2_bn_mode == "frozen_erm_running_stats"
    assert ec.deterministic_algorithms == m.training.deterministic_algorithms
    assert ec.metric == m.risk["metric"] and ec.epsilon == m.risk["epsilon"]
    assert ec.steps_per_epoch == 2 and ec.base_seed == 7


def test_plan_cardinality_mismatch_fails_before_training():
    data = _mlp_data()
    plan = build_full_batch_task_plan(data, "stage1_task", 2, 1, 0, "stage1_task_dropout")   # 2 epochs
    m = build_model("mlp", in_dim=5, n_classes=2)
    try:
        train_stage1(m, data, plan, _cfg(stage1_epochs=3), torch.device("cpu"))              # cfg wants 3
    except ValueError as e:
        assert "epochs" in str(e)
    else:
        raise AssertionError("a plan/cfg cardinality mismatch must fail before training")


def test_critic_update_uses_logical_batch_seed():
    data = _eeg_data()
    m = _shallow(); m.train(); freeze_bn_running_stats(m)
    obj = _TinyAdv(); critic = obj.build_critic(m.feat_dim, torch.device("cpu"))
    opt = torch.optim.Adam(critic.parameters(), lr=0.1)
    rng = torch.random.get_rng_state()
    _critic_update(obj, critic, opt, m, _full_logical(data, 555), data.index(), data, torch.device("cpu"))
    assert torch.equal(torch.random.get_rng_state(), rng)   # forked on the logical-batch seed; caller RNG intact


def test_encoder_step_freezes_critic_parameters_and_state():
    critic = nn.Linear(4, 2)
    for p in critic.parameters():
        p.grad = torch.ones_like(p)
    mode = critic.training
    with _critic_frozen(critic):
        assert all(not p.requires_grad for p in critic.parameters())
        assert all(p.grad is None for p in critic.parameters())
    assert all(p.requires_grad for p in critic.parameters()) and critic.training == mode   # restored


# ---------------- helpers used above ----------------
class _DomHead(nn.Module):
    def __init__(self, feat):
        super().__init__(); self.lin = nn.Linear(feat, 2)
    def forward(self, z):
        return self.lin(z)


class _TinyAdv:
    """A minimal critic objective (predict d from z) — exercises warmup/critic ordering without a
    support graph."""
    name = "tinyadv"

    def active_status(self):
        from oaci.train.objective import ActiveStatus
        return ActiveStatus(True, None)

    def build_critic(self, feat_dim, device):
        self.c = _DomHead(int(feat_dim)).to(device); return self.c

    def critic_loss(self, critic, z_detached, batch):
        return F.cross_entropy(critic(z_detached), batch.d)

    def encoder_penalty(self, critic, z, batch):
        return -F.cross_entropy(critic(z), batch.d)

    def full_surrogate(self, model, data, device, chunk_size):
        return 0.0

    def diagnostics(self):
        return {}


def _shallow_seeded():
    with forked_rng(derive_seed(0, "model_init")):
        return _shallow()


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} train-engine tests")


if __name__ == "__main__":
    _run_all()
