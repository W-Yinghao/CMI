"""A2a formal methods: OACI eligibility, full-domain posterior critic, global-LPC / uniform priors,
shared alignment plan, and per-method activity. Small MLP + synthetic support graphs.

Standalone (``python -m oaci.tests.test_methods``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np
import torch

from oaci.methods import (GlobalLPCObjective, OACIObjective, UniformObjective, all_method_status,
                          erm_result, method_status)
from oaci.models import build_model
from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior
from oaci.train.batch_plan import (eligible_rows, materialize_full_domain_alignment_plan,
                                   materialize_oaci_alignment_plan)
from oaci.train.bn import all_eval
from oaci.train.data import TrainingData
from oaci.train.objective import BatchView


def _setup(per=(12, 12, 12), nc=2, m=4, seed=0, drop=None):
    """`per[dom]` rows per class in domain `dom`; `drop=(dom,cls)` removes that cell's rows."""
    rng = np.random.default_rng(seed)
    y, d = [], []
    for dom, k in enumerate(per):
        for c in range(nc):
            if drop is not None and (dom, c) == drop:
                continue
            y += [c] * k; d += [dom] * k
    y, d = np.array(y), np.array(d)
    counts = counts_from_labels(d, y, n_domains=len(per), n_classes=nc)
    sg = build_support_graph(counts, m=m, cell_mass=counts.astype(float),
                             reference_prior=empirical_class_prior(counts))
    X = rng.standard_normal((len(y), 5)).astype(np.float32)
    data = TrainingData(X=torch.tensor(X), y=torch.tensor(y), sample_id=tuple(f"r{i}" for i in range(len(y))),
                        sample_mass=torch.ones(len(y)), n_classes=nc, d=torch.tensor(d),
                        group=tuple(f"g{int(x)}" for x in d)).validate()
    return data, sg


def _class_mass(data, nc=2):
    return {c: float(data.sample_mass[data.y == c].sum()) for c in range(nc)}


def _bv(data):
    return BatchView(data.y, data.d, data.sample_mass)


# ---------------- OACI ----------------
def test_oaci_rejects_any_ineligible_alignment_row():
    data, sg = _setup(drop=(2, 1))                      # domain 2 has no class-1 rows -> 2 ∉ S_1
    assert 2 not in sg.support_of_class[1]
    obj = OACIObjective(sg)
    critic = obj.build_critic(5, torch.device("cpu"))
    bad = BatchView(torch.tensor([1]), torch.tensor([2]), torch.tensor([1.0]))   # (y=1, d=2): ineligible
    try:
        obj.critic_loss(critic, torch.randn(1, 5), bad)
    except ValueError:
        pass
    else:
        raise AssertionError("OACI must reject an ineligible alignment row, not silently drop it")


def test_oaci_full_and_streamed_surrogates_match():
    data, sg = _setup()
    obj = OACIObjective(sg)
    m = build_model("mlp", in_dim=5, n_classes=2)
    obj.build_critic(m.feat_dim, torch.device("cpu"))
    assert abs(obj.full_surrogate(m, data, torch.device("cpu"), None)
               - obj.full_surrogate(m, data, torch.device("cpu"), 7)) < 1e-5


# ---------------- full-domain coverage ----------------
def test_full_domain_critic_uses_every_observed_cell():
    data, sg = _setup(per=(12, 12, 2), m=4)             # cell (2,*) has 2 rows < m: present, not eligible
    full = materialize_full_domain_alignment_plan(data, 1, 1, 1, 0)
    full_ids = set(full.warmup_batches[0].microbatches[0].sample_ids)
    oaci = materialize_oaci_alignment_plan(data, sg, 1, 1, 1, 0)
    oaci_ids = set(oaci.warmup_batches[0].microbatches[0].sample_ids)
    assert len(full_ids) == len(data)                   # full-domain = ALL observed rows
    assert oaci_ids < full_ids                          # OACI strictly fewer (eligible only)


def test_full_domain_critic_never_fabricates_missing_rows():
    data, sg = _setup(drop=(2, 1))                      # (2,1) has zero rows
    full = materialize_full_domain_alignment_plan(data, 1, 1, 1, 0)
    ids = full.warmup_batches[0].microbatches[0].sample_ids
    idx = data.index()
    for sid in ids:                                     # no plan row is a (d=2,y=1) row
        i = idx[sid]
        assert not (int(data.d[i]) == 2 and int(data.y[i]) == 1)
    obj = GlobalLPCObjective([0, 1, 2], sg.cell_mass, _class_mass(data), sg.reference_prior, [0, 1], alpha=1.0)
    assert obj.prior_vector(1)[2] > 0                   # missing cell still has positive PRIOR mass


# ---------------- global-LPC / uniform priors ----------------
def test_global_lpc_prior_uses_current_cell_mass_and_level0_universe():
    data, sg = _setup(per=(20, 8, 12))
    obj = GlobalLPCObjective([0, 1, 2], sg.cell_mass, _class_mass(data), sg.reference_prior, [0, 1], alpha=1.0)
    pri = obj.prior_vector(0)
    assert len(pri) == 3 and abs(pri.sum() - 1.0) < 1e-12 and (pri > 0).all()
    exp = sg.cell_mass[:, 0] + 1.0; exp = exp / exp.sum()
    assert np.allclose(pri, exp)                        # ∝ (M_{d,0}+α) over the level-0 universe


def test_global_lpc_missing_cell_has_positive_prior_and_zero_rows():
    data, sg = _setup(drop=(2, 1))
    obj = GlobalLPCObjective([0, 1, 2], sg.cell_mass, _class_mass(data), sg.reference_prior, [0, 1], alpha=0.5)
    pri = obj.prior_vector(1)
    assert abs(pri.sum() - 1.0) < 1e-12 and pri[2] > 0  # smoothed prior mass on the deleted cell
    assert int((data.d[data.y == 1] == 2).sum()) == 0   # but zero observed rows


def test_uniform_target_is_exact_and_deletion_invariant():
    d0, sg0 = _setup()
    d1, sg1 = _setup(drop=(2, 1))                        # level-1 (a cell deleted) — same universe
    u0 = UniformObjective([0, 1, 2], _class_mass(d0), sg0.reference_prior, [0, 1])
    u1 = UniformObjective([0, 1, 2], _class_mass(d1), sg1.reference_prior, [0, 1])
    assert np.allclose(u0.prior_vector(0), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(u0.prior_vector(0), u1.prior_vector(0))   # depends only on |D0|


def test_global_lpc_and_uniform_share_alignment_plan():
    data, sg = _setup()
    a = materialize_full_domain_alignment_plan(data, 2, 3, 1, 0)
    b = materialize_full_domain_alignment_plan(data, 2, 3, 1, 0)
    assert a.plan_hash == b.plan_hash                    # both baselines share this exact plan
    assert a.plan_hash != materialize_oaci_alignment_plan(data, sg, 2, 3, 1, 0).plan_hash


# ---------------- learning dynamics ----------------
def _frozen_z(model, data):
    with all_eval(model), torch.no_grad():
        return model(data.X).z


def test_full_domain_critic_step_reduces_weighted_domain_ce():
    data, sg = _setup()
    obj = GlobalLPCObjective([0, 1, 2], sg.cell_mass, _class_mass(data), sg.reference_prior, [0, 1], alpha=1.0)
    m = build_model("mlp", in_dim=5, n_classes=2)
    critic = obj.build_critic(m.feat_dim, torch.device("cpu"))
    opt = torch.optim.Adam(critic.parameters(), lr=0.05)
    z = _frozen_z(m, data); bv = _bv(data)
    cd0 = float(obj.critic_loss(critic, z, bv))
    for _ in range(30):
        opt.zero_grad(); obj.critic_loss(critic, z, bv).backward(); opt.step()
    assert float(obj.critic_loss(critic, z, bv)) < cd0   # critic learns to predict D


def test_lpc_encoder_step_reduces_weighted_posterior_kl():
    data, sg = _setup()
    obj = GlobalLPCObjective([0, 1, 2], sg.cell_mass, _class_mass(data), sg.reference_prior, [0, 1], alpha=1.0)
    m = build_model("mlp", in_dim=5, n_classes=2)
    critic = obj.build_critic(m.feat_dim, torch.device("cpu"))
    opt_c = torch.optim.Adam(critic.parameters(), lr=0.05)
    z = _frozen_z(m, data); bv = _bv(data)
    for _ in range(30):                                  # warm the critic first
        opt_c.zero_grad(); obj.critic_loss(critic, z, bv).backward(); opt_c.step()
    for p in critic.parameters():
        p.requires_grad_(False)
    pen0 = float(obj.encoder_penalty(critic, m(data.X).z, bv))
    opt_m = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(30):
        opt_m.zero_grad(); obj.encoder_penalty(critic, m(data.X).z, bv).backward(); opt_m.step()
    assert float(obj.encoder_penalty(critic, m(data.X).z, bv)) < pen0   # encoder pushes q -> π


# ---------------- activity ----------------
def test_method_activity_rules():
    data, sg = _setup()
    st = all_method_status(sg, level0_universe_size=3, n_observed_source_domains=3)
    assert all(st[m].active for m in ("ERM", "OACI", "global_lpc", "uniform"))
    st1 = all_method_status(sg, level0_universe_size=1, n_observed_source_domains=1)
    assert not st1["global_lpc"].active and not st1["uniform"].active and st1["ERM"].active
    _, sg_single = _setup(per=(12,))                     # one domain -> no comparable class
    assert sg_single.comparable_classes == []
    assert not method_status("OACI", sg_single, 1, 1).active


def test_full_surrogate_restores_model_and_critic_modes():
    data, sg = _setup()
    obj = OACIObjective(sg)
    m = build_model("mlp", in_dim=5, n_classes=2)
    critic = obj.build_critic(m.feat_dim, torch.device("cpu"))
    m.train(); critic.train()
    obj.full_surrogate(m, data, torch.device("cpu"), None)
    assert m.training and critic.training                # full_surrogate restores both modes
    gl = GlobalLPCObjective([0, 1, 2], sg.cell_mass, _class_mass(data), sg.reference_prior, [0, 1], alpha=1.0)
    c2 = gl.build_critic(m.feat_dim, torch.device("cpu")); m.train(); c2.train()
    gl.full_surrogate(m, data, torch.device("cpu"), None)
    assert m.training and c2.training


def test_erm_result_is_byte_exact_passthrough():
    from oaci.train.checkpoint import CheckpointRecord, ERMStage
    state = {"w": torch.zeros(3)}
    from oaci.train.checkpoint import state_hash
    rec = CheckpointRecord(epoch=-1, optimizer_step=0, model_state=state, model_hash=state_hash(state),
                           R_src=0.4, balanced_err=0.1, train_surrogate=0.0, lam=0.0)
    stage = ERMStage(checkpoint=rec, R_ERM_hat=0.4, tau=0.43, task_plan_hash="t", stage1_invocation_id="i")
    res = erm_result(stage)
    assert res.method_name == "ERM" and res.active and res.trajectory == []
    assert res.erm_record.model_hash == rec.model_hash and res.alignment_plan_hash is None


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} methods tests")


if __name__ == "__main__":
    _run_all()
