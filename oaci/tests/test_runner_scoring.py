"""A2b runner interface fixes: sampling-design / accumulation enforcement, the selection scoring
session (ERM cache 4/1/3, score-once), and mass-weighted eval aggregation.

Standalone (``python -m oaci.tests.test_runner_scoring``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np

from oaci.data.eeg.units import aggregate_mean_prob
from oaci.data.plan_materialize import materialize_full_domain_alignment_plan, materialize_oaci_alignment_plan
from oaci.data.plan_sampler import UnitIndex
from oaci.leakage import LeakageScoreCache, LeakageScoreKey
from oaci.runner import SelectionScoringSession
from oaci.support_graph import build_support_graph, empirical_class_prior
from oaci.train.checkpoint import CheckpointRecord, ERMStage, TrainResult, state_hash


def _setup(per_cell=4, n_dom=3, nc=2, m=4):
    sid, y, d, grp, unit, mass = [], [], [], [], [], []
    uid = 0
    counts = np.zeros((n_dom, nc), dtype=int)
    for dom in range(n_dom):
        for c in range(nc):
            for k in range(per_cell):
                sid.append(f"s{dom}_{c}_{k}"); y.append(c); d.append(dom)
                grp.append(f"rec{dom}-{k}"); unit.append(f"u{uid}"); mass.append(1.0); uid += 1
            counts[dom, c] = per_cell
    idx = UnitIndex(tuple(sid), np.array(y), np.array(d), tuple(grp), tuple(unit), np.array(mass, float))
    sg = build_support_graph(counts, m=m, cell_mass=counts.astype(float),
                             reference_prior=empirical_class_prior(counts))
    return idx, sg


POP = "pop-sig"


# ---------------- fix 1/2: sampling design + accumulation ----------------
def test_sampling_design_hash_binds_actual_mass_unit_mapping():
    # one recording, 4 windows, 2 mass units — same COUNTS, different unit->window partition
    sid = ("a", "b", "c", "e"); y = np.zeros(4, int); d = np.zeros(4, int)
    grp = ("r0", "r0", "r0", "r0"); mass = np.full(4, 0.5)
    a = UnitIndex(sid, y, d, grp, ("u0", "u0", "u1", "u1"), mass)
    b = UnitIndex(sid, y, d, grp, ("u0", "u1", "u0", "u1"), mass)
    assert a.design_hash() != b.design_hash()                # the actual mapping is bound


def test_alignment_plan_hash_binds_role_and_sampling_design():
    idx, sg = _setup()
    base = materialize_full_domain_alignment_plan(idx, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    diff = materialize_full_domain_alignment_plan(idx, POP, 1, 2, 1, per_cell=3, micro_size=64, base_seed=0)
    assert diff.plan_hash != base.plan_hash                  # sampling_design (per_cell) is in plan_hash
    oaci = materialize_oaci_alignment_plan(idx, sg, POP, 1, 2, 1, per_cell=2, micro_size=64, base_seed=0)
    assert oaci.plan_hash != base.plan_hash and oaci.role != base.role   # role is in plan_hash


def test_adv_accumulation_capacity_is_enforced():
    idx, sg = _setup(per_cell=4)                              # 6 observed cells x4 = 24 rows
    try:
        materialize_full_domain_alignment_plan(idx, POP, 1, 1, 1, per_cell=4, micro_size=4,
                                               base_seed=0, accumulation_steps=2)   # 6 microbatches > 2
    except ValueError as e:
        assert "accumulation" in str(e)
    else:
        raise AssertionError("a logical batch exceeding adv_accumulation_steps must fail")
    materialize_full_domain_alignment_plan(idx, POP, 1, 1, 1, per_cell=4, micro_size=4,
                                           base_seed=0, accumulation_steps=8)        # fits -> ok


# ---------------- fix 4: selection scoring session ----------------
def _ckpt(epoch, leak):
    return CheckpointRecord(epoch=epoch, optimizer_step=epoch + 1,
                            model_state={"w": _T(epoch)}, model_hash=f"s2-{epoch}", R_src=0.4,
                            balanced_err=0.0, train_surrogate=leak, lam=0.0)


def _T(v):
    import torch
    return torch.tensor([float(v)])


def _result(traj, erm_hash="erm"):
    erm = CheckpointRecord(epoch=-1, optimizer_step=0, model_state={"w": _T(-1)}, model_hash=erm_hash,
                           R_src=0.4, balanced_err=0.0, train_surrogate=0.5, lam=0.0)
    stage = ERMStage(checkpoint=erm, R_ERM_hat=0.4, tau=0.5, task_plan_hash="t", stage1_invocation_id="i")
    return TrainResult(method_name="OACI", active=True, inactive_reason=None, erm_stage=stage,
                       erm_record=erm, trajectory=traj, initial_model_hash=erm_hash,
                       task_plan_hash="t", alignment_plan_hash="a")


def test_selector_scores_each_unique_checkpoint_once():
    from oaci.train.selector import select_checkpoint
    calls = {"n": 0}

    def ck_scorer(record):
        calls["n"] += 1
        return {"bootstrap_ucl": -record.epoch}              # lower for later epochs

    cache = LeakageScoreCache()
    key = LeakageScoreKey("erm", "z", "p", "s", "f", "b", "c")
    erm_scorer = lambda: {"bootstrap_ucl": 0.9}
    sess = SelectionScoringSession("OACI", cache, key, "erm", erm_scorer, ck_scorer)
    res = _result([_ckpt(0, 0.5), _ckpt(1, 0.4)])
    select_checkpoint(res, score_fn=sess.score)
    assert calls["n"] == 2                                   # two unique Stage-2 checkpoints, scored once each


def test_erm_cache_counts_are_4_1_3():
    cache = LeakageScoreCache()
    key = LeakageScoreKey("erm", "z", "p", "s", "f", "b", "c")
    compute = {"n": 0}

    def erm_scorer():
        compute["n"] += 1
        return {"bootstrap_ucl": 0.7}

    cache.get_or_compute(key, erm_scorer)                    # level prefetch: request 1, compute 1
    for name in ("OACI", "global_lpc", "uniform"):           # three Stage-2 selectors
        SelectionScoringSession(name, cache, key, "erm", erm_scorer, lambda r: {"bootstrap_ucl": 0.0})
    assert cache.request_count(key) == 4 and cache.compute_count(key) == 1 and cache.hit_count(key) == 3
    assert compute["n"] == 1


# ---------------- section 11: mass-weighted eval aggregation ----------------
def test_mass_weighted_eval_aggregation_is_duplication_invariant():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    eu = ["u0", "u0", "u1"]
    _, a1, _ = aggregate_mean_prob(logits, eu)               # arithmetic (natural equal mass)
    logits2 = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])   # duplicate u0's first window
    eu2 = ["u0", "u0", "u0", "u1"]
    mass2 = [0.25, 0.25, 0.5, 1.0]                           # split the duplicated window's mass
    _, a2, _ = aggregate_mean_prob(logits2, eu2, sample_mass=mass2)
    assert np.allclose(a1, a2)                               # duplication + mass split leaves it unchanged


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} runner-scoring tests")


if __name__ == "__main__":
    _run_all()
