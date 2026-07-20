"""C87 production-module unit tests (fast; no full control gate). Validates the estimator identities,
the v3 cross-fit semantics, the gate verdict caps, and policy/generator contracts."""
import numpy as np
import pytest

from oaci.active_testing.c87 import estimand as E
from oaci.active_testing.c87 import generator as G
from oaci.active_testing.c87 import lure as L
from oaci.active_testing.c87 import policies as P
from oaci.active_testing.c87 import gate as GT


def test_lure_uniform_identity():
    N, M = 40, 15
    w = np.ones(N)
    order = np.random.default_rng(0).choice(N, M, replace=False)
    q = L.without_replacement_proposal_sequence(w, order)
    v = L.lure_weights(N, q)
    assert np.allclose(v, 1.0, atol=1e-12)                       # uniform proposal => all weights 1
    loss = np.random.default_rng(1).random(M)
    assert np.isclose(L.lure_risk(loss, q, N), loss.mean())       # == naive mean


def test_lure_unbiased_nonuniform():
    rng = np.random.default_rng(2); N = 30
    full = rng.random(N); truth = full.mean()
    w = rng.random(N) + 0.1; w /= w.sum()
    ests = []
    for _ in range(3000):
        order = rng.choice(N, 15, replace=False, p=w)
        q = L.without_replacement_proposal_sequence(w, order)
        ests.append(L.lure_risk(full[order], q, N))
    assert abs(np.mean(ests) - truth) < 0.01                     # unbiased under any proposal


def test_crossfit_reference_can_be_negative():
    """v3: cross-fitted excess loss MAY be negative (held-SELECTION reference, not an unbiased oracle),
    and the reference is >= the optimistic in-sample argmin ON AVERAGE (winner's-curse correction)."""
    Ts, refs, mins = [], [], []
    for s in range(40):
        w = G.make_world("POS_DENSE", A=648, n_pat=400, E=1, seed=7000 + s)[0]
        Lbar, pat = E.patient_mean_loss(E.binary_nll(w.probs, w.y), w.patient_of)
        cf = E.cross_fit(Lbar, pat)
        Ts.append(E.transport_gap_cf(Lbar, w.aC, cf))
        refs.append(cf.ref); mins.append(E.held_view_loss(Lbar).min())
    assert all(np.isfinite(Ts))
    assert min(Ts) < 0                                   # excess loss CAN be negative (v3)
    assert np.mean(refs) >= np.mean(mins)                # cross-fit ref less optimistic than in-sample min


def test_s_e_is_full_set_dispersion_no_crossfit():
    w = G.make_world("POS", A=120, n_pat=150, E=1, seed=4)[0]
    Lbar, _ = E.patient_mean_loss(E.binary_nll(w.probs, w.y), w.patient_of)
    assert np.isclose(E.dispersion_s_e(Lbar), np.std(E.held_view_loss(Lbar), ddof=0))


def test_gate_georgia_exclusion_caps_verdict():
    """A surviving cell whose Georgia is vacuous can NOT be an unqualified DEMONSTRATED (v2/v3 cap)."""
    def cell(gv):
        return GT.GridCell("MODEL-SELECTOR", 32, [
            GT.CohortResult("chapman", 0.4, vacuous=False, is_georgia=False),
            GT.CohortResult("ningbo", 0.4, vacuous=False, is_georgia=False),
            GT.CohortResult("georgia", 0.4, vacuous=gv, is_georgia=True)])
    v_ok, _ = GT.program_verdict([cell(False)])
    v_vac, _ = GT.program_verdict([cell(True)])
    assert v_ok == "DEMONSTRATED"
    assert v_vac in ("WITH_CAVEAT", "INCONCLUSIVE") and v_vac != "DEMONSTRATED"


def test_gate_no_survivors_is_no_advantage():
    cell = GT.GridCell("MODEL-SELECTOR", 32, [
        GT.CohortResult("chapman", 0.0, is_georgia=False),
        GT.CohortResult("ningbo", 0.0, is_georgia=False),
        GT.CohortResult("georgia", 0.0, is_georgia=True)])
    v, _ = GT.program_verdict([cell])
    assert v == "NO_ADVANTAGE"


def test_gate_fewer_than_three_cohorts_inconclusive():
    cell = GT.GridCell("MODEL-SELECTOR", 32, [
        GT.CohortResult("chapman", 0.4, is_georgia=False),
        GT.CohortResult("ningbo", 0.4, is_georgia=False)])
    v, _ = GT.program_verdict([cell])
    assert v == "INCONCLUSIVE"


def test_policies_run_and_shapes():
    w = G.make_world("POS", A=80, n_pat=100, E=1, seed=5)[0]
    for pol in P.PRIMARY:
        a, q = pol.select(w.probs, w.y, w.patient_of, 16, np.random.default_rng(0))
        assert 0 <= a < w.probs.shape[0] and len(q) <= 16


def test_label_adaptive_beats_p0_on_pos_mean():
    """Sanity (aggregate over many worlds x seeds to beat MC noise): the label-adaptive MODEL-SELECTOR
    has strictly lower mean held regret than P0 on POS at a small budget."""
    regs = {"P0": [], "MODEL-SELECTOR": []}
    for s in range(25):
        w = G.make_world("POS", A=648, n_pat=400, E=1, seed=1000 + s)[0]
        Lbar, _ = E.patient_mean_loss(E.binary_nll(w.probs, w.y), w.patient_of)
        Lh = E.held_view_loss(Lbar); best = Lh.min()
        for name, pol in [("P0", P.P0()), ("MODEL-SELECTOR", P.ModelSelector())]:
            for k in range(4):
                a = pol.select(w.probs, w.y, w.patient_of, 16, np.random.default_rng(10 + s * 4 + k))[0]
                regs[name].append(Lh[a] - best)
    assert np.mean(regs["MODEL-SELECTOR"]) < np.mean(regs["P0"])


def test_generator_worlds_shapes():
    for kind in ("POS", "POS_DENSE", "NEG_A", "NEG_B", "CALIB"):
        w = G.make_world(kind, A=50, n_pat=60, E=3, seed=0)
        assert len(w) == 3
        for coh in w:
            assert coh.probs.shape[0] == 50
            assert coh.y.shape[0] == coh.probs.shape[1] == coh.patient_of.shape[0]
            assert set(np.unique(coh.y).tolist()) <= {0, 1}
