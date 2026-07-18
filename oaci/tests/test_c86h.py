"""C86H prep tests — synthetic / metadata / failure only. No real EEG/label is touched.

Covers the §12 bindings, the §11 registered thresholds (NOT the C86D TAU=0.02), the
two-level taxonomy (formal C86-A..E + L1..L4 vs the interpretive descriptor with
POLICY_LIMITED fixed to NOT_IDENTIFIABLE), the confirmatory inference (sign-flip max-T,
tail CVaR, favorable/worst/cells/LOTO), the locked label-blind split (deterministic +
fail-closed), INPUT_UNAVAILABLE, and the gated entrypoint refusals.
"""
import numpy as np
import pytest

from oaci.active_testing.c86h import analysis as A
from oaci.active_testing.c86h import contract as K
from oaci.active_testing.c86h import entrypoint as E


# ------------------------------------------------------------------- 12 bindings
def test_bindings_verify_ok():
    out = K.verify_bindings()
    assert out["ok"], out["mismatches"]


def test_thresholds_are_registered_not_c86d_tau():
    # C86D development used TAU=0.02; C86H must use the registered materiality 0.05.
    assert K.MATERIALITY_MARGIN == 0.05
    assert K.FAMILYWISE_ALPHA == 0.05
    assert K.MAXT_DRAWS == 65_536
    assert K.FAVORABLE_TARGET_FRACTION == 0.75
    assert K.WORST_TARGET_EFFECT_FLOOR == -0.10
    assert K.POSITIVE_CELLS_MIN == 6
    assert K.TAIL_CVAR90_MARGIN == 0.05
    assert K.LOTO_PRESERVATION_MIN == 0.75
    assert K.ACTIVE_CHAINS == 2_048
    assert K.POOLED_DATASET_PVALUE == "FORBIDDEN"


def test_population_and_field_metadata():
    assert K.N_TARGETS == 53
    assert K.COHORTS["Brandl2020_CANONICAL_ADULT_V1"]["n_adult"] == 16
    ds = K.COHORTS["OpenNeuro_ds007221_HYBRID_ADULT_V1"]
    assert ds["n_adult"] == 37
    assert ds["subjects"] == tuple(f"sub-{i}" for i in range(37, 74))
    assert ds["task"] == "hybrid"
    assert K.INTERFACE_EVENTS == ("left_hand", "right_hand")
    assert len(K.INTERFACE_CHANNELS) == 11
    assert K.UNIQUE_TRAINED_MODELS == 648
    assert K.TARGET_CONTEXTS == 424
    assert K.METHOD_REGISTRY == ("P0", "A1", "A2H")


def test_maxt_family_is_realized_two_active():
    # frozen registry has 2 active methods -> family = {A1,A2H} x 4 finite budgets = 8
    assert K.ACTIVE_METHODS == ("A1", "A2H")
    assert len(K.ACTIVE_METHODS) * len(K.FINITE_BUDGETS) == 8


# ------------------------------------------------------------- confirmatory inference
def test_maxt_flags_strong_and_ignores_null():
    rng = np.random.default_rng(0)
    n = 53
    strong = rng.normal(0.30, 0.05, n)      # clearly positive
    nulle = rng.normal(0.00, 0.05, n)       # centred on zero
    fam = {("A1", 8): strong, ("A2H", 8): nulle,
           ("A1", 16): nulle, ("A2H", 16): nulle}
    out = A.maxt_familywise(fam, seed=A.maxt_seed("cohortX"), draws=2000)
    assert out["significant"][("A1", 8)] is True
    assert out["significant"][("A2H", 8)] is False


def test_tail_qualification_positive_and_negative():
    p0 = np.linspace(0.2, 0.9, 60)          # heavier tail
    active = p0 - 0.10                       # uniformly lower loss
    q = A.tail_qualification(p0, active)
    assert q["qualified"] and q["primary_ok"] and q["all_alpha_nonnegative"]
    # active heavier in the tail -> not qualified
    worse = p0.copy(); worse[-5:] += 0.5
    q2 = A.tail_qualification(p0, worse)
    assert q2["qualified"] is False


def test_point_stats():
    e = np.array([0.1, 0.2, -0.05, 0.3, 0.15])
    assert 0.0 <= A.favorable_fraction(e) <= 1.0
    assert A.worst_target(e) == -0.05
    assert A.positive_cells([0.1, -0.1, 0.2, 0.0, 0.3, 0.05, -0.2, 0.4]) == 5
    good = np.full(20, 0.2)
    assert A.loto_preservation(good) == 1.0
    assert A.loto_preservation(np.full(20, 0.0)) == 0.0


def test_budget_status_input_unavailable_no_substitution():
    assert A.budget_status(8, pool_size=40) == "SUPPORTED"
    assert A.budget_status(64, pool_size=40) == "INPUT_UNAVAILABLE"
    assert A.budget_status("FULL", pool_size=40) == "SUPPORTED"


# --------------------------------------------------------------- formal gate C86-A..E
def _uniform(mean_ok, tail_ok):
    fam = [(m, b) for m in K.ACTIVE_METHODS for b in K.FINITE_BUDGETS]
    return {"mean": {mb: mean_ok for mb in fam}, "tail": {mb: tail_ok for mb in fam}}


def test_gate_A_when_mean_and_tail_everywhere():
    pc = {"c1": _uniform(True, True), "c2": _uniform(True, True)}
    assert A.formal_gate(pc) == "C86-A"


def test_gate_B_mean_but_not_tail():
    pc = {"c1": _uniform(True, False), "c2": _uniform(True, False)}
    assert A.formal_gate(pc) == "C86-B"


def test_gate_C_no_mean_anywhere():
    pc = {"c1": _uniform(False, False), "c2": _uniform(False, False)}
    assert A.formal_gate(pc) == "C86-C"


def test_gate_D_heterogeneous():
    # mean qualifies in c1 only -> not A/B (not every cohort), not C (some mean) -> D
    pc = {"c1": _uniform(True, True), "c2": _uniform(False, False)}
    assert A.formal_gate(pc) == "C86-D"


def test_gate_E_blocker_precedence():
    pc = {"c1": _uniform(True, True), "c2": _uniform(True, True)}
    assert A.formal_gate(pc, blocker=True) == "C86-E"


# ------------------------------------------------------------- label frontier L1..L4
def _mean_map(spec):
    """spec[cohort][method] = set of qualifying finite budgets."""
    fam = [(m, b) for m in K.ACTIVE_METHODS for b in K.FINITE_BUDGETS]
    pc = {}
    for c, methods in spec.items():
        mm = {mb: False for mb in fam}
        for m, budgets in methods.items():
            for b in budgets:
                mm[(m, b)] = True
        pc[c] = {"mean": mm, "tail": {mb: False for mb in fam}}
    return pc


def test_frontier_L1_small_budget_same_method():
    pc = _mean_map({"c1": {"A1": {8, 16, 32}}, "c2": {"A1": {8, 16, 32}}})
    assert A.label_frontier(pc) == "C86-L1"


def test_frontier_L2_mid_budget():
    pc = _mean_map({"c1": {"A1": {16, 32}}, "c2": {"A1": {16, 32}}})
    assert A.label_frontier(pc) == "C86-L2"


def test_frontier_L3_heterogeneous_methods():
    pc = _mean_map({"c1": {"A1": {8, 16, 32}}, "c2": {"A2H": {8, 16, 32}}})
    assert A.label_frontier(pc) == "C86-L3"


def test_frontier_L4_absent_in_a_cohort():
    pc = _mean_map({"c1": {"A1": {8, 16, 32}}, "c2": {}})
    assert A.label_frontier(pc) == "C86-L4"


# ----------------------------------------------------- interpretive descriptor (L2)
def test_policy_limited_fixed_not_identifiable():
    for gate in ("C86-A", "C86-B", "C86-C", "C86-D"):
        d = A.interpretive_descriptor(gate)
        assert d["policy_limited"] == "NOT_IDENTIFIABLE_IN_C86H"
        assert d["descriptor"] != "POLICY_LIMITED"   # never inferred from results
    assert A.interpretive_descriptor("C86-E")["descriptor"] is None


def test_classify_two_levels_separate():
    pc = {"c1": _uniform(True, True), "c2": _uniform(True, True)}
    out = A.classify(pc)
    assert out["formal_gate"] == "C86-A"
    assert out["label_frontier"] in K.LABEL_FRONTIER
    assert out["interpretive"]["policy_limited"] == "NOT_IDENTIFIABLE_IN_C86H"
    assert out["pooled_dataset_pvalue"] == "FORBIDDEN"


# ---------------------------------------------------------- label-blind split (12.3)
def test_split_deterministic_and_disjoint():
    trials = [f"t{i}" for i in range(100)]
    pool1, ev1 = E.split_target("Brandl2020", "3", trials)
    pool2, ev2 = E.split_target("Brandl2020", "3", trials)
    assert (pool1, ev1) == (pool2, ev2)              # deterministic
    assert set(pool1).isdisjoint(ev1)
    assert len(pool1) == 50 and len(ev1) == 50


def test_split_subject_and_dataset_bound():
    trials = [f"t{i}" for i in range(100)]
    a, _ = E.split_target("Brandl2020", "3", trials)
    b, _ = E.split_target("Brandl2020", "4", trials)
    c, _ = E.split_target("OpenNeuro_ds007221", "3", trials)
    assert a != b and a != c                         # salt+dataset+subject bound


def test_split_fails_closed_below_minimum():
    with pytest.raises(Exception):
        E.split_target("Brandl2020", "3", [f"t{i}" for i in range(79)])  # < 80 trials


def test_class_support_gate():
    assert E.class_support_ok([0] * 8 + [1] * 8) is True
    assert E.class_support_ok([0] * 8 + [1] * 7) is False   # one class short
    assert E.class_support_ok([0] * 40) is False            # single class


# --------------------------------------------------------------- gated entrypoint
def test_execute_refuses_without_authorization():
    with pytest.raises(SystemExit):
        E.execute("授权 C86D")           # wrong token
    with pytest.raises(SystemExit):
        E.execute("")


def test_execute_refuses_real_run_even_with_token():
    # correct token but the untouched field does not exist -> fail-closed, no real access
    with pytest.raises(RuntimeError):
        E.execute(E.AUTHORIZATION_TOKEN)


def test_preflight_is_outcome_free_and_ready():
    out = E.preflight()
    assert out["authorization_present"] is False
    assert out["bindings"]["ok"] is True
    assert out["field_present"] is False
    assert out["active_maxt_family_size"] == 8
    assert out["ready_for_review"] is True
