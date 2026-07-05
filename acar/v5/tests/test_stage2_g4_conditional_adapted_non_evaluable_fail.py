"""Guard (Stage-2A, PROP 9): the G4 gate the Stage-2 runner will use (ltt.gate_disease) keeps harm_among_adapted CONDITIONAL on
adapted subjects, and treats "no subject adapts" as NON-EVALUABLE ⇒ FAIL. Synthetic (data-free; no labels read from real data)."""
from __future__ import annotations
from acar.v5 import ltt as LTT
from acar.v5 import stage2_selection_runner as RUN2
from acar.v5.tests._util import ok


def _subjects(n, adapted, harmful, n_batches=4):
    return [{"subject": f"s{i}", "batches": [{"adapted": adapted, "harmful": harmful} for _ in range(n_batches)]}
            for i in range(n)]


def test_certifier_is_ltt_gate_disease():
    assert RUN2.GATE_CERTIFIER is LTT.gate_disease
    ok("the Stage-2 runner's gate certifier IS ltt.gate_disease (the conditional-on-adapted G4) (PROP 9)")


def test_no_subject_adapts_is_non_evaluable_fail():
    r = RUN2.GATE_CERTIFIER(_subjects(12, adapted=False, harmful=False))
    assert r["n_adapting"] == 0
    assert r["h2_evaluable"] is False
    assert r["harm_among_adapted_ucb"] is None
    assert r["G4_harm_among_adapted"] is False
    assert r["certification_pass"] is False
    ok("no subject adapts → H2 non-evaluable → G4 FAILS → certification fails (closes the v4 low-coverage degeneracy) (PROP 9)")


def test_high_adapted_harm_fails_g4_even_when_evaluable():
    r = RUN2.GATE_CERTIFIER(_subjects(12, adapted=True, harmful=True))
    assert r["h2_evaluable"] is True                                # subjects DO adapt
    assert r["harm_among_adapted_ucb"] is not None
    assert r["G4_harm_among_adapted"] is False                      # adapted-harm ~1.0 > 0.30 → "adapt rarely but badly" fails
    ok("adapting subjects with high adapted-harm → G4 FAILS (conditional-on-adapted, not diluted) (PROP 9)")


def main():
    print("ACAR v5 Stage-2A guard: G4 conditional-on-adapted, no-adapt=non-evaluable=fail (PROP 9)")
    test_certifier_is_ltt_gate_disease()
    test_no_subject_adapts_is_non_evaluable_fail()
    test_high_adapted_harm_fails_g4_even_when_evaluable()
    print("ALL V5 STAGE2A-G4-NON-EVALUABLE GUARDS PASS")


if __name__ == "__main__":
    main()
