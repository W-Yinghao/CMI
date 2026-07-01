"""Guard: the v4 low-coverage degeneracy is REJECTED — a policy that adapts rarely but harmfully passes G3 (all-batch harm) yet
FAILS G4 (conditional adapted-harm). Synthetic only; harm labels used ONLY in evaluation."""
from __future__ import annotations
from acar.v5 import ltt
from acar.v5 import protocol as P
from acar.v5.tests._util import ok


def _subj(name, n_total, n_adapt, n_harm_adapt):
    b = []
    for i in range(n_total):
        adapted = i < n_adapt
        harmful = adapted and (i < n_harm_adapt)
        b.append({"adapted": adapted, "harmful": harmful})
    return {"subject": name, "batches": b}


def test_low_coverage_high_adapted_harm_fails_G4_not_G3():
    # 200 subjects: 190 never adapt; 10 adapt 2/20 batches and BOTH are harmful (adapted-harm ~1.0).
    recs = [_subj(f"n{i}", 20, 0, 0) for i in range(190)]
    recs += [_subj(f"a{i}", 20, 2, 2) for i in range(10)]
    g = ltt.gate_disease(recs)
    assert g["G3_l_harm_all"] is True, ("all-batch harm is tiny → G3 passes", g["l_harm_all_ucb"])
    assert g["G4_harm_among_adapted"] is False, ("conditional adapted-harm ~1.0 → G4 must fail", g["harm_among_adapted_ucb"])
    assert g["certification_pass"] is False
    ok(f"low-coverage/high-adapted-harm: G3 pass ({g['l_harm_all_ucb']:.3f}≤0.10) but G4 FAIL ({g['harm_among_adapted_ucb']:.3f}>0.30) → rejected")


def test_no_subject_adapts_is_non_evaluable_fail():
    recs = [_subj(f"n{i}", 20, 0, 0) for i in range(50)]
    g = ltt.gate_disease(recs)
    assert g["h2_evaluable"] is False and g["G4_harm_among_adapted"] is False and g["certification_pass"] is False
    ok("no subject adapts → G4 non-evaluable → FAIL (safe-but-useless is not a pass)")


def test_healthy_policy_can_pass_certification():
    # 300 subjects, each adapts ~40% of batches, adapted-harm ~10% → G1/G3/G4 all pass.
    recs = [_subj(f"h{i}", 20, 8, 1) for i in range(300)]
    g = ltt.gate_disease(recs)
    assert g["G1_coverage"] and g["G3_l_harm_all"] and g["G4_harm_among_adapted"] and g["certification_pass"]
    ok(f"healthy policy passes G1({g['coverage_lcb']:.2f})+G3({g['l_harm_all_ucb']:.2f})+G4({g['harm_among_adapted_ucb']:.2f})")


def test_g4_is_load_bearing():
    # sanity: if we (wrongly) ignored G4, the degenerate policy would look fine on G1?/G3. Confirm G3 alone is insufficient.
    recs = [_subj(f"n{i}", 20, 0, 0) for i in range(190)] + [_subj(f"a{i}", 20, 2, 2) for i in range(10)]
    g = ltt.gate_disease(recs)
    assert g["G3_l_harm_all"] and not g["G4_harm_among_adapted"], "G4 is the gate that catches the degeneracy"
    ok("G4 is load-bearing (G3 alone would have passed the degenerate policy)")


def main():
    print("ACAR v5 guard: low-coverage degeneracy (v4 failure mode)")
    test_low_coverage_high_adapted_harm_fails_G4_not_G3()
    test_no_subject_adapts_is_non_evaluable_fail()
    test_healthy_policy_can_pass_certification()
    test_g4_is_load_bearing()
    print("ALL V5 LOW-COVERAGE-DEGENERACY GUARDS PASS")


if __name__ == "__main__":
    main()
