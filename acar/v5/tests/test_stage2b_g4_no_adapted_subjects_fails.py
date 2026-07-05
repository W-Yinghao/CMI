"""Guard (Stage-2B0): G4 (harm_among_adapted) stays conditional on adapted subjects on CAL — if NO subject adapts, H2 is
non-evaluable ⇒ raw p = 1 ⇒ never certifiable ⇒ certification fails. Synthetic only."""
from __future__ import annotations
from acar.v5 import stage2_gates as G
from acar.v5.tests._util import ok


def _records(adapted, harmful, n=20):
    return [{"subject": f"s{i}", "batches": [{"adapted": adapted, "harmful": harmful}]} for i in range(n)]


def test_no_adapted_subject_h2_non_evaluable():
    raw = G.cal_raw_pvalues(_records(adapted=False, harmful=False))
    assert raw["h2_evaluable"] is False
    assert raw["H2"] == 1.0
    assert raw["bounds"]["G4_harm_among_adapted"] is False
    adj = G.holm_adjust([raw["H1"], raw["H2"], raw["H3"]])
    assert not G.cert_pass_from_adjusted(adj[0], adj[1], adj[2])   # H2 raw=1 -> can never certify
    ok("no adapting CAL subject → H2 non-evaluable → raw p=1 → certification fails (conditional-on-adapted preserved)")


def test_adapting_but_harmful_is_evaluable_but_bounded():
    raw = G.cal_raw_pvalues(_records(adapted=True, harmful=True))
    assert raw["h2_evaluable"] is True                            # subjects DO adapt
    assert raw["H2"] == 1.0                                       # all-harmful adapted → UCB≈1 > 0.30 → never certifiable
    assert raw["bounds"]["G4_harm_among_adapted"] is False
    ok("adapting-but-fully-harmful CAL subjects → H2 evaluable but never certifiable (G4 fails)")


def main():
    print("ACAR v5 Stage-2B0 guard: G4 no-adapted-subjects fails")
    test_no_adapted_subject_h2_non_evaluable()
    test_adapting_but_harmful_is_evaluable_but_bounded()
    print("ALL V5 STAGE2B0-G4-NON-EVALUABLE GUARDS PASS")


if __name__ == "__main__":
    main()
