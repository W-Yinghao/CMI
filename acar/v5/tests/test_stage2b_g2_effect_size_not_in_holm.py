"""Guard (Stage-2B0): G2 is the pre-registered effect-size gate (red>0 AND red−v2_replay ≥ 0.02 NLL per disease + macro), a
POINT estimate — NOT a Holm hypothesis. It is computed separately from the H1–H3 certification. Synthetic only."""
from __future__ import annotations
import inspect
from acar.v5 import protocol as P
from acar.v5 import stage2_gates as G
from acar.v5.tests._util import ok


def test_g2_effect_size_semantics():
    assert G.g2_per_disease(red=0.10, v2_replay_red=0.00) is True        # 0.10>0 and margin 0.10 ≥ 0.02
    assert G.g2_per_disease(red=0.10, v2_replay_red=0.09) is False       # margin 0.01 < 0.02
    assert G.g2_per_disease(red=-0.10, v2_replay_red=-0.50) is False     # red ≤ 0 fails even with a big margin
    assert G.g2_macro(0.10, 0.00) is True
    assert G.g2_macro(0.03, 0.02) is False                              # macro margin 0.01 < 0.02
    assert P.UTILITY_EPS == 0.02
    ok("G2 = red>0 AND (red − v2_replay) ≥ 0.02 per disease + macro (point-estimate effect size)")


def test_g2_not_in_holm_family():
    # the Holm certifier's inputs are H1/H2/H3 only; g2 lives in separate functions
    cert_params = [p for p in inspect.signature(G.cert_pass_from_adjusted).parameters if p != "alpha"]
    assert "g2" not in cert_params and "red" not in cert_params
    assert hasattr(G, "g2_per_disease") and hasattr(G, "g2_macro")       # G2 is its own gate
    ok("G2 is excluded from the Holm family (a separate effect-size gate, not a certified hypothesis)")


def main():
    print("ACAR v5 Stage-2B0 guard: G2 effect-size gate is not in the Holm family")
    test_g2_effect_size_semantics()
    test_g2_not_in_holm_family()
    print("ALL V5 STAGE2B0-G2-NOT-IN-HOLM GUARDS PASS")


if __name__ == "__main__":
    main()
