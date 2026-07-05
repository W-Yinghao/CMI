"""Guard (Stage-2B0): Holm certification applies to exactly the H1/H2/H3 hypotheses (per candidate × disease); the step-down
adjuster is correct, and CAL certification requires all three adjusted p ≤ family-wise α. Synthetic only."""
from __future__ import annotations
import inspect
from acar.v5 import protocol as P
from acar.v5 import stage2_gates as G
from acar.v5.tests._util import ok


def test_holm_adjust_correct():
    import numpy as np
    adj = G.holm_adjust([0.01, 0.02, 0.03, 0.04])                 # factors 4,3,2,1 → 0.04,0.06,0.06,0.04 → monotone
    assert np.allclose(adj, [0.04, 0.06, 0.06, 0.06])
    ok("holm_adjust reproduces the step-down FWER adjustment (monotone in rank)")


def test_cert_uses_exactly_h1h2h3():
    # cert_pass_from_adjusted takes exactly the three adjusted p-values (H1,H2,H3) + alpha — G2 is not among them
    params = [p for p in inspect.signature(G.cert_pass_from_adjusted).parameters if p != "alpha"]
    assert params == ["adj_h1", "adj_h2", "adj_h3"]
    assert G.cert_pass_from_adjusted(0.04, 0.04, 0.04) is True
    assert G.cert_pass_from_adjusted(0.04, 0.06, 0.04) is False   # any one > α fails
    assert G.cert_pass_from_adjusted(P.ALPHA, P.ALPHA, P.ALPHA) is True
    ok("certification requires all three Holm-adjusted H1/H2/H3 ≤ α; the family is exactly {H1,H2,H3}")


def main():
    print("ACAR v5 Stage-2B0 guard: Holm family is candidate×disease×{H1,H2,H3} only")
    test_holm_adjust_correct()
    test_cert_uses_exactly_h1h2h3()
    print("ALL V5 STAGE2B0-HOLM-FAMILY GUARDS PASS")


if __name__ == "__main__":
    main()
