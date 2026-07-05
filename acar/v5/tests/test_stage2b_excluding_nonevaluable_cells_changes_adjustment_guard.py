"""Guard (Stage-2B0b): the fix MATTERS — excluding non-evaluable cells (the old, shrunk family) would UNDER-correct and certify
a candidate the FIXED 132-cell family correctly rejects. This is the selection-bias the Stage-2B0b correction prevents. Synthetic."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_gates as G
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import ok, stage2b_holm_per


def test_shrunk_family_would_over_certify():
    # Only 2 of 22 candidates evaluable (12 evaluable cells, 120 non-evaluable). Raw p chosen so 12·p < 0.05 < 132·p.
    p = 0.001
    per = stage2b_holm_per(evaluable_p=p, nonevaluable_ids=list(P.CANDIDATE_IDS[2:]))
    cid0 = P.CANDIDATE_IDS[0]

    # FIXED family (L=132): the correct, conservative adjustment
    full = ENG.holm_adjusted_map(per, P.CANDIDATE_MANIFEST)
    assert not G.cert_pass_from_adjusted(full[(cid0, "PD", "H1")], full[(cid0, "PD", "H2")], full[(cid0, "PD", "H3")])

    # SHRUNK family (evaluable-only, the OLD behavior): recompute Holm over just the evaluable p-values
    keys, pvals = ENG.holm_family_keys_pvalues(per, P.CANDIDATE_MANIFEST)
    ev = [(k, pv) for k, pv in zip(keys, pvals) if pv < 1.0]
    shrunk_vals = G.holm_adjust([pv for _, pv in ev])
    shrunk = {ev[i][0]: shrunk_vals[i] for i in range(len(ev))}
    assert G.cert_pass_from_adjusted(shrunk[(cid0, "PD", "H1")], shrunk[(cid0, "PD", "H2")], shrunk[(cid0, "PD", "H3")])

    # the FIXED family's adjusted p is strictly larger (more conservative) than the shrunk family's
    assert full[(cid0, "PD", "H1")] > shrunk[(cid0, "PD", "H1")]
    ok("shrinking the Holm family (dropping non-evaluable cells) would over-certify; the fixed 132-cell family correctly rejects")


def main():
    print("ACAR v5 Stage-2B0b guard: excluding non-evaluable cells changes the adjustment (fix matters)")
    test_shrunk_family_would_over_certify()
    print("ALL V5 STAGE2B0B-EXCLUSION-CHANGES-ADJUSTMENT GUARDS PASS")


if __name__ == "__main__":
    main()
