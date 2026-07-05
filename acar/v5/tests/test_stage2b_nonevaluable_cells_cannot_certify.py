"""Guard (Stage-2B0b): a non-evaluable candidate×disease cell can never be certified — its Holm-adjusted H1/H2/H3 are 1.0 and
cert_pass is False. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_gates as G
from acar.v5 import stage2_selection_engine as ENG
from acar.v5.tests._util import ok, stage2b_holm_per


def test_nonevaluable_cells_cannot_certify():
    per = stage2b_holm_per(evaluable_p=0.0001, nonevaluable_ids=["V5-P5-001"])
    adj = ENG.holm_adjusted_map(per, P.CANDIDATE_MANIFEST)
    for d in ENG.DISEASES:
        assert adj[("V5-P5-001", d, "H1")] == 1.0
        assert adj[("V5-P5-001", d, "H2")] == 1.0
        assert adj[("V5-P5-001", d, "H3")] == 1.0
        assert not G.cert_pass_from_adjusted(adj[("V5-P5-001", d, "H1")], adj[("V5-P5-001", d, "H2")],
                                             adj[("V5-P5-001", d, "H3")])
    ok("a non-evaluable candidate×disease cell has Holm-adjusted p = 1.0 for H1/H2/H3 → cannot certify")


def main():
    print("ACAR v5 Stage-2B0b guard: non-evaluable cells cannot certify")
    test_nonevaluable_cells_cannot_certify()
    print("ALL V5 STAGE2B0B-NONEVALUABLE-CANNOT-CERTIFY GUARDS PASS")


if __name__ == "__main__":
    main()
