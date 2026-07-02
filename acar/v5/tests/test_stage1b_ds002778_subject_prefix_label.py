"""Guard (Stage-1B10): ds002778 (which has NO diagnosis column) derives the label from the subject-id prefix — strip 'sub-',
casefold, prefix 'hc'→control(0), 'pd'→case(1); anything else FAILS closed. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.tests._util import expect_raises, ok


def test_prefix_control_and_case():
    for s in ("sub-hc1", "sub-hc29", "HC5", "hc10"):          # sub- optional; case-insensitive
        assert CLS.resolve_label("PD", "ds002778", s, None) == 0, s
    for s in ("sub-pd3", "sub-pd15", "PD7"):
        assert CLS.resolve_label("PD", "ds002778", s, None) == 1, s
    ok("ds002778 subject-id prefix: hc*→control(0), pd*→case(1) (sub- stripped, case-insensitive)")


def test_unrecognized_prefix_fails():
    for s in ("sub-ctrl1", "sub-x", "sub-", "sub-patient2"):
        expect_raises(CLS.CohortLabelSpecError, lambda s=s: CLS.resolve_label("PD", "ds002778", s, None))
        assert CLS.label_resolvable("PD", "ds002778", s, None) is False
    ok("a subject id with neither hc nor pd prefix → CohortLabelSpecError; label_resolvable → False")


def main():
    print("ACAR v5 Stage-1B10 guard: ds002778 subject-prefix label")
    test_prefix_control_and_case()
    test_unrecognized_prefix_fails()
    print("ALL V5 STAGE1B-DS002778-PREFIX-LABEL GUARDS PASS")


if __name__ == "__main__":
    main()
