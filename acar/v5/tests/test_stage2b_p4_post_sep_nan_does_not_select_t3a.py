"""Guard (Stage-2B1): P4's post_sep selector cannot accidentally pick t3a because of its NaN post_sep (z_post=None). The post_sep
argmax uses strict improvement (v > best + eps), so NaN never wins. Torch-free (synthetic batch). Synthetic."""
from __future__ import annotations
from acar.v5 import scalarization as SCAL
from acar.v5.tests._util import ok, batch


def _b(t3a_post_sep):
    return batch("b", matched_coral={"d_margin": 0.2, "post_sep": 0.5, "JS": 0.3, "flip_rate": 0.1},
                 spdim={"d_margin": 0.1, "post_sep": 0.4, "JS": 0.4, "flip_rate": 0.2},
                 t3a={"d_margin": 0.9, "post_sep": t3a_post_sep, "JS": 0.1, "flip_rate": 0.1})


def test_post_sep_argmax_never_selects_nan_t3a():
    b = _b(float("nan"))
    # the post_sep vote goes to the finite maximum (matched_coral), never the NaN t3a
    assert SCAL._argmax_action(b, "post_sep") == "matched_coral"
    assert SCAL._argmax_action(b, "post_sep") != "t3a"
    # P4's three votes: margin-best=t3a, post_sep-best=matched_coral, js-min=t3a. t3a wins by margin+JS — NOT via post_sep.
    p4 = {"family": "P4", "params": {"k": 2, "veto_q": "q90"}}
    proposed = SCAL.proposed_action(p4, b)
    assert proposed in ("t3a", "matched_coral")                   # sanity: a valid proposal
    # sanity: had t3a's post_sep been the finite maximum it WOULD win the post_sep vote — proving NaN (not value) is the exclusion
    assert SCAL._argmax_action(_b(9.0), "post_sep") == "t3a"
    ok("NaN post_sep for t3a never wins the post_sep argmax (strict-improvement guard); a finite max would")


def main():
    print("ACAR v5 Stage-2B1 guard: P4 post_sep NaN does not select t3a")
    test_post_sep_argmax_never_selects_nan_t3a()
    print("ALL V5 STAGE2B1-P4-POSTSEP-NAN GUARDS PASS")


if __name__ == "__main__":
    main()
