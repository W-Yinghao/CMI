"""Guard (Stage-2B1): the real action seam consumes only source_state + the z batch — never a label. Its signatures carry no
label parameter, and apply_action is invoked with exactly (name, adapter_state, z). Torch-free (monkeypatched). Synthetic."""
from __future__ import annotations
import inspect
import numpy as np
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok, stage2b_synthetic_source_state

_LABELS = {"label", "labels", "y", "y_te", "y_true", "y_pred", "diagnosis", "target", "case_control",
           "group", "participant_group", "outcome"}


def test_seam_signatures_label_free():
    for fn in (RAP.real_action_provider, RAP.validated_real_action, AR.production_action_provider,
               AR.subject_action_outputs):
        assert not (set(inspect.signature(fn).parameters) & _LABELS)
    ok("real action-provider signatures expose no label parameter")


def test_apply_action_called_without_label():
    import acar.actions as A
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=5, seed=2))
    Z = np.random.RandomState(1).randn(8, 5)
    seen = []
    orig = A.apply_action

    def _rec(*args, **kwargs):
        seen.append((len(args), tuple(kwargs)))
        n = np.asarray(args[2]).shape[0]
        return np.full((n, 2), 0.5), None if args[0] == "t3a" else np.asarray(args[2], float)

    A.apply_action = _rec
    try:
        for a in P.ACTIONS:
            AR.production_action_provider(a, lda, Z)
    finally:
        A.apply_action = orig
    # apply_action always called positionally as (name, state, z) — 3 args, no keyword (no label kwarg)
    assert all(nargs == 3 and kw == () for nargs, kw in seen)
    ok("apply_action is invoked as (name, adapter_state, z) with no label argument")


def main():
    print("ACAR v5 Stage-2B1 guard: real actions are label-free")
    test_seam_signatures_label_free()
    test_apply_action_called_without_label()
    print("ALL V5 STAGE2B1-REAL-ACTIONS-LABEL-FREE GUARDS PASS")


if __name__ == "__main__":
    main()
