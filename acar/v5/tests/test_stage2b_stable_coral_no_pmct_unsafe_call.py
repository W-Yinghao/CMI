"""Guard (Stage-2B2): the Stage-2 matched_coral path uses stable_matched_coral_v1 and NEVER pmct_predict_serialized /
acar.actions / cmi.eval; spdim & t3a still route through the frozen provider. Torch-free (monkeypatched). Synthetic."""
from __future__ import annotations
import inspect
import numpy as np
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5 import stage2_stable_coral as SC
from acar.v5.tests._util import ok, stage2b_synthetic_source_state


def test_stable_coral_imports_or_calls_no_unsafe_path():
    # check code patterns (imports / calls), NOT docstring mentions of what it deliberately avoids
    src = inspect.getsource(SC)
    for pat in ("import cmi", "from cmi", "import acar.actions", "from acar.actions", "from acar import actions",
                "pmct_predict_serialized(", "apply_action(", "production_action_provider(", "import torch"):
        assert pat not in src, f"stage2_stable_coral imports/calls the unsafe frozen path: {pat!r}"
    ok("stage2_stable_coral imports/calls no pmct / acar.actions / cmi.eval / torch (numpy-only)")


def test_matched_coral_bypasses_frozen_provider_but_spdim_t3a_do_not():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=32, seed=1))
    Z = np.random.RandomState(0).randn(16, 32)
    calls = []
    orig = AR.production_action_provider

    def _rec(name, sl, z):
        calls.append(name)
        n = np.asarray(z).shape[0]
        return np.full((n, 2), 0.5), (None if name == "t3a" else np.asarray(z, float))

    AR.production_action_provider = _rec
    try:
        RAP.real_action_provider("matched_coral", lda, Z)                     # must NOT call the frozen provider
        assert "matched_coral" not in calls
        RAP.real_action_provider("spdim", lda, Z)
        RAP.real_action_provider("t3a", lda, Z)
        assert calls == ["spdim", "t3a"]                                      # spdim/t3a DO route through the frozen provider
    finally:
        AR.production_action_provider = orig
    ok("Stage-2 matched_coral bypasses the frozen pmct provider; spdim/t3a still use it")


def main():
    print("ACAR v5 Stage-2B2 guard: no pmct in the matched_coral path")
    test_stable_coral_imports_or_calls_no_unsafe_path()
    test_matched_coral_bypasses_frozen_provider_but_spdim_t3a_do_not()
    print("ALL V5 STAGE2B2-STABLE-CORAL-NO-PMCT GUARDS PASS")


if __name__ == "__main__":
    main()
