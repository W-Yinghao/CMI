"""Guard (V6-A0a): the primary action set is the V5 final admitted Stage-2B set (identity + matched_coral/spdim/t3a) and the
matched_coral IMPLEMENTATION is stable_matched_coral_v1 — the old unsafe pmct path is NEVER used. Synthetic, torch-free."""
from __future__ import annotations
import inspect
from acar.v5 import protocol as P
from acar.v5 import v6_a0_action_viability as AV
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok


def test_primary_action_set_and_stable_impl():
    assert AV.PRIMARY_ACTIONS == P.ACTIONS == ("matched_coral", "spdim", "t3a")
    assert SP.PRIMARY_ACTIONS == P.ACTIONS
    # the intended real provider routes matched_coral -> stable_matched_coral_v1 (NOT pmct)
    rap_src = inspect.getsource(RAP.real_action_provider)
    assert "stable_matched_coral_v1" in rap_src and "matched_coral" in rap_src
    # the V6-A0 modules never import/call the unsafe frozen CORAL path
    for mod in (AV, SP):
        src = inspect.getsource(mod)
        for pat in ("pmct_predict_serialized", "apply_action(", "production_action_provider(", "import torch", "from cmi"):
            assert pat not in src, f"{mod.__name__} references the unsafe/frozen path: {pat!r}"
    ok("primary action set = {matched_coral(stable), spdim, t3a}; matched_coral impl = stable_matched_coral_v1; no pmct")


def main():
    print("ACAR v5 V6-A0a guard: primary action set uses stable matched_coral")
    test_primary_action_set_and_stable_impl()
    print("ALL V6A0-PRIMARY-ACTION-SET GUARDS PASS")


if __name__ == "__main__":
    main()
