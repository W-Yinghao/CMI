"""Guard (Stage-2B1): the v2-replay comparator supplies ONLY v2_replay_red; it never routes candidates, fits candidate thresholds,
or selects a candidate. It references no scalarization/deploy/threshold API. Synthetic (source inspection)."""
from __future__ import annotations
import inspect
from acar.v5 import stage2_v2_replay as VR
from acar.v5.tests._util import ok

_FORBIDDEN_TOKENS = ("scalarization", "deploy", "fit_quantiles", "stage2_thresholds", "fit_thresholds",
                     ".decide(", "route(", "run_selection", "selection_engine")


def test_v2_replay_touches_no_routing_or_thresholds():
    src = inspect.getsource(VR)
    for tok in _FORBIDDEN_TOKENS:
        assert tok not in src, f"stage2_v2_replay references routing/threshold/selection code: {tok!r}"
    # the module exposes no routing/threshold/select entry points
    for attr in ("decide", "route", "fit_thresholds", "fit_quantiles", "select"):
        assert not hasattr(VR, attr)
    # its public output is a float (v2_replay_red); it does not import scalarization/deploy/thresholds
    assert "scalarization" not in src and "deploy" not in src
    ok("v2-replay references no routing/scalarization/threshold/select code — it only supplies v2_replay_red for G2")


def main():
    print("ACAR v5 Stage-2B1 guard: v2-replay does not enter routing or thresholds")
    test_v2_replay_touches_no_routing_or_thresholds()
    print("ALL V5 STAGE2B1-V2REPLAY-NO-ROUTING GUARDS PASS")


if __name__ == "__main__":
    main()
