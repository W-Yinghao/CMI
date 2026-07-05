"""Guard (Stage-2B1): even with the REAL v2-replay comparator wired into the engine, the runner still fails closed without a valid
Stage-2B authorization bound to the admitted package — no global enable flag, no candidate produced without auth. Synthetic
(torch-free synthetic action provider). Synthetic."""
from __future__ import annotations
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2b_authorization as AUTH
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_v2_replay as VR
from acar.v5.tests._util import expect_raises, ok, stage2b_auth, stage2b_disease_inputs

SYN = AR.synthetic_action_provider


def _run(auth, *, registry_sha):
    prov = VR.make_engine_v2_replay_provider(action_provider=SYN)          # the REAL v2-replay comparator, wired in
    return ENG.run_selection(auth, stage1b_run_id=auth["stage1b_run_id"], stage1b_registry_sha256=registry_sha,
                             disease_inputs=stage2b_disease_inputs(seed=4, n_windows=40),
                             action_provider=SYN, v2_replay_provider=prov)


def test_real_seams_do_not_bypass_the_gate():
    a = stage2b_auth()
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: _run(stage2b_auth(statement="nope"),
                                                               registry_sha=a["stage1b_registry_sha256"]))
    expect_raises(AUTH.Stage2bAuthorizationError, lambda: _run(a, registry_sha="b" * 64))   # package-binding mismatch
    ok("with the real v2-replay wired, an invalid/unbound Stage-2B auth still fails closed")


def test_valid_auth_runs_with_real_v2_replay():
    a = stage2b_auth()
    rep = _run(a, registry_sha=a["stage1b_registry_sha256"])
    assert rep["outcome"] in ("SELECTED", "DEV_STOP")
    assert rep["notes"]["holm_family_size"] == 132                        # Stage-2B0b family still fixed
    ok("a valid bound auth runs the engine with the real v2-replay comparator → a valid report")


def main():
    print("ACAR v5 Stage-2B1 guard: real runner still fails without a valid Stage-2B auth")
    test_real_seams_do_not_bypass_the_gate()
    test_valid_auth_runs_with_real_v2_replay()
    print("ALL V5 STAGE2B1-REAL-RUNNER-FAILS-WITHOUT-AUTH GUARDS PASS")


if __name__ == "__main__":
    main()
