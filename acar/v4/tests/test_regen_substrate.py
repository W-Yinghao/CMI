"""Guards for acar/v4/regen_substrate.py — the all-DEV substrate regeneration SKELETON (NO training). Pure validation +
the pre-registered numeric compatibility-replay pass-line. NO retrain, NO torch/cmi, NO external read. Run:
python -m acar.v4.tests.test_regen_substrate
"""
import copy
import os
import shutil
import tempfile

from acar.v4 import regen_substrate as R


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                          # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def _env(base):
    p = os.path.join(base, "env_lock.json")
    with open(p, "w") as f:
        f.write("{}")
    return p


def test_validate_substrate_request():
    base = tempfile.mkdtemp()
    try:
        env = _env(base); out = os.path.join(base, "sub_out")                   # absent
        rep = R.validate_substrate_request("PD", list(R.DEV_SCOPE["PD"]), out, env_lock_path=env)
        assert rep["authorized_to_train"] is False and rep["disease"] == "PD"
        assert "encoder_checkpoint_path" in rep["expected_artifacts"]
        assert rep["substrate_kind"].startswith("NEW all-DEV") and "NOT a recovered original" in rep["substrate_kind"]
        R.validate_substrate_request("SCZ", list(R.DEV_SCOPE["SCZ"]), out, env_lock_path=env)   # SCZ scope ok
        # wrong cohort set (missing one)
        _expect(ValueError, lambda: R.validate_substrate_request("PD", ["ds002778", "ds003490"], out, env_lock_path=env))
        # external/rejected cohort in scope
        _expect(ValueError, lambda: R.validate_substrate_request("PD", ["ds002778", "ds003490", "ds007526"], out,
                                                                 env_lock_path=env))
        # pipeline mismatch
        _expect(ValueError, lambda: R.validate_substrate_request("PD", list(R.DEV_SCOPE["PD"]), out, env_lock_path=env,
                                                                 pipeline_config={**R.FROZEN_PIPELINE, "resample_fs": 250}))
        # seed != 0
        _expect(ValueError, lambda: R.validate_substrate_request("PD", list(R.DEV_SCOPE["PD"]), out, seed=1,
                                                                 env_lock_path=env))
        # env lock missing
        _expect(ValueError, lambda: R.validate_substrate_request("PD", list(R.DEV_SCOPE["PD"]), out,
                                                                 env_lock_path=os.path.join(base, "nope.json")))
        # bad disease
        _expect(ValueError, lambda: R.validate_substrate_request("ZZ", [], out, env_lock_path=env))
        # output dir exists
        os.makedirs(out)
        _expect(ValueError, lambda: R.validate_substrate_request("PD", list(R.DEV_SCOPE["PD"]), out, env_lock_path=env))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_train_not_authorized():
    base = tempfile.mkdtemp()
    try:
        env = _env(base); out = os.path.join(base, "o")
        rep = R.train_all_dev_substrate("PD", list(R.DEV_SCOPE["PD"]), out, env_lock_path=env, dry_run=True)
        assert rep["authorized_to_train"] is False                              # dry-run validates, does not train
        _expect(R.SubstrateTrainingNotAuthorizedError,
                lambda: R.train_all_dev_substrate("PD", list(R.DEV_SCOPE["PD"]), out, env_lock_path=env, dry_run=False))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _ok_stats():
    return {"PD": dict(lambda_certified=True, coverage=0.5, red=0.2, L_harm_all_eval=0.03, v2_evaluable=True,
                       v2_replay_red=0.1),
            "SCZ": dict(lambda_certified=True, coverage=0.5, red=0.3, L_harm_all_eval=0.02, v2_evaluable=True,
                        v2_replay_red=0.1)}


def test_compatibility_replay_pass():
    auth, _ = R.compatibility_replay_pass(_ok_stats())
    assert auth
    # each absolute gate, alone, flips to NOT AUTHORIZED
    for d, key, val in (("PD", "red", 0.0), ("SCZ", "coverage", 0.10), ("PD", "L_harm_all_eval", 0.2),
                        ("SCZ", "lambda_certified", False)):
        s = _ok_stats(); s[d][key] = val
        assert not R.compatibility_replay_pass(s)[0], (d, key)
    # per-disease v2 sub-gate: red <= v2_replay (still evaluable) → fail
    s = _ok_stats(); s["PD"]["v2_replay_red"] = 0.5
    assert not R.compatibility_replay_pass(s)[0]
    # v2 not-evaluable for one disease → that v2 sub-gate WAIVED; absolute gates still pass → authorized
    s = _ok_stats(); s["PD"]["v2_evaluable"] = False; s["PD"].pop("v2_replay_red")
    assert R.compatibility_replay_pass(s)[0]
    # neither v2 evaluable → macro v2 WAIVED; absolute pass → authorized
    s = _ok_stats()
    for d in ("PD", "SCZ"):
        s[d]["v2_evaluable"] = False; s[d].pop("v2_replay_red")
    assert R.compatibility_replay_pass(s)[0]
    # missing a disease → fail-closed
    assert not R.compatibility_replay_pass({"PD": _ok_stats()["PD"]})[0]
    # the candidate is fixed (no reselection knobs in the API)
    assert R.FIXED_CANDIDATE == {"score_family": "shift_margin", "policy": "benefit_ranked", "loss": "harm_indicator"}


def main():
    print("ACAR v4 regen_substrate guards (skeleton; NO training):")
    for t in (test_validate_substrate_request, test_train_not_authorized, test_compatibility_replay_pass):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 REGEN-SUBSTRATE GUARDS PASS")


if __name__ == "__main__":
    main()
