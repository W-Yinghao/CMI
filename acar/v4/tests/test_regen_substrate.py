"""Guards for acar/v4/regen_substrate.py — the all-DEV substrate regeneration SKELETON (NO training). Pure validation +
the pre-registered numeric compatibility-replay pass-line. NO retrain, NO torch/cmi, NO external read. Run:
python -m acar.v4.tests.test_regen_substrate
"""
import copy  # noqa: F401  (kept for fixture deep-copies if needed)
import json
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
    # per-disease v2 sub-gate: red <= v2_replay → fail
    s = _ok_stats(); s["PD"]["v2_replay_red"] = 0.5
    assert not R.compatibility_replay_pass(s)[0]
    # v2 is a HARD requirement (NO waiver): v2 not-evaluable for EITHER disease → FAIL
    s = _ok_stats(); s["PD"]["v2_evaluable"] = False; s["PD"].pop("v2_replay_red")
    ok, why = R.compatibility_replay_pass(s)
    assert not ok and "v2_replay NOT evaluable" in why
    s = _ok_stats()
    for d in ("PD", "SCZ"):
        s[d]["v2_evaluable"] = False; s[d].pop("v2_replay_red")
    assert not R.compatibility_replay_pass(s)[0]
    # PD passes but SCZ fails → overall fail (no per-disease leniency)
    s = _ok_stats(); s["SCZ"]["red"] = 0.0
    assert not R.compatibility_replay_pass(s)[0]
    # missing a disease → fail-closed
    assert not R.compatibility_replay_pass({"PD": _ok_stats()["PD"]})[0]
    # the candidate is fixed (no reselection knobs in the API)
    assert R.FIXED_CANDIDATE == {"score_family": "shift_margin", "policy": "benefit_ranked", "loss": "harm_indicator"}


# ----------------------------------------------------------------------------- frozen command-contract manifests + CLIs

def _regen_manifest(base, disease="PD", cohorts=None, **over):
    env = os.path.join(base, "regen_env_lock.json")
    with open(env, "w") as f:
        f.write("{}")
    cohorts = list(R.DEV_SCOPE[disease]) if cohorts is None else list(cohorts)
    m = {"protocol_commit": "a" * 40, "repo_clean_required": True, "disease": disease, "dev_cohorts": cohorts,
         "source_kind": "canonical_features", "source_paths": {c: base for c in cohorts},
         "subject_list_sha256": "b" * 64, "diagnosis_label_sha256": "b" * 64, "pipeline_config_sha256": "b" * 64,
         "env_lock_path": env, "env_lock_sha256": "b" * 64, "seed": 0}
    m.update(over)
    p = os.path.join(base, "regen.json")
    with open(p, "w") as f:
        json.dump(m, f)
    return p, os.path.join(base, "out")


def test_validate_regen_manifest():
    base = tempfile.mkdtemp()
    try:
        p, _ = _regen_manifest(base)
        with open(p) as f:
            R.validate_regen_manifest(json.load(f))                  # happy
        bad = lambda **o: R.validate_regen_manifest(json.load(open(_regen_manifest(base, **o)[0])))
        _expect(ValueError, lambda: bad(protocol_commit="x"))                                   # bad commit
        _expect(ValueError, lambda: bad(repo_clean_required=False))                             # clean not required
        _expect(ValueError, lambda: bad(cohorts=["ds002778", "ds003490", "ds007526"]))        # external id in scope
        _expect(ValueError, lambda: bad(cohorts=["ds002778", "ds003490"]))                     # wrong scope
        _expect(ValueError, lambda: bad(source_kind="bogus"))                                  # bad source_kind
        _expect(ValueError, lambda: bad(seed=1))                                               # seed != 0
        _expect(ValueError, lambda: bad(subject_list_sha256="short"))                          # bad hash
        _expect(ValueError, lambda: bad(source_paths={"ds002778": "rel/path"}))                # rel + key mismatch
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _sub_manifest(**over):
    sd = {d: {"encoder_checkpoint_path": "/p", "encoder_checkpoint_sha256": "b" * 64,
              "source_state_path": "/s", "source_state_sha256": "b" * 64,
              "encoder_provenance_path": "/ep", "source_state_provenance_path": "/sp"} for d in ("PD", "SCZ")}
    m = {"protocol_commit": "a" * 40, "candidate": dict(R.FIXED_CANDIDATE), "alpha": R.ALPHA, "budget": R.BUDGET,
         "coverage_min": R.COVERAGE_MIN, "substrates": sd,
         "dev_cohorts": {"PD": list(R.DEV_SCOPE["PD"]), "SCZ": list(R.DEV_SCOPE["SCZ"])}, "env_lock_sha256": "b" * 64}
    m.update(over)
    return m


def test_validate_substrate_manifest():
    assert R.validate_substrate_manifest(_sub_manifest()) is not None                          # happy
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(candidate={})))    # reselection
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(alpha=0.2)))       # op-point drift
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(protocol_commit="x")))
    sd = _sub_manifest(); sd["substrates"]["PD"]["encoder_checkpoint_sha256"] = "short"
    _expect(ValueError, lambda: R.validate_substrate_manifest(sd))                             # bad artifact sha
    sd = _sub_manifest(); del sd["substrates"]["SCZ"]
    _expect(ValueError, lambda: R.validate_substrate_manifest(sd))                             # missing disease


def _fake_git(commit="a" * 40, clean=True):
    import types

    def g(root, *a):
        if a[:1] == ("rev-parse",):
            out = commit
        elif a[:1] == ("status",):
            out = "" if clean else " M acar/v4/x.py"
        else:
            out = ""
        return types.SimpleNamespace(returncode=0, stdout=out + "\n", stderr="")
    return g


def test_run_regen_substrate_fail_closed():
    from acar.v4 import run_regen_substrate as RRS
    base = tempfile.mkdtemp(); saved = RRS._git
    try:
        RRS._git = _fake_git()
        p, out = _regen_manifest(base)
        _expect(R.SubstrateTrainingNotAuthorizedError, lambda: RRS.run(p, out))                # preflight passes → gated
        assert not os.path.exists(out)                                                          # nothing written
        # manifest defects raise BEFORE the gate (and before git, where applicable)
        pe, oe = _regen_manifest(base, cohorts=["ds002778", "ds003490", "ds007526"])
        _expect(ValueError, lambda: RRS.run(pe, oe))                                            # external id
        ph, oh = _regen_manifest(base, protocol_commit="c" * 40)
        _expect(ValueError, lambda: RRS.run(ph, oh))                                            # HEAD mismatch
        RRS._git = _fake_git(clean=False)
        pd_, od_ = _regen_manifest(base)
        _expect(ValueError, lambda: RRS.run(pd_, od_))                                          # dirty worktree
        RRS._git = _fake_git()
        pen, oen = _regen_manifest(base, env_lock_path=os.path.join(base, "nope.json"))
        _expect(ValueError, lambda: RRS.run(pen, oen))                                          # env lock missing
        px, ox = _regen_manifest(base, disease="SCZ"); os.makedirs(ox + "_x")
        _expect(FileExistsError, lambda: RRS.run(px, ox + "_x"))                                # output exists
    finally:
        RRS._git = saved; shutil.rmtree(base, ignore_errors=True)


def test_run_substrate_compatibility_fail_closed():
    from acar.v4 import run_regen_substrate as RRS
    from acar.v4 import run_substrate_compatibility as RSC
    base = tempfile.mkdtemp(); saved = RRS._git
    try:
        RRS._git = _fake_git()
        p = os.path.join(base, "sub.json"); out = os.path.join(base, "compat_out")
        with open(p, "w") as f:
            json.dump(_sub_manifest(), f)
        _expect(R.SubstrateCompatibilityNotAuthorizedError, lambda: RSC.run(p, out))            # preflight passes → gated
        assert not os.path.exists(out)
        pb = os.path.join(base, "sub_bad.json")
        with open(pb, "w") as f:
            json.dump(_sub_manifest(candidate={"x": 1}), f)
        _expect(ValueError, lambda: RSC.run(pb, out))                                           # reselection → before gate
    finally:
        RRS._git = saved; shutil.rmtree(base, ignore_errors=True)


def main():
    print("ACAR v4 regen_substrate guards (skeleton + B1-preflight command contract; NO training):")
    for t in (test_validate_substrate_request, test_train_not_authorized, test_compatibility_replay_pass,
              test_validate_regen_manifest, test_validate_substrate_manifest, test_run_regen_substrate_fail_closed,
              test_run_substrate_compatibility_fail_closed):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 REGEN-SUBSTRATE GUARDS PASS")


if __name__ == "__main__":
    main()
