"""Guards for acar/v4/regen_substrate.py — the all-DEV substrate regeneration SKELETON (NO training). Pure validation +
the pre-registered numeric compatibility-replay pass-line. NO retrain, NO torch/cmi, NO external read. Run:
python -m acar.v4.tests.test_regen_substrate
"""
import copy  # noqa: F401  (kept for fixture deep-copies if needed)
import hashlib
import json
import os
import shutil
import sys
import tempfile

from acar.v4 import regen_substrate as R
from acar.v4 import regen_envlock as EL


def _fsha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _captured_lock(commit, pipeline_sha, device_kind="cuda"):
    lk = EL.schema_only_template(protocol_commit=commit, pipeline_config_sha256=pipeline_sha)
    lk.update(status="CAPTURED_AND_VERIFIED", python_version="3.13.14", torch_version="2.6.0+cu124",
              torchvision_version="0.21.0+cu124", torchaudio_version="2.6.0+cu124", moabb_version="1.5.0",
              mne_version="1.12.1", skorch_version="1.4.0", braindecode_version="1.5.2", numpy_version="2.4.4",
              scipy_version="1.18.0", sklearn_version="1.9.0", device_kind=device_kind,
              device_name=("NVIDIA A100" if device_kind == "cuda" else "cpu"))
    if device_kind == "cuda":                                    # CAPTURED cuda lock requires the cuda fields non-empty
        lk.update(cuda_version="12.4", cudnn_version="90100", driver_version="550.54.15")
    return lk


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
        # seed STRICT int 0 (reject bool / "0" / 0.0 / 0.9)
        for s in (1, True, False, "0", 0.0, 0.9):
            _expect(ValueError, lambda s=s: R.validate_substrate_request("PD", list(R.DEV_SCOPE["PD"]), out, seed=s,
                                                                         env_lock_path=env))
        # pipeline EXACT match (extra key rejected)
        _expect(ValueError, lambda: R.validate_substrate_request("PD", list(R.DEV_SCOPE["PD"]), out, env_lock_path=env,
                                                                 pipeline_config={**R.FROZEN_PIPELINE, "extra": 1}))
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

def _regen_manifest(base, disease="PD", cohorts=None, lock_status="CAPTURED_AND_VERIFIED", break_env_sha=False,
                    lock_device_kind="cuda", **over):
    cohorts = list(R.DEV_SCOPE[disease]) if cohorts is None else list(cohorts)
    pcfg = R.canonical_pipeline_config_sha256()
    commit = over.get("protocol_commit", "a" * 40)
    if lock_status == "CAPTURED_AND_VERIFIED":
        lk = _captured_lock(commit, pcfg, device_kind=lock_device_kind)
    else:
        lk = EL.schema_only_template(protocol_commit=commit, pipeline_config_sha256=pcfg)
        if lock_status == "CAPTURE_FAILED":
            lk["status"] = "CAPTURE_FAILED"; lk["capture_note"] = "probe failed (synthetic)"
    env = os.path.join(base, f"regen_env_{disease}.json")
    with open(env, "w") as f:
        json.dump(lk, f)
    env_sha = "0" * 64 if break_env_sha else _fsha(env)
    m = {"protocol_commit": commit, "repo_clean_required": True, "disease": disease, "dev_cohorts": cohorts,
         "source_kind": "raw_bids", "source_paths": {c: base for c in cohorts},
         "source_file_manifest_sha256": "b" * 64,
         "per_cohort_source_file_manifest_sha256": {c: "b" * 64 for c in cohorts},
         "eligible_subject_list_sha256": "b" * 64,
         "per_cohort_eligible_subject_list_sha256": {c: "b" * 64 for c in cohorts},
         "n_eligible_subjects": R.EXACT_ELIGIBLE[disease], "excluded_subjects": {},
         "subject_list_sha256": "b" * 64, "diagnosis_label_sha256": "b" * 64, "pipeline_config_sha256": pcfg,
         "env_lock_path": env, "env_lock_sha256": env_sha, "seed": 0}
    m.update(over)
    p = os.path.join(base, f"regen_{disease}.json")
    with open(p, "w") as f:
        json.dump(m, f)
    return p, os.path.join(base, f"out_{disease}")


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
        _expect(ValueError, lambda: bad(source_kind="canonical_features"))                      # B1b training = raw_bids only
        for s in (1, True, "0", 0.0):
            _expect(ValueError, lambda s=s: bad(seed=s))                                       # seed STRICT int 0
        _expect(ValueError, lambda: bad(subject_list_sha256="short"))                          # bad hash
        _expect(ValueError, lambda: bad(pipeline_config_sha256="b" * 64))                      # != canonical FROZEN hash
        _expect(ValueError, lambda: bad(source_file_manifest_sha256="short"))                  # raw-list provenance hash
        _expect(ValueError, lambda: bad(per_cohort_source_file_manifest_sha256={"ds002778": "b" * 64}))  # key mismatch
        _expect(ValueError, lambda: bad(eligible_subject_list_sha256="short"))                 # eligible hash
        _expect(ValueError, lambda: bad(n_eligible_subjects=999))                              # != EXACT_ELIGIBLE[PD]=230
        _expect(ValueError, lambda: bad(n_eligible_subjects=True))                             # strict int
        _expect(ValueError, lambda: bad(excluded_subjects={"noslash": "r"}))                   # bad excluded key
        _expect(ValueError, lambda: bad(excluded_subjects={"ds002778/sub-1": ""}))             # empty reason
        _expect(ValueError, lambda: bad(per_cohort_eligible_subject_list_sha256={"ds002778": "b" * 64}))  # key mismatch
        _expect(ValueError, lambda: bad(source_paths={"ds002778": "rel/path"}))                # rel + key mismatch
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _sub_manifest(**over):
    sd = {d: {"encoder_checkpoint_path": "/p", "encoder_state_dict_sha256": "b" * 64,
              "encoder_checkpoint_file_sha256": "b" * 64, "source_state_path": "/s",
              "source_state_artifact_sha256": "b" * 64, "source_state_file_sha256": "b" * 64,
              "encoder_provenance_path": "/ep", "source_state_provenance_path": "/sp",
              "dev_input_manifest_path": "/dim", "dev_input_manifest_sha256": "b" * 64} for d in ("PD", "SCZ")}
    m = {"substrate_protocol_commit": "f" * 40, "compatibility_protocol_commit": "a" * 40,   # two-commit split (C1)
         "candidate": dict(R.FIXED_CANDIDATE), "alpha": R.ALPHA, "budget": R.BUDGET,
         "coverage_min": R.COVERAGE_MIN, "substrates": sd, "env_lock_path": "/abs/env_lock.json",
         "dev_cohorts": {"PD": list(R.DEV_SCOPE["PD"]), "SCZ": list(R.DEV_SCOPE["SCZ"])}, "env_lock_sha256": "b" * 64}
    m.update(over)
    return m


def test_validate_substrate_manifest():
    assert R.validate_substrate_manifest(_sub_manifest()) is not None                          # happy
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(candidate={})))    # reselection
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(alpha=0.2)))       # op-point drift
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(substrate_protocol_commit="x")))    # bad commit
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(compatibility_protocol_commit="x")))
    _expect(ValueError, lambda: R.validate_substrate_manifest(_sub_manifest(protocol_commit="a" * 40)))  # retired single field
    m = _sub_manifest(); del m["env_lock_path"]
    _expect(ValueError, lambda: R.validate_substrate_manifest(m))                              # env_lock_path required
    sd = _sub_manifest(); sd["substrates"]["PD"]["encoder_checkpoint_file_sha256"] = "short"
    _expect(ValueError, lambda: R.validate_substrate_manifest(sd))                             # bad artifact sha
    sd = _sub_manifest(); sd["substrates"]["PD"]["dev_input_manifest_sha256"] = "short"
    _expect(ValueError, lambda: R.validate_substrate_manifest(sd))                             # bad dev-input sha
    sd = _sub_manifest(); del sd["substrates"]["PD"]["dev_input_manifest_path"]
    _expect(ValueError, lambda: R.validate_substrate_manifest(sd))                             # dev-input path required
    sd = _sub_manifest(); del sd["substrates"]["SCZ"]
    _expect(ValueError, lambda: R.validate_substrate_manifest(sd))                             # missing disease
    for hf in ("encoder_state_dict_sha256", "encoder_checkpoint_file_sha256",                  # each of the 4 unambiguous
               "source_state_artifact_sha256", "source_state_file_sha256"):                    # hash fields is required
        sd = _sub_manifest(); del sd["substrates"]["PD"][hf]
        _expect(ValueError, lambda sd=sd: R.validate_substrate_manifest(sd))
    for legacy in ("encoder_checkpoint_sha256", "source_state_sha256"):                        # retired ambiguous names rejected
        sd = _sub_manifest(); sd["substrates"]["PD"][legacy] = "b" * 64
        _expect(ValueError, lambda sd=sd: R.validate_substrate_manifest(sd))


def _compat_auth(**over):
    a = {"compatibility_protocol_commit": "a" * 40, "substrate_protocol_commit": "f" * 40,
         "substrate_manifest_sha256": "b" * 64, "env_lock_sha256": "c" * 64, "output_path": "/out",
         "authorized_by": "yinghao", "authorization_time": "2026-06-30T12:00:00Z", "statement": R.REQUIRED_COMPAT_STATEMENT}
    a.update(over)
    return a


def test_validate_compat_authorization():
    assert R.validate_compat_authorization(_compat_auth()) is not None
    _expect(ValueError, lambda: R.validate_compat_authorization(_compat_auth(compatibility_protocol_commit="x")))
    _expect(ValueError, lambda: R.validate_compat_authorization(_compat_auth(substrate_protocol_commit="x")))
    _expect(ValueError, lambda: R.validate_compat_authorization(_compat_auth(substrate_manifest_sha256="short")))
    _expect(ValueError, lambda: R.validate_compat_authorization(_compat_auth(statement="ok go")))   # wrong statement
    bad = _compat_auth(); bad["extra"] = 1
    _expect(ValueError, lambda: R.validate_compat_authorization(bad))                          # extra field
    bad = _compat_auth(); del bad["authorized_by"]
    _expect(ValueError, lambda: R.validate_compat_authorization(bad))                          # missing field


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
    base = tempfile.mkdtemp(); saved = RRS._git; saved_ve = RRS._verify_eligible_subjects
    try:
        RRS._git = _fake_git()
        RRS._verify_eligible_subjects = lambda spec: None                                       # tmp source_paths have no sub-*
        p, out = _regen_manifest(base)
        _expect(R.SubstrateTrainingNotAuthorizedError, lambda: RRS.run(p, out))                # full preflight passes → gated (no auth)
        assert not os.path.exists(out)                                                          # nothing written
        assert "torch" not in sys.modules and "cmi" not in sys.modules                         # no heavy import on this path
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
        _expect(ValueError, lambda: RRS.run(pen, oen))                                          # env lock file missing
        pbs, obs = _regen_manifest(base, break_env_sha=True)
        _expect(ValueError, lambda: RRS.run(pbs, obs))                                          # env_lock_sha256 mismatch
        pso, oso = _regen_manifest(base, lock_status="SCHEMA_ONLY_NOT_CAPTURED")
        _expect(ValueError, lambda: RRS.run(pso, oso))                                          # SCHEMA_ONLY rejected
        pcf, ocf = _regen_manifest(base, lock_status="CAPTURE_FAILED")
        _expect(ValueError, lambda: RRS.run(pcf, ocf))                                          # CAPTURE_FAILED rejected
        pcpu, ocpu = _regen_manifest(base, lock_device_kind="cpu")
        _expect(ValueError, lambda: RRS.run(pcpu, ocpu))                                        # CPU env lock rejected at preflight
        px, ox = _regen_manifest(base, disease="SCZ"); os.makedirs(ox + "_x")
        _expect(FileExistsError, lambda: RRS.run(px, ox + "_x"))                                # output exists
        assert "torch" not in sys.modules                                                       # still no torch
    finally:
        RRS._git = saved; RRS._verify_eligible_subjects = saved_ve; shutil.rmtree(base, ignore_errors=True)


def _b1_auth(commit, dim_sha, env_sha, output, **over):
    a = {"protocol_commit": commit, "disease": "PD", "dev_input_manifest_sha256": dim_sha, "env_lock_sha256": env_sha,
         "output_path": output, "authorized_by": "yinghao", "authorization_time": "2026-06-30T12:00:00Z",
         "statement": R.REQUIRED_AUTH_STATEMENT}
    a.update(over)
    return a


def test_validate_b1_authorization():
    ok = _b1_auth("a" * 40, "b" * 64, "c" * 64, "/out")
    assert R.validate_b1_authorization(ok) is ok
    _expect(ValueError, lambda: R.validate_b1_authorization(_b1_auth("x", "b" * 64, "c" * 64, "/out")))   # bad commit
    _expect(ValueError, lambda: R.validate_b1_authorization(_b1_auth("a" * 40, "b" * 64, "c" * 64, "/out",
                                                                     statement="ok go")))                 # wrong statement
    _expect(ValueError, lambda: R.validate_b1_authorization(_b1_auth("a" * 40, "b" * 64, "c" * 64, "/out", disease="ZZ")))
    bad = _b1_auth("a" * 40, "b" * 64, "c" * 64, "/out"); bad["extra"] = 1
    _expect(ValueError, lambda: R.validate_b1_authorization(bad))                                          # extra field
    bad = _b1_auth("a" * 40, "b" * 64, "c" * 64, "/out"); del bad["authorized_by"]
    _expect(ValueError, lambda: R.validate_b1_authorization(bad))                                          # missing field


def test_check_eligible_subjects():
    dis = "SCZ"; cohorts = list(R.DEV_SCOPE[dis]); c0 = cohorts[0]; n = R.EXACT_ELIGIBLE[dis]   # 225
    elig = [f"s{i:04d}" for i in range(n)]

    def mani(excluded, raw_c0):
        raw = {c: [] for c in cohorts}; raw[c0] = list(raw_c0)
        return raw, {"dev_cohorts": cohorts, "excluded_subjects": dict(excluded),
                     "eligible_subject_list_sha256": R.canonical_subject_list_sha256([f"{c0}/{s}" for s in elig]),
                     "per_cohort_eligible_subject_list_sha256": {
                         c: R.canonical_subject_list_sha256(sorted({f"{c}/{s}" for s in raw[c]} - set(excluded)))
                         for c in cohorts}}
    raw, m = mani({f"{c0}/x0": "qc drop"}, elig + ["x0"])                                       # 226 raw = 225 elig + 1 excl
    assert len(R.check_eligible_subjects(dis, raw, m)) == n                                     # happy
    raw, m = mani({}, elig + ["extra1"])
    _expect(ValueError, lambda: R.check_eligible_subjects(dis, raw, m))                         # extra raw not excluded
    raw, m = mani({}, elig[:-1])
    _expect(ValueError, lambda: R.check_eligible_subjects(dis, raw, m))                         # missing eligible
    raw, m = mani({f"{c0}/ghost": "x"}, elig)
    _expect(ValueError, lambda: R.check_eligible_subjects(dis, raw, m))                         # excluded not in raw
    raw, m = mani({}, elig[:-1] + ["sZZZZ"])
    _expect(ValueError, lambda: R.check_eligible_subjects(dis, raw, m))                         # right count, wrong member


def _good_runtime():
    return {"device_kind": "cuda", "torch_intraop_threads": 1, "torch_interop_threads": 1, "omp_num_threads": 1,
            "python_version": "3.11.0", "torch_version": "2.8.0+cu128", "torchvision_version": "0.23.0",
            "torchaudio_version": "2.8.0", "braindecode_version": "1.1.0", "moabb_version": "1.2.0",
            "mne_version": "1.12.1", "skorch_version": "1.2.0", "numpy_version": "2.1.0", "scipy_version": "1.14.0",
            "sklearn_version": "1.5.0", "cuda_version": "12.8", "cudnn_version": "90100", "driver_version": "550.54.15"}


def test_require_cuda():
    assert R.require_cuda({"device": "cuda"}, True) == "cuda"
    _expect(RuntimeError, lambda: R.require_cuda({"device": "cuda"}, False))                    # no silent CPU fallback
    _expect(ValueError, lambda: R.require_cuda({"device": "cpu"}, True))                        # schedule must pin cuda


def test_check_runtime_matches_lock():
    rt = _good_runtime(); lock = dict(rt)
    assert R.check_runtime_matches_lock(lock, rt) is True                                       # exact match
    _expect(RuntimeError, lambda: R.check_runtime_matches_lock(lock, {**rt, "device_kind": "cpu"}))   # runtime not cuda
    _expect(ValueError, lambda: R.check_runtime_matches_lock({**lock, "device_kind": "cpu"}, rt))     # lock not cuda
    _expect(ValueError, lambda: R.check_runtime_matches_lock(lock, {**rt, "torch_intraop_threads": 4}))  # threads != 1
    _expect(ValueError, lambda: R.check_runtime_matches_lock({**lock, "omp_num_threads": 8}, rt))     # lock threads != 1
    _expect(ValueError, lambda: R.check_runtime_matches_lock(lock, {**rt, "torch_version": "2.7.0"}))  # version drift
    _expect(ValueError, lambda: R.check_runtime_matches_lock(lock, {**rt, "moabb_version": "9.9"}))    # any version drift
    _expect(ValueError, lambda: R.check_runtime_matches_lock(lock, {**rt, "cuda_version": "12.1"}))    # cuda toolkit drift
    _expect(ValueError, lambda: R.check_runtime_matches_lock(lock, {**rt, "cudnn_version": "80000"}))  # cudnn drift
    _expect(ValueError, lambda: R.check_runtime_matches_lock(lock, {**rt, "driver_version": "9.9"}))   # driver drift
    _expect(ValueError, lambda: R.check_runtime_matches_lock(lock, {**rt, "skorch_version": ""}))      # empty != vacuous match
    _expect(ValueError, lambda: R.check_runtime_matches_lock({**lock, "mne_version": ""}, rt))         # lock empty rejected


def test_assert_finite():
    import numpy as np
    assert R.assert_finite([1.0, 2.0, 3.0], "x") is True
    _expect(ValueError, lambda: R.assert_finite([1.0, np.nan], "x"))
    _expect(ValueError, lambda: R.assert_finite([1.0, np.inf], "x"))
    _expect(ValueError, lambda: R.assert_finite([], "x"))                                       # empty -> fail


def test_single_subject_label():
    assert R.single_subject_label([1, 1, 1], "c/s") == 1
    assert R.single_subject_label([0, 0], "c/s") == 0
    _expect(ValueError, lambda: R.single_subject_label([], "c/s"))                              # 0 windows
    _expect(ValueError, lambda: R.single_subject_label([0, 1], "c/s"))                          # mixed within subject
    _expect(ValueError, lambda: R.single_subject_label([2], "c/s"))                             # label outside {0,1}


def test_check_training_set():
    assert R.check_training_set([0, 0, 1, 1], ["A", "A", "B", "B"], {"A", "B"}) is True
    _expect(ValueError, lambda: R.check_training_set([0, 1], ["A", "B"], {"A", "B", "C"}))       # eligible C has no windows
    _expect(ValueError, lambda: R.check_training_set([0, 0], ["A", "A"], {"A"}))                 # single class
    _expect(ValueError, lambda: R.check_training_set([0, 2], ["A", "B"], {"A", "B"}))            # label outside {0,1}
    _expect(ValueError, lambda: R.check_training_set([0, 1], ["A", "Z"], {"A"}))                 # window from non-eligible


def test_canonical_state_dict_sha256():
    import numpy as np
    sd = {"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "b": np.array([0.5], dtype=np.float32)}
    h0 = R.canonical_state_dict_sha256(sd)
    assert h0 == R.canonical_state_dict_sha256({"b": sd["b"], "w": sd["w"]})                    # order-independent
    h_val = R.canonical_state_dict_sha256({"w": sd["w"] * 2, "b": sd["b"]})
    h_dtype = R.canonical_state_dict_sha256({"w": sd["w"].astype(np.float64), "b": sd["b"]})
    h_shape = R.canonical_state_dict_sha256({"w": sd["w"].reshape(1, 4), "b": sd["b"]})
    h_name = R.canonical_state_dict_sha256({"W": sd["w"], "b": sd["b"]})
    assert len({h0, h_val, h_dtype, h_shape, h_name}) == 5                                      # sensitive to value/dtype/shape/name
    be = {"w": sd["w"].astype(">f4"), "b": sd["b"].astype(">f4")}                               # big-endian, same values
    assert R.canonical_state_dict_sha256(be) == h0                                              # serialization/endianness-independent
    _expect(ValueError, lambda: R.canonical_state_dict_sha256({}))                              # empty -> fail


def test_run_regen_substrate_authorized_runs_gated_trainer():
    from acar.v4 import run_regen_substrate as RRS
    base = tempfile.mkdtemp()
    saved = (RRS._git, RRS._verify_eligible_subjects, RRS._train_substrate, RRS._verify_runtime_matches_lock)
    calls = []

    def fake_train(spec, output):
        calls.append(output)
        enc = os.path.join(output, "enc.pt"); ss = os.path.join(output, "ss.npz")
        with open(enc, "wb") as f: f.write(b"E")
        with open(ss, "wb") as f: f.write(b"S")
        return {"encoder_checkpoint_path": enc, "source_state_path": ss,                        # trainer reports BOTH:
                "encoder_state_dict_sha256": "c" * 64, "source_state_artifact_sha256": "d" * 64}  # canonical (semantic) shas
    try:
        RRS._git = _fake_git(); RRS._verify_eligible_subjects = lambda spec: None
        RRS._train_substrate = fake_train
        RRS._verify_runtime_matches_lock = lambda spec: None                                    # runtime==lock checked elsewhere
        p, out = _regen_manifest(base)                                                          # PD; protocol_commit "a"*40
        dim_sha = _fsha(p)
        env_sha = json.load(open(p))["env_lock_sha256"]                                         # bind auth to the run's env lock
        authp = os.path.join(base, "auth.json")
        with open(authp, "w") as f:
            json.dump(_b1_auth("a" * 40, dim_sha, env_sha, out), f)
        body = RRS.run(p, out, b1_authorization=authp)                                          # AUTHORIZED -> gated trainer runs
        assert calls == [out]                                                                   # trainer called exactly once
        assert os.path.isfile(os.path.join(out, "RESULT.json")) and os.path.isfile(os.path.join(out, "manifest.json"))
        a = body["artifacts"]                                                                   # BOTH file + canonical shas recorded
        assert a["encoder_state_dict_sha256"] == "c" * 64 and a["source_state_artifact_sha256"] == "d" * 64
        assert len(a["encoder_checkpoint_file_sha256"]) == 64 and len(a["source_state_file_sha256"]) == 64
        assert a["encoder_checkpoint_file_sha256"] != a["encoder_state_dict_sha256"]            # file-bytes hash != semantic hash
        assert "source_state_sha256" not in a and "encoder_checkpoint_sha256" not in a          # retired ambiguous names gone
        res = json.load(open(os.path.join(out, "RESULT.json")))
        assert {"encoder_state_dict_sha256", "encoder_checkpoint_file_sha256",
                "source_state_artifact_sha256", "source_state_file_sha256"} <= set(res)
        # runtime mismatch -> fail BEFORE training + BEFORE output claim
        def bad_runtime(spec):
            raise ValueError("runtime torch_version != env lock")
        RRS._verify_runtime_matches_lock = bad_runtime
        outR = os.path.join(base, "outR")
        with open(os.path.join(base, "authR.json"), "w") as f:
            json.dump(_b1_auth("a" * 40, dim_sha, env_sha, outR), f)
        _expect(ValueError, lambda: RRS.run(p, outR, b1_authorization=os.path.join(base, "authR.json")))
        assert calls == [out] and not os.path.exists(outR)                                      # trainer not called; no output
        RRS._verify_runtime_matches_lock = lambda spec: None
        # auth mismatch (wrong output_path) -> fail before training, no second call, output not created
        out2 = os.path.join(base, "out2")
        authp2 = os.path.join(base, "auth2.json")
        with open(authp2, "w") as f:
            json.dump(_b1_auth("a" * 40, dim_sha, env_sha, "/WRONG"), f)
        _expect(ValueError, lambda: RRS.run(p, out2, b1_authorization=authp2))
        assert calls == [out] and not os.path.exists(out2)
        # trainer returns paths but OMITS the canonical (semantic) shas -> fail closed + cleanup
        def nohash(spec, output):
            enc = os.path.join(output, "enc.pt"); ss = os.path.join(output, "ss.npz")
            with open(enc, "wb") as f: f.write(b"E")
            with open(ss, "wb") as f: f.write(b"S")
            return {"encoder_checkpoint_path": enc, "source_state_path": ss}                    # no encoder_state_dict_sha256
        RRS._train_substrate = nohash
        out4 = os.path.join(base, "out4")
        with open(os.path.join(base, "auth4.json"), "w") as f:
            json.dump(_b1_auth("a" * 40, dim_sha, env_sha, out4), f)
        _expect(RuntimeError, lambda: RRS.run(p, out4, b1_authorization=os.path.join(base, "auth4.json")))
        assert not os.path.exists(out4)                                                         # no manifest/RESULT without canonical shas
        RRS._train_substrate = fake_train
        # trainer failure -> output cleaned
        def boom(spec, output):
            os.path.exists(output); raise RuntimeError("train boom")
        RRS._train_substrate = boom
        out3 = os.path.join(base, "out3")
        authp3 = os.path.join(base, "auth3.json")
        with open(authp3, "w") as f:
            json.dump(_b1_auth("a" * 40, dim_sha, env_sha, out3), f)
        _expect(RuntimeError, lambda: RRS.run(p, out3, b1_authorization=authp3))
        assert not os.path.exists(out3)                                                         # claimed dir removed on abort
    finally:
        RRS._git, RRS._verify_eligible_subjects, RRS._train_substrate, RRS._verify_runtime_matches_lock = saved
        shutil.rmtree(base, ignore_errors=True)


def _sub_manifest_files(base, *, missing=False, bad=None, no_env=False):   # bad in {enc,ss,dim,env}; no_env drops the env lock file
    subs = {}
    env = os.path.join(base, "ABSENT_env_lock.json" if no_env else "env_lock.json")            # absent path => FileNotFoundError
    if not missing and not no_env:
        with open(env, "wb") as f:
            f.write(b"ENVLOCK")
    env_sha = ("0" * 64 if bad == "env" else (_fsha(env) if (not missing and not no_env) else "b" * 64))
    for d in ("PD", "SCZ"):
        enc = os.path.join(base, f"enc_{d}.pt"); ss = os.path.join(base, f"ss_{d}.npz")
        dim = os.path.join(base, f"dim_{d}.json")
        if not missing:
            for path, payload in ((enc, "enc-" + d), (ss, "ss-" + d), (dim, "dim-" + d)):
                with open(path, "wb") as f:
                    f.write(payload.encode())
        def _sha(path, key):                                                                   # FILE-byte hash (preflight verifies)
            return "0" * 64 if bad == key else (_fsha(path) if not missing else "b" * 64)
        subs[d] = {"encoder_checkpoint_path": enc, "encoder_checkpoint_file_sha256": _sha(enc, "enc"),
                   "encoder_state_dict_sha256": "b" * 64,                                       # canonical (verified at replay)
                   "source_state_path": ss, "source_state_file_sha256": _sha(ss, "ss"),
                   "source_state_artifact_sha256": "b" * 64,
                   "encoder_provenance_path": enc + ".prov.json", "source_state_provenance_path": ss + ".prov.json",
                   "dev_input_manifest_path": dim, "dev_input_manifest_sha256": _sha(dim, "dim")}
    return _sub_manifest(substrates=subs, env_lock_path=env, env_lock_sha256=env_sha)


def test_load_eligible_windows_excludes_before_open_and_cohort_aware():
    from acar.v4 import run_regen_substrate as RRS
    import numpy as np
    opened = []

    def fake_loader(disease, cohort, subject, cohort_dir, cfg):
        opened.append((disease, cohort, subject, cohort_dir))
        return np.zeros((2, cfg["canon_channels"], int(cfg["resample_fs"] * cfg["window_sec"])), dtype="<f4"), 0
    dis = "SCZ"; cohorts = list(R.DEV_SCOPE[dis]); c0, c1 = cohorts[0], cohorts[1]
    spec = {"disease": dis, "dev_cohorts": cohorts, "source_paths": {c0: "/raw/" + c0, c1: "/raw/" + c1}}
    # allowlist is cohort-aware: same local id 'sub-1' in two cohorts is distinct; 'sub-999' in c0 is EXCLUDED (not listed)
    allowlist = {f"{c0}/sub-1", f"{c1}/sub-1"}
    X, y, subj = RRS.load_eligible_windows(spec, allowlist, signal_loader=fake_loader)
    assert sorted(o[:3] for o in opened) == sorted([(dis, c0, "sub-1"), (dis, c1, "sub-1")])   # only eligible; cohort-aware
    assert all((dis, c, "sub-999") not in [o[:3] for o in opened] for c in (c0, c1))           # excluded NEVER opened
    assert {o[3] for o in opened} == {"/raw/" + c0, "/raw/" + c1}                              # cohort dir threaded through
    assert X.shape == (4, 19, 512)                                                             # 2 cohorts × 1 subj × 2 windows
    assert set(subj) == {f"{c0}/sub-1", f"{c1}/sub-1"} and len(subj) == 4


def test_cmi_load_cohort_honors_subject_allowlist():
    """The loader edit: subjects=allowlist skips every other subject at file-discovery, BEFORE _read_raw opens signal."""
    try:
        from cmi.data import bids_data as B
    except Exception as e:                                                        # env without mne/cmi -> skip (suite stays green)
        print(f"  [skip] cmi load_cohort allowlist (import: {type(e).__name__})"); return
    saved = (B.glob.glob, B._read_participants, B._read_raw)
    opened = []

    def fake_glob(pattern, *a, **k):
        if pattern.endswith("sub-*"):
            return [os.path.join("/d", "sub-1"), os.path.join("/d", "sub-2")]     # two subjects on disk
        if "task-" in pattern:
            sd = pattern.split(os.sep + "**")[0]                                  # the subject dir embedded in the task glob
            return [os.path.join(sd, "eeg", "x_task-rest_eeg.set")]
        return []

    def fake_read_raw(path):
        opened.append(path); raise RuntimeError("stop after open (test)")         # record + skip the heavy mne pipeline
    try:
        B.glob.glob = fake_glob
        B._read_participants = lambda ds: {"sub-1": {}, "sub-2": {}}
        B._read_raw = fake_read_raw
        B.load_cohort("/d", "rest", lambda row, sid: 1, subjects={"sub-1"})       # allowlist = only sub-1
        assert [p for p in opened if os.sep + "sub-1" + os.sep in p] == opened     # only sub-1 opened
        assert all(os.sep + "sub-2" + os.sep not in p for p in opened)            # sub-2 NEVER opened (excluded)
        assert len(opened) >= 1
    finally:
        B.glob.glob, B._read_participants, B._read_raw = saved


def test_run_substrate_compatibility_fail_closed():
    from acar.v4 import run_regen_substrate as RRS
    from acar.v4 import run_substrate_compatibility as RSC
    base = tempfile.mkdtemp(); saved = RRS._git
    try:
        RRS._git = _fake_git()                                                                  # HEAD == "a"*40 == compatibility_protocol_commit
        out = os.path.join(base, "compat_out")
        p = os.path.join(base, "sub_ok.json")
        with open(p, "w") as f:
            json.dump(_sub_manifest_files(base), f)                                             # real artifacts + dev-input + env-lock + sha
        _expect(R.SubstrateCompatibilityNotAuthorizedError, lambda: RSC.run(p, out))            # full preflight passes → gated (no auth)
        assert not os.path.exists(out) and "torch" not in sys.modules                           # no output, no heavy import on this path
        # HEAD must == compatibility_protocol_commit (NOT the substrate commit)
        ph = os.path.join(base, "sub_headmm.json")
        with open(ph, "w") as f:
            json.dump(_sub_manifest_files(base, ), f)                                           # compatibility_protocol_commit="a"*40
        RRS._git = _fake_git(commit="e" * 40)
        _expect(ValueError, lambda: RSC.run(ph, out))                                           # HEAD != compatibility_protocol_commit
        RRS._git = _fake_git()
        pb = os.path.join(base, "sub_bad.json")
        with open(pb, "w") as f:
            json.dump(_sub_manifest(candidate={"x": 1}), f)
        _expect(ValueError, lambda: RSC.run(pb, out))                                           # reselection → before gate
        pp = os.path.join(base, "sub_proto.json")
        with open(pp, "w") as f:
            json.dump(_sub_manifest_files(base, ) | {"protocol_commit": "a" * 40}, f)           # retired single commit field
        _expect(ValueError, lambda: RSC.run(pp, out))
        pm = os.path.join(base, "sub_missing.json")
        with open(pm, "w") as f:
            json.dump(_sub_manifest_files(os.path.join(base, "nope"), missing=True), f)
        _expect(FileNotFoundError, lambda: RSC.run(pm, out))                                    # artifact path missing
        pne = os.path.join(base, "sub_noenv.json")
        with open(pne, "w") as f:
            json.dump(_sub_manifest_files(base, no_env=True), f)
        _expect(FileNotFoundError, lambda: RSC.run(pne, out))                                   # env-lock file missing
        for key in ("enc", "ss", "dim", "env"):                                                # EACH file-byte sha mismatch branch
            pk = os.path.join(base, f"sub_bad_{key}.json")
            with open(pk, "w") as f:
                json.dump(_sub_manifest_files(base, bad=key), f)
            _expect(ValueError, lambda pk=pk: RSC.run(pk, out))
        assert "torch" not in sys.modules                                                       # no heavy import on any preflight path
    finally:
        RRS._git = saved; shutil.rmtree(base, ignore_errors=True)


def test_run_substrate_compatibility_authorized_runs_gated_replay():
    from acar.v4 import run_regen_substrate as RRS
    from acar.v4 import run_substrate_compatibility as RSC
    base = tempfile.mkdtemp()
    saved = (RRS._git, RSC._run_compatibility_replay)
    calls = []

    def fake_replay(spec, output, status="SUBSTRATE_COMPATIBILITY_PASS"):
        calls.append(output)
        return {"status": status, "reason": "synthetic", "per_disease": {"PD": {}, "SCZ": {}}}
    try:
        RRS._git = _fake_git()
        p = os.path.join(base, "sub_ok.json")
        with open(p, "w") as f:
            json.dump(_sub_manifest_files(base), f)
        ims = _fsha(p); env_sha = json.load(open(p))["env_lock_sha256"]
        out = os.path.join(base, "compat_run")

        def _auth(outp, **over):
            a = {"compatibility_protocol_commit": "a" * 40, "substrate_protocol_commit": "f" * 40,
                 "substrate_manifest_sha256": ims, "env_lock_sha256": env_sha, "output_path": outp,
                 "authorized_by": "yinghao", "authorization_time": "2026-06-30T12:00:00Z",
                 "statement": R.REQUIRED_COMPAT_STATEMENT}
            a.update(over); ap = os.path.join(base, f"auth_{os.path.basename(outp)}.json")
            with open(ap, "w") as f: json.dump(a, f)
            return ap
        # PASS verdict -> compat_manifest + compat_RESULT(status) written; replay called once
        RSC._run_compatibility_replay = fake_replay
        body = RSC.run(p, out, compat_authorization=_auth(out))
        assert calls == [out]
        res = json.load(open(os.path.join(out, "compat_RESULT.json")))
        man = json.load(open(os.path.join(out, "compat_manifest.json")))
        assert res["status"] == "SUBSTRATE_COMPATIBILITY_PASS" and res["candidate"] == R.FIXED_CANDIDATE
        assert man["verdict"]["status"] == "SUBSTRATE_COMPATIBILITY_PASS"
        assert man["substrate_protocol_commit"] == "f" * 40 and man["compatibility_protocol_commit"] == "a" * 40
        assert "SELECT" not in json.dumps(res) and "DEV_STOP" not in json.dumps(res)            # no selection/external vocab
        # FAIL verdict is a normal written result (NOT an abort)
        outf = os.path.join(base, "compat_fail")
        RSC._run_compatibility_replay = lambda spec, output: fake_replay(spec, output, status="SUBSTRATE_COMPATIBILITY_FAIL")
        RSC.run(p, outf, compat_authorization=_auth(outf))
        assert json.load(open(os.path.join(outf, "compat_RESULT.json")))["status"] == "SUBSTRATE_COMPATIBILITY_FAIL"
        assert calls == [out, outf]                                                             # replay called exactly once per run
        # invalid-but-present authorizations -> fail in _load_compat_authorization BEFORE replay/output (replay NOT called)
        for over in ({"substrate_manifest_sha256": "d" * 64}, {"env_lock_sha256": "d" * 64},
                     {"compatibility_protocol_commit": "e" * 40}, {"statement": "ok go"}):
            n = len(calls); outx = os.path.join(base, "compat_inv_" + list(over)[0])
            _expect(ValueError, lambda outx=outx, over=over: RSC.run(p, outx, compat_authorization=_auth(outx, **over)))
            assert not os.path.exists(outx) and len(calls) == n and "torch" not in sys.modules   # no replay, no output, no torch
        # replay returns a non-verdict status -> abort + cleanup (OPERATIONALLY_ABORTED_NO_VERDICT; never read as FAIL)
        out3 = os.path.join(base, "compat_nonverdict")
        RSC._run_compatibility_replay = lambda spec, output: {"status": "OPERATIONALLY_ABORTED_NO_VERDICT"}
        _expect(RuntimeError, lambda: RSC.run(p, out3, compat_authorization=_auth(out3)))
        assert not os.path.exists(out3) and "torch" not in sys.modules
        # replay raises operationally -> output cleaned (no partial)
        out4 = os.path.join(base, "compat_boom")
        def boom(spec, output): raise R.SubstrateReplayNotWiredError("frontier")
        RSC._run_compatibility_replay = boom
        _expect(R.SubstrateReplayNotWiredError, lambda: RSC.run(p, out4, compat_authorization=_auth(out4)))
        assert not os.path.exists(out4) and "torch" not in sys.modules
    finally:
        RRS._git, RSC._run_compatibility_replay = saved
        shutil.rmtree(base, ignore_errors=True)


def test_compat_replay_inner_frontiers_are_controlled():
    """The re-embed + fixed-candidate stat-extraction frontiers raise a CONTROLLED SubstrateReplayNotWiredError (never a
    silently-wrong verdict) until finalized at the authorized C-run."""
    from acar.v4 import run_substrate_compatibility as RSC
    _expect(R.SubstrateReplayNotWiredError, lambda: RSC._reembed_dev_under_substrate({}, "/tmp/x"))
    _expect(R.SubstrateReplayNotWiredError, lambda: RSC._extract_fixed_candidate_stats(object(), {}))


def main():
    print("ACAR v4 regen_substrate guards (skeleton + B1-preflight command contract; NO training):")
    for t in (test_validate_substrate_request, test_train_not_authorized, test_compatibility_replay_pass,
              test_validate_regen_manifest, test_validate_substrate_manifest, test_check_eligible_subjects,
              test_validate_b1_authorization, test_validate_compat_authorization, test_require_cuda,
              test_check_runtime_matches_lock, test_assert_finite,
              test_single_subject_label, test_check_training_set, test_canonical_state_dict_sha256,
              test_run_regen_substrate_fail_closed,
              test_run_regen_substrate_authorized_runs_gated_trainer,
              test_load_eligible_windows_excludes_before_open_and_cohort_aware,
              test_run_substrate_compatibility_fail_closed,
              test_run_substrate_compatibility_authorized_runs_gated_replay,
              test_compat_replay_inner_frontiers_are_controlled,
              test_cmi_load_cohort_honors_subject_allowlist):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 REGEN-SUBSTRATE GUARDS PASS")


if __name__ == "__main__":
    main()
