"""
CSC Route B3 confirmatory DRY-RUN + fail-closed tests. Validates the freeze-candidate manifest is
self-consistent and that the dry-run validator fails closed on tampering and refuses --execute. Runs
standalone:  python -m csc.tests.test_b3_confirmatory
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

from csc.mininfo import run_b3_confirmatory as R


def test_manifest_self_consistent():
    mani = R.load_manifest()
    assert not R.verify_code_hashes(mani), f"code hashes drifted: {R.verify_code_hashes(mani)}"
    assert not R.verify_scenarios(mani), f"scenario configs drifted: {R.verify_scenarios(mani)}"
    assert not R.verify_seed_disjoint(mani), f"seed overlap: {R.verify_seed_disjoint(mani)}"
    print("OK manifest self-consistent (code hashes, scenario configs, seed disjointness)")


def test_scenarios_match_code():
    from csc.mininfo.run_b3_p23 import SCENARIOS
    mani = R.load_manifest()
    assert mani["scenario_configs"] == SCENARIOS, "manifest scenario_configs must equal run_b3_p23.SCENARIOS"
    print("OK manifest scenario_configs == run_b3_p23.SCENARIOS (exact)")


def test_seed_disjoint_from_A_confirmatory():
    mani = R.load_manifest()
    base = mani["seed_spec"]["confirmatory_base_seed"]
    assert base != 900000, "B3 confirmatory seed must NOT reuse A-line 900000"
    assert base >= 1_000_000, "B3 confirmatory seed should be a fresh high block"
    assert not R.verify_seed_disjoint(mani)
    print(f"OK confirmatory seed {base} disjoint from A(900000) + B dev + test range")


def test_fail_closed_on_code_tamper(tmp=None):
    mani = copy.deepcopy(R.load_manifest())
    # corrupt one recorded hash -> provenance must FAIL
    k = next(iter(mani["code_hashes_sha256"]))
    mani["code_hashes_sha256"][k] = "0" * 64
    assert R.verify_code_hashes(mani), "tampered code hash must be caught (fail closed)"
    # corrupt a scenario config -> scenario check must FAIL
    mani2 = copy.deepcopy(R.load_manifest())
    mani2["scenario_configs"]["baseline"] = {"cfg_subject_tau": 9.9}
    assert R.verify_scenarios(mani2), "drifted scenario config must be caught (fail closed)"
    # move seed onto A block -> disjointness must FAIL
    mani3 = copy.deepcopy(R.load_manifest())
    mani3["seed_spec"]["confirmatory_base_seed"] = 900000
    assert R.verify_seed_disjoint(mani3), "seed overlap with A block must be caught (fail closed)"
    print("OK fail-closed: code-tamper, scenario-drift, seed-overlap each caught")


def test_execute_is_refused():
    r = subprocess.run([sys.executable, "-m", "csc.mininfo.run_b3_confirmatory", "--execute"],
                       cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.abspath(
                           os.path.join(os.path.dirname(__file__), "..", ".."))})
    assert r.returncode == 2, f"--execute must exit 2 (refused), got {r.returncode}"
    assert "NOT AUTHORIZED" in (r.stdout + r.stderr), "must state NOT AUTHORIZED"
    print("OK --execute refused (exit 2, NOT AUTHORIZED)")


def test_dry_run_clean():
    assert R.dry_run() is True, "dry-run provenance must be clean on the committed manifest"
    print("OK dry-run provenance CLEAN on committed manifest")


if __name__ == "__main__":
    test_manifest_self_consistent()
    test_scenarios_match_code()
    test_seed_disjoint_from_A_confirmatory()
    test_fail_closed_on_code_tamper()
    test_execute_is_refused()
    test_dry_run_clean()
    print("\nall CSC B3 confirmatory dry-run + fail-closed tests passed")
