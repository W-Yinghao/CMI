"""
CSC Route B3 confirmatory FREEZE-PACKAGE tests: manifest self-consistency, seed-schedule completeness +
disjointness, fail-closed provenance (manifest-hash / code / scenario / seed tampering), --execute refused
without the frozen tag, and a SMOKE execute (tiny grid, NOT base 1200000) that exercises the run+artifact+
criteria machinery. Standalone:  python -m csc.tests.test_b3_confirmatory
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile

from csc.mininfo import run_b3_confirmatory as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _h(rel):
    return hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()


def test_manifest_self_consistent():
    m = R.load_manifest()
    assert not R.verify_manifest_hash(m), R.verify_manifest_hash(m)
    assert not R.verify_code_hashes(m), R.verify_code_hashes(m)
    assert not R.verify_scenarios(m), R.verify_scenarios(m)
    errs, n, rng = R.verify_seed_schedule(m["frozen_payload"])
    assert not errs, errs
    print(f"OK manifest self-consistent (hash/code/scenario/seed); {n} seeds range {rng}")


def test_scenarios_match_code():
    from csc.mininfo.run_b3_p23 import SCENARIOS
    assert R.load_manifest()["frozen_payload"]["scenario_configs"] == SCENARIOS
    print("OK frozen_payload.scenario_configs == run_b3_p23.SCENARIOS (exact)")


def test_seed_schedule_complete_and_disjoint():
    fp = R.load_manifest()["frozen_payload"]
    sched = R.generate_seed_schedule(fp)
    all_seeds = [s for c in sched for s in c["seeds"]]
    assert fp["seed_spec"]["base_seed"] == 1200000
    assert len(all_seeds) == len(set(all_seeds)), "seeds must be unique"
    assert len(sched) == 112 and len(all_seeds) == 5376, (len(sched), len(all_seeds))
    assert min(all_seeds) >= 1_000_000, "confirmatory seeds must be a fresh high block"
    errs, _, _ = R.verify_seed_schedule(fp)
    assert not errs, errs
    print(f"OK seed schedule complete ({len(sched)} cells x 48 = {len(all_seeds)}) + disjoint from A/B/test")


def test_fail_closed_tampering():
    base = R.load_manifest()
    # manifest-hash mismatch
    m = copy.deepcopy(base); m["frozen_payload"]["method_lock"]["C"] = 0.123
    assert R.verify_manifest_hash(m), "tampered payload must break manifest hash"
    # code-hash mismatch
    m = copy.deepcopy(base); k = next(iter(m["frozen_payload"]["code_hashes_sha256"]))
    m["frozen_payload"]["code_hashes_sha256"][k] = "0" * 64
    assert R.verify_code_hashes(m), "tampered code hash must be caught"
    # scenario drift
    m = copy.deepcopy(base); m["frozen_payload"]["scenario_configs"]["baseline"] = {"cfg_subject_tau": 9.9}
    assert R.verify_scenarios(m), "scenario drift must be caught"
    # seed overlap with A
    m = copy.deepcopy(base); m["frozen_payload"]["seed_spec"]["base_seed"] = 900000
    assert R.verify_seed_schedule(m["frozen_payload"])[0], "seed overlap with A must be caught"
    print("OK fail-closed: manifest-hash / code / scenario / seed-overlap each caught")


def test_execute_refused_without_tag():
    # the frozen tag does not exist yet -> --execute must fail closed (exit 2)
    r = subprocess.run([sys.executable, "-m", "csc.mininfo.run_b3_confirmatory", "--execute"],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 2, f"--execute must be refused (exit 2) without frozen tag, got {r.returncode}"
    assert "REFUSED" in (r.stdout + r.stderr)
    print("OK --execute refused without frozen tag (exit 2, fail-closed)")


def _smoke_manifest(tmpdir):
    """Valid tiny smoke manifest: all 6 scenario_configs (for verify_scenarios), tiny grid, base 500000
    (NOT 1200000, disjoint from A/B/test), 2 replicates, n_boot small."""
    from csc.mininfo.run_b3_p23 import SCENARIOS
    fp = {
        "protocol": "csc-b3-confirmatory-SMOKE", "version": "smoke",
        "calibration_version": "smoke", "expected_code_ref": "refs/tags/does-not-exist",
        "method_lock": {"method": "pc_centered_calibrated", "h1_basis": "pc", "condition_coding": "centered",
                        "rank": 3, "C": 0.5, "null": "x", "finite_sample_gate": "x", "alpha_family": 0.05,
                        "positive_decision_budgets": [20], "alpha_budget": 0.025, "lcb_level": 0.975,
                        "min_confirm_pairs": 20, "pair_integrity_min": 0.95, "min_epochs_per_condition": 8,
                        "n_boot": 20, "n_folds": 3, "n_subjects": 36, "clusters_per_cell": 2},
        "grid": {"control_kinds": ["clean", "missing_pair"], "control_scenarios": ["baseline"],
                 "primary_positive_kinds": ["paired_concept"], "primary_scenarios": ["baseline"],
                 "primary_budgets": [20], "secondary_kinds": [], "decision_budgets": [20]},
        "scenario_configs": SCENARIOS,
        "seed_spec": {"base_seed": 500000, "cell_stride": 10000, "replicates": 2,
                      "development_seed_exclusion": {"A_confirmatory": 900000,
                                                     "B_development": [0, 1000, 2000, 3000, 4000, 700000],
                                                     "smoke_test_range": "<100000"}},
        "pass_criteria": {"type": "smoke"},
        "code_hashes_sha256": {r: _h(r) for r in ("csc/mininfo/paired_calibrated.py",
                                                  "csc/mininfo/paired_conditional_test.py",
                                                  "csc/mininfo/paired_sim.py")},
    }
    mani = {"protocol": "csc-b3-confirmatory-SMOKE", "version": "smoke",
            "manifest_hash": R.canonical_manifest_hash(fp), "status": "SMOKE", "frozen_payload": fp}
    p = os.path.join(tmpdir, "smoke_manifest.json")
    json.dump(mani, open(p, "w"))
    assert "1200000" not in open(p).read(), "smoke manifest must NOT contain base 1200000"
    return p


def test_smoke_execute_writes_artifact_and_evaluates():
    with tempfile.TemporaryDirectory() as td:
        mp = _smoke_manifest(td); out = os.path.join(td, "smoke_result.json")
        rc = R.execute(mp, n_jobs=1, out=out, require_git=False)   # no git guard for smoke
        assert rc == 0, f"smoke execute should run (rc={rc})"
        assert os.path.exists(out), "smoke execute must write an artifact (even on scientific FAIL)"
        art = json.load(open(out))
        assert "verdict" in art and "criteria" in art["verdict"], "artifact must carry the C1-C5 verdict"
        assert art["verdict"]["red_team_required"] is True
        assert art["verdict"]["criteria"]["C6_independent_verification"]["passed"] is None
        print(f"OK smoke execute wrote artifact + evaluated C1-C5 (verdict "
              f"{art['verdict']['preliminary_scientific_verdict_excluding_C6']}, C6 pending)")


def test_dry_run_clean():
    assert R.dry_run() is True
    print("OK dry-run provenance CLEAN on committed manifest")


if __name__ == "__main__":
    test_manifest_self_consistent()
    test_scenarios_match_code()
    test_seed_schedule_complete_and_disjoint()
    test_fail_closed_tampering()
    test_execute_refused_without_tag()
    test_smoke_execute_writes_artifact_and_evaluates()
    test_dry_run_clean()
    print("\nall CSC B3 confirmatory freeze-package tests passed")
