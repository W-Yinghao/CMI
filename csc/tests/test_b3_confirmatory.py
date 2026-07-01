"""
CSC Route B3 confirmatory FREEZE-PACKAGE tests: manifest self-consistency, seed-schedule completeness +
disjointness, fail-closed provenance (manifest-hash / code / scenario / seed tampering), --execute refused
without the frozen tag, a SMOKE execute (tiny grid, NOT the real base 3000000) that exercises the
run+artifact+criteria machinery, structured smoke-bypass refusal, and sbatch wrapper shape.
Standalone:  python -m csc.tests.test_b3_confirmatory
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
    fp = R.load_manifest()["frozen_payload"]; ss = fp["seed_spec"]
    sched = R.generate_seed_schedule(fp)
    cluster = [s for c in sched for s in c["seeds"]]
    off = ss.get("seed_target_offset", 0)
    used = set(cluster) | {s + off for s in cluster}          # cluster AND target-offset RNG seeds
    assert ss["base_seed"] == 3000000, ss["base_seed"]
    assert len(sched) == 112 and len(cluster) == 5376, (len(sched), len(cluster))
    assert len(used) == 2 * len(cluster), "cluster and target seeds must not collide (stride > offset+reps)"
    assert min(cluster) >= 1_000_000
    errs, _, rng = R.verify_seed_schedule(fp)
    assert not errs, errs
    # explicit A-line stream disjointness (source 900000..65 AND target 1800000..65)
    assert not (used & set(range(900000, 900066))), "must not reuse A source seeds"
    assert not (used & set(range(1800000, 1800066))), "must not reuse A target seeds"
    print(f"OK seed schedule complete ({len(cluster)} cluster + {len(cluster)} target seeds) "
          f"range {rng}, disjoint from A source+target / B dev / test")


def test_a_target_stream_excluded():
    fp = R.load_manifest()["frozen_payload"]
    forb = R._forbidden(fp["seed_spec"]["development_seed_exclusion"], 100000, 48)
    assert 900000 in forb and 1800000 in forb, "A source AND target streams must be excluded"
    # a base that lands a cell on the A target stream must be caught (the 1200000/stride-10000 bug)
    import copy
    m = copy.deepcopy(fp); m["seed_spec"]["base_seed"] = 1200000; m["seed_spec"]["cell_stride"] = 10000
    assert R.verify_seed_schedule(m)[0], "base 1200000/stride 10000 (hits A target 1800000) must be caught"
    print("OK A confirmatory source(900000) AND target(1800000) streams both excluded; 1200000 bug caught")


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
    """Valid tiny smoke manifest: all 6 scenario_configs (for verify_scenarios), tiny grid, base 200000
    (NOT the real 3000000; disjoint from A/B/test), 2 replicates, n_boot small."""
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
        "seed_spec": {"base_seed": 200000, "cell_stride": 100000, "replicates": 2, "seed_target_offset": 10000,
                      "development_seed_exclusion": {"A_confirmatory_source_seeds": [900000, 900065],
                                                     "A_confirmatory_target_seeds": [1800000, 1800065],
                                                     "B_development_blocks": [0, 1000, 2000, 3000, 4000, 700000],
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
        # provenance payload present (git head/ref/commit/clean + seed schedule summary + slurm)
        for k in ("git_head", "expected_code_ref", "expected_code_commit", "git_status_clean"):
            assert k in art["code_provenance"], f"missing code_provenance.{k}"
        assert art["seed_schedule"]["base_seed"] == 200000 and art["seed_schedule"]["disjointness_checked"]
        assert "slurm" in art
        print(f"OK smoke execute wrote artifact + C1-C5 + provenance payload (verdict "
              f"{art['verdict']['preliminary_scientific_verdict_excluding_C6']}, C6 pending)")


def test_smoke_refuses_real_confirmatory():
    # --smoke must NOT be usable to run the real manifest or the real base_seed (git-guard bypass path)
    def smoke(mp):
        return subprocess.run([sys.executable, "-m", "csc.mininfo.run_b3_confirmatory", "--smoke", mp],
                              cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": ROOT})
    r = smoke(R.MANIFEST)                                  # the real manifest
    assert r.returncode == 2 and "REFUSED" in (r.stdout + r.stderr), "--smoke on real manifest must refuse"
    with tempfile.TemporaryDirectory() as td:
        # a smoke manifest that (illegally) uses the real base_seed 3000000 must also be refused
        m = json.load(open(R.MANIFEST)); m["frozen_payload"]["protocol"] = "smoke-x"
        m["frozen_payload"]["seed_spec"]["base_seed"] = 3000000
        m["manifest_hash"] = R.canonical_manifest_hash(m["frozen_payload"])
        p = os.path.join(td, "bad.json"); json.dump(m, open(p, "w"))
        rr = smoke(p)
        assert rr.returncode == 2 and "REFUSED" in (rr.stdout + rr.stderr), "--smoke base 3000000 must refuse"
    print("OK --smoke refuses real manifest AND real base_seed 3000000 (structured, not string match)")


def test_sbatch_wrapper_shape():
    p = os.path.join(ROOT, "csc/mininfo/run_b3_confirmatory.sbatch")
    txt = open(p).read().splitlines()
    assert subprocess.run(["bash", "-n", p]).returncode == 0, "sbatch must pass bash -n"
    assert txt[0].startswith("#!/bin/bash"), "shebang line 1"
    assert any(l.startswith("#SBATCH --chdir=") for l in txt), "needs #SBATCH --chdir on its own line"
    assert "CMI_AAAI_csc_b3_frozen" in "\n".join(txt), "must run from the frozen worktree, not main workdir"
    # every non-blank, non-#! line must be a comment, #SBATCH, or shell code -- no un-commented prose
    for l in txt:
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        assert not s.startswith("This wrapper"), f"un-commented prose line: {l!r}"
    body = "\n".join(txt)
    assert 'rm -f "$OUT"' in body and 'mv "$TMP_OUT" "$OUT"' in body, "stale-rm + temp->final mv required"
    assert "infra_fail" in body and "sha256sum" in body, "fail-closed + integrity hash required"
    assert "manifest_hash" in body and "base_seed" in body and "git_status_clean" in body, "freshness re-verify"
    print(f"OK sbatch wrapper shape valid (bash -n, #SBATCH lines, frozen worktree, rm/mv, freshness re-verify)")


def test_dry_run_clean():
    assert R.dry_run() is True
    print("OK dry-run provenance CLEAN on committed manifest")


if __name__ == "__main__":
    test_manifest_self_consistent()
    test_scenarios_match_code()
    test_seed_schedule_complete_and_disjoint()
    test_a_target_stream_excluded()
    test_fail_closed_tampering()
    test_execute_refused_without_tag()
    test_smoke_execute_writes_artifact_and_evaluates()
    test_smoke_refuses_real_confirmatory()
    test_sbatch_wrapper_shape()
    test_dry_run_clean()
    print("\nall CSC B3 confirmatory freeze-package tests passed")
