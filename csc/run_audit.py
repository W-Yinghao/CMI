"""
csc.run_audit — one machine-readable audit artifact per commit (CSC-P1.3).

Everything goes through the SINGLE frozen path (csc.protocol.run_frozen_protocol). The audit
records the full serializable ProtocolConfig MANIFEST (tau_detect/tau_label as RULES, not
numbers) + its hash, both validity BANKS, and full provenance.

Run via SLURM (CPU partition), NOT the login node:  sbatch csc/run_audit.sbatch
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import warnings

import numpy as np

from csc.protocol import ProtocolConfig, ood_power_bank, synthetic_null_bank
from csc.run_synthetic import run as run_syn
from csc.sim.shift_simulator import SimConfig, make_source
from csc.calibration.lodo import nested_lodo, VISIBLE_CONCEPT, COVARIATE_STABLE, AMBIGUOUS

TEST_MODULES = ["test_design_and_pairs", "test_validity_gate", "test_null_calibration",
                "test_power", "test_protocol", "test_cluster_inference",
                "test_paired_and_accounting", "test_p143_contracts"]


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def p142_diagnostics(cfg, seeds):
    """CSC-P1.4.2-specific diagnostics retained in the audit artifact: per-source null invalid
    counts + conservative-p, SOURCE_INVALID triggers, paired-unit invariants, cross-split
    separability response, and the deterministic NAMED stage-seed derivations."""
    from csc.certificate import analyze_source
    from csc.certificate.residual_test import stage_seed, aggregate_subject_loss
    from csc.certificate.atlas import (support_signature_strata, stratified_subject_resample,
                                       cov_concept_angle)
    from csc.sim.shift_simulator import make_source, make_paired_subjects

    per_source = []
    for s in seeds:
        src = make_source(SimConfig(seed=s), n_domains=8, concept_domains=3, seed=s)
        sa = analyze_source(src.Z, src.Y, src.D, n_boot=cfg.n_boot, n_dir_boot=cfg.n_dir_boot,
                            alpha=cfg.alpha, var_keep=cfg.var_keep, C=cfg.C,
                            n_folds=cfg.source_cv_folds, invalid_frac_max=cfg.invalid_null_frac_max,
                            group_ids=src.group_ids, seed=s)
        d = sa.detail
        per_source.append(dict(
            seed=s, source_status=sa.source_status, source_test_status=sa.test.status,
            residual_null_invalid=int(getattr(sa.test, "null_invalid", 0)),
            residual_p_value=float(sa.test.p_value),
            geom_null_invalid=int(d.get("geom_null_invalid", 0)),
            geom_null_estimable=bool(d.get("geom_null_estimable", True)),
            cov_boot_invalid=int(d.get("cov_boot_invalid", 0)),
            n_dir_boot=cfg.n_dir_boot, invalid_frac_max=cfg.invalid_null_frac_max,
            # CSC-P1.4.3 #2: source_invalid covers ALL stages (support/residual/geometry/separability)
            source_invalid_triggered=(sa.source_status != "VALID"),
            concept_disagreement_deg=float(d.get("concept_disagreement_deg", float("nan"))),
            concept_stable=bool(d.get("concept_stable", False)),
            separability_assessable=bool(d.get("separability_assessable", False))))

    # paired ON/OFF invariants (numeric evidence, not just test pass/fail)
    Zp, Yp, Dp, Gp = make_paired_subjects(SimConfig(seed=0), n_subjects=20, concept_delta=0.0, seed=0)
    base = aggregate_subject_loss(np.arange(len(Yp), dtype=float), Gp, Dp)
    dup = (Gp == Gp[0]) & (Dp == Dp[0])
    G2 = np.concatenate([Gp, Gp[dup]]); D2 = np.concatenate([Dp, Dp[dup]])
    L2 = np.concatenate([np.arange(len(Yp), dtype=float), np.arange(len(Yp), dtype=float)[dup]])
    after = aggregate_subject_loss(L2, G2, D2)
    dup_max_delta = max(abs(base[k] - after[k]) for k in base)
    idx_by, strata = support_signature_strata(Gp, Dp, Yp)
    rng = np.random.default_rng(7)
    both_cond = []
    for _ in range(50):
        idx, gid = stratified_subject_resample(idx_by, strata, rng)
        both_cond.append(all(set(Dp[idx[gid == u]].tolist()) == {0, 1} for u in np.unique(gid)))
    paired = dict(duplication_invariance_max_delta=float(dup_max_delta),
                  paired_bootstrap_all_intact=bool(all(both_cond)), n_paired_bootstraps=len(both_cond))

    # cross-split separability angle response (controlled directions)
    d6 = 6; u = np.zeros(d6); u[0] = 1.0
    A = (np.arange(1, 9, dtype=float))[:, None] * u[None, :]
    sep_response = {}
    for th in (90.0, 60.0, 30.0, 10.0):
        w = np.cos(np.radians(th)) * u + np.sin(np.radians(th)) * np.eye(d6)[1]
        sep_response[th] = round(cov_concept_angle(A, (w / np.linalg.norm(w))[:, None]), 2)

    # deterministic NAMED stage-seed derivations (sha256, NOT builtin hash())
    stages = ["analyze_source", "residual_cv", "residual_null", "geometry_null", "cov_bootstrap",
              "calibration", "oracle", "certify_robust", "separability_split"]
    named_seeds = {st: stage_seed(0, st) for st in stages}

    return dict(per_source=per_source, paired_invariants=paired,
                separability_angle_response=sep_response,
                named_stage_seeds_at_root0=named_seeds,
                seed_hash_algo="sha256", builtin_python_hash_used=False,
                invalid_null_frac_max=cfg.invalid_null_frac_max,
                method_changed_vs_p141=True)


def _env_info():
    import numpy, sklearn, scipy
    pkgs = dict(python=sys.version.split()[0], numpy=numpy.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__)
    h = hashlib.sha256(json.dumps(pkgs, sort_keys=True).encode()).hexdigest()[:16]
    return pkgs, h


def run_tests():
    out = {}
    for m in TEST_MODULES:
        r = subprocess.run([sys.executable, "-m", f"csc.tests.{m}"],
                           capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines()[-1:] or r.stderr.strip().splitlines()[-1:])
        out[m] = dict(passed=(r.returncode == 0), tail=tail)
    out["all_passed"] = all(v["passed"] for k, v in out.items() if k != "all_passed")
    return out


def calibration_null_bank(cfg: ProtocolConfig, seeds):
    """LODO ORACLE-SANITY bank (estimator sanity only). Goes through the SINGLE executor via
    nested_lodo(cfg). False-concept CONTROL is validated by synthetic_null_bank, not here."""
    n_stable = n_visible = n_amb = 0
    false_concept = 0
    valid = 0
    for s in seeds:
        src = make_source(SimConfig(seed=s), n_domains=8, concept_domains=3, seed=s)
        res = nested_lodo(src.Z, src.Y, src.D, cfg=cfg, group_ids=src.group_ids, seed=s)
        sc = res.scorecard
        n_stable += sc["n_stable"]; n_visible += sc["n_visible"]; n_amb += sc["n_ambiguous"]
        if sc["n_stable"]:
            false_concept += int(round((sc["false_concept_on_stable"] or 0.0) * sc["n_stable"]))
        valid += int(res.valid_bank)
    return dict(bank="CALIBRATION_NULL_BANK(oracle-sanity)", n_source_seeds=len(seeds),
                n_stable=n_stable, n_visible=n_visible, n_ambiguous=n_amb,
                false_concept_on_stable_count=false_concept,
                seeds_with_valid_null_bank=valid,
                note="estimator sanity only; false-concept control = synthetic_null_bank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=40)
    ap.add_argument("--n_dir_boot", type=int, default=120)
    ap.add_argument("--target_n_boot", type=int, default=120)
    ap.add_argument("--oracle_boot", type=int, default=150)
    ap.add_argument("--bank_seeds", type=int, default=4)
    ap.add_argument("--out", type=str, default="csc/results/audit.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = ProtocolConfig(n_boot=args.n_boot, n_dir_boot=args.n_dir_boot,
                         target_n_boot=args.target_n_boot, oracle_boot=args.oracle_boot)
    start = datetime.datetime.now().isoformat(timespec="seconds")
    env_pkgs, env_hash = _env_info()
    audited = _git("rev-parse", "HEAD")
    csc_clean = all("csc/results/" in ln or not ln.strip()
                    for ln in _git("status", "--porcelain", "csc/").splitlines())
    print(f"[audit] audited_code_commit={audited[:7]} manifest={cfg.hash()} clean={csc_clean}")

    print("[audit] self-tests ...")
    tests = run_tests()
    print(f"[audit] tests all_passed={tests['all_passed']}")
    # FAIL-FAST: a self-test failure invalidates the audit; do NOT spend the expensive banks.
    if not tests["all_passed"]:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(status="CSC-P1.4.2 AUDIT ABORTED — self-tests failed (fail-fast)",
                           audited_code_commit=audited, protocol_manifest_hash=cfg.hash(),
                           git_status_clean_csc=csc_clean, tests=tests, provenance_ok=False), f, indent=2)
        print(f"[audit] FAIL-FAST: self-tests failed -> wrote {args.out}, exit 1")
        sys.exit(1)

    print("[audit] run_synthetic via FROZEN PATH (DEVELOPMENT) ...")
    syn = run_syn(seeds=args.seeds, cfg=cfg, label="DEVELOPMENT", quiet=True)

    print("[audit] SYNTHETIC_NULL_BANK (generator-truth-stable) ...")
    syn_null = synthetic_null_bank(cfg, list(range(args.bank_seeds)),
                                   min_stable=max(2, args.bank_seeds))
    print("[audit] CALIBRATION_NULL_BANK (LODO oracle sanity) ...")
    null_bank = calibration_null_bank(cfg, list(range(args.bank_seeds)))
    print("[audit] OOD_POWER_BANK (generator-truth) ...")
    power_bank = ood_power_bank(cfg, list(range(args.bank_seeds)),
                                min_visible=max(2, args.bank_seeds))
    print("[audit] CSC-P1.4.2 diagnostics ...")
    p142 = p142_diagnostics(cfg, list(range(args.bank_seeds)))

    audit = dict(
        status="CSC-P1.4.2 DEVELOPMENT — frozen-path audit; NO FREEZE, NO CONFIRMATORY, NO P2. "
               "Inference procedure CHANGED vs P1.4.1 -> these are NEW-METHOD development numbers, "
               "NOT poolable with P1.4.1 for confidence intervals.",
        audited_code_commit=audited, audited_code_commit_short=audited[:7],
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_status_clean_csc=csc_clean,
        protocol_manifest=cfg.manifest(), protocol_manifest_hash=cfg.hash(),
        development_defaults=cfg.to_dict(),
        environment=env_pkgs, environment_hash=env_hash,
        exact_command=" ".join(sys.argv),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        start_time=start, end_time=datetime.datetime.now().isoformat(timespec="seconds"),
        seed_provenance="DEVELOPMENT (used during iteration; NOT confirmatory). A frozen "
                        "confirmatory run requires a separate, previously-unseen seed set.",
        tests=tests, run_synthetic=syn, synthetic_null_bank=syn_null,
        calibration_null_bank=null_bank, ood_power_bank=power_bank,
        p142_diagnostics=p142,
    )
    # provenance gate: an audit is only trustworthy if the tree is clean AND the audited commit
    # is HEAD. Recorded in the artifact; ENFORCED by a nonzero exit (fail-closed).
    head_match = (audited == _git("rev-parse", "HEAD"))
    audit["provenance_ok"] = bool(tests["all_passed"] and csc_clean and head_match)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[audit] wrote {args.out}")
    print(f"[audit] SUMMARY tests={tests['all_passed']} manifest={cfg.hash()[:12]} clean={csc_clean}")
    print(f"  run_synthetic PRIMARY any_forbidden_full={syn['any_forbidden_full_suite']}/{syn['seeds']} "
          f"(CP-UB {syn['exact_cp_ub_full_suite']:.3f})")
    print(f"  SYNTHETIC_NULL_BANK clusters_fail={syn_null['seed_cluster_failures']}/"
          f"{syn_null['n_source_clusters']} CP-UB {syn_null['false_concept_cp_upper_cluster']:.3f} "
          f"evaluable={syn_null['evaluable']} control_pass={syn_null['control_pass']}")
    print(f"  OOD_POWER_BANK power={power_bank['concept_power']} CP-LB "
          f"{power_bank['concept_power_cp_lower']:.3f} atlas={power_bank['atlas_availability']} "
          f"evaluable={power_bank['evaluable']} decomp={power_bank['binding_failure_decomposition']}")
    n_inv_src = sum(r["source_invalid_triggered"] for r in p142["per_source"])
    print(f"  P1.4.2 dup_invariance_delta={p142['paired_invariants']['duplication_invariance_max_delta']:.2e} "
          f"paired_intact={p142['paired_invariants']['paired_bootstrap_all_intact']} "
          f"sep_response={p142['separability_angle_response']} source_invalid={n_inv_src}/{len(p142['per_source'])}")
    if not audit["provenance_ok"]:
        print(f"[audit] FAIL-CLOSED: tests_all={tests['all_passed']} clean_csc={csc_clean} "
              f"head_match={head_match} -> exit 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
