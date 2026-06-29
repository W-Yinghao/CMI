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
                "test_paired_and_accounting", "test_p143_contracts", "test_p144_contracts",
                "test_p145_contracts", "test_confirmatory"]


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def contract_diagnostics(cfg, seeds):
    """CSC-P1.4.5 inference-contract diagnostics retained in the audit artifact: per-source null
    invalid counts + conservative-p, source-status (all stages), concept-attribution stability,
    the FULL residual_decoder_test T/p/status/certificate invariance under epoch duplication, and
    the deterministic NAMED stage-seed derivations."""
    from csc.certificate import analyze_source
    from csc.certificate.residual_test import (stage_seed, residual_decoder_test,
                                               _subject_fold_assignment)
    from csc.certificate.atlas import cov_concept_angle
    from csc.sim.shift_simulator import make_source, make_paired_subjects

    per_source = []
    for s in seeds:
        src = make_source(SimConfig(seed=s), n_domains=8, concept_domains=3, seed=s)
        sa = analyze_source(src.Z, src.Y, src.D, n_boot=cfg.n_boot, n_dir_boot=cfg.n_dir_boot,
                            alpha=cfg.alpha, var_keep=cfg.var_keep, C=cfg.C,
                            n_folds=cfg.source_cv_folds, invalid_frac_max=cfg.invalid_null_frac_max,
                            concept_eigengap_min=cfg.concept_eigengap_min,
                            concept_stability_max_deg=cfg.concept_stability_max_deg,
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
            source_invalid_triggered=(sa.source_status != "VALID"),   # covers ALL stages
            concept_disagreement_deg=float(d.get("concept_disagreement_deg", float("nan"))),
            has_dominant_concept_dir=bool(d.get("has_dominant_concept_dir", False)),
            concept_attribution_unstable=bool(d.get("concept_attribution_unstable", False)),
            concept_stable=bool(d.get("concept_stable", False))))

    # FULL residual_decoder_test invariance under within-cell epoch duplication (CSC-P1.4.4 #5):
    # record T, p, status AND the fold-assignment hash before & after -- not just the loss helper.
    # Use a SUBJECT-level source (label_unit='subject'): duplicating a subject's epochs is redundant
    # at every stage (per-cell loss, per-fold weighted scaler, per-subject null), so T, p AND status
    # are ALL invariant. (For label_unit='trial' epochs ARE the unit, so duplication adds null draws
    # and p need not be invariant -- a different, correct estimand.)
    sd = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    Zp, Yp, Dp, Gp = sd.Z, sd.Y, sd.D, sd.group_ids
    def _fold_hash(G, Y):
        # hash the SUBJECT->fold MAP (sorted by subject), which is invariant to row multiplicity --
        # the per-row fold vector changes length under duplication and is not comparable.
        f, _, _ = _subject_fold_assignment(G, Y, cfg.source_cv_folds, stage_seed(0, "residual_cv"))
        G = np.asarray(G)
        smap = np.array([[s, f[G == s][0]] for s in np.unique(G)])
        return hashlib.sha256(smap.tobytes()).hexdigest()[:12]
    r1 = residual_decoder_test(Zp, Yp, Dp, n_folds=cfg.source_cv_folds, n_boot=cfg.n_boot,
                               group_ids=Gp, C=cfg.C, label_unit="subject", seed=0)
    dup = (Gp == Gp[0])                                    # duplicate one whole subject's epochs
    Z2 = np.concatenate([Zp, Zp[dup]]); Y2 = np.concatenate([Yp, Yp[dup]])
    D2 = np.concatenate([Dp, Dp[dup]]); G2 = np.concatenate([Gp, Gp[dup]])
    r2 = residual_decoder_test(Z2, Y2, D2, n_folds=cfg.source_cv_folds, n_boot=cfg.n_boot,
                               group_ids=G2, C=cfg.C, label_unit="subject", seed=0)
    full_T = dict(T_before=float(r1.T), T_after=float(r2.T), abs_delta_T=abs(float(r1.T - r2.T)),
                  p_before=float(r1.p_value), p_after=float(r2.p_value),
                  abs_delta_p=abs(float(r1.p_value - r2.p_value)),
                  status_before=r1.status, status_after=r2.status,
                  significant_before=bool(r1.significant), significant_after=bool(r2.significant),
                  fold_hash_before=_fold_hash(Gp, Yp), fold_hash_after=_fold_hash(G2, Y2))

    # FULL-PROTOCOL invariance (CSC-P1.4.5 audit): run the ENTIRE execute_protocol before & after
    # duplicating one SOURCE subject's epochs, and compare every downstream output -- atlas subspace
    # hashes, source_status, tau_detect/label, cov_stable AND the final robust certificate.
    from csc.protocol import execute_protocol, ProtocolConfig
    from csc.sim.shift_simulator import make_target
    def _atlas_hash(a):
        b = b"".join(np.round(x, 6).tobytes() for x in (a.cov_dirs, a.concept_dirs, a.label_dirs,
                                                        a.pooled_mean))
        return hashlib.sha256(b).hexdigest()[:12]
    pcfg = ProtocolConfig(n_boot=cfg.n_boot, n_dir_boot=cfg.n_dir_boot, target_n_boot=cfg.target_n_boot)
    tgt = make_target("covariate", SimConfig(seed=0), geom=sd.geom, seed=100)
    tcid = np.zeros(len(tgt.Z), int)
    o1 = execute_protocol(sd.Z, sd.Y, sd.D, tgt.Z, pcfg, src_group_ids=Gp,
                          tgt_group_ids=tgt.group_ids, tgt_condition_ids=tcid, seed=0)
    o2 = execute_protocol(Z2, Y2, D2, tgt.Z, pcfg, src_group_ids=G2,
                          tgt_group_ids=tgt.group_ids, tgt_condition_ids=tcid, seed=0)
    full_protocol = dict(
        atlas_hash_before=_atlas_hash(o1["analysis"].atlas), atlas_hash_after=_atlas_hash(o2["analysis"].atlas),
        source_status_before=o1["analysis"].source_status, source_status_after=o2["analysis"].source_status,
        tau_detect_before=round(o1["tau_detect"], 8), tau_detect_after=round(o2["tau_detect"], 8),
        tau_label_before=round(o1["tau_label"], 8), tau_label_after=round(o2["tau_label"], 8),
        cov_stable_before=bool(o1["analysis"].cov_stable), cov_stable_after=bool(o2["analysis"].cov_stable),
        certificate_before=o1["certificate"].state, certificate_after=o2["certificate"].state)

    # principal-angle primitive response (controlled directions): rank-1 cov along u, concept @theta
    d6 = 6; u = np.zeros(d6); u[0] = 1.0
    A = (np.arange(1, 9, dtype=float))[:, None] * u[None, :]
    angle_response = {}
    for th in (90.0, 60.0, 30.0, 10.0):
        w = np.cos(np.radians(th)) * u + np.sin(np.radians(th)) * np.eye(d6)[1]
        angle_response[th] = round(cov_concept_angle(A, (w / np.linalg.norm(w))[:, None]), 2)

    # ACTUAL executor-derived stage seeds (CSC-P1.4.5 audit): the executor derives a per-stage seed
    # from the ROOT via ctx.seed(stage); the source-internal stages (residual/geometry/cov) are then
    # derived AGAIN from the analyze_source seed -- record the real chain, not a flat root-level calc.
    root = 0
    s_analyze = stage_seed(root, "analyze_source")
    named_seeds = {"root": root,
                   "analyze_source": s_analyze,
                   "calibrate_thresholds": stage_seed(root, "calibrate_thresholds"),
                   "certify_robust": stage_seed(root, "certify_robust"),
                   "analyze_source/residual_cv": stage_seed(s_analyze, "residual_cv"),
                   "analyze_source/residual_null": stage_seed(s_analyze, "residual_null"),
                   "analyze_source/geometry_null": stage_seed(s_analyze, "geometry_null"),
                   "analyze_source/cov_bootstrap": stage_seed(s_analyze, "cov_bootstrap"),
                   "analyze_source/concept_stability": stage_seed(s_analyze, "concept_stability")}

    return dict(per_source=per_source, full_residual_test_duplication_invariance=full_T,
                full_protocol_duplication_invariance=full_protocol,
                principal_angle_response=angle_response,
                named_stage_seeds_at_root0=named_seeds,
                seed_hash_algo="sha256", builtin_python_hash_used=False,
                invalid_null_frac_max=cfg.invalid_null_frac_max,
                concept_eigengap_min=cfg.concept_eigengap_min,
                concept_stability_max_deg=cfg.concept_stability_max_deg,
                method_changed_vs_prior_round=True)


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
            json.dump(dict(status="CSC-P1.4.5 AUDIT ABORTED — self-tests failed (fail-fast)",
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
    print("[audit] CSC-P1.4.5 contract diagnostics ...")
    diag = contract_diagnostics(cfg, list(range(args.bank_seeds)))

    audit = dict(
        status="CSC-P1.4.5 DEVELOPMENT — frozen-path audit; NO P1.5 PRODUCTION SWEEP, NO FREEZE, "
               "NO CONFIRMATORY, NO P2. Inference procedure CHANGED vs P1.4.4 -> these are "
               "NEW-METHOD development numbers, NOT poolable with prior rounds for confidence intervals.",
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
        contract_diagnostics=diag,
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
    n_inv_src = sum(r["source_invalid_triggered"] for r in diag["per_source"])
    ft = diag["full_residual_test_duplication_invariance"]
    print(f"  P1.4.5 full-T dup: dT={ft['abs_delta_T']:.2e} dp={ft['abs_delta_p']:.2e} "
          f"status {ft['status_before']}=={ft['status_after']} fold_hash "
          f"{ft['fold_hash_before']}=={ft['fold_hash_after']}; angle_resp="
          f"{diag['principal_angle_response']} source_invalid={n_inv_src}/{len(diag['per_source'])}")
    if not audit["provenance_ok"]:
        print(f"[audit] FAIL-CLOSED: tests_all={tests['all_passed']} clean_csc={csc_clean} "
              f"head_match={head_match} -> exit 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
