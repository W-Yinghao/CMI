"""
csc.run_audit — one machine-readable audit artifact per commit (CSC-P1.1).

Produces a single JSON capturing everything a reviewer needs to verify a (NON-frozen,
DEVELOPMENT) state without re-running:

  commit hash, frozen-config hash, conda env, seed list, all-test pass/fail, per-shift
  confusion counts, concept power, covariate-compatible coverage, abstention/invalid rates,
  EXACT cluster-level Clopper-Pearson bound, LODO per-fold thresholds, oracle class counts
  and valid_bank status.

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

from csc.run_synthetic import run as run_syn
from csc.sim.shift_simulator import SimConfig, make_source
from csc.calibration.lodo import nested_lodo, VISIBLE_CONCEPT, COVARIATE_STABLE, AMBIGUOUS
from csc.certificate import CertifierConfig
from csc.certificate.atlas import MIN_PRINCIPAL_ANGLE_DEG

TEST_MODULES = ["test_design_and_pairs", "test_validity_gate",
                "test_null_calibration", "test_power"]


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _config_hash():
    cfg = CertifierConfig()
    payload = json.dumps(dict(
        tau_detect=cfg.tau_detect, tau_label=cfg.tau_label,
        tau_resid=cfg.tau_resid, tau_margin=cfg.tau_margin,
        min_principal_angle_deg=MIN_PRINCIPAL_ANGLE_DEG,
    ), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16], json.loads(payload)


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
        out[m] = dict(passed=(r.returncode == 0),
                      tail=r.stdout.strip().splitlines()[-1:] or r.stderr.strip().splitlines()[-1:])
    out["all_passed"] = all(v["passed"] for k, v in out.items() if k != "all_passed")
    return out


def run_lodo_audit(seeds, n_boot, n_dir_boot, oracle_boot):
    """Mechanism-group-out LODO per seed: hold out the concept-domain group (oracle should
    see VISIBLE_CONCEPT) and covariate halves (should be STABLE/AMBIGUOUS). Aggregate."""
    folds_records = []
    verdict_counts = {VISIBLE_CONCEPT: 0, COVARIATE_STABLE: 0, AMBIGUOUS: 0}
    valid_banks = 0
    for s in seeds:
        cfg = SimConfig(seed=s)
        src = make_source(cfg, n_domains=8, concept_domains=3, seed=s)
        concept = [i for i, d in enumerate(src.domains) if d.c != 0]
        cov = [i for i, d in enumerate(src.domains) if d.c == 0]
        folds = [concept, cov[:len(cov) // 2], cov[len(cov) // 2:]]
        res = nested_lodo(src.Z, src.Y, src.D, folds=folds, n_boot=n_boot,
                          n_dir_boot=n_dir_boot, oracle_boot=oracle_boot, seed=s)
        valid_banks += int(res.valid_bank)
        for r in res.records:
            verdict_counts[r.oracle.verdict] = verdict_counts.get(r.oracle.verdict, 0) + 1
            folds_records.append(dict(
                seed=s, fold=r.fold, cert=r.cert_state, tau_detect=round(r.tau_detect, 4),
                concept_atlas=r.concept_atlas, oracle_verdict=r.oracle.verdict,
                oracle_point=round(r.oracle.point, 4),
                oracle_ci=[round(r.oracle.lb, 4), round(r.oracle.ub, 4)]))
    return dict(n_seeds=len(seeds), valid_banks=valid_banks,
                oracle_verdict_counts=verdict_counts, fold_records=folds_records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=40)
    ap.add_argument("--n_dir_boot", type=int, default=120)
    ap.add_argument("--oracle_boot", type=int, default=150)
    ap.add_argument("--lodo_seeds", type=int, default=4)
    ap.add_argument("--out", type=str, default="csc/results/audit.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    start_time = datetime.datetime.now().isoformat(timespec="seconds")
    cfg_hash, cfg_payload = _config_hash()
    env_pkgs, env_hash = _env_info()
    # the METHOD commit being audited (determinable from Git history); NOT the later commit
    # that will CONTAIN this audit.json -- recording that would be a self-reference.
    audited_commit = _git("rev-parse", "HEAD")
    # clean-check IGNORES csc/results/ (the audit's own output dir) -- otherwise the artifact
    # it is about to write would mark its own tree dirty.
    _status = [ln for ln in _git("status", "--porcelain", "csc/").splitlines()
               if ln.strip() and "csc/results/" not in ln]
    csc_clean = (len(_status) == 0)
    print(f"[audit] audited_code_commit={audited_commit[:7]} config_hash={cfg_hash} "
          f"csc_clean={csc_clean}")

    print("[audit] running self-tests ...")
    tests = run_tests()
    print(f"[audit] tests all_passed={tests['all_passed']}")

    print("[audit] run_synthetic (DEVELOPMENT) ...")
    syn = run_syn(seeds=args.seeds, n_boot=args.n_boot, n_dir_boot=args.n_dir_boot,
                  label="DEVELOPMENT", quiet=True)

    print("[audit] nested LODO (mechanism-group-out) ...")
    lodo = run_lodo_audit(list(range(args.lodo_seeds)), args.n_boot, args.n_dir_boot,
                          args.oracle_boot)

    audit = dict(
        status="CSC-P1.2 DEVELOPMENT — AUDIT; NO FREEZE, NO P2",
        audited_code_commit=audited_commit,
        audited_code_commit_short=audited_commit[:7],
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_status_clean_csc=csc_clean,
        config_hash=cfg_hash, frozen_config=cfg_payload,
        environment=env_pkgs, environment_hash=env_hash,
        exact_command=" ".join(sys.argv),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        start_time=start_time,
        end_time=datetime.datetime.now().isoformat(timespec="seconds"),
        seed_provenance="DEVELOPMENT (used during iteration; NOT confirmatory). A frozen "
                        "confirmatory run requires a separate, previously-unseen seed set.",
        tests=tests, run_synthetic=syn, lodo=lodo,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[audit] wrote {args.out}")
    print(f"[audit] SUMMARY: tests={tests['all_passed']} "
          f"false_cert_total={syn['false_cert_total']} "
          f"cluster_bound={syn['exact_cp_upper_bound_cluster']:.3f} "
          f"power={syn['power_visible_concept']:.2f} "
          f"cov_cov={syn['covariate_compatible_coverage']:.2f} "
          f"lodo_valid_banks={lodo['valid_banks']}/{lodo['n_seeds']} "
          f"oracle={lodo['oracle_verdict_counts']}")


if __name__ == "__main__":
    main()
