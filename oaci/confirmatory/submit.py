"""BNCI2014_001 LOSO staged submitter (C6b) -- builds the 18-job plan (9 Phase-A GPU + 9 Phase-B CPU) and
prints it before submitting. Each Phase-A (V100) job records a fold's store and chains its matching Phase-B
(CPU) job; the submitter only sbatches the nine Phase-A jobs. All artifacts/stores live OUTSIDE the repo.

    python -m oaci.confirmatory.submit --loso-root <outside-repo> --repo-root <repo> \
        --datalake-root <datalake> [--launch]      # default: dry-run (print only)
"""
from __future__ import annotations

import argparse
import os
import sys

from .loso_plan import SUBJECTS, loso_plan

_PHASE_A_SCRIPT = "oaci/slurm_confirmatory_staged_a.sh"
_PHASE_B_SCRIPT = "oaci/slurm_confirmatory_staged_b.sh"
_EXPECTED_V100 = "~1h40m"
_EXPECTED_CPU = "~5h44m"


def _outside_repo(path, repo_root) -> bool:
    p, r = os.path.abspath(path), os.path.abspath(repo_root)
    return not (p == r or p.startswith(r + os.sep))


def build_job_plan(loso_root, repo_root, *, model_seed=0, bootstrap_mode="full", subjects=SUBJECTS,
                   leakage_jobs=16) -> list:
    """The 18-job plan: for each fold a Phase-A (V100, records the store + chains Phase-B) and a Phase-B
    (CPU, replays + writes the artifact, depends on its Phase-A)."""
    jobs = []
    for spec in loso_plan(subjects=subjects):
        out_root = os.path.join(loso_root, spec["fold_id"])
        env = {"OACI_TARGET_SUBJECT": str(spec["target"]), "OACI_MODEL_SEED": str(int(model_seed)),
               "OACI_BOOTSTRAP_MODE": str(bootstrap_mode), "OACI_LEAKAGE_JOBS": str(int(leakage_jobs)),
               "OACI_OUT_ROOT": out_root}
        jobs.append({"kind": "phase_a", "fold_id": spec["fold_id"], "target": spec["target"],
                     "source_audit_subjects": spec["source_audit_subjects"],
                     "source_train_subjects": spec["source_train_subjects"], "deleted_cell": spec["deleted_cell"],
                     "partition": "V100", "gres": "gpu:1", "script": _PHASE_A_SCRIPT, "out_root": out_root,
                     "staging": os.path.join(out_root, "staging"), "env": env, "depends_on": None,
                     "expected_wall": _EXPECTED_V100})
        jobs.append({"kind": "phase_b", "fold_id": spec["fold_id"], "target": spec["target"],
                     "partition": "CPU", "gres": None, "script": _PHASE_B_SCRIPT,
                     "artifact_root": os.path.join(out_root, "artifacts"), "leakage_jobs": int(leakage_jobs),
                     "depends_on": f"{spec['fold_id']}:phase_a (chained by Phase A)",
                     "expected_wall": _EXPECTED_CPU})
    return jobs


def validate_launch(loso_root, repo_root, datalake_root) -> None:
    if not _outside_repo(loso_root, repo_root):
        raise ValueError(f"LOSO artifact/store root must be OUTSIDE the repo: {loso_root}")
    if not os.path.isdir(os.path.join(datalake_root, "MNE-bnci-data")):
        raise ValueError(f"datalake missing BNCI at {datalake_root}")


def print_plan(jobs, *, loso_root, file=sys.stdout) -> None:
    a = [j for j in jobs if j["kind"] == "phase_a"]
    print(f"LOSO staged plan: {len(a)} Phase-A (V100) + {len(jobs) - len(a)} Phase-B (CPU) = {len(jobs)} jobs",
          file=file)
    print(f"  root (outside repo): {loso_root}", file=file)
    print(f"  Phase A: V100 gpu:1, 1 fold/job, concurrency 1 (V100 GRES); each chains its Phase B", file=file)
    print(f"  Phase B: CPU, {a[0]['env']['OACI_LEAKAGE_JOBS'] if a else 16} workers, depends on its Phase A", file=file)
    for j in jobs:
        if j["kind"] == "phase_a":
            print(f"  [A] {j['fold_id']} target={j['target']} audit={j['source_audit_subjects']} "
                  f"train={j['source_train_subjects']} deleted={j['deleted_cell']['domain_id']}|"
                  f"{j['deleted_cell']['class_name']} store={j['staging']} V100~{j['expected_wall']}", file=file)
        else:
            print(f"  [B] {j['fold_id']} artifact={j['artifact_root']} depends={j['depends_on']} "
                  f"CPU~{j['expected_wall']}", file=file)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.submit")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--datalake-root", required=True)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--bootstrap-mode", choices=("full", "validation"), default="full")
    ap.add_argument("--leakage-jobs", type=int, default=16)
    ap.add_argument("--launch", action="store_true", help="actually sbatch the 9 Phase-A jobs (default: dry-run)")
    args = ap.parse_args(argv)
    jobs = build_job_plan(args.loso_root, args.repo_root, model_seed=args.model_seed,
                          bootstrap_mode=args.bootstrap_mode, leakage_jobs=args.leakage_jobs)
    print_plan(jobs, loso_root=args.loso_root)
    if not args.launch:
        print("\n(dry-run; pass --launch to submit the 9 Phase-A jobs)")
        return 0
    validate_launch(args.loso_root, args.repo_root, args.datalake_root)
    import subprocess
    logs = "/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out"
    for j in [x for x in jobs if x["kind"] == "phase_a"]:
        env = {**os.environ, **j["env"], "OACI_DATALAKE_ROOT": args.datalake_root}
        out = subprocess.run(["sbatch", "--parsable", f"--output={logs}", j["script"]],
                             env=env, capture_output=True, text=True)
        print(f"submitted {j['fold_id']} phase A: {out.stdout.strip() or out.stderr.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
