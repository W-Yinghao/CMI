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
                   leakage_jobs=16, phase_b_cap=3) -> list:
    """The 18-job plan: for each fold a Phase-A (V100, records the store) and a Phase-B (CPU, replays +
    writes the artifact). Phase-A jobs run in PARALLEL (no inter-A dependency). Phase-B is submitter-managed
    and CAPPED at `phase_b_cap` concurrent via a rolling afterany:B_{i-cap} chain (in loso_plan order), each
    B additionally gated afterok on its own Phase-A. Phase-A carries OACI_CHAIN_PHASE_B=0 so it does NOT
    self-chain (the submitter owns the dependency graph)."""
    jobs = []
    specs = loso_plan(subjects=subjects)
    fold_ids = [s["fold_id"] for s in specs]
    for i, spec in enumerate(specs):
        out_root = os.path.join(loso_root, spec["fold_id"])
        common = {"OACI_TARGET_SUBJECT": str(spec["target"]), "OACI_MODEL_SEED": str(int(model_seed)),
                  "OACI_BOOTSTRAP_MODE": str(bootstrap_mode), "OACI_LEAKAGE_JOBS": str(int(leakage_jobs)),
                  "OACI_OUT_ROOT": out_root, "OACI_REPO": os.path.abspath(repo_root)}
        a_env = {**common, "OACI_CHAIN_PHASE_B": "0"}   # submitter manages Phase B; do not self-chain
        jobs.append({"kind": "phase_a", "fold_id": spec["fold_id"], "target": spec["target"],
                     "source_audit_subjects": spec["source_audit_subjects"],
                     "source_train_subjects": spec["source_train_subjects"], "deleted_cell": spec["deleted_cell"],
                     "partition": "V100", "gres": "gpu:1", "script": _PHASE_A_SCRIPT, "out_root": out_root,
                     "staging": os.path.join(out_root, "staging"), "env": a_env, "depends_on": None,
                     "expected_wall": _EXPECTED_V100})
        prior_b = fold_ids[i - phase_b_cap] if i >= phase_b_cap else None       # the B that must finish first
        dep_str = f"{spec['fold_id']}:phase_a (afterok)" + (
            f" + {prior_b}:phase_b (afterany, cap={phase_b_cap})" if prior_b else "")
        jobs.append({"kind": "phase_b", "fold_id": spec["fold_id"], "target": spec["target"],
                     "partition": "CPU", "gres": None, "script": _PHASE_B_SCRIPT,
                     "artifact_root": os.path.join(out_root, "artifacts"), "leakage_jobs": int(leakage_jobs),
                     "env": dict(common), "depends_on": dep_str, "depends_on_phase_a": spec["fold_id"],
                     "depends_on_prior_phase_b": prior_b, "phase_b_cap": int(phase_b_cap),
                     "expected_wall": _EXPECTED_CPU})
    return jobs


def validate_launch(loso_root, repo_root, datalake_root) -> None:
    if not _outside_repo(loso_root, repo_root):
        raise ValueError(f"LOSO artifact/store root must be OUTSIDE the repo: {loso_root}")
    if not os.path.isdir(os.path.join(datalake_root, "MNE-bnci-data")):
        raise ValueError(f"datalake missing BNCI at {datalake_root}")


def print_plan(jobs, *, loso_root, file=sys.stdout) -> None:
    a = [j for j in jobs if j["kind"] == "phase_a"]
    b = [j for j in jobs if j["kind"] == "phase_b"]
    cap = b[0]["phase_b_cap"] if b else 3
    print(f"LOSO staged plan: {len(a)} Phase-A (V100) + {len(b)} Phase-B (CPU) = {len(jobs)} jobs",
          file=file)
    print(f"  root (outside repo): {loso_root}", file=file)
    print(f"  Phase A: V100 gpu:1, 1 fold/job, all {len(a)} run in PARALLEL (no inter-A dependency)", file=file)
    print(f"  Phase B: CPU, {a[0]['env']['OACI_LEAKAGE_JOBS'] if a else 16} workers, submitter-managed, "
          f"capped {cap} concurrent (afterok:A + rolling afterany:B_i-{cap})", file=file)
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
    ap.add_argument("--phase-b-cap", type=int, default=3, help="max concurrent Phase-B (CPU) jobs")
    ap.add_argument("--launch", action="store_true",
                    help="actually sbatch all 9 Phase-A + 9 Phase-B jobs (default: dry-run)")
    args = ap.parse_args(argv)
    jobs = build_job_plan(args.loso_root, args.repo_root, model_seed=args.model_seed,
                          bootstrap_mode=args.bootstrap_mode, leakage_jobs=args.leakage_jobs,
                          phase_b_cap=args.phase_b_cap)
    print_plan(jobs, loso_root=args.loso_root)
    if not args.launch:
        print("\n(dry-run; pass --launch to submit all 9 Phase-A + 9 Phase-B jobs)")
        return 0
    validate_launch(args.loso_root, args.repo_root, args.datalake_root)
    import subprocess
    logs = "/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out"

    def _sbatch(job, extra):
        env = {**os.environ, **job["env"], "OACI_DATALAKE_ROOT": args.datalake_root}
        out = subprocess.run(["sbatch", "--parsable", f"--output={logs}", *extra, job["script"]],
                             env=env, capture_output=True, text=True)
        return out.stdout.strip(), out.stderr.strip()

    # Phase A: all in parallel (no dependency); each carries OACI_CHAIN_PHASE_B=0 so it does not self-chain.
    aids, bids = {}, {}
    for j in [x for x in jobs if x["kind"] == "phase_a"]:
        jid, err = _sbatch(j, [])
        aids[j["fold_id"]] = jid
        print(f"submitted {j['fold_id']} phase A (V100): {jid or ('FAILED: ' + err)}")
    # Phase B: submitter-managed, capped -- each depends afterok on its own Phase A and afterany on B_{i-cap}.
    for j in [x for x in jobs if x["kind"] == "phase_b"]:
        deps = []
        aid = aids.get(j["depends_on_phase_a"])
        if aid:
            deps.append(f"afterok:{aid}")
        prior = j.get("depends_on_prior_phase_b")
        if prior and bids.get(prior):
            deps.append(f"afterany:{bids[prior]}")
        extra = [f"--dependency={','.join(deps)}"] if deps else []
        jid, err = _sbatch(j, extra)
        bids[j["fold_id"]] = jid
        print(f"submitted {j['fold_id']} phase B (CPU): {jid or ('FAILED: ' + err)} deps={deps or 'none'}")
    n_ok = sum(1 for v in list(aids.values()) + list(bids.values()) if v)
    print(f"\nsubmitted {n_ok}/{len(jobs)} jobs; verify with: squeue -u $USER -o '%.10i %.20j %.8T %R'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
