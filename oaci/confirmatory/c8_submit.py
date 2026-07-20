"""BNCI2014_001 LOSO MULTI-SEED staged submitter (C8) — 9 targets × seeds [0,1,2] = 27 Phase-A (V100) + 27
Phase-B (CPU, native K1/K2 decisions) + 1 aggregation. Prints the full job graph; --launch submits it.

    BNCI2014-001 minimum-seed K1/K2 run. Seeds [0,1,2] meet the configured K2 minimum.
    This is not yet the full 5-seed manifest sweep.

Phase-A runs in parallel (C6-style GPU scheduling); Phase-B is submitter-managed and capped (afterok on its
own Phase-A + a rolling afterany:B_{i-cap}); the aggregation depends afterok on ALL 27 Phase-B. Every
Phase-B carries OACI_COMPUTE_DECISIONS=1 (2000-permutation K1). All artifacts/stores live OUTSIDE the repo.
"""
from __future__ import annotations

import argparse
import os
import sys

from .loso_plan import SUBJECTS, loso_plan
from .submit import (_EXPECTED_CPU, _EXPECTED_V100, _PHASE_A_SCRIPT, _PHASE_B_SCRIPT, _outside_repo)

_AGG_SCRIPT = "oaci/slurm_bnci001_loso_c8_aggregate.sh"
_LABEL = ("BNCI2014-001 minimum-seed K1/K2 run. Seeds [0,1,2] meet the configured K2 minimum. "
          "This is not yet the full 5-seed manifest sweep.")


def build_c8_job_plan(loso_root, repo_root, *, seeds=(0, 1, 2), bootstrap_mode="full", subjects=SUBJECTS,
                      leakage_jobs=16, phase_b_cap=3, k1_permutations=2000, bootstrap_budgets=None) -> list:
    """27 Phase-A + 27 Phase-B + 1 aggregation. Fold-run order = (seed, target); the Phase-B cap is a rolling
    afterany:B_{i-cap}. Each fold-run gets a DISTINCT out root ``<loso_root>/seed-<s>/target-00N/``."""
    specs = loso_plan(subjects=subjects)
    order = [(int(s), spec) for s in seeds for spec in specs]        # 27 (seed, target) pairs
    fold_uids = [f"seed-{s}/{spec['fold_id']}" for s, spec in order]
    repo_abs = os.path.abspath(repo_root)
    a_jobs, b_jobs = [], []
    for i, (s, spec) in enumerate(order):
        out_root = os.path.join(loso_root, f"seed-{s}", spec["fold_id"])
        uid = fold_uids[i]
        common = {"OACI_TARGET_SUBJECT": str(spec["target"]), "OACI_MODEL_SEED": str(s),
                  "OACI_BOOTSTRAP_MODE": str(bootstrap_mode), "OACI_LEAKAGE_JOBS": str(int(leakage_jobs)),
                  "OACI_OUT_ROOT": out_root, "OACI_REPO": repo_abs}
        a_jobs.append({"kind": "phase_a", "fold_uid": uid, "seed": s, "target": spec["target"],
                       "source_audit_subjects": spec["source_audit_subjects"],
                       "source_train_subjects": spec["source_train_subjects"], "deleted_cell": spec["deleted_cell"],
                       "partition": "V100", "gres": "gpu:1", "script": _PHASE_A_SCRIPT, "out_root": out_root,
                       "staging": os.path.join(out_root, "staging"),
                       "env": {**common, "OACI_CHAIN_PHASE_B": "0"}, "depends_on": None,
                       "expected_wall": _EXPECTED_V100})
        prior_b = fold_uids[i - phase_b_cap] if i >= phase_b_cap else None
        dep = f"{uid}:phase_a (afterok)" + (f" + {prior_b}:phase_b (afterany, cap={phase_b_cap})" if prior_b else "")
        b_jobs.append({"kind": "phase_b", "fold_uid": uid, "seed": s, "target": spec["target"],
                       "partition": "CPU", "gres": None, "script": _PHASE_B_SCRIPT,
                       "artifact_root": os.path.join(out_root, "artifacts"), "leakage_jobs": int(leakage_jobs),
                       "env": {**common, "OACI_COMPUTE_DECISIONS": "1"}, "depends_on": dep,
                       "depends_on_phase_a": uid, "depends_on_prior_phase_b": prior_b, "phase_b_cap": int(phase_b_cap),
                       "k1_permutations": int(k1_permutations), "bootstrap_budgets": bootstrap_budgets,
                       "expected_wall": "~11h (audit + 2000-perm K1)"})
    agg = {"kind": "aggregation", "partition": "CPU", "gres": None, "script": _AGG_SCRIPT,
           "depends_on": f"all {len(b_jobs)} phase_b (afterok)", "depends_on_all_phase_b": list(fold_uids),
           "loso_root": loso_root, "seeds": list(int(s) for s in seeds),
           "report": ["oaci/reports/C8_BNCI001_LOSO_SEEDS012_K1K2.md",
                      "oaci/reports/C8_BNCI001_LOSO_SEEDS012_K1K2.json"], "expected_wall": "~20m"}
    return a_jobs + b_jobs + [agg]


def validate_c8_launch(loso_root, repo_root, datalake_root) -> None:
    if not _outside_repo(loso_root, repo_root):
        raise ValueError(f"LOSO artifact/store root must be OUTSIDE the repo: {loso_root}")
    if not os.path.isdir(os.path.join(datalake_root, "MNE-bnci-data")):
        raise ValueError(f"datalake missing BNCI at {datalake_root}")
    if os.environ.get("OACI_ALLOW_DIRTY") != "1":
        import subprocess
        dirty = subprocess.run(["git", "-C", os.path.abspath(repo_root), "status", "--porcelain", "--", "oaci"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            raise ValueError("repo oaci/ tree is dirty; commit before launching (folds must share one commit)")
    a = [j for j in build_c8_job_plan(loso_root, repo_root) if j["kind"] == "phase_a"]
    uids = [j["fold_uid"] for j in a]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate (target, seed) fold-run in the plan")


def print_c8_plan(jobs, *, loso_root, file=sys.stdout) -> None:
    a = [j for j in jobs if j["kind"] == "phase_a"]
    b = [j for j in jobs if j["kind"] == "phase_b"]
    agg = next(j for j in jobs if j["kind"] == "aggregation")
    cap = b[0]["phase_b_cap"] if b else 3
    print(_LABEL, file=file)
    print(f"C8 staged plan: {len(a)} Phase-A (V100) + {len(b)} Phase-B (CPU) + 1 aggregation = {len(jobs)} jobs",
          file=file)
    print(f"  root (outside repo): {loso_root}", file=file)
    print(f"  Phase A: V100 gpu:1, all {len(a)} parallel (C6-style GPU scheduling; QOS GRES throttles)", file=file)
    print(f"  Phase B: CPU, submitter-managed, capped {cap} concurrent (afterok:A + rolling afterany:B_i-{cap}); "
          f"OACI_COMPUTE_DECISIONS=1", file=file)
    print(f"  Aggregation: depends afterok on ALL {len(b)} Phase-B -> {agg['report'][0]} (+ .json)", file=file)
    bud = b[0]["bootstrap_budgets"] if b else None
    for j in b:
        a_match = next(x for x in a if x["fold_uid"] == j["fold_uid"])
        print(f"  [{j['fold_uid']}] target={j['target']} seed={j['seed']} "
              f"audit={a_match['source_audit_subjects']} train={a_match['source_train_subjects']} "
              f"deleted={a_match['deleted_cell']['domain_id']}|{a_match['deleted_cell']['class_name']}", file=file)
        print(f"       A store={a_match['staging']}  V100~{a_match['expected_wall']}", file=file)
        print(f"       B artifact={j['artifact_root']}  depends={j['depends_on']}  CPU {j['expected_wall']}",
              file=file)
        print(f"       K1 permutations={j['k1_permutations']}  bootstrap_budgets={bud}", file=file)


def _bootstrap_budgets(protocol_path):
    from .schema import load_confirmatory
    p = load_confirmatory(protocol_path)
    pr, ev = p.block("probe"), p.block("evaluation")
    return {"selection_bootstrap": pr.get("selection_bootstrap"), "audit_bootstrap": pr.get("audit_bootstrap"),
            "paired_bootstrap": ev.get("paired_bootstrap")}, int(p.block("k1")["n_permutations"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.c8_submit")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--datalake-root", required=True)
    ap.add_argument("--protocol", default="oaci/protocol/confirmatory_v2.yaml")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--bootstrap-mode", choices=("full", "validation"), default="full")
    ap.add_argument("--leakage-jobs", type=int, default=16)
    ap.add_argument("--phase-b-cap", type=int, default=3)
    ap.add_argument("--launch", action="store_true", help="actually submit the 27+27+1 jobs (default: dry-run)")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    budgets, k1_perm = _bootstrap_budgets(args.protocol)
    jobs = build_c8_job_plan(args.loso_root, args.repo_root, seeds=seeds, bootstrap_mode=args.bootstrap_mode,
                             leakage_jobs=args.leakage_jobs, phase_b_cap=args.phase_b_cap,
                             k1_permutations=k1_perm, bootstrap_budgets=budgets)
    print_c8_plan(jobs, loso_root=args.loso_root)
    if not args.launch:
        print("\n(dry-run; validate paths/clean-tree/datalake, then pass --launch)")
        return 0
    validate_c8_launch(args.loso_root, args.repo_root, args.datalake_root)
    import subprocess
    logs = "/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out"

    def _sb(job, extra):
        env = {**os.environ, **job["env"], "OACI_DATALAKE_ROOT": args.datalake_root}
        out = subprocess.run(["sbatch", "--parsable", f"--output={logs}", *extra, job["script"]],
                             env=env, capture_output=True, text=True)
        return out.stdout.strip(), out.stderr.strip()

    aids, bids = {}, {}
    for j in [x for x in jobs if x["kind"] == "phase_a"]:
        jid, err = _sb(j, [])
        aids[j["fold_uid"]] = jid
        print(f"submitted {j['fold_uid']} phase A (V100): {jid or ('FAILED: ' + err)}")
    for j in [x for x in jobs if x["kind"] == "phase_b"]:
        deps = []
        if aids.get(j["depends_on_phase_a"]):
            deps.append(f"afterok:{aids[j['depends_on_phase_a']]}")
        if j["depends_on_prior_phase_b"] and bids.get(j["depends_on_prior_phase_b"]):
            deps.append(f"afterany:{bids[j['depends_on_prior_phase_b']]}")
        jid, err = _sb(j, [f"--dependency={','.join(deps)}"] if deps else [])
        bids[j["fold_uid"]] = jid
        print(f"submitted {j['fold_uid']} phase B (CPU): {jid or ('FAILED: ' + err)} deps={deps or 'none'}")
    print(f"\nsubmitted {sum(1 for v in list(aids.values()) + list(bids.values()) if v)}/{len(jobs) - 1} A+B jobs; "
          f"run {_AGG_SCRIPT} after all Phase-B complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
