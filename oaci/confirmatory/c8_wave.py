"""C8 THROTTLED WAVE controller — trickle the 27 (seed, target) fold-runs into free submit-cap slots.

The cluster caps total submitted (pending+running) jobs per user (QOSMaxSubmitJobPerUserLimit ~30), which is
shared with the user's OTHER work (e.g. the TOS pilot). So C8 cannot fire 54 jobs at once. This controller
runs as ONE persistent CPU job that repeatedly: counts my current jobs, and if there is room (under an
overall cap AND a C8-specific cap) submits the next Phase-A fold-run. Each Phase-A SELF-CHAINS its
decision-enabled Phase-B (OACI_CHAIN_PHASE_B=1 + OACI_COMPUTE_DECISIONS=1), so the controller only throttles
Phase-A submission — the running Phase-B jobs count toward my job total and thus self-limit the next wave.

Resume-safe: a fold-run whose committed artifact already exists is skipped. After all 27 Phase-A are
submitted the controller exits; the self-chained Phase-B jobs finish on their own, then run the aggregation.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

from .loso_plan import SUBJECTS, loso_plan

_PHASE_A_SCRIPT = "oaci/slurm_confirmatory_staged_a.sh"
_LOGS = "/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out"


def c8_fold_runs(loso_root, repo_root, *, seeds, subjects=SUBJECTS, leakage_jobs=16, bootstrap_mode="full") -> list:
    """The 27 fold-runs in (seed, target) order, each with the Phase-A env that self-chains a decision Phase-B."""
    repo_abs = os.path.abspath(repo_root)
    out = []
    for s in (int(x) for x in seeds):
        for spec in loso_plan(subjects=subjects):
            out_root = os.path.join(loso_root, f"seed-{s}", spec["fold_id"])
            out.append({"fold_uid": f"seed-{s}/{spec['fold_id']}", "seed": s, "target": int(spec["target"]),
                        "out_root": out_root,
                        "env": {"OACI_TARGET_SUBJECT": str(spec["target"]), "OACI_MODEL_SEED": str(s),
                                "OACI_BOOTSTRAP_MODE": str(bootstrap_mode), "OACI_LEAKAGE_JOBS": str(int(leakage_jobs)),
                                "OACI_OUT_ROOT": out_root, "OACI_REPO": repo_abs,
                                "OACI_CHAIN_PHASE_B": "1", "OACI_COMPUTE_DECISIONS": "1"}})
    return out


def slots_to_submit(my_total, my_c8, max_total, max_c8, n_remaining) -> int:
    """How many Phase-A to submit now: bounded by the overall cap headroom, the C8-specific cap, and the
    remaining fold-runs. Never negative."""
    return max(0, min(int(n_remaining), int(max_total) - int(my_total), int(max_c8) - int(my_c8)))


def has_committed_artifact(out_root) -> bool:
    return len(glob.glob(os.path.join(out_root, "artifacts", "*", "COMMITTED.json"))) >= 1


# ---- live cluster side-effects (mocked in tests) ----
def _my_total(user) -> int:
    r = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i"], capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def _my_c8(user) -> int:
    r = subprocess.run(["squeue", "-u", user, "-h", "-o", "%j"], capture_output=True, text=True)
    return sum(1 for l in r.stdout.splitlines() if l.strip().startswith("oaci-staged-"))


def _sbatch_a(fold, datalake_root) -> str:
    env = {**os.environ, **fold["env"], "OACI_DATALAKE_ROOT": datalake_root}
    out = subprocess.run(["sbatch", "--parsable", f"--output={_LOGS}", _PHASE_A_SCRIPT],
                         env=env, capture_output=True, text=True)
    jid = out.stdout.strip()
    return jid if jid.isdigit() else ""


def run_wave(loso_root, repo_root, datalake_root, *, seeds=(0, 1, 2), subjects=SUBJECTS, leakage_jobs=16,
             bootstrap_mode="full", max_total=28, max_c8=8, poll=120, user=None,
             count_total=_my_total, count_c8=_my_c8, submit=_sbatch_a, sleep=time.sleep,
             done_check=has_committed_artifact, out=sys.stdout) -> dict:
    user = user or os.environ.get("USER", "")
    folds = c8_fold_runs(loso_root, repo_root, seeds=seeds, subjects=subjects, leakage_jobs=leakage_jobs,
                         bootstrap_mode=bootstrap_mode)
    submitted, skipped, i = [], [], 0
    print(f"[c8-wave] {len(folds)} fold-runs; max_total={max_total} max_c8={max_c8} poll={poll}s user={user}",
          file=out, flush=True)
    while i < len(folds):
        if done_check(folds[i]["out_root"]):                    # resume: already done
            skipped.append(folds[i]["fold_uid"]); i += 1; continue
        k = slots_to_submit(count_total(user), count_c8(user), max_total, max_c8, len(folds) - i)
        for _ in range(k):
            if done_check(folds[i]["out_root"]):
                skipped.append(folds[i]["fold_uid"]); i += 1; continue
            jid = submit(folds[i], datalake_root)
            if not jid:                                          # QOS hit despite our count -> back off
                break
            print(f"[c8-wave] submitted {folds[i]['fold_uid']} phase A: {jid}", file=out, flush=True)
            submitted.append((folds[i]["fold_uid"], jid)); i += 1
        if i < len(folds):
            sleep(poll)
    print(f"[c8-wave] DONE: {len(submitted)} Phase-A submitted, {len(skipped)} already-done skipped; "
          f"each self-chains its decision Phase-B.", file=out, flush=True)
    return {"submitted": submitted, "skipped": skipped, "n_folds": len(folds)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.c8_wave")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--datalake-root", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--leakage-jobs", type=int, default=16)
    ap.add_argument("--max-total", type=int, default=28, help="overall cap headroom (shared with other work)")
    ap.add_argument("--max-c8", type=int, default=8, help="max concurrent C8 (staged-A/B) jobs")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args(argv)
    run_wave(args.loso_root, args.repo_root, args.datalake_root,
             seeds=tuple(int(s) for s in args.seeds.split(",")), leakage_jobs=args.leakage_jobs,
             max_total=args.max_total, max_c8=args.max_c8, poll=args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
