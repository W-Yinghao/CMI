"""C8 THROTTLED WAVE controller — trickle the 27 (seed, target) fold-runs into free submit-cap slots, and
MANAGE both phases itself (no self-chaining — a self-chained Phase-B sbatch fails under a full shared cap and
is lost).

The cluster caps total submitted jobs per user (QOSMaxSubmitJobPerUserLimit ~30), shared with the user's
other work, and separately caps per-user GPUs. So C8 cannot fire its 54 jobs at once. This controller runs as
ONE persistent CPU job that each cycle: reads disk + squeue state, then (within an overall cap AND a
C8-specific cap) submits Phase-B for any fold whose Phase-A has finished (priority — never waste the GPU work)
and Phase-A for any not-yet-started fold. Every submit is retried on the next cycle if it is rejected (QOS),
so nothing is lost. Resume-safe: fold state is derived from disk (committed artifact / Phase-A staging) plus a
persisted job-id map, so a restarted controller picks up exactly where it left off.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from .loso_plan import SUBJECTS, loso_plan

_PHASE_A = "oaci/slurm_confirmatory_staged_a.sh"
_PHASE_B = "oaci/slurm_confirmatory_staged_b.sh"
_LOGS = "/projects/EEG-foundation-model/yinghao/oaci-confirmatory-logs/%x-%j.out"
_STATE = "wave_state.json"


def c8_fold_runs(loso_root, repo_root, *, seeds, subjects=SUBJECTS, leakage_jobs=16, bootstrap_mode="full") -> list:
    """The 27 fold-runs in (seed, target) order, each with the shared env (the controller adds the
    A-specific / B-specific bits at submit time)."""
    repo_abs = os.path.abspath(repo_root)
    out = []
    for s in (int(x) for x in seeds):
        for spec in loso_plan(subjects=subjects):
            out_root = os.path.join(loso_root, f"seed-{s}", spec["fold_id"])
            out.append({"fold_uid": f"seed-{s}/{spec['fold_id']}", "seed": s, "target": int(spec["target"]),
                        "out_root": out_root, "staging": os.path.join(out_root, "staging"),
                        "common_env": {"OACI_TARGET_SUBJECT": str(spec["target"]), "OACI_MODEL_SEED": str(s),
                                       "OACI_BOOTSTRAP_MODE": str(bootstrap_mode),
                                       "OACI_LEAKAGE_JOBS": str(int(leakage_jobs)), "OACI_OUT_ROOT": out_root,
                                       "OACI_REPO": repo_abs}})
    return out


def slots_to_submit(my_total, my_c8, max_total, max_c8, n_remaining) -> int:
    return max(0, min(int(n_remaining), int(max_total) - int(my_total), int(max_c8) - int(my_c8)))


def has_committed_artifact(out_root) -> bool:
    return len(glob.glob(os.path.join(out_root, "artifacts", "*", "COMMITTED.json"))) >= 1


def phase_a_complete(out_root) -> bool:
    return os.path.exists(os.path.join(out_root, "staging", "phase_a.json"))


def classify(fold, live, state, *, done=has_committed_artifact, a_complete=phase_a_complete) -> str:
    """One fold's action: 'done' | 'needs_b' (A finished, B not live) | 'needs_a' (no A yet, A not live) |
    'in_progress' (an A/B job is live, or A is still running)."""
    uid = fold["fold_uid"]
    if done(fold["out_root"]):
        return "done"
    st = state.get(uid, {})
    if a_complete(fold["out_root"]):
        return "in_progress" if st.get("b_jid") in live else "needs_b"
    return "in_progress" if st.get("a_jid") in live else "needs_a"


# ---- live cluster side-effects (mocked in tests) ----
def _live_jids(user) -> set:
    r = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i"], capture_output=True, text=True)
    return {l.strip() for l in r.stdout.splitlines() if l.strip()}


def _my_total(user) -> int:
    return len(_live_jids(user))


def _my_c8(user) -> int:
    r = subprocess.run(["squeue", "-u", user, "-h", "-o", "%j"], capture_output=True, text=True)
    return sum(1 for l in r.stdout.splitlines() if l.strip().startswith("oaci-staged-"))


def _sbatch(script, env, datalake_root, extra=()) -> str:
    e = {**os.environ, **env, "OACI_DATALAKE_ROOT": datalake_root}
    out = subprocess.run(["sbatch", "--parsable", f"--output={_LOGS}", *extra, script],
                         env=e, capture_output=True, text=True)
    jid = out.stdout.strip()
    return jid if jid.isdigit() else ""


def _submit_a(fold, datalake_root, submit=_sbatch) -> str:
    return submit(_PHASE_A, {**fold["common_env"], "OACI_CHAIN_PHASE_B": "0"}, datalake_root)   # controller owns B


def _submit_b(fold, datalake_root, submit=_sbatch) -> str:
    return submit(_PHASE_B, {**fold["common_env"], "OACI_COMPUTE_DECISIONS": "1"}, datalake_root)


def _load_state(loso_root) -> dict:
    p = os.path.join(loso_root, _STATE)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def _save_state(loso_root, state) -> None:
    os.makedirs(loso_root, exist_ok=True)
    with open(os.path.join(loso_root, _STATE), "w") as f:
        json.dump(state, f)


def run_wave(loso_root, repo_root, datalake_root, *, seeds=(0, 1, 2), subjects=SUBJECTS, leakage_jobs=16,
             bootstrap_mode="full", max_total=28, max_c8=8, poll=120, max_cycles=None, user=None,
             live_jids=_live_jids, count_total=_my_total, count_c8=_my_c8, submit_a=_submit_a, submit_b=_submit_b,
             done=has_committed_artifact, a_complete=phase_a_complete, load_state=_load_state,
             save_state=_save_state, sleep=time.sleep, out=sys.stdout) -> dict:
    user = user or os.environ.get("USER", "")
    folds = c8_fold_runs(loso_root, repo_root, seeds=seeds, subjects=subjects, leakage_jobs=leakage_jobs,
                         bootstrap_mode=bootstrap_mode)
    state = load_state(loso_root)
    print(f"[c8-wave] {len(folds)} fold-runs; max_total={max_total} max_c8={max_c8} poll={poll}s user={user}",
          file=out, flush=True)
    cycles = 0
    while True:
        if all(done(f["out_root"]) for f in folds):
            break
        if max_cycles is not None and cycles >= max_cycles:
            break
        cycles += 1
        live = live_jids(user)
        cls = {f["fold_uid"]: classify(f, live, state, done=done, a_complete=a_complete) for f in folds}
        total, c8 = count_total(user), count_c8(user)
        # Phase B first (a finished Phase-A must not sit idle), then Phase A.
        for phase, want in (("needs_b", submit_b), ("needs_a", submit_a)):
            for f in folds:
                if cls[f["fold_uid"]] != phase:
                    continue
                if not (total < max_total and c8 < max_c8):
                    break
                jid = want(f, datalake_root)
                if not jid:                                      # QOS/failure -> retry next cycle
                    continue
                key = "b_jid" if phase == "needs_b" else "a_jid"
                state.setdefault(f["fold_uid"], {})[key] = jid
                total += 1; c8 += 1
                print(f"[c8-wave] submitted {f['fold_uid']} {'phase B' if phase=='needs_b' else 'phase A'}: {jid}",
                      file=out, flush=True)
        save_state(loso_root, state)
        n_done = sum(1 for f in folds if done(f["out_root"]))
        if n_done < len(folds):
            sleep(poll)
    n_done = sum(1 for f in folds if done(f["out_root"]))
    print(f"[c8-wave] EXIT after {cycles} cycles: {n_done}/{len(folds)} fold-runs committed.", file=out, flush=True)
    return {"state": state, "n_done": n_done, "n_folds": len(folds), "cycles": cycles}


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
