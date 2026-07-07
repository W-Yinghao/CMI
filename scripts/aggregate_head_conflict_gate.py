#!/usr/bin/env python
"""FSR Phase 7C Q7C-a — HELD-IN LEARNABILITY gate (fail-closed). On the TRAINING subjects, did the linear head
actually SATISFY the task-conflicting labels (fit on the relabeled subset beyond a task-only floor, rising with
gamma) under an achieved corruption rate that tracks gamma and an exactly-P(y)-preserving construction? This is a
capability check (can the shortcut be linearly learned at all) -- the structured-beats-control TRANSFER test is
Q7C-b (aggregate_head_conflict_transfer.py). If Q7C-a fails -> STOP: shortcut not linearly learnable; 7C-full
does not run. The subject-reliance-on-train columns are reported as DIAGNOSTICS only, not gate conditions."""
import csv, json
from pathlib import Path
import numpy as np

R = Path("results/fsr_head_only_label_conflict")
RNG = np.random.default_rng(0)
CONFIRM = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def clu_of(rows):
    seen, idx = {}, []
    for r in rows:
        idx.append(seen.setdefault((r["dataset"], r["target_subject"]), len(seen)))
    return np.array(idx)


def _res(clu, rng):
    u = np.unique(clu); pick = rng.choice(u, len(u), replace=True)
    return np.concatenate([np.where(clu == c)[0] for c in pick])


def boot_lo(v, clu, nb=3000):
    m = np.isfinite(v)
    if m.sum() == 0:
        return None
    v, clu = v[m], clu[m]
    b = [v[_res(clu, RNG)].mean() for _ in range(nb)]
    return round(float(np.percentile(b, 2.5)), 4)


def col(rows, name):
    return np.array([fl(r.get(name)) for r in rows], float)


def main():
    rows = [r for r in csv.DictReader(open(R / "heldin_learnability_gate.csv")) if int(fl(r["token_seed"])) in CONFIRM]
    py_exact = int(max(int(r["global_Py_delta"]) for r in rows)) == 0
    dose = {}
    for g in ("0.0", "0.2", "0.4"):
        rr = [r for r in rows if r["gamma"] == g]; clu = clu_of(rr)
        cf = col(rr, "conflict_fit"); mf = col(rr, "conflict_fit_minus_floor")
        ach = col(rr, "achieved_conflict_frac")
        l5cs = col(rr, "l5_conflict_train") - col(rr, "l5_shuffle_train_max")     # diagnostic
        l5cr = col(rr, "l5_conflict_train") - col(rr, "l5_random_train")          # diagnostic
        td = col(rr, "heldin_task_drop_train")
        dose[g] = dict(
            achieved_conflict_frac=round(float(np.nanmean(ach)), 4),
            conflict_fit=round(float(np.nanmean(cf)), 4) if np.isfinite(cf).any() else None,
            conflict_fit_minus_floor=round(float(np.nanmean(mf)), 4) if np.isfinite(mf).any() else None,
            conflict_fit_minus_floor_ci_lo=boot_lo(mf, clu),
            l5_conflict_minus_shuffle_train=round(float(np.nanmean(l5cs)), 4),     # diagnostic only
            l5_conflict_minus_random_train=round(float(np.nanmean(l5cr)), 4),      # diagnostic only
            heldin_task_drop_train=round(float(np.nanmean(td)), 4), n=len(rr))
    # Q7C-a pass: exact histogram + achieved rate tracks gamma + head satisfies conflict labels beyond task floor,
    # rising with gamma (a linear-learnability capability check; NOT the transfer/control test which is Q7C-b).
    achieved_monotone = bool(dose["0.0"]["achieved_conflict_frac"] <= 1e-6 and
                             dose["0.2"]["achieved_conflict_frac"] > dose["0.0"]["achieved_conflict_frac"] and
                             dose["0.4"]["achieved_conflict_frac"] > dose["0.2"]["achieved_conflict_frac"])
    cf4, cf2 = dose["0.4"]["conflict_fit"], dose["0.2"]["conflict_fit"]
    conflict_rises = bool(cf4 is not None and cf2 is not None and cf4 > cf2)
    beats_floor = bool(dose["0.4"]["conflict_fit_minus_floor_ci_lo"] is not None and
                       dose["0.4"]["conflict_fit_minus_floor_ci_lo"] > 0)
    learnability_pass = bool(py_exact and achieved_monotone and conflict_rises and beats_floor)

    if not py_exact:
        reason = "histogram_not_preserved"
    elif not achieved_monotone:
        reason = "achieved_corruption_rate_does_not_track_gamma (construction saturated / mis-dosed)"
    elif not (conflict_rises and beats_floor):
        reason = ("shortcut_not_linearly_learnable: the linear head does NOT satisfy the task-conflicting labels "
                  "beyond a task-only floor / not rising with gamma -> a linear head on frozen 4B latents cannot "
                  "memorize the subject shortcut; no weaponization inference (7C-full does not run)")
    else:
        reason = "head_learns_subject_shortcut_under_task_conflict (memorizes conflict labels beyond task floor)"

    verdict = dict(
        stage="7C_Q7Ca_heldin_learnability_gate", global_Py_preserved=py_exact, achieved_rate_tracks_gamma=achieved_monotone,
        conflict_fit_rises_with_gamma=conflict_rises, conflict_fit_beats_task_floor=beats_floor,
        heldin_learnability_pass=learnability_pass, dose_response=dose, gate_reason=reason,
        proceed_to_7C_full=learnability_pass,
        note=("FAIL-CLOSED Q7C-a. Learnability = the head satisfies the P(y)-exact, gamma-tracking task-conflict "
              "labels on TRAINING subjects beyond a task-only floor (memorization capability). The structured-"
              "beats-shuffle/random TRANSFER test binds at Q7C-b, which only runs if this passes. Subject-reliance-"
              "on-train columns here are diagnostics, not gate conditions." if not learnability_pass else
              "Q7C-a passes: the linear head can memorize the subject shortcut under task-conflict. Run 7C-full; "
              "Q7C-b TRANSFER (structured beats shuffle+random on held-out subjects) is the binding weaponization gate."),
    )
    (R / "label_conflict_verdict.json").write_text(json.dumps(dict(
        heldin_learnability_pass=learnability_pass, pseudo_target_transferability_pass=None,
        weaponization_confirmed=None, primary_gamma=0.4, global_Py_preserved=py_exact,
        gate_verdict=verdict, target_labels_used_for_fit=False, target_labels_used_for_selection=False,
        repair_claim_level=None, pc2_gpu_gate="paused"), indent=2) + "\n")

    print("Phase 7C Q7C-a held-in learnability gate (fail-closed):")
    print(f"  global P(y) preserved exactly = {py_exact}")
    for g in ("0.0", "0.2", "0.4"):
        d = dose[g]
        print(f"  gamma={g}: achieved={d['achieved_conflict_frac']} conflict_fit={d['conflict_fit']} "
              f"(minus_floor={d['conflict_fit_minus_floor']} ci_lo={d['conflict_fit_minus_floor_ci_lo']}) | "
              f"[diag] l5(c-shuf)={d['l5_conflict_minus_shuffle_train']} l5(c-rnd)={d['l5_conflict_minus_random_train']} "
              f"task_drop={d['heldin_task_drop_train']}")
    print(f"  achieved_monotone={achieved_monotone}  conflict_rises={conflict_rises}  beats_floor={beats_floor}")
    print(f"  ==> heldin_learnability_pass={learnability_pass}  reason={reason}  proceed_to_7C_full={learnability_pass}")


if __name__ == "__main__":
    main()
