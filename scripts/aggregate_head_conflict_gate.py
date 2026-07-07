#!/usr/bin/env python
"""FSR Phase 7C Q7C-a — held-in learnability gate verdict (fail-closed). Did the TASK-CONFLICT corruption make the
head learn a SUBJECT shortcut (rely on subject to satisfy task-conflicting labels), beating BOTH the random-noise
(Hrandom) and subject-shuffle (Hshuffle) controls, monotone in gamma? Histogram must be exactly P(y)-preserving."""
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


def boot_ci(v, clu, nb=3000):
    v = np.asarray(v, float); b = [v[_res(clu, RNG)].mean() for _ in range(nb)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main():
    rows = [r for r in csv.DictReader(open(R / "heldin_learnability_gate.csv")) if int(fl(r["token_seed"])) in CONFIRM]
    py_exact = int(max(int(r["global_Py_delta"]) for r in rows)) == 0
    dose = {}
    for g in ("0.0", "0.2", "0.4"):
        rr = [r for r in rows if r["gamma"] == g]; clu = clu_of(rr)
        cs = np.array([fl(r["l5_conflict_minus_shuffle"]) for r in rr])
        cr = np.array([fl(r["l5_conflict_minus_random"]) for r in rr])
        td = np.array([fl(r["heldin_task_drop"]) for r in rr])
        dose[g] = dict(l5_conflict_minus_shuffle=round(float(cs.mean()), 4), cs_ci=boot_ci(cs, clu),
                       l5_conflict_minus_random=round(float(cr.mean()), 4), cr_ci=boot_ci(cr, clu),
                       heldin_task_drop=round(float(td.mean()), 4), td_ci=boot_ci(td, clu), n=len(rr))
    # Q7C-a: subject-specific reliance beats BOTH controls, monotone in gamma, CI@0.4 > 0
    beats_shuffle = bool(dose["0.4"]["l5_conflict_minus_shuffle"] > dose["0.2"]["l5_conflict_minus_shuffle"] and
                         dose["0.4"]["cs_ci"][0] > 0)
    beats_random = bool(dose["0.4"]["l5_conflict_minus_random"] > dose["0.2"]["l5_conflict_minus_random"] and
                        dose["0.4"]["cr_ci"][0] > 0)
    task_conflicts = bool(dose["0.4"]["heldin_task_drop"] > 0 and dose["0.4"]["td_ci"][0] > 0)  # corruption bites
    learnability_pass = bool(py_exact and beats_shuffle and beats_random)

    if not py_exact:
        reason = "histogram_not_preserved"
    elif not task_conflicts:
        reason = "corruption_did_not_bite (task not hurt) -> mis-constructed"
    elif not learnability_pass:
        reason = "head_did_not_learn_a_SUBJECT_shortcut (task hurt, but subject-specific reliance not > shuffle/random controls)"
    else:
        reason = "head_learned_subject_shortcut_under_task_conflict"

    verdict = dict(
        stage="7C_Q7Ca_learnability_gate", global_Py_preserved=py_exact, dose_response=dose,
        task_conflict_bites=task_conflicts, beats_shuffle_control=beats_shuffle, beats_random_control=beats_random,
        heldin_learnability_pass=learnability_pass, gate_reason=reason, proceed_to_7C_full=learnability_pass,
        note=("FAIL-CLOSED Q7C-a. Task-conflict corruption is P(y)-exact and hurts held-in task (task bites), but "
              "learnability requires the head to satisfy the conflicting labels by relying on SUBJECT beyond the "
              "matched random-noise (Hrandom) AND subject-shuffle (Hshuffle) controls, monotone in gamma. Only if "
              "this passes does 7C-full (Q7C-b transferability + target harm + repair) run." if not learnability_pass
              else "Q7C-a passes: the head learned a subject-specific shortcut under task-conflict; run 7C-full "
                   "(Q7C-b transferability is the binding gate for a target-weaponization claim)."),
    )
    (R / "label_conflict_verdict.json").write_text(json.dumps(dict(
        heldin_learnability_pass=learnability_pass, pseudo_target_transferability_pass=None,
        weaponization_confirmed=None, primary_gamma=0.4, global_Py_preserved=py_exact,
        gate_verdict=verdict, target_labels_used_for_fit=False, target_labels_used_for_selection=False,
        repair_claim_level=None, pc2_gpu_gate="paused"), indent=2) + "\n")

    print("Phase 7C Q7C-a learnability gate (fail-closed):")
    print(f"  global P(y) preserved exactly = {py_exact}")
    for g in ("0.0", "0.2", "0.4"):
        d = dose[g]
        print(f"  gamma={g}: l5(conflict-shuffle)={d['l5_conflict_minus_shuffle']} ci={d['cs_ci']} | l5(conflict-random)={d['l5_conflict_minus_random']} ci={d['cr_ci']} | heldin_task_drop={d['heldin_task_drop']} ci={d['td_ci']}")
    print(f"  task_conflict_bites={task_conflicts}  beats_shuffle={beats_shuffle}  beats_random={beats_random}")
    print(f"  ==> heldin_learnability_pass={learnability_pass}  reason={reason}  proceed_to_7C_full={learnability_pass}")


if __name__ == "__main__":
    main()
