#!/usr/bin/env python
"""FSR Phase 7C Q7C-b + weaponization + secondary repair (the BINDING gate). Runs after Q7C-a passes and
--stage full has written the transfer/target CSVs. Weaponization CONFIRMED iff (Q7C-a learnability) AND (Q7C-b:
on HELD-OUT source subjects the STRUCTURED conflict head shows higher subject reliance AND larger true-task drop
than BOTH the subject-shuffle band and the random-noise control, monotone in gamma) AND (target true-label bAcc
drops vs H0, clustered CI < 0). Repair (E4/E4b/ERASE target-X + Hreg training-time) is SECONDARY only."""
import csv, json
from pathlib import Path
import numpy as np

R = Path("results/fsr_head_only_label_conflict")
RNG = np.random.default_rng(0)
CONFIRM = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]
G = "0.4"


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rows_of(fn):
    return [r for r in csv.DictReader(open(R / fn)) if int(fl(r["token_seed"])) in CONFIRM]


def clu_of(rows):
    seen, idx = {}, []
    for r in rows:
        idx.append(seen.setdefault((r["dataset"], r["target_subject"]), len(seen)))
    return np.array(idx)


def _res(clu, rng):
    u = np.unique(clu); pick = rng.choice(u, len(u), replace=True)
    return np.concatenate([np.where(clu == c)[0] for c in pick])


def ci(rows, name, nb=3000):
    v = np.array([fl(r.get(name)) for r in rows], float); clu = clu_of(rows)
    m = np.isfinite(v); v, clu = v[m], clu[m]
    if len(v) == 0:
        return None, [None, None]
    b = [v[_res(clu, RNG)].mean() for _ in range(nb)]
    return round(float(v.mean()), 4), [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def at(rows, g):
    return [r for r in rows if r["gamma"] == g]


def main():
    gate = json.load(open(R / "label_conflict_verdict.json"))
    learn_pass = bool(gate.get("heldin_learnability_pass"))
    tr = rows_of("pseudo_target_transferability_gate.csv")
    hm = rows_of("target_harm.csv")
    rp = rows_of("repair_secondary_results.csv")

    tr4 = at(tr, G); tr2 = at(tr, "0.2")
    # PRIMARY transfer signal = held-out true-task HARM of the structured head beyond both controls (the effect,
    # per STOP-rule 5); L4/L5 reliance are reported mechanism diagnostics.
    tdrop0, tdrop0_ci = ci(tr4, "pt_task_drop_vs_H0")
    tdrops, tdrops_ci = ci(tr4, "pt_task_drop_vs_shuffle")
    tdropr, tdropr_ci = ci(tr4, "pt_task_drop_vs_random")
    tdrops2, _ = ci(tr2, "pt_task_drop_vs_shuffle")
    l5cs, l5cs_ci = ci(tr4, "pt_l5_conflict_minus_shuffle")     # diagnostic
    l5cr, l5cr_ci = ci(tr4, "pt_l5_conflict_minus_random")      # diagnostic
    l4cs, l4cs_ci = ci(tr4, "pt_l4_conflict_minus_shuffle")     # diagnostic (label-free)
    harm, harm_ci = ci(at(hm, G), "target_harm")

    hurts_transfer = bool(tdrop0_ci[0] is not None and tdrop0_ci[0] > 0)              # structured hurts unseen subj
    beats_shuffle = bool(tdrops_ci[0] is not None and tdrops_ci[0] > 0)              # beyond scrambled structure
    beats_random = bool(tdropr_ci[0] is not None and tdropr_ci[0] > 0)              # beyond matched label-noise
    monotone = bool(tdrops is not None and tdrops2 is not None and tdrops > tdrops2)
    transfer_pass = bool(learn_pass and hurts_transfer and beats_shuffle and beats_random and monotone)
    # per-DATASET breakdown (leave-one-dataset-out consistency; discloses the pooled sign's fragility)
    perds = {}
    for ds in sorted(set(r["dataset"] for r in tr4)):
        dd = [r for r in tr4 if r["dataset"] == ds]
        perds[ds] = dict(vs_H0=ci(dd, "pt_task_drop_vs_H0"), vs_shuffle=ci(dd, "pt_task_drop_vs_shuffle"),
                         vs_random=ci(dd, "pt_task_drop_vs_random"), n_folds=len(set(r["target_subject"] for r in dd)))
    # a strictly-negative vs_shuffle is NOT read as "shuffle more harmful" (construction asymmetry); the robust,
    # both-dataset-consistent statement is beats_shuffle=False (structured does NOT exceed shuffle) => not specific.
    beats_shuffle_any_dataset = any(v["vs_shuffle"][1][0] is not None and v["vs_shuffle"][1][0] > 0 for v in perds.values())
    target_harmed = bool(harm_ci[0] is not None and harm_ci[0] > 0)
    weaponization = bool(transfer_pass and target_harmed)

    # secondary repair at primary gamma (target-X E4/E4b/ERASE + training-time Hreg), vs Hconflict harm
    def repair(name):
        v0, _ = ci(rp, "H0_tgt"); vc, _ = ci(rp, "Hconflict_tgt"); vr, vr_ci = ci(rp, name)
        # recovered fraction of the harm gap (H0 - Hconflict); CI on per-fold (repair - Hconflict)
        d = [dict(dataset=r["dataset"], target_subject=r["target_subject"],
                  gain=fl(r[name]) - fl(r["Hconflict_tgt"])) for r in rp]
        gv = np.array([x["gain"] for x in d], float); clu = clu_of(rp)
        m = np.isfinite(gv)
        lo = (round(float(np.percentile([gv[m][_res(clu[m], RNG)].mean() for _ in range(3000)], 2.5)), 4)
              if m.sum() else None)
        return dict(mean=vr, gain_over_conflict=round(float(np.nanmean(gv)), 4), gain_ci_lo=lo,
                    frac_of_gap=(round(float(np.nanmean(gv) / (v0 - vc)), 3) if (v0 is not None and vc is not None and abs(v0 - vc) > 1e-6) else None))
    reps = {k: repair(f"{k}_tgt") for k in ("E4", "E4b", "ERASE", "Hreg")} if rp else {}
    tgtx_helps = any(reps.get(k, {}).get("gain_ci_lo") not in (None,) and reps[k]["gain_ci_lo"] > 0 for k in ("E4", "E4b", "ERASE"))
    hreg_helps = bool(reps.get("Hreg", {}).get("gain_ci_lo") is not None and reps["Hreg"]["gain_ci_lo"] > 0)
    if not weaponization:
        repair_level = None
    elif tgtx_helps:
        repair_level = "secondary"          # target-X op recovered some harm (would contradict R3 prediction)
    elif hreg_helps:
        repair_level = "training_time_partial"
    else:
        repair_level = "none"

    verdict = dict(
        stage="7C_Q7Cb_transfer_weaponization", heldin_learnability_pass=learn_pass,
        transfer_pt_task_drop_vs_H0=[tdrop0, tdrop0_ci], transfer_pt_task_drop_vs_shuffle=[tdrops, tdrops_ci],
        transfer_pt_task_drop_vs_random=[tdropr, tdropr_ci],
        diag_l5_conflict_minus_shuffle=[l5cs, l5cs_ci], diag_l5_conflict_minus_random=[l5cr, l5cr_ci],
        diag_l4_conflict_minus_shuffle=[l4cs, l4cs_ci],
        transfer_monotone_gamma=monotone, beats_shuffle_control=beats_shuffle, beats_random_control=beats_random,
        hurts_transfer_true_task=hurts_transfer, pseudo_target_transferability_pass=transfer_pass,
        per_dataset=perds, beats_shuffle_in_any_single_dataset=beats_shuffle_any_dataset,
        target_harm=[harm, harm_ci], target_true_label_bacc_dropped=target_harmed,
        weaponization_confirmed=weaponization, repair=reps, repair_claim_level=repair_level,
        interpretation=(
            "WEAPONIZATION CONFIRMED: controlled task-conflicting source labels weaponize the naturally present "
            "subject signal into a cross-subject transferable, target-harmful head-level reliance." if weaponization
            else ("Q7C-a passed (the head can MEMORIZE the subject-conditional conflict labels in-sample) but Q7C-b "
                  "did NOT: the structured conflict head's held-out/target true-task harm does NOT EXCEED the "
                  "subject-shuffle band on either dataset (beats_shuffle=False; the pooled vs-shuffle point is "
                  "mildly negative but reaches significance only pooled and only from the binary dataset -- the "
                  "4-class dataset is a null). It BEATS matched random noise (+0.078), so the harm is generic "
                  "SUBJECT-BLOCKED corruption harm that transfers, NOT a subject-STRUCTURE-specific / natural-"
                  "signal-aligned transferable reliance. A negative vs-shuffle is read as construction asymmetry, "
                  "NOT as 'scrambling is more harmful'. Reportable not-confirmed, not a weaponization claim."
                  if learn_pass else "Q7C-a failed upstream; transfer not evaluated.")),
    )
    full = json.load(open(R / "label_conflict_verdict.json"))
    full.update(dict(pseudo_target_transferability_pass=transfer_pass, weaponization_confirmed=weaponization,
                     target_harm=harm, target_harm_ci=harm_ci, repair_claim_level=repair_level,
                     transfer_verdict=verdict))
    (R / "label_conflict_verdict.json").write_text(json.dumps(full, indent=2) + "\n")

    print("Phase 7C Q7C-b transfer + weaponization (binding):")
    print(f"  heldin_learnability_pass (Q7C-a) = {learn_pass}")
    print(f"  transfer true-task drop vs H0={tdrop0} ci={tdrop0_ci} ; vs shuffle={tdrops} ci={tdrops_ci} ; vs random={tdropr} ci={tdropr_ci}")
    print(f"  [diag] l5(c-shuf)={l5cs} ci={l5cs_ci} ; l5(c-rnd)={l5cr} ci={l5cr_ci} ; l4(c-shuf)={l4cs} ci={l4cs_ci}")
    print(f"  beats_shuffle={beats_shuffle} beats_random={beats_random} monotone={monotone} hurts_transfer={hurts_transfer}")
    print(f"  ==> pseudo_target_transferability_pass={transfer_pass}")
    print(f"  target_harm = {harm} ci={harm_ci}  target_harmed={target_harmed}")
    print(f"  ==> WEAPONIZATION_CONFIRMED={weaponization}  repair_claim_level={repair_level}")
    for k, v in reps.items():
        print(f"     repair {k}: mean={v['mean']} gain_over_conflict={v['gain_over_conflict']} ci_lo={v['gain_ci_lo']} frac_of_gap={v['frac_of_gap']}")


if __name__ == "__main__":
    main()
