#!/usr/bin/env python
"""FSR Phase 4D — aggregate repair CSVs into pooled recovery + pre-registered pass/fail (FSR_21).

Design-red-team hardening (wtbd1lg62):
  * PRIMARY alpha is the PRE-REGISTERED CONSTANT 1.0 (never chosen from target harm). alpha=2.0 is a reported
    secondary; promoting it to the headline is stop-worthy.
  * ESTABLISH-HARM gate: if pooled (bAcc_orig - bAcc_injected) at alpha=1.0 is not clearly positive
    (>= 0.02 AND bootstrap CI excludes 0), recovery_fraction is undefined -> repair_claim_level='none'.
  * D1-vs-D3a is judged on POOLED REPAIRED bAcc directly (denominator-free) + bootstrap CI over ALL folds;
    no fold is ever dropped by a target-harm threshold.
  * Recovery-band -> claim-level mapping: rf>=0.70(+gates)->'strong'; 0<rf<0.70(+gates)->'partial'; else 'none'.
  * D0 (exact) and D2/D3b (erasure) are contrast baselines only; erasure_arms_excluded_from_headline=true.
"""
import csv, json
from pathlib import Path
import numpy as np

R = Path("results/fsr_phase4d_repair")
RNG = np.random.default_rng(0)
PRIMARY_ALPHA = 1.0          # PRE-REGISTERED constant (FSR_21); not derived from any bacc_* array
DELTA_BACC = 0.02            # declared min pooled bAcc margin D1 must beat D3a by for a STRONG pass
HARM_FLOOR = 0.02            # declared min pooled induced harm to trust recovery_fraction
SAFE_DROP = 0.01             # declared max source-val clean-task drop (task-safety gate)


def load(f):
    return list(csv.DictReader(open(R / f)))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def col(rows, c):
    return np.array([fl(r[c]) for r in rows], float)


def pooled_recovery(rows, bacc_col):
    inj, org, rep = col(rows, "bacc_injected"), col(rows, "bacc_orig"), col(rows, bacc_col)
    d = org.mean() - inj.mean()
    return (float((rep.mean() - inj.mean()) / d) if abs(d) > 1e-4 else None,
            dict(orig=round(float(org.mean()), 4), injected=round(float(inj.mean()), 4),
                 repaired=round(float(rep.mean()), 4),
                 pooled_recovery=round(float((rep.mean() - inj.mean()) / d), 4) if abs(d) > 1e-4 else None,
                 n=len(rows)))


def boot_recovery_ci(rows, bacc_col, nb=2000):
    inj, org, rep = col(rows, "bacc_injected"), col(rows, "bacc_orig"), col(rows, bacc_col)
    n = len(rows); out = []
    for _ in range(nb):
        i = RNG.integers(0, n, n)
        d = org[i].mean() - inj[i].mean()
        if abs(d) > 1e-4:
            out.append((rep[i].mean() - inj[i].mean()) / d)
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)] if out else [None, None]


def boot_mean_ci(v, nb=2000):
    v = np.asarray(v, float); n = len(v)
    b = [v[RNG.integers(0, n, n)].mean() for _ in range(nb)]
    return round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)


def boot_diff_ci(rows, colA, colB, nb=2000):
    a, b = col(rows, colA), col(rows, colB); n = len(rows); out = []
    for _ in range(nb):
        i = RNG.integers(0, n, n)
        out.append(a[i].mean() - b[i].mean())
    return round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)


def main():
    rec = load("phase4d_target_recovery.csv")
    sel = load("phase4d_source_val_selection.csv")
    alphas = sorted({fl(r["alpha"]) for r in rec})
    arms = dict(D0="D0_exact_bacc", D1="D1_cf_adapter_bacc", D2="D2_taskorth_erase_bacc",
                D3a="D3a_rand_adapter_bacc", D3b="D3b_randk_erase_bacc")

    per_alpha = {}
    for a in alphas:
        rows = [r for r in rec if fl(r["alpha"]) == a]
        d = {}
        for k, c in arms.items():
            r_, info = pooled_recovery(rows, c)
            d[k] = dict(**info, ci=boot_recovery_ci(rows, c))
        per_alpha[str(a)] = d

    def rows_at(a):
        return [r for r in rec if fl(r["alpha"]) == a]

    def clean_drop(a):
        v = [fl(r["val_clean_task_drop"]) for r in sel if r["arm"] == "D1_counterfactual" and fl(r["alpha"]) == a]
        return round(float(np.mean(v)), 4), round(float(np.max(v)), 4), boot_mean_ci(v)

    def val_inj_bacc(a):
        v = [fl(r["val_inj_task_bacc"]) for r in sel if r["arm"] == "D1_counterfactual" and fl(r["alpha"]) == a]
        return round(float(np.mean(v)), 4), boot_mean_ci(v)

    a1 = PRIMARY_ALPHA if PRIMARY_ALPHA in alphas else alphas[0]
    a2 = 2.0 if 2.0 in alphas else (alphas[-1] if len(alphas) > 1 else None)
    P1 = per_alpha[str(a1)]

    # ---- establish-harm gate (target used only for scoring the estimand) ----
    harm1 = col(rows_at(a1), "induced_harm")
    harm_mean = round(float(harm1.mean()), 4)
    harm_ci = list(boot_mean_ci(harm1))
    harm_established = bool(harm_mean >= HARM_FLOOR and harm_ci[0] > 0)

    d1_rf = P1["D1"]["pooled_recovery"]
    d1_bacc, d3a_bacc, inj_bacc = P1["D1"]["repaired"], P1["D3a"]["repaired"], P1["D1"]["injected"]
    d1_minus_d3a_bacc = round(d1_bacc - d3a_bacc, 4)
    d1_minus_d3a_ci = list(boot_diff_ci(rows_at(a1), "D1_cf_adapter_bacc", "D3a_rand_adapter_bacc"))
    improves = bool(d1_bacc > inj_bacc)
    beats_random_point = bool(d1_minus_d3a_bacc > 0)
    beats_random_strong = bool(d1_minus_d3a_bacc >= DELTA_BACC and d1_minus_d3a_ci[0] > 0)
    drop_mean, drop_max, drop_ci = clean_drop(a1)
    task_safe = bool(drop_mean <= SAFE_DROP)
    vib_mean, vib_ci = val_inj_bacc(a1)

    if not harm_established:
        level = "none"
    elif d1_rf is not None and d1_rf >= 0.70 and beats_random_strong and task_safe and improves:
        level = "strong"
    elif d1_rf is not None and d1_rf > 0 and beats_random_point and improves and task_safe:
        level = "partial"
    else:
        level = "none"
    cf_pass = bool(level in ("partial", "strong"))

    # alpha=2 secondary D1-vs-D3a margin + CI (reported, never headline)
    d1_minus_rand_a2, d1_minus_rand_a2_ci = None, None
    if a2 is not None:
        P2 = per_alpha[str(a2)]
        d1_minus_rand_a2 = round(P2["D1"]["repaired"] - P2["D3a"]["repaired"], 4)
        d1_minus_rand_a2_ci = list(boot_diff_ci(rows_at(a2), "D1_cf_adapter_bacc", "D3a_rand_adapter_bacc"))

    # target-harm-CONDITIONED diagnostics (exploratory; NOT headline; folds selected by target harm >= floor)
    diag = {}
    for a in (a1, a2):
        if a is None:
            continue
        sub = [r for r in rows_at(a) if (fl(r["induced_harm"]) or 0) >= HARM_FLOOR]
        if len(sub) < 2:
            diag[str(a)] = dict(n=len(sub), note="too few harm-established folds")
            continue
        d1r, _ = pooled_recovery(sub, "D1_cf_adapter_bacc")
        d3r, _ = pooled_recovery(sub, "D3a_rand_adapter_bacc")
        A = col(sub, "D1_cf_adapter_bacc"); B = col(sub, "D3a_rand_adapter_bacc")
        margin = float(A.mean() - B.mean())
        ci = list(boot_diff_ci(sub, "D1_cf_adapter_bacc", "D3a_rand_adapter_bacc"))
        diag[str(a)] = dict(n=len(sub),
                            d1_recovery_ratio_of_pooled_means=round(d1r, 4) if d1r is not None else None,
                            d3a_recovery_ratio_of_pooled_means=round(d3r, 4) if d3r is not None else None,
                            d1_minus_d3a_bacc=round(margin, 4), d1_minus_d3a_bacc_ci=ci,
                            d1_minus_d3a_ci_excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
                            d1_beats_d3a_folds=int((A > B).sum()), n_folds=len(sub))

    verdict = dict(
        counterfactual_repair_pass=cf_pass, repair_claim_level=level,
        primary_branch="spatial_z", primary_alpha=a1,
        alpha_is_preregistered_constant=True, alpha_selection_used_target=False,
        harm_established_alpha1=harm_established,
        injection_harm_denominator_alpha1=harm_mean, injection_harm_denominator_alpha1_ci=harm_ci,
        recovery_fraction_alpha1=round(d1_rf, 4) if d1_rf is not None else None,
        recovery_fraction_alpha1_ci=P1["D1"]["ci"],
        recovery_fraction_alpha2=(round(per_alpha[str(a2)]["D1"]["pooled_recovery"], 4)
                                  if a2 is not None and per_alpha[str(a2)]["D1"]["pooled_recovery"] is not None
                                  else None),
        d1_repaired_bacc_alpha1=d1_bacc, d3a_repaired_bacc_alpha1=d3a_bacc, injected_bacc_alpha1=inj_bacc,
        d1_minus_random_bacc_alpha1=d1_minus_d3a_bacc, d1_minus_random_bacc_alpha1_ci=d1_minus_d3a_ci,
        d1_minus_random_bacc_alpha2=d1_minus_rand_a2, d1_minus_random_bacc_alpha2_ci=d1_minus_rand_a2_ci,
        d1_beats_d3a_folds_alpha1=int((col(rows_at(a1), "D1_cf_adapter_bacc") >
                                       col(rows_at(a1), "D3a_rand_adapter_bacc")).sum()),
        d1_beats_d3a_folds_alpha2=(int((col(rows_at(a2), "D1_cf_adapter_bacc") >
                                        col(rows_at(a2), "D3a_rand_adapter_bacc")).sum()) if a2 else None),
        diagnostics_target_harm_conditioned=diag,
        diagnostics_note=("harm-conditioned subsets select folds by target harm >= HARM_FLOOR -- exploratory, "
                          "target-derived selection, NOT a pre-registered headline (STOP rule 4)."),
        random_control_beaten_point=beats_random_point, random_control_beaten_strong=beats_random_strong,
        delta_bacc_margin=DELTA_BACC,
        target_improved_over_injected=improves,
        source_val_task_safe=task_safe, source_val_clean_task_drop_mean_alpha1=drop_mean,
        source_val_clean_task_drop_max_alpha1=drop_max, source_val_clean_task_drop_ci_alpha1=list(drop_ci),
        u_generalization_val_inj_bacc_alpha1=vib_mean, u_generalization_val_inj_bacc_ci_alpha1=list(vib_ci),
        task_orth_erasure_recovery_alpha1=P1["D2"]["pooled_recovery"],
        randk_erasure_recovery_alpha1=P1["D3b"]["pooled_recovery"],
        task_orth_erasure_minus_random_alpha1=round(P1["D2"]["repaired"] - P1["D3b"]["repaired"], 4),
        exact_subtraction_recovery_alpha1=P1["D0"]["pooled_recovery"],
        erasure_arms_excluded_from_headline=True,
        target_labels_used_for_fit=False, target_labels_used_for_selection=False,
        target_labels_used_for_final_eval_only=True,
        per_alpha=per_alpha,
        claim_language=("Phase 4D repairs an INJECTED positive-control shortcut, not a natural one. "
                        f"repair_claim_level={level}. D2 (task-orth) and D3b (random-k) are ERASURE contrast "
                        "baselines ONLY -- regardless of their measured recovery, no output may frame erasure "
                        "as improving target. A pass licenses 'a counterfactual/task-protected repair recovers "
                        "a known harmful branch-local shortcut where erasure fails' -- NOT a DG method, NOT SOTA, "
                        "NOT 'spatial leakage is naturally harmful'. Headline verdict is alpha=1.0 only."),
    )
    (R / "phase4d_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    print("Phase 4D verdict:")
    print(f"  establish-harm@{a1}: pooled induced harm={harm_mean} ci={harm_ci} -> established={harm_established}")
    for a in alphas:
        pa = per_alpha[str(a)]
        star = " (PRIMARY)" if a == a1 else " (secondary)"
        print(f"  alpha={a}{star}: orig={pa['D1']['orig']} inj={pa['D1']['injected']}")
        for k in ("D0", "D1", "D2", "D3a", "D3b"):
            print(f"    {k:4s} repaired={pa[k]['repaired']} pooled_recovery={pa[k]['pooled_recovery']} ci={pa[k]['ci']}")
    print(f"  D1 recovery@{a1}={d1_rf} ci={P1['D1']['ci']}")
    print(f"  D1-vs-D3a repaired-bAcc margin@{a1}={d1_minus_d3a_bacc} ci={d1_minus_d3a_ci} "
          f"(point>{0}={beats_random_point}, strong(>= {DELTA_BACC} & ci>0)={beats_random_strong})")
    print(f"  source-val clean-task drop@{a1}: mean={drop_mean} max={drop_max} -> task_safe={task_safe}")
    print(f"  u-generalization val-inj bAcc@{a1}={vib_mean} ci={vib_ci} (low => target fail is token-shift, not repair)")
    print(f"  target improved over injected: {improves}")
    print(f"  ==> counterfactual_repair_pass={cf_pass}  repair_claim_level={level}")


if __name__ == "__main__":
    main()
