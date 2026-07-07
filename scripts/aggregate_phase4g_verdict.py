#!/usr/bin/env python
"""FSR Phase 4G — controlled second-moment verdict (FSR_29). Inherits 4F corrections: clustered bootstrap over
folds, structural veto {E4,E3} vs ERASE negative-control, leave-one-DATASET-out binding, netting, non-identity.
Primary = E4b (covariance alignment); E4 first-moment is the mean-null sanity (should be insufficient)."""
import csv, json
from pathlib import Path
import numpy as np

R = Path("results/fsr_phase4g_second_moment")
RNG = np.random.default_rng(0)
CONFIRM = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]
PRIMARY_INJ = "varmod"
HARM_FLOOR = 0.02
DELTA = 0.02
SAFE_DROP = 0.01


def load(f):
    return list(csv.DictReader(open(R / f)))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def col(rows, c):
    return np.array([fl(r[c]) for r in rows], float)


def clu_of(rows):
    seen, idx = {}, []
    for r in rows:
        idx.append(seen.setdefault((r["dataset"], r["target_subject"]), len(seen)))
    return np.array(idx)


def _resample(clu, rng):
    u = np.unique(clu); pick = rng.choice(u, size=len(u), replace=True)
    return np.concatenate([np.where(clu == c)[0] for c in pick])


def gains(rows, pre):
    g, d = [], []
    for r in rows:
        inj = fl(r[f"{pre}_inj_bacc"]); cln = fl(r.get(f"{pre}_cln_bacc"))
        o = fl(r["bacc_orig"]); ij = fl(r["bacc_injected"])
        gc = (cln - o) if cln is not None else 0.0
        g.append((inj - ij) - gc); d.append(o - ij)
    return np.array(g), np.array(d)


def pooled_ratio(g, d):
    return float(g.mean() / d.mean()) if abs(d.mean()) > 1e-4 else None


def boot_mean_ci(v, clu, nb=3000):
    v = np.asarray(v, float); b = [v[_resample(clu, RNG)].mean() for _ in range(nb)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def boot_diff_ci(ga, gb, clu, nb=3000):
    out = []
    for _ in range(nb):
        i = _resample(clu, RNG); out.append(ga[i].mean() - gb[i].mean())
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def boot_ratio_ci(g, d, clu, nb=3000):
    out = []
    for _ in range(nb):
        i = _resample(clu, RNG)
        if abs(d[i].mean()) > 1e-4:
            out.append(g[i].mean() / d[i].mean())
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)] if out else [None, None]


def task_safe(rows, pre, orig_mean, inj_mean, harm):
    ib = col(rows, f"{pre}_inj_bacc"); cb = col(rows, f"{pre}_cln_bacc")
    raw = float((ib - col(rows, "bacc_injected")).sum() / harm.sum())
    clean_drop = float(orig_mean - cb.mean())
    reg = bool(abs(ib.mean() - cb.mean()) <= 0.01 and ib.mean() < orig_mean - 0.02 and cb.mean() < orig_mean - 0.02)
    return dict(raw=round(raw, 4), clean_drop=round(clean_drop, 4), regression_to_floor=reg,
                task_safe=bool(clean_drop <= SAFE_DROP and raw > 0 and ib.mean() > inj_mean and not reg))


def verdict_for(res, injt):
    conf = [r for r in res if int(fl(r["token_seed"])) in CONFIRM and r["is_alpha_star"] == "True" and r["injtype"] == injt]
    if len(conf) < 2:
        return dict(injtype=injt, n=len(conf), note="insufficient")
    clu = clu_of(conf)
    orig_mean = float(col(conf, "bacc_orig").mean()); inj_mean = float(col(conf, "bacc_injected").mean())
    harm = col(conf, "induced_harm"); harm_mean = round(float(harm.mean()), 4); harm_ci = boot_mean_ci(harm, clu)
    harm_est = bool(harm_mean >= HARM_FLOOR and harm_ci[0] > 0)
    mean_disp = round(float(col(conf, "mean_disp").mean()), 5)

    gE4b, d = gains(conf, "E4b"); gE4, _ = gains(conf, "E4"); gE3, _ = gains(conf, "E3")
    e4b_net = pooled_ratio(gE4b, d); e4b_ci = boot_ratio_ci(gE4b, d, clu)
    e4_net = pooled_ratio(gE4, d)
    ts4b = task_safe(conf, "E4b", orig_mean, inj_mean, harm)
    e4b_e3 = round(float(gE4b.mean() - gE3.mean()), 4); e4b_e3_ci = boot_diff_ci(gE4b, gE3, clu)
    e4b_beats_e3 = bool(e4b_e3 >= DELTA and e4b_e3_ci[0] > 0)
    e4b_e4 = round(float(gE4b.mean() - gE4.mean()), 4)
    # ORACLE-E4b (shrink along the TRUE injected direction) + fail-attribution diagnostics (red-team wjdzttrhu)
    gE4bO, _ = gains(conf, "E4bO"); e4bO_net = pooled_ratio(gE4bO, d)
    e4bO_e3 = round(float(gE4bO.mean() - gE3.mean()), 4); e4bO_e3_ci = boot_diff_ci(gE4bO, gE3, clu)
    oracle_beats_e3 = bool(e4bO_e3 >= DELTA and e4bO_e3_ci[0] > 0)
    dominance = round(float(col(conf, "inj_dominance").mean()), 4)
    vc_overlap = round(float(col(conf, "vc_overlap").mean()), 4)
    arm_overlap = round(float(col(conf, "arm_overlap").mean()), 4)
    if e4b_beats_e3:
        fail_attr = "e4b_passes_specificity"
    elif oracle_beats_e3:
        fail_attr = "mis_estimation_dirs_exc_misses_true_direction"   # oracle repairs, estimated does not
    else:
        fail_attr = "genuinely_weak_second_moment_repair"             # even oracle sub-threshold
    beats_veto = bool((gE4b.mean() - gE4.mean()) > 0 and (gE4b.mean() - gE3.mean()) > 0)
    # mean-null sanity: E4 first-moment must NOT explain most recovery
    e4_sufficient = bool(e4_net is not None and e4_net >= 0.50 and e4_net >= (e4b_net or 0))
    mean_null_pass = bool(mean_disp < 0.02 and not e4_sufficient)
    # ERASE negative control (structural; falsification only)
    erase = task_safe(conf, "ERASE", orig_mean, inj_mean, harm)
    erase_valid = erase["task_safe"]
    # identity / non-identity
    e4bi = col(conf, "E4b_inj_bacc"); e4bc = col(conf, "E4b_cln_bacc"); ident = np.abs(e4bi - e4bc) < 1e-9
    mech_id = round(float(ident.mean()), 4)
    e4b_net_ni = pooled_ratio(gE4b[~ident], d[~ident]) if (~ident).sum() >= 2 else e4b_net
    # leave-one-dataset-out (binding)
    lodo, lodo_pass, signs = [], True, []
    for dsn in sorted(set(r["dataset"] for r in conf)):
        rs = [r for r in conf if r["dataset"] == dsn]; cl = clu_of(rs)
        h = col(rs, "induced_harm"); hci = boot_mean_ci(h, cl)
        ga, _ = gains(rs, "E4b"); gb, _ = gains(rs, "E3"); m = float(ga.mean() - gb.mean()); mci = boot_diff_ci(ga, gb, cl)
        he = bool(h.mean() >= HARM_FLOOR and hci[0] > 0); sp = bool(m >= DELTA and mci[0] > 0)
        lodo.append(dict(dataset=dsn, n=len(rs), harm=round(float(h.mean()), 4), harm_established=he,
                         e4b_minus_e3=round(m, 4), e4b_e3_ci=mci, specificity_pass=sp, passes=bool(he and sp)))
        signs.append(m > 0)
        if not (he and sp):
            lodo_pass = False
    per_ds_sign = bool(all(signs))

    partial_ok = bool(harm_est and mean_null_pass and ts4b["task_safe"] and e4b_beats_e3 and beats_veto
                      and not erase_valid and e4b_net is not None and (e4b_net_ni or 0) > 0.30 and per_ds_sign)
    if not partial_ok:
        level = "none"
    elif (e4b_net_ni or 0) >= 0.50 and lodo_pass:
        level = "strong"
    else:
        level = "partial"
    return dict(injtype=injt, n=len(conf), n_folds=int(len(np.unique(clu))),
                harm=harm_mean, harm_ci=harm_ci, harm_established=harm_est, mean_disp=mean_disp,
                mean_null_pass=mean_null_pass, e4_first_moment_sufficient=e4_sufficient,
                e4_netted_recovery=round(e4_net, 4) if e4_net is not None else None,
                e4b_task_safe=ts4b["task_safe"], e4b_clean_drop=ts4b["clean_drop"], e4b_raw_recovery=ts4b["raw"],
                e4b_netted_recovery=round(e4b_net, 4) if e4b_net is not None else None, e4b_netted_recovery_ci_clustered=e4b_ci,
                e4b_netted_recovery_nonidentity=round(e4b_net_ni, 4) if e4b_net_ni is not None else None,
                mechanical_identity_frac=mech_id,
                e4b_minus_e3_netted_gain=e4b_e3, e4b_minus_e3_ci_clustered=e4b_e3_ci, e4b_beats_e3=e4b_beats_e3,
                e4b_minus_e4_netted_gain=e4b_e4, e4b_beats_veto_set=beats_veto,
                oracle_e4b_netted_recovery=round(e4bO_net, 4) if e4bO_net is not None else None,
                oracle_e4b_minus_e3_netted_gain=e4bO_e3, oracle_e4b_minus_e3_ci=e4bO_e3_ci, oracle_beats_e3=oracle_beats_e3,
                injection_dominance_index=dominance, est_dir_vc_overlap=vc_overlap, arm_dir_overlap=arm_overlap,
                fail_attribution=fail_attr,
                erase_valid_repair=erase_valid, erase_raw=erase["raw"], erase_clean_drop=erase["clean_drop"],
                leave_one_dataset_out=lodo, leave_one_dataset_out_pass=lodo_pass, per_dataset_sign_consistent=per_ds_sign,
                repair_claim_level=level)


def main():
    res = load("phase4g_repair_results.csv")
    prim = verdict_for(res, PRIMARY_INJ)
    others = {injt: verdict_for(res, injt) for injt in sorted(set(r["injtype"] for r in res)) if injt != PRIMARY_INJ}
    level = prim["repair_claim_level"]
    gate = "eligible_for_review" if level in ("partial", "strong") else "paused"
    v = dict(primary_branch="spatial_z", injection_type="second_moment_controlled", primary_injection=PRIMARY_INJ,
             fresh_confirm_seeds=CONFIRM, primary_repair="E4b_second_moment_alignment",
             **{k: prim[k] for k in prim if k not in ("injtype",)},
             secondary_injection_verdicts=others,
             repair_claim_scope="controlled_second_moment_only",
             target_labels_used_for_fit=False, target_labels_used_for_selection=False,
             comparator_veto_set_used_target=False, target_labels_used_for_final_eval_only=True,
             pc2_gpu_gate=gate, pc2_gpu_run_authorized=False,
             claim_language=("Phase 4G: controlled MEAN-NULL second-moment injected shortcut on spatial_z. Primary "
                             "repair=E4b covariance alignment (target-X-only); E4 first-moment is the mean-null "
                             "sanity (should be insufficient). Structural veto {E4,E3} vs ERASE negative-control; "
                             "clustered CIs; leave-one-DATASET-out BINDING. Scope=controlled_second_moment_only; "
                             "NOT learned/natural/general/DG/SOTA. pc2_gpu_run_authorized=false (GPU paused). Does "
                             "not re-score 4E/4F."))
    (R / "phase4g_verdict.json").write_text(json.dumps(v, indent=2) + "\n")
    _w(R / "phase4g_leave_one_dataset_out.csv", prim.get("leave_one_dataset_out", []))

    print("Phase 4G verdict (primary injection=varmod, netted, clustered):")
    print(f"  n={prim['n']} folds={prim.get('n_folds')} harm={prim['harm']} ci={prim['harm_ci']} est={prim['harm_established']}")
    print(f"  mean_disp={prim['mean_disp']} mean_null_pass={prim['mean_null_pass']} E4_first_moment_sufficient={prim['e4_first_moment_sufficient']} (E4 netted {prim['e4_netted_recovery']})")
    print(f"  E4b task_safe={prim['e4b_task_safe']} raw={prim['e4b_raw_recovery']} clean_drop={prim['e4b_clean_drop']}")
    print(f"  E4b netted={prim['e4b_netted_recovery']} ci={prim['e4b_netted_recovery_ci_clustered']} nonident={prim['e4b_netted_recovery_nonidentity']} (mech_id {prim['mechanical_identity_frac']})")
    print(f"  E4b-E3={prim['e4b_minus_e3_netted_gain']} ci={prim['e4b_minus_e3_ci_clustered']} beats_e3={prim['e4b_beats_e3']}  E4b-E4={prim['e4b_minus_e4_netted_gain']} beats_veto={prim['e4b_beats_veto_set']}")
    print(f"  ERASE valid_repair={prim['erase_valid_repair']} (raw {prim['erase_raw']} clean_drop {prim['erase_clean_drop']})")
    print(f"  leave-one-dataset-out pass={prim['leave_one_dataset_out_pass']}: " + " ".join(f"{l['dataset']}(h{l['harm']}/{l['harm_established']},d{l['e4b_minus_e3']}/{l['specificity_pass']})" for l in prim['leave_one_dataset_out']))
    print(f"  ORACLE-E4b netted={prim['oracle_e4b_netted_recovery']} oracle-E3={prim['oracle_e4b_minus_e3_netted_gain']} ci={prim['oracle_e4b_minus_e3_ci']} beats_e3={prim['oracle_beats_e3']}")
    print(f"  inj_dominance={prim['injection_dominance_index']} est_dir.vc_overlap={prim['est_dir_vc_overlap']} arm_overlap={prim['arm_dir_overlap']}  FAIL_ATTRIBUTION={prim['fail_attribution']}")
    print(f"  ==> repair_claim_level={level}  scope=controlled_second_moment_only  pc2_gpu_gate={gate}")
    for injt, ov in others.items():
        print(f"  [secondary {injt}] level={ov.get('repair_claim_level')} harm={ov.get('harm')} E4b_net={ov.get('e4b_netted_recovery')} E4b-E3={ov.get('e4b_minus_e3_netted_gain')}")


def _w(p, rows):
    if not rows:
        Path(p).write_text(""); return
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
