#!/usr/bin/env python
"""FSR Phase 4F — corrected confirmatory verdict (FSR_26). Design-red-teamed (wnhbjp2rt) hardening:

  * STRUCTURAL veto set (MF-1): ERASE is a pre-registered NEGATIVE CONTROL excluded by construction (it erases
    P_S -> task-destructive by design); E1/E3 are eligible comparators by construction and E4 must beat them.
    Target-scored eligibility() is DIAGNOSTIC only + a one-sided falsification guard (a declared control scoring
    as a valid repair PAUSES the verdict; it can never relax E4's bar). Firewall attestation is thus honest.
  * CLUSTERED bootstrap (MF-4): all gate CIs resample whole (dataset, target_subject) FOLDS, not the iid
    seed-rows (21 folds are the same clusters repeated across seeds -> iid understates variance ~sqrt(#seeds)).
  * strong is a strict superset of partial (MF-5); netted/specificity gates use the CLUSTERED CI lower bound,
    not the point estimate; LOSO requires ci_lo>0 on EVERY cut (MF-6); drop-anti-harm is a required consistency
    condition (MF-7). Guarantee: clean_drop<=SAFE_DROP => NOT regression_to_floor => valid repairs never excluded.
"""
import csv, json
from pathlib import Path
import numpy as np

R = Path("results/fsr_phase4f_corrected_repair")
RNG = np.random.default_rng(0)
CONFIRM = [20260721, 20260722, 20260723, 20260724, 20260725, 20260726, 20260727, 20260728]
NEGATIVE_CONTROLS = ("ERASE",)   # pre-registered: erases P_S, task-destructive by construction
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
    u = np.unique(clu)
    pick = rng.choice(u, size=len(u), replace=True)
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
    v = np.asarray(v, float)
    b = [v[_resample(clu, RNG)].mean() for _ in range(nb)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def boot_diff_ci(ga, gb, clu, nb=3000):
    out = []
    for _ in range(nb):
        i = _resample(clu, RNG)
        out.append(ga[i].mean() - gb[i].mean())
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def boot_ratio_ci(g, d, clu, nb=3000):
    out, dropped = [], 0
    for _ in range(nb):
        i = _resample(clu, RNG)
        if abs(d[i].mean()) > 1e-4:
            out.append(g[i].mean() / d[i].mean())
        else:
            dropped += 1
    ci = [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)] if out else [None, None]
    return ci, round(dropped / nb, 4)


def eligibility(rows, pre, orig_mean):
    """DIAGNOSTIC ONLY (target-scored): is arm `pre` a valid repair? Used to falsify negative-control status,
    never to relax E4's bar. clean_drop<=SAFE_DROP provably implies NOT regression_to_floor."""
    ib = col(rows, f"{pre}_inj_bacc"); cb = col(rows, f"{pre}_cln_bacc")
    inj = col(rows, "bacc_injected"); o = col(rows, "bacc_orig")
    raw = float((ib - inj).sum() / (o - inj).sum()) if abs((o - inj).sum()) > 1e-6 else None
    clean_drop = float((o - cb).mean())
    reg_floor = bool(abs(ib.mean() - cb.mean()) <= 0.01 and ib.mean() < orig_mean - 0.02 and cb.mean() < orig_mean - 0.02)
    valid_repair = bool(raw is not None and raw > 0 and clean_drop <= SAFE_DROP and not reg_floor)
    reason = None if valid_repair else (
        "raw_recovery<=0" if (raw is None or raw <= 0) else
        "clean_drop>SAFE_DROP" if clean_drop > SAFE_DROP else "regression_to_floor")
    return dict(raw_recovery=round(raw, 4) if raw is not None else None, clean_drop=round(clean_drop, 4),
                regression_to_floor=reg_floor, valid_repair=valid_repair, reason=reason)


def main():
    res = load("phase4f_token_centering_results.csv")
    conf = [r for r in res if int(fl(r["token_seed"])) in CONFIRM and r["is_alpha_star"] == "True"]
    clu = clu_of(conf)
    orig_mean = float(col(conf, "bacc_orig").mean()); inj_mean = float(col(conf, "bacc_injected").mean())
    harm = col(conf, "induced_harm")
    harm_mean = round(float(harm.mean()), 4); harm_ci = boot_mean_ci(harm, clu)
    harm_established = bool(harm_mean >= HARM_FLOOR and harm_ci[0] > 0)
    anti_harm = int((harm <= 0).sum())

    gE4, d = gains(conf, "E4"); gE3, _ = gains(conf, "E3"); gE1, _ = gains(conf, "E1")
    e4_net = pooled_ratio(gE4, d); e4_net_ci, e4_net_drop = boot_ratio_ci(gE4, d, clu)
    e4_raw = float((col(conf, "E4_inj_bacc") - col(conf, "bacc_injected")).sum() / harm.sum())
    e4_inj_bacc = float(col(conf, "E4_inj_bacc").mean()); e4_cln_bacc = float(col(conf, "E4_cln_bacc").mean())
    e4_clean_drop = round(orig_mean - e4_cln_bacc, 4)
    e4_reg_floor = bool(abs(e4_inj_bacc - e4_cln_bacc) <= 0.01 and e4_inj_bacc < orig_mean - 0.02 and e4_cln_bacc < orig_mean - 0.02)
    e4_task_safe = bool(e4_clean_drop <= SAFE_DROP and e4_raw > 0 and e4_inj_bacc > inj_mean and not e4_reg_floor)
    e4_e3 = round(float(gE4.mean() - gE3.mean()), 4); e4_e3_ci = boot_diff_ci(gE4, gE3, clu)
    e4_beats_e3 = bool(e4_e3 >= DELTA and e4_e3_ci[0] > 0)
    e1_e4 = round(float(gE1.mean() - gE4.mean()), 4)

    # eligibility DIAGNOSTIC (target-scored) + STRUCTURAL veto set
    elig = {pre: eligibility(conf, pre, orig_mean) for pre in ("E1", "E3", "ERASE")}
    eligible = [p for p in ("E1", "E3") if p not in NEGATIVE_CONTROLS]
    ineligible = list(NEGATIVE_CONTROLS)
    neg_control_falsified = any(elig[p]["valid_repair"] for p in NEGATIVE_CONTROLS)  # one-sided pause guard
    beats_eligible = all((gE4.mean() - gains(conf, p)[0].mean()) > 0 for p in eligible)

    # leave-one-seed-out on E4-E3 (clustered CI within each cut); require ci_lo>0 on EVERY cut
    loso, loso_pass = [], True
    for s in CONFIRM:
        sub = [r for r in conf if int(fl(r["token_seed"])) != s]
        ga, _ = gains(sub, "E4"); gb, _ = gains(sub, "E3")
        m = round(float(ga.mean() - gb.mean()), 4); ci = boot_diff_ci(ga, gb, clu_of(sub))
        loso.append(dict(dropped_seed=s, e4_minus_e3=m, ci_lo=ci[0], ci_hi=ci[1]))
        if not (ci[0] > 0):
            loso_pass = False
    loso_min_margin = round(min(l["e4_minus_e3"] for l in loso), 4)

    # per-dataset (N=2: descriptive; require each dataset point sign positive AND raw>0)
    perds, signs = [], []
    for dsname in sorted(set(r["dataset"] for r in conf)):
        rs = [r for r in conf if r["dataset"] == dsname]
        ga, dd = gains(rs, "E4"); gb, _ = gains(rs, "E3")
        raw = float((col(rs, "E4_inj_bacc") - col(rs, "bacc_injected")).sum() / (col(rs, "bacc_orig") - col(rs, "bacc_injected")).sum())
        net = pooled_ratio(ga, dd); m = float(ga.mean() - gb.mean())
        perds.append(dict(dataset=dsname, n=len(rs), e4_raw=round(raw, 4),
                          e4_netted=round(net, 4) if net is not None else None, e4_minus_e3=round(m, 4)))
        signs.append(m > 0 and raw > 0)
    per_dataset_sign_consistent = bool(all(signs))

    keep = [r for r in conf if fl(r["induced_harm"]) > 0]
    gAk, dAk = gains(keep, "E4"); drop_anti = pooled_ratio(gAk, dAk)
    drop_anti_ok = bool(drop_anti is not None and drop_anti > 0.30)

    # ---- grade: strong is a strict superset of partial; netted/specificity gate on CLUSTERED CI ----
    partial_ok = bool(harm_established and e4_task_safe and e4_beats_e3 and beats_eligible
                      and not neg_control_falsified and e4_net is not None and e4_net > 0.30
                      and e4_net_ci[0] is not None and e4_net_ci[0] > 0
                      and per_dataset_sign_consistent and drop_anti_ok)
    if neg_control_falsified:
        level = "none"
    elif not harm_established:
        level = "none"
    elif not partial_ok:
        level = "none"
    elif e4_net >= 0.50 and loso_pass:
        level = "strong"
    else:
        level = "partial"
    gate = "eligible_for_protocol_update" if level in ("partial", "strong") else "paused"

    verdict = dict(
        fresh_confirm_seeds=CONFIRM, uses_phase4e_seeds_for_claim=False, n_confirm_seedfolds=len(conf),
        n_folds_clusters=int(len(np.unique(clu))),
        harm_established=harm_established, injection_harm_denominator=harm_mean,
        injection_harm_denominator_ci_clustered=harm_ci, anti_harm_folds=anti_harm,
        primary_repair="E4_full_mean_alignment",
        e4_task_safe=e4_task_safe, e4_clean_target_drop=e4_clean_drop, e4_raw_recovery=round(e4_raw, 4),
        e4_regression_to_floor=e4_reg_floor,
        e4_netted_recovery=round(e4_net, 4) if e4_net is not None else None,
        e4_netted_recovery_ci_clustered=e4_net_ci, e4_netted_ratio_dropped_frac=e4_net_drop,
        e4_minus_e3_netted_gain=e4_e3, e4_minus_e3_ci_clustered=e4_e3_ci, e4_beats_e3=e4_beats_e3,
        e1_minus_e4_netted_gain=e1_e4,
        veto_set_structural=list(eligible), negative_controls=list(ineligible),
        negative_control_falsified=neg_control_falsified,
        comparator_eligibility_DIAGNOSTIC={p: elig[p] for p in elig},
        e4_beats_eligible_comparators=beats_eligible,
        leave_one_seed_out=loso, leave_one_seed_out_pass=loso_pass, loso_min_margin=loso_min_margin,
        per_dataset=perds, per_dataset_sign_consistent=per_dataset_sign_consistent,
        per_dataset_note="N=2 datasets: sign-consistency is descriptive, not a powered consistency test.",
        drop_anti_harm_e4_netted=round(drop_anti, 4) if drop_anti is not None else None, drop_anti_harm_ok=drop_anti_ok,
        target_labels_used_for_fit=False, target_labels_used_for_selection=False,
        comparator_veto_set_used_target=False,
        negative_control_diagnostic_uses_target=True, eligibility_rule_preregistered=True,
        target_labels_used_for_final_eval_only=True,
        repair_claim_level=level, pc2_gpu_gate=gate,
        claim_language=("Phase 4F corrected confirmatory test on FRESH seeds. E4 first-moment mean alignment; "
                        "veto set is STRUCTURAL (E1/E3 in, ERASE a pre-declared task-destructive negative control "
                        "excluded by construction; clean-safe => not regression-to-floor => valid repairs never "
                        "excluded). All gate CIs CLUSTERED over folds. Scope = controlled constant-offset "
                        "first-moment shortcut ONLY (not general repair / DG / SOTA / natural). Phase 4E stays "
                        "none (not re-scored)."),
    )
    (R / "phase4f_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    _w(R / "phase4f_comparator_eligibility.csv", [dict(arm=p, **elig[p]) for p in elig])
    _w(R / "phase4f_leave_one_seed_out.csv", loso)
    _w(R / "phase4f_per_dataset_summary.csv", perds)
    _w(R / "phase4f_clean_target_netting.csv", [dict(
        dataset=r["dataset"], target_subject=r["target_subject"], token_seed=r["token_seed"],
        bacc_orig=r["bacc_orig"], bacc_injected=r["bacc_injected"], E4_inj_bacc=r["E4_inj_bacc"],
        E4_cln_bacc=r["E4_cln_bacc"], E4_netted_rec=r["E4_netted_rec"], E1_inj_bacc=r["E1_inj_bacc"],
        E1_cln_bacc=r["E1_cln_bacc"], E1_netted_rec=r["E1_netted_rec"]) for r in conf])
    _w(R / "phase4f_random_controls.csv", [dict(
        dataset=r["dataset"], target_subject=r["target_subject"], token_seed=r["token_seed"],
        E3_inj_bacc=r["E3_inj_bacc"], E3_cln_bacc=r["E3_cln_bacc"], E3_raw_rec=r["E3_raw_rec"],
        E3_netted_rec=r["E3_netted_rec"], ERASE_inj_bacc=r["ERASE_inj_bacc"], ERASE_cln_bacc=r["ERASE_cln_bacc"],
        ERASE_raw_rec=r["ERASE_raw_rec"], ERASE_netted_rec=r["ERASE_netted_rec"]) for r in conf])

    print("Phase 4F verdict (fresh confirm seeds, netted, CLUSTERED CIs):")
    print(f"  clusters(folds)={len(np.unique(clu))} n_seedfolds={len(conf)}  harm={harm_mean} ci={harm_ci} est={harm_established} anti={anti_harm}")
    print(f"  E4 raw={round(e4_raw,4)} clean_drop={e4_clean_drop} task_safe={e4_task_safe} reg_floor={e4_reg_floor}")
    print(f"  E4 netted={round(e4_net,4) if e4_net else None} clustered_ci={e4_net_ci} (dropped {e4_net_drop})")
    print(f"  E4-E3 gain={e4_e3} clustered_ci={e4_e3_ci} beats_e3(>= {DELTA} & CI>0)={e4_beats_e3}")
    print(f"  veto set (structural)={eligible}  negative_controls={ineligible}  neg_control_falsified={neg_control_falsified}")
    print(f"    ERASE diagnostic: raw={elig['ERASE']['raw_recovery']} clean_drop={elig['ERASE']['clean_drop']} valid_repair={elig['ERASE']['valid_repair']}")
    print(f"  E4 beats eligible={beats_eligible}  E1-E4={e1_e4}")
    print(f"  LOSO pass(every cut CI_lo>0)={loso_pass} min_margin={loso_min_margin}")
    print(f"  per-dataset sign-consistent(descriptive,N=2)={per_dataset_sign_consistent}: " + " ".join(f"{p['dataset']}(raw{p['e4_raw']},net{p['e4_netted']},d{p['e4_minus_e3']})" for p in perds))
    print(f"  drop-anti-harm E4 netted={round(drop_anti,4) if drop_anti else None} ok={drop_anti_ok}")
    print(f"  partial_ok={partial_ok}  ==> repair_claim_level={level}  pc2_gpu_gate={gate}")


def _w(p, rows):
    if not rows:
        Path(p).write_text(""); return
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
