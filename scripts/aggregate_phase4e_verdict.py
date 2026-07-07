#!/usr/bin/env python
"""FSR Phase 4E — aggregate token-centering CSVs into the multi-seed CONFIRM, NETTED verdict (FSR_24).

Primary = E4 (deployable full mean alignment). Verdict on CONFIRM seeds, at alpha_star, NETTED (token-specific =
injected-effect - clean-effect). E1 only earns a sub-claim if E1 netted > E4 netted (CI>0). E0 oracle + E3/ERASE
controls excluded from headline. Establish-harm gate; mechanism gate (E1 in-scope iff median captured>=0.5).
"""
import csv, json
from pathlib import Path
import numpy as np

R = Path("results/fsr_phase4e_token_centering")
RNG = np.random.default_rng(0)
CONFIRM_SEEDS = [20260707, 20260708, 20260709]
DEV_SEED = 0
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


def gains(rows, pre):
    """per-row netted gain = (E_inj - injected) - (E_cln - orig); denom = orig - injected."""
    g, d = [], []
    for r in rows:
        inj = fl(r[f"{pre}_inj_bacc"]); cln = fl(r.get(f"{pre}_cln_bacc"))
        o = fl(r["bacc_orig"]); ij = fl(r["bacc_injected"])
        if inj is None or o is None or ij is None:
            continue
        gc = (cln - o) if cln is not None else 0.0
        g.append((inj - ij) - gc); d.append(o - ij)
    return np.array(g), np.array(d)


def pooled_ratio(g, d):
    return float(g.mean() / d.mean()) if abs(d.mean()) > 1e-4 else None


def boot_ratio_ci(g, d, nb=2000):
    n = len(g); out = []
    for _ in range(nb):
        i = RNG.integers(0, n, n)
        if abs(d[i].mean()) > 1e-4:
            out.append(g[i].mean() / d[i].mean())
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)] if out else [None, None]


def boot_diff_ci(ga, gb, nb=2000):
    n = len(ga); out = []
    for _ in range(nb):
        i = RNG.integers(0, n, n)
        out.append(ga[i].mean() - gb[i].mean())
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def boot_mean_ci(v, nb=2000):
    v = np.asarray(v, float); n = len(v)
    b = [v[RNG.integers(0, n, n)].mean() for _ in range(nb)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main():
    res = load("phase4e_token_centering_results.csv")
    mech = load("phase4e_mechanism_capture.csv")
    man = load("phase4e_manifest.csv")
    # verdict rows: CONFIRM seeds, at alpha_star
    conf = [r for r in res if int(fl(r["token_seed"])) in CONFIRM_SEEDS and r["is_alpha_star"] == "True"]
    dev = [r for r in res if int(fl(r["token_seed"])) == DEV_SEED and r["is_alpha_star"] == "True"]
    n_seedfolds = len(conf)

    denom_all = np.array([fl(r["induced_harm"]) for r in conf])
    harm_mean = round(float(denom_all.mean()), 4)
    harm_ci = boot_mean_ci(denom_all)
    harm_established = bool(harm_mean >= HARM_FLOOR and harm_ci[0] > 0)
    anti_harm_folds = int((denom_all <= 0).sum())

    arms = {}
    for pre in ("E0", "E4", "E1", "E3", "ERASE"):
        if pre == "E0":
            rr = [fl(r["E0_recovery"]) for r in conf if fl(r["E0_recovery"]) is not None]
            arms[pre] = dict(recovery=round(float(np.mean(rr)), 4) if rr else None)
            continue
        g, d = gains(conf, pre)
        pr = pooled_ratio(g, d)
        arms[pre] = dict(netted_recovery=(round(pr, 4) if pr is not None else None),
                         netted_recovery_ci=boot_ratio_ci(g, d), mean_net_gain=round(float(g.mean()), 4))
    # E2 netted recovery (from column)
    e2 = [fl(r["E2_netted_rec"]) for r in conf if fl(r["E2_netted_rec"]) is not None]
    arms["E2"] = dict(netted_recovery=round(float(np.mean(e2)), 4) if e2 else None)

    gE4, d = gains(conf, "E4"); gE1, _ = gains(conf, "E1"); gE3, _ = gains(conf, "E3"); gER, _ = gains(conf, "ERASE")
    e4_vs_e3 = round(float(gE4.mean() - gE3.mean()), 4); e4_vs_e3_ci = boot_diff_ci(gE4, gE3)
    e4_vs_er = round(float(gE4.mean() - gER.mean()), 4); e4_vs_er_ci = boot_diff_ci(gE4, gER)
    e1_vs_e4 = round(float(gE1.mean() - gE4.mean()), 4); e1_vs_e4_ci = boot_diff_ci(gE1, gE4)
    # clean-target TTA gain (E4 on clean)
    tta = np.array([fl(r["E4_cln_bacc"]) - fl(r["bacc_orig"]) for r in conf])
    tta_gain = round(float(tta.mean()), 4)
    # repaired > injected (netted): mean net gain > 0
    e4_improves = bool(gE4.mean() > 0)
    e4_beats_e3 = bool(e4_vs_e3 > 0); e4_beats_er = bool(e4_vs_er > 0)
    e4_beats_strong = bool(e4_vs_e3 >= DELTA and e4_vs_e3_ci[0] > 0 and e4_vs_er >= DELTA and e4_vs_er_ci[0] > 0)
    # source-heldout transfer
    sel_net = np.array([fl(r["sel_netted_rec"]) for r in man if int(fl(r["token_seed"])) in CONFIRM_SEEDS])
    transfers = bool(sel_net.mean() > 0)
    # task safety (clean drop from selection)
    clean_drop = np.array([fl(r["sel_clean_drop"]) for r in man if int(fl(r["token_seed"])) in CONFIRM_SEEDS])
    task_safe = bool(clean_drop.mean() <= SAFE_DROP)
    # mechanism
    cap = np.array([fl(r["captured_fraction"]) for r in mech if int(fl(r["token_seed"])) in CONFIRM_SEEDS])
    cap_med = round(float(np.median(cap)), 4)
    e1_in_scope = bool(cap_med >= 0.5)
    stress_unmet_frac = round(float(np.mean([1.0 if r["stress_unmet"] == "True" else 0.0
                                             for r in man if int(fl(r["token_seed"])) in CONFIRM_SEEDS])), 4)

    e4_rec = arms["E4"]["netted_recovery"]
    if not harm_established:
        level = "none"
    elif e4_rec is not None and e4_rec >= 0.50 and e4_beats_strong and task_safe and e4_improves and transfers:
        level = "strong"
    elif e4_rec is not None and e4_beats_e3 and e4_beats_er and e4_improves and task_safe and transfers:
        level = "partial"
    else:
        level = "none"
    cf_pass = bool(level in ("partial", "strong"))
    e1_adds = bool(e1_vs_e4 > 0 and e1_vs_e4_ci[0] > 0)

    verdict = dict(
        fresh_token_seeds=CONFIRM_SEEDS, dev_seed=DEV_SEED, n_confirm_seedfolds=n_seedfolds,
        alpha_selected_by_source_only=True, stress_unmet_frac=stress_unmet_frac,
        harm_established_confirm=harm_established, injection_harm_denominator=harm_mean,
        injection_harm_denominator_ci=harm_ci, anti_harm_folds=anti_harm_folds,
        mechanism_captured_fraction_median=cap_med, e1_mechanistically_in_scope=e1_in_scope,
        primary_repair="E4_full_mean_alignment",
        e4_netted_recovery=e4_rec, e4_netted_recovery_ci=arms["E4"]["netted_recovery_ci"],
        e4_raw_recovery=None,  # see results CSV appendix
        e1_netted_recovery=arms["E1"]["netted_recovery"], e1_netted_recovery_ci=arms["E1"]["netted_recovery_ci"],
        e1_minus_e4_netted_gain=e1_vs_e4, e1_minus_e4_netted_gain_ci=e1_vs_e4_ci, e1_adds_value_over_e4=e1_adds,
        e4_minus_e3_netted_gain=e4_vs_e3, e4_minus_e3_netted_gain_ci=e4_vs_e3_ci,
        e4_minus_erase_netted_gain=e4_vs_er, e4_minus_erase_netted_gain_ci=e4_vs_er_ci,
        e3_random_netted_recovery=arms["E3"]["netted_recovery"], erase_netted_recovery=arms["ERASE"]["netted_recovery"],
        e2_netted_recovery=arms["E2"]["netted_recovery"], exact_subtraction_recovery=arms["E0"]["recovery"],
        clean_target_tta_gain=tta_gain, source_heldout_transfers=transfers, source_val_task_safe=task_safe,
        target_labels_used_for_fit=False, target_labels_used_for_selection=False,
        target_labels_used_for_final_eval_only=True,
        repair_claim_level=level, counterfactual_repair_pass=cf_pass,
        pc2_gpu_gate=("eligible" if cf_pass else "paused"),
        claim_language=("Phase 4E shows a DESCRIPTIVE within-scope signal for repairing an injected "
                        "constant-offset shortcut, but the BINDING repair_claim_level is NONE (E4 fails the "
                        "frozen beat-ERASE-on-netted clause = tie). Primary=E4 deployable full first-moment "
                        "mean alignment, NETTED against clean-target TTA. E0 oracle + E3/ERASE controls "
                        "excluded from headline; ERASE netted is a regression-to-floor artifact (task-"
                        "destructive). Allowed: 'E4 produced a within-scope repair SIGNAL / descriptively "
                        "reversed the injected first-moment offset / did not clear the confirmatory bar'. "
                        "NOT allowed: 'E4 repairs shortcuts / E4 passed / deployable repair'. NOT a DG "
                        "method / SOTA / natural-harm claim; E0/E3/E4 not byte-comparable to PC1."),
    )
    (R / "phase4e_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    # confirmatory CSV (verdict driver, per confirm seed)
    conf_rows = []
    for sd in CONFIRM_SEEDS:
        rs = [r for r in conf if int(fl(r["token_seed"])) == sd]
        if not rs:
            continue
        g4, dd = gains(rs, "E4"); g1, _ = gains(rs, "E1")
        conf_rows.append(dict(token_seed=sd, n=len(rs), pooled_harm=round(float(dd.mean()), 4),
                              E4_netted_recovery=pooled_ratio(g4, dd), E1_netted_recovery=pooled_ratio(g1, dd),
                              E4_mean_net_gain=round(float(g4.mean()), 4)))
    with open(R / "phase4e_fresh_seed_confirmatory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conf_rows[0].keys())); w.writeheader()
        for r in conf_rows:
            w.writerow(r)

    print("Phase 4E verdict (CONFIRM aggregate, netted):")
    print(f"  n_confirm_seedfolds={n_seedfolds}  harm={harm_mean} ci={harm_ci} established={harm_established} anti_harm={anti_harm_folds}")
    print(f"  mechanism captured_fraction median={cap_med} -> E1 in-scope={e1_in_scope}   stress_unmet_frac={stress_unmet_frac}")
    print(f"  E0 exact recovery={arms['E0']['recovery']}")
    print(f"  E4 (PRIMARY) netted recovery={e4_rec} ci={arms['E4']['netted_recovery_ci']}  raw see CSV")
    print(f"  E1 netted={arms['E1']['netted_recovery']}   E1-E4 gain={e1_vs_e4} ci={e1_vs_e4_ci} -> E1 adds value={e1_adds}")
    print(f"  E4-E3 gain={e4_vs_e3} ci={e4_vs_e3_ci}   E4-ERASE gain={e4_vs_er} ci={e4_vs_er_ci}")
    print(f"  E3 random netted={arms['E3']['netted_recovery']} ERASE netted={arms['ERASE']['netted_recovery']} E2 netted={arms['E2']['netted_recovery']}")
    print(f"  clean-target TTA gain (E4 on clean)={tta_gain}   transfers={transfers}  task_safe={task_safe}")
    print(f"  ==> repair_claim_level={level}  pc2_gpu_gate={verdict['pc2_gpu_gate']}")


if __name__ == "__main__":
    main()
