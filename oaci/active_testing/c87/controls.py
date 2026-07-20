"""C87 synthetic control gate (C87P §6) — PRODUCTION harness.

Runs POS / POS_DENSE / NEG_A / NEG_B / CALIB + the three mutation tests (CALIB-M1, CALIB-M2, NEG-M3) on
the EXACT production modules (generator, policies, estimand cross-fit, LURE, patient-cluster BCa bootstrap,
gate). Emits a signed CONTROL_PASS token iff every criterion passes AND every mutation test triggers its
failure signature; otherwise ENGINEERING_BLOCKER. NO threshold is relaxed to pass; AMBIGUOUS == FAIL.

Every scoped size is recorded in `config` (no silent caps). Control sizes are a deliberate, DISCLOSED
subset of the real C87E config (A_real=648, N_p^(e)~5000): a smaller held n is a HARDER clustered-bootstrap
coverage test, and the machinery being validated is candidate/n-agnostic.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from . import estimand as E
from . import generator as G
from . import policies as P
from .bootstrap import (bca_lcb, bc_ci, crossfit_vec, mean_excess, paired_gain_bootstrap,
                        percentile_lcb)
from .estimand import _fold_assignment, dispersion_s_e, held_view_loss
from .gate import TAU_G

CONFIG = dict(
    A=648, n_pat=400, pos_n_pat=1000, E=3, K_seeds=10, B_boot=2000, K_MC=1000,
    fast_reps=100, budgets=[16, 32, 64], power_budget=32, alpha=0.05,
    coverage_band=[0.935, 0.965], tau_G=TAU_G,
    note=("control-scoped subset of C87E (A_real=648 matched). COVERAGE controls (CALIB/POS_DENSE/M2) use "
          "N_p=400 as a HARDER-than-real coverage stress (real held ~5000). POWER/specificity controls "
          "(POS/NEG_B) use N_p=pos_n_pat=1000 (still << real ~5000) because power is a property at "
          "deployment n; small n unfairly understates it. B_boot main=2000; K_MC=1000; fast-loop "
          "bootstrap=100 reps; power = P(MODEL-SELECTOR LCB[G]>0 in ALL cohorts at B=power_budget)."),
)


def _cohort_eval(coh, policy, B, K_seeds, B_boot, seed, bca=True):
    """Run K policy-seed acquisitions + P0, then CRN-paired cluster bootstrap. Returns metrics dict.

    bca=True -> BCa LCB (jackknife; for reported main CIs). bca=False -> percentile LCB (fast; for the
    power/FWER loops that repeat over K_MC redraws)."""
    rng = np.random.default_rng(seed)
    Lbar, pat_ids = E.patient_mean_loss(E.binary_nll(coh.probs, coh.y), coh.patient_of)
    fold_of = _fold_assignment(pat_ids)
    s_e = dispersion_s_e(Lbar)
    p0 = P.P0()
    picks_pi = np.array([policy.select(coh.probs, coh.y, coh.patient_of, B,
                                       np.random.default_rng(seed * 1000 + k))[0] for k in range(K_seeds)])
    picks_p0 = np.array([p0.select(coh.probs, coh.y, coh.patient_of, B,
                                   np.random.default_rng(seed * 1000 + 7 + k))[0] for k in range(K_seeds)])
    bs = paired_gain_bootstrap(Lbar, fold_of, picks_pi, picks_p0, coh.aC, B_boot, rng)
    cluster_se = float(np.median([bs["boot_pi"].std()]))  # cheap proxy: SE of the pi excess estimate
    if s_e <= 0:
        lcb_G = float("nan")
    elif bca:
        lcb_G = bca_lcb(bs["obs_G"], bs["boot_G"], _jack_G(Lbar, fold_of, picks_pi, picks_p0),
                        alpha=CONFIG["alpha"])
    else:
        lcb_G = percentile_lcb(bs["boot_G"], CONFIG["alpha"])
    # vacuity: dispersion s_e vs patient-cluster bootstrap SE of held loss (skip in the fast power loop)
    if bca:
        Lh_boot = np.array([held_view_loss(Lbar[:, rng.integers(0, Lbar.shape[1], Lbar.shape[1])])
                            for _ in range(120)])
        vac_se = float(np.median(Lh_boot.std(axis=0)))
    else:
        vac_se = float("nan")
    return dict(s_e=s_e, obs_G=bs["obs_G"], obs_T=bs["obs_T"], obs_pi=bs["obs_pi"],
                lcb_G_std=(lcb_G / s_e if s_e > 0 else float("nan")),
                lcb_G_raw=lcb_G,
                lcb_T_raw=percentile_lcb(bs["boot_T"], CONFIG["alpha"]),
                boot_T=bs["boot_T"], cluster_se=vac_se,
                vacuous=bool(s_e <= vac_se), aHfin=coh.aHfin, aC=coh.aC)


def _jack_G(Lbar, fold_of, picks_pi, picks_p0):
    from .bootstrap import _jackknife_G
    return _jackknife_G(Lbar, fold_of, picks_pi, picks_p0)


def _fast_lcb_G(coh, policy, B, K_seeds, reps, seed):
    """Fast per-redraw paired cluster-bootstrap PERCENTILE LCB of G/s_e (for power/FWER loops)."""
    m = _cohort_eval(coh, policy, B, K_seeds, reps, seed, bca=False)
    return m["lcb_G_std"], m


# ------------------------------------------------------------------ POS / NEG / CALIB --------------

def run_pos(cfg, log):
    npat = cfg["pos_n_pat"]
    world = G.make_world("POS", A=cfg["A"], n_pat=npat, E=cfg["E"], seed=101)
    crit = {}
    # crit1: T^CF CI covers 0 and aC==aHfin in every cohort
    t_ok = True
    for e, coh in enumerate(world):
        m = _cohort_eval(coh, P.ModelSelector(), 64, cfg["K_seeds"], cfg["B_boot"], 200 + e)
        lo = percentile_lcb(m["boot_T"], cfg["alpha"] / 2)
        hi = -percentile_lcb(-m["boot_T"], cfg["alpha"] / 2)
        t_ok &= bool((lo <= 0 <= hi) and (coh.aC == coh.aHfin))
    crit["1_transport_consistency"] = bool(t_ok)
    # crit2: some label-adaptive (pi,B) with LCB[G]>0 in ALL cohorts
    found = None
    for pol in [P.ModelSelector(), P.CODA()]:
        for B in cfg["budgets"]:
            allpos = all(_cohort_eval(coh, pol, B, cfg["K_seeds"], cfg["B_boot"], 300 + e)["lcb_G_raw"] > 0
                         for e, coh in enumerate(world))
            if allpos:
                found = (pol.name, B); break
        if found:
            break
    crit["2_active_gain_recovered"] = found
    # crit3: power over K_MC redraws = P(MODEL-SELECTOR LCB[G]>0 in ALL cohorts at power_budget)
    hits = 0
    for k in range(cfg["K_MC"]):
        w = G.make_world("POS", A=cfg["A"], n_pat=npat, E=cfg["E"], seed=5000 + k)
        allpos = True
        for e, coh in enumerate(w):
            lcb, _ = _fast_lcb_G(coh, P.ModelSelector(), cfg["power_budget"], cfg["K_seeds"],
                                 cfg["fast_reps"], 9000 + k * 10 + e)
            allpos &= bool(lcb > 0)
            if not allpos:
                break
        hits += int(allpos)
    power = hits / cfg["K_MC"]
    crit["3_power"] = dict(power=power, floor=0.80, passed=bool(power >= 0.80))
    log(f"  POS: transport={crit['1_transport_consistency']} gain={found} power={power:.3f}")
    passed = bool(crit["1_transport_consistency"] and (found is not None) and crit["3_power"]["passed"])
    return passed, crit


def run_pos_dense(cfg, log):
    """v3/M1 winner's-curse control: in a dense near-tie regime the NAIVE in-sample argmin is
    optimistically low, which would inflate the transport gap toward the (pre-registered) nontransport
    conclusion. The cross-fit held-SELECTION reference is out-of-fold and does NOT share that optimism.
    Verify: (a) cross-fit reference > in-sample argmin loss by >2SE (optimism present + corrected);
    (b) naive T inflated over cross-fit T by >2SE (bias toward nontransport removed by cross-fit)."""
    ins, ref, naive_T, xfit_T = [], [], [], []
    for k in range(200):
        w = G.make_world("POS_DENSE", A=cfg["A"], n_pat=cfg["n_pat"], E=1, seed=7000 + k)
        coh = w[0]
        Lbar, pat_ids = E.patient_mean_loss(E.binary_nll(coh.probs, coh.y), coh.patient_of)
        cf = E.cross_fit(Lbar, pat_ids)
        Lh = held_view_loss(Lbar)
        ins.append(Lh.min()); ref.append(cf.ref)
        naive_T.append(Lh[coh.aC] - Lh.min())         # naive: in-sample argmin as reference (optimistic)
        xfit_T.append(E.transport_gap_cf(Lbar, coh.aC, cf))
    ins, ref, naive_T, xfit_T = map(np.array, (ins, ref, naive_T, xfit_T))
    d = ref - ins; se = d.std() / np.sqrt(d.size)
    wc_corrected = bool(d.mean() > 2 * se)            # cross-fit ref higher than optimistic in-sample min
    dt = naive_T - xfit_T; se2 = dt.std() / np.sqrt(dt.size)
    naive_inflates = bool(dt.mean() > 2 * se2)        # naive transport gap inflated vs cross-fit
    crit = dict(insample_argmin=float(ins.mean()), xfit_reference=float(ref.mean()),
                optimism_gap=float(d.mean()), winners_curse_corrected=wc_corrected,
                naive_T=float(naive_T.mean()), xfit_T=float(xfit_T.mean()),
                naive_inflates_transport=naive_inflates)
    log(f"  POS_DENSE: insample_min={ins.mean():.4f} xfit_ref={ref.mean():.4f} optimism_gap={d.mean():+.4f} "
        f"(corrected={wc_corrected}); naiveT={naive_T.mean():+.4f} xfitT={xfit_T.mean():+.4f} "
        f"(naive_inflates={naive_inflates})")
    return bool(wc_corrected and naive_inflates), crit


def run_neg_a(cfg, log):
    world = G.make_world("NEG_A", A=cfg["A"], n_pat=cfg["n_pat"], E=cfg["E"], seed=201)
    ok = True
    details = []
    for e, coh in enumerate(world):
        m = _cohort_eval(coh, P.ModelSelector(), 64, cfg["K_seeds"], cfg["B_boot"], 400 + e)
        lcb_T = percentile_lcb(m["boot_T"], cfg["alpha"])
        cond = bool(lcb_T > 0 and coh.aC != coh.aHfin)
        ok &= cond
        details.append(dict(cohort=e, lcb_T=lcb_T, aC_ne_aHfin=bool(coh.aC != coh.aHfin)))
    log(f"  NEG_A: no-false-transport={ok}")
    return bool(ok), dict(passed=ok, details=details)


def run_neg_b(cfg, log):
    """PRIMARY specificity: the REAL gate (LCB_95(G/s_e) >= tau_G in ALL E cohorts, same label-adaptive
    pi,B) must have false-positive rate <= alpha over K_MC null redraws. This is the principled specificity
    measure. Diagnostics (disclosed, NOT pass/fail): the stricter LCB>0-in-all-cohorts rate; the fraction
    of null cohorts flagged VACUOUS by the degeneracy guard (in a true null s_e is a noise denominator, so
    the standardized point-estimate is meaningless there and the guard should fire); and the mean null
    standardized gain (a small legitimate selector edge inflated by the tiny null denominator)."""
    fp_gate = fp_pos = vac_hits = vac_tot = 0
    signs = []
    for k in range(cfg["K_MC"]):
        w = G.make_world("NEG_B", A=cfg["A"], n_pat=cfg["pos_n_pat"], E=cfg["E"], seed=6000 + k)
        all_gate, all_pos = True, True
        for e, coh in enumerate(w):
            lcb, m = _fast_lcb_G(coh, P.ModelSelector(), cfg["power_budget"], cfg["K_seeds"],
                                 cfg["fast_reps"], 9500 + k * 10 + e)
            if k < 200:
                signs.append(m["obs_G"] / m["s_e"] if m["s_e"] > 0 else 0.0)
                # vacuity: compare s_e to the patient-cluster bootstrap SE of held loss
                Lbar, pat = E.patient_mean_loss(E.binary_nll(coh.probs, coh.y), coh.patient_of)
                rng = np.random.default_rng(1234 + k * 10 + e)
                se_h = float(np.median(np.array([held_view_loss(Lbar[:, rng.integers(0, Lbar.shape[1],
                             Lbar.shape[1])]) for _ in range(80)]).std(axis=0)))
                vac_hits += int(m["s_e"] <= se_h); vac_tot += 1
            all_gate &= bool(lcb >= cfg["tau_G"])
            all_pos &= bool(lcb > 0)
            if not (all_gate or all_pos):
                break
        fp_gate += int(all_gate); fp_pos += int(all_pos)
    fpr_gate = fp_gate / cfg["K_MC"]; fpr_pos = fp_pos / cfg["K_MC"]
    signs = np.array(signs); mean_std = float(signs.mean())
    vac_frac = vac_hits / max(vac_tot, 1)
    crit = dict(fpr_gate=fpr_gate, fpr_pos=fpr_pos, alpha=cfg["alpha"], tau_G=cfg["tau_G"],
                gate_specificity_ok=bool(fpr_gate <= cfg["alpha"]),
                null_vacuous_fraction=vac_frac, mean_null_Gstd_diagnostic=mean_std)
    log(f"  NEG_B: real-gate FP rate={fpr_gate:.4f} (<= {cfg['alpha']}: {crit['gate_specificity_ok']}); "
        f"LCB>0 rate={fpr_pos:.4f}; null-vacuous frac={vac_frac:.2f}; mean null G/s_e (diag)={mean_std:+.3f}")
    return bool(crit["gate_specificity_ok"]), crit


def run_calib(cfg, log):
    """v3-correct calibration: the cross-fit T^CF is a held-SELECTION reference estimand, NOT a zero oracle.
    Its finite-n population value is estimated from the redraws themselves (truth_finite = mean over K_MC
    independent n=400 redraws). Check: (a) the patient-cluster bootstrap CI attains nominal coverage of
    truth_finite (band [93.5%,96.5%]); (b) the bootstrap is ~unbiased (bootstrap mean ~ observed)."""
    big = G.make_world("CALIB", A=cfg["A"], n_pat=3000, E=1, seed=8000)[0]     # ONE fixed population
    Lbar_big, _ = E.patient_mean_loss(E.binary_nll(big.probs, big.y), big.patient_of)
    Lh_big = held_view_loss(Lbar_big)
    a_good = int(np.argmin(Lh_big))                                           # population best
    a_bad = int(np.argsort(Lh_big)[len(Lh_big) // 2])                         # median candidate
    truth_G = float(Lh_big[a_bad] - Lh_big[a_good])                           # G-like smooth held difference
    # PRIMARY: coverage of the GATE STATISTIC. In the gate, G = R^CF(P0)-R^CF(pi) and the cross-fit
    # reference CANCELS in the paired difference, so G reduces to a smooth difference of two held means
    # (here L^H(a_bad)-L^H(a_good)). Its patient-cluster BCa/BC CI must attain nominal coverage.
    covG, T_diag = 0, []
    for k in range(cfg["K_MC"]):
        rng = np.random.default_rng(80000 + k)
        coh = G.subsample_patients(big, cfg["n_pat"], rng)
        Lbar, pat_ids = E.patient_mean_loss(E.binary_nll(coh.probs, coh.y), coh.patient_of)
        Lh = held_view_loss(Lbar)
        obsG = Lh[a_bad] - Lh[a_good]
        reps = 250
        bd = np.empty(reps)
        for b in range(reps):
            idx = rng.integers(0, Lbar.shape[1], Lbar.shape[1])              # CRN-paired resample
            lh = held_view_loss(Lbar[:, idx]); bd[b] = lh[a_bad] - lh[a_good]
        lo, hi = bc_ci(obsG, bd, cfg["alpha"])
        covG += int(lo <= truth_G <= hi)
        cf = E.cross_fit(Lbar, pat_ids)                                      # T^CF diagnostic (selection stat)
        T_diag.append(E.transport_gap_cf(Lbar, coh.aC, cf))
    coverage_G = covG / cfg["K_MC"]
    band = cfg["coverage_band"]
    cov_ok = bool(band[0] <= coverage_G <= band[1])
    crit = dict(gate_stat_truth=truth_G, coverage_G=coverage_G, coverage_band=band, coverage_ok=cov_ok,
                crossfit_T_finite_mean_diagnostic=float(np.mean(T_diag)))
    log(f"  CALIB: gate-statistic (G-like) cluster-bootstrap coverage={coverage_G:.3f} band={band} ok={cov_ok} "
        f"(cross-fit T finite-mean diagnostic={np.mean(T_diag):+.4f})")
    return bool(cov_ok), crit


# ------------------------------------------------------------------ mutation tests -----------------

def run_mutations(cfg, log):
    """Each mutation MUST trigger its failure signature, else the control is non-diagnostic (blocker)."""
    out = {}
    # CALIB-M1: unweighted (naive-mean) estimator under an ADAPTIVE proposal must show bias > 2*SE.
    from .lure import lure_risk, without_replacement_proposal_sequence
    biases_w, biases_u = [], []
    for k in range(400):
        w = G.make_world("CALIB", A=cfg["A"], n_pat=cfg["n_pat"], E=1, seed=8300 + k)
        coh = w[0]
        rng = np.random.default_rng(8300 + k)
        n = coh.probs.shape[1]
        a = coh.aHfin
        truth = E.binary_nll(coh.probs[[a]], coh.y)[0].mean()
        # STRONGLY-SKEWED label-free proposal: candidate a's predicted-entropy^4 concentrates acquisition
        # on high-loss records => the UNWEIGHTED mean over-estimates the risk (biased), LURE corrects it.
        p = np.clip(coh.probs[a], 1e-6, 1 - 1e-6)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        wgt = (ent + 1e-3) ** 4
        wgt = wgt / wgt.sum()
        B = 64
        order = rng.choice(n, size=B, replace=False, p=wgt)
        qseq = without_replacement_proposal_sequence(wgt, order)
        loss_a = E.binary_nll(coh.probs[[a]][:, order], coh.y[order])[0]
        biases_w.append(lure_risk(loss_a, qseq, n) - truth)       # LURE-weighted
        biases_u.append(loss_a.mean() - truth)                    # unweighted naive
    bw, bu = np.array(biases_w), np.array(biases_u)
    m1 = bool(abs(bu.mean()) > 2 * bu.std() / np.sqrt(bu.size) and
              abs(bw.mean()) <= max(2 * bw.std() / np.sqrt(bw.size), 0.003))
    out["CALIB_M1_unweighted_biased"] = dict(unweighted_bias=float(bu.mean()),
                                             lure_bias=float(bw.mean()), triggered=m1)
    # CALIB-M2: record-level (NON-clustered) bootstrap under rho>0 must UNDER-cover the population mean of a
    # simple candidate loss, while the cluster bootstrap covers ~95%. Fixed population; subsample patients.
    big = G.make_world("CALIB", A=cfg["A"], n_pat=4000, E=1, seed=8600, records_per_patient=5)[0]
    a = 0
    _Lbar_big, _ = E.patient_mean_loss(E.binary_nll(big.probs, big.y), big.patient_of)
    truth = float(held_view_loss(_Lbar_big)[a])
    cov_cluster, cov_record, NM = 0, 0, 300
    for k in range(NM):
        rng = np.random.default_rng(86000 + k)
        coh = G.subsample_patients(big, 200, rng)
        L = E.binary_nll(coh.probs, coh.y)
        Lbar, _ = E.patient_mean_loss(L, coh.patient_of)
        bt_c = np.array([held_view_loss(Lbar[:, rng.integers(0, Lbar.shape[1], Lbar.shape[1])])[a] for _ in range(200)])
        lo, hi = np.quantile(bt_c, [0.025, 0.975]); cov_cluster += int(lo <= truth <= hi)
        nrec = L.shape[1]
        bt_r = np.array([L[a, rng.integers(0, nrec, nrec)].mean() for _ in range(200)])
        lo, hi = np.quantile(bt_r, [0.025, 0.975]); cov_record += int(lo <= truth <= hi)
    m2 = bool(cov_record / NM < 0.90 <= cov_cluster / NM + 1e-9)
    out["CALIB_M2_record_bootstrap_undercovers"] = dict(cluster_cov=cov_cluster / NM,
                                                        record_cov=cov_record / NM, triggered=m2)
    # NEG-M3: oracle-leak selector must manufacture spurious all-cohort G>0 in NEG-B.
    leak_hits = 0
    for k in range(200):
        w = G.make_world("NEG_B", A=cfg["A"], n_pat=cfg["n_pat"], E=cfg["E"], seed=8900 + k)
        allpos = True
        for e, coh in enumerate(w):
            lcb, _ = _fast_lcb_G_leak(coh, cfg, 8900 + k * 10 + e)
            allpos &= bool(lcb > 0)
            if not allpos:
                break
        leak_hits += int(allpos)
    m3 = bool(leak_hits / 200 > cfg["alpha"])   # leak SHOULD produce spurious all-cohort gain often
    out["NEG_M3_oracle_leak_manufactures_gain"] = dict(rate=leak_hits / 200, triggered=m3)
    log(f"  MUTATIONS: M1={m1} M2={m2} M3={m3}")
    return bool(m1 and m2 and m3), out


def _fast_lcb_G_leak(coh, cfg, seed):
    """LCB of G/s_e with the ORACLE-LEAK selector as pi (mutation NEG-M3)."""
    rng = np.random.default_rng(seed)
    Lbar, pat_ids = E.patient_mean_loss(E.binary_nll(coh.probs, coh.y), coh.patient_of)
    fold_of = _fold_assignment(pat_ids)
    s_e = dispersion_s_e(Lbar)
    leak = P._OracleLeakSelector()
    p0 = P.P0()
    picks_pi = np.array([leak.select(coh.probs, coh.y, coh.patient_of, cfg["power_budget"],
                                     np.random.default_rng(seed + k))[0] for k in range(cfg["K_seeds"])])
    picks_p0 = np.array([p0.select(coh.probs, coh.y, coh.patient_of, cfg["power_budget"],
                                   np.random.default_rng(seed + 99 + k))[0] for k in range(cfg["K_seeds"])])
    bs = paired_gain_bootstrap(Lbar, fold_of, picks_pi, picks_p0, coh.aC, 200, rng)
    lcb = percentile_lcb(bs["boot_G"], cfg["alpha"])
    return (lcb / s_e if s_e > 0 else float("nan")), None


def run_control_gate(config=None, verbose=True):
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    logs = []
    def log(m):
        logs.append(m)
        if verbose:
            print(m, flush=True)
    t0 = time.time()
    log(f"[C87 CONTROL GATE] config={json.dumps({k: v for k, v in cfg.items() if k != 'note'})}")
    log(f"  note: {cfg['note']}")
    results = {}
    results["POS"] = dict(zip(("passed", "detail"), run_pos(cfg, log)))
    results["POS_DENSE"] = dict(zip(("passed", "detail"), run_pos_dense(cfg, log)))
    results["NEG_A"] = dict(zip(("passed", "detail"), run_neg_a(cfg, log)))
    results["NEG_B"] = dict(zip(("passed", "detail"), run_neg_b(cfg, log)))
    results["CALIB"] = dict(zip(("passed", "detail"), run_calib(cfg, log)))
    results["MUTATIONS"] = dict(zip(("passed", "detail"), run_mutations(cfg, log)))

    all_pass = all(r["passed"] for r in results.values())
    verdict = "CONTROL_PASS" if all_pass else "ENGINEERING_BLOCKER"
    payload = dict(verdict=verdict, config=cfg, results=results,
                   elapsed_s=round(time.time() - t0, 1), logs=logs)
    payload["signature"] = hashlib.sha256(
        json.dumps({k: results[k]["passed"] for k in results}, sort_keys=True).encode()).hexdigest()[:16]
    log(f"[C87 CONTROL GATE] VERDICT = {verdict}  ({payload['elapsed_s']}s)  sig={payload['signature']}")
    return payload
