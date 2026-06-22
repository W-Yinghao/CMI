"""ACAR go/no-go harness v2 (leak-proof). Implements notes/ACAR_FROZEN_v2.md.

Phase-1 (y-FREE, via scoring.score_actions): per natural batch & action -> label-free φ_a. Phase-2 (y ONLY here):
paired incremental risk ΔR_a and harm = 1[ΔR_a>0]. PRIMARY endpoint = disease-stratified SUBJECT-clustered CV:
subject-disjoint FIT/CAL/EVAL folds; ĝ_a on FIT; joint conformal q on CAL subjects; G1/G2/coverage on EVAL.
SECONDARY = LOCO cohort descriptive robustness (no coverage claim). Guards: metamorphic φ-invariance + ΔR
label-sensitivity (here) and the full route_batch/serialize guards in acar/tests/. Decision: PROCEED /
MEASUREMENT_ONLY / TERMINATE (binding only because the protocol is now valid).

Run:  ACAR_FEAT_DUMP=<dir> python -m acar.run_gonogo --alpha 0.1 --delta 0.0 --out results/acar_gonogo
"""
from __future__ import annotations
import argparse, json, os, hashlib, subprocess
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C
from .config import (ACARConfig, DISEASE, ACTIONS, NON_IDENTITY, PAIRED_FEATURES, AUROC_GATE, RETAIN_FRAC, FIT_FRAC)
from .data import load_all, dump_sha256
from .scoring import score_actions
from .risk import delta_risk, harm_label
from .conformal import subject_fold, split_fit_cal, fit_routers
from .deploy import route_fvec


# ---------- records (Phase-1 + Phase-2) ----------
def build_records(data, cfg, guard_fail):
    recs = []
    for disease, cohorts in data.items():
        for coh in cohorts:
            for b in coh.batches:
                base = dict(disease=disease, cohort=coh.cohort, subject=b.subject, fallback=bool(b.fallback), n=len(b.z))
                if b.fallback:                                # retained, forced identity, never adapted
                    recs.append({**base, "fvec": None, "phi": None,
                                 "dr": {a: 0.0 for a in ACTIONS}, "harm": {a: 0 for a in ACTIONS}})
                    continue
                s1 = score_actions(coh.state, b.z, NON_IDENTITY)     # label-free (no y); determinism covered by the
                # unit guards + the outer double-run hash, so we do NOT recompute the expensive action set per batch.
                p0 = s1["identity"]["p"]
                dr = {"identity": 0.0}; harm = {"identity": 0}
                for a in NON_IDENTITY:
                    d = float(delta_risk(p0, s1[a]["p"], b.y, cfg.risk)); dr[a] = d; harm[a] = harm_label(d)
                # ΔR must be label-sensitive (estimand really uses y)
                yperm = np.random.default_rng(cfg.seed + 7).permutation(b.y)
                if abs(delta_risk(p0, s1[NON_IDENTITY[0]]["p"], yperm, cfg.risk) - dr[NON_IDENTITY[0]]) < 1e-12 and np.std(b.y) > 0:
                    guard_fail.append(f"{disease}/{coh.cohort}: ΔR insensitive to y")
                recs.append({**base, "fvec": {a: s1[a]["fvec"] for a in NON_IDENTITY},
                             "phi": {a: s1[a]["phi"] for a in NON_IDENTITY}, "dr": dr, "harm": harm})
    return recs


# ---------- metrics ----------
def _auc(s, lab):
    s = np.asarray(s, float); lab = np.asarray(lab)
    ok = np.isfinite(s); s, lab = s[ok], lab[ok]
    if len(set(lab.tolist())) != 2 or np.std(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(lab, s))


def _orient(fit_recs, action, feat):
    """Train-only orientation sign for a paired feature (+1 if higher => more harm on FIT, else -1)."""
    vals = [r["phi"][action][feat] for r in fit_recs]
    harm = [r["harm"][action] for r in fit_recs]
    a = _auc(vals, harm)
    if np.isnan(a):
        return 1.0
    return 1.0 if a >= 0.5 else -1.0


def _best_fixed(fit_recs):
    """Best fixed-action policy selected on FIT subjects ONLY (never CAL/EVAL) — so CAL/EVAL labels cannot leak into
    the baseline ACAR must beat (review: split isolation). identity has reduction 0 by definition."""
    red = {f: (0.0 if f == "identity" else -np.mean([r["dr"][f] for r in fit_recs])) for f in ACTIONS}
    return max(ACTIONS, key=lambda f: red[f])


# ---------- PRIMARY: subject-clustered CV per disease ----------
def subject_cv(recs, cfg):
    out = {}
    nonfb = [r for r in recs if not r["fallback"]]
    # train-only orientation per (action,feature) from ALL FIT subjects pooled (folds chosen below; here use full
    # non-EVAL pool per fold). For a single global sign we orient on the full non-fallback set MINUS each EVAL fold;
    # to keep one stable sign we compute it per fold and store the oriented EVAL values.
    for disease in DISEASE:
        drecs = [r for r in recs if r["disease"] == disease]
        dnonfb = [r for r in drecs if not r["fallback"]]
        subjects = sorted({r["subject"] for r in drecs})
        # accumulators (pooled EVAL across folds)
        feat_vals = {a: {f: [] for f in PAIRED_FEATURES} for a in NON_IDENTITY}      # (oriented_value, harm, cohort)
        feat_harm = {a: {f: [] for f in PAIRED_FEATURES} for a in NON_IDENTITY}
        feat_coh = {a: {f: [] for f in PAIRED_FEATURES} for a in NON_IDENTITY}
        reg_pred = {a: [] for a in NON_IDENTITY}; reg_harm = {a: [] for a in NON_IDENTITY}
        routed_dr, routed_coh, abstain = [], [], []
        fixed_dr = {f: [] for f in ACTIONS}
        bestfixed_dr = []
        oracle_benefit, router_benefit = [], []
        cov_subj = defaultdict(lambda: True); cov_seen = set()
        q_finite_folds = 0; n_folds_used = 0
        folds_log = []                                        # per-fold n_fit/n_cal/n_eval/k/q (manifest)
        eval_trace = []                                       # record-level routing trace (determinism hash)
        for j in range(cfg.k_folds):
            eval_subj = {s for s in subjects if subject_fold(s, cfg.k_folds, cfg.seed) == j}
            train_subj = [s for s in subjects if s not in eval_subj]
            if not eval_subj or not train_subj:
                continue
            fit_subj, cal_subj = split_fit_cal(train_subj, FIT_FRAC, cfg.seed)
            fit_recs = [r for r in dnonfb if r["subject"] in fit_subj]
            cal_recs = [r for r in dnonfb if r["subject"] in cal_subj]
            if len(fit_recs) < 8 or not cal_recs:
                continue
            n_folds_used += 1
            routers, diag = fit_routers(fit_recs, cal_recs, NON_IDENTITY, cfg.alpha, cfg.delta, cfg.seed)
            q_finite_folds += int(diag["q_informative"])
            signs = {a: {f: _orient(fit_recs, a, f) for f in PAIRED_FEATURES} for a in NON_IDENTITY}
            bestfixed = _best_fixed(fit_recs)                 # FIT-only (no CAL/EVAL label leak)
            eval_records = [r for r in drecs if r["subject"] in eval_subj]
            folds_log.append(dict(disease=disease, fold=j, n_fit=diag["n_fit"], n_cal=diag["n_cal"],
                                  n_eval=len(eval_records), k=diag["k"], q=diag["q"], bestfixed=bestfixed,
                                  fit_subj=sorted(fit_subj), cal_subj=sorted(cal_subj), eval_subj=sorted(eval_subj),
                                  cal_scores=sorted(round(s, 8) for s in diag["cal_scores"])))
            # EVAL: all records of EVAL subjects (incl fallback -> identity)
            for r in eval_records:
                if r["fallback"]:
                    routed_dr.append(0.0); routed_coh.append(r["cohort"]); abstain.append(True)
                    for f in ACTIONS:
                        fixed_dr[f].append(0.0)
                    bestfixed_dr.append(0.0); oracle_benefit.append(0.0); router_benefit.append(0.0)
                    continue
                chosen, U = route_fvec(routers, r["fvec"])
                rd = 0.0 if chosen == "identity" else r["dr"][chosen]
                routed_dr.append(rd); routed_coh.append(r["cohort"]); abstain.append(chosen == "identity")
                eval_trace.append((r["subject"], r["cohort"], j, chosen, tuple(round(U[a], 8) for a in NON_IDENTITY)))
                for f in ACTIONS:
                    fixed_dr[f].append(0.0 if f == "identity" else r["dr"][f])
                bestfixed_dr.append(0.0 if bestfixed == "identity" else r["dr"][bestfixed])
                ob = max(0.0, max(-r["dr"][a] for a in ACTIONS)); oracle_benefit.append(ob)
                router_benefit.append(max(0.0, -rd))
                # coverage (subject-level): all actions' ΔR <= U
                cov_seen.add(r["subject"])
                if not all(r["dr"][a] <= U[a] for a in NON_IDENTITY):
                    cov_subj[r["subject"]] = False
                # G1 accumulation
                for a in NON_IDENTITY:
                    for f in PAIRED_FEATURES:
                        feat_vals[a][f].append(signs[a][f] * r["phi"][a][f])
                        feat_harm[a][f].append(r["harm"][a]); feat_coh[a][f].append(r["cohort"])
                    reg_pred[a].append(float(routers.regs[a].predict(r["fvec"][a][None])[0]))
                    reg_harm[a].append(r["harm"][a])
        # ---- G1 per action (pooled out-of-fold AUROC + explicit per-cohort evaluability) ----
        g1 = {}
        for a in NON_IDENTITY:
            feats, per_cohort = {}, {}
            for f in PAIRED_FEATURES:
                feats[f] = _auc(feat_vals[a][f], feat_harm[a][f])
                vals = np.asarray(feat_vals[a][f], float); harm = np.asarray(feat_harm[a][f]); coh = np.asarray(feat_coh[a][f])
                pc = {}
                for c in DISEASE[disease]:
                    m = coh == c
                    pc[c] = (_auc(vals[m], harm[m]) if m.any() else float("nan"))
                per_cohort[f] = pc
            reg_auc = _auc(reg_pred[a], reg_harm[a])
            g1[a] = dict(feature_auc=feats, regressor_auc=reg_auc, per_cohort_feature_auc=per_cohort)
        # ---- G2 ----
        rd = np.array(routed_dr); n = len(rd)
        red_router = float(-rd.mean()) if n else float("nan")
        red_fixed = {f: (float(-np.mean(fixed_dr[f])) if fixed_dr[f] else float("nan")) for f in ACTIONS}
        red_bestfixed = float(-np.mean(bestfixed_dr)) if bestfixed_dr else float("nan")
        abst_rate = float(np.mean(abstain)) if n else float("nan")
        adapt_coverage = 1 - abst_rate                       # p = fraction of batches actually adapted
        # matched-coverage random policy: adapt a RANDOM p-fraction with best-fixed, abstain the rest. The abstained
        # contribute 0 reduction, so E[reduction] = p * red_bestfixed (p = ADAPTATION coverage, not abstention).
        red_random = adapt_coverage * red_bestfixed if np.isfinite(red_bestfixed) else float("nan")
        denom = float(np.sum(oracle_benefit)); retention = float(np.sum(router_benefit) / denom) if denom > 1e-12 else float("nan")
        # cohort-macro: router helps (mean ΔR<0) per cohort
        coh_help = {}
        for c in sorted(set(routed_coh)):
            vals = [routed_dr[i] for i in range(n) if routed_coh[i] == c]
            coh_help[c] = bool(np.mean(vals) < 0) if vals else None
        n_eval_coh = sum(1 for v in coh_help.values() if v is not None)
        cohort_macro_pass = (n_eval_coh >= 2 and sum(1 for v in coh_help.values() if v) >= n_eval_coh - 1)
        coverage = float(np.mean([cov_subj[s] for s in cov_seen])) if cov_seen else float("nan")
        out[disease] = dict(
            g1=g1, n_eval_batches=n, abstain_rate=abst_rate, adapt_coverage=adapt_coverage,
            red_router=red_router, red_fixed=red_fixed, red_bestfixed=red_bestfixed, red_random=red_random,
            retention=retention, cohort_help=coh_help, cohort_macro_pass=bool(cohort_macro_pass),
            coverage=coverage, q_informative_folds=q_finite_folds, n_folds=n_folds_used,
            folds_log=folds_log, eval_trace=sorted(eval_trace, key=lambda t: (t[0], t[1], t[2])))
    return out


def _stable_across_cohorts(cv, a, f):
    """Stability with the denominator FIXED at n_total (all cohorts), not the evaluable count — so undefined AUROCs
    cannot silently disappear (review point 1). Require BOTH: the feature is evaluable (non-NaN oriented AUROC) in
    >= n_total-1 cohorts, AND oriented AUROC > 0.5 in >= n_total-1 cohorts. (v2 §A3, tightened)"""
    n_total = sum(len(DISEASE[d]) for d in DISEASE)          # 7 cohorts
    aucs = []
    for d in DISEASE:
        pc = cv[d]["g1"][a]["per_cohort_feature_auc"][f]
        aucs += [pc[c] for c in DISEASE[d]]                  # include NaNs (non-evaluable cohorts)
    n_evaluable = sum(1 for v in aucs if not np.isnan(v))
    n_pass = sum(1 for v in aucs if not np.isnan(v) and v > 0.5)
    return n_evaluable >= n_total - 1 and n_pass >= n_total - 1


def passes_g1(cv):
    """>=1 non-identity action with a paired feature (pooled AUROC>=GATE on BOTH diseases AND per-cohort stable)
    OR ĝ_a (pooled AUROC>=GATE on BOTH diseases)."""
    for a in NON_IDENTITY:
        feat_ok = any(
            (not np.isnan(cv["PD"]["g1"][a]["feature_auc"][f]) and not np.isnan(cv["SCZ"]["g1"][a]["feature_auc"][f])
             and cv["PD"]["g1"][a]["feature_auc"][f] >= AUROC_GATE and cv["SCZ"]["g1"][a]["feature_auc"][f] >= AUROC_GATE
             and _stable_across_cohorts(cv, a, f))
            for f in PAIRED_FEATURES)
        reg_ok = (not np.isnan(cv["PD"]["g1"][a]["regressor_auc"]) and not np.isnan(cv["SCZ"]["g1"][a]["regressor_auc"])
                  and cv["PD"]["g1"][a]["regressor_auc"] >= AUROC_GATE and cv["SCZ"]["g1"][a]["regressor_auc"] >= AUROC_GATE)
        if feat_ok or reg_ok:
            return True, a
    return False, None


def passes_g2(cv):
    for d in DISEASE:
        r = cv[d]
        if not np.isfinite(r["red_router"]) or not np.isfinite(r["retention"]):
            return False
        cond = (r["red_router"] > 0 and r["red_router"] > r["red_bestfixed"] and
                r["red_router"] > r["red_random"] and r["cohort_macro_pass"] and r["retention"] >= RETAIN_FRAC)
        if not cond:
            return False
    return True


# ---------- SECONDARY: LOCO descriptive robustness (no coverage claim) ----------
def loco_descriptive(recs, cfg):
    out = {}
    for disease in DISEASE:
        cohs = DISEASE[disease]; per = {}
        for held in cohs:
            tr = [r for r in recs if r["disease"] == disease and r["cohort"] != held and not r["fallback"]]
            te = [r for r in recs if r["disease"] == disease and r["cohort"] == held]
            te_nf = [r for r in te if not r["fallback"]]
            if len(tr) < 8 or not te_nf:
                per[held] = None; continue
            routers, _ = fit_routers(tr, tr, NON_IDENTITY, cfg.alpha, cfg.delta, cfg.seed)   # in-cohort cal (descriptive only)
            routed = []
            for r in te:
                if r["fallback"]:
                    routed.append(0.0); continue
                chosen, _ = route_fvec(routers, r["fvec"]); routed.append(0.0 if chosen == "identity" else r["dr"][chosen])
            always = [r["dr"]["matched_coral"] for r in te_nf]
            per[held] = dict(red_router=float(-np.mean(routed)), red_always_coral=float(-np.mean(always)),
                             harm_always=int(np.sum(np.array(always) > 0)),
                             harm_router=int(np.sum(np.array([x for x in routed]) > 0)))
        out[disease] = per
    return out


# ---------- driver ----------
def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _record_digest(recs):
    """RECORD-LEVEL φ_a and ΔR_a (review point 2): aggregate equality must not conceal different inputs."""
    rows = []
    for r in sorted(recs, key=lambda x: (x["disease"], x["cohort"], x["subject"], x["n"])):
        fv = None if r["fvec"] is None else {a: [round(float(v), 8) for v in r["fvec"][a]] for a in NON_IDENTITY}
        rows.append((r["disease"], r["cohort"], r["subject"], r["n"], r["fallback"],
                     fv, {a: round(float(r["dr"][a]), 8) for a in ACTIONS}))
    return rows


def canonical_hash(cv, recs, decision):
    """Full record-level digest: split assignments, φ_a, ΔR_a, ĝ_a (via U_a), selected action, subject scores, q,
    AND aggregate metrics. Two runs that route any batch differently get different hashes."""
    core = dict(decision=decision, records=_record_digest(recs))
    for d in DISEASE:
        r = cv[d]
        core[d] = dict(
            red_router=round(r["red_router"], 8), red_bestfixed=round(r["red_bestfixed"], 8),
            red_random=round(r["red_random"], 8) if np.isfinite(r["red_random"]) else None,
            retention=round(r["retention"], 8) if np.isfinite(r["retention"]) else None,
            coverage=round(r["coverage"], 8) if np.isfinite(r["coverage"]) else None,
            cohort_macro_pass=r["cohort_macro_pass"],
            folds=r["folds_log"],                            # n_fit/n_cal/n_eval/k/q + split assignments + cal scores
            eval_trace=r["eval_trace"],                      # per-record (subject,cohort,fold,chosen,U_a)
            g1={a: dict(feat={f: (round(v, 8) if v == v else None) for f, v in r["g1"][a]["feature_auc"].items()},
                        per_cohort={f: {c: (round(v, 8) if v == v else None) for c, v in pc.items()}
                                    for f, pc in r["g1"][a]["per_cohort_feature_auc"].items()},
                        reg=(round(r["g1"][a]["regressor_auc"], 8) if r["g1"][a]["regressor_auc"] == r["g1"][a]["regressor_auc"] else None))
                for a in NON_IDENTITY})
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=C.B)
    ap.add_argument("--risk", choices=["nll", "01"], default="nll")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/acar_gonogo")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = ACARConfig(alpha=a.alpha, delta=a.delta, batch=a.batch, risk=a.risk, seed=a.seed, out=a.out)

    data = load_all(batch_size=cfg.batch)
    if a.smoke:
        data = {d: cohs[:2] for d, cohs in data.items()}

    guard_fail = []
    recs = build_records(data, cfg, guard_fail)
    if guard_fail:
        print("METAMORPHIC GUARD FAILED (label leakage):"); [print("  ", g) for g in guard_fail[:10]]; raise SystemExit(3)
    cv = subject_cv(recs, cfg)
    G1, g1_action = passes_g1(cv); G2 = bool(passes_g2(cv))
    decision = "PROCEED" if (G1 and G2) else ("MEASUREMENT_ONLY" if G1 else "TERMINATE")
    h1 = canonical_hash(cv, recs, decision)

    if not a.smoke:
        gf2 = []
        recs2 = build_records(data, cfg, gf2)
        if gf2:
            print("GUARD FAILED on rerun"); raise SystemExit(3)
        cv2 = subject_cv(recs2, cfg); G1b, _ = passes_g1(cv2); G2b = bool(passes_g2(cv2))
        h2 = canonical_hash(cv2, recs2, "PROCEED" if (G1b and G2b) else ("MEASUREMENT_ONLY" if G1b else "TERMINATE"))
        if h1 != h2:
            print(f"NON-DETERMINISTIC ({h1} != {h2}) — abort"); raise SystemExit(4)

    loco = loco_descriptive(recs, cfg)
    out = f"{cfg.out}/{_git_sha()}_{cfg.config_hash()}"
    os.makedirs(out, exist_ok=True)
    summary = dict(decision=decision, G1=bool(G1), G1_action=g1_action, G2=G2,
                   metamorphic_guard="PASS", double_run_hash=h1, pre_registration="notes/ACAR_FROZEN_v2.md",
                   alpha=cfg.alpha, delta=cfg.delta, risk=cfg.risk, k_folds=cfg.k_folds,
                   subject_cv=cv, loco_descriptive=loco, n_records=len(recs))
    json.dump(summary, open(f"{out}/acar_gonogo_summary.json", "w"), indent=2, default=str)
    manifest = dict(pre_registration="notes/ACAR_FROZEN_v2.md", git_sha=_git_sha(), config_hash=cfg.config_hash(),
                    double_run_hash=h1, feat_dump=C.feat_dump_dir(), alpha=cfg.alpha, delta=cfg.delta, seed=cfg.seed,
                    coverage_event=("Pr[ forall B in B(S), forall a: dR_a(B) <= g_a(phi_a(B)) + q ] >= 1-alpha; "
                                    "B(S) = fixed finite batching protocol; NOT an unbounded future stream; "
                                    "stratified by DISEASE TASK FAMILY (PD vs SCZ) only, NEVER by the HC/Patient "
                                    "label y; exchangeability claimed w.r.t. disease-stratified CAL subjects only; "
                                    "calibration unit = subject cluster; m = per-fold CAL subjects; LOCO = empirical."),
                    folds={d: cv[d]["folds_log"] for d in DISEASE},   # per disease/fold: n_fit,n_cal,n_eval,k,q
                    dump_sha256={f"{d}/{c}": dump_sha256(d, c) for d in DISEASE for c in DISEASE[d]})
    json.dump(manifest, open(f"{out}/run_manifest.json", "w"), indent=2, default=str)

    print(f"=== ACAR GO/NO-GO v2: {decision} ===  (G1={G1} via {g1_action}, G2={G2}, guard=PASS, hash={h1}, n={len(recs)})")
    for d in DISEASE:
        r = cv[d]
        print(f"  [{d}] G1 best paired-feature AUROC per action (oriented, out-of-fold):")
        for a in NON_IDENTITY:
            fa = r["g1"][a]["feature_auc"]
            best = max(PAIRED_FEATURES, key=lambda f: (fa[f] if fa[f] == fa[f] else -1))
            print(f"      {a:14s} {best:9s}={fa[best] if fa[best]==fa[best] else float('nan'):.3f}  ĝ={r['g1'][a]['regressor_auc'] if r['g1'][a]['regressor_auc']==r['g1'][a]['regressor_auc'] else float('nan'):.3f}")
        print(f"  [{d}] G2 NLLred router={r['red_router']:+.4f} bestfixed={r['red_bestfixed']:+.4f} "
              f"random={r['red_random'] if np.isfinite(r['red_random']) else float('nan'):+.4f} "
              f"retention={r['retention'] if np.isfinite(r['retention']) else float('nan'):.2f} "
              f"cohort_macro={r['cohort_macro_pass']} coverage={r['coverage'] if np.isfinite(r['coverage']) else float('nan'):.3f} "
              f"q_inf_folds={r['q_informative_folds']}/{r['n_folds']} abstain={r['abstain_rate']:.2f}")
    print(f"  -> {out}/acar_gonogo_summary.json")


if __name__ == "__main__":
    main()
