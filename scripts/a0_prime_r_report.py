"""A0'-R reporting-completeness pass. Deterministic (SAME SHA seed map as a0_prime_r), NO new seed/score/generator/
decision freedom. Fills the originally-pre-registered metrics a0_prime_r.py did not emit, expands the
serialized-equivalence hard gate to ALL 7 cohorts + every adapting batch, and writes a verifiable artifact.
"""
import os, json, hashlib
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from cmi.eval.label_shift import transduct_predict
from cmi.eval.source_state import fit_source_state, pmct_predict_serialized
from a0_prime_r import (sha_seed, gen, state_score_metric, sample_scores_sf, base_adapt_ztilde, _load,
                        DISEASE, V4, B, SCORES, SEV, CONTEST, R)

CONTROLS = ["clean", "covariate_shift_beneficial"]; KFRAC = 0.20

def gen_any(name, sev, z, y, state, rng):
    if name == "clean":
        return z.copy(), y.copy()
    return gen(name, sev, z, y, state, rng)

def _auc(s, lab):
    lab = np.asarray(lab); return float(roc_auc_score(lab, s)) if len(set(lab)) == 2 and np.std(s) > 1e-12 else np.nan

def selective(batches, score, by_random=False, oracle=False):
    """Within-batch abstain top floor(0.2*n) by score (or random/oracle); retained NLL, 0-1 risk, harm-flip rate."""
    nll, zo, hf, cov = [], [], [], []
    for b in batches:
        n = b["n"]; k = int(np.floor(KFRAC * n))
        if oracle:
            order = np.argsort(-(b["harm_flip"].astype(float) + 1e-6 * b["la"]))     # abstain the actual harm flips first
        elif by_random:
            order = np.random.default_rng(sha_seed("rand", b["batch"], n, 0)).permutation(n)
        else:
            order = np.argsort(-b["scores"][score])
        keep = np.ones(n, bool); keep[order[:k]] = False
        nll.append(b["la"][keep].mean()); zo.append(b["adapt_err"][keep].mean()); cov.append(keep.mean())
        bc = b["base_correct"] & keep
        hf.append(b["harm_flip"][bc].mean() if bc.any() else np.nan)
    return dict(retained_nll=float(np.nanmean(nll)), retained_01=float(np.nanmean(zo)),
                retained_harmflip=float(np.nanmean(hf)), achieved_coverage=float(np.nanmean(cov)))

def main():
    fz = json.load(open("results/freeze_a1/manifest.json"))
    # ---- EXPANDED serialized-equivalence hard gate: ALL 7 cohorts, EVERY adapting batch (incl tail >= 8) ----
    maxd = 0.0; n_checked = 0
    for cond, cohs in DISEASE.items():
        for coh in cohs:
            zev, yev, zte, _ = _load(cond, coh); st = fit_source_state(zev, yev, 2, rho=0.1)
            pi = np.bincount(yev, minlength=2).astype(float) / len(yev)
            for s in range(0, len(zte), B):
                bz = zte[s:s+B]
                if len(bz) < 8:
                    continue                                                  # fallback batch -> identity (no serialized transport)
                on = transduct_predict(zev, yev, bz, pi, 2, mode="matched_coral", shrink=0.1)["prob"]
                ad = pmct_predict_serialized(st, bz, ref="pooled", tmap="wc", em_iters=3)[0]
                maxd = max(maxd, float(np.abs(np.asarray(ad, float) - on).max())); n_checked += 1
    if maxd > 1e-9:
        print(f"EXPANDED SERIALIZED-EQUIVALENCE FAILED ({maxd:.2e} over {n_checked} batches) — abort"); raise SystemExit(2)

    # ---- re-run all cells (contest + controls), deterministic ----
    store = defaultdict(list)
    for cond, cohs in DISEASE.items():
        for coh in cohs:
            zev, yev, zte, yte = _load(cond, coh); state = fit_source_state(zev, yev, 2, rho=0.1); Winv = state_score_metric(state)
            for fam in CONTEST + CONTROLS:
                for sev in (SEV[fam] if fam != "clean" else [0]):
                    for real in range(R):
                        zp, yp = gen_any(fam, sev, zte, yte, state, np.random.default_rng(sha_seed(coh, fam, sev, real)))
                        # reuse cell logic but with this realization's seed (re-derive deterministically)
                        rows = []
                        for bi, s in enumerate(range(0, len(zp), B)):
                            bz, by = zp[s:s+B], yp[s:s+B]; fb = len(bz) < 8
                            base = state["clf"].predict_proba(bz) if fb else None
                            if fb:
                                adapt = base; ztil = bz
                            else:
                                base, adapt, ztil = base_adapt_ztilde(state, bz)
                            sc = sample_scores_sf(state, ztil, adapt, Winv)
                            bc = base.argmax(1) == by; ae = adapt.argmax(1) != by; bw = base.argmax(1) != by
                            la = -np.log(np.clip(adapt[np.arange(len(by)), by], 1e-9, 1)); lb = -np.log(np.clip(base[np.arange(len(by)), by], 1e-9, 1))
                            rows.append(dict(batch=bi, fallback=fb, n=len(bz), scores={k: sc[k] for k in SCORES},
                                             base_correct=bc, base_err=bw, adapt_err=ae, harm_flip=bc & ae,
                                             ben_flip=bw & (adapt.argmax(1) == by), la=la, lb=lb))
                        store[(cond, coh, fam)] += rows

    rep = {}
    # 1) flips (per disease, contest) + 2) adapted-error AUROC + cohort-equal family macro
    def disease_rows(d, fams): return [r for coh in DISEASE[d] for fam in fams for r in store[(d, coh, fam)]]
    rep["flips"] = {}
    for d in DISEASE:
        rows = disease_rows(d, CONTEST)
        h = float(np.mean([x for r in rows for x in r["harm_flip"]])); b = float(np.mean([x for r in rows for x in r["ben_flip"]]))
        rep["flips"][d] = dict(harmful=h, beneficial=b, net=h - b)
    def wb_auc(rows, score, target, subset):
        out = []
        for r in rows:
            m = r["base_correct"] if subset == "bc" else np.ones(r["n"], bool)
            if m.sum() < 8: continue
            a = _auc(r["scores"][score][m], r[target][m])
            if not np.isnan(a): out.append(a)
        return out
    rep["within_batch_auc"] = {s: {tgt: {d: float(np.mean([v for coh in DISEASE[d] for fam in CONTEST for v in wb_auc(store[(d, coh, fam)], s, tgt, sub)] or [np.nan]))
                                          for d in DISEASE} for tgt, sub in (("harm_flip", "bc"), ("base_err", "all"), ("adapt_err", "all"))} for s in SCORES}
    rep["family_macro_cohort_equal"] = {s: {f"{d}/{fam}": float(np.nanmean([np.mean(wb_auc(store[(d, coh, fam)], s, "harm_flip", "bc") or [np.nan]) for coh in DISEASE[d]]))
                                            for d in DISEASE for fam in CONTEST} for s in SCORES}
    rep["family_pooled"] = {s: {f"{d}/{fam}": float(np.mean([v for coh in DISEASE[d] for v in wb_auc(store[(d, coh, fam)], s, "harm_flip", "bc")] or [np.nan]))
                                for d in DISEASE for fam in CONTEST} for s in SCORES}
    # 3) selective NLL / 0-1 / harm-flip at fixed within-batch coverage (s_sep, g_unc, random, oracle), per disease (contest)
    rep["selective"] = {}
    for d in DISEASE:
        rows = disease_rows(d, CONTEST)
        rep["selective"][d] = dict(s_sep=selective(rows, "s_sep"), g_unc=selective(rows, "g_unc"),
                                   random=selective(rows, None, by_random=True), oracle=selective(rows, None, oracle=True),
                                   no_abstain=dict(retained_nll=float(np.mean([r["la"].mean() for r in rows])),
                                                   retained_01=float(np.mean([r["adapt_err"].mean() for r in rows]))))
    # 4) global vs within-batch top-20% harm enrichment (s_sep)
    rep["top20_enrichment"] = {}
    for d in DISEASE:
        rows = disease_rows(d, CONTEST)
        alls = np.concatenate([r["scores"]["s_sep"][r["base_correct"]] for r in rows if r["base_correct"].any()])
        allh = np.concatenate([r["harm_flip"][r["base_correct"]] for r in rows if r["base_correct"].any()])
        gt = np.argsort(-alls)[:int(0.2 * len(alls))]
        wb = [r["harm_flip"][r["base_correct"]][np.argsort(-r["scores"]["s_sep"][r["base_correct"]])[:max(1, int(0.2 * r["base_correct"].sum()))]].mean()
              for r in rows if r["base_correct"].sum() >= 5]
        rep["top20_enrichment"][d] = dict(base_rate=float(allh.mean()), global_top20=float(allh[gt].mean()), withinbatch_top20=float(np.nanmean(wb)))
    # 5) per cohort/family coverage accounting
    rep["coverage_accounting"] = {}
    for d in DISEASE:
        for coh in DISEASE[d]:
            for fam in CONTEST:
                rows = store[(d, coh, fam)]
                elig = sum(1 for r in rows if r["base_correct"].sum() >= 8 and len(set(r["harm_flip"][r["base_correct"]])) == 2)
                rep["coverage_accounting"][f"{d}/{coh}/{fam}"] = dict(
                    n_batches=len(rows), eligible_batches=elig, fallback_frac=float(np.mean([r["fallback"] for r in rows])),
                    harm_events=int(sum(int(r["harm_flip"].sum()) for r in rows)))
    # 6) matched-coverage no-harm guard on controls (clean, covariate_beneficial): net protection ~ 0, not strongly negative
    rep["no_harm_guard"] = {}
    for fam in CONTROLS:
        for d in DISEASE:
            rows = [r for coh in DISEASE[d] for r in store[(d, coh, fam)]]
            prevented = lost = 0.0; nb = 0
            for r in rows:
                n = r["n"]; k = int(np.floor(KFRAC * n)); ab = np.argsort(-r["scores"]["s_sep"])[:k]
                m = np.zeros(n, bool); m[ab] = True
                prevented += r["harm_flip"][m].sum(); lost += r["ben_flip"][m].sum(); nb += 1
            rep["no_harm_guard"][f"{fam}/{d}"] = dict(prevented_per_batch=float(prevented / max(nb, 1)),
                                                      lost_per_batch=float(lost / max(nb, 1)), net=float((prevented - lost) / max(nb, 1)))

    rep["serialized_equivalence"] = dict(max_abs=maxd, batches_checked=n_checked, all_7_cohorts=True)
    rep["note"] = ("reporting-completeness only; decision (POST_ALIGN_ABSTENTION_ONLY, hash 82719b20f12a2bbc) is "
                   "from a0_prime_r.py and unchanged. base-error AUROC ~0.52-0.53 = WEAKLY aligned w/ base difficulty.")
    out = f"results/a0_prime_r/{fz['hash'][:16]}"; os.makedirs(out, exist_ok=True)
    json.dump(rep, open(f"{out}/a0primer_report.json", "w"), indent=2, default=str)
    print(f"=== A0'-R reporting-completeness (serialized-equiv ALL 7 cohorts: max|d|={maxd:.0e} over {n_checked} batches) ===")
    print("  flips (contest) PD/SCZ:", {d: {k: round(v, 3) for k, v in rep["flips"][d].items()} for d in DISEASE})
    print("  selective retained-NLL (lower=better)  s_sep / random / oracle / no-abstain:")
    for d in DISEASE:
        sv = rep["selective"][d]; print(f"    {d}: {sv['s_sep']['retained_nll']:.3f} / {sv['random']['retained_nll']:.3f} / {sv['oracle']['retained_nll']:.3f} / {sv['no_abstain']['retained_nll']:.3f}  (cov={sv['s_sep']['achieved_coverage']:.3f})")
    print("  selective retained harm-flip rate  s_sep / random:")
    for d in DISEASE:
        sv = rep["selective"][d]; print(f"    {d}: {sv['s_sep']['retained_harmflip']:.3f} / {sv['random']['retained_harmflip']:.3f}")
    print("  cohort-equal family macro (s_sep harm AUROC):", {k: round(v, 3) for k, v in rep["family_macro_cohort_equal"]["s_sep"].items()})
    print("  no-harm guard net/batch (s_sep on controls; ~0 good):", {k: round(v["net"], 3) for k, v in rep["no_harm_guard"].items()})
    print(f"  -> {out}/a0primer_report.json")

if __name__ == "__main__":
    main()
