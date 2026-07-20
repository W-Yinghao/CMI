"""A0-PILOT — closed-loop sample-abstention pilot. Implements notes/A0_SAMPLE_ABSTENTION_PILOT_FROZEN.md.
4 fixed SHA seed blocks x 5 realizations; action arms; matched within-batch budget k_b=floor(0.2*n); endpoints
ADAPT_SSEP vs ADAPT_RANDOM; Freeze-B gate. No new gate/score/generator/severity/coverage freedom. GPU-free.
"""
import os, json, hashlib
from collections import defaultdict
import numpy as np
from cmi.eval.source_state import fit_source_state, pmct_predict_serialized
from a0_prime_r import (gen, state_score_metric, sample_scores_sf, base_adapt_ztilde, _load,
                        DISEASE, V4, B, CONTEST)

SEV = {"lowmargin_rot": [10, 20, 30, 45], "highmargin_cbw": [0.05, 0.10, 0.15], "covariate_shift_beneficial": [0.5, 1.0, 2.0]}
CONTROLS = ["clean", "covariate_shift_beneficial"]; BLOCKS = 4; R = 5; KFRAC = 0.20
ARMS = ["BASE", "ALWAYS_ADAPT", "ADAPT_RANDOM_ABSTAIN", "ADAPT_GUNC_ABSTAIN", "ADAPT_SSEP_ABSTAIN",
        "ORACLE_HARM_ABSTAIN", "BASE_WITH_SSEP_MASK"]

def pseed(block, *parts):
    return int(hashlib.sha256(("A0PILOT_V1|block_%d|" % block + "|".join(map(str, parts))).encode()).hexdigest()[:8], 16)

def gen_any(name, sev, z, y, st, rng):
    return (z.copy(), y.copy()) if name == "clean" else gen(name, sev, z, y, st, rng)

def batch_eval(state, bz, by, Winv, randseed):
    """All arms for one batch. Returns per-arm endpoint contributions + (for guard) mask/coverage/retained-preds."""
    fb = len(bz) < 8
    base = state["clf"].predict_proba(bz)
    if fb:
        adapt = base
    else:
        _, adapt, ztil = base_adapt_ztilde(state, bz)
        adapt = np.asarray(adapt, float)
    ztil = bz if fb else ztil
    sc = sample_scores_sf(state, ztil, adapt, Winv)
    n = len(bz); k = int(np.floor(KFRAC * n))
    bc = base.argmax(1) == by; ae = adapt.argmax(1) != by; bw = base.argmax(1) != by
    hf = bc & ae; bfl = bw & (adapt.argmax(1) == by)
    la = -np.log(np.clip(adapt[np.arange(n), by], 1e-9, 1)); lb = -np.log(np.clip(base[np.arange(n), by], 1e-9, 1))
    def mask_top(score): m = np.zeros(n, bool); m[np.argsort(-score)[:k]] = True; return m
    masks = dict(BASE=np.zeros(n, bool), ALWAYS_ADAPT=np.zeros(n, bool),
                 ADAPT_RANDOM_ABSTAIN=(lambda m: (m.__setitem__(np.random.default_rng(randseed).permutation(n)[:k], True) or m))(np.zeros(n, bool)),
                 ADAPT_GUNC_ABSTAIN=mask_top(sc["g_unc"]), ADAPT_SSEP_ABSTAIN=mask_top(sc["s_sep"]),
                 ORACLE_HARM_ABSTAIN=hf.copy(), BASE_WITH_SSEP_MASK=mask_top(sc["s_sep"]))
    out = {}
    for arm in ARMS:
        m = masks[arm]; keep = ~m
        pred_nll = lb if arm in ("BASE", "BASE_WITH_SSEP_MASK") else la
        pred_err = bw if arm in ("BASE", "BASE_WITH_SSEP_MASK") else ae
        out[arm] = dict(retained_nll=float(pred_nll[keep].mean()) if keep.any() else np.nan,
                        retained_01=float(pred_err[keep].mean()) if keep.any() else np.nan,
                        retained_harmflip=float(hf[bc & keep].mean()) if (bc & keep).any() else np.nan,
                        prevented=float(hf[m].sum()), lost=float(bfl[m].sum()),
                        coverage=float(keep.mean()), mask=m)
    return out

def run():
    store = defaultdict(lambda: defaultdict(list)); guard_fail = []; guard_n = 0
    for block in range(BLOCKS):
        for cond, cohs in DISEASE.items():
            for coh in cohs:
                zev, yev, zte, yte = _load(cond, coh); state = fit_source_state(zev, yev, 2, rho=0.1); Winv = state_score_metric(state)
                for fam in CONTEST + CONTROLS:
                    for sev in (SEV[fam] if fam != "clean" else [0]):
                        for real in range(R):
                            sd = pseed(block, coh, fam, sev, real)
                            zp, yp = gen_any(fam, sev, zte, yte, state, np.random.default_rng(sd))
                            for s in range(0, len(zp), B):
                                bz, by = zp[s:s+B], yp[s:s+B]
                                ev = batch_eval(state, bz, by, Winv, pseed(block, coh, fam, sev, real, "rand", s))
                                store[(cond, fam)][block].append((coh, ev))
                            # METAMORPHIC GUARD (1 realization per cell, block 0): permute y -> masks/coverage identical
                            if block == 0 and real == 0:
                                yperm = np.random.default_rng(sd + 7).permutation(yte)
                                zp2, _ = gen_any(fam, sev, zte, yperm, state, np.random.default_rng(sd))
                                for s in range(0, len(zp2), B):
                                    bz = zp2[s:s+B]; by1 = yp[s:s+B]; by2 = yperm[s:s+B]
                                    e1 = batch_eval(state, bz, by1, Winv, pseed(block, coh, fam, sev, real, "rand", s))
                                    e2 = batch_eval(state, bz, by2, Winv, pseed(block, coh, fam, sev, real, "rand", s))
                                    guard_n += 1
                                    for arm in ["ADAPT_SSEP_ABSTAIN", "ADAPT_GUNC_ABSTAIN", "ADAPT_RANDOM_ABSTAIN"]:
                                        if not np.array_equal(e1[arm]["mask"], e2[arm]["mask"]):
                                            guard_fail.append(f"{cond}/{coh}/{fam}: {arm} mask depends on y")
    return store, guard_fail, guard_n

def agg(store, fam_set, arm, field, disease, excl_coh=None, block=None):
    """macro batch->family->cohort for a disease (optionally leave-one-cohort-out / single block)."""
    pc = []
    for coh in DISEASE[disease]:
        if coh == excl_coh: continue
        fm = []
        for fam in fam_set:
            blocks = [block] if block is not None else range(BLOCKS)
            vals = [ev[arm][field] for b in blocks for (c, ev) in store[(disease, fam)][b] if c == coh and not np.isnan(ev[arm][field])]
            if vals: fm.append(np.mean(vals))
        if fm: pc.append(np.mean(fm))
    return float(np.mean(pc)) if pc else np.nan

def main():
    fz = json.load(open("results/freeze_a1/manifest.json"))
    store, guard_fail, guard_n = run()
    if guard_fail:
        print("METAMORPHIC GUARD FAILED:"); [print("  ", g) for g in guard_fail[:10]]; raise SystemExit(3)
    res = {}
    for d in DISEASE:
        res[d] = {arm: {f: agg(store, CONTEST, arm, f, d) for f in ("retained_nll", "retained_01", "retained_harmflip", "coverage")} for arm in ARMS}
        for arm in ARMS:
            res[d][arm]["net_protection"] = agg(store, CONTEST, arm, "prevented", d) - agg(store, CONTEST, arm, "lost", d)
    # Freeze-B gate
    def improves(d, metric):  # SSEP better (lower) than RANDOM on metric, both diseases
        return res[d]["ADAPT_SSEP_ABSTAIN"][metric] < res[d]["ADAPT_RANDOM_ABSTAIN"][metric]
    c1 = all(improves(d, "retained_nll") and improves(d, "retained_01") for d in DISEASE)
    c2 = all(agg(store, [fam], "ADAPT_SSEP_ABSTAIN", "retained_nll", d) <
             agg(store, [fam], "ADAPT_RANDOM_ABSTAIN", "retained_nll", d) for d in DISEASE for fam in CONTEST)  # no harm family reverses (NLL)
    c3 = all(all((agg(store, CONTEST, "ADAPT_SSEP_ABSTAIN", "retained_nll", d, excl_coh=coh) or 9) <
                 (agg(store, CONTEST, "ADAPT_RANDOM_ABSTAIN", "retained_nll", d, excl_coh=coh) or 0) for coh in DISEASE[d]) for d in DISEASE)  # LOCO
    c4 = all(all(agg(store, CONTEST, "ADAPT_SSEP_ABSTAIN", "retained_nll", d, block=b) <
                 agg(store, CONTEST, "ADAPT_RANDOM_ABSTAIN", "retained_nll", d, block=b) for b in range(BLOCKS)) for d in DISEASE)  # seed blocks
    ctrl = {f"{fam}/{d}": (agg(store, [fam], "ADAPT_SSEP_ABSTAIN", "prevented", d) - agg(store, [fam], "ADAPT_SSEP_ABSTAIN", "lost", d))
            for fam in CONTROLS for d in DISEASE}
    c5 = all((-(agg(store, [fam], "ADAPT_SSEP_ABSTAIN", "lost", d) - agg(store, [fam], "ADAPT_SSEP_ABSTAIN", "prevented", d))) >= -0.02 for fam in CONTROLS for d in DISEASE)
    c6 = all(res[d]["ADAPT_SSEP_ABSTAIN"]["net_protection"] >= res[d]["ADAPT_GUNC_ABSTAIN"]["net_protection"] - 0.03 for d in DISEASE)
    gate = dict(c1_nll_and_01_improve=bool(c1), c2_no_family_reversal=bool(c2), c3_loco_robust=bool(c3),
                c4_seedblocks_consistent=bool(c4), c5_controls_no_harm=bool(c5), c6_ssep_not_worse_than_gunc=bool(c6))
    freeze_b = all(gate.values())
    decision = "FREEZE_B: CITA-no-LPC + post-alignment s_sep selective abstention" if freeze_b else "DIAGNOSTIC_ONLY"
    out = f"results/a0_pilot/{fz['hash'][:16]}"; os.makedirs(out, exist_ok=True)
    summary = dict(decision=decision, freeze_b=freeze_b, gate=gate, metamorphic_guard="PASS", guard_batches_checked=guard_n,
                   per_disease=res, controls_net=ctrl, blocks=BLOCKS, R=R, kfrac=KFRAC, freeze_a1_hash=fz["hash"],
                   note="ADAPT_SSEP vs ADAPT_RANDOM at matched coverage; oracle = ceiling; achieved coverage reported.")
    json.dump(summary, open(f"{out}/a0pilot_summary.json", "w"), indent=2, default=str)
    json.dump(dict(pre_registration="notes/A0_SAMPLE_ABSTENTION_PILOT_FROZEN.md", blocks=BLOCKS, dumps=V4), open(f"{out}/run_manifest.json", "w"), indent=2)
    print(f"=== A0-PILOT DECISION: {decision} ===  (guard=PASS over {guard_n} batches)")
    print(f"  achieved coverage: {res['PD']['ADAPT_SSEP_ABSTAIN']['coverage']:.3f}")
    print("  arm endpoints (macro, contest) — retained NLL | 0-1 | harm-flip | net-protection:")
    for arm in ARMS:
        print(f"    {arm:22s} PD: {res['PD'][arm]['retained_nll']:.3f} | {res['PD'][arm]['retained_01']:.3f} | {res['PD'][arm]['retained_harmflip']:.3f} | {res['PD'][arm]['net_protection']:+.2f}"
              f"   SCZ: {res['SCZ'][arm]['retained_nll']:.3f} | {res['SCZ'][arm]['retained_01']:.3f} | {res['SCZ'][arm]['retained_harmflip']:.3f} | {res['SCZ'][arm]['net_protection']:+.2f}")
    print("  Freeze-B gate:", gate)
    print(f"  -> {out}/a0pilot_summary.json")

if __name__ == "__main__":
    main()
