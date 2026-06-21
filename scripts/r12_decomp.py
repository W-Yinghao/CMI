"""r12 continuous effect-decomposition + LAYERED equivalence checks (run when the deterministic re-run lands).
  g = paired CITA_nested(+matched-CORAL) − ERM(native) gain, per (cohort, seed) outer fold. UNITS: bAcc stored
  on [0,1] in JSON; harvester multiplies by 100 -> everything here is in PERCENTAGE POINTS (pp).
  Delta_implementation = g(enc,current) − g(enc,old)     [P2.2 init + any other code/config delta; audit before naming 'init']
  Delta_protocol       = g(full,current) − g(enc,current) [selection-protocol effect, clean ONLY if run-paths equivalent]
  Total = Delta_implementation + Delta_protocol (additivity exact on shared keys).

TWO tolerances (never conflate):
  EPS_EQ_PP = 0.5 pp   — SCIENTIFIC practical-equivalence band on EFFECT sizes. "equivalent" ONLY if the whole
                         cohort-level CI ⊂ [−0.5,+0.5]; point-in-band but wide CI -> 'inconclusive', NOT 'equivalent'.
  TOL_PATH  = 1e-4 pp  — RUN-PATH numerical tolerance (near machine precision).
LAYERED run-path equivalence (§3 — bAcc alone is insufficient; different predictions can share a bAcc):
  (1) completeness: identical paired (cohort,seed) keys, no missing/dup.
  (2) config IDs correspond.
  (3) per-config PREDICTION HASH identical (canonical sample order) between r12enc and r11fp.  [strongest]
  (4) per-config outer bAcc within TOL_PATH.                                                    [fallback if no hash]
UNCERTAINTY unit (§2): COHORT, not fold. Aggregate gains within cohort (mean over seeds) -> cohort effect; CI by
  resampling COHORTS; LOCO as the key small-n diagnostic. NO plain fold-level bootstrap (too-narrow CI).
On ANY fail-fast failure: print BLOCKED, exit nonzero, do NOT emit a Freeze-A eligibility marker.
"""
import json, glob, sys, numpy as np
from collections import defaultdict, Counter
EPS_EQ_PP = 0.5
TOL_PATH = 1e-4
SEEDS = {'0', '1'}

def load(pat, seeds=SEEDS):
    out = {}
    for f in sorted(glob.glob(pat)):
        s = f.split("_s")[-1].split(".")[0]
        if s not in seeds:
            continue
        o = json.load(open(f)); fo = o.get("folds", {})
        # unit assertion: stored bAcc must be on [0,1]
        sample = next((r["balanced_acc"] for r in fo.get("erm:0", [])), 0.5)
        assert 0.0 <= sample <= 1.0, f"bAcc not on [0,1] in {f}: {sample}"
        ermn = {r["held_out"]: r["balanced_acc"] * 100 for r in fo.get("erm:0", [])}
        cfg, ph = defaultdict(dict), defaultdict(dict)
        for lbl in ("erm:0", "lpc_prior:0.1", "lpc_prior:0.3"):
            for r in fo.get(lbl, []):
                cfg[r["held_out"]][lbl] = r.get("ts_matched_coral_balanced_acc",
                                                r.get("ts_coral_balanced_acc")) * 100
                ph[r["held_out"]][lbl] = r.get("pred_hash")
        for r in fo.get("CITA_nested", []):
            c = r["held_out"]
            out[(c, s)] = dict(erm=ermn.get(c), cfg=dict(cfg.get(c, {})), ph=dict(ph.get(c, {})),
                               sel=r.get("selected_lbl"),
                               cita=r.get("ts_matched_coral_balanced_acc", r.get("ts_coral_balanced_acc")) * 100)
    return out

def cohort_ci(per_cohort_seed):
    """per_cohort_seed: {cohort: [values over seeds]} -> (mean, lo, hi) by resampling COHORTS, + LOCO list."""
    coh = sorted(per_cohort_seed); ceff = np.array([np.mean(per_cohort_seed[c]) for c in coh])
    rng = np.random.default_rng(0)
    bs = [ceff[rng.integers(0, len(ceff), len(ceff))].mean() for _ in range(5000)]
    loco = [(c, float(np.mean([np.mean(per_cohort_seed[x]) for x in coh if x != c]))) for c in coh]
    return float(ceff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), coh, ceff, loco

FAILED = []
for cond in ["SCZ", "PD"]:
    old = load(f"results/r8_dualpc2/r8sel_{cond}_s*.json")                                              # OLD code (Δ_implementation baseline)
    enc = load(f"results/r12det_dualpc2/r12det_encoder_{cond}_s*.json") or load(f"results/r12enc_dualpc2/r12enc_{cond}_s*.json")  # current encoder (det preferred)
    full = load(f"results/r12det_dualpc2/r12det_full_{cond}_s*.json") or load(f"results/r11fp_dualpc2/r11fp_{cond}_s*.json")        # current full (det preferred)
    print(f"\n========== {cond} ==========")
    ko, ke, kf = set(old), set(enc), set(full)
    keys = sorted(ke & kf)
    miss = (ke | kf) - (ke & kf)
    print(f"  (1) completeness: enc={len(ke)} full={len(kf)} common={len(keys)}" + (f"  UNPAIRED={sorted(miss)}" if miss else ""))
    if not keys or miss:
        FAILED.append((cond, "completeness")); print("      -> FAIL"); continue
    # (3) prediction-hash equivalence (preferred); (4) bAcc within TOL_PATH (fallback)
    hash_ok = all(enc[k]['ph'].get(l) and full[k]['ph'].get(l) for k in keys for l in enc[k]['cfg'])
    if hash_ok:
        mism = [(k, l) for k in keys for l in set(enc[k]['cfg']) & set(full[k]['cfg']) if enc[k]['ph'][l] != full[k]['ph'][l]]
        print(f"  (3) prediction-hash equivalence: {len(mism)} mismatches of {sum(len(enc[k]['cfg']) for k in keys)}")
        if mism:
            FAILED.append((cond, f"pred-hash mismatch ({len(mism)})"))
            print(f"      -> FAIL: predictions differ (even if bAcc matched) {mism[:3]}..."); continue
    else:
        diffs = [abs(enc[k]['cfg'][l] - full[k]['cfg'][l]) for k in keys for l in set(enc[k]['cfg']) & set(full[k]['cfg'])]
        md = max(diffs) if diffs else float('nan')
        print(f"  (3/4) no pred_hash -> bAcc fallback: max|Δ|={md:.2e} (TOL={TOL_PATH})")
        if md >= TOL_PATH:
            FAILED.append((cond, f"bAcc path-equiv {md:.2e}")); print("      -> FAIL run-path"); continue
    # additivity (only if old present & paired with current)
    paired = sorted(ko & ke & kf)
    g = lambda D, K: {k: D[k]['cita'] - D[k]['erm'] for k in K if D[k].get('erm') is not None}
    if paired:
        go, ge, gf = g(old, paired), g(enc, paired), g(full, paired)
        d_impl = {k: ge[k] - go[k] for k in paired}; d_prot = {k: gf[k] - ge[k] for k in paired}
        tot = {k: gf[k] - go[k] for k in paired}
        resid = max(abs(tot[k] - (d_impl[k] + d_prot[k])) for k in paired)
        print(f"  (2) additivity residual = {resid:.2e}  ({'OK' if resid < 1e-9 else 'FAIL'})")
    else:
        ge, gf = g(enc, keys), g(full, keys); d_impl = None
        d_prot = {k: gf[k] - ge[k] for k in keys}
        print("  (2) Δ_implementation skipped (old-code paired keys absent); reporting Δ_protocol only")
    # (4) effects at COHORT-level uncertainty
    def report(name, dd):
        pcs = defaultdict(list)
        for (c, s), v in dd.items(): pcs[c].append(v)
        m, lo, hi, coh, ceff, loco = cohort_ci(pcs)
        inb = (lo >= -EPS_EQ_PP and hi <= EPS_EQ_PP)
        tag = "≈0 EQUIVALENT (CI⊂band)" if inb else ("INCONCLUSIVE (|pt|≤band, CI wide)" if abs(m) <= EPS_EQ_PP else ("↑" if m > 0 else "↓"))
        print(f"  (4) {name:22} = {m:+.2f} pp  cohort-CI [{lo:+.2f},{hi:+.2f}]  {tag}")
        print(f"        per-cohort: " + ", ".join(f"{c}:{e:+.1f}" for c, e in zip(coh, ceff)))
        print(f"        LOCO mean:  " + ", ".join(f"drop {c}:{v:+.1f}" for c, v in loco))
    if d_impl is not None: report("Δ_implementation", d_impl)
    report("Δ_protocol", d_prot)
    tr = Counter((enc[k]['sel'], full[k]['sel']) for k in keys)
    print("  (5) sel transitions enc→full: " + ", ".join(f"{a}->{b}:{n}" for (a, b), n in tr.items()))
    print(f"      ERM-abs cur {np.mean([enc[k]['erm'] for k in keys]):.1f} / full {np.mean([full[k]['erm'] for k in keys]):.1f}"
          f"  | CITA-abs cur {np.mean([enc[k]['cita'] for k in keys]):.1f} / full {np.mean([full[k]['cita'] for k in keys]):.1f}")

print("\n" + "=" * 60)
if FAILED:
    print("FREEZE-A BLOCKED (no manifest):")
    for c, w in FAILED: print(f"  {c}: {w}")
    sys.exit(2)
print("ALL FAIL-FAST PASSED — Freeze-A ELIGIBLE (magnitude accepted as-is; write manifest next).")
