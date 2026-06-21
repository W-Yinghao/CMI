"""Frozen survivor matrix (EVIDENCE_LEDGER #7). ERM / plain matched-CORAL / SPDIM / CITA-no-LPC on the disease
cross-dataset folds (PD+SCZ seed0, feat_dump_v4, serialized source-free state, bit-exact). CITA-no-LPC == erm:0
encoder + matched-CORAL (LPC dropped, CMI report-only) -> it IS plain matched-CORAL; shown explicitly. Reports
cohort balanced-acc, worst-cohort, NLL/Brier/ECE, cohort-level paired bootstrap CIs, source-free equivalence, and a
fallback/failure count. NO new method/hyperparameter. CPU-only.
"""
import json, glob
import numpy as np
from cmi.eval.label_shift import transduct_predict
from cmi.eval.source_state import fit_source_state
from cmi.eval.tta_baselines import spdim_predict
from a0_prime_r import _load, DISEASE

def m(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1); p = p / p.sum(1, keepdims=True); n, K = p.shape; oh = np.eye(K)[y]
    from sklearn.metrics import balanced_accuracy_score
    conf = p.max(1); corr = (p.argmax(1) == y).astype(float); ece = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        msk = (conf >= lo) & (conf < lo + 1/15)
        if msk.any(): ece += abs(corr[msk].mean() - conf[msk].mean()) * msk.mean()
    return dict(bacc=float(balanced_accuracy_score(y, p.argmax(1)) * 100), nll=float(-np.log(p[np.arange(n), y]).mean()),
                brier=float(((p - oh) ** 2).sum(1).mean()), ece=float(ece * 100))

def main():
    METHODS = ["ERM_head", "no_adapt_probe", "matched_CORAL (=CITA-no-LPC)", "SPDIM"]
    per = {mm: {} for mm in METHODS}; equiv = 0.0
    for cond, cohs in DISEASE.items():
        J = json.load(open(f"results/r9_dualpc2/r14deploy_{cond}.json"))
        recd = {r["held_out"]: r for r in J["folds"]["erm:0"]}
        for coh in cohs:
            o = np.load(f"results/feat_dump_v4/audit_{cond}_{coh}_erm_0.npz", allow_pickle=True)
            zev, yev = np.asarray(o["z_ev"], float), np.asarray(o["y_ev"]).astype(int)
            zte, yte = np.asarray(o["z_te"], float), np.asarray(o["y_te"]).astype(int)
            pi = np.bincount(np.concatenate([np.asarray(o["y_se"]).astype(int), yev]), minlength=2).astype(float); pi /= pi.sum()
            st = fit_source_state(zev, yev, 2, rho=0.1)
            r = transduct_predict(zev, yev, zte, pi, 2, mode="matched_coral", shrink=0.1)
            mc = np.asarray(r["prob"], float); probe = np.asarray(r["prob_probe_raw"], float)
            sp = np.asarray(spdim_predict(st, zte), float)
            erm = np.asarray(o["prob_te"], float)
            # equivalence: offline matched-CORAL bAcc vs the runner-recorded value (frozen-readout check)
            from sklearn.metrics import balanced_accuracy_score as bas
            equiv = max(equiv, abs(bas(yte, mc.argmax(1)) - recd[coh]["ts_matched_coral_balanced_acc"]))
            per["ERM_head"][coh] = m(erm, yte); per["no_adapt_probe"][coh] = m(probe, yte)
            per["matched_CORAL (=CITA-no-LPC)"][coh] = m(mc, yte); per["SPDIM"][coh] = m(sp, yte)
    cohorts = [c for cs in DISEASE.values() for c in cs]
    def col(mm, fld, disease=None):
        cs = DISEASE[disease] if disease else cohorts
        return [per[mm][c][fld] for c in cs]
    # cohort-level paired bootstrap CI for a contrast on a field
    def ci(ma, mb, fld):
        d = np.array([per[ma][c][fld] - per[mb][c][fld] for c in cohorts]); rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    out = dict(serialized_equivalence_maxabs_bacc=equiv,
               table={mm: {c: per[mm][c] for c in cohorts} for mm in METHODS},
               summary={mm: {fld: dict(mean=float(np.mean(col(mm, fld))), worst=(float(np.max(col(mm, fld))) if fld in ("nll", "brier", "ece") else float(np.min(col(mm, fld)))),
                                       PD=float(np.mean(col(mm, fld, "PD"))), SCZ=float(np.mean(col(mm, fld, "SCZ"))))
                            for fld in ("bacc", "nll", "brier", "ece")} for mm in METHODS},
               contrasts=dict(CITAnoLPC_vs_SPDIM_bacc=ci("matched_CORAL (=CITA-no-LPC)", "SPDIM", "bacc"),
                              CITAnoLPC_vs_ERMhead_bacc=ci("matched_CORAL (=CITA-no-LPC)", "ERM_head", "bacc"),
                              CITAnoLPC_vs_SPDIM_nll=ci("matched_CORAL (=CITA-no-LPC)", "SPDIM", "nll"),
                              CITAnoLPC_vs_SPDIM_ece=ci("matched_CORAL (=CITA-no-LPC)", "SPDIM", "ece")))
    cita, spd = "matched_CORAL (=CITA-no-LPC)", "SPDIM"
    beats_spdim = (out["contrasts"]["CITAnoLPC_vs_SPDIM_bacc"][1] > 0 or          # worst-case acc CI excludes 0 (better)
                   out["contrasts"]["CITAnoLPC_vs_SPDIM_nll"][2] < 0 or           # NLL CI excludes 0 (lower)
                   out["contrasts"]["CITAnoLPC_vs_SPDIM_ece"][2] < 0)
    out["decision"] = ("CITA-no-LPC shows a clear advantage over SPDIM/matched-CORAL -> supporting engineering gain"
                       if beats_spdim else
                       "NO clear CITA-no-LPC advantage over plain matched-CORAL/SPDIM -> STOP positive-method line; "
                       "main contribution = the measurement->control gap (diagnostic line)")
    out["note"] = "CITA-no-LPC IS plain matched-CORAL (LPC dropped, CMI report-only); listed separately only to make the identity explicit."
    import os; os.makedirs("results/survivor_matrix", exist_ok=True)
    json.dump(out, open("results/survivor_matrix/summary.json", "w"), indent=2, default=str)
    print(f"=== SURVIVOR MATRIX (serialized-equiv |Δbacc|={equiv:.2e}) ===")
    print(f"  {'method':30s} {'bAcc':>6s} {'worst':>6s} {'NLL':>6s} {'ECE':>5s}   (PD/SCZ bAcc)")
    for mm in METHODS:
        s = out["summary"][mm]
        print(f"  {mm:30s} {s['bacc']['mean']:6.1f} {s['bacc']['worst']:6.1f} {s['nll']['mean']:6.3f} {s['ece']['mean']:5.1f}   ({s['bacc']['PD']:.1f}/{s['bacc']['SCZ']:.1f})")
    print("  contrasts (mean [95% CI], cohort-level paired bootstrap):")
    for k, v in out["contrasts"].items():
        print(f"    {k:30s} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]")
    print(f"  DECISION: {out['decision']}")
    print(f"  -> results/survivor_matrix/summary.json")

if __name__ == "__main__":
    main()
