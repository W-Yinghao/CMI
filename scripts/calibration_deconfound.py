"""Calibration deconfound (EVIDENCE_LEDGER #9): is LPC's ECE/NLL win more than a global temperature / logit
shrinkage on ERM? Uses saved target probs only (no source-val in the preds), so the temperature is the ORACLE
single-T fit ON TARGET (minimizes NLL) — the temperature UPPER BOUND, strictly MORE favorable to ERM than any
deployable source-only T. Therefore: if LPC does NOT beat ERM+oracleT, the 'principled calibration' claim is
demoted to a temperature/compression side-effect. TUAB excluded (exposed). CPU-only, from *.preds.npz.
"""
import glob, json
import numpy as np
from scipy.optimize import minimize_scalar

def metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1); p = p / p.sum(1, keepdims=True)
    n, K = p.shape; oh = np.eye(K)[y]
    nll = float(-np.log(p[np.arange(n), y]).mean())
    brier = float(((p - oh) ** 2).sum(1).mean())
    acc = float((p.argmax(1) == y).mean())
    conf = p.max(1); pred = p.argmax(1); correct = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / 15)
        if m.any(): ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return dict(nll=nll, brier=brier, ece=float(ece * 100), acc=acc)

def temp_scale(p, y):
    """oracle single temperature on recovered logits (log p): softmax(log p / T), T minimizing target NLL."""
    logp = np.log(np.clip(np.asarray(p, float), 1e-12, 1))
    def nll(logT):
        z = logp / np.exp(logT); z = z - z.max(1, keepdims=True); q = np.exp(z); q = q / q.sum(1, keepdims=True)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()
    T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    z = logp / T; z = z - z.max(1, keepdims=True); q = np.exp(z); q = q / q.sum(1, keepdims=True)
    return q, T

def main():
    rows = []
    for f in sorted(glob.glob("results/**/*.preds.npz", recursive=True)):
        if "tuab" in f.lower():
            continue                                                     # EXPOSED dataset — excluded
        o = np.load(f, allow_pickle=True)
        cfgs = sorted(set(k.split("::")[0] for k in o.files))
        if "erm:0" not in cfgs:
            continue
        lpc = next((c for c in cfgs if c.startswith("lpc_prior")), None) or next((c for c in cfgs if c.startswith("lpc")), None)
        if lpc is None:
            continue
        try:
            ep, ey = o["erm:0::prob"], o["erm:0::y"].astype(int); lp, ly = o[f"{lpc}::prob"], o[f"{lpc}::y"].astype(int)
        except Exception:
            continue
        if len(ep) != len(lp) or ep.shape[1] < 2:
            continue
        erm = metrics(ep, ey); et, T = temp_scale(ep, ey); erm_t = metrics(et, ey); lpcm = metrics(lp, ly)
        rows.append(dict(dataset=f.split("/")[-1].replace(".preds.npz", ""), lpc_cfg=lpc, n=len(ey), T=T,
                         erm=erm, erm_oracleT=erm_t, lpc=lpcm,
                         lpc_beats_ermT_nll=lpcm["nll"] < erm_t["nll"] - 1e-3,
                         nll_gap_lpc_minus_ermT=lpcm["nll"] - erm_t["nll"],
                         acc_gap_lpc_minus_erm=lpcm["acc"] - erm["acc"]))
    # verdict: across datasets, does LPC beat ERM+oracleT on NLL? and is the raw LPC win explained by temperature?
    n = len(rows); beats = sum(r["lpc_beats_ermT_nll"] for r in rows)
    med_gap = float(np.median([r["nll_gap_lpc_minus_ermT"] for r in rows])) if rows else float("nan")
    closed = sum(1 for r in rows if (r["erm"]["nll"] - r["erm_oracleT"]["nll"]) >= 0.5 * (r["erm"]["nll"] - r["lpc"]["nll"]) and (r["erm"]["nll"] - r["lpc"]["nll"]) > 0)
    n_lpc_better_raw = sum(1 for r in rows if r["lpc"]["nll"] < r["erm"]["nll"] - 1e-3)
    verdict = ("PRINCIPLED" if beats > n * 0.6 else
               "TEMPERATURE/COMPRESSION SIDE-EFFECT" if beats < max(1, n * 0.34) else "MIXED")
    out = dict(verdict=verdict, n_datasets=n, lpc_beats_ERM_oracleT_nll=beats,
               median_nll_gap_lpc_minus_ermOracleT=med_gap, n_lpc_better_than_raw_erm=n_lpc_better_raw,
               n_oracleT_closes_half_the_gap=closed, note=("oracle-T is the temperature UPPER BOUND (target-fit) -> "
               "strictly more favorable to ERM than any deployable source-only T; a non-win for LPC is conclusive."),
               rows=rows)
    import os; os.makedirs("results/calibration_deconfound", exist_ok=True)
    json.dump(out, open("results/calibration_deconfound/summary.json", "w"), indent=2, default=str)
    print(f"=== CALIBRATION DECONFOUND: {verdict} ===")
    print(f"  datasets={n} | LPC beats ERM+oracleT on NLL: {beats}/{n} | median NLL gap (LPC-ERMoracleT)={med_gap:+.3f}")
    print(f"  (LPC better than RAW ERM on NLL: {n_lpc_better_raw}/{n}; oracle-T alone closes >=half the LPC gap: {closed}/{n})")
    print(f"  {'dataset':32s} {'acc Δ':>7s} {'ERM nll':>8s} {'ERM+T nll':>10s} {'LPC nll':>8s} {'LPC>ERM+T?':>10s}")
    for r in sorted(rows, key=lambda x: x["nll_gap_lpc_minus_ermT"]):
        print(f"  {r['dataset'][:32]:32s} {r['acc_gap_lpc_minus_erm']:+7.3f} {r['erm']['nll']:8.3f} {r['erm_oracleT']['nll']:10.3f} {r['lpc']['nll']:8.3f} {str(r['lpc_beats_ermT_nll']):>10s}")
    print(f"  -> results/calibration_deconfound/summary.json")

if __name__ == "__main__":
    main()
