"""ECE / NLL / Brier from the per-fold probability sidecars (results/*.preds.npz) — NO GPU, no retrain.
Tests whether removing conditional leakage improves CALIBRATION (lower ECE/NLL) vs ERM — a reviewer-requested
metric where a less-shortcut-reliant representation should help. ERM vs the smallest-λ lpc_prior (no-cost point)."""
import numpy as np, glob, os


def ece(prob, y, nbins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, nbins + 1); e = 0.0; N = len(y)
    for i in range(nbins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += m.sum() / N * abs(correct[m].mean() - conf[m].mean())
    return e * 100


def nll(prob, y):
    return -np.log(np.clip(prob[np.arange(len(y)), y], 1e-9, 1)).mean()


def lam(cfg):
    try: return float(cfg.split(":")[1])
    except Exception: return 1e9


def main():
    out = ["# Calibration (ECE% / NLL) — ERM vs lpc_prior, from saved `*.preds.npz` (no GPU/retrain)\n",
           "_Lower is better. lpc_prior = smallest-λ (no-cost) config. Hypothesis: less subject-shortcut → better calibration._\n",
           "| dataset | ERM ECE | lpc ECE | ΔECE | ERM NLL | lpc NLL | ΔNLL |",
           "|---|---|---|---|---|---|---|"]
    nwin = ntot = 0
    for f in sorted(glob.glob("results/*.preds.npz")):
        name = os.path.basename(f)[:-10]
        try: d = np.load(f)
        except Exception: continue
        cfgs = set(k.split("::")[0] for k in d.files)
        lps = [c for c in cfgs if c.startswith("lpc_prior")]
        if "erm:0" not in cfgs or not lps:
            continue
        def m(c):
            p = d[f"{c}::prob"]; y = d[f"{c}::y"].astype(int)
            return ece(p, y), nll(p, y)
        ee, en = m("erm:0")
        lp = min(lps, key=lam); le, ln = m(lp)
        ntot += 1; nwin += le <= ee + 0.2
        out.append(f"| {name} | {ee:.1f} | {le:.1f} | {le-ee:+.1f} | {en:.3f} | {ln:.3f} | {ln-en:+.3f} |")
    out.append(f"\n*lpc_prior calibration ≤ ERM (ΔECE≤0.2) on {nwin}/{ntot} datasets.*")
    open("notes/calibration.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
