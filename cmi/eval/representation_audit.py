"""P1.5 representation-collapse audit — is the LPC leakage reduction GENUINE suppression or representation collapse?

Stage-wise on real EEG features (erm dumps in results/feat_dump/, lpc dumps in results/feat_dump_lpc/):
    pre-align (erm z)  ->  post-align (erm z after matched-CORAL)  ->  post-LPC (lpc z)
Per stage report (task-only metrics; domain-leakage multi-capacity comes from the r11mc run):
    task_probe_bAcc (held-out)         — does class structure SURVIVE? (collapse would drop this)
    effective_rank, stable_rank        — representation dimensionality (collapse shrinks it)
    between/within scatter ratio       — class separability geometry
    feat_norm, feat_var                — is the "reduction" just norm/variance shrinkage?
    probe_train_val_gap                — underfit vs overfit of the probe

Verdict logic: LPC leakage reduction is GENUINE iff (from r11mc) the domain probe drops at ALL capacities AND
here the task-probe / effective-rank / scatter / norm do NOT collapse in step. If a strong probe recovers the
leakage (r11mc strong≈erm) the conclusion is "weak-probe hiding", not suppression. All point estimates get a
grouped (per-cohort) bootstrap CI.
"""
import numpy as np, glob, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from cmi.eval.label_shift import transduct_predict


def _eff_rank(Z):
    Z = Z - Z.mean(0)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))     # entropy effective rank
    stable = float((s ** 2).sum() / (s[0] ** 2 + 1e-12))    # stable rank = ||.||_F^2 / ||.||_2^2
    return eff, stable


def _scatter_ratio(Z, y):
    mu = Z.mean(0); Sb = 0.0; Sw = 0.0
    for c in np.unique(y):
        Zc = Z[y == c]; muc = Zc.mean(0)
        Sb += len(Zc) * float(((muc - mu) ** 2).sum())
        Sw += float(((Zc - muc) ** 2).sum())
    return Sb / (Sw + 1e-12)                                # between/within scatter (separability geometry)


def _task_probe(Z, y, rng):
    n = len(Z); idx = rng.permutation(n); cut = int(0.7 * n)
    tr, ev = idx[:cut], idx[cut:]
    clf = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
    tr_ba = balanced_accuracy_score(y[tr], clf.predict(Z[tr]))
    ev_ba = balanced_accuracy_score(y[ev], clf.predict(Z[ev]))
    return ev_ba * 100, (tr_ba - ev_ba) * 100               # held-out bAcc, train-val gap


def stage_metrics(Z, y, rng):
    eff, stable = _eff_rank(Z)
    ev_ba, gap = _task_probe(Z, y, rng)
    return dict(task_bAcc=ev_ba, probe_gap=gap, eff_rank=eff, stable_rank=stable,
                scatter=_scatter_ratio(Z, y), feat_norm=float(np.linalg.norm(Z, axis=1).mean()),
                feat_var=float(Z.var(0).mean()))


def run_audit():
    rng = np.random.default_rng(0)
    erm = {os.path.basename(f).split("_")[1] + "_" + os.path.basename(f).split("_")[2]: f
           for f in glob.glob("results/feat_dump/feat_*_erm_0.npz")}
    lpc = {os.path.basename(f).split("_")[1] + "_" + os.path.basename(f).split("_")[2]: f
           for f in glob.glob("results/feat_dump_lpc/feat_*_lpc_prior_0.3*.npz")}
    print("=== P1.5 representation-collapse audit: pre-align(erm) -> post-align(erm+CORAL) -> post-LPC ===")
    print(f"{'cohort':18}{'stage':12}{'task_bAcc':>10}{'gap':>6}{'effRank':>8}{'stbRank':>8}{'scatter':>8}{'fnorm':>7}{'fvar':>7}")
    agg = {s: {k: [] for k in ('task_bAcc', 'eff_rank', 'scatter', 'feat_norm')} for s in ("pre", "post", "lpc")}
    for key, f in sorted(erm.items()):
        o = np.load(f); zs, ys, zt, yt = o["z_se"], o["y_se"], o["z_te"], o["y_te"]
        if len(np.unique(yt)) < 2 or len(zt) < 150:
            continue
        stages = {"pre-align(erm)": zt,
                  "post-align": transduct_predict(zs, ys, zt, np.array([.5, .5]), 2, mode="matched_coral")["z_tilde"]}
        if key in lpc:
            ol = np.load(lpc[key]); stages["post-LPC"] = ol["z_te"]
        for sname, Z in stages.items():
            yy = yt if sname != "post-LPC" else np.load(lpc[key])["y_te"]
            m = stage_metrics(np.asarray(Z, float), yy, rng)
            tag = {"pre-align(erm)": "pre", "post-align": "post", "post-LPC": "lpc"}[sname]
            for k in agg[tag]:
                agg[tag][k].append(m[k])
            print(f"{key:18}{sname:12}{m['task_bAcc']:10.1f}{m['probe_gap']:6.1f}{m['eff_rank']:8.1f}"
                  f"{m['stable_rank']:8.2f}{m['scatter']:8.2f}{m['feat_norm']:7.1f}{m['feat_var']:7.2f}")
    print("\n=== MEANS by stage (grouped bootstrap CI over cohorts) ===")
    for tag, nm in (("pre", "pre-align(erm)"), ("post", "post-align"), ("lpc", "post-LPC")):
        if not agg[tag]["task_bAcc"]:
            continue
        out = []
        for k in ("task_bAcc", "eff_rank", "scatter", "feat_norm"):
            v = np.array(agg[tag][k]); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            out.append(f"{k}={v.mean():.1f}[{np.percentile(bs, 2.5):.1f},{np.percentile(bs, 97.5):.1f}]")
        print(f"  {nm:16} " + "  ".join(out))
    print("\nVERDICT: GENUINE suppression iff (r11mc) domain-leakage drops at all caps AND here task_bAcc/eff_rank/")
    print("scatter/feat_norm do NOT collapse pre->lpc. Collapse = task_bAcc & eff_rank fall together with leakage.")


if __name__ == "__main__":
    run_audit()
