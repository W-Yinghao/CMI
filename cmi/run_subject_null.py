"""Validate the D=subject lead on PD medication-state: is I(Y_med;D_subject|Z)=0.040 a genuine SUBJECT-SPECIFIC
levodopa boundary, or an artifact of the 40-domain residual probe?

Extract Z once (EEGNet ERM on Y=med-state), compute the observed residual decoder CMI with the TRUE subject
labels, then compare against two permutation nulls (same Z, same probe, many refits):

  (1) FAKE-SUBJECT (capacity null): permute the subject label d -> 40 RANDOM groups of the SAME sizes, REAL Y.
      Isolates the 40-domain probe-capacity baseline. If observed ~ this null, 0.040 is just the multi-domain
      residual floor (the §3.5 race-4-way inflation), NOT subject-specificity.
  (2) WITHIN-SUBJECT Y-PERM (label null): keep real subjects, scramble ON/OFF WITHIN each subject (preserves
      per-subject counts/proportions). If observed ~ this null, 0.040 does not need the real medication labels.

Genuine subject-specific med-response  <=>  observed >> BOTH nulls.
"""
import argparse, json
import numpy as np, torch

from cmi.run_scps_crossdataset import load as load_xs
from cmi.run_glsvae import extract_features
from cmi.run_concept_control import build_domain, fit_decoders


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, subj, coh, classes = load_xs("PDMED", ["ds002778", "ds003490"])
    d_full, names = build_domain("subject", subj, coh, y, "PDMED", min_count=args.min_count)
    keep = d_full >= 0
    X, y, d = X[keep], y[keep], d_full[keep]
    n_dom = len(names)
    print(f"[PDMED D=subject] X={X.shape} n_dom={n_dom} (kept {keep.sum()}/{len(keep)})", flush=True)

    Z = extract_features(X, y, d, 2, "EEGNet", args.bb_epochs, args.bs, device, args.seed)
    obs = fit_decoders(Z, y, d, n_dom, seed=args.seed)
    rng = np.random.default_rng(1234 + args.seed)

    # (1) fake-subject: permute d (same group-size multiset, real Y)
    fake = np.array([fit_decoders(Z, y, rng.permutation(d), n_dom, seed=args.seed)["residual"]
                     for _ in range(args.n_null)], dtype=float)

    # (2) within-subject Y-permutation: scramble ON/OFF inside each subject
    def perm_within(yv, dv):
        yp = yv.copy()
        for k in range(n_dom):
            idx = np.where(dv == k)[0]
            yp[idx] = rng.permutation(yv[idx])
        return yp
    winp = np.array([fit_decoders(Z, perm_within(y, d), d, n_dom, seed=args.seed)["residual"]
                     for _ in range(args.n_null)], dtype=float)

    def summ(null, name):
        m, s = float(null.mean()), float(null.std() + 1e-9)
        z = (obs["residual"] - m) / s
        p = float((np.sum(null >= obs["residual"]) + 1) / (len(null) + 1))
        verdict = "CONFIRM" if (z > 2 and p < 0.05) else "not-sig"
        print(f"  {name:34s} null={m:.4f}±{s:.4f}  obs={obs['residual']:.4f}  z={z:+.2f}  p={p:.3f}  -> {verdict}",
              flush=True)
        return dict(null_mean=m, null_std=s, z=z, p=p, confirm=bool(z > 2 and p < 0.05))

    print(f"[observed] RC_res={obs['residual']:.4f}  Y|Z_acc={obs['acc']:.3f}  (n_dom={n_dom})", flush=True)
    r1 = summ(fake, "fake-subject (capacity null)")
    r2 = summ(winp, "within-subj Y-perm (label null)")
    out = dict(seed=int(args.seed), n_dom=int(n_dom), observed_residual=float(obs["residual"]),
               observed_acc=float(obs["acc"]), fake_subject=r1, within_subj_yperm=r2)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"  [saved] {args.out}", flush=True)
    print(f"DONE seed={args.seed}: obs={obs['residual']:.4f} | fake-subj null={r1['null_mean']:.4f} "
          f"(z={r1['z']:+.1f}) | within-subj null={r2['null_mean']:.4f} (z={r2['z']:+.1f}) | "
          f"BOTH_confirm={r1['confirm'] and r2['confirm']}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_count", type=int, default=60)
    ap.add_argument("--bb_epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n_null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    run(ap.parse_args())
