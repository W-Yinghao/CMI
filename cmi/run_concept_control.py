"""REAL-DATA positive control for the concept-shift detectors.

Closing validation of `notes/CONCEPT_SHIFT_SECTION.md`: we have shown both detectors read NULL on real
cross-site EEG, and FIRE on *synthetic* concept shift. The one missing link is firing on *real* EEG. Here we
inject a KNOWN, dial-able cohort-dependent concept shift into real cross-site recordings and confirm BOTH
detectors (Route-A delta_d ELBO test; Route-C intercept-residual decoder CMI) rise with the injected strength
alpha and stay silent at alpha=0.

Injection (label-BALANCED => pure CONCEPT shift, not label shift; z-ENCODABLE => the trained Z represents it):
  f = random linear combo of per-channel log-power  (an EEGNet-encodable power feature).
  In ONE cohort ("shifted"), redefine the label rule by f: swap the alpha-fraction of class-0 with the HIGHEST
  f to class-1, and the alpha-fraction of class-1 with the LOWEST f to class-0. Equal counts swapped => the
  per-class marginal is preserved (no label shift), but p(y|f) now differs across cohorts (concept shift), and
  f is a power feature EEGNet learns => Z encodes it => a domain-aware decoder can recover the cohort rule while
  an intercept-only decoder cannot. alpha=0 => identity (the genuine null).

Run one (condition, cohorts, alpha, seed) per process; sweep alpha across SLURM jobs.
"""
import argparse, csv, glob, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

from cmi.run_scps_crossdataset import load as load_xs
from cmi.run_glsvae import extract_features, concept_test_null

RAW = "/projects/EEG-foundation-model/datalake/raw"


# --------------------------------------------------------------------------- domain (D) selection
def _demographics(disease):
    """subject_id ('sub-XXXX') -> {age, sex, race} from all participants.tsv of the disease."""
    M = {}
    for f in glob.glob(f"{RAW}/scps/{disease}/*/participants.tsv"):
        for r in csv.DictReader(open(f), delimiter="\t"):
            age = r.get("age") or r.get("AGE") or ""
            sex = (r.get("gender") or r.get("sex") or r.get("GENDER") or "").strip().upper()[:1]
            race = (r.get("race") or "").strip()
            M[r["participant_id"]] = dict(age=age, sex=sex, race=race)
    return M


def build_domain(dvar, subj, coh, y, disease, n_age_bins=3, min_count=150):
    """Return (d[int, -1=drop], domain_names). Non-degeneracy: keep only D-values that span >=2 classes
    and have >= min_count samples; D=cohort uses the recording site directly."""
    n = len(subj)
    key = lambda s: str(s).split("/")[-1]                       # 'ds003944/sub-1448' -> 'sub-1448'
    dom = np.array(["__drop__"] * n, dtype=object)
    if dvar == "cohort":
        dom = np.array([str(c) for c in coh], dtype=object)
    elif dvar == "subject":                                     # valid ONLY when each subject spans >=2 classes
        dom = np.array([str(s) for s in subj], dtype=object)    # (e.g. PD med-state task: each subj has ON & OFF)
    else:
        M = _demographics(disease)
        if dvar == "sex":
            for i, s in enumerate(subj):
                v = M.get(key(s), {}).get("sex", "")
                if v in ("M", "F"):
                    dom[i] = v
        elif dvar == "race":
            for i, s in enumerate(subj):
                v = M.get(key(s), {}).get("race", "")
                if v:
                    dom[i] = v
        elif dvar == "age":
            ages = np.array([float(M.get(key(s), {}).get("age") or "nan") for s in subj])
            valid = ~np.isnan(ages)
            qs = np.quantile(ages[valid], np.linspace(0, 1, n_age_bins + 1))
            qs[0] -= 1; qs[-1] += 1
            binid = np.clip(np.digitize(ages, qs) - 1, 0, n_age_bins - 1)
            lab = ["young", "mid", "old"] if n_age_bins == 3 else [f"age{i}" for i in range(n_age_bins)]
            for i in range(n):
                if valid[i]:
                    dom[i] = lab[binid[i]]
        else:
            raise ValueError(f"unknown dvar {dvar}")
    keep = np.zeros(n, bool); names = []
    for v in sorted(set(dom.tolist()) - {"__drop__"}):
        m = (dom == v)
        if m.sum() >= min_count and len(set(y[m].tolist())) >= 2:
            keep |= m; names.append(v)
    idmap = {v: i for i, v in enumerate(names)}
    d = np.array([idmap.get(dom[i], -1) for i in range(n)], dtype=np.int64)
    d[~keep] = -1
    return d, names


# --------------------------------------------------------------------------- injection
def _feature(X, kind, feat_seed):
    """An EEGNet-encodable scalar feature per trial, from per-channel log-power."""
    logpow = np.log(X.reshape(X.shape[0], X.shape[1], -1).var(axis=2) + 1e-6)    # [N, n_ch]
    if kind == "sumpow":
        f = logpow.sum(1)                                                        # overall amplitude (very encodable)
    elif kind == "pc1":
        Lc = (logpow - logpow.mean(0)) / (logpow.std(0) + 1e-8)
        _, _, Vt = np.linalg.svd(Lc - Lc.mean(0), full_matrices=False)
        f = Lc @ Vt[0]                                                           # dominant power direction
    else:  # randpow
        f = logpow @ np.random.default_rng(feat_seed).normal(size=logpow.shape[1])
    return (f - f.mean()) / (f.std() + 1e-8)


def inject_concept(X, y, d, alpha, shifted_dom, feat_seed=0, feat_kind="pc1"):
    """Label-balanced, EEGNet-encodable cohort-dependent concept injection. Returns (y_injected, f, n_swapped)."""
    f = _feature(X, feat_kind, feat_seed)
    y2 = y.copy().astype(np.int64)
    n_swap = 0
    if alpha > 0:
        m = (d == shifted_dom)
        for c, take_high in [(0, True), (1, False)]:        # class-0 high-f -> 1 ; class-1 low-f -> 0
            idx = np.where(m & (y == c))[0]
            if len(idx) == 0:
                continue
            order = idx[np.argsort(f[idx])]                 # ascending f
            k = int(round(alpha * len(idx)))
            if k == 0:
                continue
            pick = order[-k:] if take_high else order[:k]
            y2[pick] = 1 - c
            n_swap += k
    return y2, f.astype("float32"), n_swap


# --------------------------------------------------------------------------- Route C (discriminative)
def _mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def fit_decoders(Z, y, d, n_dom, epochs=300, lr=2e-3, seed=0):
    """a(Y|Z) blind, h(Y|Z,D) full, h0=u(Z)+b_D intercept-only. Held-out raw=CE(a)-CE(h), residual=CE(h0)-CE(h)."""
    torch.manual_seed(seed)
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n); cut = n // 2
    fit, ev = idx[:cut], idx[cut:]
    Zt = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y); dt = torch.tensor(d)
    doh = F.one_hot(dt, n_dom).float()
    a = _mlp(Zt.shape[1], 2); h = _mlp(Zt.shape[1] + n_dom, 2); u = _mlp(Zt.shape[1], 2)
    bD = torch.zeros(n_dom, 2, requires_grad=True)
    opt = torch.optim.Adam(list(a.parameters()) + list(h.parameters()) + list(u.parameters()) + [bD], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (F.cross_entropy(a(Zt[fit]), yt[fit])
                + F.cross_entropy(h(torch.cat([Zt[fit], doh[fit]], 1)), yt[fit])
                + F.cross_entropy(u(Zt[fit]) + bD[dt[fit]], yt[fit]))
        loss.backward(); opt.step()
    with torch.no_grad():
        ce_a = F.cross_entropy(a(Zt[ev]), yt[ev]).item()
        ce_h = F.cross_entropy(h(torch.cat([Zt[ev], doh[ev]], 1)), yt[ev]).item()
        ce_0 = F.cross_entropy(u(Zt[ev]) + bD[dt[ev]], yt[ev]).item()
        pa = a(Zt[ev]).argmax(1)                                  # blind Y|Z decodability (balanced acc)
        accs = [(pa[yt[ev] == c] == c).float().mean().item() for c in (0, 1) if (yt[ev] == c).any()]
    return dict(raw=max(ce_a - ce_h, 0.0), residual=max(ce_0 - ce_h, 0.0),
                acc=float(np.mean(accs)) if accs else 0.0)


def route_c_test(Z, y, d, n_dom, n_perm, seed):
    """Observed residual + a domain-permutation null (preserves per-domain counts) => z, p, fired."""
    obs = fit_decoders(Z, y, d, n_dom, seed=seed)
    rng = np.random.default_rng(9000 + seed)
    null = np.array([fit_decoders(Z, y, rng.permutation(d), n_dom, seed=seed)["residual"]
                     for _ in range(n_perm)], dtype=float)
    # NB: this permute-d null DESTROYS d-Z covariate correlation, so it collapses to ~0 with ~0 variance and
    # FALSE-fires on the small covariate-leakage residual present even at alpha=0. The honest null is the
    # alpha=0 real-data baseline (handled in the dose-response analysis), so we floor std and treat the
    # z/p below as a secondary diagnostic only.
    nm, ns = float(null.mean()), float(max(null.std(), 1e-3))
    z = float((obs["residual"] - nm) / ns)
    pv = float((np.sum(null >= obs["residual"]) + 1) / (n_perm + 1))
    return dict(raw=obs["raw"], residual=obs["residual"], acc=obs.get("acc", 0.0), null_mean=nm,
                null_std=ns, z_score=z, p_value=pv, fired=bool(pv < 0.05 and z > 2.0), n_perm=int(n_perm))


# --------------------------------------------------------------------------- main
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, y, subj_s, coh_s, classes = load_xs(args.condition, args.cohorts)
    assert len(classes) == 2, f"injection assumes binary y, got {classes}"
    d_full, dom_names = build_domain(args.dvar, subj_s, coh_s, y, args.condition,
                                     n_age_bins=args.n_age_bins, min_count=args.min_count)
    keep = d_full >= 0
    X, y, d = X[keep], y[keep], d_full[keep]
    if args.collapse_binary:                                # {target domain vs rest} — well-powered 2-way contrast
        tgt = (len(dom_names) - 1) if args.shifted_dom < 0 else args.shifted_dom
        d = (d == tgt).astype(np.int64)
        dom_names = ["rest", dom_names[tgt]]
    n_cls, n_dom = len(classes), len(dom_names)
    assert n_dom >= 2, f"dvar={args.dvar} yielded <2 non-degenerate domains: {dom_names}"
    shifted = args.shifted_dom if args.shifted_dom >= 0 else (n_dom - 1)   # default: last domain
    # per-domain class balance (sanity: confirms non-degeneracy)
    bal = {dom_names[k]: np.bincount(y[d == k], minlength=2).tolist() for k in range(n_dom)}
    print(f"[{args.condition}/D={args.dvar}] X={X.shape} classes={classes} n_dom={n_dom} "
          f"domains={dom_names} shifted={dom_names[shifted]} alpha={args.alpha} "
          f"(kept {keep.sum()}/{len(keep)}; per-domain [n0,n1]={bal}) ({time.time()-t0:.0f}s)", flush=True)

    y_inj, f, n_swap = inject_concept(X, y, d, args.alpha, shifted, feat_seed=args.feat_seed,
                                      feat_kind=args.feat)
    print(f"  [inject] alpha={args.alpha} swapped {n_swap} labels in cohort {shifted}; "
          f"class balance pre={np.bincount(y)} post={np.bincount(y_inj)}", flush=True)

    Z = extract_features(X, y_inj, d, n_cls, args.backbone, args.bb_epochs, args.bs, device, args.seed)

    rc = route_c_test(Z, y_inj, d, n_dom, args.rc_perm, args.seed)
    print(f"  [Route-C] Y|Z_acc={rc['acc']:.3f} raw={rc['raw']:.3f} residual={rc['residual']:.3f} "
          f"null={rc['null_mean']:.3f}±{rc['null_std']:.3f} z={rc['z_score']:.2f} p={rc['p_value']:.3f} "
          f"fired={rc['fired']}", flush=True)

    ra = concept_test_null(Z, y_inj, d, n_cls, n_dom, args, device)
    print(f"  [Route-A] gain={ra['observed_gain']:.4f} null={ra['null_mean']:.4f}±{ra['null_std']:.4f} "
          f"z={ra['z_score']:.2f} p={ra['p_value']:.3f} fired={ra['concept_shift_detected']}", flush=True)

    out = dict(condition=args.condition, dvar=args.dvar, domains=dom_names, shifted_dom=int(shifted),
               alpha=float(args.alpha), seed=int(args.seed), n_swapped=int(n_swap), feat=args.feat,
               route_c=rc, route_a={k: v for k, v in ra.items() if k != "null_gains"})
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"  [saved] {args.out}", flush=True)
    print(f"DONE alpha={args.alpha} seed={args.seed}: "
          f"RC residual={rc['residual']:.3f} (z={rc['z_score']:.1f} fired={rc['fired']}) | "
          f"RA gain={ra['observed_gain']:.4f} (z={ra['z_score']:.1f} fired={ra['concept_shift_detected']})",
          flush=True)
    return out


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="SCZ")
    ap.add_argument("--cohorts", nargs="*", default=["ds003944", "ds003947"])
    ap.add_argument("--dvar", default="cohort", choices=["cohort", "subject", "sex", "age", "race"],
                    help="domain variable D (alpha=0 measures the REAL I(Y;D|Z) across this split)")
    ap.add_argument("--n_age_bins", type=int, default=3)
    ap.add_argument("--min_count", type=int, default=150, help="drop D-values with fewer samples")
    ap.add_argument("--collapse_binary", action="store_true",
                    help="collapse D to {target-vs-rest} for a well-powered 2-way contrast")
    ap.add_argument("--alpha", type=float, default=0.0, help="injected concept-shift strength (0=null)")
    ap.add_argument("--shifted_dom", type=int, default=-1, help="cohort index to inject into (-1=last)")
    ap.add_argument("--feat_seed", type=int, default=0, help="fixed across alpha so the feature direction is shared")
    ap.add_argument("--feat", default="pc1", choices=["pc1", "sumpow", "randpow"],
                    help="encodable injection feature (pc1=dominant power direction)")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--bb_epochs", type=int, default=120)
    ap.add_argument("--rc_perm", type=int, default=100, help="Route-C residual permutation-null size")
    # Route-A (GLS-VAE) knobs consumed by concept_test_null:
    ap.add_argument("--vae_epochs", type=int, default=300)
    ap.add_argument("--delta_epochs", type=int, default=300)
    ap.add_argument("--n_perm", type=int, default=80, help="Route-A domain-permutation null size")
    ap.add_argument("--zy", type=int, default=8)
    ap.add_argument("--zd", type=int, default=6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    return ap.parse_args()


if __name__ == "__main__":
    run(build_args())
