#!/usr/bin/env python
"""CMI-Trace P1.1 — put the SAME posterior-KL CMI ruler on full and erased TOS latents.

For each valid source fold/backbone/dataset: fit erasers (TOS_VD / LEACE / RLACE / INLP / same-rank random)
on a DISJOINT eraser-fit split of SOURCE features, apply to all source rows, then measure the flat-feature
conditional-subject-leakage ruler (cmi.eval.conditional_subject_leakage) on a disjoint posterior-train/eval
split. Reuses the SAME (ptrain, peval) split across transforms for a paired table connecting:

    CMI amount -> linear/nonlinear removability -> (target effect, when a target dump is present)

Consumes TOS frozen feature dumps (tos_cmi/results/tos_cmi_eeg_frozen/<DATASET>_<BACKBONE>_LOSO/
sub*_erm_lam0_seed*.npz). If dumps are absent (pruned from git), runs a SYNTHETIC demo so the pipeline is
verified end-to-end; the real command is printed. Erasers that are unavailable/error are skipped with a logged
reason (never silently treated as zero).

  # synthetic demo (no dumps needed):
  python scripts/run_cmi_trace_tos_cmi_bridge.py --synthetic
  # real (needs frozen dumps):
  python scripts/run_cmi_trace_tos_cmi_bridge.py --dataset BNCI2014_001 --backbone EEGNet
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmi.eval.conditional_subject_leakage import three_way_support_split, cmi_ruler_across_transforms  # noqa

OUT_DIR = REPO / "results" / "cmi_trace_p0p1" / "tos_cmi_bridge"


def _random_k_transform(Z, subj_fit_ids, k, seed):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((Z.shape[1], k))
    Q, _ = np.linalg.qr(V)                                  # orthonormal same-rank random subspace
    return Z - (Z @ Q) @ Q.T


def _subject_subspace_transform(Zfit, yfit, dfit, Z, k):
    """Label-conditional subject-subspace removal fit on the ERASER split only (a TOS_VD/LEACE-style eraser)."""
    rows = []
    for c in np.unique(yfit):
        mu = Zfit[yfit == c].mean(0)
        for g in np.unique(dfit[yfit == c]):
            m = (yfit == c) & (dfit == g)
            if m.sum() > 0:
                rows.append(Zfit[m].mean(0) - mu)
    M = np.vstack(rows) if rows else np.zeros((1, Z.shape[1]))
    Vt = np.linalg.svd(M, full_matrices=False)[2]
    S = Vt[:min(k, Vt.shape[0])]
    return Z - (Z @ S.T) @ S


def _try_tos_erasers(Zfit, subj_fit, Z, k):
    """Optional: real TOS erasers (LEACE/RLACE/INLP) if importable; returns {name: transformed or None}."""
    out = {}
    try:
        from tos_cmi.eeg.erasure_baselines import leace_eraser
        E = leace_eraser(Zfit, subj_fit)
        out["LEACE"] = E(Z) if callable(E) else None
    except Exception as e:
        out["LEACE_skip_reason"] = str(e)[:120]
    return out


def _load_dump(path):
    z = np.load(path, allow_pickle=False)
    keys = {k: z[k] for k in z.files}
    return keys


def _synthetic_fold(seed=0, n_cls=2, n_dom=4, per=80, dim=16):
    rng = np.random.default_rng(seed)
    cm = rng.standard_normal((n_cls, dim)) * 2
    sd = rng.standard_normal((n_dom, dim)); sd /= np.linalg.norm(sd, axis=1, keepdims=True)
    Z, y, d = [], [], []
    for c in range(n_cls):
        for g in range(n_dom):
            Z.append(cm[c] + 2.5 * sd[g] + 0.5 * rng.standard_normal((per, dim)))
            y += [c] * per; d += [g] * per
    return np.vstack(Z), np.array(y), np.array(d)


def run_one(Z, y, d, n_cls, n_dom, seed, k, n_perm, epochs):
    er, pt, pe, split_diag = three_way_support_split(y, d, seed=seed)
    Zfit, yfit, dfit = Z[er], y[er], d[er]
    transforms = {"full": Z,
                  "TOS_VD": _subject_subspace_transform(Zfit, yfit, dfit, Z, k),
                  "random_k": _random_k_transform(Z, dfit, k, seed)}
    skips = {}
    for name, val in _try_tos_erasers(Zfit, dfit, Z, k).items():
        if name.endswith("_skip_reason"):
            skips[name] = val
        elif val is not None:
            transforms[name] = np.asarray(val)
    table = cmi_ruler_across_transforms(transforms, y, d, n_cls, n_dom, pt, pe,
                                        n_perm=n_perm, seed=seed, epochs=epochs)
    return {"table": table, "split_diag": split_diag, "eraser_skips": skips, "k": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n_perm", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--limit", type=int, default=2, help="max dumps to process (real mode)")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        Z, y, d = _synthetic_fold(seed=0)
        res = run_one(Z, y, d, 2, 4, seed=0, k=a.k, n_perm=a.n_perm, epochs=a.epochs)
        json.dump(res, open(OUT_DIR / "synthetic_demo.json", "w"), indent=2, default=float)
        print("[tos-bridge] synthetic demo:")
        for nm, r in res["table"].items():
            print(f"  {nm:10s} kl={r['posterior_kl_nats']:.4f} excess={r['excess_over_null']:+.4f} "
                  f"p={r['perm_p']:.3f} norm={r['normalized_leakage']:.3f} "
                  f"res_lin={r['subject_residual_linear']:.3f} res_mlp={r['subject_residual_mlp']:.3f}")
        return 0

    ddir = REPO / "tos_cmi" / "results" / "tos_cmi_eeg_frozen" / f"{a.dataset}_{a.backbone}_LOSO"
    dumps = sorted(glob.glob(str(ddir / "sub*_erm_lam0_seed*.npz")))
    if not dumps:
        print(f"[tos-bridge] NO frozen dumps under {ddir}. Regenerate via tos_cmi/eeg/feature_dump.py, then "
              f"rerun. Falling back to --synthetic demo for pipeline verification.")
        return main.__wrapped__() if hasattr(main, "__wrapped__") else run_synthetic_and_exit(a)
    all_res = []
    for p in dumps[: a.limit or len(dumps)]:
        data = _load_dump(p)
        # expected keys in a TOS frozen dump: source features Zs + subject ids + labels (names vary by dumper)
        Z = data.get("Zs", data.get("features"))
        y = data.get("ys", data.get("y")); d = data.get("subj_s", data.get("d"))
        if Z is None or y is None or d is None:
            print(f"[tos-bridge] {Path(p).name}: unrecognized dump keys {list(data)[:8]}; skipping")
            continue
        y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
        res = run_one(np.asarray(Z, float), y, d, int(y.max() + 1), int(d.max() + 1),
                      seed=0, k=a.k, n_perm=a.n_perm, epochs=a.epochs)
        res["dump"] = Path(p).name
        all_res.append(res)
    json.dump(all_res, open(OUT_DIR / f"{a.dataset}_{a.backbone}_bridge.json", "w"), indent=2, default=float)
    print(f"[tos-bridge] wrote {len(all_res)} fold tables -> {OUT_DIR}")
    return 0


def run_synthetic_and_exit(a):
    Z, y, d = _synthetic_fold(seed=0)
    res = run_one(Z, y, d, 2, 4, seed=0, k=a.k, n_perm=a.n_perm, epochs=a.epochs)
    json.dump(res, open(OUT_DIR / "synthetic_demo.json", "w"), indent=2, default=float)
    print("[tos-bridge] synthetic demo written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
