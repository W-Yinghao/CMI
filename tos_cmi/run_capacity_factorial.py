"""Track C -- architecture x latent-dimension factorial (frozen-feature, BCI-IV-2a LOSO).
Dumps frozen ERM Z at a chosen (backbone, latent dim) over all LOSO folds, into a dim-tagged dir, so the
existing ablation/removability analysis can be run per cell. Goal: disentangle representation TYPE (SPD vs
conv) from latent DIMENSION in the "low-rank removability is representation-dependent" finding (C7 caveat).
  TSMNet dim knob = subspacedims m (tangent dim m(m+1)/2);  EEGNet dim knob = F2 (penultimate width).

  python -m tos_cmi.run_capacity_factorial --backbone TSMNet --dim 8 --seed 0
  python -m tos_cmi.run_capacity_factorial --backbone EEGNet --dim 64 --seed 0
"""
import argparse
import json
import os
from tos_cmi.eeg.feature_dump import dump_fold

# 2a has 9 subjects -> 9 LOSO folds
FOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def backbone_kw_for(backbone, dim):
    if backbone == "TSMNet":
        return {"subspacedims": int(dim)}                      # tangent dim = dim*(dim+1)/2
    if backbone == "EEGNet":
        return {"F1": max(8, int(dim) // 2), "D": 2, "F2": int(dim)}   # penultimate width = F2 = dim
    raise ValueError("unsupported backbone %s" % backbone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", required=True, choices=["TSMNet", "EEGNet"])
    ap.add_argument("--dim", type=int, required=True, help="TSMNet subspacedims m, or EEGNet F2 width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default="tos_cmi/results/tos_cmi_eeg_frozen/factorial")
    args = ap.parse_args()
    kw = backbone_kw_for(args.backbone, args.dim)
    cell = "%s_dim%d" % (args.backbone, args.dim)            # e.g. TSMNet_dim8, EEGNet_dim64
    base = "%s/%s_%s" % (args.out_root, args.dataset, cell)
    os.makedirs(base, exist_ok=True)
    rows = []
    for f in FOLDS:
        out = "%s/sub%d_erm_lam0_seed%d.npz" % (base, f, args.seed)
        try:
            dump_fold(args.dataset, f, "erm", 0.0, args.seed, out, backbone=args.backbone,
                      epochs=args.epochs, device=args.device, backbone_kw=kw)
            rows.append({"fold": f, "ok": True})
            print("[ok] %s sub%d -> %s" % (cell, f, out), flush=True)
        except Exception as e:
            rows.append({"fold": f, "ok": False, "err": repr(e)[:200]})
            print("[FAIL] %s sub%d : %r" % (cell, f, e), flush=True)
    json.dump({"backbone": args.backbone, "dim": args.dim, "kw": kw, "seed": args.seed, "rows": rows},
              open("%s/_manifest_seed%d.json" % (base, args.seed), "w"), indent=1)
    print("FACTORIAL_CELL_DONE %s %d/%d ok" % (cell, sum(r["ok"] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
