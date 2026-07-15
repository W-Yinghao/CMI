"""Phase 2.0 driver -- frozen-feature EEG pilot (BNCI2014_001 / 2a, TSMNet LogEig, LOSO).

Dumps frozen Z/logits for ERM + global-LPC over LOSO folds x configs x seed; the score-Fisher
DIAGNOSTIC is run separately by tos_cmi.eeg.report on the dumped artifacts. No trainer wiring, no
selective-penalty training, no deletion. GPU required (TSMNet); run via scripts/tos_eeg_frozen_pilot.sbatch.

  # plumbing (one fold, reduced epochs):
  python -m tos_cmi.run_eeg_frozen_pilot --target-subjects 1 --configs erm:0 lpc_prior:0.3 lpc_prior:1.0 \
      --epochs 40 --seed 0 --device cuda
  # full first round:
  python -m tos_cmi.run_eeg_frozen_pilot --target-subjects all --configs erm:0 lpc_prior:0.03 lpc_prior:0.1 \
      lpc_prior:0.3 lpc_prior:1.0 lpc_prior:3.0 --epochs 300 --seed 0 --device cuda
"""
import argparse
import json
import os
import time

from tos_cmi.eeg.feature_dump import dump_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="TSMNet")
    ap.add_argument("--target-subjects", nargs="+", default=["1"], help="ints or 'all'")
    ap.add_argument("--configs", nargs="+", default=["erm:0", "lpc_prior:0.3", "lpc_prior:1.0"],
                    help="method:lam (global-LPC = lpc_prior:LAM; ERM = erm:0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--domain-mode", default="subject", choices=["subject", "subject_session"])
    ap.add_argument("--out-root", default="tos_cmi/results/tos_cmi_eeg_frozen")
    args = ap.parse_args()

    if args.target_subjects == ["all"]:
        import moabb.datasets as D
        subjects = [int(s) for s in getattr(D, args.dataset)().subject_list]  # real LOSO folds (NOT hardcoded 9)
    else:
        subjects = [int(s) for s in args.target_subjects]
    print("LOSO over %d subjects for %s: %s" % (len(subjects), args.dataset, subjects), flush=True)
    base = "%s/%s_%s_LOSO" % (args.out_root, args.dataset, args.backbone)
    os.makedirs(base, exist_ok=True)
    rows = []
    for tgt in subjects:
        for cfg in args.configs:
            method, lam = cfg.split(":"); lam = float(lam)
            tag = "sub%d_%s_lam%g_seed%d" % (tgt, method, lam, args.seed)
            out = "%s/%s.npz" % (base, tag)
            if os.path.exists(out):                         # idempotent: never recompute a banked fold
                print("[skip-existing] %s" % tag, flush=True); continue
            t0 = time.time()
            try:
                dump_fold(args.dataset, tgt, method, lam, args.seed, out, backbone=args.backbone,
                          epochs=args.epochs, device=args.device, domain_mode=args.domain_mode)
                rows.append({"tag": tag, "target": tgt, "method": method, "lam": lam,
                             "out": out, "ok": True, "secs": round(time.time() - t0, 1)})
                print("[ok] %s (%.0fs) -> %s" % (tag, time.time() - t0, out), flush=True)
            except Exception as e:
                rows.append({"tag": tag, "target": tgt, "method": method, "lam": lam,
                             "ok": False, "error": repr(e)[:300]})
                print("[FAIL] %s : %r" % (tag, e), flush=True)
    summ = "%s/summary.json" % base
    json.dump({"dataset": args.dataset, "backbone": args.backbone, "seed": args.seed,
               "epochs": args.epochs, "configs": args.configs, "rows": rows}, open(summ, "w"), indent=1)
    print("wrote", summ)
    print("EEG_FROZEN_PILOT_DONE %d/%d ok" % (sum(r["ok"] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
