"""Project A — audited real-EEG mini-grid over (target subject, seed).

Loads the dataset ONCE and runs `run_real_audited._run_pilot` for each (target, seed), writing
one audited run directory per config (`dataset=..._target=..._seed=...`). Each run gets the same
4 artifacts as the single-run pilot; a failed/unavailable config writes a legal SKIP artifact and
the grid continues. Feed the resulting root to `h2cmi.observability.validate_results` to emit the
tracked summary digest. Not a SOTA run — an audited claim-boundary validation.

  python -m h2cmi.run_real_audited_grid --dataset BNCI2014_001 --subjects 1 2 3 \
      --target-subjects 1 2 3 --seeds 0 1 2 --epochs 20 --fast --align-factor subject \
      --device cuda --root-outdir notes/project_A_observability/results/step8_bnci2014_001_minigrid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A audited real-EEG mini-grid")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--target-subjects", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--align-factor", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root-outdir", required=True)
    args = ap.parse_args(argv)

    from h2cmi import run_real_audited as R

    root = Path(args.root_outdir)
    root.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version.split()[0], "device": args.device}
    align_factor = args.align_factor or ("site" if args.dataset == "synthetic" else "subject")

    # ---- load ONCE (reused across all target/seed configs) ------------------------------
    load_err = None
    subject_map = None
    try:
        if args.dataset == "synthetic":
            X, y, dag, domains, subj_col, _s, n_classes, info = R._load_synthetic(args.n_classes, 0)
        else:
            X, y, dag, domains, subj_col, _s, n_classes, info = R._load_moabb(
                args.dataset, args.subjects, 0)
            subject_map = info["subject_map"]
    except Exception as exc:                                   # reason-coded; every run skips
        load_err = f"{type(exc).__name__}: {exc}"

    # domain-factor structure is invariant across (target, seed) — compute once
    factor_levels = align_degenerate = None
    if load_err is None:
        factor_levels = {f.name: int(f.n_levels) for f in dag.factors}
        align_degenerate = factor_levels.get(align_factor, 0) <= 1

    n_ok = n_skip = 0
    for tsub in args.target_subjects:
        for seed in args.seeds:
            run_dir = root / f"dataset={args.dataset}_target={tsub}_seed={seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_args = argparse.Namespace(dataset=args.dataset, subjects=args.subjects,
                                          target_subject=tsub)
            if load_err is not None:
                R._write_skip(run_dir, f"load failed: {load_err}", run_args, environment)
                n_skip += 1
                continue
            # resolve target index
            if args.dataset == "synthetic":
                target_idx = tsub
            else:
                if str(tsub) not in subject_map:
                    R._write_skip(run_dir, f"target {tsub} not in {sorted(subject_map)}",
                                  run_args, environment)
                    n_skip += 1
                    continue
                target_idx = subject_map[str(tsub)]
            try:
                cfg = R._build_cfg(n_classes, X.shape[1], X.shape[2], args.epochs, args.device,
                                   seed, args.fast)
                _res, _leak, raw = R._run_pilot(X, y, dag, domains, subj_col, target_idx, cfg,
                                                n_classes, args.n_perm, align_factor=align_factor)
            except Exception as exc:
                R._write_skip(run_dir, f"pilot failed: {type(exc).__name__}: {exc}",
                              run_args, environment)
                n_skip += 1
                continue
            data = R._finalize(run_dir, raw, dataset=args.dataset, subjects=args.subjects,
                               target_subject=tsub, target_index=target_idx, seed=seed,
                               epochs=args.epochs, fast=args.fast, device=args.device,
                               align_factor=align_factor, factor_levels=factor_levels,
                               align_degenerate=align_degenerate, n_classes=n_classes,
                               n_chans=X.shape[1], n_times=X.shape[2],
                               dataset_info={k: v for k, v in info.items() if k != "subject_map"},
                               environment=environment)
            n_ok += 1
            s = data["summary"]
            print(f"  ok  target={tsub} seed={seed}  claims={s['n_claims']} "
                  f"allowed={s['n_allowed']} rejected={s['n_rejected']} "
                  f"violations={len(data['forbidden_claims_violated'])}")

    print(f"grid done -> {root}  (ok={n_ok} skipped={n_skip})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
