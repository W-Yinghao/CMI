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
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_subjects_arg(vals):
    """None | ['all'] | list of int-strings -> None | 'all' | list[int]."""
    if not vals:
        return None
    if len(vals) == 1 and str(vals[0]).strip().lower() == "all":
        return "all"
    return [int(v) for v in vals]


def _moabb_subject_list(name):
    """The dataset's full subject-id list WITHOUT loading trial data, so an 'all' grid can still
    enumerate expected cells even when the data load fails (dataset invalid for the loader/paradigm,
    or the cache is unavailable)."""
    import moabb.datasets as D
    return sorted(int(s) for s in getattr(D, name)().subject_list)


def _atomic_write_json(path, obj, tmp_token=""):
    """Write JSON via a unique temp file + atomic rename (safe when parallel shards each write the
    same identical grid_manifest.json)."""
    import os
    path = Path(path)
    tmp = Path(f"{path}.tmp{tmp_token}")
    tmp.write_bytes((json.dumps(obj, indent=2) + "\n").encode("utf-8"))
    os.replace(str(tmp), str(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A audited real-EEG mini-grid")
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subject ids to LOAD, or 'all' (MOABB subject_list); default = all")
    ap.add_argument("--target-subjects", nargs="+", required=True,
                    help="target subject ids to iterate, or 'all' (resolved after load)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--align-factor", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch intra-op threads for CPU multi-core (0 = leave torch default)")
    ap.add_argument("--root-outdir", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose run dir already has an ok/skipped manifest")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run dir first")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)
    if args.resume and args.overwrite:
        ap.error("--resume and --overwrite are mutually exclusive")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        ap.error("require num_shards >= 1 and 0 <= shard_index < num_shards")

    from h2cmi import run_real_audited as R
    R._maybe_set_threads(args.threads)                         # CPU multi-core (SLURM passes --threads)

    subjects_spec = _resolve_subjects_arg(args.subjects)       # None | 'all' | list[int]
    targets_spec = _resolve_subjects_arg(args.target_subjects)  # 'all' | list[int]
    load_subjects = None if subjects_spec in (None, "all") else subjects_spec

    root = Path(args.root_outdir)
    root.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version.split()[0], "device": args.device}
    align_factor = args.align_factor or ("site" if args.dataset == "synthetic" else "subject")

    # ---- load ONCE (reused across all target/seed configs) ------------------------------
    load_err = None
    subject_map = None
    n_classes = None
    try:
        if args.dataset == "synthetic":
            X, y, dag, domains, subj_col, _s, n_classes, info = R._load_synthetic(args.n_classes, 0)
        else:
            X, y, dag, domains, subj_col, _s, n_classes, info = R._load_moabb(
                args.dataset, load_subjects, 0)
            subject_map = info["subject_map"]
    except Exception as exc:                                   # reason-coded; every run skips
        load_err = f"{type(exc).__name__}: {exc}"

    # resolve 'all' target subjects AFTER load, so grid_manifest records CONCRETE ids. Falls back to
    # the dataset's subject_list if the data load fails (so skips still enumerate every cell).
    if targets_spec == "all":
        if args.dataset == "synthetic":
            target_subjects = sorted({int(x) for x in subj_col}) if load_err is None else []
        elif subject_map is not None:
            target_subjects = sorted(int(k) for k in subject_map)
        else:
            target_subjects = _moabb_subject_list(args.dataset)
    else:
        target_subjects = [int(t) for t in targets_spec]

    # domain-factor structure is invariant across (target, seed) — compute once
    factor_levels = align_degenerate = None
    if load_err is None:
        factor_levels = {f.name: int(f.n_levels) for f in dag.factors}
        align_degenerate = factor_levels.get(align_factor, 0) <= 1

    # grid_manifest.json records the RESOLVED expected grid so the validator distinguishes a MISSING
    # cell (no dir at all) from a legal skip. Written atomically by every shard (identical content).
    expected_cells = [{"target_subject": int(t), "seed": int(s)}
                      for t in target_subjects for s in args.seeds]
    _atomic_write_json(root / "grid_manifest.json", {
        "dataset": args.dataset,
        "subjects": ("all" if subjects_spec == "all" else load_subjects),
        "target_subjects": [int(t) for t in target_subjects],
        "seeds": [int(s) for s in args.seeds],
        "epochs": int(args.epochs), "fast": bool(args.fast), "align_factor": align_factor,
        "n_classes": int(n_classes) if (load_err is None and n_classes is not None) else None,
        "expected_cells": expected_cells,
    }, tmp_token=str(args.shard_index))

    cells = [(t, s) for t in target_subjects for s in args.seeds]
    n_ok = n_skip = n_resume = 0
    for tsub in target_subjects:
        for seed in args.seeds:
            if cells.index((tsub, seed)) % args.num_shards != args.shard_index:
                continue                                    # deterministic sharding (SLURM array)
            run_dir = root / f"dataset={args.dataset}_target={tsub}_seed={seed}"
            if args.resume and (run_dir / "run_manifest.json").exists():   # skip completed cells
                try:
                    prev = json.loads((run_dir / "run_manifest.json").read_text())
                except Exception:
                    prev = {}
                if prev.get("status") in ("ok", "skipped"):
                    print(f"  resume-skip target={tsub} seed={seed}")
                    n_resume += 1
                    continue
            if args.overwrite and run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            run_args = argparse.Namespace(dataset=args.dataset, subjects=args.subjects,
                                          target_subject=tsub, seed=seed)
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

    print(f"grid done -> {root}  (ok={n_ok} skipped={n_skip} resume-skipped={n_resume})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
