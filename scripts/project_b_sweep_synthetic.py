"""Project B Step-2A: synthetic substrate repair sweep.

Goal: find ONE reproducible "recoverable" world (raw OFFLINE_TTA beats IDENTITY) and ONE
reproducible "harmful" world (raw OFFLINE_TTA clearly hurts, and the held-out target site is
actually a concept-shifted site) so that Project B has a valid go/no-go substrate. The Step-1
baselines showed the `--fast` (4-epoch) worlds are under-trained and do NOT separate; this sweep
uses full training (no `--fast`) and records `target_concept_hit`.

This script does NOT modify any h2cmi core code. It reuses the h2cmi library functions in the
EXACT same order as `h2cmi.run_synthetic.main()` (in-process, rather than subprocess+parse-JSON),
because the driver's JSON never emits `target_sites` and we must know whether the held-out target
site is one of the concept-shifted sites. The train/target split is additionally reconstructed
from its RNGs (cheap, no training) so harmful configs can be pre-scanned for hit-seeds, and every
full run asserts the reconstruction matches the real simulator.

Modes:
  --mode selftest            validate the RNG split-reconstruction against a real EEGSimulator build
  --mode prescan  --config H1 [--nseeds 51] [--need 3]
                             cheap: list the first `need` seeds whose target site hits a concept site
  --mode run --world recoverable|harmful --config R1 --seed 0 --out DIR
                             one full training run -> DIR/<world>_<cid>_seed<seed>.json {row, report}
  --mode aggregate --dir DIR --out DIR/summary.csv
                             read all run JSONs -> summary.csv + apply acceptance criteria + best configs

Threads are pinned to 1 (process-level parallelism via a SLURM array is the intended launcher).
"""
from __future__ import annotations

# pin BLAS/OMP threads BEFORE importing numpy/torch (process-level parallelism)
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import glob
import json
import math
import sys

# make `import h2cmi` resolve to THIS worktree's package regardless of cwd / how we are launched
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# --------------------------------------------------------------------------- config tables
COMMON = dict(classes=3, chans=16, times=128, fs=128.0, sites=6, subjects=4, sessions=2,
              trials=60, noise=0.25, label_rho=0.0, epochs=30, bs=64, device="cpu",
              eval_unit="subject")

RECOVERABLE = {
    "R1": dict(cov=0.8, prior=0.4, montage=0.2, concept=0.0, concept_frac=0.0),
    "R2": dict(cov=1.2, prior=0.4, montage=0.2, concept=0.0, concept_frac=0.0),
    "R3": dict(cov=1.6, prior=0.4, montage=0.2, concept=0.0, concept_frac=0.0),
    "R4": dict(cov=1.2, prior=0.8, montage=0.2, concept=0.0, concept_frac=0.0),
    "R5": dict(cov=1.2, prior=0.4, montage=0.4, concept=0.0, concept_frac=0.0),
    "R6": dict(cov=1.6, prior=0.8, montage=0.4, concept=0.0, concept_frac=0.0),
}
HARMFUL = {
    "H1": dict(cov=0.8, prior=0.4, montage=0.2, concept=0.6, concept_frac=0.17),
    "H2": dict(cov=0.8, prior=0.4, montage=0.2, concept=0.8, concept_frac=0.17),
    "H3": dict(cov=0.8, prior=0.4, montage=0.2, concept=1.0, concept_frac=0.17),
    "H4": dict(cov=1.2, prior=0.4, montage=0.2, concept=0.8, concept_frac=0.17),
    "H5": dict(cov=0.8, prior=0.8, montage=0.2, concept=0.8, concept_frac=0.17),
    "H6": dict(cov=1.2, prior=0.8, montage=0.3, concept=1.0, concept_frac=0.17),
}

SUMMARY_COLS = [
    "world", "config", "seed", "epochs", "sites", "subjects", "sessions", "trials",
    "cov", "prior", "montage", "noise", "concept", "concept_frac",
    "target_sites", "concept_sites", "target_concept_hit",
    "strict_bacc", "offline_delta_bacc",
    "offline_gain_mean", "offline_gain_lo", "offline_gain_hi", "offline_p_gt0",
    "gate_coverage", "gate_avoided_harm", "gate_missed_benefit", "gate_selective_gain",
    "gate_auroc", "gate_n_pseudo",
    "leakage_site_excess", "leakage_subject_excess", "leakage_session_excess",
]


def _params(world: str, cid: str) -> dict:
    table = RECOVERABLE if world == "recoverable" else HARMFUL
    if cid not in table:
        raise SystemExit(f"unknown config {cid} for world {world}")
    return table[cid]


# --------------------------------------------------------------------------- RNG split reconstruction
# These mirror EXACTLY how h2cmi/data/eeg_simulator.py draws the concept sites (first consumption of
# default_rng(seed+1) inside sample(); line ~162) and the target site (default_rng(seed) inside
# train_target_split; line ~250). Reconstruction is validated against the real simulator in run_one.
def reconstruct_concept_sites(n_sites: int, concept_frac: float, seed: int) -> list[int]:
    n_concept = int(round(concept_frac * n_sites))
    if n_concept <= 0:
        return []
    rng = np.random.default_rng(seed + 1)
    return sorted(rng.choice(n_sites, size=n_concept, replace=False).tolist())


def reconstruct_target_site(n_sites: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    sites = np.arange(n_sites)          # == np.unique(sim.site); all sites are populated
    return int(rng.choice(sites, size=min(1, n_sites - 1), replace=False)[0])


def target_concept_hit(n_sites: int, concept_frac: float, seed: int) -> bool:
    return reconstruct_target_site(n_sites, seed) in set(
        reconstruct_concept_sites(n_sites, concept_frac, seed))


# --------------------------------------------------------------------------- one full run (in-process)
def run_one(world: str, cid: str, seed: int, out_dir: str, leakage_nperm: int = 20) -> dict:
    # heavy imports deferred so selftest/prescan/aggregate stay light
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    from h2cmi.config import H2Config
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    from h2cmi.train.trainer import train_h2, reference_prior
    from h2cmi.eval.harness import run_three_settings
    from h2cmi.eval.leakage import crossfit_conditional_leakage

    p = _params(world, cid)
    shift = ShiftSpec(cov=p["cov"], prior=p["prior"], concept=p["concept"],
                      concept_site_frac=p["concept_frac"], montage=p["montage"],
                      noise=COMMON["noise"], label_mechanism_rho=COMMON["label_rho"])
    sim = EEGSimulator(COMMON["classes"], COMMON["chans"], COMMON["times"], COMMON["fs"],
                       shift=shift, seed=seed).sample(
        COMMON["sites"], COMMON["subjects"], COMMON["sessions"], COMMON["trials"])
    src_idx, tgt_idx = train_target_split(sim, n_target_sites=1, seed=seed)

    # config MIRRORS run_synthetic.build_config but with fast DISABLED (full training)
    cfg = H2Config(n_classes=COMMON["classes"])
    cfg.encoder.n_chans = COMMON["chans"]
    cfg.encoder.n_times = COMMON["times"]
    cfg.encoder.fs = COMMON["fs"]
    cfg.train.epochs = COMMON["epochs"]
    cfg.train.device = COMMON["device"]
    cfg.train.seed = seed
    cfg.train.batch_size = COMMON["bs"]

    Xs, ys = sim.X[src_idx], sim.y[src_idx]
    src_domains = sim.domains.subset(src_idx)
    model, hcmi, dual, hist = train_h2(Xs, ys, src_domains, sim.dag, cfg,
                                       align_factor="site", verbose=False)
    pi_star = reference_prior(ys, COMMON["classes"], cfg.align.reference_prior)

    Xt, yt = sim.X[tgt_idx], sim.y[tgt_idx]
    tgt_unit = sim.domains.subset(tgt_idx).factor(COMMON["eval_unit"])
    src_unit = src_domains.factor("subject")            # gate pseudo-targets = source subjects
    results = run_three_settings(model, Xt, yt, tgt_unit, cfg, pi_star,
                                 X_src=Xs, y_src=ys, gate_pseudo_levels=src_unit,
                                 device=cfg.train.device)

    Zs = model.embed(Xs, device=cfg.train.device)
    leak = crossfit_conditional_leakage(Zs, ys, src_domains, sim.dag, COMMON["classes"],
                                        device=cfg.train.device, n_perm=leakage_nperm, seed=seed)

    # split facts from the REAL simulator + validate the cheap reconstruction agrees
    tgt_sites = sorted(np.unique(sim.site[tgt_idx]).tolist())
    concept_sites = [int(s) for s in sim.meta["concept_sites"]]
    hit = bool(set(tgt_sites) & set(concept_sites))
    assert tgt_sites == [reconstruct_target_site(COMMON["sites"], seed)], \
        f"target-site reconstruction mismatch: real={tgt_sites} recon={reconstruct_target_site(COMMON['sites'], seed)}"
    assert concept_sites == reconstruct_concept_sites(COMMON["sites"], p["concept_frac"], seed), \
        f"concept-site reconstruction mismatch: real={concept_sites}"

    sr = results["offline_tta"]["selective_risk"]
    gb = results["offline_tta"]["gain_bootstrap"]
    hm = results["gate_info"]["harm_metrics"]
    row = dict(
        world=world, config=cid, seed=seed, epochs=cfg.train.epochs,
        sites=COMMON["sites"], subjects=COMMON["subjects"], sessions=COMMON["sessions"],
        trials=COMMON["trials"], cov=p["cov"], prior=p["prior"], montage=p["montage"],
        noise=COMMON["noise"], concept=p["concept"], concept_frac=p["concept_frac"],
        target_sites=tgt_sites, concept_sites=concept_sites, target_concept_hit=hit,
        strict_bacc=float(results["strict_dg"]["balanced_acc"]),
        offline_delta_bacc=float(results["offline_tta"]["delta_adapt"]["d_balanced_acc"]),
        offline_gain_mean=float(gb["mean"]), offline_gain_lo=float(gb["lo"]),
        offline_gain_hi=float(gb["hi"]), offline_p_gt0=float(gb["p_gt0"]),
        gate_coverage=float(sr["coverage"]), gate_avoided_harm=float(sr["avoided_harm"]),
        gate_missed_benefit=float(sr["missed_benefit"]), gate_selective_gain=float(sr["selective_gain"]),
        gate_auroc=float(hm["auroc"]), gate_n_pseudo=int(results["gate_info"]["n_pseudo"]),
        leakage_site_excess=float(leak["site"]["excess"]),
        leakage_subject_excess=float(leak["subject"]["excess"]),
        leakage_session_excess=float(leak["session"]["excess"]),
    )

    # report mirrors run_synthetic's core report + the split facts (deliverable: 3 JSON reports)
    report = dict(
        config=dict(world=world, cid=cid, seed=seed, classes=COMMON["classes"],
                    sites=COMMON["sites"], shift=vars(shift), epochs=cfg.train.epochs,
                    eval_unit=COMMON["eval_unit"], concept_sites=concept_sites,
                    target_sites=tgt_sites, target_concept_hit=hit),
        strict_dg={k: v for k, v in results["strict_dg"].items() if k != "per_domain_bacc"},
        offline_tta=dict(delta_adapt=results["offline_tta"]["delta_adapt"],
                         delta_selective=results["offline_tta"]["delta_selective"],
                         gain_bootstrap=results["offline_tta"]["gain_bootstrap"],
                         selective_risk=results["offline_tta"]["selective_risk"]),
        online_tta={k: v for k, v in results["online_tta"].items() if k != "per_domain_bacc"},
        gate_info=results["gate_info"],
        leakage=leak,
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{world}_{cid}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(dict(row=row, report=report), f, indent=2, default=float)
    print(f"[run] {world}/{cid}/seed{seed}: strict={row['strict_bacc']:.3f} "
          f"d_off={row['offline_delta_bacc']:+.3f} hit={hit} -> {path}")
    return row


# --------------------------------------------------------------------------- aggregate + acceptance
def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _recoverable_pass(rows: list[dict]) -> bool:
    d = [r["offline_delta_bacc"] for r in rows]
    return (_mean([r["strict_bacc"] for r in rows]) >= 0.55
            and _mean(d) >= 0.05
            and sum(1 for x in d if x > 0) >= 2
            and _mean([r["offline_p_gt0"] for r in rows]) >= 0.60)


def _harmful_pass(rows: list[dict]) -> bool:
    d = [r["offline_delta_bacc"] for r in rows]
    return (all(r["target_concept_hit"] for r in rows)
            and _mean([r["strict_bacc"] for r in rows]) >= 0.55
            and _mean(d) <= -0.05
            and sum(1 for x in d if x < 0) >= 2
            and _mean([r["offline_p_gt0"] for r in rows]) <= 0.40)


def aggregate(run_dir: str, out_csv: str) -> None:
    rows = []
    for fp in sorted(glob.glob(os.path.join(run_dir, "*_seed*.json"))):
        with open(fp) as f:
            rows.append(json.load(f)["row"])
    if not rows:
        raise SystemExit(f"no run JSONs found in {run_dir}")

    def _fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.6g}"
        if isinstance(v, list):
            return "|".join(str(x) for x in v)
        return str(v)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLS)
        for r in sorted(rows, key=lambda r: (r["world"], r["config"], r["seed"])):
            w.writerow([_fmt(r.get(c)) for c in SUMMARY_COLS])
    print(f"[aggregate] wrote {out_csv} ({len(rows)} rows)")

    # group by (world, config)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["world"], r["config"]), []).append(r)

    print("\n===== per-config summary =====")
    rec_pass, harm_pass = [], []
    for (world, cid), rs in sorted(groups.items()):
        mb = _mean([r["strict_bacc"] for r in rs])
        md = _mean([r["offline_delta_bacc"] for r in rs])
        mp = _mean([r["offline_p_gt0"] for r in rs])
        hits = sum(1 for r in rs if r["target_concept_hit"])
        pos = sum(1 for r in rs if r["offline_delta_bacc"] > 0)
        neg = sum(1 for r in rs if r["offline_delta_bacc"] < 0)
        ok = _recoverable_pass(rs) if world == "recoverable" else _harmful_pass(rs)
        print(f"  {world:11s} {cid}  n={len(rs)}  strict={mb:.3f}  d_off={md:+.3f}  "
              f"p_gt0={mp:.2f}  hit={hits}/{len(rs)}  pos/neg={pos}/{neg}  PASS={ok}")
        if world == "recoverable" and ok:
            rec_pass.append((md, mb, cid, rs))
        if world == "harmful" and ok:
            harm_pass.append((md, mb, cid, rs))

    print("\n===== best configs (vs SS3.4 acceptance) =====")
    if rec_pass:
        rec_pass.sort(key=lambda t: (-t[0], -t[1]))          # biggest reliable gain, then higher strict
        md, mb, cid, rs = rec_pass[0]
        print(f"  BEST RECOVERABLE = {cid}  d_off={md:+.3f} strict={mb:.3f} seeds={[r['seed'] for r in rs]}")
    else:
        print("  BEST RECOVERABLE = NONE PASSED (see per-config table for closest)")
    if harm_pass:
        harm_pass.sort(key=lambda t: (t[0], -t[1]))          # most harm (most negative), then higher strict
        md, mb, cid, rs = harm_pass[0]
        print(f"  BEST HARMFUL     = {cid}  d_off={md:+.3f} strict={mb:.3f} seeds={[r['seed'] for r in rs]}")
    else:
        print("  BEST HARMFUL     = NONE PASSED (see per-config table for closest)")


# --------------------------------------------------------------------------- selftest / prescan
def selftest() -> None:
    """Validate the RNG split-reconstruction against a real EEGSimulator build (no training)."""
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec, train_target_split
    checks = 0
    for cid, p in list(HARMFUL.items())[:3] + list(RECOVERABLE.items())[:1]:
        for seed in range(6):
            shift = ShiftSpec(cov=p["cov"], prior=p["prior"], concept=p["concept"],
                              concept_site_frac=p["concept_frac"], montage=p["montage"],
                              noise=COMMON["noise"], label_mechanism_rho=COMMON["label_rho"])
            sim = EEGSimulator(COMMON["classes"], COMMON["chans"], COMMON["times"], COMMON["fs"],
                               shift=shift, seed=seed).sample(
                COMMON["sites"], COMMON["subjects"], COMMON["sessions"], COMMON["trials"])
            _, tgt_idx = train_target_split(sim, n_target_sites=1, seed=seed)
            real_tgt = sorted(np.unique(sim.site[tgt_idx]).tolist())
            real_con = sorted(int(s) for s in sim.meta["concept_sites"])
            assert real_tgt == [reconstruct_target_site(COMMON["sites"], seed)], (cid, seed, real_tgt)
            assert real_con == reconstruct_concept_sites(COMMON["sites"], p["concept_frac"], seed), \
                (cid, seed, real_con)
            checks += 1
    print(f"[selftest] RNG split reconstruction matches the real simulator on {checks} (config,seed) pairs OK")


def prescan(cid: str, nseeds: int, need: int) -> None:
    p = _params("harmful", cid)
    hits = [s for s in range(nseeds) if target_concept_hit(COMMON["sites"], p["concept_frac"], s)]
    chosen = hits[:need]
    print(f"[prescan] {cid} concept_frac={p['concept_frac']} n_sites={COMMON['sites']}: "
          f"{len(hits)}/{nseeds} hit seeds; first {need} = {chosen}")
    print("HITSEEDS " + cid + " " + " ".join(str(s) for s in chosen))


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Project B Step-2A synthetic substrate sweep")
    ap.add_argument("--mode", required=True,
                    choices=["selftest", "prescan", "run", "aggregate"])
    ap.add_argument("--world", choices=["recoverable", "harmful"])
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--nseeds", type=int, default=51)
    ap.add_argument("--need", type=int, default=3)
    ap.add_argument("--leakage_nperm", type=int, default=20)
    ap.add_argument("--out", default="/tmp/project_b_step2a_sweep")
    ap.add_argument("--dir", default="/tmp/project_b_step2a_sweep")
    args = ap.parse_args()

    if args.mode == "selftest":
        selftest()
    elif args.mode == "prescan":
        if not args.config:
            raise SystemExit("--config required for prescan")
        prescan(args.config, args.nseeds, args.need)
    elif args.mode == "run":
        if not (args.world and args.config and args.seed is not None):
            raise SystemExit("--world --config --seed required for run")
        run_one(args.world, args.config, args.seed, args.out, leakage_nperm=args.leakage_nperm)
    elif args.mode == "aggregate":
        out_csv = args.out if args.out.endswith(".csv") else os.path.join(args.dir, "summary.csv")
        aggregate(args.dir, out_csv)


if __name__ == "__main__":
    main()
