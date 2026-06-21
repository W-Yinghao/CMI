"""Stage-B1b A*: the proper NESTED canonical-site null (review). For each (data_seed, unordered
excluded site-pair {a,b}) it trains a source model on the OTHER three canonical sites and scores
each excluded site (a and b) as a NO-SHIFT held-out pseudo-target -- a clean null the source model
never saw. Per (seed, pair) trains ONE model (cached by the unordered pair) and emits TWO null
units. With 5 sites that is C(5,2)=10 pairs -> 10 models -> 20 null units per seed (3 seeds = 30
models / 60 null units per difficulty).

It records, per (seed, pseudo_target_site, action): the raw evidence gain and the B reproducibility
signals (direction cosine / effect-to-noise / prediction stability), plus an audit of the actual
adaptation outcome. This feeds analyze_b1a_astar.py, which sets the empirical max-null eligibility
threshold (other-seed) and runs N1/N2 against the EXISTING real-target signals. No C signals, no
SPD/rotation/CMI. Shardable by --shard-pairs so each GPU trains only a few models (never the whole
LOSO sweep on one device).

  python -m h2cmi.run_nested_null --difficulty standard --grid-seeds 0,1,2 --shard-pairs 0-1,0-2 \
      --bundle-root results/h2cmi/nested_bundles_standard --out results/h2cmi/nested_null_standard_s.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.domains import compact_domain_labels
from h2cmi.data.paired_simulator import PairedEEGSimulator, difficulty_kwargs
from h2cmi.train.trainer import train_h2, reference_prior, H2Model
from h2cmi.eval.harness import _embed, _predict_transform, _predict_generative
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME, Transform,
                                         reference_weighted_source_moments)
from h2cmi.tta import router_signals as rs
from h2cmi.run_shift_grid import build_cfgs
from h2cmi.grid_io import (SCHEMA_VERSION, load_done_keys, append_row, hash_state, require_clean_git,
                           source_data_hash, source_training_signature, source_code_signature,
                           build_data_spec, build_manifest, validate_or_create_manifest,
                           source_bundle_paths, save_source_bundle, load_source_bundle, stable_hash_int)

ACTIONS = ("pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag")


def _all_pairs(n):
    return [f"{a}-{b}" for a, b in itertools.combinations(range(n), 2)]


def _bacc(model, U, transform, prior, yt):
    return float(balanced_accuracy_score(yt, _predict_transform(model, U, transform, prior).argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="standard", choices=["standard", "hard"])
    ap.add_argument("--grid-seeds", default="0,1,2")
    ap.add_argument("--shard-pairs", default="all", help="'all' or e.g. '0-1,2-3' (canonical a<b)")
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--chans", type=int, default=16)
    ap.add_argument("--times", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="results/h2cmi/nested_null.jsonl")
    ap.add_argument("--bundle-root", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    runner_commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root])
    code_sig = source_code_signature()
    seeds = [int(x) for x in args.grid_seeds.split(",") if x != ""]
    all_pairs = _all_pairs(args.sites)
    shard_pairs = all_pairs if args.shard_pairs == "all" else [p for p in args.shard_pairs.split(",") if p]
    cfg, _ = build_cfgs(args)
    dkw = difficulty_kwargs(args.difficulty)
    data_spec = build_data_spec(simulator="PairedEEGSimulator", n_sites=args.sites,
                                subjects_per_site=args.subjects, sessions_per_subject=args.sessions,
                                trials_per_session=args.trials, n_classes=args.classes, n_chans=args.chans,
                                n_times=args.times, source_scenario="population_null",
                                train_seed_policy="train_seed=data_seed", difficulty=args.difficulty,
                                difficulty_spec=dkw)
    # experiment = the GLOBAL nested grid (all pairs x both pseudo-sites x actions); shard = this subset
    manifest = build_manifest(cfg, global_seeds=seeds, global_sites=list(range(args.sites)),
                              scenarios=["nested_null"], items=sorted(ACTIONS), item_field="action",
                              cmi_arms=["off"], shard_spec={"seeds": sorted(seeds), "pairs": sorted(shard_pairs)},
                              cli={"nested": True, "difficulty": args.difficulty}, data_spec=data_spec)
    validate_or_create_manifest(args.out, manifest)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                import json as _j
                r = _j.loads(line)
                done.add((r["data_seed"], r["excluded_site_pair"], r["pseudo_target_site"], r["action"]))
    print(f"[nested-null] diff={args.difficulty} pairs={shard_pairs} seeds={seeds} -> {args.out} ({len(done)} done)", flush=True)

    uni = np.full(args.classes, 1.0 / args.classes)
    log_uni = torch.log(torch.tensor(uni, dtype=torch.float32, device=args.device).clamp_min(1e-8))
    for seed in seeds:
        sim = PairedEEGSimulator(args.classes, args.chans, args.times, base_noise=dkw["base_noise"],
                                 subj_anatomy=dkw["subj_anatomy"], class_signal_scale=dkw["class_signal_scale"],
                                 data_seed=seed)
        full = sim.sample(args.sites, args.subjects, args.sessions, args.trials, target_site=0,
                          scenario="population_null")          # no shift anywhere (target_site irrelevant)
        for pair in shard_pairs:
            a, b = (int(x) for x in pair.split("-"))
            excl = {a, b}
            src = ~np.isin(full.site, list(excl))
            Xs, ys = full.X[src], full.y[src]
            src_dag, src_dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
            src_dhash = source_data_hash(Xs, ys, src_dom)
            pi_star = reference_prior(ys, args.classes, "uniform")
            cfg.train.seed = seed
            pair_code = 100 + a * 10 + b                       # unique tsig key per UNORDERED pair
            tsig = source_training_signature(cfg, seed, pair_code, "off", source_code_signature=code_sig,
                                             data_spec=data_spec)
            pt, jsf = source_bundle_paths(args.bundle_root, tsig, seed, pair_code, "off")
            if os.path.exists(pt) and os.path.exists(jsf):
                model, _ = load_source_bundle(pt, jsf, build_model=lambda c=cfg: H2Model(c, pi_star).to(args.device),
                                              expected_training_signature=tsig, expected_source_data_hash=src_dhash,
                                              expected_source_code_signature=code_sig, expected_pi_star=pi_star)
            else:
                model, _, _, hist = train_h2(Xs, ys, src_dom, src_dag, cfg, align_factor="site")
                save_source_bundle(pt, jsf, model, training_signature=tsig, source_data_hash=src_dhash,
                                   source_code_signature=code_sig, pi_star=pi_star, commit_sha=runner_commit, history=hist)
            ckpt = hash_state(model)
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, args.classes, args.device)
            Us = _embed(model, Xs, args.device)
            pooled_ref = reference_weighted_source_moments(Us, ys, pi_star)
            sd_S = Us.std(0, unbiased=False).cpu().numpy()
            for ps in (a, b):                                  # each excluded site = a clean null pseudo-target
                pm = full.site == ps
                Xp, yp = full.X[pm], full.y[pm]
                subj = full.subject[pm]
                U = _embed(model, Xp, args.device)
                Tid = Transform(U.shape[1], "diag_affine", device=args.device)
                id_full = _bacc(model, U, Tid, uni, yp)
                id_oof = tta.grouped_heldout(U, subj, B1A_VARIANTS_BY_NAME["identity"], true_labels=yp,
                                             decision_prior=uni, seed_parts=(seed, ps, "nested", args.difficulty))
                for an in ACTIONS:
                    key = (seed, pair, ps, an)
                    if key in done:
                        continue
                    spec = B1A_VARIANTS_BY_NAME[an]
                    tseed = stable_hash_int(seed, pair, ps, args.difficulty, an)
                    fit = tta.fit_variant(U, spec, pooled_ref=pooled_ref, tta_seed=tseed)
                    s = rs.loso_evidence_gain(tta, U, subj, spec, pooled_ref=pooled_ref, seed=tseed)
                    bsig = rs.replicate_stability(tta, U, subj, spec, sd_S=sd_S, pooled_ref=pooled_ref, decision_prior=uni)
                    g = tta.grouped_heldout(U, subj, spec, true_labels=yp, pooled_ref=pooled_ref,
                                            decision_prior=uni, seed_parts=(seed, ps, "nested", args.difficulty))
                    row = dict(schema_version=SCHEMA_VERSION, runner_commit_sha=runner_commit,
                               config_signature=manifest["config_signature"],
                               experiment_signature=manifest["experiment_signature"], difficulty=args.difficulty,
                               data_seed=seed, excluded_site_pair=pair, pseudo_target_site=int(ps), action=an,
                               nested_source_checkpoint_hash=ckpt, source_data_hash=src_dhash,
                               raw_evidence_score=s,
                               transform_direction_cosine=bsig["transform_direction_cosine"],
                               transform_effect_to_noise_ratio=bsig["transform_effect_to_noise_ratio"],
                               crossfit_prediction_disagreement=bsig["crossfit_prediction_disagreement"],
                               full_refit_delta_bacc=_bacc(model, U, fit.transform, uni, yp) - id_full,
                               oof_delta_bacc=(g["grouped_oof_bacc"] - id_oof["grouped_oof_bacc"]
                                               if g["grouped_oof_bacc"] == g["grouped_oof_bacc"] else float("nan")))
                    append_row(args.out, row); done.add(key)
            print(f"  seed={seed} pair={pair} ckpt={ckpt[:12]} done", flush=True)
    print(f"[nested-null] complete: {len(done)} in {args.out}", flush=True)


if __name__ == "__main__":
    main()
