"""Stage-B2a step 2: freeze the canonical class-conditional ESTIMATOR (gen_oneshot vs gen_iterative)
on SOURCE-ONLY data, before any B2a target evaluation. Reuses the existing nested LOSO bundles (no
retraining): for each held-out SOURCE pseudo-target site, draw synthetic geometry+prior episodes
from the B2a coupler, apply them to that site, and score both estimators with the (known) SOURCE
labels. No target data, no target labels.

Pre-registered decision (fixed BEFORE running):
  metric              = bacc_uniform (full-refit, uniform decision prior)
  in-seed aggregation = mean over (pair, pseudo-site, episode)
  per-seed delta      = mean(bacc[gen_oneshot] - bacc[gen_iterative])
  overall delta       = mean over seeds
  substantive_threshold = 0.01
  rule: |overall delta| < threshold -> gen_oneshot (parsimony tie-break); else the better performer

Emits a freeze-record JSON with the choice + provenance (bundle hashes, code SHA, row SHA-256).

  python -m h2cmi.run_canonical_freeze --grid-seeds 0,1,2 \
      --bundle-root results/h2cmi/nested_bundles_standard --out results/h2cmi/canonical_freeze.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.data.paired_simulator import PairedEEGSimulator, difficulty_kwargs
from h2cmi.data.metadata_substrate import sample_episode
from h2cmi.domains import compact_domain_labels
from h2cmi.train.trainer import reference_prior, H2Model
from h2cmi.eval.harness import _embed, _predict_transform
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME,
                                         reference_weighted_source_moments)
from h2cmi.run_shift_grid import build_cfgs
from h2cmi.grid_io import (require_clean_git, source_data_hash, source_training_signature,
                           source_code_signature, build_data_spec, source_bundle_paths,
                           load_source_bundle, hash_state, stable_hash_int, sha256_file, append_row)

SUBSTANTIVE_THRESHOLD = 0.01           # pre-registered, fixed before running
ESTIMATORS = ("gen_oneshot_diag", "gen_iterative_diag")


def freeze_decision(per_seed_delta: dict, threshold: float = SUBSTANTIVE_THRESHOLD) -> tuple[str, bool, float]:
    """Pre-registered rule: overall delta = mean over seeds of (oneshot - iterative). |overall| <
    threshold -> gen_oneshot (parsimony tie-break); else the better performer. Returns
    (selected, tie_break_triggered, overall_delta)."""
    overall = float(np.mean(list(per_seed_delta.values())))
    tie = abs(overall) < threshold
    selected = "gen_oneshot_diag" if (tie or overall > 0) else "gen_iterative_diag"
    return selected, bool(tie), overall


def _bacc(model, U, transform, prior, y):
    return float(balanced_accuracy_score(y, _predict_transform(model, U, transform, prior).argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-seeds", default="0,1,2")
    ap.add_argument("--sites", type=int, default=5); ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=2); ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--classes", type=int, default=3); ap.add_argument("--chans", type=int, default=16)
    ap.add_argument("--times", type=int, default=128); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--difficulty", default="standard")
    ap.add_argument("--episodes-per-site", type=int, default=8)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--out", default="results/h2cmi/canonical_freeze.jsonl")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root])
    code_sig = source_code_signature()
    seeds = [int(x) for x in args.grid_seeds.split(",") if x]
    pairs = [f"{a}-{b}" for a, b in itertools.combinations(range(args.sites), 2)]
    cfg, _ = build_cfgs(args)
    dkw = difficulty_kwargs(args.difficulty)
    data_spec = build_data_spec(simulator="PairedEEGSimulator", n_sites=args.sites,
                                subjects_per_site=args.subjects, sessions_per_subject=args.sessions,
                                trials_per_session=args.trials, n_classes=args.classes, n_chans=args.chans,
                                n_times=args.times, source_scenario="population_null",
                                train_seed_policy="train_seed=data_seed", difficulty=args.difficulty,
                                difficulty_spec=dkw)
    uni = np.full(args.classes, 1.0 / args.classes)
    if os.path.exists(args.out):
        os.remove(args.out)
    bundle_hashes = []
    for seed in seeds:
        sim = PairedEEGSimulator(args.classes, args.chans, args.times, base_noise=dkw["base_noise"],
                                 subj_anatomy=dkw["subj_anatomy"], class_signal_scale=dkw["class_signal_scale"],
                                 data_seed=seed)
        full = sim.sample(args.sites, args.subjects, args.sessions, args.trials, target_site=0, scenario="population_null")
        for pair in pairs:
            a, b = (int(x) for x in pair.split("-"))
            src = ~np.isin(full.site, [a, b])
            Xs, ys = full.X[src], full.y[src]
            dag, dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
            src_dhash = source_data_hash(Xs, ys, dom)
            pi_star = reference_prior(ys, args.classes, "uniform")
            cfg.train.seed = seed
            tsig = source_training_signature(cfg, seed, 100 + a * 10 + b, "off",
                                             source_code_signature=code_sig, data_spec=data_spec)
            pt, jsf = source_bundle_paths(args.bundle_root, tsig, seed, 100 + a * 10 + b, "off")
            if not (os.path.exists(pt) and os.path.exists(jsf)):
                raise FileNotFoundError(f"missing nested bundle {pt}")
            model, _ = load_source_bundle(pt, jsf, build_model=lambda c=cfg: H2Model(c, pi_star).to(args.device),
                                          expected_training_signature=tsig, expected_source_data_hash=src_dhash,
                                          expected_source_code_signature=code_sig, expected_pi_star=pi_star)
            bundle_hashes.append(hash_state(model))
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, args.classes, args.device)
            Us = _embed(model, Xs, args.device)
            pooled_ref = reference_weighted_source_moments(Us, ys, pi_star)
            for ps in (a, b):                                  # held-out SOURCE pseudo-target site
                for ei in range(args.episodes_per_site):
                    rng = np.random.default_rng((seed, stable_hash_int(pair), ps, ei))
                    ep = sample_episode(rng)
                    shifted = sim.sample(args.sites, args.subjects, args.sessions, args.trials,
                                         target_site=ps, scenario=ep.scenario)
                    pm = shifted.site == ps
                    Xp, yp = shifted.X[pm], shifted.y[pm]      # SOURCE labels (known)
                    U = _embed(model, Xp, args.device)
                    for est in ESTIMATORS:
                        tseed = stable_hash_int(seed, pair, ps, ei, est)
                        fit = tta.fit_variant(U, B1A_VARIANTS_BY_NAME[est], pooled_ref=pooled_ref, tta_seed=tseed)
                        append_row(args.out, dict(data_seed=seed, pair=pair, pseudo_site=int(ps), episode=ei,
                                                  estimator=est, latent_stratum=ep.latent_stratum,
                                                  bacc_uniform=_bacc(model, U, fit.transform, uni, yp)))
            print(f"  seed={seed} pair={pair} done", flush=True)

    # pre-registered decision
    rows = [json.loads(l) for l in open(args.out)]
    by_seed = {}
    for s in seeds:
        o = np.mean([r["bacc_uniform"] for r in rows if r["data_seed"] == s and r["estimator"] == "gen_oneshot_diag"])
        i = np.mean([r["bacc_uniform"] for r in rows if r["data_seed"] == s and r["estimator"] == "gen_iterative_diag"])
        by_seed[s] = float(o - i)
    selected, tie, overall = freeze_decision(by_seed, SUBSTANTIVE_THRESHOLD)
    record = dict(
        marker="CANONICAL_CC_ESTIMATOR_FROZEN", selection_data="source-only nested pseudo-target episodes",
        target_data_used=False, target_labels_used=False, selected=selected,
        substantive_threshold=SUBSTANTIVE_THRESHOLD, tie_break_triggered=bool(tie),
        metric="bacc_uniform_full_refit", in_seed_aggregation="mean_over_pair_pseudosite_episode",
        per_seed_delta_oneshot_minus_iterative=by_seed, overall_delta=overall,
        code_sha=commit, source_code_signature=code_sig,
        input_bundle_checkpoint_hashes_sha256=hashlib.sha256(
            "".join(sorted(bundle_hashes)).encode()).hexdigest(),
        rows_sha256=sha256_file(args.out), n_rows=len(rows), n_episodes_per_site=args.episodes_per_site)
    rec_path = args.out.replace(".jsonl", ".freeze.json")
    with open(rec_path, "w") as f:
        json.dump(record, f, indent=2)
    print("FREEZE RECORD:", json.dumps(record, indent=2), flush=True)
    print(f"-> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
