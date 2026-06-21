"""Stage-B2b source-only futility checkpoint (review (i')). BEFORE any seeds-3..5 target training,
test whether the FROZEN evidence score has route-conditioned power. Reuses the seeds-0..2 nested
LOSO bundles (source-only gate design is allowed on the frozen B2a seeds; NO target eval here).

Per route r in {pooled (DIAG x SAME -> pooled_empirical_diag), cc (DIAG x DIFFERENT -> gen_oneshot_
diag)} and bank in {null (net geometry 0), power (frozen B2a gain dist)}, draw metadata-route-
positive episodes, apply them to held-out SOURCE pseudo-target sites, and score the route's single
action by the frozen change-of-variable evidence gain. The route-conditioned gate uses an OTHER-SEED
empirical null CDF; threshold at route-null FPR <= 0.10; retention = fraction of POWER passing.

Pre-frozen decision (no re-tune -- score/alpha/route/coupler/threshold all fixed): if the cross-
fitted aggregate (pooled+cc) route retention at route-null FPR <= 0.10 is < 0.25 ->
B2B_SOURCE_POWER_FAIL and target evaluation is NOT launched.

  python -m h2cmi.run_b2b_source_power --grid-seeds 0,1,2 --device cuda \
      --bundle-root results/h2cmi/nested_bundles_standard --out results/h2cmi/b2b_source_power.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from h2cmi.data.paired_simulator import PairedEEGSimulator, difficulty_kwargs
from h2cmi.data.metadata_substrate import route_bank_episode, ROUTE_VARIANT
from h2cmi.domains import compact_domain_labels
from h2cmi.train.trainer import reference_prior, H2Model
from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME,
                                         reference_weighted_source_moments)
from h2cmi.tta import router_signals as rs
from h2cmi.run_shift_grid import build_cfgs
from h2cmi.grid_io import (require_clean_git, source_data_hash, source_training_signature,
                           source_code_signature, build_data_spec, source_bundle_paths,
                           load_source_bundle, stable_hash_int, sha256_file, append_row)

ROUTES = ("pooled", "cc")
ALPHA = 0.10                 # route-null FPR target
RETENTION_MIN = 0.25         # pre-frozen futility threshold


def _route_scores(rows, route, bank, seeds):
    return np.array([r["score"] for r in rows if r["route"] == route and r["bank"] == bank
                     and r["data_seed"] in seeds], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-seeds", default="0,1,2")
    ap.add_argument("--sites", type=int, default=5); ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=2); ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--classes", type=int, default=3); ap.add_argument("--chans", type=int, default=16)
    ap.add_argument("--times", type=int, default=128); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--difficulty", default="standard")
    ap.add_argument("--episodes", type=int, default=15)         # per (pseudo-site, route, bank)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--signature-device", default="",
                    help="device baked into the bundle SIGNATURE (default: --device). Set 'cuda' to "
                         "reuse cuda-trained bundles on a CPU node (evaluation only; no training here).")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--out", default="results/h2cmi/b2b_source_power.jsonl")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root])
    code_sig = source_code_signature()
    seeds = [int(x) for x in args.grid_seeds.split(",") if x]
    pairs = [f"{a}-{b}" for a, b in itertools.combinations(range(args.sites), 2)]
    cfg, _ = build_cfgs(args)
    cfg.train.device = args.signature_device or args.device   # tsig matches bundles trained on this
                                                              # device; execution still uses args.device
    dkw = difficulty_kwargs(args.difficulty)
    data_spec = build_data_spec(simulator="PairedEEGSimulator", n_sites=args.sites,
                                subjects_per_site=args.subjects, sessions_per_subject=args.sessions,
                                trials_per_session=args.trials, n_classes=args.classes, n_chans=args.chans,
                                n_times=args.times, source_scenario="population_null",
                                train_seed_policy="train_seed=data_seed", difficulty=args.difficulty,
                                difficulty_spec=dkw)
    if os.path.exists(args.out):
        os.remove(args.out)
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
            tsig = source_training_signature(cfg, seed, 100 + a * 10 + b, "off", source_code_signature=code_sig, data_spec=data_spec)
            pt, jsf = source_bundle_paths(args.bundle_root, tsig, seed, 100 + a * 10 + b, "off")
            if not (os.path.exists(pt) and os.path.exists(jsf)):
                raise FileNotFoundError(f"missing nested bundle {pt}")
            model, _ = load_source_bundle(pt, jsf, build_model=lambda c=cfg: H2Model(c, pi_star).to(args.device),
                                          expected_training_signature=tsig, expected_source_data_hash=src_dhash,
                                          expected_source_code_signature=code_sig, expected_pi_star=pi_star)
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, args.classes, args.device)
            Us = _embed(model, Xs, args.device)
            pooled_ref = reference_weighted_source_moments(Us, ys, pi_star)
            for ps in (a, b):
                for route in ROUTES:
                    var = ROUTE_VARIANT[route]
                    for bank in ("null", "power"):
                        for ei in range(args.episodes):
                            rng = np.random.default_rng((seed, stable_hash_int(pair, route, bank), ps, ei))
                            ep = route_bank_episode(rng, route, bank)
                            tf = sim.sample(args.sites, args.subjects, args.sessions, args.trials, target_site=ps, scenario=ep.scenario)
                            pm = tf.site == ps
                            U = _embed(model, tf.X[pm], args.device)
                            subj = tf.subject[pm]
                            s = rs.loso_evidence_gain(tta, U, subj, B1A_VARIANTS_BY_NAME[var], pooled_ref=pooled_ref,
                                                      seed=stable_hash_int(seed, pair, ps, route, bank, ei))
                            append_row(args.out, dict(data_seed=seed, pair=pair, pseudo_site=int(ps), route=route,
                                                      bank=bank, episode=ei, score=float(s),
                                                      geom_magnitude=ep.geom_magnitude, prior_magnitude=ep.prior_magnitude))
            print(f"  seed={seed} pair={pair} done", flush=True)

    rows = [json.loads(l) for l in open(args.out)]
    per_route = {}
    for route in ROUTES:
        fpr, ret = [], []
        for s in seeds:
            others = [x for x in seeds if x != s] or seeds
            null_other = _route_scores(rows, route, "null", set(others))
            if len(null_other) < 10:
                continue
            tau = float(np.quantile(null_other, 1 - ALPHA))      # threshold at route-null FPR=ALPHA (other-seed)
            null_s = _route_scores(rows, route, "null", {s})
            pow_s = _route_scores(rows, route, "power", {s})
            if len(null_s):
                fpr.append(float(np.mean(null_s >= tau)))
            if len(pow_s):
                ret.append(float(np.mean(pow_s >= tau)))
        # ROC ceiling (all-data, diagnostic): retention at threshold = (1-alpha)-quantile of all null
        null_all = _route_scores(rows, route, "null", set(seeds))
        pow_all = _route_scores(rows, route, "power", set(seeds))
        roc = float(np.mean(pow_all >= np.quantile(null_all, 1 - ALPHA))) if len(null_all) and len(pow_all) else float("nan")
        per_route[route] = dict(crossfit_route_null_fpr=float(np.mean(fpr)) if fpr else float("nan"),
                                crossfit_route_retention=float(np.mean(ret)) if ret else float("nan"),
                                roc_ceiling_retention_at_fpr10=roc, n_null=len(null_all), n_power=len(pow_all))
    agg = float(np.mean([per_route[r]["crossfit_route_retention"] for r in ROUTES
                         if per_route[r]["crossfit_route_retention"] == per_route[r]["crossfit_route_retention"]]))
    decision = "PROCEED_TO_TARGET_EVAL" if agg >= RETENTION_MIN else "B2B_SOURCE_POWER_FAIL"
    record = dict(marker="B2B_SOURCE_POWER_CHECKPOINT", alpha=ALPHA, retention_min=RETENTION_MIN,
                  per_route=per_route, aggregate_crossfit_retention=agg, decision=decision,
                  code_sha=commit, source_code_signature=code_sig, rows_sha256=sha256_file(args.out), n_rows=len(rows))
    rec_path = args.out.replace(".jsonl", ".checkpoint.json")
    json.dump(record, open(rec_path, "w"), indent=2)
    print("SOURCE-POWER CHECKPOINT:", json.dumps(record, indent=2), flush=True)
    print(f"-> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
