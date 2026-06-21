"""Stage-B2a step 3: metadata-conditioned diagonal operator selection (review (a')). Per (seed,
target site) draws B2a metadata episodes (observable metadata -> latent acquisition transform +
prior), applies each to the target site, and evaluates SIX comparators. The router sees ONLY the
metadata + the source-calibrated evidence; target labels are used solely for final scoring.

Action set (diagonal-only; SPD/rotation frozen): identity / pooled_empirical_diag /
canonical_fixed_prior_class_conditional_diag (= gen_oneshot_diag, FROZEN by canonical_freeze).

Comparators:
  identity            always identity
  always_pooled       always pooled_empirical_diag
  always_canonical    always the frozen canonical class-conditional
  n1_target_ranking   the FAILED A* N1 target-stat ranking (argmax Z>tau else identity)
  metadata_gated      a_meta = rule(metadata); deploy a_meta iff Z_{a_meta} > tau (frozen gate)  [METHOD]
  metadata_oracle     best operator by realised bAcc (upper bound on a perfect metadata map)

Gate: the frozen A* nested-null calibration (other-seed q50/q90 + empirical max-null tau), reused
unchanged. Reuses the B1a source bundles (no retraining).

  python -m h2cmi.run_b2a_grid --grid-seeds 0,1,2 --episodes-per-unit 12 \
      --bundle-root results/h2cmi/b1a_bundles_standard \
      --nested results/h2cmi/nested_null_standard.jsonl --out results/h2cmi/b2a_grid.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.data.paired_simulator import PairedEEGSimulator, difficulty_kwargs
from h2cmi.data.metadata_substrate import sample_episode, metadata_to_operator, CANONICAL_CC
from h2cmi.domains import compact_domain_labels
from h2cmi.train.trainer import reference_prior, H2Model
from h2cmi.eval.harness import _embed, _predict_transform
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME, Transform,
                                         reference_weighted_source_moments)
from h2cmi.tta import router_signals as rs
from h2cmi.run_shift_grid import build_cfgs
from h2cmi.grid_io import (SCHEMA_VERSION, append_row, require_clean_git, source_data_hash,
                           source_training_signature, source_code_signature, build_data_spec,
                           source_bundle_paths, load_source_bundle, stable_hash_int, sha256_file)
from h2cmi.analyze_b1a_astar import _nested_units, _calibration

CANONICAL_VARIANT = "gen_oneshot_diag"        # FROZEN by canonical_freeze.freeze.json
OP_TO_VARIANT = {"identity": "identity", "pooled_empirical_diag": "pooled_empirical_diag",
                 CANONICAL_CC: CANONICAL_VARIANT}
GATE_ACTIONS = ("pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag")
COMPARATORS = ("identity", "always_pooled", "always_canonical", "n1_target_ranking",
               "metadata_gated", "metadata_oracle")


def _bacc(model, U, T, prior, y):
    return float(balanced_accuracy_score(y, _predict_transform(model, U, T, prior).argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-seeds", default="0,1,2"); ap.add_argument("--grid-target-sites", default="all")
    ap.add_argument("--sites", type=int, default=5); ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=2); ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--classes", type=int, default=3); ap.add_argument("--chans", type=int, default=16)
    ap.add_argument("--times", type=int, default=128); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--difficulty", default="standard")
    ap.add_argument("--episodes-per-unit", type=int, default=12)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--bundle-root", required=True); ap.add_argument("--nested", required=True)
    ap.add_argument("--freeze-record", required=True, help="canonical_freeze.freeze.json (consistency gate)")
    ap.add_argument("--out", default="results/h2cmi/b2a_grid.jsonl")
    args = ap.parse_args()

    # launch gate: the canonical estimator MUST match the immutable freeze record
    freeze = json.load(open(args.freeze_record))
    if freeze.get("marker") != "CANONICAL_CC_ESTIMATOR_FROZEN" or freeze.get("selected") != CANONICAL_VARIANT:
        raise RuntimeError(f"freeze-record mismatch: selected={freeze.get('selected')} != {CANONICAL_VARIANT}")
    freeze_sha = sha256_file(args.freeze_record)
    out_dir = os.path.dirname(args.out) or "."
    commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root])
    code_sig = source_code_signature()
    seeds = [int(x) for x in args.grid_seeds.split(",") if x]
    sites = list(range(args.sites)) if args.grid_target_sites == "all" else [int(x) for x in args.grid_target_sites.split(",")]
    cfg, _ = build_cfgs(args)
    dkw = difficulty_kwargs(args.difficulty)
    data_spec = build_data_spec(simulator="PairedEEGSimulator", n_sites=args.sites,
                                subjects_per_site=args.subjects, sessions_per_subject=args.sessions,
                                trials_per_session=args.trials, n_classes=args.classes, n_chans=args.chans,
                                n_times=args.times, source_scenario="population_null",
                                train_seed_policy="train_seed=data_seed", difficulty=args.difficulty,
                                difficulty_spec=dkw)
    nested_units = _nested_units([json.loads(l) for l in open(args.nested) if l.strip()])
    nested_sha = sha256_file(args.nested)
    uni = np.full(args.classes, 1.0 / args.classes)
    if os.path.exists(args.out):
        os.remove(args.out)
    print(f"[b2a] commit={commit[:12]} nested_sha={nested_sha[:12]} canonical={CANONICAL_VARIANT} "
          f"seeds={seeds} sites={sites} eps={args.episodes_per_unit} -> {args.out}", flush=True)

    for seed in seeds:
        sim = PairedEEGSimulator(args.classes, args.chans, args.times, base_noise=dkw["base_noise"],
                                 subj_anatomy=dkw["subj_anatomy"], class_signal_scale=dkw["class_signal_scale"],
                                 data_seed=seed)
        other = set(s for s in {k[0] for k in nested_units} if s != seed) or {k[0] for k in nested_units}
        cal, tau, Z = _calibration(nested_units, other)          # FROZEN gate calibration (other-seed)
        for tsite in sites:
            full = sim.sample(args.sites, args.subjects, args.sessions, args.trials, target_site=tsite,
                              scenario="population_null")
            src = full.site != tsite
            Xs, ys = full.X[src], full.y[src]
            dag, dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
            src_dhash = source_data_hash(Xs, ys, dom)
            pi_star = reference_prior(ys, args.classes, "uniform")
            cfg.train.seed = seed
            tsig = source_training_signature(cfg, seed, tsite, "off", source_code_signature=code_sig, data_spec=data_spec)
            pt, jsf = source_bundle_paths(args.bundle_root, tsig, seed, tsite, "off")
            if not (os.path.exists(pt) and os.path.exists(jsf)):
                raise FileNotFoundError(f"missing B1a source bundle {pt}")
            model, _ = load_source_bundle(pt, jsf, build_model=lambda c=cfg: H2Model(c, pi_star).to(args.device),
                                          expected_training_signature=tsig, expected_source_data_hash=src_dhash,
                                          expected_source_code_signature=code_sig, expected_pi_star=pi_star)
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, args.classes, args.device)
            Us = _embed(model, Xs, args.device)
            pooled_ref = reference_weighted_source_moments(Us, ys, pi_star)
            for ei in range(args.episodes_per_unit):
                ep = sample_episode(np.random.default_rng((seed, tsite, ei)))
                tf = sim.sample(args.sites, args.subjects, args.sessions, args.trials, target_site=tsite, scenario=ep.scenario)
                tm = tf.site == tsite
                Xt, yt, subj = tf.X[tm], tf.y[tm], tf.subject[tm]
                U = _embed(model, Xt, args.device)
                Tid = Transform(U.shape[1], "diag_affine", device=args.device)
                bacc = {"identity": _bacc(model, U, Tid, uni, yt)}
                zscore = {}
                for a in GATE_ACTIONS:
                    tseed = stable_hash_int(seed, tsite, ei, a)
                    fit = tta.fit_variant(U, B1A_VARIANTS_BY_NAME[a], pooled_ref=pooled_ref, tta_seed=tseed)
                    bacc[a] = _bacc(model, U, fit.transform, uni, yt)
                    s = rs.loso_evidence_gain(tta, U, subj, B1A_VARIANTS_BY_NAME[a], pooled_ref=pooled_ref, seed=tseed)
                    zscore[a] = Z(a, s)
                a_meta = metadata_to_operator(ep.delta)
                meta_var = OP_TO_VARIANT[a_meta]
                gate_pass = (meta_var != "identity") and (zscore.get(meta_var, -9) > tau)
                # comparator selections
                sel = {}
                sel["identity"] = "identity"
                sel["always_pooled"] = "pooled_empirical_diag"
                sel["always_canonical"] = CANONICAL_VARIANT
                elig = {a: zscore[a] for a in GATE_ACTIONS if zscore[a] > tau}
                sel["n1_target_ranking"] = max(elig, key=elig.get) if elig else "identity"
                sel["metadata_gated"] = meta_var if gate_pass else "identity"
                sel["metadata_oracle"] = max(("identity",) + GATE_ACTIONS, key=lambda a: bacc[a])
                common = dict(schema_version=SCHEMA_VERSION, runner_commit_sha=commit, code_sha=commit,
                              source_code_signature=code_sig, nested_calibration_sha256=nested_sha,
                              freeze_record_sha256=freeze_sha,
                              canonical_variant=CANONICAL_VARIANT, frozen_tau=float(tau),
                              data_seed=seed, target_site=tsite, episode=ei,
                              geometry_compatibility=ep.delta.geometry_compatibility,
                              prevalence_risk=ep.delta.prevalence_risk, latent_stratum=ep.latent_stratum,
                              eff_geom=ep.meta["eff_geom"], geom_magnitude=ep.geom_magnitude,
                              prior_magnitude=ep.prior_magnitude, metadata_operator=a_meta,
                              metadata_gate_pass=bool(gate_pass), identity_bacc=bacc["identity"])
                for comp in COMPARATORS:
                    s_op = sel[comp]
                    append_row(args.out, dict(common, comparator=comp, selected_op=s_op,
                                              adapted=(s_op != "identity"),
                                              dbacc_full=float(bacc[s_op] - bacc["identity"]),
                                              selected_bacc=float(bacc[s_op])))
            print(f"  seed={seed} site={tsite} done", flush=True)
    print(f"[b2a] complete -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
