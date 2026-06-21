"""Stage-B1b-2: compute the A/B/C router signals per (seed, site, scenario, difficulty, action),
REUSING the frozen B1a source bundles (no retraining). Output feeds analyze_b1a_routers.py, which
composes R0..R3 and the frozen pass criteria. The method is unchanged -- this only adds diagnostics.

Per source unit (seed, site) it also computes SOURCE-NULL distributions (no-shift pseudo-targets
drawn from the source subjects) and stores per-action calibration quantiles for the conjunction:
source-q05 of the reproducibility signals, source-q95 of the anchor-flip rate, and source-q95 of
delta_snd. The empirical-null conformal p uses a pooled source-subject null (m~80) -- a prototype
(training-included) null; a final version needs a nested retrain to exclude the held-out source
domain.

  python -m h2cmi.run_b1a_router_signals --difficulty standard --grid-seeds 0,1,2 \
      --bundle-root results/h2cmi/b1a_bundles_standard --out results/h2cmi/b1a_router_signals_standard.jsonl
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from h2cmi.domains import compact_domain_labels
from h2cmi.data.paired_simulator import PairedEEGSimulator, PRESET_SCENARIOS, difficulty_kwargs
from h2cmi.train.trainer import reference_prior, H2Model
from h2cmi.eval.harness import _embed
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME,
                                         reference_weighted_source_moments)
from h2cmi.tta import router_signals as rs
from h2cmi.run_shift_grid import build_cfgs
from h2cmi.grid_io import (SCHEMA_VERSION, load_done_keys, append_row, require_clean_git,
                           source_data_hash, source_training_signature, source_code_signature,
                           build_data_spec, build_manifest, validate_or_create_manifest,
                           source_bundle_paths, load_source_bundle, stable_hash_int)

ACTIONS = ("pooled_empirical_diag", "gen_oneshot_diag", "gen_iterative_diag")   # non-identity deployable
DEFAULT_STANDARD = "population_null,matched_domain_null,cov,prior,cov_prior,conditional_rotation,cov_conditional_rotation"
from sklearn.metrics import balanced_accuracy_score


def _parse_int_list(s, n):
    return list(range(n)) if s == "all" else [int(x) for x in s.split(",") if x != ""]


def _bacc(model, U, transform, prior, yt):
    from h2cmi.eval.harness import _predict_transform
    return float(balanced_accuracy_score(yt, _predict_transform(model, U, transform, prior).argmax(1)))


def _source_calibration(tta, model, Us, ys, src_subj, pooled_ref, sd_S, log_uni, *, n_draws=40, seed=0):
    """Per-action source-null distributions of the B/C signals (no-shift pseudo-targets) ->
    calibration quantiles. Returns {action: {q05_cosine,q05_predstab,q05_etn,q95_anchor,q95_dsnd}}."""
    uq = np.unique(src_subj)
    cal = {}
    for an in ACTIONS:
        spec = B1A_VARIANTS_BY_NAME[an]
        cos, pst, etn, anc, dsnd = [], [], [], [], []
        for d in range(n_draws):
            rng = np.random.default_rng((seed, d, hash(an) & 0xffff))
            grp = rng.choice(uq, size=min(3, len(uq)), replace=False)
            m = np.isin(src_subj, grp)
            if m.sum() < tta.cfg.min_target:
                continue
            sub = src_subj[m]
            b = rs.replicate_stability(tta, Us[m], sub, spec, sd_S=sd_S, pooled_ref=pooled_ref)
            cos.append(b["transform_direction_cosine"]); etn.append(b["transform_effect_to_noise_ratio"])
            pst.append(1.0 - b["crossfit_prediction_disagreement"])
            f = tta.fit_variant(Us[m], spec, pooled_ref=pooled_ref, tta_seed=int(d))
            cs = rs.class_structure(model, Us[m], f.transform, log_uni)
            dsnd.append(cs["delta_snd"])
            thr = rs.source_confidence_threshold(model, Us[m], log_uni)
            fl, _ = rs.anchor_flip_rate(model, Us[m], f.transform, log_uni, thr)
            anc.append(fl)
        def q(v, p):
            v = np.asarray([x for x in v if x == x], float)
            return float(np.quantile(v, p)) if len(v) else float("nan")
        cal[an] = dict(q05_cosine=q(cos, 0.05), q05_predstab=q(pst, 0.05), q05_etn=q(etn, 0.05),
                       q95_anchor=q(anc, 0.95), q95_dsnd=q(dsnd, 0.95))
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=DEFAULT_STANDARD)
    ap.add_argument("--difficulty", default="standard", choices=["standard", "hard"])
    ap.add_argument("--grid-seeds", default="0,1,2")
    ap.add_argument("--grid-target-sites", default="all")
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
    ap.add_argument("--null-draws", type=int, default=80)
    ap.add_argument("--cal-draws", type=int, default=40)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="results/h2cmi/b1a_router_signals.jsonl")
    ap.add_argument("--bundle-root", required=True, help="EXISTING B1a bundle root to reuse")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    runner_commit = require_clean_git(allow_dirty=args.allow_dirty, ignore_prefixes=[out_dir, args.bundle_root])
    code_sig = source_code_signature()
    scen_canon = sorted({PRESET_SCENARIOS[s].name for s in args.scenarios.split(",") if s})
    seeds = _parse_int_list(args.grid_seeds, args.sites)
    sites = _parse_int_list(args.grid_target_sites, args.sites)
    cfg, _ = build_cfgs(args)
    dkw = difficulty_kwargs(args.difficulty)
    data_spec = build_data_spec(simulator="PairedEEGSimulator", n_sites=args.sites,
                                subjects_per_site=args.subjects, sessions_per_subject=args.sessions,
                                trials_per_session=args.trials, n_classes=args.classes, n_chans=args.chans,
                                n_times=args.times, source_scenario="population_null",
                                train_seed_policy="train_seed=data_seed", difficulty=args.difficulty,
                                difficulty_spec=dkw)
    manifest = build_manifest(cfg, global_seeds=seeds, global_sites=sites, scenarios=scen_canon,
                              items=sorted(ACTIONS), item_field="action", cmi_arms=["off"],
                              shard_spec={"seeds": sorted(seeds), "sites": sorted(sites)},
                              cli={"router_signals": True, "difficulty": args.difficulty}, data_spec=data_spec)
    validate_or_create_manifest(args.out, manifest)
    done = load_done_keys(args.out, item_field="action", manifest=manifest)
    print(f"[router-signals] diff={args.difficulty} exp_sig={manifest['experiment_signature']} "
          f"scenarios={scen_canon} -> {args.out} ({len(done)} done)", flush=True)

    uni = np.full(args.classes, 1.0 / args.classes)
    log_uni = torch.log(torch.tensor(uni, dtype=torch.float32, device=args.device).clamp_min(1e-8))
    cmi = "off"
    for seed in seeds:
        for tsite in sites:
            sim = PairedEEGSimulator(args.classes, args.chans, args.times, base_noise=dkw["base_noise"],
                                     subj_anatomy=dkw["subj_anatomy"], class_signal_scale=dkw["class_signal_scale"],
                                     data_seed=seed)
            full = sim.sample(args.sites, args.subjects, args.sessions, args.trials, target_site=tsite,
                              scenario="population_null")
            src = full.site != tsite
            Xs, ys = full.X[src], full.y[src]
            src_dag, src_dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
            src_dhash = source_data_hash(Xs, ys, src_dom)
            pi_star = reference_prior(ys, args.classes, "uniform")
            cfg.train.seed = seed
            tsig = source_training_signature(cfg, seed, tsite, cmi, source_code_signature=code_sig, data_spec=data_spec)
            pt, js = source_bundle_paths(args.bundle_root, tsig, seed, tsite, cmi)
            if not (os.path.exists(pt) and os.path.exists(js)):
                raise FileNotFoundError(f"missing source bundle for seed={seed} site={tsite}: {pt}")
            model, _ = load_source_bundle(pt, js, build_model=lambda c=cfg: H2Model(c, pi_star).to(args.device),
                                          expected_training_signature=tsig, expected_source_data_hash=src_dhash,
                                          expected_source_code_signature=code_sig, expected_pi_star=pi_star)
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, args.classes, args.device)
            Us = _embed(model, Xs, args.device)
            pooled_ref = reference_weighted_source_moments(Us, ys, pi_star)
            sd_S = Us.std(0, unbiased=False).cpu().numpy()
            src_subj = full.subject[src]
            # source-null scores (A) per action + B/C calibration quantiles
            nulls = {a: rs.source_null_scores(tta, Us, src_subj, B1A_VARIANTS_BY_NAME[a], pooled_ref=pooled_ref,
                                              n_draws=args.null_draws, base_seed=seed) for a in ACTIONS}
            cal = _source_calibration(tta, model, Us, ys, src_subj, pooled_ref, sd_S, log_uni,
                                      n_draws=args.cal_draws, seed=seed)
            common = dict(schema_version=SCHEMA_VERSION, runner_commit_sha=runner_commit,
                          config_signature=manifest["config_signature"],
                          experiment_signature=manifest["experiment_signature"], difficulty=args.difficulty,
                          data_seed=seed, train_seed=seed, target_site=tsite, cmi=cmi,
                          source_data_hash=src_dhash, source_code_signature=code_sig,
                          n_source_null=int(min(len(v) for v in nulls.values())))
            for scen_c in scen_canon:
                tf = sim.sample(args.sites, args.subjects, args.sessions, args.trials, target_site=tsite, scenario=scen_c)
                tm = tf.site == tsite
                Xt, yt = tf.X[tm], tf.y[tm]
                subj = tf.subject[tm]
                U = _embed(model, Xt, args.device)
                from h2cmi.tta.class_conditional import Transform
                Tid = Transform(U.shape[1], "diag_affine", device=args.device)
                id_full = _bacc(model, U, Tid, uni, yt)
                id_oof = tta.grouped_heldout(U, subj, B1A_VARIANTS_BY_NAME["identity"], true_labels=yt,
                                             decision_prior=uni, seed_parts=(seed, tsite, scen_c, args.difficulty))
                thr = rs.source_confidence_threshold(model, Us, log_uni)
                for an in ACTIONS:
                    key = (seed, tsite, scen_c, an, cmi)
                    if key in done:
                        continue
                    spec = B1A_VARIANTS_BY_NAME[an]
                    tseed = stable_hash_int(seed, tsite, scen_c, args.difficulty, an)
                    fit = tta.fit_variant(U, spec, pooled_ref=pooled_ref, tta_seed=tseed)
                    s_t = rs.loso_evidence_gain(tta, U, subj, spec, pooled_ref=pooled_ref, seed=tseed)
                    p_a = rs.conformal_pvalue(s_t, nulls[an])
                    b = rs.replicate_stability(tta, U, subj, spec, sd_S=sd_S, pooled_ref=pooled_ref, decision_prior=uni)
                    cs = rs.class_structure(model, U, fit.transform, log_uni)
                    dga = rs.disc_gen_agreement(model, U, fit.transform, log_uni)
                    dga_id = rs.disc_gen_agreement(model, U, Tid, log_uni)
                    flip, n_anchor = rs.anchor_flip_rate(model, U, fit.transform, log_uni, thr)
                    g = tta.grouped_heldout(U, subj, spec, true_labels=yt, pooled_ref=pooled_ref,
                                            decision_prior=uni, seed_parts=(seed, tsite, scen_c, args.difficulty))
                    row = dict(common, scenario=scen_c, action=an,
                               evidence_target=s_t, null_pvalue=p_a, n_actions=len(ACTIONS),
                               delta_disc_gen_agreement=float(dga - dga_id),
                               anchor_flip_rate=flip, anchor_n=n_anchor,
                               bacc_uniform=_bacc(model, U, fit.transform, uni, yt),
                               grouped_oof_bacc=g["grouped_oof_bacc"],
                               identity_bacc_uniform=id_full, identity_grouped_oof_bacc=id_oof["grouped_oof_bacc"],
                               **{k: b[k] for k in b}, **{k: cs[k] for k in cs},
                               **{f"src_{k}": cal[an][k] for k in cal[an]})
                    append_row(args.out, row); done.add(key)
                print(f"  seed={seed} site={tsite} scen={scen_c} done", flush=True)
    print(f"[router-signals] complete: {len(done)} in {args.out}", flush=True)


if __name__ == "__main__":
    main()
