"""Stage-B0 action-decomposition grid (review §3, §4, §7-B0).

Decomposes the current joint EM into four ACTIONS and evaluates each under TWO decision
priors, separating the fit prior from the decision prior (review §3 confound):

  A0 identity        T=I,  pi_fit=pi_S
  A1 prior_only      T=I,  estimate pi_T          (cannot change uniform-decision bAcc)
  A2 geometry_only   fit T, pi_fit=pi_S
  A3 joint           fit T, estimate pi_T

For each action:
  * bacc_uniform_decision  -- balanced accuracy under the UNIFORM decision prior (primary,
    since bAcc is the uniform-prior risk; prior-only is identity here by construction);
  * accuracy / NLL / Brier / ECE under the TARGET (fit) decision prior -- where prior
    correction actually pays off.

Also records the multi-head strict picture (disc / gen / blend) so we can see whether CMI
moved the discriminative/blend task representation, not only the generative head (review §4),
and which posterior responsibilities should come from.

Reuses the paired simulator + one Source-A/B pair per (seed,site) (frozen). JSONL, atomic
append, resumable. NO gate / SSL / disentangle / alignment / online / --full / real-EEG.

Tiny integration:
  python -m h2cmi.run_action_grid --scenarios population_null,matched_domain_null,cov,prior,\
conditional_rotation --seeds 0 --target-sites all --sites 3 --subjects 2 --sessions 2 \
--trials 16 --epochs 2 --fast --out results/h2cmi/action_grid_smoke.jsonl
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from h2cmi.config import H2Config, core_config
from h2cmi.domains import compact_domain_labels
from h2cmi.data.paired_simulator import PairedEEGSimulator, PRESET_SCENARIOS
from h2cmi.train.trainer import train_h2, reference_prior, H2Model
from h2cmi.eval.metrics import classification_metrics, _ece
from h2cmi.eval.harness import _predict_transform, _embed
from h2cmi.tta.class_conditional import ClassConditionalTTA, Transform
from h2cmi.run_shift_grid import build_cfgs
from h2cmi.grid_io import (SCHEMA_VERSION, load_done_keys, append_row, hash_state,
                           require_clean_git, source_data_hash, source_training_signature,
                           source_code_signature, build_data_spec, build_manifest,
                           validate_or_create_manifest, bundle_expected_keys,
                           source_bundle_paths, save_source_bundle, load_source_bundle)

ACTIONS = ("identity", "prior_only", "geometry_only", "joint")
CMI_ARMS = ("off", "on")


def _parse_int_list(s, n_default):
    return list(range(n_default)) if s == "all" else [int(x) for x in s.split(",") if x != ""]


@torch.no_grad()
def _multihead_strict(model, U, y, device):
    """Strict (no-adaptation) disc/gen/blend predictions under the UNIFORM decision prior."""
    uni = torch.log(torch.full((model.cfg.n_classes,), 1.0 / model.cfg.n_classes, device=device))
    disc = F.softmax(model.head.disc_logits(U), dim=1).cpu().numpy()
    gen = model.head.density.class_posterior(U, uni).cpu().numpy()
    blend = 0.5 * (disc + gen)
    return dict(
        strict_disc_bacc=float(balanced_accuracy_score(y, disc.argmax(1))),
        strict_gen_bacc=float(balanced_accuracy_score(y, gen.argmax(1))),
        strict_blend_bacc=float(balanced_accuracy_score(y, blend.argmax(1))),
        disc_ece=float(_ece(disc, y)), gen_ece=float(_ece(gen, y)), blend_ece=float(_ece(blend, y)),
        disc_gen_disagreement=float((disc.argmax(1) != gen.argmax(1)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="population_null,matched_domain_null,cov,prior,conditional_rotation")
    # GLOBAL grid (defines the experiment identity, shared by every shard)
    ap.add_argument("--grid-seeds", default="0")
    ap.add_argument("--grid-target-sites", default="all")
    # THIS shard's subset (default: the whole grid -> a single non-sharded run)
    ap.add_argument("--shard-seeds", default="")
    ap.add_argument("--shard-target-sites", default="")
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
    ap.add_argument("--allow-dirty", action="store_true", help="dev only: skip the clean-git guard")
    ap.add_argument("--out", default="results/h2cmi/action_grid.jsonl")
    ap.add_argument("--bundle-root", default="", help="provenance-keyed source cache (default: <out>.bundles)")
    args = ap.parse_args()

    bundle_root = args.bundle_root or (args.out + ".bundles")
    out_dir = os.path.dirname(args.out) or "."
    # the run writes untracked result artifacts under its out dir + bundle root; exclude those
    # from the clean-git guard, but still reject any other dirty/untracked file (e.g. code).
    runner_commit = require_clean_git(allow_dirty=args.allow_dirty,
                                      ignore_prefixes=[out_dir, bundle_root])
    code_sig = source_code_signature()
    scenarios = [s for s in args.scenarios.split(",") if s]
    for s in scenarios:
        if s not in PRESET_SCENARIOS:
            raise ValueError(f"unknown scenario {s}; have {sorted(PRESET_SCENARIOS)}")
    scen_canon = sorted({PRESET_SCENARIOS[s].name for s in scenarios})
    global_seeds = _parse_int_list(args.grid_seeds, args.sites)
    global_sites = _parse_int_list(args.grid_target_sites, args.sites)
    shard_seeds = _parse_int_list(args.shard_seeds, args.sites) if args.shard_seeds else global_seeds
    shard_sites = _parse_int_list(args.shard_target_sites, args.sites) if args.shard_target_sites else global_sites
    if not set(shard_seeds) <= set(global_seeds) or not set(shard_sites) <= set(global_sites):
        raise ValueError("shard seeds/sites must be subsets of the global grid")
    cfg_off, cfg_on = build_cfgs(args)

    data_spec = build_data_spec(simulator="PairedEEGSimulator", n_sites=args.sites,
                                subjects_per_site=args.subjects, sessions_per_subject=args.sessions,
                                trials_per_session=args.trials, n_classes=args.classes,
                                n_chans=args.chans, n_times=args.times,
                                source_scenario="population_null",
                                train_seed_policy="train_seed=data_seed", difficulty="standard")
    cli = {k: getattr(args, k) for k in ("scenarios", "grid_seeds", "grid_target_sites",
            "shard_seeds", "shard_target_sites", "sites", "subjects", "sessions", "trials",
            "classes", "chans", "times", "epochs", "fast")}
    manifest = build_manifest(cfg_on, global_seeds=global_seeds, global_sites=global_sites,
                              scenarios=scen_canon, items=sorted(ACTIONS), item_field="action",
                              cmi_arms=sorted(CMI_ARMS),
                              shard_spec={"seeds": sorted(shard_seeds), "sites": sorted(shard_sites)},
                              cli=cli, data_spec=data_spec)
    validate_or_create_manifest(args.out, manifest)
    done = load_done_keys(args.out, item_field="action", manifest=manifest)
    print(f"[action-grid] commit={runner_commit[:12]} exp_sig={manifest['experiment_signature']} "
          f"shard={manifest['shard_spec']} scenarios={scen_canon} -> {args.out} ({len(done)} done)",
          flush=True)

    uni = np.full(args.classes, 1.0 / args.classes)
    for seed in shard_seeds:
        for tsite in shard_sites:
            sim = PairedEEGSimulator(args.classes, args.chans, args.times, data_seed=seed)
            full = sim.sample(args.sites, args.subjects, args.sessions, args.trials,
                              target_site=tsite, scenario="population_null")
            src = full.site != tsite
            Xs, ys = full.X[src], full.y[src]
            src_dag, src_dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
            src_dhash = source_data_hash(Xs, ys, src_dom)        # X + y + domain levels + DAG
            pi_star = reference_prior(ys, args.classes, "uniform")
            for cfg, cmi in ((cfg_off, "off"), (cfg_on, "on")):
                cfg.train.seed = seed
                # compute-resume: if every result key for this source bundle is done, skip
                # training entirely (do not load data/build model/train).
                if bundle_expected_keys(seed, tsite, cmi, scen_canon, ACTIONS) <= done:
                    print(f"  seed={seed} site={tsite} cmi={cmi} complete -> skip train", flush=True)
                    continue
                tsig = source_training_signature(cfg, seed, tsite, cmi,            # provenance key
                                                 source_code_signature=code_sig, data_spec=data_spec)
                pt_path, json_path = source_bundle_paths(bundle_root, tsig, seed, tsite, cmi)
                if os.path.exists(pt_path) and os.path.exists(json_path):
                    model, bmeta = load_source_bundle(
                        pt_path, json_path, build_model=lambda c=cfg: H2Model(c, pi_star).to(args.device),
                        expected_training_signature=tsig, expected_source_data_hash=src_dhash,
                        expected_source_code_signature=code_sig, expected_pi_star=pi_star)
                    src_commit = bmeta.get("source_training_commit_sha")
                else:
                    model, _, _, hist = train_h2(Xs, ys, src_dom, src_dag, cfg, align_factor="site")
                    save_source_bundle(pt_path, json_path, model, training_signature=tsig,
                                       source_data_hash=src_dhash, source_code_signature=code_sig,
                                       pi_star=pi_star, commit_sha=runner_commit, history=hist)
                    src_commit = runner_commit
                ckpt = hash_state(model)
                common = dict(schema_version=SCHEMA_VERSION, runner_commit_sha=runner_commit,
                              source_training_commit_sha=src_commit, source_training_signature=tsig,
                              source_code_signature=code_sig,
                              config_signature=manifest["config_signature"],
                              experiment_signature=manifest["experiment_signature"],
                              data_seed=seed, train_seed=seed, tta_seed=seed, target_site=tsite,
                              cmi=cmi, source_checkpoint_hash=ckpt, source_data_hash=src_dhash)
                for scen_c in scen_canon:                     # canonical, deduped scenario names
                    tf = sim.sample(args.sites, args.subjects, args.sessions, args.trials,
                                    target_site=tsite, scenario=scen_c)
                    tm = tf.site == tsite
                    Xt, yt = tf.X[tm], tf.y[tm]
                    U = _embed(model, Xt, args.device)
                    true_prior = np.bincount(yt, minlength=args.classes) / len(yt)
                    strict = _multihead_strict(model, U, yt, args.device)
                    tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, args.classes, args.device)
                    for action in ACTIONS:
                        key = (seed, tsite, scen_c, action, cmi)
                        if key in done:
                            continue
                        res = tta.fit_action(U, action)
                        # primary: balanced acc under UNIFORM decision prior
                        p_uni = _predict_transform(model, U, res.transform, uni)
                        bacc_uni = balanced_accuracy_score(yt, p_uni.argmax(1))
                        # calibrated: under the TARGET (fit) decision prior
                        p_tgt = _predict_transform(model, U, res.transform, res.pi_T)
                        m = classification_metrics(p_tgt, yt)
                        acc_tgt = float((p_tgt.argmax(1) == yt).mean())
                        row = dict(common, scenario=scen_c, action=action,
                                   target_size=int(len(yt)), decision_prior="uniform",
                                   bacc_uniform_decision=float(bacc_uni),
                                   accuracy_target_prior=acc_tgt,
                                   nll_target_prior=m["nll"], brier_target_prior=m["brier"],
                                   ece_target_prior=m["ece"],
                                   transform_norm=res.diagnostics.get("transform_norm", 0.0),
                                   held_out_evidence_gain=(tta._crossfit_evidence_gain(U)
                                                           if action == "joint" else float("nan")),
                                   fit_prior=list(np.asarray(res.pi_T)), true_prior=list(true_prior),
                                   prior_l1_error=float(np.abs(np.asarray(res.pi_T) - true_prior).sum()),
                                   **strict)
                        append_row(args.out, row); done.add(key)
                print(f"  seed={seed} site={tsite} cmi={cmi} ckpt={ckpt} done", flush=True)
    print(f"[action-grid] complete: {len(done)} units in {args.out}", flush=True)


if __name__ == "__main__":
    main()
