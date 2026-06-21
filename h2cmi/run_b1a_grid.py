"""Stage-B1a responsibility x transform-family grid (review Stage-B1a).

A NEW runner (the Stage-B0 ``run_action_grid`` is left untouched). It freezes the seven-variant
matrix that decomposes the current joint EM along three axes -- how soft class assignments are
obtained, what moves, and the transform family -- to localise WHY the joint underperforms
geometry-only:

  identity                no adaptation (baseline)
  pooled_diag             classless diagonal moment match           (is p(z|y) load-bearing?)
  gen_oneshot_diag        responsibility generated ONCE on identity, frozen   (Q0_U0)
  gen_iterative_diag      responsibility re-estimated each EM round            (Q0_U2)
  oracle_oneshot_diag     true-label responsibility, frozen          (responsibility ceiling)
  oracle_oneshot_lowrank  ditto, low-rank family               (diag-vs-lowrank, C_family)
  joint_iterative_diag    transform + prior M-step                   (the current joint)

Main experiment is CMI-OFF only (a CMI-on retention arm is added after a single candidate is
chosen). Two difficulty regimes share the runner: B1a-standard (the 7 paired scenarios) and
B1a-hard-null (matched_domain_null only, ``--difficulty hard`` -- a GLOBAL SNR drop, NOT a
target-only mechanism, so the identity-null stays an identity-null but at lower strict bAcc).

Every non-identity variant is scored by per-target-SUBJECT LOSO held-out evidence (fit on the
other subjects, judge the held-out one) -- never an in-sample / random-split number. Per-unit
TTA randomness is keyed by a stable hash of (seed, site, scenario, difficulty, variant, fold),
so results are invariant to the order variants/folds run in.

Reuses the paired simulator + one frozen source bundle per (seed,site); JSONL, atomic append,
provenance-bound + resumable via the v3 grid_io infra. NO gate / SSL / disentangle / alignment
/ online / --full / real-EEG. Optional ``--b0-ref`` asserts the standard-difficulty source
checkpoint is byte-identical to Stage B0 (confirming only the adaptation side changed).

  python -m h2cmi.run_b1a_grid --difficulty standard --grid-seeds 0,1,2 \
      --out results/h2cmi/b1a_standard.jsonl
  python -m h2cmi.run_b1a_grid --difficulty hard --scenarios matched_domain_null \
      --grid-seeds 0,1,2 --out results/h2cmi/b1a_hard_null.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.domains import compact_domain_labels
from h2cmi.data.paired_simulator import PairedEEGSimulator, PRESET_SCENARIOS, difficulty_kwargs
from h2cmi.train.trainer import train_h2, reference_prior, H2Model
from h2cmi.eval.harness import _predict_transform, _predict_generative, _embed
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS, B1A_VARIANTS_BY_NAME,
                                         reference_weighted_source_moments)
from h2cmi.run_shift_grid import build_cfgs
from h2cmi.grid_io import (SCHEMA_VERSION, load_done_keys, append_row, hash_state, require_clean_git,
                           source_data_hash, source_training_signature, source_code_signature,
                           build_data_spec, build_manifest, validate_or_create_manifest,
                           bundle_expected_keys, source_bundle_paths, save_source_bundle,
                           load_source_bundle, stable_hash_int, checkpoint_hash_for_scheme)

VARIANT_NAMES = tuple(v.name for v in B1A_VARIANTS)
CMI_ARMS = ("off",)                                   # main B1a is CMI-off only
DEFAULT_STANDARD = "population_null,matched_domain_null,cov,prior,cov_prior,conditional_rotation,cov_conditional_rotation"


def _parse_int_list(s, n_default):
    return list(range(n_default)) if s == "all" else [int(x) for x in s.split(",") if x != ""]


def _nll(p, y) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, None)).mean())


def load_b0_checkpoints(path: str) -> dict:
    """Map (data_seed, target_site, 'off') -> source_checkpoint_hash from a Stage-B0 JSONL.
    STRICT: aborts if the same (seed,site,off) carries two different hashes."""
    m = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("cmi") != "off":
                continue
            h = r.get("source_checkpoint_hash")
            if not h:
                continue
            k = (r["data_seed"], r["target_site"], "off")
            if k in m and m[k] != h:
                raise RuntimeError(f"--b0-ref: conflicting source_checkpoint_hash for {k}: "
                                   f"{m[k]} vs {h}")
            m[k] = h
    return m


def _resp_stats(r_np, yt, K) -> tuple[float, float, float]:
    """(balanced acc of argmax vs true, mean entropy, min class occupancy) for an [N,K] resp."""
    bacc = float(balanced_accuracy_score(yt, r_np.argmax(1)))
    ent = float(-(r_np * np.log(np.clip(r_np, 1e-8, None))).sum(1).mean())
    occ = float((r_np.sum(0) / max(1, len(yt))).min())
    return bacc, ent, occ


@torch.no_grad()
def variant_metrics(model, U, fit, yt, uni, p_id, bacc_id, nll_id) -> dict:
    """Transform + responsibility + identity-vs-adapted diagnostics for one VariantFit (in-sample;
    LOSO held-out numbers are computed separately). Records the responsibilities ACTUALLY used --
    r_initial (generated on identity), r_last_used (consumed by the final transform M-step), and
    r_final (post-fit posterior) -- so feedback vs responsibility-quality can be read off."""
    T, K = fit.transform, model.cfg.n_classes
    d = U.shape[1]
    A = T.matrix()
    I = torch.eye(d, device=A.device)
    svals = torch.linalg.svdvals(A)
    p_ad = _predict_transform(model, U, T, uni)        # uniform decision prior, adapted geometry
    bacc_ad = float(balanced_accuracy_score(yt, p_ad.argmax(1)))
    fin_bacc, fin_ent, fin_occ = _resp_stats(np.asarray(fit.r_final.cpu().numpy()), yt, K)
    out = dict(
        bacc_uniform=bacc_ad, bacc_uniform_identity=float(bacc_id),
        transform_norm=float(((A - I) ** 2).sum().sqrt().cpu()),
        bias_norm=float((T.b ** 2).sum().sqrt().cpu()),
        logdet=float(T.logdet().cpu()),
        condition_number=float((svals.max() / svals.clamp_min(1e-8).min()).cpu()),
        prediction_disagreement=float((p_ad.argmax(1) != p_id.argmax(1)).mean()),
        delta_bacc_uniform=float(bacc_ad - bacc_id),
        delta_nll_uniform=float(nll_id - _nll(p_ad, yt)),
        fit_objective=float(fit.objective),
        final_resp_bacc=fin_bacc, final_resp_entropy=fin_ent, final_class_occupancy=fin_occ)
    nan = float("nan")
    if fit.r_initial is not None:
        ib, ie, io = _resp_stats(np.asarray(fit.r_initial.cpu().numpy()), yt, K)
    else:
        ib = ie = io = nan
    if fit.r_last_used is not None:
        lb, le, lo = _resp_stats(np.asarray(fit.r_last_used.cpu().numpy()), yt, K)
    else:
        lb = le = lo = nan
    out.update(initial_resp_bacc=ib, initial_resp_entropy=ie, initial_class_occupancy=io,
               last_used_resp_bacc=lb, last_used_resp_entropy=le)
    # legacy aliases kept so existing tooling that read these still resolves
    out["responsibility_bacc"] = fin_bacc if np.isnan(lb) else lb
    out["responsibility_entropy"] = fin_ent if np.isnan(le) else le
    out["class_occupancy"] = fin_occ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=DEFAULT_STANDARD)
    ap.add_argument("--variants", default="all",
                    help="comma list of variant names (default all 7); 'identity' for the preflight")
    ap.add_argument("--difficulty", default="standard", choices=["standard", "hard"])
    # GLOBAL grid (defines the experiment identity, shared by every shard)
    ap.add_argument("--grid-seeds", default="0,1,2")
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
    ap.add_argument("--out", default="results/h2cmi/b1a.jsonl")
    ap.add_argument("--bundle-root", default="", help="provenance-keyed source cache (default: <out>.bundles)")
    ap.add_argument("--b0-ref", default="", help="Stage-B0 JSONL: assert standard source checkpoints match")
    ap.add_argument("--b0-ref-soft", action="store_true",
                    help="record b0_match but WARN instead of abort on mismatch (cross-device runs: "
                         "byte-exact reproduction needs B0's original device; rely on source_code_"
                         "signature + source_data_hash for the device-independent invariant)")
    args = ap.parse_args()

    bundle_root = args.bundle_root or (args.out + ".bundles")
    out_dir = os.path.dirname(args.out) or "."
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
    cfg, _cfg_on = build_cfgs(args)                   # CMI-OFF only for the main B1a study
    dkw = difficulty_kwargs(args.difficulty)
    # variant selection: experiment identity is the SELECTED subset (so the identity preflight is
    # its own experiment with items=[identity]); default is the full frozen 7.
    sel = list(VARIANT_NAMES) if args.variants == "all" else [v for v in args.variants.split(",") if v]
    for v in sel:
        if v not in B1A_VARIANTS_BY_NAME:
            raise ValueError(f"unknown variant {v}; have {list(VARIANT_NAMES)}")
    variants = [B1A_VARIANTS_BY_NAME[v] for v in sel]
    variant_names = tuple(v.name for v in variants)
    b0ref = load_b0_checkpoints(args.b0_ref) if args.b0_ref else None
    if b0ref is not None and args.difficulty != "standard":
        raise ValueError("--b0-ref only valid with --difficulty standard (B0 is the standard regime)")
    if b0ref is not None:                              # require complete coverage of the grid
        need = {(s, t, "off") for s in global_seeds for t in global_sites}
        miss = sorted(need - set(b0ref))
        if miss:
            raise RuntimeError(f"--b0-ref missing {len(miss)} source units (e.g. {miss[:4]}); "
                               "a partial reference cannot certify the adaptation-only invariant")

    data_spec = build_data_spec(simulator="PairedEEGSimulator", n_sites=args.sites,
                                subjects_per_site=args.subjects, sessions_per_subject=args.sessions,
                                trials_per_session=args.trials, n_classes=args.classes,
                                n_chans=args.chans, n_times=args.times,
                                source_scenario="population_null",
                                train_seed_policy="train_seed=data_seed",
                                difficulty=args.difficulty, difficulty_spec=dkw)
    cli = {k: getattr(args, k) for k in ("scenarios", "difficulty", "grid_seeds", "grid_target_sites",
            "shard_seeds", "shard_target_sites", "sites", "subjects", "sessions", "trials",
            "classes", "chans", "times", "epochs", "fast")}
    manifest = build_manifest(cfg, global_seeds=global_seeds, global_sites=global_sites,
                              scenarios=scen_canon, items=sorted(variant_names), item_field="variant",
                              cmi_arms=sorted(CMI_ARMS),
                              shard_spec={"seeds": sorted(shard_seeds), "sites": sorted(shard_sites)},
                              cli=cli, data_spec=data_spec)
    validate_or_create_manifest(args.out, manifest)
    done = load_done_keys(args.out, item_field="variant", manifest=manifest)
    print(f"[b1a] commit={runner_commit[:12]} diff={args.difficulty} exp_sig={manifest['experiment_signature']} "
          f"variants={list(variant_names)} shard={manifest['shard_spec']} scenarios={scen_canon} "
          f"-> {args.out} ({len(done)} done)", flush=True)

    uni = np.full(args.classes, 1.0 / args.classes)
    cmi = "off"
    for seed in shard_seeds:
        for tsite in shard_sites:
            if bundle_expected_keys(seed, tsite, cmi, scen_canon, variant_names) <= done:
                print(f"  seed={seed} site={tsite} complete -> skip", flush=True)
                continue
            sim = PairedEEGSimulator(args.classes, args.chans, args.times,
                                     base_noise=dkw["base_noise"], subj_anatomy=dkw["subj_anatomy"],
                                     class_signal_scale=dkw["class_signal_scale"], data_seed=seed)
            full = sim.sample(args.sites, args.subjects, args.sessions, args.trials,
                              target_site=tsite, scenario="population_null")
            src = full.site != tsite
            Xs, ys = full.X[src], full.y[src]
            src_dag, src_dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
            src_dhash = source_data_hash(Xs, ys, src_dom)
            pi_star = reference_prior(ys, args.classes, "uniform")
            cfg.train.seed = seed
            tsig = source_training_signature(cfg, seed, tsite, cmi,
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
            # EMPIRICAL pooled source-latent moments (computed ONCE; the pooled_empirical_diag ref)
            Us = _embed(model, Xs, args.device)
            pooled_ref = reference_weighted_source_moments(Us, ys, pi_star)
            b0_fields = {}
            if b0ref is not None:                          # adaptation-only invariant vs Stage B0
                exp = b0ref[(seed, tsite, "off")]          # coverage already guaranteed present
                actual, scheme = checkpoint_hash_for_scheme(model, exp)
                match = actual == exp
                b0_fields = dict(b0_checkpoint_hash=exp, b0_actual_hash=actual,
                                 b0_hash_scheme=scheme, b0_match=bool(match))
                if not match:
                    msg = (f"source checkpoint diverged from B0 at seed={seed} site={tsite}: "
                           f"{actual} != B0 {exp} ({scheme})")
                    if args.b0_ref_soft:
                        print(f"  WARN {msg} -- soft mode: continuing (rely on source_code_signature "
                              f"+ source_data_hash for the device-independent invariant)", flush=True)
                    else:
                        raise RuntimeError(msg + "; byte-exact match needs B0's original device -- "
                                           "pass --b0-ref-soft to downgrade to a recorded warning")
                print(f"  seed={seed} site={tsite} b0-check match={match} scheme={scheme}", flush=True)
            common = dict(schema_version=SCHEMA_VERSION, runner_commit_sha=runner_commit,
                          source_training_commit_sha=src_commit, source_training_signature=tsig,
                          source_code_signature=code_sig, config_signature=manifest["config_signature"],
                          experiment_signature=manifest["experiment_signature"], difficulty=args.difficulty,
                          data_seed=seed, train_seed=seed, target_site=tsite, cmi=cmi,
                          source_checkpoint_hash=ckpt, source_data_hash=src_dhash, **b0_fields)
            tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, args.classes, args.device)
            for scen_c in scen_canon:
                expected = {(seed, tsite, scen_c, v, cmi) for v in variant_names}
                if expected <= done:
                    continue
                tf = sim.sample(args.sites, args.subjects, args.sessions, args.trials,
                                target_site=tsite, scenario=scen_c)
                tm = tf.site == tsite
                Xt, yt = tf.X[tm], tf.y[tm]
                subj = tf.subject[tm]
                U = _embed(model, Xt, args.device)
                true_prior = np.bincount(yt, minlength=args.classes) / len(yt)
                p_id = _predict_generative(model, U, uni)
                bacc_id = float(balanced_accuracy_score(yt, p_id.argmax(1)))
                nll_id = _nll(p_id, yt)
                # compute the SELECTED variants for this (unit,scenario) so action_regret is consistent
                fits = {}
                for v in variants:
                    ol = yt if v.responsibility == "oracle" else None
                    tseed = stable_hash_int(seed, tsite, scen_c, args.difficulty, v.name)
                    fit = tta.fit_variant(U, v, oracle_labels=ol, pooled_ref=pooled_ref, tta_seed=tseed)
                    grouped = tta.grouped_heldout(U, subj, v, true_labels=yt, oracle_labels=ol,
                                                  pooled_ref=pooled_ref, decision_prior=uni,
                                                  seed_parts=(seed, tsite, scen_c, args.difficulty))
                    fits[v.name] = (v, fit, grouped, tseed)
                # action regret vs the best OOF action INCLUDING identity (review §4)
                valid = {nm: g["grouped_oof_bacc"] for nm, (_v, _f, g, _s) in fits.items()
                         if not np.isnan(g["grouped_oof_bacc"])}
                best_oof = max(valid.values()) if valid else float("nan")
                best_var = max(valid, key=valid.get) if valid else ""
                for v in variants:
                    key = (seed, tsite, scen_c, v.name, cmi)
                    if key in done:
                        continue
                    _v, fit, grouped, tseed = fits[v.name]
                    m = variant_metrics(model, U, fit, yt, uni, p_id, bacc_id, nll_id)
                    g_oof = grouped["grouped_oof_bacc"]
                    regret = float(best_oof - g_oof) if not (np.isnan(best_oof) or np.isnan(g_oof)) else float("nan")
                    pi_np = np.asarray(fit.pi_T.cpu().numpy())
                    row = dict(common, scenario=scen_c, variant=v.name, responsibility=v.responsibility,
                               update=v.update, transform_kind=v.kind, restarts=int(v.restarts),
                               tta_seed=int(tseed), target_size=int(len(yt)), decision_prior="uniform",
                               true_prior=list(np.asarray(true_prior)), fit_prior=list(pi_np),
                               prior_l1_error=float(np.abs(pi_np - true_prior).sum()),
                               action_regret=regret, oracle_best_variant=best_var, **m, **grouped)
                    append_row(args.out, row); done.add(key)
            print(f"  seed={seed} site={tsite} ckpt={ckpt[:12]} done", flush=True)
    print(f"[b1a] complete: {len(done)} units in {args.out}", flush=True)


if __name__ == "__main__":
    main()
