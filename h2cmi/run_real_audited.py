"""Project A — audited real-EEG (or synthetic) Tier-2 pilot runner.

Trains H2-CMI source-only on one dataset, runs the three-setting harness on a held-out target
subject, computes source leakage diagnostics, and emits BOTH raw metrics and a machine-checked
observability report (via the audited evaluation bridge). Every target metric is oracle/
evaluation-only; the estimated target prior is reported but NOT claimed identified (no TU-1
contracts are asserted for a pilot). This is an interface + claim-boundary validation, NOT a
performance claim.

`--dataset synthetic` uses the EEG simulator (fully self-contained; validates the whole path).
`--dataset <MOABB name>` loads offline via cmi.data.moabb_data; if the raw cache is unavailable
it writes a legal SKIP artifact and exits 0 (never fails the pilot on a missing local cache).

Example:
  python -m h2cmi.run_real_audited --dataset synthetic --target-subject 0 --epochs 2 --fast \
      --outdir notes/project_A_observability/results/synthetic_pilot
  python -m h2cmi.run_real_audited --dataset BNCI2014_001 --subjects 1 2 3 --target-subject 1 \
      --epochs 2 --fast --device cpu --outdir notes/project_A_observability/results/bnci_s1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ------------------------------------------------------------------ data loading
def _moabb_domains(meta):
    """Build a (DAG, DomainLabels, subject_col, session_col) from MOABB metadata."""
    from h2cmi.domains.dag import DomainDAG, DomainFactor, DomainLabels
    subj = meta["subject"].astype(str)
    sess_key = subj + "|" + meta["session"].astype(str)
    subj_map = {s: i for i, s in enumerate(sorted(subj.unique()))}
    sess_map = {s: i for i, s in enumerate(sorted(sess_key.unique()))}
    subj_col = subj.map(subj_map).to_numpy().astype(np.int64)
    sess_col = sess_key.map(sess_map).to_numpy().astype(np.int64)
    site_col = np.zeros(len(meta), dtype=np.int64)
    dag = DomainDAG([
        DomainFactor("site", 1, (), "invariant", 0.02),
        DomainFactor("subject", len(subj_map), ("site",), "random_effect", 0.05),
        DomainFactor("session", len(sess_map), ("subject",), "random_effect", 0.10),
    ])
    domains = DomainLabels(dag, np.stack([site_col, subj_col, sess_col], axis=1))
    return dag, domains, subj_col, sess_col, subj_map


def _load_synthetic(n_classes, seed):
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec
    sim = EEGSimulator(n_classes, 12, 128,
                       shift=ShiftSpec(cov=1.0, prior=0.3, montage=0.2, noise=0.3),
                       seed=seed).sample(n_sites=2, subjects_per_site=3,
                                         sessions_per_subject=2, trials_per_session=32)
    info = {"dataset": "synthetic", "n_subjects": int(sim.subject.max()) + 1,
            "n_chans": int(sim.X.shape[1]), "n_times": int(sim.X.shape[2])}
    return sim.X, sim.y, sim.dag, sim.domains, sim.subject, sim.session, n_classes, info


def _load_moabb(name, subjects, seed):
    from cmi.data import moabb_data
    X, y, meta, classes = moabb_data.load(name, subjects=subjects)
    dag, domains, subj_col, sess_col, subj_map = _moabb_domains(meta)
    info = {"dataset": name, "subjects": subjects, "classes": list(classes),
            "subject_map": subj_map, "n_chans": int(X.shape[1]), "n_times": int(X.shape[2])}
    return X, y, dag, domains, subj_col, sess_col, len(classes), info


# -------------------------------------------------- label-free R1 representation/prior diagnostics
def _pairwise_sq(A, B):
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def _nn_dist(A, B, exclude_self=False):
    D = _pairwise_sq(A, B)
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return np.sqrt(D.min(1))


def _mmd_rbf(X, Y):
    both = np.vstack([X, Y])
    g = 1.0 / (float(np.median(_pairwise_sq(both, both))) + 1e-12)      # median-heuristic bandwidth
    kxx = np.exp(-g * _pairwise_sq(X, X)).mean()
    kyy = np.exp(-g * _pairwise_sq(Y, Y)).mean()
    kxy = np.exp(-g * _pairwise_sq(X, Y)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def _representation_diagnostics(Zs, Zt, seed=0, max_n=400):
    """LABEL-FREE source/target representation-shift proxies (MMD-RBF, centroid distance, target->
    source kNN distance, off-source mass). Uses embeddings only — NO labels. Reason-codes on failure."""
    keys = ["source_target_mmd_rbf", "source_target_centroid_distance",
            "target_knn_distance_mean", "target_off_source_mass_proxy"]
    try:
        rng = np.random.default_rng(int(seed))
        Zs = np.asarray(Zs, dtype=np.float64)
        Zt = np.asarray(Zt, dtype=np.float64)

        def _sub(Z):
            return Z[rng.choice(len(Z), max_n, replace=False)] if len(Z) > max_n else Z

        Zs_, Zt_ = _sub(Zs), _sub(Zt)
        d_ts = _nn_dist(Zt_, Zs_)
        thr = float(np.percentile(_nn_dist(Zs_, Zs_, exclude_self=True), 95))
        return ({"source_target_mmd_rbf": round(_mmd_rbf(Zs_, Zt_), 6),
                 "source_target_centroid_distance": round(float(np.linalg.norm(Zs.mean(0) - Zt.mean(0))), 6),
                 "target_knn_distance_mean": round(float(d_ts.mean()), 6),
                 "target_off_source_mass_proxy": round(float(np.mean(d_ts > thr)), 6)}, {})
    except Exception as exc:                                  # reason-coded, never a silent 0
        return ({k: None for k in keys},
                {k: f"repr diagnostics failed: {type(exc).__name__}: {exc}" for k in keys})


def _prior_diagnostics(pi_dict, source_prior):
    """LABEL-FREE target-prior-estimate diagnostics from the exported per-domain pi_T (no labels)."""
    keys = ["target_prior_entropy_hat", "target_prior_shift_l1_hat",
            "target_prior_shift_l1_from_source", "prior_estimate_max_mass"]
    pis = [np.asarray(v, dtype=np.float64) for v in (pi_dict or {}).values() if v is not None]
    if not pis:
        return {k: None for k in keys}
    mean_pi = np.mean(pis, axis=0)
    u = np.full(len(mean_pi), 1.0 / len(mean_pi))
    l1_src = (float(np.abs(mean_pi - np.asarray(source_prior, dtype=np.float64)).sum())
              if source_prior is not None else None)
    return {"target_prior_entropy_hat": round(float(-(mean_pi * np.log(mean_pi + 1e-12)).sum()), 6),
            "target_prior_shift_l1_hat": round(float(np.abs(mean_pi - u).sum()), 6),
            "target_prior_shift_l1_from_source": round(l1_src, 6) if l1_src is not None else None,
            "prior_estimate_max_mass": round(float(mean_pi.max()), 6)}


# ------------------------------------------------------------------ pilot
def _build_cfg(n_classes, n_chans, n_times, epochs, device, seed, fast):
    from h2cmi.config import H2Config
    cfg = H2Config(n_classes=n_classes)
    cfg.encoder.n_chans = n_chans
    cfg.encoder.n_times = n_times
    cfg.train.device = device
    cfg.train.seed = seed
    if fast:
        cfg.encoder.z_c_dim = 16
        cfg.encoder.z_n_dim = 8
        cfg.cmi.critic_inner = 1
        cfg.tta.em_iters = 6
        cfg.tta.min_target = 12
    cfg.train.epochs = epochs
    return cfg


def _run_pilot(X, y, dag, domains, subj_col, target_subject_idx, cfg, n_classes, n_perm,
               align_factor="site"):
    from h2cmi.eval.harness import _json_safe, run_three_settings
    from h2cmi.eval.leakage import crossfit_conditional_leakage
    from h2cmi.train.trainer import reference_prior, train_h2

    src_idx = np.where(subj_col != target_subject_idx)[0]
    tgt_idx = np.where(subj_col == target_subject_idx)[0]
    if len(src_idx) == 0 or len(tgt_idx) == 0:
        raise ValueError(f"empty source/target split for target subject {target_subject_idx}")

    Xs, ys = X[src_idx], y[src_idx]
    src_domains = domains.subset(src_idx)
    Xt, yt = X[tgt_idx], y[tgt_idx]
    tgt_unit = domains.subset(tgt_idx).factor("session")     # target sessions as TTA domains
    src_unit = src_domains.factor("subject")                 # source subjects for gate pseudo-targets

    model, _hcmi, _dual, hist = train_h2(Xs, ys, src_domains, dag, cfg, align_factor=align_factor)
    pi_star = reference_prior(ys, n_classes, cfg.align.reference_prior)
    res = run_three_settings(model, Xt, yt, tgt_unit, cfg, pi_star, X_src=Xs, y_src=ys,
                             gate_pseudo_levels=src_unit, device=cfg.train.device)
    Zs = model.embed(Xs, device=cfg.train.device)            # match model's device (train_h2 left it on cuda)
    Zt = model.embed(Xt, device=cfg.train.device)            # target embeddings (label-free)
    leak = crossfit_conditional_leakage(Zs, ys, src_domains, dag, n_classes,
                                        n_perm=n_perm, seed=cfg.train.seed)

    # split the offline-TTA output: label-free prediction diagnostics + oracle per-trial block
    off = res["offline_tta"]
    r1_pred = off.pop("r1_prediction_diagnostics", {})       # label-free (from predictions only)
    per_trial = off.pop("per_trial_oracle_predictions", {})  # oracle/evaluation-only (R2 curves)
    r1_rep, r1_rep_missing = _representation_diagnostics(Zs, Zt, seed=cfg.train.seed)
    r1_prior = _prior_diagnostics(off.get("per_domain_pi_T"), pi_star)
    r1_diagnostics = {**r1_pred, **r1_rep, **r1_prior}
    r1_missing = {k: "null (not computable from available outputs)"
                  for k, v in r1_diagnostics.items() if v is None}
    r1_missing.update(r1_rep_missing)

    raw = {"strict_dg": _json_safe(res["strict_dg"]),
           "offline_tta": _json_safe(off),
           "online_tta": _json_safe(res["online_tta"]),
           "gate_info": _json_safe(res.get("gate_info", {})),
           "leakage": _json_safe(leak),
           "r1_diagnostics": _json_safe(r1_diagnostics),          # LABEL-FREE target-unlabeled diagnostics
           "r1_diagnostics_missing": r1_missing,
           "per_trial_oracle_predictions": _json_safe(per_trial),  # oracle/evaluation-only (R2 only)
           "train_history": _json_safe(hist),
           "n_source_trials": int(len(src_idx)), "n_target_trials": int(len(tgt_idx))}
    return res, leak, raw


def _maybe_set_threads(n):
    """Set torch intra-op threads for CPU multi-core runs (SLURM CPU jobs pass --threads)."""
    if n and n > 0:
        import torch
        torch.set_num_threads(int(n))


def _write_skip(out_dir, reason, args, environment):
    skip = {"status": "skipped", "skip_reason": reason, "dataset": args.dataset,
            "requested_subjects": args.subjects, "target_subject": args.target_subject,
            "seed": getattr(args, "seed", None),           # record seed so the cell matches its grid slot
            "environment": environment}
    (out_dir / "run_manifest.json").write_text(json.dumps(skip, indent=2))
    # stub report for a uniform file set (consumers still branch on manifest['status'])
    (out_dir / "observability_report.json").write_text(json.dumps(
        {"title": f"skipped: {args.dataset}", "status": "skipped", "skip_reason": reason,
         "summary": {"n_claims": 0, "n_allowed": 0, "n_rejected": 0},
         "forbidden_claims_checked": [], "forbidden_claims_violated": [], "claims": []}, indent=2))
    (out_dir / "observability_report.md").write_text(
        f"# Real-EEG audited pilot — SKIPPED\n\n- reason: **{reason}**\n- dataset: `{args.dataset}`\n"
        f"- requested subjects: {args.subjects}\n- target subject: {args.target_subject}\n\n"
        f"No target metric was produced, so no claim boundary is asserted.\n")
    print(f"REAL EEG AUDIT PILOT SKIPPED: {reason}")


def _finalize(out_dir, raw, *, dataset, subjects, target_subject, target_index, seed, epochs,
              fast, device, align_factor, factor_levels, align_degenerate, n_classes, n_chans,
              n_times, dataset_info, environment):
    """Build the audited report + write the 4 artifacts + manifest for ONE run. Shared by the
    single-run `main` and the grid runner so their reports/manifests never drift."""
    from h2cmi.observability import (assert_forbidden_claims_not_made, build_audited_eval_report,
                                     write_observability_report_json, write_observability_report_md)
    report = build_audited_eval_report(
        f"Audited Tier-2 pilot — {dataset} target-subject {target_subject} seed {seed}",
        strict_dg=raw["strict_dg"], offline_tta=raw["offline_tta"], online_tta=raw["online_tta"],
        leakage=raw["leakage"], prior_contracts=None, prior_conclusion=False,  # honest: not identified
        has_oracle_target_labels=True)
    assert_forbidden_claims_not_made(report)                  # clean: no rejected CONCLUSION
    (out_dir / "raw_results.json").write_text(json.dumps(raw, indent=2))
    data = write_observability_report_json(report, str(out_dir / "observability_report.json"))
    write_observability_report_md(report, str(out_dir / "observability_report.md"))
    manifest = {"status": "ok", "dataset": dataset, "subjects": subjects,
                "target_subject": target_subject, "target_index": int(target_index), "seed": seed,
                "epochs": epochs, "fast": fast, "device": device, "align_factor": align_factor,
                "domain_factor_levels": factor_levels,
                "alignment_factor_degenerate": bool(align_degenerate),
                "n_source_trials": raw["n_source_trials"], "n_target_trials": raw["n_target_trials"],
                "n_classes": n_classes, "n_chans": int(n_chans), "n_times": int(n_times),
                "dataset_info": dataset_info, "environment": environment,
                "audit_summary": data["summary"],
                "forbidden_claims_violated": data["forbidden_claims_violated"]}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description="Project A audited real-EEG/synthetic Tier-2 pilot")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--target-subject", type=int, default=0,
                    help="MOABB subject id (or synthetic subject index)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--align-factor", default=None,
                    help="domain factor to align on (default: 'subject' for MOABB, 'site' for synthetic)")
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch intra-op threads for CPU multi-core (0 = leave torch default)")
    ap.add_argument("--outdir", default="notes/project_A_observability/results/audited_pilot")
    args = ap.parse_args(argv)
    _maybe_set_threads(args.threads)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version.split()[0], "device": args.device}

    # ---- load (graceful skip on MOABB-unavailable) --------------------------------------
    try:
        if args.dataset == "synthetic":
            X, y, dag, domains, subj_col, _sess, n_classes, info = _load_synthetic(
                args.n_classes, args.seed)
            target_idx = args.target_subject
        else:
            X, y, dag, domains, subj_col, _sess, n_classes, info = _load_moabb(
                args.dataset, args.subjects, args.seed)
            # map the requested MOABB subject id to its contiguous column index
            smap = info["subject_map"]
            key = str(args.target_subject)
            if key not in smap:
                _write_skip(out_dir, f"target subject {key} not in loaded subjects {sorted(smap)}",
                            args, environment)
                return 0
            target_idx = smap[key]
    except Exception as exc:                                   # reason-coded graceful skip
        _write_skip(out_dir, f"{type(exc).__name__}: {exc}", args, environment)
        return 0

    # ---- resolve alignment factor + record domain-factor structure ---------------------
    align_factor = args.align_factor or ("site" if args.dataset == "synthetic" else "subject")
    factor_levels = {f.name: int(f.n_levels) for f in dag.factors}
    align_degenerate = factor_levels.get(align_factor, 0) <= 1   # e.g. single-site MOABB

    # ---- train + evaluate (skip on any pilot failure — pilot must not hard-fail) --------
    try:
        cfg = _build_cfg(n_classes, X.shape[1], X.shape[2], args.epochs, args.device,
                         args.seed, args.fast)
        _res, _leak, raw = _run_pilot(X, y, dag, domains, subj_col, target_idx, cfg,
                                      n_classes, args.n_perm, align_factor=align_factor)
    except Exception as exc:
        _write_skip(out_dir, f"pilot run failed: {type(exc).__name__}: {exc}", args, environment)
        return 0

    # ---- audited observability report (prior reported but NOT claimed identified) -------
    data = _finalize(out_dir, raw, dataset=args.dataset, subjects=args.subjects,
                     target_subject=args.target_subject, target_index=target_idx, seed=args.seed,
                     epochs=args.epochs, fast=args.fast, device=args.device,
                     align_factor=align_factor, factor_levels=factor_levels,
                     align_degenerate=align_degenerate, n_classes=n_classes, n_chans=X.shape[1],
                     n_times=X.shape[2],
                     dataset_info={k: v for k, v in info.items() if k != "subject_map"},
                     environment=environment)

    s = data["summary"]
    print(f"AUDITED PILOT OK  dataset={args.dataset}  target={args.target_subject}  "
          f"claims={s['n_claims']} allowed={s['n_allowed']} rejected={s['n_rejected']}  "
          f"violations={len(data['forbidden_claims_violated'])}")
    print(f"wrote {out_dir}/raw_results.json, observability_report.json/.md, run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
