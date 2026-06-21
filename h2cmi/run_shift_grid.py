"""Shift-grid driver (review §"Shift-grid 的关键设计"): the CMI x TTA factorial across
orthogonal shift mechanisms, with oracle diagnostics, paired source models and resume.

Per (data_seed, target_site):
  * generate the CANONICAL source data once (scenario-independent);
  * train Source-A (CMI off) and Source-B (CMI on) ONCE from the same init + batch order;
  * for each scenario, generate the target-site data and evaluate
        M0 = A + identity   M1 = B + identity   M2 = A + TTA   M3 = B + TTA
    plus oracle_prior / oracle_labels / oracle_supervised_transform on each source model.

M0/M2 literally share Source-A; M1/M3 share Source-B. Rows are appended atomically to a
JSONL so a SLURM interruption can resume (completed (seed,site,scenario,method,cmi) units
are skipped). Aggregate to CSV/Parquet afterwards.

Tiny integration first:
  python -m h2cmi.run_shift_grid --scenarios no_shift,cov,prior,concept --seeds 0 \
    --target-sites all --sites 3 --subjects 2 --sessions 2 --trials 12 --epochs 2 --fast \
    --out results/h2cmi/shift_grid_smoke.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.config import H2Config, core_config
from h2cmi.domains import compact_domain_labels
from h2cmi.data.paired_simulator import PairedEEGSimulator, PRESET_SCENARIOS
from h2cmi.train.trainer import train_h2, reference_prior
from h2cmi.eval.metrics import classification_metrics
from h2cmi.eval.harness import _predict_generative, _predict_transform, _embed
from h2cmi.eval.leakage import crossfit_conditional_leakage
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.tta.oracles import (oracle_prior, oracle_labels, oracle_supervised_oof,
                               crossfit_supervised_gain)


# --------------------------------------------------------------------- hashing
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__) or ".").decode().strip()
    except Exception:
        return "unknown"


def _hash(obj) -> str:
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]


def hash_array(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]


def hash_state(model) -> str:
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def config_hash(cfg: H2Config, args) -> str:
    return _hash((cfg.n_classes, cfg.encoder.z_c_dim, cfg.encoder.use_spd, cfg.density.df,
                  cfg.tta.transform, cfg.train.epochs, args.sites, args.subjects,
                  args.sessions, args.trials))


# --------------------------------------------------------------------- io / resume
def load_done_keys(path: str, item_field: str = "method") -> set:
    """Strict resume index keyed by (data_seed, target_site, scenario, item_field, cmi).

    ``item_field`` is the row's unit column ("method" for the shift grid, "action" for the
    action grid). FAILS LOUDLY on malformed JSON, missing fields or duplicate keys instead
    of silently skipping (the old broad ``except`` made the action grid non-resumable: it
    read "method", action-grid rows have "action", so every row was dropped and re-runs
    appended duplicates)."""
    keys: set = set()
    if not os.path.exists(path):
        return keys
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_no}") from exc
            required = {"data_seed", "target_site", "scenario", item_field, "cmi"}
            missing = required - row.keys()
            if missing:
                raise KeyError(f"{path}:{line_no} missing fields {sorted(missing)}")
            key = (row["data_seed"], row["target_site"], row["scenario"],
                   row[item_field], row["cmi"])
            if key in keys:
                raise ValueError(f"Duplicate result key at {path}:{line_no}: {key}")
            keys.add(key)
    return keys


def append_row(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
        f.flush(); os.fsync(f.fileno())


# --------------------------------------------------------------------- config
def build_cfgs(args):
    base = core_config(H2Config(n_classes=args.classes))
    base.encoder.n_chans = args.chans
    base.encoder.n_times = args.times
    base.train.epochs = args.epochs
    base.train.device = args.device
    if args.fast:
        base.cmi.critic_inner = 1
        base.tta.em_iters = 8
    import copy
    cfg_off = copy.deepcopy(base); cfg_off.cmi.enabled = False
    cfg_on = copy.deepcopy(base); cfg_on.cmi.enabled = True
    return cfg_off, cfg_on


# --------------------------------------------------------------------- eval helpers
def _metrics_row(p, y):
    m = classification_metrics(p, y)
    return m["balanced_acc"], m["macro_f1"], m["nll"], m["brier"], m["ece"]


def _transform_fields(res):
    d = res.diagnostics
    T = res.transform
    return dict(adapted=bool(res.adapted),
                rollback_reason=d.get("reason", ""),
                crossfit_evidence_gain=d.get("crossfit_evidence_gain", float("nan")),
                nll_before=d.get("nll_before", float("nan")),
                nll_after=d.get("nll_after", float("nan")),
                transform_norm=d.get("transform_norm", 0.0),
                bias_norm=float(np.linalg.norm(T.b.detach().cpu().numpy())),
                logdet=float(T.logdet().detach().cpu()),
                condition_number=d.get("condition_number", 1.0))


def run_unit(model, U, y_tgt, strict_bacc, pi_star, true_prior, base_row, cmi_tag,
             tta_cfg, n_classes, device, out_path, done, groups=None):
    """Emit identity / tta / oracle rows for one source model on one target."""
    def emit(method, p, extra=None, factorial_cell=None):
        key = (base_row["data_seed"], base_row["target_site"], base_row["scenario"], method, cmi_tag)
        if key in done:
            return
        bacc, f1, nll, brier, ece = _metrics_row(p, y_tgt)
        row = dict(base_row, method=method, cmi=cmi_tag, factorial_cell=factorial_cell,
                   strict_bacc=strict_bacc, adapted_bacc=bacc, delta_bacc=bacc - strict_bacc,
                   macro_f1=f1, nll=nll, brier=brier, ece=ece, true_prior=list(true_prior))
        if extra:
            row.update(extra)
        append_row(out_path, row); done.add(key)

    # identity (M0/M1)
    p_id = _predict_generative(model, U, pi_star)
    emit("identity", p_id, factorial_cell=("M1" if cmi_tag == "on" else "M0"))

    # unsupervised TTA (M2/M3)
    tta = ClassConditionalTTA(model.head.density, pi_star, tta_cfg, n_classes, device)
    res = tta.fit(U, pseudo_labels=p_id.argmax(1))
    p_ad = _predict_transform(model, U, res.transform, res.pi_T)
    tf = _transform_fields(res)
    tf.update(estimated_prior=list(np.asarray(res.pi_T)),
              prior_l1_error=float(np.abs(np.asarray(res.pi_T) - true_prior).sum()))
    emit("tta", p_ad, extra=tf, factorial_cell=("M3" if cmi_tag == "on" else "M2"))

    # transductive oracles (apply a full-data transform): true prior / true responsibilities
    for name, fn in (("oracle_prior", oracle_prior), ("oracle_labels", oracle_labels)):
        ores = fn(tta, U, y_tgt)
        p_or = _predict_transform(model, U, ores.transform, ores.pi_T)
        of = _transform_fields(ores)
        of.update(estimated_prior=list(np.asarray(ores.pi_T)),
                  prior_l1_error=float(np.abs(np.asarray(ores.pi_T) - true_prior).sum()))
        emit(name, p_or, extra=of)
    # TRUE held-out supervised ceiling (out-of-fold by subject) -- the honest accuracy ceiling
    proba_oof, oof_info = oracle_supervised_oof(tta, U, y_tgt, groups=groups)
    emit("oracle_supervised_oof", proba_oof,
         extra=dict(crossfit_supervised_gain=crossfit_supervised_gain(tta, U, y_tgt),
                    oof_groups=oof_info["oof_groups"], oof=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="no_shift,cov,prior,concept")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--target-sites", default="all")
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--chans", type=int, default=16)
    ap.add_argument("--times", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--leak-perm", type=int, default=0, help="permutation null reps for leakage (0=skip)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="results/h2cmi/shift_grid.jsonl")
    args = ap.parse_args()

    scenarios = [s for s in args.scenarios.split(",") if s]
    for s in scenarios:
        if s not in PRESET_SCENARIOS:
            raise ValueError(f"unknown scenario {s}; have {list(PRESET_SCENARIOS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    sites = (list(range(args.sites)) if args.target_sites == "all"
             else [int(s) for s in args.target_sites.split(",")])
    cfg_off, cfg_on = build_cfgs(args)
    sha = git_sha(); chash = config_hash(cfg_on, args)
    done = load_done_keys(args.out, item_field="method")
    print(f"[shift-grid] sha={sha} scenarios={scenarios} seeds={seeds} sites={sites} "
          f"-> {args.out} ({len(done)} units already done)", flush=True)

    for seed in seeds:
        for tsite in sites:
            sim = PairedEEGSimulator(args.classes, args.chans, args.times, data_seed=seed)
            # canonical source (scenario-independent): take non-target sites of any scenario
            full = sim.sample(args.sites, args.subjects, args.sessions, args.trials,
                              target_site=tsite, scenario="no_shift")
            src = full.site != tsite
            Xs, ys = full.X[src], full.y[src]
            src_dag, src_dom, _ = compact_domain_labels(full.domains.subset(np.where(src)[0]))
            src_data_hash = hash_array(Xs) + "_" + hash_array(ys)
            pi_star = reference_prior(ys, args.classes, "uniform")

            for cfg, cmi_tag in ((cfg_off, "off"), (cfg_on, "on")):
                cfg.train.seed = seed
                model, hcmi, dual, hist = train_h2(Xs, ys, src_dom, src_dag, cfg, align_factor="site")
                ckpt_hash = hash_state(model)
                last = hist[-1] if hist else {}
                leak = crossfit_conditional_leakage(model.embed(Xs, device=args.device), ys,
                                                    src_dom, src_dag, args.classes,
                                                    device=args.device, n_perm=args.leak_perm, seed=seed)
                leak_fields = {f"{f}_leakage": leak.get(f, {}).get("I_hat", float("nan"))
                               for f in ("site", "subject", "session")}
                base_common = dict(commit_sha=sha, config_hash=chash, data_seed=seed,
                                   train_seed=seed, tta_seed=seed, target_site=tsite,
                                   source_data_hash=src_data_hash, source_checkpoint_hash=ckpt_hash,
                                   final_lambdas=last.get("lambda_", {}), critic_ce=last.get("critic_ce", 0.0),
                                   **leak_fields)
                for scen in scenarios:
                    scen_canon = PRESET_SCENARIOS[scen].name        # write canonical names only
                    tgt_full = sim.sample(args.sites, args.subjects, args.sessions, args.trials,
                                          target_site=tsite, scenario=scen)
                    tm = tgt_full.site == tsite
                    Xt, yt = tgt_full.X[tm], tgt_full.y[tm]
                    groups = tgt_full.domains.subset(np.where(tm)[0]).factor("subject")  # OOF folds
                    U = _embed(model, Xt, args.device)
                    strict = balanced_accuracy_score(yt, _predict_generative(model, U, pi_star).argmax(1))
                    true_prior = np.bincount(yt, minlength=args.classes) / len(yt)
                    base_row = dict(base_common, scenario=scen_canon, target_size=int(len(yt)))
                    run_unit(model, U, yt, strict, pi_star, true_prior, base_row, cmi_tag,
                             cfg.tta, args.classes, args.device, args.out, done, groups=groups)
                print(f"  seed={seed} site={tsite} cmi={cmi_tag} ckpt={ckpt_hash} done", flush=True)
    print(f"[shift-grid] complete: {len(done)} units in {args.out}", flush=True)


if __name__ == "__main__":
    main()
