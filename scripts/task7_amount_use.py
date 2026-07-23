"""CMI-Trace Task 7 — cross-backbone AMOUNT (lambda) vs USE (tau) subject spectrum on the tos frozen dumps.

Ports the E1 whitened subject-spectrum (cmi.eval.subject_spectrum.subject_spectrum) onto the tos_cmi frozen
LOSO dumps to test whether the four-object separation (subject info ENCODED but NOT functionally USED) holds
across BOTH backbones (EEGNet d_z=16, TSMNet d_z=210), i.e. it is not an EEGNet artifact.

USE (tau) measure:
  * PRIMARY (both backbones): a SOURCE-fit multinomial-logistic PROBE head -> representation reliance. Linear,
    so head-replay is exact BY CONSTRUCTION (comparable across backbones).
  * VALIDATION (TSMNet only): the EXACT deployed head recovered from stored logits (lstsq Z_src->logits_src;
    replay max|d|~1e-6). Shows probe-tau tracks deployed-tau. EEGNet's stored 16-d penultimate->logits map is
    NON-linear (replay ~2 nats) so the deployed head is NOT exact-recoverable -> EEGNet USE is representation-
    level only (disclosed).

FIREWALL: whitening / subject subspace / lambda / tau / probe head are ALL fit on SOURCE rows; target rows are
eval-only and carry a distinct domain tag. No target labels touch any fit.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
import numpy as np

sys.path.insert(0, ".")
from cmi.eval.subject_spectrum import subject_spectrum

TOS_ROOT = "/home/infres/yinwang/CMI_AAAI_tos/tos_cmi/results/tos_cmi_eeg_frozen"
OUT_DIR = "results/task7_amount_use"
DATASETS = ["Lee2019_MI", "Cho2017", "BNCI2015_001"]
BACKBONES = ["EEGNet", "TSMNet"]


def _remap(a):
    u = np.unique(a); return np.searchsorted(u, a).astype(np.int64), len(u)


def _probe_head(Zs, ys, n_cls, seed=0):
    """SOURCE-fit multinomial-logistic head -> symmetric n_cls logits (replay-exact linear head)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    lr.fit(Zs, ys)
    if n_cls == 2:                                    # binary: build symmetric 2-logit head
        w = lr.coef_[0]; b0 = lr.intercept_[0]
        W = np.vstack([-0.5 * w, 0.5 * w]); b = np.array([-0.5 * b0, 0.5 * b0])
    else:
        W = lr.coef_; b = lr.intercept_
    return W, b


def _exact_head(Zs, Ls):
    """Recover the deployed linear head from stored source logits (lstsq). Returns (W,b,max_abs_diff)."""
    A = np.hstack([Zs, np.ones((len(Zs), 1))])
    sol, *_ = np.linalg.lstsq(A, Ls, rcond=None)
    W = sol[:-1].T; b = sol[-1]
    diff = float(np.abs(A @ sol - Ls).max())
    return W, b, diff


def build_data(path, head="probe", seed=0):
    """tos dump -> audit-npz-like `data` dict consumable by subject_spectrum (replay-exact head)."""
    d = np.load(path, allow_pickle=True)
    Zs = d["Z_source"].astype(float); ys = d["y_source"].astype(np.int64)
    Zt = d["Z_target"].astype(float); yt = d["y_target"].astype(np.int64)
    ds_src, n_dom = _remap(np.asarray(d["subject_source"]))
    n_cls = int(d["n_cls"])
    Z = np.vstack([Zs, Zt]); y = np.concatenate([ys, yt])
    dd = np.concatenate([ds_src, np.full(len(Zt), n_dom, np.int64)])   # target tag = n_dom (distinct)
    src_idx = np.arange(len(Zs)); tgt_idx = np.arange(len(Zs), len(Zs) + len(Zt))
    if head == "exact":
        W, b, diff = _exact_head(Zs, d["logits_source"].astype(float))
        model_logits = Z @ W.T + b; replay_ok = bool(diff <= 1e-5); head_kind = "exact_deployed"
    else:
        W, b = _probe_head(Zs, ys, n_cls, seed); model_logits = Z @ W.T + b
        replay_ok = True; head_kind = "source_probe"
    data = {"graph_z": Z, "y": y, "d": dd, "model_logits": model_logits,
            "source_indices": src_idx, "target_indices": tgt_idx,
            "task_head_weight": W, "task_head_bias": b, "task_head_input": "graph_z",
            "task_head_kind": "linear", "task_head_replay_ok": replay_ok,
            "dataset": str(np.asarray(d["dataset"])), "backbone": str(np.asarray(d["backbone"])),
            "method": "erm", "seed": int(np.asarray(d.get("seed", seed))),
            "fold": int(re.search(r"sub(\d+)_", path.split("/")[-1]).group(1))}
    return data, head_kind


def run_one(path, representation="graph_z", head="probe", k_spec=16, n_perm=50, n_random=50, seed=0):
    data, head_kind = build_data(path, head=head, seed=seed)
    t0 = time.time()
    spec = subject_spectrum(data, representation=representation, k_spec=k_spec,
                            n_perm=n_perm, n_random=n_random, seed=seed)
    spec["head_kind"] = head_kind; spec["path"] = path; spec["secs"] = round(time.time() - t0, 1)
    return spec


def _dumps(ds, bb, n=None, seed=0):
    ps = sorted(glob.glob(f"{TOS_ROOT}/{ds}_{bb}_LOSO/sub*_erm_lam0_seed{seed}.npz"),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    return ps[:n] if n else ps


# ------------------------------------------------------------------ fleet (per-dump JSON, skip-if-done)
def _tasks(n_folds):
    """(ds, bb, head, path). head=probe for both backbones; exact ALSO for TSMNet (E7.5 validation)."""
    out = []
    for ds in DATASETS:
        for bb in BACKBONES:
            for p in _dumps(ds, bb, n=n_folds):
                out.append((ds, bb, "probe", p))
                if bb == "TSMNet":
                    out.append((ds, bb, "exact", p))
    return out


def _out_path(ds, bb, head, path):
    fold = int(re.search(r"sub(\d+)_", path.split("/")[-1]).group(1))
    return f"{OUT_DIR}/{ds}_{bb}_{head}_sub{fold}.json"


def _run_task(ds, bb, head, path, k_spec, n_perm, n_random):
    op = _out_path(ds, bb, head, path)
    if os.path.exists(op):
        return op, "skip"
    try:
        s = run_one(path, head=head, k_spec=k_spec, n_perm=n_perm, n_random=n_random)
        s.update(dataset=ds, backbone=bb, head=head)
        json.dump(s, open(op, "w"))
        return op, "ok"
    except Exception as e:                                     # reason-code, never silent
        return op, "ERR:%s" % (repr(e)[:200])


def fleet(n_folds=12, k_spec=16, n_perm=100, n_random=50, n_jobs=8):
    from joblib import Parallel, delayed
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = _tasks(n_folds)
    print(f"[fleet] {len(tasks)} dump-tasks (n_folds={n_folds}, k_spec={k_spec}, n_perm={n_perm}) n_jobs={n_jobs}")
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_task)(ds, bb, h, p, k_spec, n_perm, n_random) for (ds, bb, h, p) in tasks)
    ok = sum(r[1] == "ok" for r in res); sk = sum(r[1] == "skip" for r in res)
    errs = [r for r in res if r[1].startswith("ERR")]
    print(f"[fleet] ok={ok} skip={sk} err={len(errs)}")
    for r in errs[:20]:
        print("  ERR", r[0], r[1])
    print("TASK7_FLEET_DONE" if not errs else "TASK7_FLEET_DONE_WITH_ERRORS")


# ------------------------------------------------------------------ aggregate (fold-cluster bootstrap)
def _fold_boot(folds, stat, n_boot=10000, seed=0):
    folds = list(folds)
    if not folds:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed); point = stat(folds); boots = []
    for _ in range(n_boot):
        samp = [folds[i] for i in rng.integers(0, len(folds), len(folds))]
        v = stat(samp)
        if np.isfinite(v):
            boots.append(v)
    return float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _load(head):
    cells = {}
    for fp in glob.glob(f"{OUT_DIR}/*_{head}_sub*.json"):
        s = json.load(open(fp))
        cells[(s["dataset"], s["backbone"], s["fold"])] = s
    return cells


def _fold_entry(s):
    D = s["directions"]
    lam = np.array([r["lambda_excess_over_null"] for r in D])
    tau = np.array([r["tau_ce_reliance"] for r in D])
    taur = np.array([r["tau_random_mean"] for r in D])
    pp = np.array([r["lambda_perm_p"] for r in D])
    rank = np.arange(len(D), dtype=float)                     # direction index j = descending-energy rank
    return {"lam": lam, "tau": tau, "taur": taur, "rank": rank,
            "frac_sig": float((pp < 0.05).mean()),
            "mean_abs_tau": float(np.abs(tau).mean()),
            "mean_tau_minus_r": float((tau - taur).mean()),
            "mean_lam": float(lam.mean())}


def _corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _partial(a, b, c):
    """Partial corr(a,b | c). lambda is largely an energy-rank proxy, so the raw corr(lambda,tau) is confounded
    by direction energy; controlling for rank j is the adversarially-verified correct estimand (V3)."""
    rab, rac, rbc = _corr(a, b), _corr(a, c), _corr(b, c)
    if any(np.isnan([rab, rac, rbc])):
        return np.nan
    den = (1 - rac ** 2) * (1 - rbc ** 2)
    if den <= 1e-12:
        return np.nan
    return (rab - rac * rbc) / np.sqrt(den)


def _endpoints(fold_entries, n_boot):
    def sc(key):
        return _fold_boot(fold_entries, lambda F: float(np.mean([e[key] for e in F])), n_boot)
    corr_lt = _fold_boot(fold_entries,
                         lambda F: _corr(np.concatenate([e["lam"] for e in F]),
                                         np.concatenate([e["tau"] for e in F])), n_boot)
    # energy-controlled partial corr (V3): lambda is ~an energy-rank proxy, so the raw corr is confounded.
    part_lt = _fold_boot(fold_entries,
                         lambda F: _partial(np.concatenate([e["lam"] for e in F]),
                                            np.concatenate([e["tau"] for e in F]),
                                            np.concatenate([e["rank"] for e in F])), n_boot)
    return {
        "n_folds": len(fold_entries),
        "E7.1_frac_lambda_sig": sc("frac_sig"),
        "E7.2_mean_tau_minus_random": sc("mean_tau_minus_r"),
        "E7.2_mean_abs_tau": sc("mean_abs_tau"),
        "mean_lambda_excess": sc("mean_lam"),
        "E7.3_corr_lambda_tau_RAW_confounded": corr_lt,
        "E7.3_partial_corr_lambda_tau_given_energyrank": part_lt,
    }


def aggregate(n_boot=10000, n_folds=12):
    cells = _load("probe")
    # coverage (proper unique count) + firewall gate
    exp = [(ds, bb, f) for ds in DATASETS for bb in BACKBONES
           for f in sorted({int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
                            for p in _dumps(ds, bb, n=n_folds)})]
    missing = [k for k in exp if k not in cells]
    bad_fw = [k for k, s in cells.items() if not s.get("firewall_passed", False)]
    bad_hr = [k for k, s in cells.items() if not s.get("head_replay_verified", False)]
    out = {"n_boot": n_boot, "n_folds_per_cell": n_folds, "coverage": {"expected": len(exp),
           "have": len(cells), "missing": [f"{a}/{b}/sub{c}" for a, b, c in missing]},
           "firewall_fail": [f"{a}/{b}/sub{c}" for a, b, c in bad_fw],
           "head_replay_fail": [f"{a}/{b}/sub{c}" for a, b, c in bad_hr]}
    if missing or bad_fw or bad_hr:
        out["status"] = "INCOMPLETE_OR_QC_FAIL"
        json.dump(out, open(f"{OUT_DIR}/task7_results.json", "w"), indent=1)
        print(json.dumps(out, indent=1)); return out
    out["status"] = "COMPLETE"
    # per backbone (pooled over datasets; fold-cluster unit = (ds,fold))
    out["per_backbone"] = {}
    for bb in BACKBONES:
        fe = [_fold_entry(s) for (ds, b, f), s in cells.items() if b == bb]
        out["per_backbone"][bb] = _endpoints(fe, n_boot)
    # per (dataset, backbone)
    out["per_cell"] = {}
    for ds in DATASETS:
        for bb in BACKBONES:
            fe = [_fold_entry(s) for (d, b, f), s in cells.items() if d == ds and b == bb]
            out["per_cell"][f"{ds}/{bb}"] = _endpoints(fe, n_boot)
    # E7.5 probe-vs-exact (TSMNet): paired per fold on matched directions
    exact = _load("exact")
    fe5 = []
    for k, se in exact.items():
        sp = cells.get(k)
        if sp is None:
            continue
        tp = np.array([r["tau_ce_reliance"] for r in sp["directions"]])
        te = np.array([r["tau_ce_reliance"] for r in se["directions"]])
        n = min(len(tp), len(te))
        fe5.append({"tp": tp[:n], "te": te[:n], "absdiff": float(np.abs(tp[:n] - te[:n]).mean())})
    if fe5:
        out["E7.5_probe_vs_exact_TSMNet"] = {
            "n_folds": len(fe5),
            "corr_tau_probe_exact": _fold_boot(fe5, lambda F: _corr(np.concatenate([e["tp"] for e in F]),
                                                                    np.concatenate([e["te"] for e in F])), n_boot),
            "mean_abs_diff": _fold_boot(fe5, lambda F: float(np.mean([e["absdiff"] for e in F])), n_boot)}
    json.dump(out, open(f"{OUT_DIR}/task7_results.json", "w"), indent=1)
    # console table
    def f3(t):
        return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"
    print(f"\n{'cell':<26}{'λ_frac_sig':<22}{'τ−random':<24}{'|τ|':<20}{'corr(λ,τ)raw':<22}{'partial(λ,τ|E)':<22}")
    for name, e in {**{f"BACKBONE:{b}": out['per_backbone'][b] for b in BACKBONES}, **out["per_cell"]}.items():
        print(f"{name:<26}{f3(e['E7.1_frac_lambda_sig']):<22}{f3(e['E7.2_mean_tau_minus_random']):<24}"
              f"{f3(e['E7.2_mean_abs_tau']):<20}{f3(e['E7.3_corr_lambda_tau_RAW_confounded']):<22}"
              f"{f3(e['E7.3_partial_corr_lambda_tau_given_energyrank']):<22}")
    if "E7.5_probe_vs_exact_TSMNet" in out:
        e = out["E7.5_probe_vs_exact_TSMNet"]
        print(f"\nE7.5 TSMNet probe-vs-exact: corr={f3(e['corr_tau_probe_exact'])}  |Δτ|={f3(e['mean_abs_diff'])}")
    print(f"\n-> {OUT_DIR}/task7_results.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fleet", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--n_folds", type=int, default=12)
    ap.add_argument("--k_spec", type=int, default=16)
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--n_random", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()
    if a.fleet:
        fleet(a.n_folds, a.k_spec, a.n_perm, a.n_random, a.n_jobs)
    elif a.aggregate:
        aggregate(a.n_boot, a.n_folds)
    else:                                                      # default: single-dump probe
        for ds in ["Lee2019_MI"]:
            for bb, heads in [("EEGNet", ["probe"]), ("TSMNet", ["probe", "exact"])]:
                p = _dumps(ds, bb, n=1)[0]
                for h in heads:
                    s = run_one(p, head=h, k_spec=8, n_perm=10, n_random=10)
                    D = s["directions"]; lam = np.array([r["lambda_excess_over_null"] for r in D])
                    tau = np.array([r["tau_ce_reliance"] for r in D]); taur = np.array([r["tau_random_mean"] for r in D])
                    print(f"{ds}/{bb} head={h} firewall={s['firewall_passed']} replay={s['head_replay_verified']} "
                          f"{s['secs']}s | λtop={lam[:3].round(3)} |τ|={np.abs(tau).mean():.4f} "
                          f"USE>rand={int((tau>taur).sum())}/{len(D)}")
        print("TASK7_PROBE_DONE")
