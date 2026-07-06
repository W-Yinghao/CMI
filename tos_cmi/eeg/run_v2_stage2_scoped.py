"""V2 Stage-2 SCOPED driver (per V2_STAGE1B_VERDICT.md): World A on EEGNet only (clean ceiling robustness
across source_subject_counts + seeds + folds); Worlds B/C on EEGNet+TSMNet (refusal / no-false-accept
robustness across source_subject_counts). TSMNet World A is SKIPPED (Stage-1B verdict carried forward).
Reuses the tested run_v2_certificate internals; gate thresholds FROZEN. Encodes the Stage-2 stop conditions.

  python -m tos_cmi.eeg.run_v2_stage2_scoped [--folds 15] [--tag stage2_scoped]
Scope (world->backbones, datasets, seeds, source_subject_counts, folds, alphas, interventions) is read from
tos_cmi/eeg/configs/v2_stage2_scoped.yaml. Writes tos_cmi/results/method_deepen/v2_stage2/v2_<tag>_*.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import numpy as np
from joblib import Parallel, delayed

from tos_cmi.eeg.run_v2_certificate import (_one, _dumps, aggregate, build_manifest, _nuisance_m,
                                           INTERVENTIONS, N_JOBS)
from tos_cmi.eeg.report_v2 import _worldA_taxonomy, PRINCIPLED, DIAGNOSTIC

CONFIG = "tos_cmi/eeg/configs/v2_stage2_scoped.yaml"
OUT = "tos_cmi/results/method_deepen/v2_stage2"


def _cfg():
    import yaml
    return yaml.safe_load(open(CONFIG))


def target_leak_structural_check(datasets, nmode, nfrac, nmin):
    """Pre-run STRUCTURAL gate for stop-condition #2 (target labels used only in the audit). Enforces:
      (a) oracle diagnostic interventions are NOT in the deployable set (never a method result);
      (b) permuting the target labels used for SCORING does not change ANY source-only gate signal
          (task_drop / source-LOSO benefit / domain_gain) -- i.e. y_target enters only the final audit.
    Raises AssertionError (-> job halts) on failure; prints TARGET_LEAK_STRUCTURAL_PASS on success."""
    from tos_cmi.eeg.v2_worlds import DEPLOYABLE as DEP, DIAGNOSTIC as DIAG, FACTORIES
    from tos_cmi.eeg.run_v2_certificate import eval_v2
    from tos_cmi.eeg.semi_synthetic_real_latent import inject
    from tos_cmi.eeg.erasure_baselines import _ids
    assert set(DIAG).isdisjoint(set(DEP)), "oracle diagnostic intervention leaked into the deployable set"
    ps = _dumps(datasets[0], "EEGNet", 0, 1)
    d = np.load(ps[0], allow_pickle=True)
    Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
    Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
    subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
    m = _nuisance_m(Zs.shape[1], nmode, nfrac, nmin)
    inj = inject("A", Zs, ys, subj, Zt, yt, seed=0, m=m)
    F = FACTORIES["leace_baseline"]
    s1 = eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], yt, n_cls, F, 0, 6)
    ytp = np.random.default_rng(1).permutation(yt)
    s2 = eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], ytp, n_cls, F, 0, 6)
    assert abs(s1["task_drop"] - s2["task_drop"]) < 1e-9, "task_drop changed under target-label permutation"
    assert abs(s1["domain_gain"] - s2["domain_gain"]) < 1e-9, "domain_gain changed under target-label permutation"
    assert np.allclose(s1["benefit"], s2["benefit"]), "source-LOSO benefit changed under target-label permutation"
    assert abs(s1["tgt_bacc_full"] - s2["tgt_bacc_full"]) > 1e-6, "sanity: the target AUDIT should change"
    print("TARGET_LEAK_STRUCTURAL_PASS", flush=True)


def stop_conditions(summary, manifest, bt):
    """Return (halt, findings). Encodes the 7 Stage-2 stop conditions."""
    f = []
    dep = [v for v in summary.values() if v["deployable"]]
    # 1 any false ACCEPT (gate ACCEPT a non-target-beneficial cell)
    fa = [v for v in dep if v["gate_action"] == "ACCEPT" and not (v["is_safe"] and v["dtgt_bacc_lo"] > bt)]
    f.append(("1 false-accept", len(fa)))
    # 3 EEGNet World-A clean positives per n_source (flag if any n_source has 0)
    A_eeg = [v for v in summary.values() if v["world"] == "A" and v["backbone"] == "EEGNet"]
    per_ns = {}
    for ns in sorted(set(v["n_source"] for v in A_eeg)):
        cells = [v for v in A_eeg if v["n_source"] == ns]
        tax = _worldA_taxonomy(cells, bt)
        per_ns[ns] = sum(t["n_clean"] for t in tax)
    f.append(("3 EEGNet World-A clean positives by n_source", per_ns))
    # 4 random-k reproduces EEGNet World-A gain
    rnd_repro = [v for v in A_eeg if v["intervention"] == "random_k" and v["dtgt_bacc_lo"] > bt]
    f.append(("4 random reproduces (EEGNet World-A random_k LCB>bt)", len(rnd_repro)))
    # 5 World-B unsafe accept
    B_acc = [v for v in summary.values() if v["world"] == "B" and v["deployable"] and v["gate_action"] == "ACCEPT"]
    f.append(("5 World-B unsafe accept", len(B_acc)))
    # 6 World-C principled accept
    C_acc = [v for v in summary.values() if v["world"] == "C" and v["intervention"] in PRINCIPLED and v["gate_action"] == "ACCEPT"]
    f.append(("6 World-C principled accept", len(C_acc)))
    # 7 TSMNet B/C degeneracy >20%
    tsm_bc_bad = [k for k, v in manifest.items() if v["backbone"] == "TSMNet" and v["frac_skipped"] > 0.20]
    f.append(("7 TSMNet B/C degeneracy>20%", len(tsm_bc_bad)))
    halt = bool(len(fa) or len(rnd_repro) or len(B_acc) or len(C_acc) or len(tsm_bc_bad)
                or any(c == 0 for c in per_ns.values()))
    return halt, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)          # override config (validation/tuning)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--n-source", nargs="+", default=None)
    ap.add_argument("--alphas", nargs="+", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print task count/composition, do not run")
    ap.add_argument("--tag", default="stage2_scoped")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    C = _cfg()
    outdir = a.outdir
    datasets = a.datasets or C["datasets"]; seeds = a.seeds or C["seeds"]
    nsrc = [str(x) for x in (a.n_source or C["source_subject_counts"])]
    folds = a.folds if a.folds is not None else int(C.get("folds", 15))
    alphas = a.alphas or [float(x) for x in C["alpha_grid"]]; n_pseudo = int(C["n_pseudo"])
    interv = C.get("interventions", INTERVENTIONS)
    nmode = C.get("nuisance_dim_mode", "fraction_of_z_dim"); nfrac = float(C.get("nuisance_fraction", 0.20))
    nmin = int(C.get("nuisance_dim_min", 4))
    safety_eps = float(C["thresholds"]["safety_reject_task_drop_ucb"])
    benefit_thr = float(C["thresholds"]["benefit_accept_lcb"])
    world_bb = {"A": C["world_A"]["include_backbones"], "B": C["world_B"]["include_backbones"],
                "C": C["world_C"]["include_backbones"]}
    cfg_hash = hashlib.sha256(open(CONFIG).read().encode()).hexdigest()[:12]
    tasks = []
    for w in ["A", "B", "C"]:
        for ds in datasets:
            for bb in world_bb[w]:
                for sd in seeds:
                    for p in _dumps(ds, bb, sd, folds):
                        for ns in nsrc:
                            for al in alphas:
                                for iv in interv:
                                    tasks.append((w, ds, bb, sd, p, ns, al, iv))
    from collections import Counter
    comp = Counter((t[0], t[2]) for t in tasks)   # (world, backbone) composition
    print("V2 Stage-2 scoped %s: %d tasks (n_jobs=%d), config %s | World A backbones=%s ; B/C backbones=%s ; "
          "n_source=%s folds=%d seeds=%s ; thresholds safety<=%.3f benefit>%.3f (FROZEN)"
          % (a.tag, len(tasks), N_JOBS, cfg_hash, world_bb["A"], world_bb["B"], nsrc, folds, seeds,
             safety_eps, benefit_thr), flush=True)
    print("  task composition (world,backbone): %s" % dict(sorted(comp.items())), flush=True)
    if a.dry_run:
        print("DRY_RUN (no execution). tasks=%d" % len(tasks)); return
    # stop-condition #2: target-leak structural gate (HALTs the run on failure)
    target_leak_structural_check(datasets, nmode, nfrac, nmin)
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_one)(w, ds, bb, sd, p, ns, al, iv, 0.15, 1.0, nmin, nmode, nfrac, 0.1, n_pseudo)
        for (w, ds, bb, sd, p, ns, al, iv) in tasks)
    ndeg = sum(1 for r in rows if r.get("degenerate")); nfail = sum(1 for r in rows if r.get("fail"))
    exp = {}
    for w in ["A", "B", "C"]:
        for ds in datasets:
            for bb in world_bb[w]:
                for sd in seeds:
                    exp[(ds, bb, sd)] = len(_dumps(ds, bb, sd, folds))
    manifest = build_manifest(rows, exp)
    summary = aggregate(rows, safety_eps, benefit_thr)
    os.makedirs(outdir, exist_ok=True)
    cols = ["world", "dataset", "backbone", "seed", "fold", "n_source", "alpha", "intervention",
            "ground_truth", "z_dim", "m_eff", "degenerate", "cond", "skip_reason", "src_task_full",
            "src_task_eras", "task_drop", "z_full", "z_eras", "domain_gain", "tgt_bacc_full",
            "tgt_bacc_eras", "router_acc"]
    with open("%s/v2_%s_rows.csv" % (outdir, a.tag), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump({"config_hash": cfg_hash, "thresholds": {"safety_eps": safety_eps, "benefit_lcb": benefit_thr},
               "params": {"nuisance_mode": nmode, "nuisance_fraction": nfrac, "nuisance_dim_min": nmin,
                          "m_EEGNet": _nuisance_m(16, nmode, nfrac, nmin), "n_pseudo": n_pseudo, "phi": 0.15,
                          "beta": 1.0, "folds": folds, "source_subject_counts": nsrc},
               "world_backbones": world_bb, "n_tasks": len(rows), "n_fail": nfail, "n_degenerate": ndeg,
               "summary": summary},
              open("%s/v2_%s_summary.json" % (outdir, a.tag), "w"), indent=1)
    json.dump({"config_hash": cfg_hash, "manifest": manifest},
              open("%s/v2_%s_manifest.json" % (outdir, a.tag), "w"), indent=1)
    # per-n_source Stage-2 view
    print("\n=== World A / EEGNet clean ceiling by source_subject_count ===")
    for ns in nsrc:
        cells = [v for v in summary.values() if v["world"] == "A" and v["backbone"] == "EEGNet" and v["n_source"] == ns]
        tax = _worldA_taxonomy(cells, benefit_thr)
        nclean = sum(t["n_clean"] for t in tax); nacc = sum(t["n_accept"] for t in tax)
        print("  n_source=%-4s : clean positives %d ; principled ACCEPT %d ; status %s"
              % (ns, nclean, nacc, ",".join(t["status"] for t in tax)))
    print("\n=== World B/C refusal by backbone x source_subject_count ===")
    for wk in ["B", "C"]:
        for bb in world_bb[wk]:
            for ns in nsrc:
                cells = [v for v in summary.values() if v["world"] == wk and v["backbone"] == bb and v["n_source"] == ns
                         and v["intervention"] in PRINCIPLED]
                nacc = sum(1 for v in cells if v["gate_action"] == "ACCEPT")
                hidg = sum(1 for v in cells if v["domain_gain"] > 0.05 and not (v["is_safe"] and v["dtgt_bacc_lo"] > benefit_thr))
                print("  World %s %-7s n_source=%-4s : principled ACCEPT %d ; high-dg-useless %d" % (wk, bb, ns, nacc, hidg))
    halt, findings = stop_conditions(summary, manifest, benefit_thr)
    print("\n=== stop conditions ===")
    print("  %-45s : %s" % ("2 target-leak (PRE-RUN structural gate)", "TARGET_LEAK_STRUCTURAL_PASS (asserted before run)"))
    for name, val in findings:
        print("  %-45s : %s" % (name, val))
    print("  degenerate %d / fail %d of %d tasks" % (ndeg, nfail, len(rows)))
    print("  STAGE2_HALT" if halt else "  STAGE2_CLEAN")
    print("V2_STAGE2_%s_DONE" % a.tag.upper())


if __name__ == "__main__":
    main()
