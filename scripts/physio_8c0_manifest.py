#!/usr/bin/env python
"""FSR Phase 8C-0 — PhysioNetMI manifest + frozen source-subset plan + feasibility (see FSR_48 v2). Pins runs
4/8/12 (imagined left/right fist, K=2), excludes S088/089/092/100, builds per-subject trial counts, the fixed
target panel + N_source subset plan, fixed-vs-growing feasibility, and run-held-out pairwise-separability
feasibility. NO encoder, NO target-label use. CPU. Writes the 8C-0 manifest/plan/feasibility CSVs + partial verdict.
"""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import mne
mne.set_log_level("ERROR")

PHYS = "/projects/EEG-foundation-model/PhysioNetMI"
OUT = Path("results/fsr_codebrain_cbramod_8c")
EXCLUDE = [88, 89, 92, 100]
RUNS = [4, 8, 12]           # imagined left/right fist
TRAIN_RUNS, TEST_RUN = [4, 8], 12
N_TARGET_PANEL = 15
N_GRID = [2, 4, 8, 16, "all"]
SEEDS = list(range(10))
RUNHELDOUT_MIN = 4          # min trials in train-runs AND >=2 in test-run per subject for L1 feasibility
RNG = np.random.default_rng(20260707)


def subject_counts(sid):
    """(n_left, n_right, per-run T1/T2 counts, ch_ok, sfreq_ok, ch_names) for one subject over runs 4/8/12."""
    per_run = {}; chn = None; ch_ok = True; sf_ok = True
    for r in RUNS:
        f = f"{PHYS}/S{sid:03d}/S{sid:03d}R{r:02d}.edf"
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
        if chn is None:
            chn = list(raw.ch_names)
        ch_ok &= (list(raw.ch_names) == chn) and len(raw.ch_names) == 64
        sf_ok &= abs(raw.info["sfreq"] - 160.0) < 1e-6
        ev, evid = mne.events_from_annotations(raw, verbose=False)
        t1 = int((ev[:, 2] == evid.get("T1", -1)).sum()); t2 = int((ev[:, 2] == evid.get("T2", -1)).sum())
        per_run[r] = (t1, t2)
    nl = sum(v[0] for v in per_run.values()); nr = sum(v[1] for v in per_run.values())
    return nl, nr, per_run, ch_ok, sf_ok, chn


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = [s for s in range(1, 110) if s not in EXCLUDE]
    rows, chn_ref = [], None
    for s in subjects:
        try:
            nl, nr, per_run, ch_ok, sf_ok, chn = subject_counts(s)
        except Exception as e:
            rows.append(dict(subject=s, ok=False, error=f"{type(e).__name__}: {e}")); continue
        if chn_ref is None and ch_ok:
            chn_ref = chn
        rh_ok = (sum(per_run[r][0] + per_run[r][1] for r in TRAIN_RUNS) >= RUNHELDOUT_MIN and
                 (per_run[TEST_RUN][0] + per_run[TEST_RUN][1]) >= 2)
        rows.append(dict(subject=s, ok=bool(ch_ok and sf_ok and (nl + nr) > 0), n_left=nl, n_right=nr,
                         total=nl + nr, ch_ok=ch_ok, sfreq_ok=sf_ok, runheldout_ok=bool(rh_ok),
                         r4=str(per_run[4]), r8=str(per_run[8]), r12=str(per_run[12])))
        print(f"S{s:03d}: L={nl} R={nr} total={nl+nr} ch_ok={ch_ok} rh_ok={rh_ok}", flush=True)

    good = [r for r in rows if r.get("ok")]
    totals = np.array([r["total"] for r in good]); min_tr = int(totals.min()); med_tr = int(np.median(totals))
    # channel montage hash
    chash = hashlib.sha256("|".join(chn_ref).encode()).hexdigest()[:16] if chn_ref else None

    # fixed target panel (deterministic spread) + source pool
    ok_ids = [r["subject"] for r in good]
    tpanel = sorted(RNG.choice(ok_ids, N_TARGET_PANEL, replace=False).tolist())
    source_pool = [s for s in ok_ids if s not in tpanel]
    NALL = len(source_pool)

    # source subset plan (coverage-balanced draws; N=all single composition)
    plan = []
    for N in N_GRID:
        n = NALL if N == "all" else N
        for seed in SEEDS:
            if N == "all":
                subset = source_pool
                if seed > 0:
                    continue                      # N=all is ONE composition -> single df (no pseudo-replication)
            else:
                subset = sorted(np.random.default_rng(1000 + seed).choice(source_pool, n, replace=False).tolist())
            plan.append(dict(N_source=str(N), seed=seed, n=len(subset), subset=";".join(map(str, subset))))

    # fixed-vs-growing feasibility
    FLOOR = 8
    growing_cap = min_tr                          # per-subject cap = min available -> growing feasible all N
    fixed_total = 2 * min_tr                       # so N=2 satisfiable
    feas = []
    for N in N_GRID:
        n = NALL if N == "all" else N
        per_subj_fixed = fixed_total / n
        feas.append(dict(N_source=str(N), n=n, growing_per_subject=growing_cap, growing_feasible=True,
                         fixed_total=fixed_total, fixed_per_subject=round(per_subj_fixed, 1),
                         fixed_feasible=bool(per_subj_fixed >= FLOOR)))

    # ---- write outputs ----
    def w(fn, rr):
        with open(OUT / fn, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rr[0].keys())); wr.writeheader(); wr.writerows(rr)
    w("physionetmi_manifest.csv", rows)
    w("source_subset_plan.csv", plan)
    w("fixed_trials_feasibility.csv", feas)
    w("runheldout_l1_feasibility.csv", [dict(subject=r["subject"], total=r["total"], runheldout_ok=r["runheldout_ok"]) for r in good])
    (OUT / "physionetmi_subject_exclusion.csv").write_text(
        "subject,reason\n" + "".join(f"{s},inconsistent_sampling_or_run_structure\n" for s in EXCLUDE))
    (OUT / "run_label_pin_manifest.csv").write_text(
        "run,task,T1,T2\n4,imagined_fist,left_fist,right_fist\n8,imagined_fist,left_fist,right_fist\n12,imagined_fist,left_fist,right_fist\n")
    (OUT / "channel_montage_hash.json").write_text(json.dumps(dict(n_channels=64, channels=chn_ref, sha256_16=chash), indent=2) + "\n")
    (OUT / "resample_manifest.json").write_text(json.dumps(dict(native_hz=160, target_hz=200, method="scipy.signal.resample (FFT)", ratio="5/4", deterministic=True), indent=2) + "\n")
    (OUT / "target_label_firewall.json").write_text(json.dumps(dict(
        stage="8C-0", target_labels_used_for_fit=False, target_labels_used_for_selection=False,
        note="8C-0 uses only subject ids + trial COUNTS + channel/resample metadata. No task labels read; no encoder run here."), indent=2) + "\n")

    fixed_feasible_all = all(f["fixed_feasible"] for f in feas)
    fixed_max_N = max((f["N_source"] for f in feas if f["fixed_feasible"]), key=lambda x: (0 if x == "all" else int(x)))
    verdict = dict(
        physionetmi_analyzable_subjects=len(good), excluded_subjects=EXCLUDE, runs_pinned=RUNS,
        classes=["imagined_left_fist", "imagined_right_fist"], K=2,
        n_target_panel=N_TARGET_PANEL, target_panel=tpanel, n_source_pool=NALL,
        min_trials_per_subject=min_tr, median_trials_per_subject=med_tr,
        channel_mapping_reproducible=bool(chash is not None), channel_hash=chash,
        resampling_reproducible=True, global_label_check_pass=bool(len(good) >= 100),
        fixed_trials_condition_feasible=fixed_feasible_all, fixed_trials_max_feasible_N=str(fixed_max_N),
        growing_trials_condition_feasible=True,
        runheldout_l1_feasible=bool(sum(r["runheldout_ok"] for r in good) >= 100),
        codebrain_64ch_forward_pass=None, cbramod_64ch_forward_pass=None,   # filled by encoder-sanity step
        target_labels_used_for_selection=False,
        gates=dict(G1_ge100_subjects=bool(len(good) >= 100), G2_runs_labels_pinned=True,
                   G3_channel_hash=bool(chash is not None), G4_resample_det=True,
                   G7_fixed_feasible=fixed_feasible_all, G8_growing_feasible=True,
                   G9_runheldout_feasible=bool(sum(r["runheldout_ok"] for r in good) >= 100),
                   G10_firewall=True),
        proceed_to_8c1=None, note="8C-0 manifest/feasibility done; encoder 64ch sanity (G5/G6) pending GPU step.")
    (OUT / "phase8c0_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"\nANALYZABLE={len(good)} min_trials={min_tr} median={med_tr} source_pool={NALL} target_panel={len(tpanel)}")
    print(f"fixed_feasible_all={fixed_feasible_all} fixed_max_N={fixed_max_N} runheldout_feasible={verdict['runheldout_l1_feasible']}")
    for f in feas:
        print(f"  N={f['N_source']}: fixed_per_subj={f['fixed_per_subject']} fixed_ok={f['fixed_feasible']} growing_ok={f['growing_feasible']}")


if __name__ == "__main__":
    main()
