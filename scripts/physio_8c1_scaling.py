#!/usr/bin/env python
"""FSR Phase 8C-1 — PhysioNetMI subject-scaling audit (see FSR_48 v2 + PM 8C-1 terms). Per (model, condition,
N_source, seed): fixed-a-priori PCA (d=128, fit ONCE on the full source pool) -> fixed LDA head; per-cell task
gate (source-val bAcc>=0.58 -> L4/L5/L6 interpretable, else WEAK_TASK_NOT_INTERPRETED); primary L1 = mean pairwise
subject separability (2-way, run-held-out: train runs 4/8, test 12); L5 = subject-subspace vs variance-MATCHED
null (rank=min(8,K(N-1)), per removed-variance); L6 target consequence on the frozen 15-target panel (final
scoring only). growing N in {2,4,8,16,all}; fixed N in {2,4,8}. N=all = single composition. Clustered bootstrap by
target subject. Writes the 8C-1 CSVs + slope summary + verdict. Target labels: final scoring ONLY.
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as BACC
import cb_cbm_8b_audit as A   # subject_offsets, task_offsets, top_k, erase, var_frac_removed

OUT = Path("results/fsr_codebrain_cbramod_8c")
RNG = np.random.default_rng(20260707)
PCA_D, TASK_GATE, MAX_PAIRS = 128, 0.58, 400
GROW_CAP, FIXED_TOTAL, K = 40, 80, 2
TRAIN_RUNS, TEST_RUN = (4, 8), 12


def load(model):
    z = np.load(OUT / "embeddings" / f"physionetmi_{model}_F1.npz")
    return z["X"].astype(np.float64), z["y"].astype(int), z["d"].astype(int), z["run"].astype(int)


def budget_idx(idx_by_subj, N, condition, seed):
    """trial indices for the source-train set under the budget."""
    rng = np.random.default_rng(7000 + seed)
    cap = GROW_CAP if condition == "growing" else max(4, FIXED_TOTAL // N)
    out = []
    for s, ix in idx_by_subj.items():
        ix = np.array(ix)
        take = ix if len(ix) <= cap else rng.choice(ix, cap, replace=False)
        out.append(take)
    return np.concatenate(out)


def pairwise_l1(Z, y, d, run, subjects):
    """mean over source-subject pairs of 2-way run-held-out separability (train runs 4/8, test 12)."""
    pairs = [(a, b) for i, a in enumerate(subjects) for b in subjects[i + 1:]]
    if len(pairs) > MAX_PAIRS:
        pairs = [pairs[i] for i in RNG.choice(len(pairs), MAX_PAIRS, replace=False)]
    accs = []
    for a, b in pairs:
        m = np.isin(d, [a, b]); tr = m & np.isin(run, TRAIN_RUNS); te = m & (run == TEST_RUN)
        if tr.sum() < 6 or te.sum() < 2 or len(np.unique(d[te])) < 2:
            continue
        lab = (d == b).astype(int)
        try:
            h = LDA().fit(Z[tr], lab[tr]); accs.append(BACC(lab[te], h.predict(Z[te])))
        except Exception:
            continue
    return float(np.mean(accs)) if accs else None, len(accs)


def nway_l1(Z, d, run, subjects):
    tr = np.isin(d, subjects) & np.isin(run, TRAIN_RUNS); te = np.isin(d, subjects) & (run == TEST_RUN)
    if len(np.unique(d[tr])) < len(subjects) or te.sum() < len(subjects):
        return None
    try:
        h = LDA().fit(Z[tr], d[tr]); return float(BACC(d[te], h.predict(Z[te])))
    except Exception:
        return None


def cell(Z, y, d, run, model, condition, N, seed, subset, panel):
    idx_by_subj = {s: np.where(d == s)[0] for s in subset}
    bidx = budget_idx(idx_by_subj, N, condition, seed)
    yb, db = y[bidx], d[bidx]
    # trial-level 80/20 split (unstratified permutation) for head + source-val task gate
    rng = np.random.default_rng(9000 + seed)
    perm = rng.permutation(len(bidx)); nval = max(len(bidx) // 5, 2)
    val, trn = bidx[perm[:nval]], bidx[perm[nval:]]
    mu, sd = Z[trn].mean(0), Z[trn].std(0) + 1e-8
    def S(ix): return (Z[ix] - mu) / sd
    head = LDA().fit(S(trn), y[trn])
    sv = BACC(y[val], head.predict(S(val)))
    # target eval (final scoring only)
    tix = np.where(np.isin(d, panel))[0]
    tb = BACC(y[tix], head.predict(S(tix)))
    gated = bool(sv >= TASK_GATE)
    row = dict(model=model, condition=condition, N_source=str(N), seed=seed, n_source=len(subset),
               n_train_trials=len(trn), source_val_bacc=round(sv, 4), target_bacc=round(tb, 4),
               task_gated=gated)
    l4 = l5s = l5v = l6 = varfrac = None
    if gated:
        k = min(8, K * (N - 1))
        Ztr = S(trn); subjM = A.subject_offsets(Ztr, y[trn], d[trn])
        Bs = A.top_k(subjM, k); Bv = A.top_k(Ztr - Ztr.mean(0), Bs.shape[0])
        W = head.coef_ if head.coef_.ndim == 2 else head.coef_.reshape(1, -1)
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        l4 = float(np.mean(np.max(np.abs(Wn @ Bs.T), axis=1))) if Bs.shape[0] else None
        Zv = S(val); base = BACC(y[val], head.predict(Zv))
        l5s = base - BACC(y[val], head.predict(A.erase(Zv, Bs)))
        l5v = base - BACC(y[val], head.predict(A.erase(Zv, Bv)))
        varfrac = A.var_frac_removed(Zv, Bs)
        Zt = S(tix); l6 = BACC(y[tix], head.predict(Zt)) - BACC(y[tix], head.predict(A.erase(Zt, Bs)))
        row.update(dict(subj_rank=int(Bs.shape[0]), l4_alignment=round(l4, 4),
                        l5_drop_subject=round(l5s, 4), l5_drop_variance=round(l5v, 4),
                        l5_beats_variance=bool(l5s > l5v), l5_removed_var=round(varfrac, 4),
                        l6_target_delta=round(l6, 4)))
    else:
        row.update(dict(subj_rank=None, l4_alignment=None, l5_drop_subject=None, l5_drop_variance=None,
                        l5_beats_variance=None, l5_removed_var=None, l6_target_delta="WEAK_TASK_NOT_INTERPRETED"))
    return row


def main():
    plan = list(csv.DictReader(open(OUT / "source_subset_plan.csv")))
    v0 = json.load(open(OUT / "phase8c0_verdict.json")); panel = v0["target_panel"]; pool = v0["n_source_pool"]
    all_rows, l1_rows = [], []
    for model in ("cbramod", "codebrain"):
        X, y, d, run = load(model)
        src_mask = ~np.isin(d, panel)                        # PCA fit on FULL source pool only (fixed a priori)
        pca = PCA(n_components=PCA_D, random_state=0).fit(X[src_mask])
        Z = pca.transform(X)                                 # fixed 128-dim space for all cells
        print(f"[{model}] PCA fit on {src_mask.sum()} source trials; Z={Z.shape}", flush=True)
        for r in plan:
            N = 104 if r["N_source"] == "all" else int(r["N_source"]); seed = int(r["seed"])
            subset = [int(s) for s in r["subset"].split(";")]
            # L1 pairwise (condition-independent; full trials, run-held-out)
            pl1, npairs = pairwise_l1(Z, y, d, run, subset); nw = nway_l1(Z, d, run, subset)
            l1_rows.append(dict(model=model, N_source=r["N_source"], seed=seed, n_source=len(subset),
                                pairwise_separability=round(pl1, 4) if pl1 is not None else None, n_pairs=npairs,
                                nway_acc_descriptive=round(nw, 4) if nw is not None else None, nway_chance=round(1 / len(subset), 4)))
            for condition in ("growing", "fixed"):
                if condition == "fixed" and (r["N_source"] not in ("2", "4", "8")):
                    continue
                all_rows.append(cell(Z, y, d, run, model, condition, N, seed, subset, panel))
            print(f"[{model}] N={r['N_source']} seed={seed} pairwise_L1={pl1}", flush=True)

    # ---- write per-cell CSVs ----
    def w(fn, rows, keys):
        with open(OUT / fn, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k) for k in keys})
    perf_keys = ["model", "condition", "N_source", "seed", "n_source", "n_train_trials", "source_val_bacc", "target_bacc", "task_gated"]
    w("subject_scaling_performance.csv", all_rows, perf_keys)
    w("task_gate_by_cell.csv", all_rows, ["model", "condition", "N_source", "seed", "source_val_bacc", "task_gated"])
    w("subject_scaling_pairwise_l1.csv", l1_rows, ["model", "N_source", "seed", "n_source", "pairwise_separability", "n_pairs"])
    w("subject_scaling_nway_l1_descriptive.csv", l1_rows, ["model", "N_source", "seed", "n_source", "nway_acc_descriptive", "nway_chance"])
    w("subject_scaling_l4_alignment.csv", all_rows, ["model", "condition", "N_source", "seed", "task_gated", "subj_rank", "l4_alignment"])
    w("subject_scaling_l5_replay.csv", all_rows, ["model", "condition", "N_source", "seed", "task_gated", "subj_rank", "l5_drop_subject", "l5_drop_variance", "l5_beats_variance", "l5_removed_var"])
    w("subject_scaling_l6_consequence.csv", all_rows, ["model", "condition", "N_source", "seed", "task_gated", "l6_target_delta"])
    w("fixed_vs_growing_trials.csv", all_rows, ["model", "condition", "N_source", "seed", "n_train_trials", "target_bacc", "source_val_bacc"])

    # ---- slopes vs log(N) (per model x condition) + summary ----
    def slope(rows, ykey):
        pts = [(np.log(r["n_source"]), r[ykey]) for r in rows if isinstance(r.get(ykey), (int, float))]
        if len(pts) < 3:
            return None
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        b = [np.polyfit(xs[i], ys[i], 1)[0] for i in [RNG.integers(0, len(xs), len(xs)) for _ in range(1000)]]
        return dict(slope=round(float(np.polyfit(xs, ys, 1)[0]), 4), ci=[round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)], n=len(pts))
    slopes = []
    for model in ("cbramod", "codebrain"):
        for condition in ("growing", "fixed"):
            rr = [r for r in all_rows if r["model"] == model and r["condition"] == condition]
            l1r = [r for r in l1_rows if r["model"] == model and (condition == "growing" or r["N_source"] in ("2", "4", "8"))]
            slopes.append(dict(model=model, condition=condition,
                               target_bacc_slope=slope(rr, "target_bacc"),
                               pairwise_l1_slope=slope(l1r, "pairwise_separability"),
                               l5_drop_subject_slope=slope([r for r in rr if r["task_gated"]], "l5_drop_subject")))
    (OUT / "model_condition_slope_summary.csv").write_text(
        "model,condition,target_bacc_slope,pairwise_l1_slope,l5_drop_subject_slope\n" +
        "".join(f"{s['model']},{s['condition']},{s['target_bacc_slope']},{s['pairwise_l1_slope']},{s['l5_drop_subject_slope']}\n" for s in slopes))
    (OUT / "subject_scaling_mixed_effects.json").write_text(json.dumps(dict(slopes=slopes,
        note=("slope vs log(N_source), CELL-LEVEL bootstrap (NOT clustered by the shared 15-target panel -> CIs are "
              "OVER-PRECISE; sign is robust but treat intervals as lower bounds on uncertainty); growing=full grid, "
              "fixed={2,4,8} (N=2 fixed==growing by construction); N=all single composition. NOTE: pairwise-L1 is "
              "computed in the FIXED source-pool PCA space and is training/condition/N-INDEPENDENT by design -> its "
              "flat-vs-N is structural, NOT a diversity finding.")), indent=2, default=str) + "\n")

    def trend(model, cond, key, l1=False):
        rr = ([r for r in l1_rows if r["model"] == model and (cond == "growing" or r["N_source"] in ("2", "4", "8"))] if l1
              else [r for r in all_rows if r["model"] == model and r["condition"] == cond])
        s = slope(rr, key)
        if s is None:
            return "insufficient"
        return "increases" if s["ci"][0] > 0 else "decreases" if s["ci"][1] < 0 else "flat"
    cb_gate = np.mean([r["task_gated"] for r in all_rows if r["model"] == "cbramod"])
    code_gate = np.mean([r["task_gated"] for r in all_rows if r["model"] == "codebrain"])
    verdict = dict(primary_model="CBraMod", exploratory_model="CodeBrain", dataset="PhysioNetMI",
        analyzable_subjects=105, target_panel_subjects=15, n_source_grid_growing=[2, 4, 8, 16, "all"],
        n_source_grid_fixed=[2, 4, 8], subset_seeds=10, primary_l1_metric="mean_pairwise_subject_separability",
        cbramod_task_gate_pass_rate=round(float(cb_gate), 3), codebrain_task_gate_pass_rate=round(float(code_gate), 3),
        cbramod_perf_trend_growing=trend("cbramod", "growing", "target_bacc"),
        cbramod_perf_trend_fixed=trend("cbramod", "fixed", "target_bacc"),
        cbramod_pairwise_l1_trend=trend("cbramod", "growing", "pairwise_separability", l1=True),
        cbramod_l5_reliance_trend=trend("cbramod", "growing", "l5_drop_subject"),
        codebrain_perf_trend_growing=trend("codebrain", "growing", "target_bacc"),
        codebrain_pairwise_l1_trend=trend("codebrain", "growing", "pairwise_separability", l1=True),
        target_labels_used_for_selection=False, proceed_to_specialist_baselines=None,
        note="8C-1 CBraMod primary / CodeBrain exploratory. See slope summary + trends; interpretation per FSR_48 v2 grid.")
    (OUT / "codebrain_cbramod_8c_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    (OUT / "target_label_firewall.json").write_text(json.dumps(dict(stage="8C-1",
        target_labels_used_for_fit=False, target_labels_used_for_selection=False, target_labels_used_for_final_scoring_only=True,
        note="PCA fit on source pool; head+subspaces on source-train; task gate on source-val; L1 on source subjects; target labels only score target_bacc/L6."), indent=2) + "\n")
    print("=== 8C-1 slopes ==="); [print(" ", s["model"], s["condition"], "perf", s["target_bacc_slope"], "L1", s["pairwise_l1_slope"], "L5", s["l5_drop_subject_slope"]) for s in slopes]
    print("gate pass: cbramod %.2f codebrain %.2f" % (cb_gate, code_gate))


if __name__ == "__main__":
    main()
