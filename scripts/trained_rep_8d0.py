#!/usr/bin/env python
"""FSR Phase 8D-0 — metric-power gate (see FSR_51 v2). Proves the bottleneck-code L1 metric CAN move: train a
subject-ERASE positive-control adapter (A1-SE: bottleneck z'=z+W2 relu(W1 z), task head + adversarial subject
classifier via gradient reversal on h=relu(W1 z)) and measure mean pairwise subject separability on h for source
subjects HELD OUT from adapter training, vs A0 (frozen z). CBraMod, N_source=8, subset_seeds=3, train_seeds=5.
PASS iff held-out L1 on h drops vs A0 by >=0.10 abs OR >=20% rel, CI_lo>0, source-val task not collapsed. No target
labels (source-only; PhysioNetMI 15-target panel excluded entirely). Writes trained_rep_metric_power_gate.csv."""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as BACC

OUT = Path("results/fsr_trained_rep_scaling")
C8 = Path("results/fsr_codebrain_cbramod_8c")
PCA_D, BOTTLENECK, N_SOURCE = 128, 16, 8
SUBSET_SEEDS, TRAIN_SEEDS = [0, 1, 2], [0, 1, 2, 3, 4]
EPOCHS, LR, LAMBDA_ADV = 120, 1e-3, 1.0
TRAIN_RUNS, TEST_RUN = (4, 8), 12


def pairwise_l1(H, y, d, run, subjects, rng):
    pairs = [(a, b) for i, a in enumerate(subjects) for b in subjects[i + 1:]]
    accs = []
    for a, b in pairs:
        m = np.isin(d, [a, b]); tr = m & np.isin(run, TRAIN_RUNS); te = m & (run == TEST_RUN)
        if tr.sum() < 6 or te.sum() < 2 or len(np.unique(d[te])) < 2:
            continue
        lab = (d == b).astype(int)
        try:
            accs.append(BACC(lab[te], LDA().fit(H[tr], lab[tr]).predict(H[te])))
        except Exception:
            continue
    return float(np.mean(accs)) if accs else None, len(accs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch, torch.nn as nn
    torch.use_deterministic_algorithms(True, warn_only=True)
    z = np.load(C8 / "embeddings" / "physionetmi_cbramod_F1.npz")
    X, y, d, run = z["X"].astype(np.float64), z["y"].astype(int), z["d"].astype(int), z["run"].astype(int)
    panel = json.load(open(C8 / "phase8c0_verdict.json"))["target_panel"]
    pool = sorted(set(int(s) for s in np.unique(d)) - set(panel))              # 90 source subjects
    src = ~np.isin(d, panel)
    pca = PCA(n_components=PCA_D, random_state=0).fit(X[src]); Z = pca.transform(X).astype(np.float32)

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lamb): ctx.lamb = lamb; return x.view_as(x)
        @staticmethod
        def backward(ctx, g): return -ctx.lamb * g, None

    rows = []
    for ss in SUBSET_SEEDS:
        rng = np.random.default_rng(5000 + ss)
        subset = sorted(rng.choice(pool, N_SOURCE, replace=False).tolist())
        rng.shuffle(subset); atr_subj, l1_subj = sorted(subset[:4]), sorted(subset[4:])   # disjoint: 4 train / 4 L1-eval
        atr = np.isin(d, atr_subj); loc = {s: i for i, s in enumerate(atr_subj)}
        dloc = np.array([loc.get(s, -1) for s in d])
        # source-val split within adapter-train (trial-level 80/20)
        idx = np.where(atr)[0]; rng.shuffle(idx); nval = max(len(idx) // 5, 4)
        vmask = np.zeros(len(d), bool); vmask[idx[:nval]] = True
        tmask = atr & ~vmask
        # A0 frozen L1 on z (held-out subjects)
        a0_ho, _ = pairwise_l1(Z, y, d, run, l1_subj, rng)
        a0_in, _ = pairwise_l1(Z, y, d, run, atr_subj, rng)
        for ts in TRAIN_SEEDS:
            torch.manual_seed(1000 + ts); np.random.seed(1000 + ts)
            Zt = torch.tensor(Z)
            W1 = nn.Sequential(nn.Linear(PCA_D, BOTTLENECK), nn.ReLU())
            W2 = nn.Linear(BOTTLENECK, PCA_D); ln = nn.LayerNorm(PCA_D)
            head = nn.Linear(PCA_D, int(y.max()) + 1); adv = nn.Linear(BOTTLENECK, len(atr_subj))
            params = list(W1.parameters()) + list(W2.parameters()) + list(ln.parameters()) + list(head.parameters()) + list(adv.parameters())
            opt = torch.optim.Adam(params, lr=LR)
            yt = torch.tensor(y); dt = torch.tensor(np.where(dloc < 0, 0, dloc))
            tr_i = torch.tensor(np.where(tmask)[0]); ce = nn.CrossEntropyLoss()
            for ep in range(EPOCHS):
                opt.zero_grad()
                h = W1(Zt[tr_i]); zp = ln(Zt[tr_i] + W2(h))
                loss = ce(head(zp), yt[tr_i]) + LAMBDA_ADV * ce(adv(GRL.apply(h, 1.0)), dt[tr_i])
                loss.backward(); opt.step()
            with torch.no_grad():
                H = W1(Zt).numpy()                                  # bottleneck code for ALL trials
                zp_v = ln(Zt[torch.tensor(np.where(vmask)[0])] + W2(W1(Zt[torch.tensor(np.where(vmask)[0])])))
                sv = BACC(y[vmask], head(zp_v).argmax(1).numpy())
            a1_ho, npho = pairwise_l1(H, y, d, run, l1_subj, rng)
            a1_in, npin = pairwise_l1(H, y, d, run, atr_subj, rng)
            rows.append(dict(subset_seed=ss, train_seed=ts,
                             A0_L1_heldout=round(a0_ho, 4) if a0_ho else None, A1SE_L1_heldout=round(a1_ho, 4) if a1_ho else None,
                             drop_heldout=round(a0_ho - a1_ho, 4) if (a0_ho and a1_ho) else None,
                             A0_L1_insample=round(a0_in, 4) if a0_in else None, A1SE_L1_insample=round(a1_in, 4) if a1_in else None,
                             drop_insample=round(a0_in - a1_in, 4) if (a0_in and a1_in) else None,
                             source_val_bacc=round(float(sv), 4), n_pairs_heldout=npho))
            print(f"ss={ss} ts={ts}: A0_ho={a0_ho:.3f} A1SE_ho={a1_ho:.3f} drop_ho={a0_ho-a1_ho:+.3f} | "
                  f"A0_in={a0_in:.3f} A1SE_in={a1_in:.3f} drop_in={a0_in-a1_in:+.3f} | sval={sv:.3f}", flush=True)

    with open(OUT / "trained_rep_metric_power_gate.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    def ci(key):
        v = np.array([r[key] for r in rows if isinstance(r.get(key), (int, float))], float)
        if len(v) == 0:
            return None, [None, None]
        rng = np.random.default_rng(0); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(3000)]
        return round(float(v.mean()), 4), [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    dho, dho_ci = ci("drop_heldout"); din, din_ci = ci("drop_insample")
    a0m, _ = ci("A0_L1_heldout"); svm, _ = ci("source_val_bacc")
    rel = (dho / a0m) if (dho is not None and a0m) else None
    task_ok = bool(svm is not None and svm > 0.52)
    passed = bool(dho is not None and dho_ci[0] is not None and dho_ci[0] > 0 and (dho >= 0.10 or (rel is not None and rel >= 0.20)) and task_ok)
    verdict = dict(stage="8D-0_metric_power_gate", model="CBraMod", n_source=N_SOURCE, bottleneck=BOTTLENECK,
        l1_locus="bottleneck_code_h", A0_L1_heldout=a0m, drop_heldout=[dho, dho_ci], rel_drop_heldout=round(rel, 3) if rel else None,
        drop_insample=[din, din_ci], source_val_bacc=svm, task_not_collapsed=task_ok,
        metric_power_gate_pass=passed,
        interpretation=("A1-SE subject-erase moves held-out bottleneck-code L1 below the frozen ceiling -> the metric "
            "CAN move -> 8D-1 diversity test is interpretable." if passed else
            "A1-SE does NOT move held-out bottleneck-code L1 sufficiently -> metric SATURATED under this light-adapter "
            "design -> 8D-1 cannot test diversity-erasure -> STOP Phase 8 (method-limit, NOT a foundation-model null)."),
        target_labels_used=False)
    (OUT / "trained_rep_metric_power_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    (OUT / "adapter_architecture_manifest.json").write_text(json.dumps(dict(
        adapter="z'=LayerNorm(z + W2 relu(W1 z))", bottleneck=BOTTLENECK, pca_d=PCA_D, task_head="linear on z'",
        subject_adversary="linear on h + gradient-reversal (DANN)", epochs=EPOCHS, lr=LR, lambda_adv=LAMBDA_ADV), indent=2) + "\n")
    print(f"\n=== 8D-0 metric-power gate ===\n  A0 held-out L1={a0m} drop_heldout={dho} ci={dho_ci} rel={rel} "
          f"drop_insample={din} source_val={svm} task_ok={task_ok}\n  ==> PASS={passed}")


if __name__ == "__main__":
    main()
