#!/usr/bin/env python
"""FSR Phase 8C-0 — 64-channel PhysioNetMI encoder-sanity gate (G5/G6; see FSR_48 v2 MAJOR-B). Do NOT inherit 8B's
19-32ch QC. Loads a few PhysioNetMI subjects (runs 4/8/12, imagined L/R fist, 64ch @160->200Hz), forwards the F1
spatial feature through CodeBrain + CBraMod, and verifies (a) determinism + batch-invariance at 64ch, (b) finite +
non-degenerate embeddings, (c) >= chance task with a source-only linear head (LOSO over the sanity subjects).
No target labels used for anything but final scoring. GPU. Writes encoder_64ch_sanity.csv + updates the verdict."""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
import mne
from scipy.signal import resample
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as BACC
mne.set_log_level("ERROR")
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_rq4/scripts")
import cb_cbm_feature_dump as FD

PHYS = "/projects/EEG-foundation-model/PhysioNetMI"
OUT = Path("results/fsr_codebrain_cbramod_8c")
RUNS = [4, 8, 12]
SANITY_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]


def load_subject(sid):
    Xs, ys, rs = [], [], []
    for r in RUNS:
        raw = mne.io.read_raw_edf(f"{PHYS}/S{sid:03d}/S{sid:03d}R{r:02d}.edf", preload=True, verbose=False)
        ev, evid = mne.events_from_annotations(raw, verbose=False)
        eid = {k: evid[k] for k in ("T1", "T2") if k in evid}
        ep = mne.Epochs(raw, ev, event_id=eid, tmin=0, tmax=4.0 - 1.0 / 160, baseline=None, preload=True, verbose=False)
        X = ep.get_data()                       # (n,64,640) @160
        lab = (ep.events[:, 2] == evid["T2"]).astype(int)   # 0=left(T1) 1=right(T2)
        Xs.append(X); ys.append(lab); rs.append(np.full(len(lab), r))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(rs)


def preprocess(X):                              # (n,64,640)@160 -> (n,64,4,200)@200 zscore
    n, C, T = X.shape
    x = resample(X, int(round(T * 200 / 160)), axis=-1)[:, :, :800]
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
    return x.reshape(n, C, 4, 200).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4); FD.set_determinism(torch)
    dev = torch.device("cuda")
    data = {s: load_subject(s) for s in SANITY_SUBJECTS}
    rows = []
    gate = {}
    for model in ("codebrain", "cbramod"):
        fwd, _, _ = (FD.build_codebrain if model == "codebrain" else FD.build_cbramod)(torch, dev)

        def f1(Xr):                              # (n,64,640)@160 -> F1 (n, 64*200)
            x = preprocess(Xr); F1 = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    feats = fwd(torch.tensor(x[i:i + 64], device=dev)).cpu().numpy()
                    F1.append(feats.mean(axis=2).reshape(feats.shape[0], feats.shape[1] * feats.shape[3]))
            return np.concatenate(F1, 0)

        # determinism QC (64ch) on subject 1
        X1 = data[SANITY_SUBJECTS[0]][0][:32]
        a = f1(X1); b = f1(X1); c = np.concatenate([f1(X1[:16]), f1(X1[16:32])], 0)
        det = dict(repeat_max=float(np.max(np.abs(a - b))), batchgroup_max=float(np.max(np.abs(a - c))))
        # per-subject F1 + non-degeneracy
        F, Y, S = [], [], []
        for s in SANITY_SUBJECTS:
            Xr, y, r = data[s]; f = f1(Xr); F.append(f); Y.append(y); S.append(np.full(len(y), s))
        F = np.concatenate(F, 0).astype(np.float64); Y = np.concatenate(Y); S = np.concatenate(S)
        pdist = float(np.median(np.linalg.norm(F[:50] - F[50:100], axis=1))) if len(F) >= 100 else None
        # above-chance task: LOSO over sanity subjects, PCA-64 + LDA
        from sklearn.decomposition import PCA
        accs = []
        for ts in SANITY_SUBJECTS:
            tr = S != ts; te = S == ts
            p = PCA(n_components=min(64, tr.sum() - 1)).fit(F[tr]); Ztr = p.transform(F[tr]); Zte = p.transform(F[te])
            mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
            h = LDA().fit((Ztr - mu) / sd, Y[tr]); accs.append(BACC(Y[te], h.predict((Zte - mu) / sd)))
        tbacc = float(np.mean(accs))
        finite = bool(np.isfinite(F).all())
        ok = bool(det["repeat_max"] < 1e-4 and finite and (pdist is None or pdist > 1e-6) and tbacc > 0.52)
        rows.append(dict(model=model, n_subjects=len(SANITY_SUBJECTS), n_trials=int(len(Y)),
                         repeat_max=round(det["repeat_max"], 6), batchgroup_max=round(det["batchgroup_max"], 6),
                         finite=finite, median_pairwise_L2=round(pdist, 4) if pdist else None,
                         loso_task_bacc=round(tbacc, 4), task_chance=0.5, sanity_pass=ok))
        gate[model] = ok
        print(f"{model}: repeat={det['repeat_max']:.2e} batchgroup={det['batchgroup_max']:.2e} finite={finite} "
              f"L2={pdist:.3f} task_bAcc={tbacc:.4f} PASS={ok}", flush=True)

    with open(OUT / "encoder_64ch_sanity.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    # update verdict G5/G6
    vp = OUT / "phase8c0_verdict.json"
    v = json.load(open(vp)) if vp.exists() else {}
    v["codebrain_64ch_forward_pass"] = gate.get("codebrain")
    v["cbramod_64ch_forward_pass"] = gate.get("cbramod")
    v.setdefault("gates", {})["G5_codebrain_64ch"] = gate.get("codebrain")
    v["gates"]["G6_cbramod_64ch"] = gate.get("cbramod")
    g = v.get("gates", {})
    v["proceed_to_8c1"] = bool(all(g.get(k) for k in g) and (gate.get("codebrain") or gate.get("cbramod")))
    v["note"] = ("8C-0 complete. Encoder 64ch sanity: " +
                 ("both pass" if gate.get("codebrain") and gate.get("cbramod") else
                  "codebrain-only" if gate.get("codebrain") else "cbramod-only" if gate.get("cbramod") else "BOTH FAIL -> STOP"))
    vp.write_text(json.dumps(v, indent=2) + "\n")
    print("proceed_to_8c1 =", v["proceed_to_8c1"])


if __name__ == "__main__":
    main()
