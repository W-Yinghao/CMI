#!/usr/bin/env python
"""D0.5-A raw-signal CSP sanity for SHU-MI. Confirms the labels / C3-C4 lateralization / preprocessing are
decodable INDEPENDENT of any foundation encoder. Manual CSP (generalized eigen) + LDA on band-passed (8-30 Hz)
19-common raw trials. Within-subject (session-held-out) is the primary gate; cross-subject (source->target) as a
cheap secondary; a label-shuffle control bounds chance. No encoder, no target-label leakage into filters/LDA."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, "s2p/scripts")
SHUMI = "/projects/EEG-foundation-model/tdoan-24/SHUMI_200hz"
CH_IDX = [0, 1, 3, 4, 12, 13, 23, 24, 30, 31, 5, 6, 14, 15, 25, 26, 2, 11, 22]   # 19-common (encoder order)
SPLIT = {"source": list(range(1, 16)), "target": list(range(21, 26))}


def load_raw(subjects):
    import lmdb, pickle
    env = lmdb.open(SHUMI, readonly=True, lock=False, readahead=False, meminit=False)
    X, y, sj, ss = [], [], [], []
    with env.begin() as txn:
        for s in subjects:
            for ses in range(1, 6):
                t = 0
                while True:
                    v = txn.get(f"sub-{s:03d}_ses-{ses:02d}_task_motorimagery_eeg-{t}".encode())
                    if v is None:
                        break
                    d = pickle.loads(v)
                    X.append(np.asarray(d["sample"], np.float32)[CH_IDX]); y.append(int(d["label"]))
                    sj.append(s); ss.append(ses); t += 1
    return np.stack(X), np.array(y), np.array(sj), np.array(ss)      # X (n,19,800) RAW


def bandpass(X, lo=8, hi=30, fs=200):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=-1)


def csp_fit(Xa, Xb, k=3):
    Ca = np.mean([np.cov(x) for x in Xa], 0); Cb = np.mean([np.cov(x) for x in Xb], 0)
    Ca /= np.trace(Ca); Cb /= np.trace(Cb)
    w, V = eigh(Ca, Ca + Cb)
    idx = np.argsort(w); sel = np.concatenate([idx[:k], idx[-k:]])
    return V[:, sel].T                                              # (2k,19)


def csp_feat(X, F):
    return np.array([np.log(np.var(F @ x, axis=1) + 1e-8) for x in X])


def decode(Xtr, ytr, Xte, yte, k=3, shuffle=False, rng=None):
    F = csp_fit(Xtr[ytr == 0], Xtr[ytr == 1], k)
    ytr2 = rng.permutation(ytr) if shuffle else ytr
    if shuffle:
        F = csp_fit(Xtr[ytr2 == 0], Xtr[ytr2 == 1], k)
    lda = LinearDiscriminantAnalysis().fit(csp_feat(Xtr, F), ytr2)
    return balanced_accuracy_score(yte, lda.predict(csp_feat(Xte, F)))


def main():
    R = Path("results/s2p_p1_downstream/d0p5_decodability_sanity"); R.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0); t0 = time.time()
    subs = sorted(set(SPLIT["source"] + SPLIT["target"] + list(range(16, 21))))
    X, y, sj, ss = load_raw(subs); Xf = bandpass(X)
    rows = []
    # within-subject: session-held-out (train sess<max, test sess==max)
    wi = []
    for s in sorted(set(sj)):
        m = sj == s; smax = ss[m].max(); tr = m & (ss < smax); te = m & (ss == smax)
        if te.sum() < 8 or len(np.unique(y[tr])) < 2:
            continue
        acc = decode(Xf[tr], y[tr], Xf[te], y[te], rng=rng)
        sh = decode(Xf[tr], y[tr], Xf[te], y[te], shuffle=True, rng=rng)
        wi.append(acc); rows.append(dict(kind="within_subject", subject=int(s), bacc=round(float(acc), 4),
                                         shuffle_bacc=round(float(sh), 4), n_test=int(te.sum())))
    # cross-subject: source(1-15) -> target(21-25)
    tr = np.isin(sj, SPLIT["source"]); te = np.isin(sj, SPLIT["target"])
    cross = decode(Xf[tr], y[tr], Xf[te], y[te], rng=rng)
    cross_sh = decode(Xf[tr], y[tr], Xf[te], y[te], shuffle=True, rng=rng)
    rows.append(dict(kind="cross_subject", subject="1-15->21-25", bacc=round(float(cross), 4),
                     shuffle_bacc=round(float(cross_sh), 4), n_test=int(te.sum())))
    import pandas as pd
    pd.DataFrame(rows).to_csv(R / "raw_csp_sanity.csv", index=False)
    wi = np.array(wi)
    verdict = dict(within_subject_median_bacc=round(float(np.median(wi)), 4),
                   within_subject_mean_bacc=round(float(wi.mean()), 4),
                   within_subject_frac_above_0p55=round(float((wi > 0.55).mean()), 3),
                   n_subjects=int(len(wi)), cross_subject_bacc=round(float(cross), 4),
                   cross_subject_shuffle_bacc=round(float(cross_sh), 4),
                   within_shuffle_median=round(float(np.median([r["shuffle_bacc"] for r in rows if r["kind"] == "within_subject"])), 4),
                   gate_within_median_ge_0p60=bool(np.median(wi) >= 0.60),
                   gate_within_clearly_above_chance=bool((wi > 0.55).mean() >= 0.6),
                   raw_csp_within_subject_pass=bool(np.median(wi) >= 0.60 or (wi > 0.55).mean() >= 0.6),
                   elapsed_s=round(time.time() - t0, 1))
    json.dump(verdict, open(R / "raw_csp_verdict.json", "w"), indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
