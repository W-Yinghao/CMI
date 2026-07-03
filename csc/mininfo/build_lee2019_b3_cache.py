"""Isolated Lee2019/OpenBMI B3 real-feature CACHE BUILDER (pre-reg v4, D2).

Reads the OpenBMI .mat files DIRECTLY (moabb exposes only session 1 despite both being on disk).
NO cmi import, NO moabb dependency. Frozen pipeline (pre-reg v4):
  montage  = SM16_no_FCz (16 sensorimotor channels; FCz dropped, NO substitute -> fail-closed)
  bandpass = 8-30 Hz, 4th-order Butterworth, zero-phase filtfilt on the continuous signal
  window   = 0.5-3.5 s post trial onset ; resample epoch to 128 Hz (384 samples)
  feature  = Z = log( var_t(x_ch) )  (normalize=None) -> 16-dim log-bandpower per trial
  label    = {left_hand:0, right_hand:1}   (OpenBMI y_dec: 1=right_hand, 2=left_hand)
  run      = MI training (offline) run per session (EEG_MI_train), matching moabb default (100 trials/session)

FEASIBILITY MODE ONLY (this authorization): builds the cache + metadata JSON + feasibility report, then STOPS.
Does NOT run any certifier, injected bank, or create a tag.

Usage (in eeg2025 or icml; scipy only):
  python build_lee2019_b3_cache.py --probe-only            # inspect one .mat structure, exit
  python build_lee2019_b3_cache.py [--subjects N] --out DIR # build feasibility cache (default all 54)
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample

# ---- frozen constants (pre-reg v4) ----
ROOT = "/projects/EEG-foundation-model/datalake/raw/MNE-lee2019-mi-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542"
MONTAGE = ["FC3", "FC1", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
           "CP3", "CP1", "CPz", "CP2", "CP4"]              # SM16_no_FCz
MONTAGE_NAME = "SM16_no_FCz"
BANDPASS = (8.0, 30.0)
WINDOW = (0.5, 3.5)          # seconds post onset
FS_RESAMPLED = 128
N_RESAMPLED = int((WINDOW[1] - WINDOW[0]) * FS_RESAMPLED)  # 384
LABEL_MAP = {1: 1, 2: 0}     # y_dec 1=right_hand->1, 2=left_hand->0
N_SUBJECTS = 54
SESSIONS = (1, 2)
MIN_TRIALS_PER_CELL = 8      # feasibility gate 6


def mat_path(subj, sess):
    return os.path.join(ROOT, f"session{sess}", f"s{subj}", f"sess0{sess}_subj{subj:02d}_EEG_MI.mat")


def _load_mat(path):
    """scipy.io.loadmat (OpenBMI is v7 per moabb); logged h5py/pymatreader fallback if v7.3."""
    try:
        return loadmat(path, squeeze_me=False, struct_as_record=False), "scipy.io.loadmat"
    except NotImplementedError:                # v7.3 HDF5
        try:
            from pymatreader import read_mat
            return read_mat(path), "pymatreader"
        except Exception as e:
            raise RuntimeError(f"v7.3 .mat and no pymatreader/h5py fallback: {e}")


def _struct(mat, key):
    v = mat[key]
    # struct_as_record=False -> mat_struct; take [0,0] if wrapped
    return v[0, 0] if hasattr(v, "shape") and v.shape == (1, 1) else v


def _chan_names(chan):
    out = []
    for c in np.asarray(chan).ravel():
        a = np.asarray(c).ravel()
        out.append(str(a[0]) if a.size else str(c))
    return out


def parse_run(path):
    """Return (x[time,ch], fs, chan_names[list], onsets[int array], y_dec[int array]) from EEG_MI_train."""
    mat, parser = _load_mat(path)
    s = _struct(mat, "EEG_MI_train")
    x = np.asarray(s.x, dtype=np.float64)              # [time, channels]
    fs = int(np.asarray(s.fs).ravel()[0])
    chan = _chan_names(s.chan)
    onsets = np.asarray(s.t, dtype=np.int64).ravel()
    y_dec = np.asarray(s.y_dec, dtype=np.int64).ravel()
    if x.shape[0] < x.shape[1]:                        # guard: ensure [time, ch]
        x = x.T
    return x, fs, chan, onsets, y_dec, parser


def features_for_run(path):
    """Return (Z[n_trials,16], y[n_trials], meta dict). Fail-closed if any montage channel absent."""
    x, fs, chan, onsets, y_dec, parser = parse_run(path)
    missing = [c for c in MONTAGE if c not in chan]
    if missing:
        raise RuntimeError(f"FAIL_CLOSED montage channel(s) absent {missing} in {os.path.basename(path)}")
    idx = [chan.index(c) for c in MONTAGE]
    xc = x[:, idx]                                     # [time, 16], continuous
    sos = butter(4, [BANDPASS[0], BANDPASS[1]], btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, xc, axis=0)                  # zero-phase bandpass on continuous
    a, b = int(WINDOW[0] * fs), int(WINDOW[1] * fs)
    Z, y = [], []
    for t0, yd in zip(onsets, y_dec):
        seg = xf[t0 + a: t0 + b, :]                    # [~3000, 16]
        if seg.shape[0] < (b - a) - 2:                 # onset too close to end -> skip (logged)
            continue
        seg = resample(seg, N_RESAMPLED, axis=0)       # -> [384, 16] (log-var ~ sample-rate-invariant)
        Z.append(np.log(seg.var(axis=0) + 1e-20))
        y.append(LABEL_MAP.get(int(yd), -1))
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    meta = dict(fs_raw=fs, n_trials=int(len(y)), n_skipped=int(len(onsets) - len(y)),
                n_channels_total=len(chan), parser=parser,
                source_file=path, source_size=os.path.getsize(path))
    return Z, y, meta


def probe_one():
    p = mat_path(1, 1)
    print("PROBE file:", p, "exists:", os.path.exists(p))
    x, fs, chan, onsets, y_dec, parser = parse_run(p)
    print("parser:", parser)
    print("x shape [time,ch]:", x.shape, "| fs:", fs)
    print("n_channels:", len(chan))
    print("first 20 channels:", chan[:20])
    print("montage present:", [c for c in MONTAGE if c in chan], "| missing:", [c for c in MONTAGE if c not in chan])
    print("n_onsets:", len(onsets), "| y_dec unique:", np.unique(y_dec, return_counts=True))
    Z, y, meta = features_for_run(p)
    print("Z shape:", Z.shape, "| Z std:", round(float(Z.std()), 4),
          "| rank:", int(np.linalg.matrix_rank(Z - Z.mean(0))))
    print("y counts:", np.unique(y, return_counts=True), "| skipped:", meta["n_skipped"])
    print("PROBE_OK")


def build(n_subjects, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    rows_Z, rows_y, rows_subj, rows_sess, rows_trial = [], [], [], [], []
    counts = {}            # (subj,sess) -> {class:count}
    missing_subjects, missing_sessions, missing_channels = [], [], []
    per_file = []
    subs = list(range(1, n_subjects + 1))
    for subj in subs:
        for sess in SESSIONS:
            p = mat_path(subj, sess)
            if not os.path.exists(p):
                missing_sessions.append([subj, sess]); print(f"  MISSING file subj{subj} sess{sess}"); continue
            try:
                Z, y, meta = features_for_run(p)
            except RuntimeError as e:
                if "FAIL_CLOSED" in str(e):
                    missing_channels.append([subj, sess, str(e)]); print(f"  {e}"); continue
                raise
            tid = np.arange(len(y))
            rows_Z.append(Z); rows_y.append(y)
            rows_subj.append(np.full(len(y), subj)); rows_sess.append(np.full(len(y), sess)); rows_trial.append(tid)
            cc = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            counts[f"{subj}|{sess}"] = cc
            per_file.append(meta)
            print(f"  subj{subj:02d} sess{sess}: n={len(y)} classes={cc} std={Z.std():.3f}")
    # check subjects present in both sessions
    have = {}
    for k in counts:
        s, ss = k.split("|"); have.setdefault(int(s), set()).add(int(ss))
    for subj in subs:
        if have.get(subj, set()) != set(SESSIONS):
            missing_subjects.append(subj)

    Z = np.concatenate(rows_Z) if rows_Z else np.zeros((0, 16))
    y = np.concatenate(rows_y) if rows_y else np.zeros(0, int)
    subject_id = np.concatenate(rows_subj) if rows_subj else np.zeros(0, int)
    session_id = np.concatenate(rows_sess) if rows_sess else np.zeros(0, int)
    trial_id = np.concatenate(rows_trial) if rows_trial else np.zeros(0, int)

    # eligible paired subjects: both sessions, both classes, >= MIN_TRIALS_PER_CELL per class per session
    eligible = []
    for subj in subs:
        ok = subj not in missing_subjects
        for ss in SESSIONS:
            cc = counts.get(f"{subj}|{ss}", {})
            if set(cc) != {0, 1} or min(cc.values() or [0]) < MIN_TRIALS_PER_CELL:
                ok = False
        if ok:
            eligible.append(subj)

    feat_std = Z.std(axis=0) if len(Z) else np.zeros(16)
    rank = int(np.linalg.matrix_rank(Z - Z.mean(0))) if len(Z) else 0
    meta = dict(
        montage_name=MONTAGE_NAME, channel_names=MONTAGE, bandpass=list(BANDPASS), window=list(WINDOW),
        fs_resampled=FS_RESAMPLED, normalize=None, label_map={"left_hand": 0, "right_hand": 1},
        run="EEG_MI_train", n_subjects_scanned=n_subjects, n_sessions_per_subject=len(SESSIONS),
        trial_counts_by_subject_session_class=counts,
        missing_subjects=missing_subjects, missing_sessions=missing_sessions, missing_channels=missing_channels,
        feature_dim=int(Z.shape[1]) if len(Z) else 0, feature_rank=rank,
        feature_std_min=float(feat_std.min()), feature_std_median=float(np.median(feat_std)),
        feature_std_max=float(feat_std.max()),
        nan_count=int(np.isnan(Z).sum()), inf_count=int(np.isinf(Z).sum()),
        eligible_paired_subjects=eligible, n_eligible=len(eligible),
        n_trials_total=int(len(y)), build_seconds=round(time.time() - t_start, 1),
        source_files=[{"file": m["source_file"], "size": m["source_size"], "parser": m["parser"],
                       "n_trials": m["n_trials"], "n_skipped": m["n_skipped"]} for m in per_file],
    )
    # ---- feasibility gates ----
    gates = {
        "G1_all16_channels_present": len(missing_channels) == 0,
        "G2_54_subjects_both_sessions_or_reported": True,   # reported via missing_* (informational)
        "G3_ge20_eligible_paired": len(eligible) >= 20,
        "G4_eligible_have_both_sessions": all(have.get(s, set()) == set(SESSIONS) for s in eligible),
        "G5_each_session_both_classes": all(set(counts.get(f"{s}|{ss}", {})) == {0, 1}
                                            for s in eligible for ss in SESSIONS),
        "G6_min_trials_per_cell": all(min(counts.get(f"{s}|{ss}", {0: 0, 1: 0}).values()) >= MIN_TRIALS_PER_CELL
                                      for s in eligible for ss in SESSIONS),
        "G7_feature_dim_16": (Z.shape[1] == 16) if len(Z) else False,
        "G8_rank_ge3": rank >= 3,
        "G9_no_nan_inf": meta["nan_count"] == 0 and meta["inf_count"] == 0,
        "G10_nondegenerate_std": float(np.median(feat_std)) > 1e-6,
    }
    primary = ["G1_all16_channels_present", "G3_ge20_eligible_paired", "G4_eligible_have_both_sessions",
               "G5_each_session_both_classes", "G6_min_trials_per_cell", "G7_feature_dim_16",
               "G8_rank_ge3", "G9_no_nan_inf", "G10_nondegenerate_std"]
    feasible = all(gates[g] for g in primary)
    meta["feasibility_gates"] = gates
    meta["feasibility_PASS"] = feasible

    np.savez_compressed(os.path.join(out_dir, "LEE2019_B3.npz"),
                        Z=Z, y=y, subject_id=subject_id, session_id=session_id, trial_id=trial_id,
                        channel_names=np.array(MONTAGE), montage_name=MONTAGE_NAME)
    with open(os.path.join(out_dir, "LEE2019_B3_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== FEASIBILITY REPORT ===")
    print(f"trials total: {len(y)} | subjects scanned: {n_subjects} | eligible paired: {len(eligible)}")
    print(f"missing_subjects: {missing_subjects}")
    print(f"missing_sessions: {missing_sessions}")
    print(f"missing_channels: {missing_channels}")
    print(f"feature dim: {meta['feature_dim']} | rank: {rank} | std median: {meta['feature_std_median']:.4f} "
          f"(min {meta['feature_std_min']:.4f} max {meta['feature_std_max']:.4f})")
    print(f"nan: {meta['nan_count']} inf: {meta['inf_count']}")
    for g in primary:
        print(f"  {'PASS' if gates[g] else 'FAIL'}  {g}")
    print(f"  info  G2 (missingness reported): missing_subjects={len(missing_subjects)} "
          f"missing_sessions={len(missing_sessions)}")
    print("FEASIBILITY_PASS" if feasible else "FEASIBILITY_FAIL (STOP + report; no montage/feature change)")
    return feasible


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--out", default="/home/infres/yinwang/realeeg_feas/cache")
    a = ap.parse_args()
    print(f"[builder] montage={MONTAGE_NAME}({len(MONTAGE)}) bandpass={BANDPASS} window={WINDOW} "
          f"fs_out={FS_RESAMPLED} run=EEG_MI_train")
    if a.probe_only:
        probe_one(); sys.exit(0)
    print("[builder] fail-fast probe of subj1/sess1 before full build...")
    probe_one()
    ok = build(a.subjects, a.out)
    sys.exit(0 if ok else 4)
