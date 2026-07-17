#!/usr/bin/env python
"""D3 (Amendment 03, Blocker 1): backfill a `session_target` SIDECAR for the DGCNN forward-graph audit dumps so the
Mechanism-Subspace Oracle can run the SAME calibration-session -> future-session macro split as EEGNet (no
session-free split). STRICT, deterministic, ZERO-GPU: the DGCNN and EEGNet target blocks are built from the same
MOABB canonical order, so EEGNet `session_target` transplants onto DGCNN by position; a MOABB recompute is the
self-certifying cross-check. Fail-closed per cell: requires element-wise y_target identity (must be exact).
Does NOT mutate the tracked .audit.npz. Only the project owner may stop a scientific line. Manuscript FROZEN.

  python scripts/backfill_dgcnn_session.py            # both datasets, all erm cells, with MOABB cross-check
  python scripts/backfill_dgcnn_session.py --no-moabb # skip the (slow) MOABB recompute, EEGNet transplant only
"""
from __future__ import annotations
import argparse, glob, hashlib, re, subprocess, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_audit_npz, feat_from_tos_dump

DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def sidecar_path(audit_path):
    """Deterministic sidecar location for a DGCNN audit npz (never mutate the tracked .audit.npz)."""
    p = Path(audit_path)
    return p.parent / "session_backfill" / (p.name.replace(".audit.npz", "") + ".session.npz")


def _eegnet_dump(ds, subject, seed):
    return REPO / "tos_cmi/results/tos_cmi_eeg_frozen" / f"{ds}_EEGNet_LOSO" / f"sub{subject}_erm_lam0_seed{seed}.npz"


def _moabb_session_for_subject(ds, subject, _cache={}):
    """Recompute sess[te] directly from MOABB (parity oracle, EEGNet-independent). Cached per dataset."""
    if ds not in _cache:
        from cmi.data.moabb_data import load
        X, y, meta, classes = load(ds)
        _cache[ds] = (np.asarray(meta["subject"]).astype(int), np.asarray(meta["session"]).astype(str), np.asarray(y).astype(int))
    subj, sess, y = _cache[ds]
    te = subj == int(subject)
    return sess[te], y[te]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-moabb", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=DATASETS); a = ap.parse_args()
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    audit = []
    for ds in a.datasets:
        audit += sorted(glob.glob(str(REPO / "results/cmi_trace_p0p1/objective_comparison" / ds / "audit" / "*erm*seed*.audit.npz")))
    n_ok = n_fail = n_moabb_ok = 0; failures = []
    for ap_ in audit:
        name = Path(ap_).name
        mds = re.search(r"^(BNCI\d+_\d+)_fold(\d+)_sub(\w+?)_erm_seed(\d+)\.audit\.npz$", name)
        if not mds:
            failures.append((name, "UNPARSED_NAME")); n_fail += 1; continue
        ds, fold, subject, seed = mds.group(1), int(mds.group(2)), mds.group(3), int(mds.group(4))
        feat = feat_from_audit_npz(ap_); yd = np.asarray(feat["y_target"]).astype(int)
        eeg = _eegnet_dump(ds, subject, seed)
        if not eeg.exists():
            failures.append((name, f"NO_EEGNET_DUMP:{eeg.name}")); n_fail += 1; continue
        fe = feat_from_tos_dump(str(eeg))
        if "session_target" not in fe:
            failures.append((name, "EEGNET_HAS_NO_SESSION")); n_fail += 1; continue
        sess = np.asarray(fe["session_target"]).astype(str); ye = np.asarray(fe["y_target"]).astype(int)
        # GUARD (fail-closed): length + element-wise y identity. count-match alone is INSUFFICIENT.
        if len(sess) != len(yd) or len(ye) != len(yd) or not np.array_equal(ye, yd):
            agree = float(np.mean(ye[:len(yd)] == yd)) if len(ye) >= len(yd) else float("nan")
            failures.append((name, f"Y_MISMATCH agree={agree}")); n_fail += 1; continue
        # MOABB cross-check (self-certifying, EEGNet-independent)
        moabb_status = "skipped"
        if not a.no_moabb:
            try:
                ms, my = _moabb_session_for_subject(ds, subject)
                moabb_status = "OK" if (len(ms) == len(sess) and np.array_equal(ms, sess) and np.array_equal(my, yd)) else "MISMATCH"
            except Exception as e:
                moabb_status = f"ERROR:{type(e).__name__}"
            if moabb_status == "MISMATCH":
                failures.append((name, "MOABB_SESSION_MISMATCH")); n_fail += 1; continue
            if moabb_status == "OK":
                n_moabb_ok += 1
        vals = sorted(np.unique(sess).tolist())
        sp = sidecar_path(ap_); sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sp, session_target=sess, y_target_ref=yd, dataset=ds, target_subject=str(subject), seed=int(seed),
                 outer_fold=int(fold), n_sessions=int(len(vals)), session_values=np.asarray(vals),
                 backfill_source="eegnet_transplant", moabb_crosscheck=moabb_status, y_agreement=1.0,
                 source_eegnet_path=str(eeg.relative_to(REPO)), audit_npz_sha=hashlib.sha256(open(ap_, "rb").read()).hexdigest()[:16],
                 moabb_load_sig="tmin0.5_tmax3.5_resample128", git_sha=sha)
        n_ok += 1
    print(f"[backfill-dgcnn-session] wrote {n_ok} sidecars; moabb_ok={n_moabb_ok}; failed={n_fail}")
    for nm, why in failures[:20]:
        print(f"  FAIL {nm}: {why}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
