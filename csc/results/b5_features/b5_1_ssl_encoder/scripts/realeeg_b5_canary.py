"""CSC-realEEG-B5.0 canary worker (feature-robustness branch). DEVELOPMENT-ONLY diagnostic: does the SM16 v2
FAIL (fitted-h0 plug-in null under-dispersion -> NULL_cov false-confirmation) PERSIST under a DEEP frozen-random
feature family Z_deep? NOT deployable, NOT confirmatory, NOT a rescue of v2, NO tag, NO method/statistic change.

Design contract (reviewer B5.0 spec):
  * Feature input = the FROZEN random-init EEGNet embedding cache LEE2019_B5_0.npz (encoder never saw any label;
    same SM16_no_FCz channels / bandpass / window / resample; only the representation differs from SM16 log-var).
  * The injection bank (EG.build_cohort) and the B3 certifier (certify_paired_calibrated) code paths are
    BYTE-UNCHANGED -- this worker only swaps the feature CACHE (Z_deep) into the byte-frozen internal-forensic
    worker `realeeg_internal_forensic.run_cohort`. "B3 certifier code path unchanged except feature input."
  * Injection rule honored by build_cohort itself: for NULL_cov / NULL_cov_plus_label / POS_concept /
    POS_concept_plus_cov the pooled boundary is REFIT on (Z_deep, y) and Y* is DRAWN FRESH from the Z_deep
    boundary -> labels REGENERATED on Z_deep, not transplanted. (NULL_label / random_label_control use the real
    trial labels, which belong to those trials -- the natural pairing, not a transplant.)
  * Fresh dev seed base disjoint from every prior line (v2 20e6, P3 50/60/70e6, R1 80/90e6).
Writes a JSONL partial + sha256 + provenance (B5 cache hash + feature-family marker); fail-closed exit 2 on error.
"""
import os, sys, json, argparse, hashlib, socket
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
FEAS = "/home/infres/yinwang/realeeg_feas"
sys.path.insert(0, FEAS)
import numpy as np
import realeeg_internal_forensic as IF          # byte-frozen certifier-internal worker (run_cohort reused verbatim)

# Feature family is parametrizable via env (defaults = B5.0 random encoder; B5.1 sets these to the SSL cache).
B5_CACHE = os.environ.get("B5_CACHE_PATH", os.path.join(FEAS, "b5_features/b5_0_random_encoder/LEE2019_B5_0.npz"))
B5_MANIFEST = os.environ.get("B5_MANIFEST_PATH", os.path.join(FEAS, "b5_features/b5_0_random_encoder/b5_0_feature_manifest.json"))
FEATURE_FAMILY = os.environ.get("B5_FAMILY", "B5_0_random_eegnet")

# ---- swap the feature cache into the frozen forensic worker (ONLY change vs the SM16 internal-forensic run) ----
_z = np.load(B5_CACHE)
assert _z["Z"].ndim == 2 and _z["Z"].shape[0] == _z["y"].shape[0], "B5 cache shape invariant"
IF.CACHE = dict(Z=_z["Z"], y=_z["y"], subject=_z["subject_id"], session=_z["session_id"])
IF.SUBJECTS = np.unique(IF.CACHE["subject"])
_B5_CACHE_SHA = hashlib.sha256(open(B5_CACHE, "rb").read()).hexdigest()

# guardrail: the swapped cache must carry NO oracle / true-generator field and must be label-frozen upstream.
_man = json.load(open(B5_MANIFEST))
assert _man.get("label_exposure") is False and _man.get("synthetic_label_exposure") is False \
    and _man.get("real_MI_label_exposure") is False, "B5 manifest must assert encoder label-freedom"


def run_cohort_b5(cond, r, base):
    rec = IF.run_cohort(cond, r, base)            # <-- byte-unchanged injection + certifier path, Z_deep features
    if isinstance(rec, dict) and "__worker_error__" not in rec:
        rec["feature_family"] = FEATURE_FAMILY
        rec["b5_cache_sha256"] = _B5_CACHE_SHA
        rec["development_only"] = True
        rec["not_deployable"] = True
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)   # cohort-index offset (for cohort-range sharding)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    assert a.condition in IF.CIDX, f"unknown condition {a.condition}"
    from joblib import Parallel, delayed
    recs = Parallel(n_jobs=a.jobs, backend="loky")(
        delayed(run_cohort_b5)(a.condition, r, a.base) for r in range(a.start, a.start + a.n))
    errs = [r for r in recs if "__worker_error__" in r]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")
    with open(a.out + ".sha256", "w") as f:
        f.write(hashlib.sha256(open(a.out, "rb").read()).hexdigest() + "  " + os.path.basename(a.out) + "\n")
    prov = dict(shard=f"B5:{a.base}:{a.condition}:{a.start}", base=a.base, condition=a.condition,
                start=a.start, n=a.n, jobs=a.jobs,
                feature_family=FEATURE_FAMILY, host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), n_records=len(recs), n_worker_errors=len(errs),
                b5_cache_path=B5_CACHE, b5_cache_sha256=_B5_CACHE_SHA,
                b5_manifest_sha256=hashlib.sha256(open(B5_MANIFEST, "rb").read()).hexdigest(),
                engine_sha256=hashlib.sha256(open(os.path.join(IF.MDIR, "realeeg_engine.py"), "rb").read()).hexdigest(),
                internal_forensic_sha256=hashlib.sha256(open(os.path.join(FEAS, "realeeg_internal_forensic.py"), "rb").read()).hexdigest(),
                b3_manifest_sha256=hashlib.sha256(open(os.path.join(IF.MDIR, "realeeg_b3_manifest.json"), "rb").read()).hexdigest(),
                ml=IF.ML, development_only=True, not_deployable=True)
    json.dump(prov, open(a.out + ".prov.json", "w"), indent=1, default=str)
    print(f"[B5 shard {a.base}:{a.condition}] {len(recs)} records, {len(errs)} errors, "
          f"{sum(1 for r in recs if r.get('false_confirm'))} false-confirm, "
          f"{sum(1 for r in recs if r.get('true_confirm'))} true-confirm -> {a.out}", flush=True)
    sys.exit(2 if errs else 0)                    # fail-closed: any worker error => infra failure exit 2
