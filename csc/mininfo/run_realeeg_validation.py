"""CSC real-EEG validation runner (v2) — dry-run by default; guarded, PARALLEL, checkpointed execute after freeze.

v2 is a PERFORMANCE-ONLY successor to the serial v1 (`csc-realeeg-v1`), which was infrastructure-infeasible
(serial ~6.3 days > the 5-day cpu-high wall, end-only write → killed with no artifact). The science is
byte-identical: same feature/montage/cache/injections/Route-A/B3 method/`n_boot`/cohorts/seed schedule/gates/
alpha/denominators. Only the EXECUTION changed — the 801 independent cohorts run in parallel and are canonically
re-sorted (condition_index, cohort_index) before the verdict, which is numerically identical to serial (proven
in tests). Each cohort streams to a JSONL partial + periodic checkpoint, so an infra kill leaves verifiable
progress and can be resumed only under the SAME tag / cache / manifest / task-table.

Default mode verifies the manifests, cache provenance, seed disjointness, pinned method/engine hashes, the
801-task table, and the fail-closed structure. The `--execute` path is DISABLED until the repository is checked
out at `refs/tags/csc-realeeg-v2` with a clean tree and matching pinned method/cache/engine hashes; with no tag
it always refuses (exit 2). A missing/duplicate/worker-errored cohort is an INFRASTRUCTURE failure (exit 2, NO
endpoint evaluated), never a silent skip. The genuine session contrast is descriptive-only and cannot affect
PASS/FAIL.

Usage:
  python -m csc.mininfo.run_realeeg_validation                       # dry-run report (exit 0 pass / 1 fail)
  python -m csc.mininfo.run_realeeg_validation --execute --jobs 24   # guarded parallel run; refuses unless frozen tag/provenance checks pass (exit 2)
  python -m csc.mininfo.run_realeeg_validation --execute --resume    # resume a killed run (same frozen inputs only)
  python -m csc.mininfo.run_realeeg_validation --smoke               # toy-cache plumbing self-test (non-real seed)
"""
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))                 # csc/mininfo
CSC_ROOT = os.path.dirname(os.path.dirname(HERE))                 # repo root (contains csc/)
CACHE_MANIFEST = os.path.join(HERE, "realeeg_lee2019_cache_manifest.json")
BANK_MANIFEST = os.path.join(HERE, "realeeg_bank_manifest.json")
ROUTEA_MANIFEST = os.path.join(HERE, "realeeg_routeA_manifest.json")
B3_MANIFEST = os.path.join(HERE, "realeeg_b3_manifest.json")

# synthetic + dev seed streams the real-EEG seeds must be disjoint from
FORBIDDEN_SEED_RANGES = [(900000, 900065), (1800000, 1800065), (3000000, 14100047)]
MONTAGE = ["FC3", "FC1", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
           "CP3", "CP1", "CPz", "CP2", "CP4"]
BANK_CONDITIONS = ["NULL_real_session", "NULL_cov", "NULL_label", "NULL_cov_plus_label",
                   "POS_concept", "POS_concept_plus_cov", "POS_pure_conditional",
                   "random_label_control", "genuine_session_contrast_descriptive"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    with open(path) as f:
        return json.load(f)


class Checks:
    def __init__(self):
        self.results = []      # (name, ok, msg)

    def check(self, name, ok, msg=""):
        self.results.append((name, bool(ok), msg))
        return bool(ok)

    def report(self):
        print("=== CSC real-EEG DRY-RUN verification ===")
        for name, ok, msg in self.results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + msg) if msg else ''}")
        n_fail = sum(1 for _, ok, _ in self.results if not ok)
        print(f"[{len(self.results) - n_fail}/{len(self.results)} checks pass]")
        return n_fail == 0


def dry_run():
    c = Checks()
    # 1. manifests load
    manifests = {}
    for key, path in [("cache", CACHE_MANIFEST), ("bank", BANK_MANIFEST),
                      ("routeA", ROUTEA_MANIFEST), ("b3", B3_MANIFEST)]:
        try:
            manifests[key] = _load(path); ok = True; msg = ""
        except Exception as e:
            manifests[key] = None; ok = False; msg = str(e)
        c.check(f"manifest_loads[{key}]", ok, msg)
    if any(m is None for m in manifests.values()):
        return c.report()
    cache, bank, rA, b3 = manifests["cache"], manifests["bank"], manifests["routeA"], manifests["b3"]

    # 2. cache manifest: frozen montage (16, exact, no substitute)
    c.check("cache_montage_is_SM16_no_FCz", cache["channels"] == MONTAGE,
            "channels must equal the frozen 16-ch montage exactly")
    c.check("cache_normalize_None", cache["normalize"] is None)
    c.check("cache_feature_pipeline",
            cache["bandpass_hz"] == [8, 30] and cache["window_sec"] == [0.5, 3.5]
            and cache["fs_resampled"] == 128 and cache["feature"] == "log_var_time")
    c.check("cache_label_map_frozen", cache["label_map"] == {"left_hand": 0, "right_hand": 1})
    c.check("cache_feature_name_frozen", cache["feature_name"] == "SM16_no_FCz_logbandpower")
    c.check("cache_run_frozen", cache["run"] == "EEG_MI_train")

    # 3. cache file: verify if present (dry-run allows absent -> rebuildable), else note
    prov = cache["provenance"]
    cpath, mpath = prov["cache_path"], prov["cache_metadata_path"]
    if os.path.exists(cpath) and os.path.exists(mpath):
        c.check("cache_sha256_matches", _sha256(cpath) == prov["cache_sha256"])
        c.check("cache_metadata_sha256_matches", _sha256(mpath) == prov["cache_metadata_sha256"])
        meta = _load(mpath)
        c.check("cache_feature_dim_16", meta.get("feature_dim") == 16)
        c.check("cache_rank_ge3", meta.get("feature_rank", 0) >= 3)
        c.check("cache_std_nondegenerate", meta.get("feature_std_median", 0) > 1e-6)
        c.check("cache_eligible_ge_min",
                meta.get("n_eligible", 0) >= cache["min_eligible_paired_subjects"])
        c.check("cache_no_nan_inf", meta.get("nan_count", 1) == 0 and meta.get("inf_count", 1) == 0)
        c.check("cache_channels_match", meta.get("channel_names") == MONTAGE)
    else:
        bf = os.path.join(CSC_ROOT, prov["builder_file"])
        c.check("cache_absent_builder_hash_verified",
                os.path.exists(bf) and _sha256(bf) == prov["builder_sha256"],
                "cache absent -> builder file must match pinned builder_sha256 to be rebuildable")

    # 4. bank manifest: all 9 conditions, gating roles, genuine contrast descriptive
    names = [x["name"] for x in bank["conditions"]]
    c.check("bank_has_9_conditions", names == BANK_CONDITIONS, f"got {names}")
    byname = {x["name"]: x for x in bank["conditions"]}
    c.check("bank_NULL_cov_gating_covariate",
            byname["NULL_cov"]["gating"] is True and "COVARIATE" in byname["NULL_cov"]["ground_truth"].upper())
    c.check("bank_genuine_contrast_descriptive_nongating",
            byname["genuine_session_contrast_descriptive"]["gating"] is False
            and "DESCRIPTIVE" in byname["genuine_session_contrast_descriptive"]["role"].upper())
    c.check("bank_power_pos_nongating",
            byname["POS_concept"]["gating"] is False and byname["POS_concept_plus_cov"]["gating"] is False)
    c.check("bank_trap_controls_gating",
            byname["NULL_label"]["gating"] is True and byname["NULL_cov_plus_label"]["gating"] is True)
    c.check("bank_random_label_control_gating", byname["random_label_control"]["gating"] is True,
            "random labels have no P(Y|Z) structure; a confirmation here is a calibration failure -> must gate")
    c.check("bank_gating_set_exact",
            bank["gating_summary"]["gating_conditions"] ==
            ["NULL_cov", "NULL_label", "NULL_cov_plus_label", "random_label_control"])
    for cond in bank["conditions"]:
        c.check(f"bank_spec_complete[{cond['name']}]",
                all(k in cond for k in ("input_sessions", "labels", "label_model", "held_fixed",
                                        "injected_shift", "ground_truth", "role", "gating", "routes")))
    # H1: the injection+verdict engine is hash-pinned in the bank manifest
    ep = bank.get("engine_provenance", {})
    engp = os.path.join(CSC_ROOT, ep.get("engine_file", "__missing__"))
    c.check("engine_sha256_pinned_matches",
            os.path.exists(engp) and _sha256(engp) == ep.get("engine_sha256"),
            "injection+verdict engine hash must match the pinned engine_sha256")

    # 5. seed schedule disjoint
    ss = bank["seed_schedule"]
    base = ss["realeeg_base_seed"]
    max_hi = max(hi for _, hi in FORBIDDEN_SEED_RANGES)
    disjoint = all(base < lo or base > hi for lo, hi in FORBIDDEN_SEED_RANGES) and base > max_hi
    c.check("seed_base_disjoint_from_synthetic", disjoint,
            f"realeeg_base_seed={base} must be outside every forbidden range and > max {max_hi}")

    # 6. route manifests: method hashes match on-disk files (self-consistency / no drift)
    for rk, rman in (("A", rA), ("B3", b3)):
        allok = True
        for rel, want in rman["code_provenance"]["method_files_sha256"].items():
            p = os.path.join(CSC_ROOT, rel)
            got = _sha256(p) if os.path.exists(p) else "MISSING"
            if got != want:
                allok = False
        c.check(f"route[{rk}]_method_hashes_match_disk", allok)
        c.check(f"route[{rk}]_cache_hash_matches_cache_manifest",
                rman["cache"]["cache_sha256"] == prov["cache_sha256"])
        c.check(f"route[{rk}]_alpha_frozen",
                rman["statistics"]["alpha_budget_per_decision_cohort"] == 0.025
                and rman["statistics"]["family_report_target"] == 0.05)
        c.check(f"route[{rk}]_invalid_cap_020", rman["statistics"]["invalid_fraction_cap"] == 0.20)
        c.check(f"route[{rk}]_bcohort_2000", rman["statistics"]["b_cohort_bootstrap"] == 2000)

    # 7. explicit gating flags: R2 power NOT gating, R5 2b NOT gating, genuine contrast NOT gating
    for rk, rman in (("A", rA), ("B3", b3)):
        gf = rman.get("gating_flags", {})
        c.check(f"route[{rk}]_R2_power_not_gating", gf.get("R2_power_is_gating") is False,
                "gating_flags.R2_power_is_gating must be explicitly false")
        c.check(f"route[{rk}]_R5_2b_not_gating", gf.get("R5_2b_is_gating") is False,
                "gating_flags.R5_2b_is_gating must be explicitly false")
        c.check(f"route[{rk}]_genuine_contrast_not_gating", gf.get("genuine_contrast_is_gating") is False)
    c.check("b3_bcertifier_200", b3["statistics"]["b_certifier_internal_null"] == 200)

    # 8. no real validation RESULT artifact (v1 or v2 FINAL) exists yet (must be absent in the dry-run package).
    #    A partial/checkpoint is progress evidence, NOT a result, and does not count here.
    result_names = ["realeeg_validation_result.json", "realeeg_validation_v2.final.json"]
    result_globs = [os.path.join(d, n)
                    for d in (os.path.join(HERE, "..", "results"), os.path.join(CSC_ROOT, "csc", "results"))
                    for n in result_names]
    c.check("no_real_validation_result_exists", not any(os.path.exists(p) for p in result_globs))

    # 9. v2 execution: the task table is deterministically 801 tasks, and the bank documents the
    #    performance-only parallel execution (n_boot / cohort counts / seeds UNCHANGED from v1).
    try:
        from . import realeeg_engine as _EG
        n_tasks = len(_EG.build_task_table(bank, bank["seed_schedule"]["realeeg_base_seed"]))
    except Exception as e:
        n_tasks = -1; print(f"  (task-table build error: {e})")
    exp_tasks = 8 * bank["run_spec"]["cohorts_per_condition"] + 1     # 8 conds x R + 1 genuine
    c.check("dry_run_task_table_is_801", n_tasks == exp_tasks == 801, f"got {n_tasks} (expected {exp_tasks})")
    exe = bank.get("execution_provenance", {})
    c.check("bank_execution_is_parallel_performance_only",
            exe.get("mode") == "cohort_parallel" and exe.get("performance_only") is True
            and exe.get("n_tasks_expected") == 801,
            "bank must document the v2 performance-only parallel execution")
    c.check("bank_execution_unchanged_science_declared",
            b3["statistics"]["b_certifier_internal_null"] == 200
            and bank["run_spec"]["cohorts_per_condition"] == 100
            and bank["run_spec"]["subjects_per_cohort"] == 30
            and bank["seed_schedule"]["realeeg_base_seed"] == 20000000,
            "v2 must not change n_boot / cohort counts / seed schedule")
    c.check("bank_execution_blas_threads_pinned_to_1",
            all(str(exe.get("blas_threads", {}).get(v)) == "1"
                for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
            "BLAS threads must be pinned to 1 to avoid oversubscription")

    ok = c.report()
    print("DRY_RUN_PASS" if ok else "DRY_RUN_FAIL")
    return ok


TAG = "csc-realeeg-v2"


def _git(*args):
    import subprocess
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=CSC_ROOT)


def verify_git_frozen():
    """HEAD == TAG^{commit} AND clean tree. Fails closed if the tag does not exist (it does not yet)."""
    tag = _git("rev-parse", f"{TAG}^{{commit}}")
    if tag.returncode != 0:
        return [f"tag {TAG} does not exist (freeze/run not authorized)"]
    errs = []
    if _git("rev-parse", "HEAD").stdout.strip() != tag.stdout.strip():
        errs.append(f"HEAD != {TAG} commit")
    if _git("status", "--porcelain").stdout.strip():
        errs.append("working tree not clean")
    return errs


def _method_hashes_ok():
    for path in (ROUTEA_MANIFEST, B3_MANIFEST):
        for rel, want in _load(path)["code_provenance"]["method_files_sha256"].items():
            p = os.path.join(CSC_ROOT, rel)
            if not os.path.exists(p) or _sha256(p) != want:
                return False
    return True


BLAS_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
MAX_JOBS = 24


def execute(out=None, n_jobs=1, resume=False, partial_path=None, checkpoint_path=None):
    """GUARDED, PARALLEL, checkpointed real run. Returns 2 (fail-closed) unless frozen at TAG + clean tree +
    method/cache/engine hashes verified. With no csc-realeeg-v2 tag this ALWAYS refuses. A missing/duplicate/
    worker-errored cohort is an INFRASTRUCTURE failure (exit 2, NO endpoint evaluated) -- never a silent skip.
    The per-cohort science is byte-identical to the serial v1 (proven by the serial==parallel identity tests)."""
    cache_man = _load(CACHE_MANIFEST); prov = cache_man["provenance"]
    checks = {"git_frozen": verify_git_frozen()}
    if not _method_hashes_ok():
        checks["method_hashes"] = ["pinned method hash mismatch"]
    cpath = prov["cache_path"]
    if not os.path.exists(cpath) or _sha256(cpath) != prov["cache_sha256"]:
        checks["cache_hash"] = ["cache missing or sha256 mismatch"]
    # gating set must include random_label_control (P1.1)
    bank = _load(BANK_MANIFEST)
    if bank["gating_summary"]["gating_conditions"] != ["NULL_cov", "NULL_label", "NULL_cov_plus_label", "random_label_control"]:
        checks["gating_set"] = ["gating set is not the four frozen controls"]
    # H1: pinned injection+verdict engine hash must match before any run
    ep = bank.get("engine_provenance", {})
    engp = os.path.join(CSC_ROOT, ep.get("engine_file", "__missing__"))
    if not (os.path.exists(engp) and _sha256(engp) == ep.get("engine_sha256")):
        checks["engine_hash"] = ["injection+verdict engine hash mismatch"]
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        print(f"[run_realeeg EXECUTE] REFUSED (fail-closed): {bad}")
        return 2

    eff_jobs = max(1, min(int(n_jobs), MAX_JOBS))
    partial_path = partial_path or ((out + ".partial.jsonl") if out else None)
    checkpoint_path = checkpoint_path or ((out + ".checkpoint.json") if out else None)
    print(f"[run_realeeg EXECUTE] provenance clean; running the frozen real-feature validation "
          f"(v2 parallel, n_jobs={eff_jobs}, resume={resume}) ...")
    import numpy as np
    from . import realeeg_engine as EG
    b3 = _load(B3_MANIFEST); mlk = b3["method_lock"]
    ml = dict(min_confirm_pairs=20, pair_integrity_min=0.95, min_epochs=8, rank=3,
              C=mlk["regularisation_C"], n_folds=mlk["n_folds"],
              n_boot=b3["statistics"]["b_certifier_internal_null"],
              alpha_family=b3["statistics"]["alpha_family"],
              n_decision_budgets=len(mlk["positive_decision_budgets"]))
    npz = np.load(cpath)
    cache = dict(Z=npz["Z"], y=npz["y"], subject=npz["subject_id"], session=npz["session_id"])
    seed_base = bank["seed_schedule"]["realeeg_base_seed"]
    engine_file = os.path.join(CSC_ROOT, bank["engine_provenance"]["engine_file"])
    head = _git("rev-parse", "HEAD").stdout.strip()
    # binding stamped into the checkpoint; resume is refused unless ALL of these still match
    binding = dict(git_head=head, cache_sha256=prov["cache_sha256"],
                   bank_manifest_sha256=_sha256(BANK_MANIFEST), engine_sha256=_sha256(engine_file),
                   expected_code_ref=f"refs/tags/{TAG}", start_time=time.time())
    n_tasks_expected = len(EG.build_task_table(bank, seed_base))
    try:
        records, tth = EG.run_validation_parallel(
            cache, bank, ml, EG.frozen_A_cfg(), seed_base, n_jobs=eff_jobs,
            partial_path=partial_path, checkpoint_path=checkpoint_path, resume=resume,
            provenance=binding, progress=True)
    except EG.InfraError as e:
        print(f"[run_realeeg EXECUTE] INFRA-FAIL (no endpoint evaluated, exit 2): {e}")
        return 2
    verdict = EG.evaluate_verdict(records, bank)
    tagc = _git("rev-parse", f"{TAG}^{{commit}}")
    ss = bank["seed_schedule"]
    payload = dict(
        protocol="csc-realeeg", schema_version="v2", route_A_label_unit="trial",
        execution=dict(
            mode="cohort_parallel", performance_only_change=True, n_jobs=eff_jobs, joblib_backend="loky",
            n_tasks=len(records), n_tasks_expected=n_tasks_expected, task_table_sha256=tth,
            canonical_sort="(condition_index, cohort_index) == serial order (bootstrap array order preserved)",
            resumed=bool(resume), partial_path=partial_path, checkpoint_path=checkpoint_path,
            blas_threads={v: os.environ.get(v) for v in BLAS_ENV_VARS},
            note="v2 = performance-only refactor of v1: identical per-cohort science, parallel over independent "
                 "cohorts + canonical sort before verdict. feature/montage/cache/injections/RouteA/B3/"
                 "n_boot=200/cohorts=100/seed schedule/gates/alpha/denominators UNCHANGED from v1."),
        manifest_provenance=dict(
            cache_manifest_sha256=_sha256(CACHE_MANIFEST), bank_manifest_sha256=_sha256(BANK_MANIFEST),
            routeA_manifest_sha256=_sha256(ROUTEA_MANIFEST), routeB3_manifest_sha256=_sha256(B3_MANIFEST),
            engine_sha256=_sha256(engine_file), runner_sha256=_sha256(os.path.abspath(__file__)),
            cache_sha256=prov["cache_sha256"], cache_metadata_sha256=prov["cache_metadata_sha256"]),
        frozen_refs=dict(
            expected_code_ref=f"refs/tags/{TAG}", git_head=head,
            expected_code_commit=(tagc.stdout.strip() if tagc.returncode == 0 else None),
            git_status_clean=(_git("status", "--porcelain").stdout.strip() == ""),
            routeA_synthetic_tag="csc-confirmatory-v1/dee8958",
            routeB3_synthetic_tag="csc-b3-confirmatory-v1/0595f64",
            synthetic_tags_untouched=True, genuine_contrast_descriptive_only=True),
        slurm=dict(job_id=os.environ.get("SLURM_JOB_ID"),
                   hostname=os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME")),
        seed_schedule=dict(base_seed=ss["realeeg_base_seed"], conditions=len(bank["conditions"]),
                           cohorts_per_condition=bank["run_spec"]["cohorts_per_condition"],
                           condition_stride=ss["condition_stride"],
                           bootstrap_seed=ss.get("b_cohort_bootstrap_seed")),
        base_seed=seed_base, verdict=verdict, per_cohort=records,
        note="Package verdict driven by TIER1 B3 real-feature safety; power (TIER2) + Route A (TIER3) reported; genuine contrast descriptive-only; red-team re-aggregation required; NO clinical/PD claim.")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[run_realeeg EXECUTE] wrote {out}")
    print(f"[run_realeeg EXECUTE] package_verdict = {verdict['package_verdict']} (red-team still required)")
    return 0


def smoke():
    """Toy-cache self-test of the engine plumbing (smoke seed, never the real base seed / real cache)."""
    from . import realeeg_engine as EG
    EG.smoke()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CSC real-EEG validation runner (dry-run default; guarded --execute).")
    ap.add_argument("--execute", action="store_true", help="GUARDED parallel real run (fails closed without the frozen tag)")
    ap.add_argument("--smoke", action="store_true", help="toy-cache plumbing self-test (non-real seed)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                    help="parallel workers (capped at 24)")
    ap.add_argument("--resume", action="store_true",
                    help="resume a killed run from its partial/checkpoint (same frozen inputs only)")
    ap.add_argument("--partial", type=str, default=None, help="JSONL partial path (default: <out>.partial.jsonl)")
    ap.add_argument("--checkpoint", type=str, default=None, help="checkpoint path (default: <out>.checkpoint.json)")
    a = ap.parse_args()
    if a.smoke:
        sys.exit(smoke())
    if a.execute:
        sys.exit(execute(out=a.out, n_jobs=a.jobs, resume=a.resume,
                         partial_path=a.partial, checkpoint_path=a.checkpoint))
    sys.exit(0 if dry_run() else 1)
