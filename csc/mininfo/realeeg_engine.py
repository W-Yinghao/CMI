"""CSC real-EEG validation ENGINE (pre-reg v4, P1.2). Implements the semi-synthetic injected bank on REAL
Lee2019 features + the Route A / Route B3 certifier invocations + cohort-level bounds + verdict.

EXECUTABLE but GUARDED: this module is imported by run_realeeg_validation.py, which only calls run_validation()
from the fail-closed --execute path (git-frozen tag + clean tree + cache/manifest hashes). It is NOT run here.
`smoke()` exercises the full plumbing on a TINY synthetic toy cache (smoke seed, never the real base seed / real
cache) so the wiring is verified without the authorized real run.

Injection semantics are frozen to the bank manifest; they are scrutinized by the P1.2 audit red-team BEFORE any
tag is created. Method locks (B3 certify_paired_calibrated, Route A run_frozen_protocol) are byte-unchanged.
"""
import hashlib, json, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression

REAL_BASE_SEED = 20_000_000        # must match the bank manifest; smoke must never use this


# ---------------------------------------------------------------- feature/label injection helpers
def _pooled_clf(Z, y):
    return LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs").fit(Z, y)


def _draw(proba, rng):
    return (rng.random(len(proba)) < proba).astype(np.int64)


def _rotate_proba(clf, Z, rng, deg=25.0):
    """Rotate the logistic boundary by `deg` in the plane of its top-2 feature-variance axes -> concept shift."""
    w = clf.coef_.ravel().astype(float)
    U, _, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)  # not used; keep boundary in feature space
    d = len(w)
    # build an orthonormal pair (w_hat, u) and rotate w toward u
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    u = rng.standard_normal(d); u -= (u @ w_hat) * w_hat; u /= (np.linalg.norm(u) + 1e-12)
    th = np.deg2rad(deg)
    w_rot = np.linalg.norm(w) * (np.cos(th) * w_hat + np.sin(th) * u)
    logit = Z @ w_rot + clf.intercept_[0]
    return 1.0 / (1.0 + np.exp(-logit))


def _prior_resample(Z, y, subj, sess, target_prior_by_sess, rng):
    """Resample trials within each (subject,session) to a target class prior (P(Z|Y) fixed, P(Y) shifted)."""
    keep = []
    for s in np.unique(subj):
        for ss in np.unique(sess):
            m = (subj == s) & (sess == ss)
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            p1 = target_prior_by_sess.get(int(ss), 0.5)
            n = len(idx)
            n1 = max(1, min(n - 1, int(round(p1 * n))))
            i1 = idx[y[idx] == 1]; i0 = idx[y[idx] == 0]
            if len(i1) == 0 or len(i0) == 0:
                keep.extend(idx.tolist()); continue
            sel = np.concatenate([rng.choice(i1, n1, replace=True), rng.choice(i0, n - n1, replace=True)])
            keep.extend(sel.tolist())
    keep = np.array(keep)
    return Z[keep], y[keep], subj[keep], sess[keep]


# ---------------------------------------------------------------- cohort builders (return Z, Y, D, G)
def build_cohort(condition, cohort, rng):
    """cohort = dict with Z, y, subject, session for a set of subjects (both sessions).
    Returns (Z, Y, D, G) for the B3 certifier (D=condition code, G=subject). Y may be real or injected."""
    Z, y, subj, sess = cohort["Z"], cohort["y"], cohort["subject"], cohort["session"]

    if condition in ("genuine_session_contrast_descriptive",):
        return Z, y, sess.copy(), subj.copy()

    if condition == "NULL_real_session":                      # within one session, random stratified A/B split
        m = sess == 1
        Zs, ys, gs = Z[m], y[m], subj[m]
        D = np.zeros(len(ys), dtype=np.int64)
        for g in np.unique(gs):
            for c in (0, 1):
                ii = np.where((gs == g) & (ys == c))[0]
                D[ii[rng.permutation(len(ii))[: len(ii) // 2]]] = 1
        return Zs, ys, D, gs

    if condition == "random_label_control":                   # marginal real, labels destroyed
        return Z, rng.permutation(y), sess.copy(), subj.copy()

    if condition == "NULL_label":                             # prior shift across sessions, P(Z|Y) fixed
        Zr, yr, gr, sr = _prior_resample(Z, y, subj, sess, {1: 0.35, 2: 0.65}, rng)
        return Zr, yr, sr, gr

    # conditions that draw Y* from a pooled boundary
    clf = _pooled_clf(Z, y)
    p = clf.predict_proba(Z)[:, 1]

    if condition == "NULL_cov":                               # real covariate (sessions) + pooled Y* -> no concept
        return Z, _draw(p, rng), sess.copy(), subj.copy()

    if condition == "NULL_cov_plus_label":
        Ystar = _draw(p, rng)
        Zr, yr, gr, sr = _prior_resample(Z, Ystar, subj, sess, {1: 0.35, 2: 0.65}, rng)
        return Zr, yr, sr, gr

    if condition in ("POS_concept", "POS_concept_plus_cov"):  # rotate boundary in session 2 -> concept
        p2 = _rotate_proba(clf, Z, rng, deg=25.0)
        Ystar = _draw(np.where(sess == 2, p2, p), rng)
        return Z, Ystar, sess.copy(), subj.copy()

    if condition == "POS_pure_conditional":                   # relabel changes P(Y|Z), marginal held (approx)
        p2 = _rotate_proba(clf, Z, rng, deg=40.0)
        Ystar = _draw(np.where(sess == 2, p2, p), rng)
        return Z, Ystar, sess.copy(), subj.copy()

    raise ValueError(f"unknown condition {condition}")


# ---------------------------------------------------------------- certifier wrappers
def certify_b3(Z, Y, D, G, m, seed, ml):
    from .paired_calibrated import certify_paired_calibrated
    from .paired_certifier import CONCEPT_CONFIRMED
    try:
        log = certify_paired_calibrated(
            Z, Y, D, G, m=m, min_confirm_pairs=ml["min_confirm_pairs"],
            pair_integrity_min=ml["pair_integrity_min"], min_epochs=ml["min_epochs"],
            rank=ml["rank"], C=ml["C"], n_folds=ml["n_folds"], n_boot=ml["n_boot"], seed=seed,
            alpha_family=ml["alpha_family"], n_decision_budgets=ml["n_decision_budgets"])
        return dict(state=log["state"], confirmed=bool(log["state"] == CONCEPT_CONFIRMED),
                    n_sampler_failures=log.get("n_sampler_failures", 0), n_boot_invalid=log.get("n_boot_invalid", 0))
    except Exception as e:
        return dict(state=f"ENGINE_ERROR:{type(e).__name__}", confirmed=False, error=str(e)[:200])


def frozen_A_cfg():
    """FROZEN Route-A config for Lee2019 MI. SAME byte-frozen A code as the synthetic dee8958 line; the
    label-generating-unit DECLARATION is matched to the substrate: MI labels are TRIAL-level (each subject
    has both L and R trials), so label_unit='trial'. analysis_unit stays 'subject' -> subject-clustered
    inference (trials are NOT treated as independent). This is a TRANSFER DIAGNOSTIC on a trial-label real
    substrate, NOT a same-config revalidation of the synthetic subject-label A (which used label_unit='subject').
    """
    from .. import protocol as P
    return P.ProtocolConfig(n_boot=40, n_dir_boot=120, target_n_boot=120, tau_n_pseudotargets=240,
                            label_unit="trial", analysis_unit="subject")


def certify_A(Z, Y, D, G, seed, cfg):
    """Route A source-anchored: source = session 1 (subject-as-domain), target = session 2 marginal.
    (Design mapping -- documented in pre-reg v4; scrutinized by the P1.2 audit.)"""
    from .. import protocol as P
    from ..certificate import CONCEPT_SUSPECT
    # GUARDRAIL: MI is trial-level; label_unit MUST be 'trial'. Fail closed BEFORE any certificate is evaluated
    # if 'subject' is used on this substrate (it would otherwise raise LabelUnitError deep inside).
    if getattr(cfg, "label_unit", None) != "trial":
        return dict(state="REFUSED_label_unit_must_be_trial_for_MI", confirmed=False,
                    error=f"Route A on Lee2019 MI requires label_unit='trial'; got {getattr(cfg,'label_unit',None)!r}")
    src = (D == 1); tgt = (D == 2)
    if src.sum() == 0 or tgt.sum() == 0:                  # within-session condition -> A (cross-session) N/A
        return dict(state="NOT_APPLICABLE_route_A_needs_cross_session", confirmed=False)
    Z_src, Y_src, G_src = Z[src], Y[src], G[src]
    D_src = G_src.copy()                                  # subject-as-domain in the source session
    Z_tgt, G_tgt = Z[tgt], G[tgt]
    tgt_cond = np.zeros(len(Z_tgt), dtype=np.int64)       # single target condition (session 2), required by API
    try:
        out = P.run_frozen_protocol(Z_src, Y_src, D_src, Z_tgt, cfg, src_group_ids=G_src,
                                    tgt_group_ids=G_tgt, tgt_condition_ids=tgt_cond, seed=seed)
        cert = out.get("certificate")
        bare = getattr(cert, "state", None) if cert is not None else out.get("state")
        bare = str(bare) if bare is not None else str(out)     # BARE state string (A1 fix; not the full repr)
        return dict(state=bare, confirmed=bool(bare == CONCEPT_SUSPECT))
    except Exception as e:
        return dict(state=f"ENGINE_ERROR:{type(e).__name__}", confirmed=False, error=str(e)[:200])


# ---------------------------------------------------------------- cohort bootstrap upper bound
def cohort_bootstrap_upper(cohort_fired, B, seed, alpha=0.05):
    """COHORT-level bootstrap: resample the per-cohort fired flags (each cohort = a fresh subject subsample of
    the run_spec) and return the (1-alpha) upper quantile of the mean fire rate. HONEST SCOPE (P1.3 blocker 4):
    subjects are the sampling unit WITHIN cohort generation, but this aggregate CI is COHORT-level, NOT a formal
    subject-cluster (subject-block) bound -- it is a descriptive real-feature safety bound. A true
    subject-block cluster bound is a future refinement, NOT claimed here."""
    x = np.asarray(cohort_fired, dtype=float)
    if len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(np.quantile(boots, 1 - alpha))


# ---------------------------------------------------------------- smoke (toy cache; NOT the real run)
def _toy_cache(rng, n_subj=8, n_trials=24, d=16):
    Z, y, subj, sess = [], [], [], []
    for s in range(1, n_subj + 1):
        for ss in (1, 2):
            zz = rng.standard_normal((n_trials, d)) + (0.4 if ss == 2 else 0.0)  # mild covariate drift
            yy = (zz[:, 0] + 0.5 * rng.standard_normal(n_trials) > 0).astype(np.int64)
            Z.append(zz); y.append(yy); subj.append(np.full(n_trials, s)); sess.append(np.full(n_trials, ss))
    return dict(Z=np.vstack(Z), y=np.concatenate(y), subject=np.concatenate(subj), session=np.concatenate(sess))


def _cp_upper(k, n, a=0.05):
    from scipy.stats import beta
    if n == 0:
        return float("nan")
    return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))


def run_validation(cache, bank, b3_ml, cfg_A, seed_base):
    """HEAVY real run -- ONLY from the guarded --execute path. Loops conditions x R cohorts x {B3, A}."""
    rs = bank["run_spec"]
    R, m = rs["cohorts_per_condition"], rs["subjects_per_cohort"]
    subjects = np.unique(cache["subject"])
    conds = bank["conditions"]
    records = []
    for ci, cond in enumerate(conds):
        name = cond["name"]; routes = cond["routes"]; gating = bool(cond["gating"])
        genuine = name == "genuine_session_contrast_descriptive"
        n_cohorts = 1 if genuine else R
        for r in range(n_cohorts):
            seed = seed_base + ci * bank["seed_schedule"]["condition_stride"] + r
            rng = np.random.default_rng(seed)
            subj = subjects if genuine else rng.choice(subjects, size=min(m, len(subjects)), replace=False)
            sel = np.isin(cache["subject"], subj)
            coh = {k: cache[k][sel] for k in ("Z", "y", "subject", "session")}
            Z, Y, D, G = build_cohort(name, coh, rng)
            rec = dict(condition=name, gating=gating, cohort=r, seed=int(seed), ground_truth=cond["ground_truth"])
            if "B3" in routes:
                rec["B3"] = certify_b3(Z, Y, D, G, m=len(subj), seed=seed, ml=b3_ml)
            if "A" in routes:
                rec["A"] = certify_A(Z, Y, D, G, seed=seed, cfg=cfg_A)
            records.append(rec)
    return records


# B3's 5-state set. DECIDED = the two states where the test actually ran and decided; the other three are
# abstain/invalid and must NOT pad the type-I denominator (they route into invalid_frac). (V1 fix)
B3_DECIDED = ("CONCEPT_CONFIRMED", "NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT")
B3_ABSTAIN_INVALID = ("NEED_MORE_LABELS", "INVALID_PAIR_STRUCTURE", "UNIDENTIFIABLE")
B3_VALID_5SET = set(B3_DECIDED) | set(B3_ABSTAIN_INVALID)
_ERR = ("ENGINE_ERROR", "NOT_APPLICABLE", "REFUSED")


def _b3_rates(records, cond, B_cohort, seed, cap):
    rs = [r["B3"] for r in records if r["condition"] == cond and "B3" in r]
    states = [str(x["state"]) for x in rs]
    decided = [x for x, s in zip(rs, states) if s in B3_DECIDED]          # denominator = DECIDED only (V1)
    n, k = len(decided), sum(1 for x in decided if x["confirmed"])
    absinv = sum(1 for s in states if s in B3_ABSTAIN_INVALID)
    err = sum(1 for s in states if s.startswith(_ERR))
    invalid_frac = (absinv + err) / len(rs) if rs else 1.0
    fired = np.array([1 if x["confirmed"] else 0 for x in decided], dtype=float)
    upper = cohort_bootstrap_upper(fired, B_cohort, seed) if n > 0 else float("nan")   # cohort bootstrap (V2; not subject-cluster)
    out_of_set = sorted({s for s in states if s not in B3_VALID_5SET and not s.startswith(_ERR)})
    if invalid_frac > cap or n == 0:
        status = "INCONCLUSIVE"
    elif out_of_set:
        status = "FAIL_silent_state"
    elif upper <= 0.05:
        status = "PASS"
    else:
        status = "FAIL"
    return dict(n_valid=n, n_confirmed=k, rate=(k / n if n else float("nan")), boot_upper=upper,
                invalid_frac=invalid_frac, out_of_set_states=out_of_set, status=status)


def _a_rates(records, cond):
    rs = [r["A"] for r in records if r["condition"] == cond and "A" in r]
    states = [str(x["state"]) for x in rs]
    valid = [x for x, s in zip(rs, states) if not s.startswith(_ERR)]
    n, k = len(valid), sum(1 for x in valid if x["confirmed"])
    abst = sum(1 for x in valid if str(x["state"]).startswith("UNIDENT"))
    return dict(n_valid=n, n_confirmed=k, rate=(k / n if n else float("nan")),
                abstention=abst, na_or_error=len(rs) - n)


def evaluate_verdict(records, bank):
    """3-tier. TIER1 B3 safety GATES the package (COHORT bootstrap upper on the 4 gating nulls,
    DECIDED-only denominators, invalid cap -> INCONCLUSIVE); TIER2 B3 power REPORTED; TIER3 Route A trial-label
    diagnostic REPORTED. Package = FAIL if any gating FAIL/silent-state, INCONCLUSIVE elif any INCONCLUSIVE,
    else PASS."""
    rs = bank["run_spec"]; cap = rs["invalid_fraction_cap"]; B_cohort = rs["b_cohort_bootstrap"]
    gating = bank["gating_summary"]["gating_conditions"]
    seed0 = bank["seed_schedule"]["realeeg_base_seed"] + 900_000_000   # bootstrap-seed offset (disjoint)
    tier1 = {c: _b3_rates(records, c, B_cohort, seed0 + i, cap) for i, c in enumerate(gating)}
    statuses = [v["status"] for v in tier1.values()]
    package = ("FAIL" if any(s.startswith("FAIL") for s in statuses)
               else "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "PASS")
    tier2 = {c: _b3_rates(records, c, B_cohort, seed0 + 100 + i, cap)
             for i, c in enumerate(("POS_concept", "POS_concept_plus_cov"))}
    tier3 = {c: _a_rates(records, c) for c in gating + ["POS_concept", "POS_concept_plus_cov"]
             if any(r["condition"] == c and "A" in r for r in records)}
    genuine = [r for r in records if r["condition"] == "genuine_session_contrast_descriptive"]
    return dict(
        package_verdict=package,
        tier1_B3_safety_gating=dict(per_condition=tier1,
                                    denominator_rule="DECIDED-only (abstain/invalid -> invalid_frac, capped -> INCONCLUSIVE); COHORT bootstrap upper (subjects sampled within cohort generation; aggregate CI is cohort-level, NOT a formal subject-cluster bound)"),
        tier2_B3_power_reported=tier2,
        tier3_routeA_trial_label_diagnostic=dict(
            per_condition=tier3,
            interpretation="transfer diagnostic on a trial-label real substrate; NOT a revalidation of the subject-label synthetic A; cautious/abstaining behavior is consistent with the A-line result but does not alone revalidate A"),
        genuine_contrast_descriptive=dict(
            b3=[r.get("B3", {}).get("state") for r in genuine],
            a=[r.get("A", {}).get("state") for r in genuine],
            note="DESCRIPTIVE only; a real CONCEPT_CONFIRMED is not validated truth"),
        note="Package verdict driven by TIER1 B3 real-feature safety (gating). Power (TIER2) and Route A (TIER3) REPORTED, non-gating. Red-team re-aggregation required for the final verdict.")


# ================================================================ v2: PERFORMANCE-ONLY parallel execution
# The serial run_validation() above is the FROZEN scientific reference (byte-unchanged from v1). v2 adds a
# cohort-level PARALLEL executor that is NUMERICALLY IDENTICAL: each cohort's rng = default_rng(seed) is created
# fresh inside the serial loop and NEVER carries state across cohorts, so every cohort is a pure deterministic
# function of (condition, seed, cache). Parallelism therefore cannot change any per-cohort record; a canonical
# sort by (condition_index, cohort_index) restores the exact serial record ORDER (required because the cohort
# bootstrap resamples the fired-flag ARRAY, whose order must match serial for byte-identity). No feature /
# montage / cache / injection / Route-A / B3 / n_boot / cohort-count / seed / gate / alpha / denominator changes.
class InfraError(RuntimeError):
    """Infrastructure/assembly failure (missing/duplicate/worker-errored task). NOT a scientific verdict:
    the runner maps this to exit 2 and evaluates NO endpoint."""


def build_task_table(bank, seed_base):
    """Canonical, deterministic task table: one entry per (condition, cohort_index), in serial order
    (condition_index asc, cohort_index asc). A task is fully self-describing: (condition, seed, routes)."""
    R = bank["run_spec"]["cohorts_per_condition"]
    stride = bank["seed_schedule"]["condition_stride"]
    tasks = []
    for ci, cond in enumerate(bank["conditions"]):
        # identity guard: the serial loop uses the ENUMERATE index for the seed; it must equal condition_index
        if cond.get("condition_index", ci) != ci:
            raise InfraError(f"condition order != condition_index at {ci}: {cond['name']}")
        name = cond["name"]
        n_cohorts = 1 if name == "genuine_session_contrast_descriptive" else R
        for r in range(n_cohorts):
            seed = seed_base + ci * stride + r
            tasks.append(dict(task_id=f"{ci:02d}:{name}:{r:04d}", condition_index=ci, cohort_index=r,
                              condition=name, seed=int(seed), routes=list(cond["routes"]),
                              is_gating=bool(cond["gating"]), ground_truth=cond["ground_truth"]))
    tasks.sort(key=lambda t: (t["condition_index"], t["cohort_index"]))
    return tasks


def task_table_sha256(tasks):
    """Stable hash of the canonical task table (the identity-bearing fields only)."""
    canon = [[t["task_id"], t["condition"], t["cohort_index"], t["seed"], list(t["routes"]), bool(t["is_gating"])]
             for t in tasks]
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_one_task(task, cache, m, b3_ml, cfg_A):
    """PURE per-cohort task: reproduces the serial run_validation body for ONE task EXACTLY. Returns a record
    with the SAME scientific keys as serial (+ task_id / condition_index for canonical ordering). Any crash
    OUTSIDE the certifiers (selection / build_cohort / OOM) is captured as __worker_error__ so assembly fails
    CLOSED -- a cohort is never silently dropped. (Certifier-internal errors already return ENGINE_ERROR states.)"""
    name = task["condition"]; seed = task["seed"]; routes = task["routes"]
    genuine = name == "genuine_session_contrast_descriptive"
    try:
        subjects = np.unique(cache["subject"])
        rng = np.random.default_rng(seed)
        subj = subjects if genuine else rng.choice(subjects, size=min(m, len(subjects)), replace=False)
        sel = np.isin(cache["subject"], subj)
        coh = {k: cache[k][sel] for k in ("Z", "y", "subject", "session")}
        Z, Y, D, G = build_cohort(name, coh, rng)
        rec = dict(task_id=task["task_id"], condition_index=task["condition_index"],
                   condition=name, gating=bool(task["is_gating"]), cohort=task["cohort_index"],
                   seed=int(seed), ground_truth=task["ground_truth"])
        if "B3" in routes:
            rec["B3"] = certify_b3(Z, Y, D, G, m=len(subj), seed=seed, ml=b3_ml)
        if "A" in routes:
            rec["A"] = certify_A(Z, Y, D, G, seed=seed, cfg=cfg_A)
        return rec
    except Exception as e:                          # pragma: no cover -- defensive; assembly turns this into InfraError
        return dict(task_id=task["task_id"], __worker_error__=f"{type(e).__name__}: {str(e)[:300]}")


def _read_jsonl(path):
    out = []
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_checkpoint(path, tasks, records, prov):
    if not path:
        return
    done = sorted(r["task_id"] for r in records if "__worker_error__" not in r)
    failed = sorted(r["task_id"] for r in records if "__worker_error__" in r)
    ck = dict(task_table_sha256=prov.get("task_table_sha256"), n_total=len(tasks),
              n_completed=len(done), n_failed=len(failed),
              completed_task_ids=done, failed_task_ids=failed,
              last_update_time=time.time())
    for k in ("start_time", "git_head", "cache_sha256", "bank_manifest_sha256", "engine_sha256",
              "expected_code_ref"):
        if k in prov:
            ck[k] = prov[k]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f, indent=2, default=str)
    os.replace(tmp, path)


def _assemble_records(tasks, records):
    """Fail-closed: exactly one non-errored record per task, no duplicate, no missing, no worker error.
    Returns records canonically sorted (condition_index, cohort_index) == serial order."""
    by_id, errors = {}, []
    for rec in records:
        if "__worker_error__" in rec:
            errors.append((rec.get("task_id"), rec["__worker_error__"])); continue
        tid = rec["task_id"]
        if tid in by_id:
            raise InfraError(f"duplicate task record: {tid}")
        by_id[tid] = rec
    if errors:
        raise InfraError(f"{len(errors)} worker task error(s); first {errors[0]}")
    want = {t["task_id"] for t in tasks}; got = set(by_id)
    if want - got:
        raise InfraError(f"{len(want - got)} missing task record(s); e.g. {sorted(want - got)[:3]}")
    if got - want:
        raise InfraError(f"unexpected task record(s): {sorted(got - want)[:3]}")
    return sorted(by_id.values(), key=lambda r: (r["condition_index"], r["cohort"]))


def run_validation_parallel(cache, bank, b3_ml, cfg_A, seed_base, n_jobs=1,
                            partial_path=None, checkpoint_path=None, resume=False,
                            provenance=None, progress=True):
    """PERFORMANCE-ONLY parallel driver. Streams each cohort record to `partial_path` (JSONL) and periodically
    rewrites `checkpoint_path`, so an infra kill leaves verifiable progress. Returns (ordered_records, tth).
    Resume is allowed ONLY when the checkpoint binds to the SAME task-table / tag / cache / manifest / engine."""
    from joblib import Parallel, delayed
    m = bank["run_spec"]["subjects_per_cohort"]
    tasks = build_task_table(bank, seed_base)
    tth = task_table_sha256(tasks)
    prov = dict(provenance or {}); prov["task_table_sha256"] = tth
    prov.setdefault("start_time", time.time())

    done = {}
    if resume and partial_path and os.path.exists(partial_path):
        ck = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                ck = json.load(f)
        if ck.get("task_table_sha256") != tth:
            raise InfraError("resume refused: checkpoint task_table_sha256 mismatch (frozen inputs changed)")
        for key in ("git_head", "cache_sha256", "bank_manifest_sha256", "engine_sha256", "expected_code_ref"):
            # FAIL CLOSED on ABSENCE too (red-team v2): if the current run binds this field, a checkpoint that
            # is missing/nulls it (truncated / hand-edited / written by a caller that omitted it) must be
            # REFUSED, not silently accepted -- else a resume could mix records from a changed frozen input.
            if key in prov and ck.get(key) != prov.get(key):
                raise InfraError(f"resume refused: checkpoint {key} mismatch or missing")
        for rec in _read_jsonl(partial_path):
            if "__worker_error__" not in rec and rec.get("task_id"):
                done[rec["task_id"]] = rec
        if progress:
            print(f"[realeeg-v2] resume: {len(done)}/{len(tasks)} cohorts already done", flush=True)
    else:
        if partial_path:                              # fresh: truncate any stale partial
            open(partial_path, "w").close()

    remaining = [t for t in tasks if t["task_id"] not in done]
    records = list(done.values())
    n_total = len(tasks)
    chunk = max(1, int(n_jobs) * 4)                   # periodic checkpoint granularity
    for i in range(0, len(remaining), chunk):
        batch = remaining[i:i + chunk]
        out = Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(run_one_task)(t, cache, m, b3_ml, cfg_A) for t in batch)
        if partial_path:
            with open(partial_path, "a") as f:
                for rec in out:
                    f.write(json.dumps(rec, default=str) + "\n")
        records.extend(out)
        _write_checkpoint(checkpoint_path, tasks, records, prov)
        errs = [r for r in out if "__worker_error__" in r]
        if progress:
            print(f"[realeeg-v2] {len([r for r in records if '__worker_error__' not in r])}/{n_total} "
                  f"cohorts done ({len(errs)} worker-error this batch)", flush=True)
        if errs:                                      # fail FAST + closed on a genuine worker crash
            raise InfraError(f"worker task error: {errs[0].get('task_id')}: {errs[0]['__worker_error__']}")
    ordered = _assemble_records(tasks, records)
    return ordered, tth


def smoke(seed=111):
    assert seed != REAL_BASE_SEED, "smoke must not use the real base seed"
    rng = np.random.default_rng(seed)
    cohort = _toy_cache(rng)
    ml = dict(min_confirm_pairs=4, pair_integrity_min=0.95, min_epochs=8, rank=3, C=0.5,
              n_folds=3, n_boot=40, alpha_family=0.05, n_decision_budgets=2)
    import csc.protocol as P
    cfg = P.ProtocolConfig(n_boot=20, n_dir_boot=40, target_n_boot=40, tau_n_pseudotargets=60,
                           label_unit="trial", analysis_unit="subject")   # MI trial-label declaration
    conds = ["NULL_real_session", "NULL_cov", "NULL_label", "random_label_control",
             "POS_concept", "genuine_session_contrast_descriptive"]
    out = {}
    for c in conds:
        Z, Y, D, G = build_cohort(c, cohort, np.random.default_rng(seed + hash(c) % 1000))
        b3 = certify_b3(Z, Y, D, G, m=8, seed=seed, ml=ml)
        a = certify_A(Z, Y, D, G, seed=seed, cfg=cfg)
        out[c] = dict(b3=b3["state"], A=a["state"])
        print(f"  smoke[{c}]: B3={b3['state']} | A={a['state']}")
    print("SMOKE_OK (engine plumbing ran on toy cache; NOT the real validation)")
    return out


if __name__ == "__main__":
    smoke()
