"""
CSC Route B3 confirmatory runner (freeze package). Modes:
  (default)  DRY-RUN  : verify manifest hash + code-hash provenance + scenario match + full seed-schedule
                        disjointness, print the pre-registered plan. No compute, no artifact.
  --execute           : GUARDED real run. Verifies manifest hash, code hashes, HEAD == expected_code_ref
                        (the frozen tag) + clean tree, scenario match, seed disjointness; then runs EXACTLY
                        the frozen grid with the deterministic seed schedule, writes a fresh JSON artifact,
                        evaluates C1-C5 with CONSERVATIVE denominators (all generated clusters), and marks
                        C6 (independent red-team) as REQUIRED -> outputs a preliminary verdict EXCLUDING C6.
                        Without the frozen tag / clean tree / matching hashes it FAILS CLOSED (exit 2).
  --smoke <manifest>  : run the --execute machinery on a SMALL smoke manifest (different seed block, tiny
                        grid) to exercise compute/artifact/criteria WITHOUT the tag and WITHOUT base 1200000.

NEVER touches real EEG. NEVER touches the A tag csc-confirmatory-v1 / dee8958.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(HERE, "b3_confirmatory_manifest.json")


# ----------------------------------------------------------------------------- manifest + hashing
def load_manifest(path=MANIFEST):
    with open(path) as f:
        return json.load(f)


def canonical_manifest_hash(frozen_payload):
    """sha256 over the frozen payload only (metadata/manifest_hash excluded), canonical JSON."""
    blob = json.dumps(frozen_payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def verify_manifest_hash(mani):
    fp = mani.get("frozen_payload")
    if fp is None:
        return ["no frozen_payload"]
    got = canonical_manifest_hash(fp)
    return [] if got == mani.get("manifest_hash") else [("manifest_hash", mani.get("manifest_hash"), got)]


def verify_code_hashes(mani):
    fp = mani["frozen_payload"]
    bad = []
    for rel, want in fp["code_hashes_sha256"].items():
        p = os.path.join(ROOT, rel)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "MISSING"
        if got != want:
            bad.append((rel, want, got))
    return bad


def verify_scenarios(mani):
    from .run_b3_p23 import SCENARIOS
    cfg = mani["frozen_payload"]["scenario_configs"]
    return [(k, cfg.get(k), SCENARIOS.get(k)) for k in set(cfg) | set(SCENARIOS) if cfg.get(k) != SCENARIOS.get(k)]


# ----------------------------------------------------------------------------- seed schedule
def _cells(fp):
    g = fp["grid"]; budgets = g["decision_budgets"]
    cells = []
    for k in g["control_kinds"]:
        for s in g["control_scenarios"]:
            for m in budgets:
                cells.append(("control", k, s, m))
    for k in g["primary_positive_kinds"]:
        for s in g["primary_scenarios"]:
            for m in budgets:
                cells.append(("primary", k, s, m))
    for k in g["secondary_kinds"]:
        for s in g["control_scenarios"]:
            for m in budgets:
                cells.append(("secondary", k, s, m))
    return sorted(cells)                                   # canonical order


def generate_seed_schedule(fp):
    """Deterministic per-cell seeds: cluster_seed = base + stride*cell_index + replicate."""
    base = fp["seed_spec"]["base_seed"]; stride = fp["seed_spec"]["cell_stride"]
    reps = fp["seed_spec"]["replicates"]
    sched = []
    for i, (phase, kind, scen, m) in enumerate(_cells(fp)):
        sched.append(dict(cell_index=i, phase=phase, kind=kind, scenario=scen, budget=m,
                          seeds=[base + stride * i + r for r in range(reps)]))
    return sched


def verify_seed_schedule(fp):
    """Every seed unique AND disjoint from A confirmatory + B dev + smoke/test ranges."""
    sched = generate_seed_schedule(fp)
    all_seeds = [s for c in sched for s in c["seeds"]]
    errs = []
    if len(all_seeds) != len(set(all_seeds)):
        errs.append("duplicate seeds in schedule")
    ex = fp["seed_spec"]["development_seed_exclusion"]
    forbidden = set(range(0, 100000))                      # smoke/test
    forbidden |= set(range(ex["A_confirmatory"], ex["A_confirmatory"] + 1000))
    reps = fp["seed_spec"]["replicates"]; stride = fp["seed_spec"]["cell_stride"]
    for b in ex["B_development"]:
        forbidden |= set(range(b, b + max(stride, reps)))
    hit = set(all_seeds) & forbidden
    if hit:
        errs.append(f"seed overlap with excluded ranges: {sorted(hit)[:5]}...")
    return errs, len(all_seeds), (min(all_seeds), max(all_seeds))


# ----------------------------------------------------------------------------- git provenance (execute)
def _git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)


def verify_git_frozen(mani):
    """HEAD == expected_code_ref^{commit} AND tree clean. Fails closed if the tag does not exist."""
    ref = mani["frozen_payload"].get("expected_code_ref") or mani.get("expected_code_ref")
    errs = []
    tag = _git("rev-parse", f"{ref}^{{commit}}")
    if tag.returncode != 0:
        return [f"frozen tag {ref} does not exist (cannot execute)"]
    head = _git("rev-parse", "HEAD")
    if head.stdout.strip() != tag.stdout.strip():
        errs.append(f"HEAD {head.stdout.strip()[:12]} != {ref} {tag.stdout.strip()[:12]}")
    st = _git("status", "--porcelain")
    if st.stdout.strip():
        errs.append("working tree not clean")
    return errs


# ----------------------------------------------------------------------------- grid run + criteria
def run_grid(fp, n_jobs=1):
    """Run the frozen grid via the deterministic seed schedule. Heavy compute -- only from --execute/--smoke."""
    import numpy as np
    from csc.sim.shift_simulator import SimConfig, make_geom
    from .paired_sim import make_paired_target, PAIRED_TRUTH
    from .paired_calibrated import certify_paired_calibrated
    from .paired_certifier import CONCEPT_CONFIRMED
    ml = fp["method_lock"]; cfgs = fp["scenario_configs"]; sched = generate_seed_schedule(fp)

    def one(phase, kind, scen, m, seed):
        sc = cfgs[scen]
        cfg = SimConfig(seed=seed, subject_tau=sc.get("cfg_subject_tau", SimConfig.subject_tau),
                        epochs_min=sc.get("cfg_epochs_min", SimConfig.epochs_min),
                        epochs_max=sc.get("cfg_epochs_max", SimConfig.epochs_max))
        geom = make_geom(cfg, np.random.default_rng(seed))
        Z, Y, D, G, truth = make_paired_target(kind, geom, cfg, n_subjects=ml["n_subjects"],
                                               seed=10_000 + seed, cov_scale=sc.get("cov_scale", 10.0),
                                               base_prior=sc.get("base_prior"),
                                               label_noise=sc.get("label_noise", 0.0))
        log = certify_paired_calibrated(Z, Y, D, G, m=m, min_confirm_pairs=ml["min_confirm_pairs"],
                                        pair_integrity_min=ml["pair_integrity_min"],
                                        min_epochs=ml["min_epochs_per_condition"], rank=ml["rank"],
                                        C=ml["C"], n_folds=ml["n_folds"], n_boot=ml["n_boot"], seed=seed,
                                        alpha_family=ml["alpha_family"],
                                        n_decision_budgets=len(ml["positive_decision_budgets"]))
        return dict(phase=phase, kind=kind, scenario=scen, budget=int(m), truth=truth, seed=int(seed),
                    state=log["state"], confirmed=bool(log["state"] == CONCEPT_CONFIRMED),
                    n_sampler_failures=log.get("n_sampler_failures", 0),
                    n_boot_invalid=log.get("n_boot_invalid", 0))
    tasks = [(c["phase"], c["kind"], c["scenario"], c["budget"], s) for c in sched for s in c["seeds"]]
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs)(delayed(one)(*t) for t in tasks)
    return [one(*t) for t in tasks]


def evaluate_criteria(records, fp):
    """C1-C5 with CONSERVATIVE denominators (all generated clusters; non-confirm states count as non-fired).
    CP thresholds computed from realized denominators. C6 left to independent red-team."""
    from csc.protocol import _cp_bound
    ml = fp["method_lock"]; g = fp["grid"]; res = {}
    ctrl = [r for r in records if r["phase"] == "control"]
    prim = [r for r in records if r["phase"] == "primary"]
    # C1 guards
    guard_fc = sum(r["confirmed"] for r in ctrl if r["kind"] in ("missing_pair", "unequal_epochs_extreme"))
    res["C1_guards"] = dict(passed=bool(guard_fc == 0), false_confirms=int(guard_fc))
    # C2 control type-I (pointwise CP upper by kind x budget over all scenarios)
    c2 = {}; c2_pass = True
    for k in g["control_kinds"]:
        for m in g["decision_budgets"]:
            rs = [r for r in ctrl if r["kind"] == k and r["budget"] == m]
            n = len(rs); fc = sum(r["confirmed"] for r in rs)
            cpu = _cp_bound(fc, n, side="upper") if n else 1.0
            c2[f"{k}|m{m}"] = dict(fc=fc, n=n, cp_upper=cpu)
            c2_pass &= (cpu <= 0.05)
    res["C2_control_typeI"] = dict(passed=bool(c2_pass), by_kind_budget=c2)
    # C3 hot cells
    hot6 = []; hot3 = {}
    for r in ctrl:
        key = (r["scenario"], r["kind"], r["budget"])
        hot3[key] = hot3.get(key, 0) + r["confirmed"]
    ge6 = [k for k, v in hot3.items() if v >= 6]
    kb3 = {}
    for (s, k, m), v in hot3.items():
        if v >= 3:
            kb3[(k, m)] = kb3.get((k, m), 0) + 1
    kind_leak = [kb for kb, c in kb3.items() if c >= 2]
    res["C3_no_hot_cell"] = dict(passed=bool(not ge6 and not kind_leak), cells_ge6=ge6, kind_budget_leak=kind_leak)
    # C4 primary power (per kind x budget pooled over primary scenarios, CP lower >= 0.60; no cell < 0.50)
    c4 = {}; c4_pass = True
    for k in g["primary_positive_kinds"]:
        for m in g["decision_budgets"]:
            rs = [r for r in prim if r["kind"] == k and r["budget"] == m]
            n = len(rs); fc = sum(r["confirmed"] for r in rs)
            cpl = _cp_bound(fc, n, side="lower") if n else 0.0
            cells = {}
            for s in g["primary_scenarios"]:
                cr = [r for r in rs if r["scenario"] == s]
                cells[s] = (sum(x["confirmed"] for x in cr) / len(cr)) if cr else 0.0
            cell_ok = all(v >= 0.50 for v in cells.values())
            c4[f"{k}|m{m}"] = dict(fc=fc, n=n, power=fc / n if n else None, cp_lower=cpl,
                                   per_scenario_power=cells, cell_floor_ok=cell_ok)
            c4_pass &= (cpl >= 0.60 and cell_ok)
    res["C4_primary_power"] = dict(passed=bool(c4_pass), by_kind_budget=c4)
    # C5 no silent failure
    bad_state = [r["state"] for r in records if r["state"] not in
                 ("CONCEPT_CONFIRMED", "NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT", "NEED_MORE_LABELS",
                  "INVALID_PAIR_STRUCTURE", "UNIDENTIFIABLE")]
    sfail = sum(r["n_sampler_failures"] for r in records)
    res["C5_no_silent_failure"] = dict(passed=bool(not bad_state and sfail == 0),
                                       sampler_failures=int(sfail), unknown_states=len(bad_state))
    res["C6_independent_verification"] = dict(passed=None, note="REQUIRES independent red-team re-aggregation")
    preliminary = all(res[c]["passed"] for c in
                      ("C1_guards", "C2_control_typeI", "C3_no_hot_cell", "C4_primary_power", "C5_no_silent_failure"))
    return dict(preliminary_scientific_verdict_excluding_C6=("PASS" if preliminary else "FAIL"),
                red_team_required=True, criteria=res)


# ----------------------------------------------------------------------------- modes
def dry_run(path=MANIFEST):
    mani = load_manifest(path); fp = mani["frozen_payload"]
    print(f"[b3-confirmatory DRY-RUN] {os.path.relpath(path, ROOT)} v{mani.get('version')} ({mani.get('status')})")
    bad_m = verify_manifest_hash(mani); bad_h = verify_code_hashes(mani); bad_s = verify_scenarios(mani)
    serr, nseed, srange = verify_seed_schedule(fp)
    print(f"  manifest hash         : {'OK' if not bad_m else 'FAIL ' + str(bad_m)}")
    print(f"  code-hash provenance  : {'OK' if not bad_h else 'FAIL ' + str(bad_h)}")
    print(f"  scenario-config match : {'OK' if not bad_s else 'FAIL ' + str(bad_s)}")
    print(f"  seed schedule         : {'OK' if not serr else 'FAIL ' + str(serr)}  ({nseed} seeds, range {srange})")
    ml = fp["method_lock"]; g = fp["grid"]
    print(f"  method {ml['method']} calib {fp['calibration_version']} ; base_seed {fp['seed_spec']['base_seed']}")
    print(f"  cells {len(_cells(fp))} x {fp['seed_spec']['replicates']} = {nseed} clusters "
          f"(controls {len(g['control_kinds'])}x{len(g['control_scenarios'])}x{len(g['decision_budgets'])}, "
          f"primary {len(g['primary_positive_kinds'])}x{len(g['primary_scenarios'])}x{len(g['decision_budgets'])})")
    ok = not (bad_m or bad_h or bad_s or serr)
    print(f"[b3-confirmatory DRY-RUN] provenance {'CLEAN' if ok else 'NOT CLEAN -- fail closed'}")
    return ok


def execute(path=MANIFEST, n_jobs=1, out=None, require_git=True, quiet=True):
    """GUARDED execute. Returns 0 on success, 2 on fail-closed provenance error."""
    if quiet:
        warnings.filterwarnings("ignore")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    mani = load_manifest(path); fp = mani["frozen_payload"]
    checks = {"manifest_hash": verify_manifest_hash(mani), "code_hashes": verify_code_hashes(mani),
              "scenarios": verify_scenarios(mani), "seed_schedule": verify_seed_schedule(fp)[0]}
    if require_git:
        checks["git_frozen"] = verify_git_frozen(mani)
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        print(f"REFUSED (fail-closed): provenance checks failed: {bad}")
        return 2
    print(f"[b3-confirmatory EXECUTE] provenance clean; running frozen grid ({len(_cells(fp))} cells)...")
    records = run_grid(fp, n_jobs=n_jobs)
    verdict = evaluate_criteria(records, fp)
    payload = dict(protocol=fp["protocol"], version=fp["version"], manifest_hash=mani["manifest_hash"],
                   calibration_version=fp["calibration_version"], base_seed=fp["seed_spec"]["base_seed"],
                   n_clusters=len(records), verdict=verdict, per_cluster=records,
                   note="preliminary verdict EXCLUDES C6 (independent red-team required); synthetic only; NO real EEG")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3-confirmatory EXECUTE] wrote {out}")
    print(f"[b3-confirmatory EXECUTE] preliminary verdict (EXCLUDING C6) = "
          f"{verdict['preliminary_scientific_verdict_excluding_C6']} ; red_team_required = True")
    return 0


def main():
    ap = argparse.ArgumentParser(description="CSC B3 confirmatory runner (dry-run default; guarded --execute).")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--smoke", type=str, default=None, help="smoke manifest path (runs execute machinery, no git guard)")
    ap.add_argument("--manifest", type=str, default=MANIFEST)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.smoke:
        assert "1200000" not in open(a.smoke).read(), "smoke manifest must not use the real base_seed 1200000"
        sys.exit(execute(a.smoke, n_jobs=a.jobs, out=a.out, require_git=False))
    if a.execute:
        sys.exit(execute(a.manifest, n_jobs=a.jobs, out=a.out, require_git=True))
    sys.exit(0 if dry_run(a.manifest) else 1)


if __name__ == "__main__":
    main()
