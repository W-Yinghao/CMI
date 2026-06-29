"""
csc.run_confirmatory — FROZEN confirmatory runner for the concept-shift certificate.

Reads the canonical frozen spec `csc/confirmatory_tag.json` (reviewer-bound: K=1, P_baseline only,
G=66, N_valid_min=59, source_invalid_cap=0.10, max_forbidden_failures=0, power_bar=0.50,
base_seed=900000, pointwise). It validates the running method against the spec's
`expected_manifest_hash`, then — ONLY under `--execute` — generates the unseen core clusters and
evaluates the two endpoints, computing the power threshold from the **realized N_valid** (never
hard-coded 37/59).

SAFETY: this module does NOT run on import and does NOT run the confirmatory sweep by default. The
default CLI is a DRY-RUN that prints the frozen plan + the realized-N_valid power-threshold table.
`--execute` is the SEPARATELY-AUTHORIZED confirmatory run; it additionally fail-closes unless the
frozen manifest matches and the base_seed is the pre-registered unseen value.

NO FREEZE / NO CONFIRMATORY RUN / NO P2 is implied by the presence of this file.
"""
from __future__ import annotations

import argparse
import json
import os

from csc.protocol import ProtocolConfig, _cp_bound
from csc.certificate import FORBIDDEN, CONCEPT_SUSPECT
from csc.run_envelope import EnvelopePoint, _cluster_record, KINDS
from csc.sim.shift_simulator import _TRUTH

TAG_PATH = os.path.join(os.path.dirname(__file__), "confirmatory_tag.json")
TGT_SEED_BASE = 900_000               # target seed stream (matches the unseen base_seed family)


def load_tag(path: str = TAG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def frozen_cfg() -> ProtocolConfig:
    """The FROZEN method config (manifest da2c0f4309...). Must match the tag's expected hash."""
    cfg = ProtocolConfig(n_boot=40, n_dir_boot=120, target_n_boot=120, tau_n_pseudotargets=240)
    cfg.validate()
    return cfg


def power_min_fired(n_valid: int, bar: float) -> int:
    """Smallest fired count k such that the one-sided 95% CP LOWER bound on power >= bar, computed
    from the REALIZED n_valid. Returns n_valid+1 (unattainable) if even k=n_valid falls short."""
    for k in range(n_valid + 1):
        if _cp_bound(k, n_valid, side="lower") >= bar:
            return k
    return n_valid + 1


def _point_from(entry: dict) -> EnvelopePoint:
    return EnvelopePoint(**entry.get("envelope_overrides", {}))


def run_point(point: EnvelopePoint, cfg: ProtocolConfig, G: int, base_seed: int,
              tgt_seed_base: int = TGT_SEED_BASE, n_jobs: int = 1) -> list:
    """Generate G independent clusters at the fixed point (deterministic per seed; identical serial
    or parallel). Returns the per-cluster records (states + source props)."""
    src_seeds = [base_seed + k for k in range(G)]
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        return list(Parallel(n_jobs=n_jobs)(
            delayed(_cluster_record)(point, cfg, s, tgt_seed_base) for s in src_seeds))
    return [_cluster_record(point, cfg, s, tgt_seed_base) for s in src_seeds]


def evaluate_point(recs: list, tag: dict) -> dict:
    """Evaluate one named point's records into the frozen endpoints. INCONCLUSIVE if too few valid
    clusters or the source-invalid cap is exceeded; else PASS iff BOTH endpoints pass (non-vacuity:
    'controls error' counts only when power also passes)."""
    G = len(recs)
    valid = [r for r in recs if r["source_status"] == "VALID"]
    n_valid = len(valid)
    src_invalid = G - n_valid
    src_invalid_frac = (src_invalid / G) if G else 1.0

    # forbidden endpoint (over valid clusters); source-invalid clusters abstain -> cannot be forbidden
    n_forbidden = sum(any(r["states"][kd] in FORBIDDEN[_TRUTH[kd]] for kd in KINDS) for r in valid)
    forbidden_cp_ub = _cp_bound(n_forbidden, n_valid, side="upper") if n_valid else 1.0

    # power endpoint (realized N_valid)
    n_fired = sum(r["states"]["boundary_coupled"] == CONCEPT_SUSPECT for r in valid)
    # unconditional numerator = valid fires ONLY (source-invalid counted as non-fires BY CONSTRUCTION,
    # enforcing the contract regardless of the certifier's state on an invalid source) -> guarantees
    # power_unconditional = n_fired/G <= power_conditional, so the exclusion can never lift the headline.
    n_fired_all = n_fired
    min_fired = power_min_fired(n_valid, tag["power_bar"]) if n_valid else None
    power_cond = (n_fired / n_valid) if n_valid else None
    power_uncond = (n_fired_all / G) if G else None
    power_cp_lb = _cp_bound(n_fired, n_valid, side="lower") if n_valid else None

    # gating
    inconclusive = (src_invalid_frac > tag["source_invalid_cap"]) or (n_valid < tag["N_valid_min"])
    forbidden_pass = (not inconclusive) and (n_forbidden <= tag["max_forbidden_failures"])
    power_pass = (not inconclusive) and (min_fired is not None) and (n_fired >= min_fired)
    if inconclusive:
        verdict = "INCONCLUSIVE"
    elif forbidden_pass and power_pass:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    # gate-failure decomposition over ALL clusters' boundary_coupled reason (already computed by
    # _cluster_record via the audited _concept_failure_reason), plus source-invalid reasons.
    decomp = {}
    for r in recs:
        decomp[r["vis_fail_reason"]] = decomp.get(r["vis_fail_reason"], 0) + 1
    src_reasons = {}
    for r in recs:
        if r["source_status"] != "VALID":
            src_reasons[r["source_status"]] = src_reasons.get(r["source_status"], 0) + 1

    return dict(
        verdict=verdict, G=G, n_valid=n_valid, source_invalid=src_invalid,
        source_invalid_frac=round(src_invalid_frac, 4),
        forbidden=n_forbidden, forbidden_cp_upper=round(forbidden_cp_ub, 4), forbidden_pass=forbidden_pass,
        fired=n_fired, min_fired_for_pass=min_fired, power_conditional=power_cond,
        power_unconditional=power_uncond, power_cp_lower=power_cp_lb, power_pass=power_pass,
        gate_failure_decomposition=decomp, source_invalid_reasons=src_reasons,
    )


def _validate_method(tag: dict, cfg: ProtocolConfig):
    h = cfg.hash()
    if h != tag["expected_manifest_hash"]:
        raise SystemExit(f"FROZEN-MANIFEST MISMATCH: running {h[:12]} != tag {tag['expected_manifest_hash'][:12]}")


def _describe(tag: dict, cfg: ProtocolConfig):
    print("=== CSC confirmatory — FROZEN PLAN (dry run; nothing executed) ===")
    print(f"status     : {tag['status']}")
    print(f"K={tag['K']} claim={tag['claim_type']} base_seed={tag['base_seed']} G={tag['G']} "
          f"N_valid_min={tag['N_valid_min']} cap={tag['source_invalid_cap']} "
          f"max_forbidden={tag['max_forbidden_failures']} power_bar={tag['power_bar']}")
    print(f"manifest   : running {cfg.hash()[:12]} vs tag {tag['expected_manifest_hash'][:12]} "
          f"({'MATCH' if cfg.hash() == tag['expected_manifest_hash'] else 'MISMATCH'})")
    print(f"core (headline): {[p['name'] for p in tag['core_points']]}")
    print(f"secondary      : {[p['name'] for p in tag['secondary_descriptive_points']]} (NOT in PASS/FAIL)")
    print("realized-N_valid power thresholds (min fired for CP_lower >= power_bar):")
    for n in (tag["N_valid_min"], 60, tag["G"], 72):
        print(f"   N_valid={n:>3d} -> need >= {power_min_fired(n, tag['power_bar'])} fired")
    print("NOT executed. The unseen-cluster confirmatory run requires --execute AND a separate "
          "authorization; default is this dry run.")


def main():
    ap = argparse.ArgumentParser(description="CSC confirmatory runner (FROZEN; dry-run by default).")
    ap.add_argument("--execute", action="store_true",
                    help="SEPARATELY-AUTHORIZED confirmatory run on the unseen core clusters")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default="csc/results/confirmatory.json")
    ap.add_argument("--tag", type=str, default=TAG_PATH)
    args = ap.parse_args()
    tag = load_tag(args.tag)
    cfg = frozen_cfg()

    if not args.execute:
        _describe(tag, cfg)
        return

    # --- separately-authorized confirmatory run (fail-closed) ---
    import warnings, datetime, socket, sys
    warnings.filterwarnings("ignore")
    _validate_method(tag, cfg)
    if tag["base_seed"] != 900_000:          # the pre-registered UNSEEN value; fail closed otherwise
        raise SystemExit(f"base_seed {tag['base_seed']} is not the pre-registered unseen value 900000")
    points = [(p["name"], _point_from(p)) for p in tag["core_points"]]              # K headline points
    results = {}
    for name, point in points:
        recs = run_point(point, cfg, tag["G"], tag["base_seed"], n_jobs=args.jobs)
        results[name] = evaluate_point(recs, tag)
        print(f"[confirmatory] {name}: {results[name]['verdict']} "
              f"(forbidden {results[name]['forbidden']}/{results[name]['n_valid']} "
              f"CP-UB {results[name]['forbidden_cp_upper']}; fired {results[name]['fired']} "
              f">= {results[name]['min_fired_for_pass']}? power_cond {results[name]['power_conditional']})")
    core_pass = all(results[n]["verdict"] == "PASS" for n, _ in points)             # CONJUNCTION
    payload = dict(
        kind="CSC confirmatory result", tag=tag, manifest_hash=cfg.hash(),
        headline_core_pass=core_pass, claim_type=tag["claim_type"],
        per_point=results, base_seed=tag["base_seed"], n_jobs=args.jobs,
        hostname=os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        time=datetime.datetime.now().isoformat(timespec="seconds"),
        note="CONFIRMATORY result. P_strong (if present) is secondary descriptive only and excluded "
             "from headline_core_pass.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[confirmatory] headline_core_pass={core_pass} -> {args.out}")
    sys.exit(0 if core_pass else 1)


if __name__ == "__main__":
    main()
