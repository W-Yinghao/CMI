"""
csc.run_envelope_p15 — CSC-P1.5 DEVELOPMENT difficulty-envelope DRIVER (preflight -> canary -> full).

Runs the reviewer-approved sequence as ONE fail-closed job and writes a single provenance-rich
artifact:

  1. PREFLIGHT   the 3 envelope-harness tests + the dry-run design (nothing scientific produced).
  2. CANARY      the FULL star grid at clusters=2 (default base_seed 500000). Purpose: confirm every
                 cell COMPLETES with the full metric block, no NaN/missing/unknown-decomp-key, and a
                 cluster-denominated count. If the canary validator finds ANY problem -> ABORT, do
                 NOT run the full sweep, exit nonzero.
  3. FULL        the FULL star grid at clusters=12 (default base_seed 600000), only if canary passed.

DEVELOPMENT only. The resulting operating-region map MAY NOT select thresholds, define the
operating region, or seed a confirmatory claim (that needs a separate UNSEEN cluster set). NO
FREEZE / NO CONFIRMATORY / NO P2 is gated on this run.

Run on SLURM CPU (csc/run_envelope_p15.sbatch), NOT the login node.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import warnings

from csc.protocol import ProtocolConfig
from csc.run_envelope import default_grid, run_envelope, validate_cells, KINDS


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


class _GateError(RuntimeError):
    pass


def _require(cond, msg):
    if not cond:
        raise _GateError(msg)


def _preflight():
    """Run the 3 harness tests in-process + record the dry-run design. Returns (ok, detail)."""
    from csc.tests import test_envelope as T
    results, ok = {}, True
    for name in ("test_axis_to_knob_mapping", "test_default_grid_is_star",
                 "test_run_cell_cluster_denominated", "test_full_gate_fails_closed"):
        try:
            getattr(T, name)()
            results[name] = "pass"
        except Exception as e:                       # noqa: BLE001 - record any failure verbatim
            results[name] = f"FAIL: {type(e).__name__}: {e}"
            ok = False
    cells = default_grid()
    design = dict(n_cells=len(cells), kinds_per_cluster=len(KINDS),
                  cell_labels=[c for c, _ in cells])
    return ok, dict(harness_tests=results, design=design)


def verify_canary_ref(canary_ref, head, manifest_hash, canary_clusters, canary_base_seed):
    """Gate for --phase full (CSC-P1.5-driver hotfix): the referenced canary artifact must be a
    PASSED canary-only run on the SAME code commit + protocol manifest + canary clusters/seed.
    Raises _GateError (or OSError/ValueError on a bad file) -- the caller fails closed."""
    _require(canary_ref is not None, "--phase full requires --canary_ref")
    with open(canary_ref) as f:
        ca = json.load(f)
    _require(ca.get("phase") == "CANARY_ONLY_PASSED",
             f"canary_ref phase={ca.get('phase')} != CANARY_ONLY_PASSED")
    _require(ca.get("canary", {}).get("validator_ok") is True, "canary_ref validator not ok")
    _require(ca.get("code_commit") == head,
             f"canary_ref code_commit {ca.get('code_commit')} != HEAD {head}")
    _require(ca.get("protocol_manifest_hash") == manifest_hash,
             "canary_ref protocol manifest hash mismatch")
    _require(ca.get("canary", {}).get("clusters_per_cell") == canary_clusters,
             "canary_ref clusters_per_cell mismatch")
    _require(ca.get("canary", {}).get("base_seed") == canary_base_seed,
             "canary_ref base_seed mismatch")
    return ca


def _phase(cells, cfg, clusters, base_seed, label):
    grid = run_envelope(cells, cfg, clusters, out=None, base_seed=base_seed)["grid"]
    ok, problems = validate_cells(grid, clusters)
    summary = dict(label=label, clusters_per_cell=clusters, base_seed=base_seed,
                   n_cells=len(grid), protocol_calls=len(grid) * clusters * len(KINDS),
                   validator_ok=ok, validator_problems=problems, grid=grid)
    return ok, summary


def main():
    ap = argparse.ArgumentParser(description="CSC-P1.5 DEVELOPMENT envelope driver (preflight->canary->full).")
    ap.add_argument("--canary_clusters", type=int, default=2)
    ap.add_argument("--canary_base_seed", type=int, default=500_000)
    ap.add_argument("--full_clusters", type=int, default=12)
    ap.add_argument("--full_base_seed", type=int, default=600_000)
    ap.add_argument("--audit_baseline", type=str, default="4ea423d",
                    help="the P1.4.5a audit commit this run builds on (recorded for lineage)")
    ap.add_argument("--n_boot", type=int, default=40)
    ap.add_argument("--n_dir_boot", type=int, default=120)
    ap.add_argument("--target_n_boot", type=int, default=120)
    ap.add_argument("--tau_n_pseudotargets", type=int, default=240)
    ap.add_argument("--phase", choices=("canary", "full", "both"), default="both",
                    help="canary: preflight+canary; full: preflight+full (assumes a prior canary "
                         "passed, ref via --canary_ref); both: preflight+canary+GATED full (one job)")
    ap.add_argument("--canary_ref", type=str, default=None,
                    help="for --phase full: the canary artifact this full sweep was gated by")
    ap.add_argument("--out", type=str, default="csc/results/envelope_p15.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = ProtocolConfig(n_boot=args.n_boot, n_dir_boot=args.n_dir_boot,
                         target_n_boot=args.target_n_boot, tau_n_pseudotargets=args.tau_n_pseudotargets)
    cfg.validate()
    cells = default_grid()
    start = datetime.datetime.now().isoformat(timespec="seconds")
    head = _git("rev-parse", "HEAD")

    art = dict(
        kind="CSC-P1.5 DEVELOPMENT difficulty-envelope (operating-region map)",
        status="DEVELOPMENT — MAY NOT select thresholds / define the operating region / seed a "
               "confirmatory claim. Denominator = independent source-target clusters.",
        code_commit=head, code_commit_short=(head[:7] if head else None),
        audit_baseline_commit=args.audit_baseline,
        git_status_clean_csc=(_git("status", "--porcelain", "csc") == ""),
        protocol_manifest=cfg.manifest(), protocol_manifest_hash=cfg.hash(),
        exact_command=" ".join(sys.argv),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        start_time=start,
        difficulty_axes_proxy_note="axis 'concept_eigengap_sep' is a PROXY (concept_domains); single "
                                   "w_concept geom today -> not a true multi-axis eigengap stress test",
        eigengap_axis_is_proxy=True,
        seed_provenance=(f"DEVELOPMENT seeds (canary base {args.canary_base_seed}, full base "
                         f"{args.full_base_seed}; NOT the 0-9 audit smoke set). A confirmatory run "
                         f"needs a separate, previously UNSEEN cluster set."),
    )

    # 1. PREFLIGHT ----------------------------------------------------------------------------------
    print("[p15] PREFLIGHT: harness tests + dry-run design ...")
    pf_ok, pf = _preflight()
    art["preflight"] = dict(ok=pf_ok, **pf)
    if not pf_ok:
        art["phase"] = "ABORTED_AT_PREFLIGHT"
        art["end_time"] = datetime.datetime.now().isoformat(timespec="seconds")
        _write(args.out, art); print("[p15] PREFLIGHT FAILED -> aborted, exit 1"); sys.exit(1)

    art["run_phase"] = args.phase

    # 2. CANARY -------------------------------------------------------------------------------------
    if args.phase in ("canary", "both"):
        print(f"[p15] CANARY: full grid x {args.canary_clusters} clusters (base {args.canary_base_seed}) ...")
        can_ok, canary = _phase(cells, cfg, args.canary_clusters, args.canary_base_seed, "canary")
        art["canary"] = canary
        if not can_ok:
            art["phase"] = "ABORTED_AT_CANARY"
            art["end_time"] = datetime.datetime.now().isoformat(timespec="seconds")
            _write(args.out, art)
            print(f"[p15] CANARY validator FAILED ({len(canary['validator_problems'])} problems) -> "
                  f"NOT running full sweep, exit 1")
            for p in canary["validator_problems"][:20]:
                print(f"        - {p}")
            sys.exit(1)
        print("[p15] CANARY passed validator.")
        if args.phase == "canary":
            art["phase"] = "CANARY_ONLY_PASSED"
            art["end_time"] = datetime.datetime.now().isoformat(timespec="seconds")
            _write(args.out, art); print(f"[p15] DONE phase={art['phase']} -> {args.out}"); return

    # 3. FULL -- only reachable when GATED by a passed canary (CSC-P1.5-driver hotfix) -------------
    #   phase=both : gated by the canary just run above (can_ok is True here, else we exited).
    #   phase=full : gated by a SEPARATE canary artifact (--canary_ref) that must be PASSED, on the
    #                SAME code commit + protocol manifest + canary clusters/seed (else fail closed).
    if args.phase == "full":
        try:
            verify_canary_ref(args.canary_ref, head, cfg.hash(),
                              args.canary_clusters, args.canary_base_seed)
        except (_GateError, OSError, ValueError) as e:
            art["phase"] = "ABORTED_AT_FULL_GATE"
            art["full_gate_error"] = str(e)
            art["end_time"] = datetime.datetime.now().isoformat(timespec="seconds")
            _write(args.out, art)
            print(f"[p15] FULL GATE FAILED: {e} -> NOT running full sweep, exit 1")
            sys.exit(1)
        art["canary_ref"] = args.canary_ref
        art["canary_ref_verified"] = True

    print(f"[p15] FULL: grid x {args.full_clusters} clusters (base {args.full_base_seed}) ...")
    full_ok, full = _phase(cells, cfg, args.full_clusters, args.full_base_seed, "full")
    art["full"] = full
    if not full_ok:                                   # FULL fails closed like CANARY (P1.5 hotfix)
        art["phase"] = "ABORTED_AT_FULL_VALIDATOR"
        art["end_time"] = datetime.datetime.now().isoformat(timespec="seconds")
        _write(args.out, art)
        print(f"[p15] FULL validator FAILED ({len(full['validator_problems'])} problems) -> exit 1")
        for p in full["validator_problems"][:20]:
            print(f"        - {p}")
        sys.exit(1)
    art["phase"] = "FULL_COMPLETE"
    art["end_time"] = datetime.datetime.now().isoformat(timespec="seconds")
    _write(args.out, art)
    print(f"[p15] DONE phase={art['phase']} -> {args.out}")


def _write(path, art):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(art, f, indent=2)


if __name__ == "__main__":
    main()
