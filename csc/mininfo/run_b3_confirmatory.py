"""
CSC Route B3 confirmatory runner -- DRY-RUN ONLY (pre-registration hardening). Loads the frozen manifest,
verifies provenance (certifier code hashes match the manifest), verifies the simulator SCENARIOS match the
manifest's exact scenario_configs, verifies the confirmatory seed block is disjoint from every A/B/dev/test
seed, and prints the pre-registered grid + CONJUNCTION pass criteria. It does NOT run the confirmatory.

`--execute` is intentionally BLOCKED: running the unseen-cluster confirmatory (and creating the tag
`csc-b3-confirmatory-v1` / manifest hash-lock) requires explicit reviewer authorization. This file is the
dry-run + fail-closed validator only.

  python -m csc.mininfo.run_b3_confirmatory                 # dry-run validation + plan
  python -m csc.mininfo.run_b3_confirmatory --execute       # refused: NOT AUTHORIZED
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(HERE, "b3_confirmatory_manifest.json")


def load_manifest(path=MANIFEST):
    with open(path) as f:
        return json.load(f)


def verify_code_hashes(mani):
    """Fail-closed provenance: current certifier source must match the manifest's recorded sha256."""
    bad = []
    for rel, want in mani["code_hashes_sha256"].items():
        p = os.path.join(ROOT, rel)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "MISSING"
        if got != want:
            bad.append((rel, want, got))
    return bad


def verify_scenarios(mani):
    """The manifest's exact scenario_configs must match csc.mininfo.run_b3_p23.SCENARIOS byte-for-byte."""
    from .run_b3_p23 import SCENARIOS
    man_cfg = mani["scenario_configs"]
    mismatch = []
    keys = set(man_cfg) | set(SCENARIOS)
    for k in keys:
        if man_cfg.get(k) != SCENARIOS.get(k):
            mismatch.append((k, man_cfg.get(k), SCENARIOS.get(k)))
    # every primary/control scenario named in the grid must exist in the configs
    for k in mani["grid"]["control_scenarios"]:
        if k not in man_cfg:
            mismatch.append((k, "IN GRID", "MISSING FROM scenario_configs"))
    return mismatch


def verify_seed_disjoint(mani):
    """The confirmatory seed block must not overlap A's block, any B dev block, or test/smoke seeds."""
    base = mani["seed_spec"]["confirmatory_base_seed"]
    clusters = mani["method_lock"]["clusters_per_cell"]
    conf = set(range(base, base + clusters))
    dj = mani["seed_spec"]["disjoint_from"]
    used = set()
    used.add(dj["A_confirmatory"])
    used |= set(range(dj["A_confirmatory"], dj["A_confirmatory"] + 200))     # A block padding
    for b in dj["B_development"]:
        used |= set(range(b, b + max(clusters, 96)))                          # each dev block + padding
    used |= set(range(0, 100000))                                            # smoke/test range
    overlap = conf & used
    return sorted(overlap)


def print_plan(mani):
    g = mani["grid"]; ml = mani["method_lock"]
    n_ctrl = len(g["control_kinds"]) * len(g["control_scenarios"]) * len(ml["positive_decision_budgets"]) \
        * ml["clusters_per_cell"]
    n_prim = len(g["primary_positive_kinds"]) * len(g["primary_scenarios"]) * len(g["primary_budgets"]) \
        * ml["clusters_per_cell"]
    print(f"  method                : {ml['method']} (calib {mani['calibration_version']})")
    print(f"  alpha_family/budget   : {ml['alpha_family']} / {ml['alpha_budget']}  LCB {ml['lcb_level']}")
    print(f"  confirmatory seed base: {mani['seed_spec']['confirmatory_base_seed']} (disjoint from A 900000 + B dev)")
    print(f"  controls              : {len(g['control_kinds'])} kinds x {len(g['control_scenarios'])} scenarios "
          f"x {ml['positive_decision_budgets']} x {ml['clusters_per_cell']} = {n_ctrl} decision runs")
    print(f"  primary positives     : {g['primary_positive_kinds']} x {g['primary_scenarios']} "
          f"x {g['primary_budgets']} x {ml['clusters_per_cell']} = {n_prim} runs")
    print(f"  secondary (reported)  : {g['secondary_kinds']}")
    print(f"  excluded from primary : {g['excluded_from_primary_positive_power']} (controls STILL evaluated there)")
    print("  PASS = CONJUNCTION:")
    for k in ("C1_guards", "C2_control_typeI", "C3_no_hot_cell", "C4_primary_power",
              "C5_no_silent_failure", "C6_independent_verification"):
        print(f"    {k}: {mani['pass_criteria'][k]}")


def dry_run(path=MANIFEST):
    mani = load_manifest(path)
    print(f"[b3-confirmatory DRY-RUN] manifest {os.path.relpath(path, ROOT)} ({mani['status']})")
    bad_h = verify_code_hashes(mani)
    bad_s = verify_scenarios(mani)
    overlap = verify_seed_disjoint(mani)
    print(f"  code-hash provenance  : {'OK' if not bad_h else 'FAIL ' + str(bad_h)}")
    print(f"  scenario-config match : {'OK' if not bad_s else 'FAIL ' + str(bad_s)}")
    print(f"  seed disjointness     : {'OK' if not overlap else 'FAIL overlap ' + str(overlap)}")
    print_plan(mani)
    ok = not (bad_h or bad_s or overlap)
    print(f"[b3-confirmatory DRY-RUN] provenance {'CLEAN -- ready for reviewer freeze authorization' if ok else 'NOT CLEAN -- fail closed'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="CSC B3 confirmatory DRY-RUN validator (no execution).")
    ap.add_argument("--execute", action="store_true", help="(blocked) run the confirmatory")
    ap.add_argument("--manifest", type=str, default=MANIFEST)
    a = ap.parse_args()
    if a.execute:
        print("REFUSED: running the B3 confirmatory (and creating tag csc-b3-confirmatory-v1 / locking the "
              "manifest hash) is NOT AUTHORIZED. Reviewer approval required. This runner is dry-run only.")
        sys.exit(2)
    ok = dry_run(a.manifest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
