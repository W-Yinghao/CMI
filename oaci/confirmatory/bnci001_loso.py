"""BNCI2014_001 LOSO seed-0 staged full-bootstrap driver (C6).

Materializes the nine LOSO fold manifests (one per held-out target) from the confirmatory protocol using
the deterministic cyclic split, and validates them WITHOUT training. The staged execution (Phase A GPU /
Phase B CPU) is orchestrated by submit.py; aggregation by aggregate.py.

    BNCI2014-001 LOSO seed-0 full-bootstrap staged run.
    This is not the final multi-seed, multi-dataset confirmatory efficacy result.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

from .loso_plan import SUBJECTS, explicit_split, loso_plan
from .materialize import VALIDATION_BOOTSTRAP, materialize_pilot_manifest
from .schema import load_confirmatory

DATASET = "BNCI2014_001"
_N_TRIALS_PER_CELL = 144                 # BNCI2014_001: 144 trials per (subject, class)
_METHODS = ("ERM", "OACI", "global_lpc", "uniform")


def materialize_loso_manifest(protocol, spec, *, out_path, model_seed=0, bootstrap_mode="full"):
    """One LOSO fold's runnable pilot manifest (cyclic split, explicit). Bit-exact to a stand-alone
    materialization for that target."""
    override = VALIDATION_BOOTSTRAP if bootstrap_mode == "validation" else None
    return materialize_pilot_manifest(
        protocol, spec["dataset_id"], target_subject=int(spec["target"]), out_path=out_path,
        model_seeds=[int(model_seed)], bootstrap_override=override, explicit_split=explicit_split(spec),
        deleted_cell=dict(spec["deleted_cell"]))


def materialize_all_loso(protocol_path, out_dir, *, subjects=SUBJECTS, model_seed=0, bootstrap_mode="full"):
    """Materialize all nine fold manifests; returns [(spec, manifest_path, ProtocolManifestV2), ...]."""
    proto = load_confirmatory(protocol_path)
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for spec in loso_plan(subjects=subjects):
        mp = os.path.join(out_dir, f"{spec['fold_id']}.yaml")
        _, m = materialize_loso_manifest(proto, spec, out_path=mp, model_seed=model_seed,
                                         bootstrap_mode=bootstrap_mode)
        out.append((spec, mp, m))
    return out


def expected_level_support(spec):
    """The EXPECTED support tables for a fold (domains = the 6 source-train subjects in plan order × 4
    classes). Level 0: every cell = 144. Level 1: the deleted (deleted_subject, feet) cell = 0, the rest
    144. The actual data is verified by the CPU dry-run; this is the structural expectation the driver and
    the aggregation check against."""
    n = len(spec["source_train_subjects"])
    classes = ("left_hand", "right_hand", "feet", "tongue")
    feet = classes.index("feet")
    del_row = spec["source_train_subjects"].index(spec["deleted_subject"])
    level0 = [[_N_TRIALS_PER_CELL] * len(classes) for _ in range(n)]
    level1 = [row[:] for row in level0]
    level1[del_row][feet] = 0
    return {"domains": list(spec["source_train_subjects"]), "classes": list(classes),
            "level0": level0, "level1": level1, "deleted_row": del_row, "deleted_col": feet,
            "p_ref": [0.25, 0.25, 0.25, 0.25]}


def validate_loso_plan(subjects=SUBJECTS) -> dict:
    """Plan-level validation (no data, no training): nine unique targets, disjoint 1/2/6 roles, every
    subject held out once, deterministic deleted cells, the four methods, and the target-001 fold matching
    the C5 split."""
    plan = loso_plan(subjects=subjects)
    targets = [f["target"] for f in plan]
    errs = []
    if sorted(targets) != sorted(int(s) for s in subjects) or len(set(targets)) != len(subjects):
        errs.append("targets are not the subjects each exactly once")
    held_out = set()
    for f in plan:
        roles = ([f["target"]], f["source_audit_subjects"], f["source_train_subjects"])
        flat = [s for r in roles for s in r]
        if len(set(flat)) != len(flat) or set(flat) != set(int(s) for s in subjects):
            errs.append(f"{f['fold_id']}: roles overlap or do not cover all subjects")
        if len(f["source_audit_subjects"]) != 2 or len(f["source_train_subjects"]) != 6:
            errs.append(f"{f['fold_id']}: role sizes != 1/2/6")
        if f["deleted_subject"] != f["source_train_subjects"][0]:
            errs.append(f"{f['fold_id']}: deleted subject != first source-train (cyclic)")
        held_out.add(f["target"])
    if held_out != set(int(s) for s in subjects):
        errs.append("not every subject held out exactly once")
    # target-001 must reproduce the C5 split
    f1 = next(f for f in plan if f["target"] == 1)
    if f1["source_audit_subjects"] != [2, 3] or f1["source_train_subjects"] != [4, 5, 6, 7, 8, 9] \
            or f1["deleted_subject"] != 4:
        errs.append("target-001 fold does not reproduce the C5 split")
    return {"ok": not errs, "errors": errs, "n_folds": len(plan), "methods": list(_METHODS), "plan": plan}


def _fold_real_check(spec, manifest, fold) -> dict:
    """Order-independent real-data checks for one built fold (NO training): role counts, target-not-fit,
    audit estimable, all four methods, and the per-level support structure (level-0 all 144; level-1 one
    deleted cell at 0)."""
    from ..runner.support import build_level_support, level0_reference_prior
    fd, maps, schedule = fold.fold_data, fold.maps, fold.deletion_schedule
    ref = level0_reference_prior(fd, maps)
    support_m = int(manifest.enabled_datasets()[DATASET].support_m)
    errs = []
    rc = (int(len(fd.source_train_idx)), int(len(fd.source_audit_idx)), int(len(fd.target_audit_idx)))
    if rc != (3456, 1152, 576):
        errs.append(f"role counts {rc} != (3456,1152,576)")
    if fold.fold_scope.source_audit.status != "estimable":
        errs.append(f"audit scope not estimable: {fold.fold_scope.source_audit.status}")
    from ..runner.bnci_data import target_seen_by_fit
    if target_seen_by_fit(fd):
        errs.append("a target id is in the preprocessing fit set")
    level_zeros = {}
    for level in (0, 1):
        ss = build_level_support(fd, maps, level, schedule, ref, support_m=support_m)
        counts = ss.support_graph.counts                       # domains x classes
        import numpy as np
        zeros = int((np.asarray(counts) == 0).sum())
        nonzero_vals = set(int(v) for v in np.asarray(counts).ravel() if v != 0)
        level_zeros[level] = zeros
        if level == 0 and (zeros != 0 or nonzero_vals - {_N_TRIALS_PER_CELL}):
            errs.append(f"level0 support not all {_N_TRIALS_PER_CELL}: zeros={zeros} vals={nonzero_vals}")
        if level == 1 and zeros != 1:
            errs.append(f"level1 deleted cells != 1 (got {zeros})")
        pref = [round(float(x), 6) for x in ss.support_graph.reference_prior.tolist()]
        if pref != [0.25, 0.25, 0.25, 0.25]:
            errs.append(f"level{level} p_ref {pref} != uniform")
    return {"fold_id": spec["fold_id"], "target": spec["target"], "ok": not errs, "errors": errs,
            "role_counts": {"source_train": rc[0], "source_audit": rc[1], "target_audit": rc[2]},
            "level0_zeros": level_zeros.get(0), "level1_zeros": level_zeros.get(1),
            "manifest_hash": fold.manifest_hash}


def loso_dry_run(protocol_path, datalake_root, out_dir, *, subjects=SUBJECTS, model_seed=0,
                 bootstrap_mode="full") -> dict:
    """Build every fold on REAL data (NO training) and check it. Offline-only."""
    from ..runner.bnci_data import build_bnci_real_fold
    plan_ok = validate_loso_plan(subjects=subjects)
    folds = materialize_all_loso(protocol_path, out_dir, subjects=subjects, model_seed=model_seed,
                                 bootstrap_mode=bootstrap_mode)
    results = []
    for spec, mp, m in folds:
        fold = build_bnci_real_fold(mp, datalake_root)         # build only; never trains
        fold.fold_data.assert_integrity()
        results.append(_fold_real_check(spec, m, fold))
    manifests = {r["fold_id"]: r["manifest_hash"] for r in results}
    return {"notice": "BNCI2014-001 LOSO seed-0 dry-run (fold build only; no training; not efficacy)",
            "plan_ok": plan_ok["ok"], "plan_errors": plan_ok["errors"],
            "all_ok": plan_ok["ok"] and all(r["ok"] for r in results) and len(results) == len(subjects),
            "n_folds": len(results), "unique_manifest_hashes": len(set(manifests.values())) == len(manifests),
            "folds": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.bnci001_loso")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--datalake-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--bootstrap-mode", choices=("full", "validation"), default="full")
    args = ap.parse_args(argv)
    real = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):           # MNE/MOABB chatter -> stderr
            rep = loso_dry_run(args.protocol, args.datalake_root, args.out_dir, model_seed=args.model_seed,
                               bootstrap_mode=args.bootstrap_mode)
        from ..artifacts.canonical_json import canonical_json_bytes
        real.buffer.write(canonical_json_bytes(rep))
        return 0 if rep["all_ok"] else 1
    except Exception as e:  # noqa: BLE001
        print(f"LOSO dry-run failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
