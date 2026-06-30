"""Two-job staged confirmatory one-fold CLIs (C4b-3).

Phase A (V100): configure determinism, materialize the pilot manifest, build the real fold, train + GPU-
prefetch every feasible candidate's features/logits into a per-level store, persist to the staging dir,
release the GPU. Phase B (CPU): rebuild the fold + context, resume select/audit/finalize from the store
(no GPU), write -> deep-verify -> read -> compare the artifact. Both phases materialize the SAME manifest
(deterministic), so Phase B's rebuilt context matches Phase A's.

    python -m oaci.confirmatory.staged_demo phase-a --protocol ... --datalake-root ... --staging-dir ...
    python -m oaci.confirmatory.staged_demo phase-b --protocol ... --staging-dir ... --output-root ... --repo-root ...

This is execution-only; the staged artifact is bit-identical to the monolithic (proven by the CPU tests).
NOT confirmatory efficacy evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

from ..artifacts.summary import compare_artifact_summary_to_memory, read_completed_artifact
from ..artifacts.canonical_json import canonical_json_bytes
from ..artifacts.verify import verify_artifact_tree
from ..artifacts.writer import write_artifact_tree_atomic
from ..leakage.parallel import leakage_parallel_report, set_leakage_parallel
from ..runner.bnci_data import build_bnci_real_fold
from ..runner.fake_artifact import build_fold_artifact_context
from ..runner.staged_fold import staged_phase_a, staged_phase_b
from ..runtime.cuda import configure_cuda_determinism
from .materialize import VALIDATION_BOOTSTRAP, materialize_pilot_manifest
from .onefold import DATASET
from .schema import load_confirmatory


def _materialize(args):
    """Materialize the confirmatory one-fold manifest using the BNCI2014_001 LOSO CYCLIC split (C6): the
    SAME deterministic split (loso_fold_spec) the submitter and the dry-run use, derived from the target so
    that loso_plan stays the single source of truth. Without this the executor fell back to the default
    SORTED split (materialize.split_subjects), which agrees with the cyclic split ONLY for target-001 -- so
    the GPU sweep would have silently run the wrong audit/train roles and wrong deleted cell for targets
    2..9. For target-001 the cyclic and the default split coincide, so the manifest is byte-identical and
    the earlier target-001 staged runs stay valid."""
    from .loso_plan import explicit_split as _loso_split, loso_fold_spec
    if str(args.dataset) != "BNCI2014_001":                            # the LOSO cyclic plan is the 9-subject BNCI
        raise ValueError(f"staged confirmatory LOSO is defined for BNCI2014_001, not {args.dataset!r}")
    proto = load_confirmatory(args.protocol)
    override = VALIDATION_BOOTSTRAP if args.bootstrap_mode == "validation" else None
    spec = loso_fold_spec(int(args.target_subject), dataset_id=str(args.dataset))
    path, manifest = materialize_pilot_manifest(
        proto, args.dataset, target_subject=int(args.target_subject), out_path=args.manifest_out,
        model_seeds=[int(args.model_seed)], bootstrap_override=override,
        explicit_split=_loso_split(spec), deleted_cell=dict(spec["deleted_cell"]))
    return path, manifest


def main_phase_a(args) -> int:
    real = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            device, runtime = configure_cuda_determinism()
            mpath, _m = _materialize(args)
            fold = build_bnci_real_fold(mpath, args.datalake_root)
            fold.fold_data.assert_integrity()
            staged_phase_a(fold, dataset_id=args.dataset, model_seed=int(args.model_seed),
                           gpu_device=device, out_dir=args.staging_dir)
            import json
            meta = json.load(open(os.path.join(args.staging_dir, "phase_a.json")))
            store_bytes = sum(os.path.getsize(os.path.join(args.staging_dir, f))
                              for f in os.listdir(args.staging_dir) if f.endswith(".pkl"))
        real.buffer.write(canonical_json_bytes({
            "phase": "A", "notice": "staged GPU record stage; not confirmatory efficacy evidence",
            "manifest_hash": fold.manifest_hash, "fold_scope_hash": fold.fold_scope.fold_scope_hash,
            "target_subject": int(args.target_subject), "model_seed": int(args.model_seed),
            "bootstrap_mode": args.bootstrap_mode, "staging_dir": args.staging_dir,
            "staging_bytes": int(store_bytes), "levels": meta["levels"],
            "runtime_driver": runtime.driver_version, "data_evidence_hash": fold.data_evidence_hash}))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"staged phase A failed: {e}", file=sys.stderr)
        return 1


def main_phase_b(args) -> int:
    if int(args.leakage_jobs) > 1:
        set_leakage_parallel(int(args.leakage_jobs), "process")
    real = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from ..runner.staged_fold import load_phase_a_fold
            fold = load_phase_a_fold(args.staging_dir)        # the EXACT Phase-A fold; NEVER re-load the data
            fold.fold_data.assert_integrity()
            fr = staged_phase_b(args.staging_dir, fold=fold)
            ctx = build_fold_artifact_context(fold, fr, repo_root=args.repo_root)
            write = write_artifact_tree_atomic(fr, ctx, args.output_root)
            rep = verify_artifact_tree(write.artifact_dir, deep=True)
            if not rep.ok:
                raise RuntimeError(f"deep verification failed: {rep.errors[:3]}")
            summ = read_completed_artifact(write.artifact_dir, deep_verify=True)
            cmp = compare_artifact_summary_to_memory(summ, fr, ctx,
                                                     artifact_scientific_hash=write.artifact_scientific_hash,
                                                     artifact_pure_science_hash=write.artifact_pure_science_hash)
            target_fit_empty = all(not lr.provenance.target_fit_ids for _, lr in fr.level_items)
        ok = rep.ok and cmp.ok and target_fit_empty
        real.buffer.write(canonical_json_bytes({
            "phase": "B", "notice": "staged CPU replay stage; not confirmatory efficacy evidence",
            "manifest_hash": fold.manifest_hash, "fold_result_hash": fr.fold_result_hash,
            "artifact_scientific_hash": write.artifact_scientific_hash,
            "artifact_pure_science_hash": write.artifact_pure_science_hash,
            "artifact_dir": write.artifact_dir, "deep_verification_ok": bool(rep.ok),
            "summary_matches_memory": bool(cmp.ok), "target_fit_ids_empty": bool(target_fit_empty),
            "n_indexed_files": write.n_indexed_files, "verified_checkpoints": rep.n_verified_checkpoints,
            "verified_plans": rep.n_verified_plans, "leakage_parallel": leakage_parallel_report()}))
        return 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        print(f"staged phase B failed: {e}", file=sys.stderr)
        return 1
    finally:
        set_leakage_parallel(1, "sequential")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.staged_demo")
    sub = ap.add_subparsers(dest="phase", required=True)
    for name in ("phase-a", "phase-b"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", required=True)
        p.add_argument("--datalake-root", required=True)
        p.add_argument("--staging-dir", required=True)
        p.add_argument("--manifest-out", required=True)
        p.add_argument("--dataset", default=DATASET)
        p.add_argument("--target-subject", type=int, default=1)
        p.add_argument("--model-seed", type=int, default=0)
        p.add_argument("--bootstrap-mode", choices=("full", "validation"), default="full")
        if name == "phase-b":
            p.add_argument("--output-root", required=True)
            p.add_argument("--repo-root", required=True)
            p.add_argument("--leakage-jobs", type=int, default=1)
    args = ap.parse_args(argv)
    return main_phase_a(args) if args.phase == "phase-a" else main_phase_b(args)


if __name__ == "__main__":
    raise SystemExit(main())
