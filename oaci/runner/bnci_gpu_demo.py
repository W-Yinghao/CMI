"""V100 two-order BNCI GPU smoke CLI (B2b-ii).

    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 ... python -m oaci.runner.bnci_gpu_demo \
        --manifest oaci/protocol/smoke_v1.yaml --datalake-root "$OACI_DATALAKE_ROOT" \
        --artifact-root "$OACI_ARTIFACT_ROOT" --repo-root "$OACI_REPO_ROOT" --model-seed 0

stdout is a single canonical-JSON report; MOABB / MNE / training logs go to stderr. The report carries
the explicit notice that any method-to-method numerical difference is NOT efficacy evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import sys

from ..artifacts.canonical_json import canonical_json_bytes
from .bnci_gpu_compare import comparison_all_equal
from .bnci_gpu_report import runtime_evidence_report
from .bnci_gpu_smoke import run_bnci_gpu_smoke

_NOTICE = ("synthetic-budget GPU execution validation only; method-to-method numerical differences are "
           "NOT efficacy evidence")


def _method_block(m):
    return {"method": m.method_name, "active": bool(m.active), "inactive_reason": m.inactive_reason,
            "selected_checkpoint": m.selection.model_hash, "selected_risk": float(m.selection.R_src),
            "selected_epoch": int(m.selection.selected_epoch),
            "selection_leakage_ucl": (None if m.selection_leakage is None else float(m.selection_leakage["bootstrap_ucl"])),
            "audit_leakage_ucl": (None if m.audit_leakage is None else float(m.audit_leakage["bootstrap_ucl"])),
            "target_pooled_bacc": m.target_metrics.pooled_reference_bacc,
            "target_pooled_nll": m.target_metrics.pooled_nll, "target_pooled_ece": m.target_metrics.pooled_ece}


def build_gpu_report(result) -> dict:
    fr = result.canonical.fold_result
    cmp = result.comparison
    levels = []
    for lvl, lr in fr.level_items:
        levels.append({"level": int(lvl), "run_key_hash": lr.run_key.run_key_hash,
                       "support_hash": lr.support_state.support_hash,
                       "eligibility_counts": lr.support_state.eligibility_counts.tolist(),
                       "p_ref": [float(x) for x in lr.support_state.support_graph.reference_prior.tolist()],
                       "erm_hash": lr.erm_stage.checkpoint.model_hash, "R_ERM_hat": float(lr.erm_stage.R_ERM_hat),
                       "tau": float(lr.erm_stage.tau), "methods": [_method_block(m) for _, m in lr.method_items]})
    return {
        "notice": _NOTICE, "report_hash": result.report_hash, "runtime": runtime_evidence_report(result.runtime),
        "rng_before": result.rng_before_hash, "rng_after_canonical": result.rng_after_canonical_hash,
        "rng_after_reversed": result.rng_after_reversed_hash,
        "rng_unchanged": (result.rng_before_hash == result.rng_after_canonical_hash == result.rng_after_reversed_hash),
        "fold_result_hash": fr.fold_result_hash, "fold_scope_hash": fr.fold_scope.fold_scope_hash,
        "data_evidence_hash": result.canonical.bnci_fold.data_evidence_hash,
        "resolved_preprocess_hash": result.canonical.bnci_fold.resolved_preprocess_hash,
        "canonical_artifact": {"dir": result.canonical.write_result.artifact_dir,
                               "artifact_scientific_hash": result.canonical.write_result.artifact_scientific_hash,
                               "n_indexed_files": result.canonical.write_result.n_indexed_files,
                               "verified_checkpoints": result.canonical.verification.n_verified_checkpoints,
                               "verified_plans": result.canonical.verification.n_verified_plans,
                               "deep_verification_ok": bool(result.canonical.verification.ok)},
        "reversed_artifact": {"dir": result.reversed.write_result.artifact_dir,
                              "artifact_scientific_hash": result.reversed.write_result.artifact_scientific_hash,
                              "deep_verification_ok": bool(result.reversed.verification.ok)},
        "order_comparison": {"fold_result_equal": cmp.fold_result_equal,
                             "artifact_scientific_hash_equal": cmp.artifact_scientific_hash_equal,
                             "checkpoint_hashes_equal": cmp.checkpoint_hashes_equal,
                             "trajectory_hashes_equal": cmp.trajectory_hashes_equal,
                             "selection_hashes_equal": cmp.selection_hashes_equal,
                             "audit_hashes_equal": cmp.audit_hashes_equal,
                             "prediction_hashes_equal": cmp.prediction_hashes_equal,
                             "metrics_hashes_equal": cmp.metrics_hashes_equal,
                             "plan_hashes_equal": cmp.plan_hashes_equal, "all_equal": comparison_all_equal(cmp),
                             "first_mismatch": (None if cmp.first_mismatch is None else cmp.first_mismatch.path)},
        "bn_all_equal_to_erm": all(a.equal_to_erm for a in result.bn_audits),
        "levels": levels}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.runner.bnci_gpu_demo")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--datalake-root", required=True)
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--model-seed", type=int, default=0)
    args = ap.parse_args(argv)
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):           # MNE/MOABB/training chatter -> stderr
            result = run_bnci_gpu_smoke(args.manifest, args.artifact_root, datalake_root=args.datalake_root,
                                        repo_root=args.repo_root, model_seed=args.model_seed)
            report = build_gpu_report(result)
        real_stdout.buffer.write(canonical_json_bytes(report))
        return 0 if report["order_comparison"]["all_equal"] and report["bn_all_equal_to_erm"] and report["rng_unchanged"] else 1
    except Exception as e:  # noqa: BLE001
        print(f"bnci gpu smoke failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
