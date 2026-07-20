"""CPU demo CLI for the fake two-level closed loop.

    python -m oaci.runner.demo --manifest oaci/protocol/fake_runner_v1.yaml \
        --output-root "$SLURM_TMPDIR/oaci-fake" --model-seed 0 \
        --method-order ERM,OACI,global_lpc,uniform --repo-root "$SLURM_SUBMIT_DIR"

stdout is a single canonical-JSON summary (no efficacy claims); errors go to stderr (exit 1).
"""
from __future__ import annotations

import argparse
import sys

from ..artifacts.canonical_json import canonical_json_bytes
from .fake_artifact import run_fake_two_level

_NOTICE = "synthetic execution validation only; method differences are not efficacy evidence"


def _sel_lambda(m):
    for c in m.train_result.trajectory:
        if c.model_hash == m.selection.model_hash:
            return float(c.lam)
    return None


def _ucl(leak):
    return None if leak is None else float(leak["bootstrap_ucl"])


def _dedup_count(lr):
    refs = 1
    uniq = {lr.erm_stage.checkpoint.model_hash}
    for _, m in lr.method_items:
        refs += 1 + len(m.train_result.trajectory)
        uniq.add(m.selection.model_hash)
        uniq.update(c.model_hash for c in m.train_result.trajectory)
    return refs - len(uniq)


def build_demo_summary(result) -> dict:
    fr, rep, summ, wr = result.fold_result, result.verification, result.loaded_summary, result.write_result
    git = result.context.git
    out = {"notice": _NOTICE, "schema_version": summ.schema_version, "git_commit": git.commit,
           "git_tree_hash": git.tree_hash, "manifest_hash": summ.manifest_hash,
           "fake_data_hash": fr.fold_scope.data_contract_hash, "fold_scope_hash": summ.fold_scope_hash,
           "fold_result_hash": summ.fold_result_hash, "context_hash": summ.context_hash,
           "pure_context_hash": summ.pure_context_hash, "provenance_hash": summ.provenance_hash,
           "artifact_scientific_hash": summ.artifact_scientific_hash,
           "artifact_pure_science_hash": summ.artifact_pure_science_hash,
           "artifact_index_sha256": summ.artifact_index_sha256, "artifact_dir": wr.artifact_dir,
           "deep_verification_ok": bool(rep.ok), "comparison_hash": result.comparison_hash,
           "n_indexed_files": summ.n_indexed_files, "n_total_files": summ.n_total_files,
           "n_verified_checkpoints": summ.n_verified_checkpoints, "n_verified_plans": summ.n_verified_plans,
           "levels": []}
    fd = result.fake_fold.fold_data
    for lvl, lr in fr.level_items:
        s = summ.levels[int(lvl)]
        del_rows = sum(1 for i in lr.support_state.source_train_idx.tolist()
                       if fd.domain_id[i] == "S0" and int(fd.y[i]) == 1)   # rows in the (S0, c1) cell
        lvl_out = {"level": int(lvl), "run_key_hash": s.run_key_hash, "support_hash": s.support_hash,
                   "eligibility_counts": [list(r) for r in s.eligibility_counts],
                   "cell_mass": [list(r) for r in s.cell_mass], "p_ref": list(s.reference_prior),
                   "deleted_cell": {"count": float(s.eligibility_counts[0][1]), "mass": float(s.cell_mass[0][1]),
                                    "rows": del_rows},
                   "stage1_invocation_count": 1, "erm_hash": s.erm_checkpoint,
                   "R_ERM_hat": s.R_ERM_hat, "tau": s.tau, "checkpoint_dedup_count": _dedup_count(lr),
                   "plan_hashes": {n: h for n, h in s.plan_hashes}, "methods": []}
        for name, m in lr.method_items:
            tm = m.target_metrics
            lvl_out["methods"].append({
                "method": name, "active": bool(m.active), "inactive_reason": m.inactive_reason,
                "selected_checkpoint": m.selection.model_hash, "selected_risk": float(m.selection.R_src),
                "selected_epoch": int(m.selection.selected_epoch), "selected_lambda": _sel_lambda(m),
                "selection_ucl": _ucl(m.selection_leakage), "audit_ucl": _ucl(m.audit_leakage),
                "target_pooled_bacc": tm.pooled_reference_bacc, "target_mean_bacc": tm.mean_domain_reference_bacc,
                "target_worst_bacc": tm.worst_domain_reference_bacc, "target_pooled_nll": tm.pooled_nll,
                "target_mean_nll": tm.mean_domain_nll, "target_worst_nll": tm.worst_domain_nll,
                "target_pooled_ece": tm.pooled_ece, "target_mean_ece": tm.mean_domain_ece,
                "target_worst_ece": tm.worst_domain_ece})
        out["levels"].append(lvl_out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.runner.demo")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--method-order", default="ERM,OACI,global_lpc,uniform")
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args(argv)
    try:
        order = tuple(args.method_order.split(","))
        result = run_fake_two_level(args.manifest, args.output_root, model_seed=args.model_seed,
                                    method_order=order, repo_root=args.repo_root)
        sys.stdout.buffer.write(canonical_json_bytes(build_demo_summary(result)))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"demo failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
