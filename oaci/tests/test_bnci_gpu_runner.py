"""GPU-only two-order BNCI smoke (B2b-ii): runs the full smoke ONCE and asserts the closed loop + the
bit-exact canonical-vs-reversed comparison. Run by slurm_gpu_smoke.sh; skips on a CPU node (exit 0).

    OACI_DATALAKE_ROOT=... OACI_ARTIFACT_ROOT=... OACI_REPO_ROOT=... python -m oaci.tests.test_bnci_gpu_runner
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

import oaci.protocol
from oaci.runner.bnci_gpu_compare import comparison_all_equal
from oaci.runner.bnci_gpu_smoke import run_bnci_gpu_smoke

_SMOKE = os.path.join(os.path.dirname(oaci.protocol.__file__), "smoke_v1.yaml")
_C = {}


def _skip():
    if not torch.cuda.is_available():
        print("SKIP  no CUDA device (run on the GPU smoke job)", file=sys.stderr)
        return True
    return False


def _result():
    if "r" not in _C:
        root = os.environ.get("OACI_DATALAKE_ROOT", "/projects/EEG-foundation-model/datalake/raw")
        repo = os.environ.get("OACI_REPO_ROOT", os.getcwd())
        art = os.environ.get("OACI_ARTIFACT_ROOT") or tempfile.mkdtemp(prefix="oaci-gpu-art-")
        _C["r"] = run_bnci_gpu_smoke(_SMOKE, art, datalake_root=root, repo_root=repo, model_seed=0)
    return _C["r"]


def test_both_orders_reach_complete_and_deep_verify():
    if _skip():
        return
    r = _result()
    from oaci.runner.provenance import RunnerPhase
    for run in (r.canonical, r.reversed):
        assert run.verification.ok
        assert all(lr.phase == RunnerPhase.COMPLETE for _, lr in run.fold_result.level_items)


def test_both_artifact_summaries_match_memory():
    if _skip():
        return
    r = _result()
    assert r.canonical.comparison_hash and r.reversed.comparison_hash      # compare-to-memory passed in each loop


def test_rng_is_unchanged_across_both_runs():
    if _skip():
        return
    r = _result()
    assert r.rng_before_hash == r.rng_after_canonical_hash == r.rng_after_reversed_hash


def test_every_stage2_checkpoint_bn_equals_erm():
    if _skip():
        return
    r = _result()
    assert r.bn_audits and all(a.equal_to_erm for a in r.bn_audits)


def test_order_reproduces_every_scientific_hash_group():
    if _skip():
        return
    cmp = _result().comparison
    assert comparison_all_equal(cmp) and cmp.first_mismatch is None
    assert (cmp.fold_result_equal and cmp.artifact_scientific_hash_equal and cmp.checkpoint_hashes_equal
            and cmp.trajectory_hashes_equal and cmp.selection_hashes_equal and cmp.audit_hashes_equal
            and cmp.prediction_hashes_equal and cmp.metrics_hashes_equal and cmp.plan_hashes_equal)


def test_bnci_gpu_exact_data_contract():
    if _skip():
        return
    fd = _result().canonical.bnci_fold.fold_data
    assert tuple(fd.X.shape) == (3456, 22, 385)
    assert (len(fd.source_train_idx), len(fd.source_audit_idx), len(fd.target_audit_idx)) == (1728, 1152, 576)


def test_bnci_gpu_target_fit_ids_empty_and_methods_active():
    if _skip():
        return
    for _, lr in _result().canonical.fold_result.level_items:
        assert not lr.provenance.target_fit_ids
        assert dict(lr.invariant_items)["oaci_rejected_ineligible_rows"] == 0


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}", file=sys.stderr)
    print(f"PASS  {len(fns)} bnci-gpu-runner tests", file=sys.stderr)


if __name__ == "__main__":
    _run_all()
