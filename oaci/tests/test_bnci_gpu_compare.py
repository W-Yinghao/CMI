"""B2b-i CPU tests: the cross-run scientific comparator (via the order-invariant fake closed loop) and
the BN-buffer audit (on a ShallowConvNet). The real GPU run is in the dedicated GPU smoke.

Standalone (``python -m oaci.tests.test_bnci_gpu_compare``) and pytest-compatible.
"""
from __future__ import annotations

import dataclasses
import os
import tempfile

import oaci.protocol
from oaci.artifacts.writer import GitEvidence, git_evidence_hash
from oaci.models import build_model
from oaci.models.bn_audit import BNBufferAudit, bn_buffer_hash, bn_buffer_state
from oaci.runner.bnci_gpu_compare import comparison_all_equal, compare_scientific_results, flatten_scientific
from oaci.runner.fake_artifact import run_fake_two_level

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")
_CANON = ("ERM", "OACI", "global_lpc", "uniform")
_REV = ("uniform", "global_lpc", "OACI", "ERM")
_C = {}


def _ge():
    c, t = "c" * 40, "t" * 40
    return GitEvidence(c, t, ("oaci",), (), True, git_evidence_hash(c, t, ("oaci",), (), True))


def _pair():
    if "p" not in _C:
        a = run_fake_two_level(_MAN, tempfile.mkdtemp(), model_seed=0, method_order=_CANON, repo_root="/x", git_evidence=_ge())
        b = run_fake_two_level(_MAN, tempfile.mkdtemp(), model_seed=0, method_order=_REV, repo_root="/x", git_evidence=_ge())
        _C["p"] = (a, b)
    return _C["p"]


def _shallow_state():
    m = build_model("shallow_convnet", in_chans=22, in_times=385, n_classes=4, temporal_filters=40,
                    temporal_kernel_samples=25, pool_kernel_samples=75, pool_stride_samples=15, dropout=0.5,
                    safe_log_eps=1e-6)
    return {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}


# ============================ comparator ============================
def test_order_comparator_is_all_equal_on_order_invariant_runs():
    a, b = _pair()
    cmp = compare_scientific_results(a, b)
    assert comparison_all_equal(cmp) and cmp.first_mismatch is None


def test_order_reproduces_every_scientific_group():
    cmp = compare_scientific_results(*_pair())
    assert (cmp.fold_result_equal and cmp.artifact_scientific_hash_equal and cmp.checkpoint_hashes_equal
            and cmp.trajectory_hashes_equal and cmp.selection_hashes_equal and cmp.audit_hashes_equal
            and cmp.prediction_hashes_equal and cmp.metrics_hashes_equal and cmp.plan_hashes_equal)


def test_order_comparator_reports_exact_first_path():
    a, b = _pair()
    # tamper one method's selection model hash in run B -> the comparator reports the exact path
    lr = b.fold_result.levels[1]
    m0 = lr.method_items[1]                                  # (name, MethodRunResult)
    bad_m = dataclasses.replace(m0[1], selection=dataclasses.replace(m0[1].selection, model_hash="DEADBEEF"))
    bad_items = tuple((n, bad_m if n == m0[0] else mm) for n, mm in lr.method_items)
    bad_lr = dataclasses.replace(lr, method_items=bad_items)
    bad_fr = dataclasses.replace(b.fold_result, level_items=tuple((lv, bad_lr if lv == 1 else l) for lv, l in b.fold_result.level_items))
    cmp = compare_scientific_results(a, dataclasses.replace(b, fold_result=bad_fr))
    assert not comparison_all_equal(cmp)
    assert cmp.first_mismatch.path == f"levels/1/methods/{m0[0]}/selection/model_hash"


def test_transport_file_hashes_are_excluded_from_scientific_comparison():
    a, _ = _pair()
    flat = flatten_scientific(a)
    keys = " ".join(flat)
    assert "artifact_index" not in keys and "file_sha" not in keys and "artifact_dir" not in keys
    assert "artifact_scientific_hash" in flat and "fold_result_hash" in flat


# ============================ BN audit ============================
def test_bn_buffer_state_matches_running_stats_only():
    bn = bn_buffer_state(_shallow_state())
    assert set(bn) == {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
    assert len(bn_buffer_hash(_shallow_state())) == 64


def test_bn_buffer_hash_changes_with_running_mean():
    st = _shallow_state()
    base = bn_buffer_hash(st)
    st["bn.running_mean"] = st["bn.running_mean"] + 1.0
    assert bn_buffer_hash(st) != base


def test_bn_audit_record_flags_equality():
    h = bn_buffer_hash(_shallow_state())
    assert BNBufferAudit(0, "OACI", "ck", h, h, True).equal_to_erm is True
    assert BNBufferAudit(0, "OACI", "ck", h, "other", False).equal_to_erm is False


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runner.bnci_gpu_smoke  # noqa: F401
    import oaci.runner.bnci_gpu_compare  # noqa: F401
    import oaci.models.bn_audit  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} bnci-gpu-compare tests")


if __name__ == "__main__":
    _run_all()
