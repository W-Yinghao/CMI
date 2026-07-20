"""Real BNCI six-subject contract -- loads the real data ONCE and asserts the exact contract. NOT in
normal CI: run only by the dedicated BNCI preflight SLURM job (needs OACI_DATALAKE_ROOT).

    OACI_DATALAKE_ROOT=/projects/EEG-foundation-model/datalake/raw \
        python -m oaci.tests.test_bnci_real_preflight
"""
from __future__ import annotations

import contextlib
import os
import sys

import oaci.protocol
from oaci.runner.bnci_data import build_bnci_real_fold, target_seen_by_fit
from oaci.runner.bnci_preflight import build_preflight_summary

_SMOKE = os.path.join(os.path.dirname(oaci.protocol.__file__), "smoke_v1.yaml")
_ROOT = os.environ.get("OACI_DATALAKE_ROOT", "/projects/EEG-foundation-model/datalake/raw")
_C = {}


def _fold():
    if "f" not in _C:
        with contextlib.redirect_stdout(sys.stderr):          # keep MNE chatter off stdout
            _C["f"] = build_bnci_real_fold(_SMOKE, _ROOT)
    return _C["f"]


def _summary():
    if "s" not in _C:
        with contextlib.redirect_stdout(sys.stderr):
            _C["s"] = build_preflight_summary(_fold())
    return _C["s"]


def test_bnci_total_trial_and_recording_counts_are_exact():
    s = _summary()
    assert s["X_shape"] == [3456, 22, 385] and s["header_count"] == 72


def test_bnci_role_counts_are_exact():
    s = _summary()
    assert s["role_trials"] == {"source_train": 1728, "source_audit": 1152, "target_audit": 576}
    assert s["role_recordings"] == {"source_train": 36, "source_audit": 24, "target_audit": 12}


def test_bnci_each_subject_class_has_144_trials():
    s = _summary()
    assert all(c == 864 for _, c in s["class_count_table"])                # 6 subjects x 144
    assert s["level0"]["cell_mass"] == [[144.0, 144.0, 144.0, 144.0]] * 3   # per source subject x class


def test_bnci_no_network_and_raw_unchanged():
    s = _summary()
    assert s["network_attempt_count"] == 0 and s["raw_fingerprint_unchanged"] and s["excluded_recordings"] == []


def test_bnci_actual_shape_sfreq_channels_and_n_times_are_exact():
    fold = _fold(); ev = fold.load_result.evidence
    assert ev.actual_sfreq == 128.0 and ev.actual_n_times == 385
    assert list(ev.common_eeg_channels) == list(fold.manifest.enabled_datasets()["BNCI2014_001"].channels)


def test_bnci_level0_support_table_is_exact():
    s = _summary()
    assert s["level0"]["eligibility_counts"] == [[144, 144, 144, 144]] * 3
    assert s["level0"]["p_ref"] == [0.25, 0.25, 0.25, 0.25]


def test_bnci_level1_deleted_cell_is_exactly_zero():
    s = _summary()
    assert s["level1"]["eligibility_counts"] == [[144, 144, 0, 144], [144, 144, 144, 144], [144, 144, 144, 144]]
    assert s["level1"]["deleted_cell"] == {"count": 0.0, "mass": 0.0, "rows": 0}
    assert s["level1"]["source_train_rows"] == 1584


def test_bnci_reference_prior_is_fixed_across_levels():
    s = _summary()
    assert s["level0"]["p_ref"] == s["level1"]["p_ref"] == [0.25, 0.25, 0.25, 0.25]


def test_bnci_all_method_statuses_are_active():
    assert all(_summary()["method_statuses"].values())


def test_bnci_audit_scope_is_estimable_with_plans():
    s = _summary()
    assert s["audit_status"] == "estimable" and s["audit_fold_plan_hash"] and s["audit_bootstrap_plan_hash"]


def test_bnci_target_never_enters_fit_ids():
    assert target_seen_by_fit(_fold().fold_data) is False and _summary()["target_seen_by_fit"] is False


def test_bnci_preflight_acceptance_ok():
    assert _summary()["acceptance_ok"]


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}", file=sys.stderr)
    print(f"PASS  {len(fns)} bnci-real-preflight tests", file=sys.stderr)


if __name__ == "__main__":
    _run_all()
