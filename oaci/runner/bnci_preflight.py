"""Six-subject BNCI CPU preflight (B1b): load real data ONCE, build FoldData/FoldScope/level supports,
assert the exact contract, and emit one canonical-JSON summary. No model training, no artifact write.

    python -m oaci.runner.bnci_preflight --manifest oaci/protocol/smoke_v1.yaml --datalake-root "$OACI_DATALAKE_ROOT"
"""
from __future__ import annotations

import argparse
import sys

from ..artifacts.canonical_json import canonical_json_bytes
from ..data.eeg.bnci import _evidence_hash  # noqa: F401  (kept for parity)
from .bnci_data import build_bnci_real_fold, target_seen_by_fit
from .support import build_level_support, level0_reference_prior

_EXPECTED = {"X_shape": [3456, 22, 385], "role_trials": {"source_train": 1728, "source_audit": 1152, "target_audit": 576},
             "role_recordings": {"source_train": 36, "source_audit": 24, "target_audit": 12}, "headers": 72}


def build_preflight_summary(fold) -> dict:
    fd = fold.fold_data
    ev = fold.load_result.evidence
    maps, sch = fold.maps, fold.deletion_schedule
    ref = level0_reference_prior(fd, maps)
    ss0 = build_level_support(fd, maps, 0, sch, ref, support_m=int(fold.manifest.enabled_datasets()["BNCI2014_001"].support_m))
    ss1 = build_level_support(fd, maps, 1, sch, ref, support_m=int(fold.manifest.enabled_datasets()["BNCI2014_001"].support_m))

    role_trials = {r: int(len(fd.role_ids(r))) for r in ("source_train", "source_audit", "target_audit")}
    role_recordings = {r: int(len({fd.group_id[i] for i in idx.tolist()}))
                       for r, idx in (("source_train", fd.source_train_idx), ("source_audit", fd.source_audit_idx),
                                      ("target_audit", fd.target_audit_idx))}
    au = fold.fold_scope.source_audit
    # deleted cell (subject-004, feet) index in the maps
    dom_idx = maps.source_domain_to_index["BNCI2014_001|subject-004"]
    cls_idx = list(fold.fold_data.class_names).index("feet")

    summary = {
        "manifest_hash": fold.manifest_hash, "raw_file_count": ev.raw_file_count,
        "raw_data_fingerprint": ev.raw_data_fingerprint, "resolved_preprocess_hash": fold.resolved_preprocess_hash,
        "full_tensor_hash": fold.full_tensor_hash, "data_contract_hash": fd.data_contract_hash,
        "split_manifest_hash": fold.split_manifest_hash, "data_evidence_hash": fold.data_evidence_hash,
        "fold_scope_hash": fold.fold_scope.fold_scope_hash,
        "network_attempt_count": int(ev.network_attempt_count),
        "raw_fingerprint_unchanged": ev.raw_data_fingerprint == ev.raw_data_fingerprint_after,
        "library_versions": [list(x) for x in ev.library_versions],
        "X_shape": list(ev.actual_shape), "dtype": ev.output_dtype, "actual_sfreq": ev.actual_sfreq,
        "actual_n_times": ev.actual_n_times, "channel_order": list(ev.common_eeg_channels),
        "header_count": ev.header_record_count, "header_hashes": [r.header_hash for r in ev.header_records],
        "class_count_table": [list(x) for x in ev.class_count_table],
        "recording_count_table": [list(x) for x in ev.recording_count_table],
        "role_trials": role_trials, "role_recordings": role_recordings,
        "shallow_geometry": fold.shallow_geometry,
        "level0": {"eligibility_counts": ss0.eligibility_counts.tolist(), "cell_mass": ss0.cell_mass.tolist(),
                   "p_ref": [float(x) for x in ss0.support_graph.reference_prior.tolist()]},
        "level1": {"eligibility_counts": ss1.eligibility_counts.tolist(), "cell_mass": ss1.cell_mass.tolist(),
                   "p_ref": [float(x) for x in ss1.support_graph.reference_prior.tolist()],
                   "deleted_cell": {"count": float(ss1.eligibility_counts[dom_idx, cls_idx]),
                                    "mass": float(ss1.cell_mass[dom_idx, cls_idx]),
                                    "rows": int(sum(1 for i in ss1.source_train_idx.tolist()
                                                    if fd.domain_id[i] == "BNCI2014_001|subject-004" and int(fd.y[i]) == cls_idx))},
                   "source_train_rows": int(len(ss1.source_train_idx))},
        "method_statuses": {n: bool(st.active) for n, st in ss1.method_status_items},
        "audit_status": au.status, "audit_fold_plan_hash": None if au.fold_plan is None else au.fold_plan.plan_hash,
        "audit_bootstrap_plan_hash": None if au.bootstrap_plan is None else au.bootstrap_plan.plan_hash,
        "target_seen_by_fit": target_seen_by_fit(fd), "excluded_recordings": list(ev.excluded_recordings)}
    summary["acceptance_ok"] = _acceptance(summary)
    return summary


def _acceptance(s) -> bool:
    return (s["X_shape"] == _EXPECTED["X_shape"] and s["role_trials"] == _EXPECTED["role_trials"]
            and s["role_recordings"] == _EXPECTED["role_recordings"] and s["header_count"] == _EXPECTED["headers"]
            and s["network_attempt_count"] == 0 and s["raw_fingerprint_unchanged"]
            and s["target_seen_by_fit"] is False and s["excluded_recordings"] == []
            and s["level0"]["eligibility_counts"] == [[144, 144, 144, 144]] * 3
            and s["level1"]["deleted_cell"] == {"count": 0.0, "mass": 0.0, "rows": 0}
            and s["level1"]["source_train_rows"] == 1584
            and all(s["method_statuses"].values()) and s["audit_status"] == "estimable"
            and all(c == 864 for _, c in s["class_count_table"])   # 6 subjects x 144 per class
            and s["level0"]["cell_mass"] == [[144.0, 144.0, 144.0, 144.0]] * 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.runner.bnci_preflight")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--datalake-root", required=True)
    args = ap.parse_args(argv)
    import contextlib
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):           # MNE/MOABB chatter -> stderr, never stdout
            fold = build_bnci_real_fold(args.manifest, args.datalake_root)
            summary = build_preflight_summary(fold)
        real_stdout.buffer.write(canonical_json_bytes(summary))
        return 0 if summary["acceptance_ok"] else 1
    except Exception as e:  # noqa: BLE001
        print(f"bnci preflight failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
