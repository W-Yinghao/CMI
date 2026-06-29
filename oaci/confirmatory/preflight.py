"""Confirmatory one-fold REAL-DATA preflight (CPU, no training, no GPU).

Materialize the pilot manifest, build the real BNCI2014_001 fold once, and report the fold contract
(X geometry, role counts, role subjects, target-not-fit, data/split hashes) so a fold-build bug is caught
on a cheap CPU node BEFORE any V100 allocation. Prints ONE canonical-JSON report to stdout; MNE/MOABB
chatter -> stderr. Offline-only (the loader forbids network).

    python -m oaci.confirmatory.preflight --protocol oaci/protocol/confirmatory_v2.yaml \
        --datalake-root "$OACI_DATALAKE_ROOT" --manifest-out "$OUT/pilot_manifest.yaml" --target-subject 1
"""
from __future__ import annotations

import argparse
import contextlib
import sys

from ..artifacts.canonical_json import canonical_json_bytes
from ..runner.bnci_data import build_bnci_real_fold, target_seen_by_fit
from .materialize import materialize_pilot_manifest
from .onefold import DATASET
from .schema import load_confirmatory


def build_preflight_report(fold, manifest_path, target_subject) -> dict:
    fd = fold.fold_data
    sm = fold.manifest.pilot
    n = int(fd.X.shape[0])
    counts = {"source_train": int(len(fd.source_train_idx)), "source_audit": int(len(fd.source_audit_idx)),
              "target_audit": int(len(fd.target_audit_idx))}
    role_sum = sum(counts.values())
    target_fit = bool(target_seen_by_fit(fd))
    acceptance_ok = (role_sum == n and all(v > 0 for v in counts.values()) and not target_fit)
    return {
        "notice": "confirmatory one-fold real-data preflight (fold build only; no training; not efficacy)",
        "manifest_path": manifest_path, "manifest_hash": fold.manifest_hash, "target_subject": int(target_subject),
        "subjects": list(sm.subjects), "target_subjects": list(sm.target_subjects),
        "source_audit_subjects": list(sm.source_audit_subjects),
        "source_train_subjects": list(sm.source_train_subjects),
        "X_shape": [int(d) for d in fd.X.shape], "n_classes": len(fd.class_names),
        "class_names": list(fd.class_names), "role_counts": counts, "role_count_sum": role_sum,
        "rows_total": n, "roles_partition_all_rows": role_sum == n,
        "target_seen_by_fit": target_fit,
        "data_evidence_hash": fold.data_evidence_hash, "resolved_preprocess_hash": fold.resolved_preprocess_hash,
        "split_manifest_hash": fold.split_manifest_hash, "shallow_geometry": fold.shallow_geometry,
        "acceptance_ok": acceptance_ok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.confirmatory.preflight")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--datalake-root", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--target-subject", type=int, default=1)
    ap.add_argument("--model-seeds", default="0,1,2")
    args = ap.parse_args(argv)
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):           # MNE/MOABB logs -> stderr, keep stdout JSON-clean
            proto = load_confirmatory(args.protocol)
            mp, _m = materialize_pilot_manifest(
                proto, args.dataset, target_subject=args.target_subject, out_path=args.manifest_out,
                model_seeds=[int(s) for s in args.model_seeds.split(",")])
            fold = build_bnci_real_fold(mp, args.datalake_root)
            fold.fold_data.assert_integrity()
            report = build_preflight_report(fold, mp, args.target_subject)
        real_stdout.buffer.write(canonical_json_bytes(report))
        return 0 if report["acceptance_ok"] else 1
    except Exception as e:  # noqa: BLE001
        print(f"confirmatory preflight failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
