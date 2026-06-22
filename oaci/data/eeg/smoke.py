"""Data pre-check / audit — reads a FEW real recordings and reports the data contract. NO
training. Run: ``python -m oaci.data.eeg.smoke``.

BNCI2014-001 (2 subjects) is read for real (full EEGBundle + eligibility/mass tables + split +
hashes); SEED-V and clinical cohorts are scanned for feasibility.
"""
from __future__ import annotations

import numpy as np

from ..eeg.audit import split_manifest_hash, tensor_hash
from ..eeg.clinical_bids import clinical_feasibility
from ..eeg.moabb import load_moabb
from ..eeg.preprocess import PreprocessSpec
from ..eeg.registry import get_entry
from ..eeg.schema import EEGBundle
from ..eeg.seed import scan_seed
from ..eeg.splits import make_loso_split
from ..eeg.units import base_mass, cell_mass, eligibility_counts


def audit_bnci(subjects=(1, 2), spec: PreprocessSpec | None = None) -> dict:
    spec = spec or PreprocessSpec()
    b: EEGBundle = load_moabb("BNCI2014_001", subjects, spec).validate()
    dom = b.domain("subject_id")
    nd, nc = int(dom.max()) + 1, len(b.class_names)
    base = base_mass(b.eval_unit_id)
    nelig = eligibility_counts(dom, b.y, b.support_unit_id, nd, nc)
    M = cell_mass(dom, b.y, base, nd, nc)
    # LOSO split (target = first source subject); MI -> domain=subject, whole-subject audit
    split = make_loso_split(dom, b.subject_id, target_domain=0, split_seed=0, ensure_train_per_domain=False)
    target_seen_by_fit = bool(set(split.target_audit.tolist()) &
                              (set(split.source_train.tolist()) | set(split.source_audit.tolist())))
    return {
        "dataset_fingerprint": b.raw_data_fingerprint, "preprocess_hash": b.preprocess_hash,
        "tensor_content_hash": b.tensor_content_hash,
        "X_shape": list(b.X.shape), "sfreq": b.sfreq, "ch_order": list(b.ch_names),
        "n_samples": b.n, "n_recordings": int(np.unique(b.recording_id).size),
        "n_support_units": int(np.unique(b.support_unit_id).size),
        "n_eval_units": int(np.unique(b.eval_unit_id).size),
        "class_names": list(b.class_names),
        "eligibility_counts": nelig.tolist(), "cell_mass": np.round(M, 2).tolist(),
        "split": {"source_train": int(len(split.source_train)), "source_audit": int(len(split.source_audit)),
                  "target_audit": int(len(split.target_audit)), "roles_disjoint": split.roles_disjoint(),
                  "n_active_source_domains": split.n_active_source_domains,
                  "method_inactive": split.method_inactive,
                  "split_manifest_hash": split_manifest_hash(split),
                  "audit_tensor_hash": tensor_hash(b.X[split.target_audit])},
        "target_seen_by_fit": target_seen_by_fit,
    }


def _demo() -> None:
    print("OACI data pre-check (NO training) — reads 2 real BNCI subjects + scans others")
    try:
        rep = audit_bnci((1, 2))
        print("\n[BNCI2014_001]")
        for k in ("dataset_fingerprint", "preprocess_hash", "tensor_content_hash", "X_shape", "sfreq"):
            print(f"  {k:22s} = {rep[k]}")
        print(f"  ch_order (first 8)     = {rep['ch_order'][:8]}")
        print(f"  n_samples/rec/support/eval = {rep['n_samples']} / {rep['n_recordings']} / "
              f"{rep['n_support_units']} / {rep['n_eval_units']}")
        print(f"  classes                = {rep['class_names']}")
        print(f"  eligibility_counts[d,y]= {rep['eligibility_counts']}")
        print(f"  cell_mass[d,y]         = {rep['cell_mass']}")
        s = rep["split"]
        print(f"  split train/audit/target = {s['source_train']}/{s['source_audit']}/{s['target_audit']}"
              f"  disjoint={s['roles_disjoint']}  active_src_domains={s['n_active_source_domains']}"
              f"  method_inactive={s['method_inactive']}")
        print(f"  split_manifest_hash    = {s['split_manifest_hash']}  audit_tensor_hash={s['audit_tensor_hash']}")
        print(f"  target_seen_by_fit     = {rep['target_seen_by_fit']}")
    except Exception as e:
        print(f"[BNCI2014_001] READ FAILED (offline): {type(e).__name__}: {e}")

    print("\n[SEED-V]", scan_seed())
    print("[PD_cross_site]", clinical_feasibility(get_entry("PD_cross_site")))
    print("[SCZ_cross_site]", clinical_feasibility(get_entry("SCZ_cross_site")))


if __name__ == "__main__":
    _demo()
