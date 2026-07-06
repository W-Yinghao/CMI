"""Project A — real-EEG audit pilot (INTERFACE CHECK ONLY; no training, no heavy data load).

Purpose: verify that the real-EEG loader interface is importable and that a real Tier-2 run
would flow through the audited evaluation bridge. It does NOT load full datasets or train models
— an actual Tier-2 run belongs on SLURM (see cmi/*.slurm; project memory: heavy jobs via sbatch,
not the login node). It writes an audited-report SKELETON with PLACEHOLDER metrics so the claim
boundary a real run must carry is explicit. It gracefully SKIPS (exit 0) if the loader is
unavailable, so Step 6 never fails on a missing local cache.

Run:  conda run -n icml python notes/project_A_observability/examples/run_real_eeg_audit_pilot.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _probe_loader():
    """Import-only probe of the MOABB loader interface. Never loads data or trains a model."""
    last = "no loader module tried"
    for modname in ("cmi.data.moabb_data", "cmi.data.processed_data"):
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:                       # reason-coded, not silently swallowed
            last = f"{modname}: {type(exc).__name__}: {exc}"
            continue
        have = {fn: hasattr(mod, fn) for fn in ("load", "domain_labels", "loso_splits")}
        if any(have.values()):
            return True, "ok", {"module": modname, "functions": have}
        last = f"{modname}: none of load/domain_labels/loso_splits present"
    return False, last, {}


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, reason, info = _probe_loader()
    if not ok:
        print(f"REAL EEG AUDIT PILOT SKIPPED: {reason}")
        return 0                                        # graceful skip; Step 6 must not fail here

    from h2cmi.observability import (ContractID as C, build_audited_eval_report, report_to_dict,
                                     write_observability_report_json,
                                     write_observability_report_md)

    # PLACEHOLDER metrics (NOT measured — real numbers require a SLURM Tier-2 run). The audited
    # report is about the claim BOUNDARY, not the values, so placeholders exercise the same audit.
    strict_dg = {"balanced_acc": None, "worst_domain_bacc": None}
    offline_tta = {"delta_adapt": {"d_balanced_acc": None}}
    online_tta = {"balanced_acc": None}
    leakage = {"site": {"I_hat": None}, "subject": {"I_hat": None}, "session": {"I_hat": None}}

    report = build_audited_eval_report(
        "REAL-EEG PILOT (interface-only, placeholder metrics) — BNCI2014_001 2a LOSO",
        strict_dg=strict_dg, offline_tta=offline_tta, online_tta=online_tta, leakage=leakage,
        prior_contracts={C.C1, C.C2, C.C3},            # declared TU-1 contracts for the TTA prior
        has_oracle_target_labels=True)

    json_path = out_dir / "real_eeg_audit_pilot.json"
    md_path = out_dir / "real_eeg_audit_pilot.md"
    data = write_observability_report_json(report, str(json_path))
    write_observability_report_md(report, str(md_path))

    s = data["summary"]
    print(f"loader interface verified: {info['module']} {info['functions']}")
    print(f"audited report skeleton: {s['n_allowed']} allowed / {s['n_rejected']} rejected "
          f"({s['n_claims']} claims); forbidden-claim violations: "
          f"{len(data['forbidden_claims_violated'])}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("REAL EEG AUDIT PILOT OK (interface verified; metrics are PLACEHOLDERS — "
          "run Tier-2 on SLURM for real numbers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
