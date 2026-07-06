"""Project A — smoke generator for the audited evaluation bridge (no training).

Feeds a fake `run_three_settings`-style output (strict_dg / offline_tta / online_tta + leakage)
through the bridge and writes an audited ObservabilityReport. It demonstrates the Step-6
discipline: strict-DG target bAcc and TTA gain are REPORTABLE oracle/evaluation-only metrics
with identifiable_estimand=null; the offline-TTA target prior is identifiable only because
C1∧C2∧C3 are declared; leakage is a diagnostic; forbidden_claims_violated == [].

Run:  conda run -n icml python notes/project_A_observability/examples/make_audited_eval_bridge_smoke.py
Outputs:
  notes/project_A_observability/examples/out/audited_eval_bridge_smoke.json
  notes/project_A_observability/examples/out/audited_eval_bridge_smoke.md
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from h2cmi.observability import (ContractID as C, assert_forbidden_claims_not_made,  # noqa: E402
                                 build_audited_eval_report, report_to_dict,
                                 write_observability_report_json, write_observability_report_md)

# fake harness output (mirrors run_three_settings shapes; no model is trained)
STRICT_DG = {"setting": "strict_dg", "balanced_acc": 0.72, "worst_domain_bacc": 0.61}
OFFLINE_TTA = {"identity": {"balanced_acc": 0.70}, "adapt": {"balanced_acc": 0.76},
               "delta_adapt": {"d_balanced_acc": 0.06},
               "selective_risk": {"coverage": 0.8, "avoided_harm": 0.03}}
ONLINE_TTA = {"setting": "online_tta", "balanced_acc": 0.73}
LEAKAGE = {"site": {"I_hat": 0.12, "excess": 0.04}, "subject": {"I_hat": 0.30, "excess": 0.21}}


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_audited_eval_report(
        "Project A audited eval bridge smoke",
        strict_dg=STRICT_DG, offline_tta=OFFLINE_TTA, online_tta=ONLINE_TTA, leakage=LEAKAGE,
        prior_contracts={C.C1, C.C2, C.C3},          # declares the TU-1 contracts for the TTA prior
        has_oracle_target_labels=True)               # benchmark eval labels available (oracle)

    assert_forbidden_claims_not_made(report)         # clean report: no rejected conclusion

    json_path = out_dir / "audited_eval_bridge_smoke.json"
    md_path = out_dir / "audited_eval_bridge_smoke.md"
    data = write_observability_report_json(report, str(json_path))
    write_observability_report_md(report, str(md_path))

    for claim, verdict in report.entries:
        if not verdict.allowed:
            tag = "REJECTED"
        elif verdict.identifiable:
            tag = "IDENTIFIABLE"
        else:
            tag = "REPORTABLE(oracle)"
        print(f"{tag:18s} {claim.regime.value:2s} {claim.estimand.value:16s} {claim.name}")
    s = data["summary"]
    print(f"\nsummary: {s['n_allowed']} allowed / {s['n_rejected']} rejected ({s['n_claims']} claims); "
          f"forbidden-claim violations: {len(data['forbidden_claims_violated'])}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("AUDITED EVAL BRIDGE SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
