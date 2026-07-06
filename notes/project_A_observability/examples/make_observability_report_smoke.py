"""Project A — smoke generator for an observability report (no training).

Constructs the 5 canonical claims of `08_experimental_protocol.md` and writes a machine- and
human-readable observability report showing which are LICENSED and which the audit REJECTS
(with the firing certificate). The two rejected claims are marked `conclusion=False` — they are
demonstrations of what the audit blocks, not finalised conclusions — so the clean report passes
`assert_forbidden_claims_not_made`.

Run:  conda run -n icml python notes/project_A_observability/examples/make_observability_report_smoke.py
Outputs:
  notes/project_A_observability/examples/out/observability_report_smoke.json
  notes/project_A_observability/examples/out/observability_report_smoke.md
"""
from __future__ import annotations

import sys
from pathlib import Path

# repo root is 3 parents up: <repo>/notes/project_A_observability/examples/<this file>
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from h2cmi.observability import (Claim, ContractID as C, Estimand, Regime,  # noqa: E402
                                 assert_forbidden_claims_not_made, build_report,
                                 write_observability_report_json, write_observability_report_md)


CLAIMS = [
    # 1. licensed: R0 source LOSO balanced accuracy is a source validation metric
    Claim("source_loso_bacc", Regime.R0, Estimand.SOURCE_LOSO,
          observed=("X_s", "Y_s", "D_s"), estimator="EEGNet + LOSO"),
    # 2. REJECTED (demo): R0 target gain — non-identifiable under R0 (TOS-1 / CE-R0-2)
    Claim("target_gain_r0", Regime.R0, Estimand.TARGET_GAIN,
          estimator="source safety gate", conclusion=False),
    # 3. licensed: R1 target prior under C1∧C2∧C3 (TU-1)
    Claim("target_prior_r1", Regime.R1, Estimand.TARGET_PRIOR,
          contracts={C.C1, C.C2, C.C3}, observed=("X_T",),
          estimator="mixture w=C^-1 mu"),
    # 4. REJECTED (demo): R1 target concept — non-identifiable from unlabeled target (TU-2 / CE-R1-1)
    Claim("target_concept_r1", Regime.R1, Estimand.TARGET_CONCEPT,
          observed=("X_T",), estimator="decoder I(Y;D|Z)", conclusion=False),
    # 5. licensed: R2 transport under C8∧C11 (MP-1)
    Claim("transport_r2", Regime.R2, Estimand.TARGET_TRANSPORT,
          contracts={C.C8, C.C11}, observed=("anchors", "paired_sessions"),
          estimator="near-identity affine (A,b)"),
]


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report("Project A smoke — 5-claim ledger", CLAIMS)

    # the two rejected items are demonstrations (conclusion=False), so a clean report passes:
    assert_forbidden_claims_not_made(report)

    json_path = out_dir / "observability_report_smoke.json"
    md_path = out_dir / "observability_report_smoke.md"
    data = write_observability_report_json(report, str(json_path))
    write_observability_report_md(report, str(md_path))

    # console summary
    for claim, verdict in report.entries:
        mark = "ALLOWED " if verdict.allowed else "REJECTED"
        cert = f"  [{verdict.failure_certificate}]" if (verdict.rejected and verdict.failure_certificate) else ""
        thm = f"  ({verdict.theorem})" if verdict.theorem else ""
        print(f"{mark}  {claim.regime.value:2s} {claim.estimand.value:20s} {claim.name}{thm}{cert}")
    s = data["summary"]
    print(f"\nsummary: {s['n_allowed']} allowed / {s['n_rejected']} rejected "
          f"({s['n_claims']} claims); forbidden-claim violations: "
          f"{len(data['forbidden_claims_violated'])}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("OBSERVABILITY REPORT SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
