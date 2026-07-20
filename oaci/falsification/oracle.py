"""C14 gate G4 — oracle rescue. The escape-hatch killer: from C10's counterfactual selector replay, can the
source_audit ORACLE (S5 — allowed to peek at held-out SOURCE, never target) pick an OACI-trajectory
checkpoint that recovers reproducible worst-domain K2 gain? If not, the failure is not a bad selector or a
bad split — the trajectory contains no source-side-identifiable downstream-winning checkpoint."""
from __future__ import annotations

from .schema import G4, ORACLE_FAIL, ORACLE_RESCUE, gate

_ORACLE = "S5_source_audit_oracle"


def g4_oracle_rescue(c10) -> dict:
    p2 = c10["part2_selector_replay"]
    sel = p2["selectors"]
    oracle_repro = bool(p2.get("oracle_reproducible"))
    source_only = list(p2.get("source_only_reproducible") or [])
    s5 = sel[_ORACLE]["k2_status"]
    s0 = p2.get("s0_current_k2")
    status = ORACLE_RESCUE if oracle_repro else ORACLE_FAIL
    return gate(G4, status, oracle_k2_status=s5, oracle_reproducible=oracle_repro,
                source_only_selectors_reproducing=source_only, s0_current_k2=s0,
                final_case=p2.get("final_case"),
                # oracle picked an OACI ckpt this often instead of ERM (from the summary table, if present)
                per_selector_k2={k: v["k2_status"] for k, v in sel.items()})
