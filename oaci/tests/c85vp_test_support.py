"""Shadow-only fixtures for C85VP process-isolation tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oaci.theory.c85v_stage_b_candidate_audit import REQUIRED_MARKERS
from oaci.theory.c85v_statement_registry import (
    FrozenProofCandidateIdentity,
    FrozenTheoremStatement,
    THEOREM_IDS,
)


def shadow_statements() -> dict[str, FrozenTheoremStatement]:
    result: dict[str, FrozenTheoremStatement] = {}
    for theorem_id in THEOREM_IDS:
        text = f"SHADOW STATEMENT {theorem_id}: synthetic review fixture only."
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result[theorem_id] = FrozenTheoremStatement(theorem_id, text, digest)
    return result


def shadow_obligations() -> dict[str, tuple[str, ...]]:
    return {
        theorem_id: (f"shadow obligation for {theorem_id}",)
        for theorem_id in THEOREM_IDS
    }


def write_shadow_candidates(
    root: Path,
    statements: dict[str, FrozenTheoremStatement],
) -> dict[str, FrozenProofCandidateIdentity]:
    root.mkdir(parents=True)
    result: dict[str, FrozenProofCandidateIdentity] = {}
    for theorem_id in THEOREM_IDS:
        statement = statements[theorem_id]
        disposition = (
            "INCOMPLETE_OPEN"
            if theorem_id == "T5"
            else "PROPOSED_COUNTEREXAMPLE"
            if theorem_id in {"T2", "T6"}
            else "PROPOSED_PROOF"
        )
        marker_text = "; ".join(REQUIRED_MARKERS[theorem_id])
        relative = f"shadow/{theorem_id}.md"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f"""# Shadow {theorem_id}

## Exact Statement

{statement.text}

Statement SHA-256: `{statement.sha256}`

## Assumptions

- shadow assumption

## Proof Candidate Or Counterexample

{marker_text}

## Boundary Cases

- shadow boundary

## Candidate Disposition

`{disposition}`

## Proof Candidate Schema And Internal Consistency

Shadow fixture only.

## Formal Status

`OPEN`
"""
        path.write_text(text)
        result[theorem_id] = FrozenProofCandidateIdentity(
            theorem_id=theorem_id,
            relative_path=relative,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            known_disposition=disposition,
        )
    return result


def write_shadow_exact_results(path: Path) -> None:
    values = {f"S{index}": {} for index in range(11)}
    values["S1"] = {
        "coarse_registered_risk": "1/2",
        "rich_unrestricted_risk": "0",
        "rich_registered_risk": "1",
    }
    values["S5"] = {
        "candidate_open_lower": "13/20",
        "candidate_open_upper": "1",
        "endpoint_policy": "both endpoints excluded",
    }
    values["S10"] = {
        "coarse_risk": "11/40",
        "rich_unrestricted_risk": "0",
        "v2_rich_risk": "3/5",
        "reversal": "13/40",
    }
    path.write_text(json.dumps(values, sort_keys=True))
