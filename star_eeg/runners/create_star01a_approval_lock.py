#!/usr/bin/env python
"""Create the post-00C SHA-named operational approval lock."""

import argparse
import json
from pathlib import Path

from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.training.approval_lock import (
    APPROVAL_HASH_FIELD,
    validate_approval_lock,
    write_approval_lock,
)
from star_eeg.training.persistence import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--approved-execution-commit", required=True)
    parser.add_argument("--approved-attempt-id", default="attempt_01")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    gate_path = repo_root / "results/star/star00c_preflight/star00c_go_nogo.json"
    gate = json.loads(gate_path.read_text())
    gate_core = {
        key: value for key, value in gate.items()
        if key != "star00c_go_nogo_hash"
    }
    if canonical_hash(gate_core) != gate.get("star00c_go_nogo_hash"):
        raise RuntimeError("STAR_00C go/no-go hash mismatch")
    if gate.get("status") != "PASS":
        raise PermissionError("STAR_00C has not passed")
    path = write_approval_lock(
        repo_root=repo_root,
        output_dir=Path(args.output_dir),
        approved_execution_commit=args.approved_execution_commit,
        approved_attempt_id=args.approved_attempt_id,
    )
    audit = validate_approval_lock(path, repo_root)
    payload = json.loads(path.read_text())
    print(json.dumps({
        "status": audit["status"],
        "approval_manifest": str(path.resolve()),
        "approval_manifest_hash": payload[APPROVAL_HASH_FIELD],
        "approval_file_sha256": sha256_file(path),
        "approved_execution_commit": audit["approved_execution_commit"],
        "approved_cells": audit["approved_cells"],
        "target_scoring": audit["target_scoring"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
