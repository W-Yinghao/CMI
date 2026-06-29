"""ACAR v4 — fixed-candidate DEV substrate-compatibility replay command (B1-gated; FAIL-CLOSED at execution).

The ONE frozen command that would (post-B1, after the substrate is trained) re-embed the OLD SEVEN DEV cohorts with the NEW
all-DEV substrate and replay the FIXED candidate (shift_margin+benefit_ranked+harm_indicator; NO reselection) to decide —
via the pre-registered numeric pass-line `regen_substrate.compatibility_replay_pass` (v2_replay a HARD requirement) —
whether external Arm B may run. It is NOT authorized here: after a full STDLIB-FIRST preflight it raises
SubstrateCompatibilityNotAuthorizedError BEFORE any torch/cmi import or DEV read. THIS IS NOT A NEW DEV SELECTION RUN.

Usage (NOT yet runnable to completion — fails closed):
    python -m acar.v4.run_substrate_compatibility --substrate-manifest /abs/substrate_manifest.json --output /abs/new_compat_dir
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import sys

from acar.v4 import regen_substrate as RS
from acar.v4.run_regen_substrate import _repo_root, _verify_commit, _verify_clean, _sha256_file  # noqa: F401


def run(substrate_manifest_path, output):
    """STDLIB-FIRST, FAIL-CLOSED. Validates the substrate manifest (fixed candidate, pinned operating point, trained-artifact
    hashes) + git/clean/output preflight, then refuses to replay (B1 not signed off). NO torch/cmi import, NO DEV read."""
    with open(substrate_manifest_path, "rb") as f:
        raw = f.read()
    input_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    spec = json.loads(raw.decode())
    RS.validate_substrate_manifest(spec)                              # fixed candidate + pinned op-point + artifact hashes
    root = _repo_root()
    _verify_commit(root, spec["protocol_commit"])
    _verify_clean(root)
    if os.path.exists(output):
        raise FileExistsError(f"output dir already exists (no overwrite): {output}")
    for d in ("PD", "SCZ"):                                           # trained-artifact file-hash preflight (no torch/DEV)
        sd = spec["substrates"][d]
        for path_key, sha_key in (("encoder_checkpoint_path", "encoder_checkpoint_sha256"),
                                  ("source_state_path", "source_state_sha256")):
            p = sd[path_key]
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{d}: substrate artifact missing: {p}")
            got = _sha256_file(p)
            if got != sd[sha_key]:
                raise ValueError(f"{d}: {path_key} sha mismatch ({got} != {sd[sha_key]})")
    report = {"input_manifest_sha256": input_manifest_sha256, "candidate": RS.FIXED_CANDIDATE,
              "pass_line": {"coverage_min": RS.COVERAGE_MIN, "budget": RS.BUDGET, "alpha": RS.ALPHA,
                            "v2_replay": "HARD requirement (no waiver)"},
              "expected_output": RS.expected_compat_output(output),
              "command": shlex.join([sys.executable, "-m", "acar.v4.run_substrate_compatibility",
                                     "--substrate-manifest", substrate_manifest_path, "--output", output])}
    raise RS.SubstrateCompatibilityNotAuthorizedError(
        "DEV substrate-compatibility replay is NOT authorized. The manifest validated (fixed candidate, pinned operating "
        "point, trained-artifact hashes) + preflight pass, but re-embedding DEV with the new substrate needs the trained "
        "artifacts + torch + DEV raw, gated behind B1. Decision uses regen_substrate.compatibility_replay_pass "
        "(v2_replay HARD). No torch/cmi import, no DEV read, no output written. report=" + json.dumps(report, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 fixed-candidate substrate-compatibility replay (B1-gated; fails closed)")
    ap.add_argument("--substrate-manifest", required=True)
    ap.add_argument("--output", required=True, help="must not exist")
    args = ap.parse_args(argv)
    return run(args.substrate_manifest, args.output)


if __name__ == "__main__":
    main()
