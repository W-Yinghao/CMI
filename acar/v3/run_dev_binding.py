"""ACAR v3 binding DEV runner — the SINGLE frozen command from seven concrete cohort dumps to the binding S2/S4 gate.

    python -m acar.v3.run_dev_binding --input-manifest <seven-cohort.json> --output <new-dir> [--protocol-commit <sha>]

Before ANY DEV metric it FAILS CLOSED unless: the output dir does not exist; the current git HEAD equals the protocol
commit (and the immutable tag `acar-v3-dev-design-v1^{}` resolves to that HEAD); the environment lock verifies; and the
seven input files exist with field-separated hashes (full_dump / source_fit / deployment_input / label / subject_list)
matching those declared in the input manifest. Only then does it build the CohortInputs and call `freeze_dev_run`. The
first real DEV run produces ONLY a SELECT (+ frozen artifacts) or `DEV_STOP / NO_LOCKBOX_CONSUMED`; it never touches an
external Arm-B endpoint or a lockbox.

Input-manifest schema:
    {"protocol_commit": "<full-sha>",
     "cohorts": [{"dataset_id": "...", "disease": "PD"|"SCZ", "path": "...",
                  "full_dump_sha256": "...", "source_fit_sha256": "...", "deployment_input_sha256": "...",
                  "label_sha256": "...", "subject_list_sha256": "..."}, ... x7]}
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys

from .loader import build_cohort_input, file_sha256
from .develop import freeze_dev_run

TAG = "acar-v3-dev-design-v1"


def _git(args):
    return subprocess.check_output(["git"] + args, stderr=subprocess.DEVNULL).decode().strip()


def verify_protocol(protocol_commit, *, require_tag=True):
    """HEAD must equal the spec's protocol commit AND the immutable tag must resolve to that HEAD."""
    head = _git(["rev-parse", "HEAD"])
    if not protocol_commit or head != protocol_commit:
        raise ValueError(f"git HEAD {head} != protocol_commit {protocol_commit}")
    if require_tag:
        try:
            tagged = _git(["rev-list", "-n", "1", TAG])
        except subprocess.CalledProcessError:
            raise ValueError(f"tag {TAG} not found")
        if tagged != head:
            raise ValueError(f"tag {TAG} -> {tagged} != HEAD {head}")
    return head


def _verify_declared_hashes(ci, decl):
    m = ci.manifest
    checks = {"full_dump_sha256": m.full_dump_sha256, "source_fit_sha256": m.source_fit_sha256,
              "deployment_input_sha256": m.deployment_input_sha256, "label_sha256": m.label_sha256,
              "subject_list_sha256": m.subject_list_sha256}
    for k, v in checks.items():
        if k in decl and decl[k] != v:
            raise ValueError(f"{ci.dataset_id}: declared {k} != computed ({decl[k]} != {v})")


def run(input_manifest_path, output, *, protocol_commit=None, require_tag=True, verify_git=True):
    with open(input_manifest_path) as f:
        spec = json.load(f)
    import os
    if os.path.exists(output):
        raise FileExistsError(f"output dir already exists: {output}")          # BEFORE anything else
    cohorts = spec["cohorts"]
    for c in cohorts:
        if not os.path.exists(c["path"]):
            raise FileNotFoundError(f"missing cohort dump: {c['path']}")
        if "full_dump_sha256" in c and file_sha256(c["path"]) != c["full_dump_sha256"]:
            raise ValueError(f"{c['dataset_id']}: file SHA-256 != declared full_dump_sha256")
    if verify_git:
        verify_protocol(protocol_commit or spec.get("protocol_commit"), require_tag=require_tag)
    inputs = []
    for c in cohorts:
        ci = build_cohort_input(c["path"], disease=c["disease"], dataset_id=c["dataset_id"])
        _verify_declared_hashes(ci, c)
        inputs.append(ci)
    cmd = "python -m acar.v3.run_dev_binding --input-manifest %s --output %s" % (input_manifest_path, output)
    return freeze_dev_run(inputs, output, protocol_commit=(protocol_commit or spec.get("protocol_commit")),
                          binding_command=cmd)


def main(argv=None):
    p = argparse.ArgumentParser(prog="acar.v3.run_dev_binding")
    p.add_argument("--input-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--protocol-commit", default=None)
    args = p.parse_args(argv)
    res, manifest = run(args.input_manifest, args.output, protocol_commit=args.protocol_commit)
    print(f"verdict={manifest['verdict']} selected={manifest.get('selected')} output={args.output}")
    return 0 if manifest["verdict"] in ("SELECT", "DEV_STOP") else 1


if __name__ == "__main__":                                                     # pragma: no cover
    sys.exit(main())
