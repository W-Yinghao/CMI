"""ACAR v4 — the UNIQUE external Arm-B CLI (binding once tagged `acar-v4-protocol`).

Runs the frozen candidate (ACAR_FROZEN_v4.md) on the audited §4 held-out strata exactly once, with a v3-style
fail-closed preflight, and writes results/acar_v4_external_001/. This module performs NO external read until ALL
preflight checks pass; it is committed and frozen TOGETHER with the protocol under the `acar-v4-protocol` tag (binding
execution path, not just docs). The synthetic guards exercise the manifest validation + preflight fail-closed branches
without any external data.

ADMISSIBLE strata are exactly the audited §4 list (notes/ACAR_V4_LOCKBOX_AUDIT.md). ASZED 14178398 (provisional) and
ds007020 (excluded) and the seven DEV cohorts are rejected. Single-site-per-disease is allowed (reported as such).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from typing import Tuple

from acar.v4 import external_adapter as EA

PROTOCOL_TAG = "acar-v4-protocol"
# audited admissible held-out strata (site, disease) — the ACAR_FROZEN_v4.md §4 list
ADMISSIBLE_STRATA = {("zenodo14808296", "SCZ"), ("ds007526", "PD")}
ADMISSIBLE_SITES = {s for s, _ in ADMISSIBLE_STRATA}
# explicitly rejected (provisional / excluded / DEV)
_REJECTED = {"14178398", "aszed", "ds007020"}
_DEV_COHORTS = {"ds002778", "ds003490", "ds004584", "ds003944", "ds003947", "ds004000", "ds004367"}


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in "0123456789abcdef" for c in s)


def validate_external_manifest(spec):
    """Fail-closed schema check for the external input manifest. Returns the normalized list of strata."""
    if not isinstance(spec, dict):
        raise ValueError("manifest must be a JSON object")
    if not _is_hex(spec.get("protocol_commit", ""), 40):
        raise ValueError("protocol_commit must be a full 40-char lowercase git SHA-1")
    strata = spec.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("strata must be a non-empty list")
    seen = set()
    for st in strata:
        if not isinstance(st, dict):
            raise ValueError("each stratum must be an object")
        site, disease = st.get("site"), st.get("disease")
        if disease not in ("PD", "SCZ"):
            raise ValueError(f"disease must be PD or SCZ, got {disease!r}")
        if site in _DEV_COHORTS or str(site).lower() in _REJECTED:
            raise ValueError(f"site {site!r} is a DEV/rejected/provisional cohort — not admissible")
        if (site, disease) not in ADMISSIBLE_STRATA:
            raise ValueError(f"({site!r},{disease!r}) is not an audited admissible stratum {sorted(ADMISSIBLE_STRATA)}")
        if (site, disease) in seen:
            raise ValueError(f"duplicate stratum ({site!r},{disease!r})")
        seen.add((site, disease))
        if not isinstance(st.get("dump_path"), str) or not st["dump_path"]:
            raise ValueError(f"{site}: dump_path must be a non-empty string")
        if not _is_hex(st.get("dump_sha256", ""), 64):
            raise ValueError(f"{site}: dump_sha256 must be a 64-char lowercase SHA-256")
    return strata


def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def verify_protocol(root, commit):
    """Fail-closed: HEAD == protocol_commit AND tag acar-v4-protocol → HEAD (unconditional, no bypass)."""
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != commit:
        raise ValueError(f"HEAD != protocol_commit ({head.stdout.strip()!r} vs {commit!r})")
    tag = _git(root, "rev-list", "-n", "1", PROTOCOL_TAG)
    if tag.returncode != 0 or tag.stdout.strip() != commit:
        raise ValueError(f"tag {PROTOCOL_TAG} does not resolve to HEAD (got {tag.stdout.strip()!r})")


def verify_clean_worktree(root):
    st = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if st.returncode != 0 or st.stdout.strip() != "":
        raise ValueError(f"worktree not clean: [{st.stdout.strip()}]")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(input_manifest_path, output):
    """Full external Arm-B run (gated): preflight fail-closed, then evaluate the frozen candidate on the admissible
    strata and write `output`. NO external read happens before all preflight checks pass."""
    with open(input_manifest_path) as f:
        spec = json.load(f)
    if os.path.exists(output):
        raise FileExistsError(f"output dir already exists (no overwrite): {output}")
    strata = validate_external_manifest(spec)
    root = _repo_root()
    verify_protocol(root, spec["protocol_commit"])
    verify_clean_worktree(root)
    for st in strata:                                          # hash the declared dumps BEFORE any modeling read
        got = _sha256_file(st["dump_path"])
        if got != st["dump_sha256"]:
            raise ValueError(f"{st['site']}: dump_sha256 mismatch ({got} != {st['dump_sha256']})")
    from acar.v3.envlock import apply_runtime                  # determinism (torch); gated past preflight
    apply_runtime()
    os.mkdir(output)                                           # atomic claim (race-free no-overwrite)
    try:
        results = []
        for st in strata:
            stratum = EA.build_stratum_from_dump(st["site"], st["disease"], st["dump_path"])
            results.append(EA.evaluate_stratum(stratum))
        ext = EA.external_taxonomy(results)
        manifest = {
            "boundary": "BINDING external Arm B (acar-v4-protocol); single confirmatory pass; lockbox-equivalent",
            "protocol_commit": spec["protocol_commit"], "protocol_tag": PROTOCOL_TAG,
            "run_status": ext.run_status, "verdict": ext.verdict, "per_disease": ext.per_disease,
            "strata": [r.__dict__ for r in ext.strata],
            "input_strata": [{"site": s["site"], "disease": s["disease"], "dump_sha256": s["dump_sha256"]} for s in strata],
        }
        with open(os.path.join(output, "manifest.json"), "w") as f:
            json.dump(manifest, f, sort_keys=True, allow_nan=False, indent=2)
        with open(os.path.join(output, "RESULT.json"), "w") as f:    # written last = completion sentinel
            json.dump({"run_status": ext.run_status, "verdict": ext.verdict,
                       "per_disease": {d: v["confirmed"] for d, v in ext.per_disease.items()}},
                      f, sort_keys=True, allow_nan=False, indent=2)
        return ext
    except BaseException:
        import shutil
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 external Arm-B (BINDING under acar-v4-protocol)")
    ap.add_argument("--input-manifest", required=True)
    ap.add_argument("--output", required=True, help="must not exist, e.g. results/acar_v4_external_001")
    args = ap.parse_args(argv)
    ext = run(args.input_manifest, args.output)
    print(f"V4_EXTERNAL run_status={ext.run_status} verdict={ext.verdict} "
          f"PD={ext.per_disease['PD']['confirmed']} SCZ={ext.per_disease['SCZ']['confirmed']}")
    return ext


if __name__ == "__main__":
    main()
