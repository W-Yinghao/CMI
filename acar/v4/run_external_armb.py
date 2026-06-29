"""ACAR v4 — the UNIQUE external Arm-B CLI (binding once tagged `acar-v4-protocol`).

Runs the frozen candidate (ACAR_FROZEN_v4.md) on the EXACT audited §4 held-out strata exactly once, with a v3-style
STDLIB-FIRST fail-closed preflight, and writes results/acar_v4_external_001/. No external read happens until every
preflight check passes AND the env lock verifies. Committed/frozen together with the protocol under `acar-v4-protocol`
(binding execution path). Synthetic guards exercise the manifest validation + preflight fail-closed branches without any
external data; the heavy path (frozen-source load, dump read, evaluate) is gated to the authorized run in eeg2025.

Leakage firewall (ACAR_FROZEN_v4.md §5): the held-out diagnosis labels never refit f_0 — each stratum supplies a
DEV-frozen source artifact (verified by sha) that external_adapter.build_stratum_from_dump applies WITHOUT fitting; held-
out labels enter only ΔR for CAL λ* + EVAL scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys

PROTOCOL_TAG = "acar-v4-protocol"
# the EXACT audited admissible held-out strata (ACAR_FROZEN_v4.md §4); a binding run requires ALL of them (not a subset)
ADMISSIBLE_STRATA = {("zenodo14808296", "SCZ"), ("ds007526", "PD")}
_REJECTED = {"14178398", "aszed", "ds007020"}                 # provisional / excluded
_DEV_COHORTS = {"ds002778", "ds003490", "ds004584", "ds003944", "ds003947", "ds004000", "ds004367"}
# full per-stratum provenance hashes (all 64-char lowercase sha-256)
_HASH_FIELDS = ("full_dump_sha256", "deployment_input_sha256", "label_sha256", "subject_list_sha256",
                "diagnosis_mapping_sha256", "resting_selection_sha256", "raw_pipeline_sha256", "source_state_sha256")


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in "0123456789abcdef" for c in s)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate_external_manifest(spec):
    """Fail-closed schema check. Requires protocol_commit (40-hex), the EXACT admissible stratum set, and full
    per-stratum provenance (dump path, all field hashes, source-state ref/sha + dev artifact path, expected counts).
    Returns the strata list."""
    if not isinstance(spec, dict):
        raise ValueError("manifest must be a JSON object")
    if not _is_hex(spec.get("protocol_commit", ""), 40):
        raise ValueError("protocol_commit must be a full 40-char lowercase git SHA-1")
    strata = spec.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("strata must be a non-empty list")
    pairs = []
    for st in strata:
        if not isinstance(st, dict):
            raise ValueError("each stratum must be an object")
        site, disease = st.get("site"), st.get("disease")
        if disease not in ("PD", "SCZ"):
            raise ValueError(f"disease must be PD or SCZ, got {disease!r}")
        if site in _DEV_COHORTS or str(site).lower() in _REJECTED:
            raise ValueError(f"site {site!r} is a DEV/rejected/provisional cohort — not admissible")
        if (site, disease) not in ADMISSIBLE_STRATA:
            raise ValueError(f"({site!r},{disease!r}) is not an audited admissible stratum")
        pairs.append((site, disease))
        if not isinstance(st.get("dump_path"), str) or not st["dump_path"]:
            raise ValueError(f"{site}: dump_path must be a non-empty string")
        if not isinstance(st.get("dev_source_artifact_path"), str) or not st["dev_source_artifact_path"]:
            raise ValueError(f"{site}: dev_source_artifact_path must be a non-empty string (DEV-frozen source)")
        if not isinstance(st.get("dataset_version"), str) or not st["dataset_version"]:
            raise ValueError(f"{site}: dataset_version must be a non-empty string")
        if not _is_hex(st.get("source_state_ref", ""), 64):
            raise ValueError(f"{site}: source_state_ref must be 64-char sha-256")
        if not _is_hex(st.get("provenance_sidecar_sha256", ""), 64):
            raise ValueError(f"{site}: provenance_sidecar_sha256 must be 64-char sha-256 (frozen-prep sidecar)")
        for hf in _HASH_FIELDS:
            if not _is_hex(st.get(hf, ""), 64):
                raise ValueError(f"{site}: {hf} must be 64-char lowercase sha-256")
        for cf in ("expected_n_subjects", "expected_embedding_dim"):
            if not _is_int(st.get(cf)) or st[cf] <= 0:
                raise ValueError(f"{site}: {cf} must be a positive int")
    if set(pairs) != ADMISSIBLE_STRATA:
        raise ValueError(f"strata must be EXACTLY {sorted(ADMISSIBLE_STRATA)} (single-site-per-disease; both required), "
                         f"got {sorted(set(pairs))}")
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate stratum")
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


def _env_lock_verify():
    """Apply the deterministic runtime + verify the env lock (fail-closed). Seam for ordering guards."""
    from acar.v3.envlock import apply_runtime, verify_env_lock
    apply_runtime()
    return verify_env_lock()


def _verify_provenance_sidecar(st):
    """Bind ALL declared hashes to the frozen-prep sidecar (PROV-1 / closure point B). The 3 prep-only hashes
    (raw_pipeline/diagnosis_mapping/resting_selection) are NOT recomputable from the .npz at run time, so instead of
    trusting hand-filled manifest values we require a `<dump>.provenance.json` emitted by prepare_dump whose sha equals the
    manifest's `provenance_sidecar_sha256`, and every manifest hash field + source_state_ref must equal the sidecar's
    value. (The 5 label-bearing hashes are ALSO independently recomputed downstream.)"""
    sidecar_path = st["dump_path"] + ".provenance.json"
    got = _sha256_file(sidecar_path)
    if got != st["provenance_sidecar_sha256"]:
        raise ValueError(f"{st['site']}: provenance_sidecar_sha256 mismatch ({got} != {st['provenance_sidecar_sha256']})")
    with open(sidecar_path) as f:
        sc = json.load(f)
    for field in (*_HASH_FIELDS, "source_state_ref"):
        if sc.get(field) != st.get(field):
            raise ValueError(f"{st['site']}: manifest {field} != frozen-prep sidecar ({st.get(field)!r} vs {sc.get(field)!r})")


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None                # NaN/Inf → None (fail-closed unevaluable), never raw
    if o is None or isinstance(o, (int, str)):
        return o
    raise TypeError(f"non-JSON-safe value of type {type(o)} in manifest")


def run(input_manifest_path, output):
    """Full external Arm-B run (gated). STDLIB-FIRST preflight (no heavy import until all checks pass), then evaluate the
    frozen candidate on the admissible strata and write `output`. NO external read before preflight + env-lock verify.
    `output` is claimed up front with a race-free `os.mkdir` (first-writer-wins; also surfaces an unwritable parent) BEFORE
    any external dump byte is read; the run is COMPLETE iff `output/RESULT.json` exists (written last); any abort removes
    the whole claimed `output` dir (no partial/half-written result)."""
    with open(input_manifest_path, "rb") as f:
        raw = f.read()
    input_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    spec = json.loads(raw.decode())
    if os.path.exists(output):
        raise FileExistsError(f"output dir already exists (no overwrite): {output}")
    strata = validate_external_manifest(spec)                 # schema + exact admissible set + full provenance
    root = _repo_root()
    verify_protocol(root, spec["protocol_commit"])
    verify_clean_worktree(root)
    os.mkdir(output)                                          # ATOMIC race-free claim, BEFORE any external dump read
    try:
        for st in strata:                                    # provenance preflight: hash dumps + verify sidecar FIRST
            got = _sha256_file(st["dump_path"])
            if got != st["full_dump_sha256"]:
                raise ValueError(f"{st['site']}: full_dump_sha256 mismatch ({got} != {st['full_dump_sha256']})")
            _verify_provenance_sidecar(st)                    # bind all 8 declared hashes to the frozen-prep sidecar
        # --- heavy section (ONLY past a fully-passing stdlib preflight) ---
        env_lock_sha = _env_lock_verify()                    # apply runtime + verify env lock (fail-closed) BEFORE model
        from acar.v4 import external_adapter as EA
        results = []
        for st in strata:
            art = EA.load_frozen_source_state_artifact_from_path(st["dev_source_artifact_path"], disease=st["disease"])
            if art.source_state_sha256 != st["source_state_sha256"] or art.source_state_ref != st["source_state_ref"]:
                raise ValueError(f"{st['site']}: frozen source artifact sha/ref mismatch vs manifest")
            stratum = EA.build_stratum_from_dump(
                st["site"], st["disease"], st["dump_path"], art,
                verify_hashes={k: st[k] for k in ("deployment_input_sha256", "label_sha256", "subject_list_sha256")})
            n_subj = len(stratum["subjects"])                 # PROV-3: bind the declared counts to the built dump
            if n_subj != st["expected_n_subjects"]:
                raise ValueError(f"{st['site']}: expected_n_subjects {st['expected_n_subjects']} != built {n_subj}")
            art_dim = getattr(art, "embedding_dim", st["expected_embedding_dim"])
            if art_dim != st["expected_embedding_dim"]:
                raise ValueError(f"{st['site']}: expected_embedding_dim {st['expected_embedding_dim']} != artifact {art_dim}")
            results.append(EA.evaluate_stratum(stratum))
        ext = EA.external_taxonomy(results)
        body = {
            "boundary": "BINDING external Arm B (acar-v4-protocol); single confirmatory pass; lockbox-equivalent",
            "command": shlex.join([sys.executable, "-m", "acar.v4.run_external_armb",
                                   "--input-manifest", input_manifest_path, "--output", output]),
            "protocol_commit": spec["protocol_commit"], "protocol_tag": PROTOCOL_TAG,
            "env_lock_sha256": env_lock_sha, "input_manifest_sha256": input_manifest_sha256,
            "run_status": ext.run_status, "verdict": ext.verdict, "per_disease": ext.per_disease,
            "strata": [r.__dict__ for r in ext.strata],
            "input_strata": [{k: s.get(k) for k in ("site", "disease", "dataset_version", "source_state_ref",
                                                    "provenance_sidecar_sha256", *_HASH_FIELDS)} for s in strata],
        }
        safe = _json_safe(body)
        safe["manifest_sha256"] = hashlib.sha256(json.dumps(safe, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with open(os.path.join(output, "manifest.json"), "w") as f:
            json.dump(safe, f, sort_keys=True, allow_nan=False, indent=2)
        with open(os.path.join(output, "RESULT.json"), "w") as f:    # written LAST = completion sentinel
            json.dump({"run_status": ext.run_status, "verdict": ext.verdict,
                       "per_disease": {d: v["confirmed"] for d, v in ext.per_disease.items()},
                       "manifest_sha256": safe["manifest_sha256"]}, f, sort_keys=True, allow_nan=False, indent=2)
        return ext
    except BaseException:
        import shutil
        shutil.rmtree(output, ignore_errors=True)            # abort: remove the claimed dir (no partial publish)
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
