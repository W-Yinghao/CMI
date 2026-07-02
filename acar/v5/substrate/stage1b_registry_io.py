"""ACAR V5 Stage-1B registry PERSISTENCE (pure/stdlib). The substrate registry must survive the run as an auditable, hash-bound
FILE artifact — Stage-2 admits a run by (registry.json + FINALIZED.json + matching registry_sha256), never by an in-memory Python
object. Canonical export is deterministic (sorted refs + sorted keys) so the same registry always hashes the same. Round-trip import
rebuilds an equivalent registry.
"""
from __future__ import annotations
import hashlib
import json
import os
from acar.v5.substrate.registry import SubstrateRegistry

SCHEMA = "acar_v5_registry_v1"
REGISTRY_FILE = "registry.json"
MARKER_FILE = "FINALIZED.json"


class RegistryIoError(RuntimeError):
    pass


def _parse_ref(ref):
    disease = ref.split("/")[0]
    fold = int(ref.split("fold")[1].split("/")[0])
    seed = int(ref.split("seed")[1])
    return disease, fold, seed


def export_registry(registry):
    """Canonical, sorted export dict for the registry (schema + n_refs + entries{ref:{hashes,meta}})."""
    entries = {}
    for ref in sorted(registry._entries):
        e = registry._entries[ref]
        entries[ref] = {"hashes": dict(e["hashes"]), "meta": dict(e["meta"])}
    return {"schema": SCHEMA, "n_refs": len(entries), "entries": entries}


def registry_canonical_bytes(registry):
    return json.dumps(export_registry(registry), sort_keys=True, separators=(",", ":")).encode("utf-8")


def registry_sha256(registry):
    return hashlib.sha256(registry_canonical_bytes(registry)).hexdigest()


def write_registry(registry, path):
    """Write the canonical registry JSON ATOMICALLY (tmp → os.replace). Returns the sha256 of the exact bytes written."""
    data = registry_canonical_bytes(registry)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return hashlib.sha256(data).hexdigest()


def load_registry_from_bytes(raw):
    """Rebuild a SubstrateRegistry from canonical registry JSON BYTES (round-trip of export_registry). Fail-closed on malformed JSON."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise RegistryIoError(f"malformed registry.json: {e}")
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise RegistryIoError(f"registry.json schema {doc.get('schema') if isinstance(doc, dict) else '?'!r} != {SCHEMA}")
    reg = SubstrateRegistry()
    for ref in sorted(doc.get("entries", {})):
        disease, fold, seed = _parse_ref(ref)
        e = doc["entries"][ref]
        reg.register(disease, fold, seed, hashes=e["hashes"], meta=e["meta"])   # re-validates hashes/meta
    return reg


def load_registry(path):
    """Rebuild a SubstrateRegistry from a canonical registry.json file (round-trip of export_registry)."""
    with open(path, "rb") as f:
        return load_registry_from_bytes(f.read())


def admit_run(output_root, run_id):
    """DOWNSTREAM (Stage-2) admission: a run is admissible ONLY if registry.json AND FINALIZED.json are both present, the marker says
    FINALIZED with n_refs==30, and the marker's registry_sha256 matches sha256(registry.json bytes). Returns the loaded registry —
    parsed from the SAME bytes that were hash-checked (no re-read → no TOCTOU swap). Fail-closed (RegistryIoError) on everything."""
    run_root = os.path.join(output_root, run_id)
    reg_path = os.path.join(run_root, REGISTRY_FILE)
    marker_path = os.path.join(run_root, MARKER_FILE)
    if not os.path.isfile(reg_path):
        raise RegistryIoError(f"run not admissible: missing {REGISTRY_FILE}")
    if not os.path.isfile(marker_path):
        raise RegistryIoError(f"run not admissible: missing {MARKER_FILE}")
    with open(marker_path, "rb") as f:
        marker_raw = f.read()
    with open(reg_path, "rb") as f:
        reg_bytes = f.read()                                  # the ONE read of registry.json; both hash-check + parse use it
    try:
        marker = json.loads(marker_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise RegistryIoError(f"run not admissible: malformed {MARKER_FILE}: {e}")
    if not isinstance(marker, dict):
        raise RegistryIoError(f"run not admissible: {MARKER_FILE} is not a JSON object")
    if marker.get("status") != "FINALIZED":
        raise RegistryIoError("run not admissible: marker status != FINALIZED")
    try:
        n_refs = int(marker.get("n_refs", -1))
    except (ValueError, TypeError) as e:
        raise RegistryIoError(f"run not admissible: marker n_refs is not an integer: {e}")
    if n_refs != 30:
        raise RegistryIoError(f"run not admissible: marker n_refs {marker.get('n_refs')} != 30")
    if marker.get("registry_sha256") != hashlib.sha256(reg_bytes).hexdigest():
        raise RegistryIoError("run not admissible: FINALIZED.registry_sha256 != sha256(registry.json)")
    reg = load_registry_from_bytes(reg_bytes)                 # parse the SAME validated bytes (not a fresh disk read)
    if len(reg._entries) != 30:
        raise RegistryIoError(f"run not admissible: registry has {len(reg._entries)} entries != 30")
    return reg
