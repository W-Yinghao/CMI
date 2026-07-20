"""Read enveloped artifact JSON + its sidecar NPZ."""
from __future__ import annotations

import os

from .canonical_json import decode_canonical_json
from .deterministic_npz import read_verified_npz
from .schema import open_envelope


def read_doc(path):
    with open(path, "rb") as f:
        return decode_canonical_json(f.read())


def read_artifact(path, expected_kind):
    """Return (logical_hash, body, arrays). ``arrays`` is read from the sidecar NPZ when body has a
    ``npz`` metadata block."""
    doc = read_doc(path)
    logical, body = open_envelope(doc, expected_kind)
    arrays = None
    if isinstance(body, dict) and "npz" in body:
        npz_path = path[:-5] + ".npz" if path.endswith(".json") else path + ".npz"
        arrays = read_verified_npz(npz_path, body["npz"])
    return logical, body, arrays
