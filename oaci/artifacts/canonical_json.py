"""Deterministic canonical JSON for scientific values.

UTF-8, sort_keys, ensure_ascii=False, compact separators, allow_nan=False, exactly one trailing
newline. Non-finite floats are tagged ({"$float":"nan"|"+inf"|"-inf"}) and ndarrays are tagged
({"$ndarray":{dtype,shape,data}}). ``$float`` / ``$ndarray`` are reserved single-key objects: an
ordinary input dict that looks like one is rejected. Object-dtype arrays are a hard failure. No
absolute path / timestamp / hostname ever enters this encoding.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

_RESERVED = ("$float", "$ndarray")


def _float_tag(x: float):
    if math.isnan(x):
        return {"$float": "nan"}
    if x == math.inf:
        return {"$float": "+inf"}
    if x == -math.inf:
        return {"$float": "-inf"}
    return None


def _encode(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        t = _float_tag(v)
        return t if t is not None else v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return _encode(float(v))
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            raise TypeError("canonical_json: object-dtype ndarray is not encodable")
        a = np.ascontiguousarray(v)
        if a.dtype.kind in ("U", "S"):
            data = [str(x) for x in a.ravel(order="C").tolist()]
        elif a.dtype.kind == "b":
            data = [bool(x) for x in a.ravel(order="C").tolist()]
        elif a.dtype.kind in ("i", "u"):
            data = [int(x) for x in a.ravel(order="C").tolist()]
        elif a.dtype.kind == "f":
            data = [_encode(float(x)) for x in a.ravel(order="C").tolist()]
        else:
            raise TypeError(f"canonical_json: unsupported ndarray dtype {a.dtype!r}")
        return {"$ndarray": {"dtype": str(a.dtype), "shape": list(a.shape), "data": data}}
    if isinstance(v, Mapping):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical_json: mapping key must be str, got {type(k)!r}")
            if k in _RESERVED:
                raise ValueError(f"canonical_json: reserved key {k!r} in ordinary mapping")
            out[k] = _encode(x)
        return out
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    raise TypeError(f"canonical_json: unsupported type {type(v)!r}")


def _decode(v):
    if isinstance(v, dict):
        if set(v.keys()) == {"$float"}:
            return {"nan": math.nan, "+inf": math.inf, "-inf": -math.inf}[v["$float"]]
        if set(v.keys()) == {"$ndarray"}:
            spec = v["$ndarray"]
            dt = np.dtype(spec["dtype"])
            flat = [(_decode(x) if isinstance(x, dict) else x) for x in spec["data"]]
            arr = np.array(flat, dtype=dt) if flat else np.zeros(0, dtype=dt)
            return arr.reshape(spec["shape"])
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def canonical_json_bytes(value) -> bytes:
    text = json.dumps(_encode(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return (text + "\n").encode("utf-8")


def decode_canonical_json(data) -> object:
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    return _decode(json.loads(data))


def canonical_json_hash(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
