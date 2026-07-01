"""ACAR V5 Stage-0 substrate registry (SPLITS §1–§2): the substrate is FROZEN-FIRST and every DEV-selection embedding must carry
a registered substrate hash set — **no hash ⇒ inadmissible for selection**. Pure (stdlib); NO training / NO real read.

An entry is keyed by (disease, fold, seed) and pins the full REGISTRY_HASH_FIELDS + REGISTRY_META_FIELDS. The registry is the
guard behind `test_substrate_hash_required`: `admit_embedding` refuses any embedding whose declared substrate_ref is unregistered
or whose hash set is incomplete/malformed.
"""
from __future__ import annotations
from acar.v5 import protocol as P

_HEX = "0123456789abcdef"


class SubstrateHashMissingError(RuntimeError):
    """Raised when a DEV-selection embedding lacks a registered, complete substrate hash set."""


def _is_hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(ch in _HEX for ch in s)


def substrate_ref(disease, fold, seed):
    """Canonical key for a fold-contained substrate (SPLITS §5/§6: per disease × outer fold × seed)."""
    if disease not in P.DEV_COHORTS:
        raise ValueError(f"unknown disease {disease!r}")
    if not (isinstance(fold, int) and 0 <= fold < P.OUTER_K):
        raise ValueError(f"fold must be in [0,{P.OUTER_K})")
    if seed not in P.S1_SEEDS:
        raise ValueError(f"seed {seed!r} not in the pinned set {P.S1_SEEDS}")
    return f"{disease}/fold{fold}/seed{seed}"


class SubstrateRegistry:
    """Fold-contained substrate registry. Stage-1 registers each frozen substrate BEFORE its embeddings are read; Stage-2
    selection may only consume embeddings whose substrate_ref is registered here."""

    def __init__(self):
        self._entries = {}                                    # ref -> {hashes..., meta...}

    def register(self, disease, fold, seed, *, hashes, meta):
        ref = substrate_ref(disease, fold, seed)
        if ref in self._entries:
            raise ValueError(f"substrate {ref} already registered (no silent overwrite)")
        missing = [f for f in P.REGISTRY_HASH_FIELDS if f not in hashes]
        if missing:
            raise SubstrateHashMissingError(f"{ref}: missing hash fields {missing}")
        bad = [f for f in P.REGISTRY_HASH_FIELDS if not _is_hex64(hashes[f])]
        if bad:
            raise SubstrateHashMissingError(f"{ref}: hash fields not 64-hex {bad}")
        meta_missing = [f for f in P.REGISTRY_META_FIELDS if f not in meta or meta[f] in (None, "")]
        if meta_missing:
            raise SubstrateHashMissingError(f"{ref}: missing/empty meta fields {meta_missing}")
        if int(meta["random_seed"]) != int(seed):
            raise ValueError(f"{ref}: meta random_seed {meta['random_seed']} != key seed {seed}")
        self._entries[ref] = {"hashes": dict(hashes), "meta": dict(meta)}
        return ref

    def is_registered(self, ref):
        return ref in self._entries

    def _rollback(self, refs):
        """Remove the given refs (used by populate_registry to guarantee all-or-none on a mid-loop register failure)."""
        for r in refs:
            self._entries.pop(r, None)

    def admit_embedding(self, embedding):
        """Fail-closed admission for a Stage-2/4/5 embedding. `embedding` MUST be a mapping carrying (1) a `substrate_ref`
        registered here AND (2) a `hashes` set that EXACTLY equals the registered substrate's hash set. Missing either ⇒
        inadmissible (Step 3b: embedded hashes are MANDATORY, not optional — a feature dump cannot merely claim a ref without
        carrying the hash set that proves it came from that substrate; this is the V4 artifact/substrate-mismatch class)."""
        if not isinstance(embedding, dict) or "substrate_ref" not in embedding:
            raise SubstrateHashMissingError("embedding has no substrate_ref — inadmissible for selection")
        ref = embedding["substrate_ref"]
        if not self.is_registered(ref):
            raise SubstrateHashMissingError(f"embedding substrate_ref {ref!r} is not registered — inadmissible for selection")
        emb_h = embedding.get("hashes")
        if emb_h is None:
            raise SubstrateHashMissingError(f"embedding {ref}: missing embedded substrate hash set — inadmissible (no hash ⇒ inadmissible)")
        if emb_h != self._entries[ref]["hashes"]:
            raise SubstrateHashMissingError(f"embedding hashes disagree with the registered substrate {ref}")
        return ref
