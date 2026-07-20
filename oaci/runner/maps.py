"""Frozen level-0 maps: class / source-domain / evaluation-domain stable-id <-> contiguous int.

A frozen dataclass that hides no mutable dict — the bidirectional mappings are stored as tuples and
exposed read-only. The source-domain integer map is EXACTLY ``0..|D0|-1`` (global-LPC indexes
``cell_mass[d, y]`` by it). Class order comes verbatim from the manifest.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .keys import feed_int64, feed_string


def _check_bijection(ids, items, name):
    ids = tuple(str(i) for i in ids)
    if len(ids) == 0:
        raise ValueError(f"{name}: empty id list")
    if any(i == "" for i in ids):
        raise ValueError(f"{name}: empty id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate ids {ids}")
    d = dict(items)
    if set(d) != set(ids):
        raise ValueError(f"{name}: index map keys != ids")
    if sorted(d.values()) != list(range(len(ids))):
        raise ValueError(f"{name}: index map is not a bijection onto 0..{len(ids) - 1}")
    return ids


@dataclass(frozen=True)
class FrozenMaps:
    class_names: tuple
    source_domain_ids: tuple                      # fixed D0
    evaluation_domain_ids: tuple
    class_to_index_items: tuple
    source_domain_to_index_items: tuple
    evaluation_domain_to_index_items: tuple
    maps_hash: str = ""

    @property
    def class_to_index(self) -> dict:
        return dict(self.class_to_index_items)

    @property
    def source_domain_to_index(self) -> dict:
        return dict(self.source_domain_to_index_items)

    @property
    def evaluation_domain_to_index(self) -> dict:
        return dict(self.evaluation_domain_to_index_items)


def _maps_hash(maps: FrozenMaps) -> str:
    h = hashlib.sha256()
    for ids, items, tag in ((maps.class_names, maps.class_to_index_items, "C"),
                            (maps.source_domain_ids, maps.source_domain_to_index_items, "S"),
                            (maps.evaluation_domain_ids, maps.evaluation_domain_to_index_items, "E")):
        h.update(tag.encode())
        for i in ids:
            feed_string(h, i)
        for k, v in items:
            feed_string(h, k); feed_int64(h, v)
    return h.hexdigest()


def build_frozen_maps(class_names, source_domain_ids, evaluation_domain_ids) -> FrozenMaps:
    """``class_names`` are kept in manifest order; source domains are sorted (D0) -> contiguous int;
    evaluation domains (over the audit + target roles) sorted -> contiguous int."""
    cn = tuple(str(c) for c in class_names)
    sd = tuple(sorted(str(d) for d in source_domain_ids))
    ed = tuple(sorted(str(d) for d in evaluation_domain_ids))
    cmap = tuple((c, i) for i, c in enumerate(cn))
    smap = tuple((d, i) for i, d in enumerate(sd))
    emap = tuple((d, i) for i, d in enumerate(ed))
    _check_bijection(cn, cmap, "class")
    _check_bijection(sd, smap, "source_domain")
    _check_bijection(ed, emap, "evaluation_domain")
    m = FrozenMaps(cn, sd, ed, cmap, smap, emap)
    return FrozenMaps(cn, sd, ed, cmap, smap, emap, _maps_hash(m))
