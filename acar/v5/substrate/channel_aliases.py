"""ACAR V5 Stage-1B channel-alias layer (pure/stdlib). The trained substrate's OUTPUT montage stays the old-10-20 canonical order
(preprocessing_config.CHANNELS_19); this layer maps a recording's raw channel names to those canonical channels BEFORE the pick, so
cohorts labelled with modern 10-10 temporal names (T7/T8/P7/P8) or upper-case Fp names (FP1/FP2) select the SAME historical
electrodes. Single source of truth: the alias pairs live in the pinned preprocessing_config (so they are part of
preprocessing_config_sha256).

Rules (fail-closed):
  * input names are stripped and case-normalized (casefold);
  * an exact canonical name (any case) maps to itself (Fp1/FP1/fp1 → Fp1; T3 → T3);
  * a pinned alias maps a modern name to its canonical electrode (T7 → T3, T8 → T4, P7 → T5, P8 → T6);
  * a name that is neither canonical nor a known alias is a NON-canonical extra → dropped;
  * if two raw channels map to the SAME canonical channel → FAIL (duplicate logical channel);
  * if any of the 19 canonical channels is missing after aliasing → FAIL.
"""
from __future__ import annotations
from acar.v5.substrate import preprocessing_config as PC


class ChannelAliasError(RuntimeError):
    pass


_CANONICAL = tuple(PC.CHANNELS_19)
_CANON_BY_CASEFOLD = {c.casefold(): c for c in _CANONICAL}
# pinned modern→canonical aliases (from the config); casefolded keys, canonical (case-correct) values
_ALIAS_BY_CASEFOLD = {str(k).strip().casefold(): v for k, v in PC.PREPROCESSING_CONFIG["input_channel_aliases"].items()}
# sanity: every alias TARGET must be a canonical channel, and an alias must NOT shadow an existing canonical name
assert set(_ALIAS_BY_CASEFOLD.values()) <= set(_CANONICAL), "alias targets must be canonical channels"
assert not (set(_ALIAS_BY_CASEFOLD) & set(_CANON_BY_CASEFOLD)), "an alias key must not also be a canonical name"


def normalize_channel(name):
    """Raw channel name → canonical name, or None if it is a non-canonical extra (to be dropped). Case-insensitive + stripped."""
    key = str(name).strip().casefold()
    if key in _CANON_BY_CASEFOLD:
        return _CANON_BY_CASEFOLD[key]
    if key in _ALIAS_BY_CASEFOLD:
        return _ALIAS_BY_CASEFOLD[key]
    return None


def resolve_canonical_sources(ch_names):
    """Map a recording's channel names to {canonical: source_name}. Fail-closed on a duplicate logical channel or any missing
    canonical channel. Extra (non-canonical) channels are simply omitted (dropped by the later pick)."""
    canon_to_src = {}
    for name in ch_names:
        canonical = normalize_channel(name)
        if canonical is None:
            continue                                          # non-canonical extra → dropped
        if canonical in canon_to_src:
            raise ChannelAliasError(f"two raw channels map to the same logical channel {canonical}: "
                                    f"{canon_to_src[canonical]!r} and {name!r}")
        canon_to_src[canonical] = name
    missing = [c for c in _CANONICAL if c not in canon_to_src]
    if missing:
        raise ChannelAliasError(f"missing canonical channels after aliasing: {missing}")
    return canon_to_src


def ordered_source_names(ch_names):
    """The recording's source channel names in CANONICAL order (length 19) — i.e. [source for T-canonical in CHANNELS_19]."""
    canon_to_src = resolve_canonical_sources(ch_names)
    return [canon_to_src[c] for c in _CANONICAL]


def logical_duplicates(names):
    """Canonical channels that MORE THAN ONE raw channel maps to (after aliasing)."""
    seen, dups = set(), set()
    for n in names:
        c = normalize_channel(n)
        if c is None:
            continue
        if c in seen:
            dups.add(c)
        seen.add(c)
    return sorted(dups)


def adjudicate_channel_source(channels_tsv_names, raw_header_names):
    """The RAW HEADER is decisive for DSP compatibility (the DSP consumes the raw header channel names). A duplicate logical channel
    in the RAW HEADER is a signal-level ambiguity → FAIL. A duplicate only in channels.tsv (with a clean raw header) is a metadata
    inconsistency → WARN (non-fatal). No 'keep-first' / no silent de-dup. Returns a verdict dict."""
    tsv_dups = logical_duplicates(channels_tsv_names) if channels_tsv_names is not None else []
    if raw_header_names is None:
        if tsv_dups:
            return {"verdict": "FAIL", "reason": f"channels.tsv duplicate logical {tsv_dups} and no raw header to adjudicate",
                    "raw_duplicates": None, "tsv_duplicates": tsv_dups}
        return {"verdict": "PASS", "reason": "no raw header; channels.tsv clean", "raw_duplicates": None, "tsv_duplicates": []}
    raw_dups = logical_duplicates(raw_header_names)
    if raw_dups:
        return {"verdict": "FAIL", "reason": f"raw header duplicate logical channel(s) {raw_dups} (signal-level ambiguity)",
                "raw_duplicates": raw_dups, "tsv_duplicates": tsv_dups}
    if tsv_dups:
        return {"verdict": "WARN_TSV_DUPLICATE",
                "reason": f"channels.tsv duplicate logical {tsv_dups} but raw header is clean (metadata inconsistency)",
                "raw_duplicates": [], "tsv_duplicates": tsv_dups}
    return {"verdict": "PASS", "reason": "raw header + channels.tsv both clean", "raw_duplicates": [], "tsv_duplicates": []}
