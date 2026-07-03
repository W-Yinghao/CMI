"""Guard (Stage-1B12): the F7-0/F7-1 variants are NEVER aliased / kept-first / averaged into the canonical F7. They normalize to
None (non-canonical → dropped), the alias layer reports canonical F7 as MISSING (no keep-first), and F7 is produced by interpolation
from good-position donors — not copied from a variant."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import montage_completion as MC
from acar.v5.tests._util import ok, expect_raises, make_mne_raw, modern_channel_names


def test_variants_do_not_alias_to_f7():
    assert CA.normalize_channel("F7-0") is None and CA.normalize_channel("F7-1") is None
    assert CA.normalize_channel("f7-0") is None and CA.normalize_channel(" F7-1 ") is None
    ok("F7-0 / F7-1 normalize to None — they are NOT aliased to the canonical F7")


def test_alias_layer_reports_f7_missing_no_keep_first():
    names = [c for c in modern_channel_names() if c != "F7"] + ["F7-0", "F7-1"]
    # resolve_canonical_sources must FAIL with canonical F7 missing — it does NOT keep-first a variant as F7
    err = expect_raises(CA.ChannelAliasError, lambda: CA.ordered_source_names(names), "F7 must be reported missing")
    assert err is True
    # logical_duplicates must NOT report F7 as a duplicate (the variants aren't F7 at all)
    assert "F7" not in CA.logical_duplicates(names)
    ok("the alias layer reports canonical F7 MISSING (no keep-first / no dedup of F7-0/F7-1 into F7)")


def test_f7_is_interpolated_not_copied_from_variant():
    present = [c for c in modern_channel_names() if c != "F7"] + ["F7-0", "F7-1", "AF3"]
    raw2, prov = MC.complete_missing_channels(make_mne_raw(present, 2048, 256.0), "SCZ", "ds004367")
    assert prov["interpolated"] == ["F7"]                     # F7 came from interpolation, recorded as such
    # donor_count counts good-position donors only; F7-0/F7-1 (no standard position) were dropped, not counted
    assert prov["donor_count"] == sum(1 for c in raw2.ch_names if c != "F7")
    assert "F7-0" not in raw2.ch_names and "F7-1" not in raw2.ch_names
    ok("F7 is interpolated from good-position donors; the variant channels are dropped, never used as the F7 source")


def main():
    print("ACAR v5 Stage-1B12 guard: ds004367 F7-0/F7-1 never kept-first/averaged into F7")
    test_variants_do_not_alias_to_f7()
    test_alias_layer_reports_f7_missing_no_keep_first()
    test_f7_is_interpolated_not_copied_from_variant()
    print("ALL V5 STAGE1B12-DS004367-F7-NO-KEEP-FIRST GUARDS PASS")


if __name__ == "__main__":
    main()
