"""Guard (Stage-1B12): ds004367 F7 is completed by INTERPOLATION only when the raw header carries the known re-referenced/split
duplicate-variant pattern (F7-0 AND F7-1) and no canonical F7. Without the variant pattern → fail-closed. SYNTHETIC mne RawArray."""
from __future__ import annotations
from acar.v5.substrate import montage_completion as MC
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.tests._util import ok, expect_raises, make_mne_raw, modern_channel_names


def _f7_variant_raw(extra=("AF3", "GSR1")):
    present = [c for c in modern_channel_names() if c != "F7"] + ["F7-0", "F7-1"] + list(extra)
    return make_mne_raw(present, 2048, 256.0)


def test_variant_pattern_completes_f7_by_interpolation():
    raw2, prov = MC.complete_missing_channels(_f7_variant_raw(), "SCZ", "ds004367")
    assert prov["interpolated"] == ["F7"] and prov["n_interpolated"] == 1 and prov["donor_count"] >= 8
    # the non-canonical variant channels are NOT used as donors and are dropped before interpolation
    assert "F7-0" not in raw2.ch_names and "F7-1" not in raw2.ch_names
    ok("ds004367 with F7-0/F7-1 and no canonical F7 → F7 interpolated (variants dropped, not donors)")


def test_end_to_end_canonical_output_with_interpolated_f7():
    sw = RMR.raw_to_windows(_f7_variant_raw(), "SCZ", "ds004367", "sub-S01")
    assert tuple(sw.channels) == PC.CHANNELS_19
    assert "interpolated=['F7']" in sw.provenance and sw.montage_completion["interpolated"] == ["F7"]
    ok("raw_to_windows emits the canonical 19 with F7 interpolated + recorded in provenance/montage_completion")


def test_missing_f7_without_variants_fails_closed():
    present = [c for c in modern_channel_names() if c != "F7"] + ["AF3", "FC1"]   # F7 missing, NO F7-0/F7-1
    expect_raises(MC.MontageCompletionError,
                  lambda: MC.complete_missing_channels(make_mne_raw(present, 2048, 256.0), "SCZ", "ds004367"),
                  "ds004367 missing F7 without the variant pattern must fail closed")
    ok("ds004367 missing F7 WITHOUT the F7-0/F7-1 variant pattern → MontageCompletionError (whitelist alone is insufficient)")


def main():
    print("ACAR v5 Stage-1B12 guard: ds004367 F7 completion only with the F7-0/F7-1 variant pattern")
    test_variant_pattern_completes_f7_by_interpolation()
    test_end_to_end_canonical_output_with_interpolated_f7()
    test_missing_f7_without_variants_fails_closed()
    print("ALL V5 STAGE1B12-DS004367-F7-VARIANT GUARDS PASS")


if __name__ == "__main__":
    main()
