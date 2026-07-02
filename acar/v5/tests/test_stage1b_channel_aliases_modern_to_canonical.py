"""Guard (Stage-1B10): the pinned channel-alias layer maps modern 10-10 temporal names to the old-10-20 canonical electrodes
(T7→T3, T8→T4, P7→T5, P8→T6) before the pick, and a recording labelled with modern names yields the canonical 19-channel
SubjectWindows in canonical order. Synthetic FakeRaw only."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import ok, make_fake_raw, modern_channel_names


def test_modern_temporal_aliases():
    assert CA.normalize_channel("T7") == "T3" and CA.normalize_channel("T8") == "T4"
    assert CA.normalize_channel("P7") == "T5" and CA.normalize_channel("P8") == "T6"
    for c in PC.CHANNELS_19:
        assert CA.normalize_channel(c) == c                   # canonical names map to themselves
    assert PC.PREPROCESSING_CONFIG["input_channel_aliases"] == {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    ok("T7/T8/P7/P8 → T3/T4/T5/T6; canonical names unchanged; alias map is pinned in preprocessing_config")


def test_ordered_sources_and_reader_output_canonical():
    modern = modern_channel_names()                           # 19 canonical electrodes in modern names, canonical order
    assert CA.ordered_source_names(modern[::-1] + ["EXG1"]) == modern   # returns source names in CANONICAL order
    sw = RMR.raw_to_windows(make_fake_raw(modern[::-1] + ["EXG1", "EXG2"], 1024), "PD", "ds002778", "sub-hc1")
    assert SW.validate_subject_windows(sw) and sw.channels == PC.CHANNELS_19 and sw.n_windows == 2
    ok("a modern-named recording → canonical-ordered SubjectWindows (channels == old-10-20 canonical order)")


def main():
    print("ACAR v5 Stage-1B10 guard: channel aliases modern→canonical")
    test_modern_temporal_aliases()
    test_ordered_sources_and_reader_output_canonical()
    print("ALL V5 STAGE1B-CHANNEL-ALIASES-MODERN GUARDS PASS")


if __name__ == "__main__":
    main()
