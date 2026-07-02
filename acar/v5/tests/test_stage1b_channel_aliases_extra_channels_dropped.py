"""Guard (Stage-1B10): extra non-canonical channels (EXG/VEOG/ECG/Status/high-density montage names) are dropped — only the 19
canonical electrodes are selected. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import real_mne_reader as RMR
from acar.v5.substrate import subject_windows as SW
from acar.v5.tests._util import ok, make_fake_raw, modern_channel_names


def test_extra_channels_dropped():
    ch = modern_channel_names() + ["EXG1", "EXG2", "VEOG", "ECG", "Status", "AF3", "FC1"]   # 19 canonical + extras
    src = CA.resolve_canonical_sources(ch)
    assert set(src) == set(PC.CHANNELS_19) and len(src) == 19
    sw = RMR.raw_to_windows(make_fake_raw(ch, 1024), "SCZ", "ds003944", "sub-1448")
    assert SW.validate_subject_windows(sw) and sw.n_channels == 19 and sw.channels == PC.CHANNELS_19
    ok("extra non-canonical channels dropped; exactly the 19 canonical electrodes selected in canonical order")


def main():
    print("ACAR v5 Stage-1B10 guard: channel aliases extra channels dropped")
    test_extra_channels_dropped()
    print("ALL V5 STAGE1B-CHANNEL-ALIASES-EXTRA GUARDS PASS")


if __name__ == "__main__":
    main()
