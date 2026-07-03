"""Guard (Stage-1B13): a GENERIC-headered BrainVision recording (EEG001..EEG0NN) in ds003944/ds003947 with a valid channels.tsv gets
the channel_names_from_channels_tsv_for_generic_brainvision repair — the staged header's [Channel Infos] is renamed from channels.tsv
by ROW ORDER (composing with the marker synth), and the repaired header opens with the real electrode names resolving all 19 canonical."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.substrate import channel_aliases as CA
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]                # 21 rows: 19 canonical (modern) + 2 non-canonical extras


def test_rename_allowed_both_cohorts():
    import mne
    for cohort in ("ds003944", "ds003947"):
        raw_dir = tempfile.mkdtemp()
        vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, n_points=300, with_marker=False,
                                        generic_header=True, write_channels_tsv=True)
        plan = BR.plan_repair("SCZ", cohort, "sub-x", vhdr)
        assert plan is not None and plan.mode == BR.MODE_CHANNEL_NAMES_FROM_TSV
        assert list(plan.channel_name_map) == _REAL and plan.marker_target == ""      # marker-less → also synthesize a marker
        repaired, man = BR.apply_repair(plan, tempfile.mkdtemp())
        assert man["channel_name_source"] == "channels.tsv" and man["generated_marker_sha256"] is not None
        r = mne.io.read_raw_brainvision(repaired, preload=False, verbose="ERROR")
        assert list(r.ch_names) == _REAL                                              # header now carries the real electrode names
        assert sum(1 for n in r.ch_names if CA.normalize_channel(n) is not None) == 19  # all 19 canonical resolve
    ok("generic EEG00N header + valid channels.tsv in ds003944/ds003947 → row-order rename + marker synth; repaired header resolves 19")


def test_rename_composes_with_existing_marker():
    # generic header but a MarkerFile already present → rename only, no marker synthesized
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-y_task-Rest_eeg", _REAL, with_marker=True,
                                    generic_header=True, write_channels_tsv=True)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-y", vhdr)
    assert plan.mode == BR.MODE_CHANNEL_NAMES_FROM_TSV and plan.marker_target != ""
    _, man = BR.apply_repair(plan, tempfile.mkdtemp())
    assert man["generated_marker_sha256"] is None                                     # existing marker kept, none synthesized
    ok("a generic header with an existing marker is renamed only (no marker synthesized)")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename allowed (ds003944/ds003947)")
    test_rename_allowed_both_cohorts()
    test_rename_composes_with_existing_marker()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-ALLOWED GUARDS PASS")


if __name__ == "__main__":
    main()
