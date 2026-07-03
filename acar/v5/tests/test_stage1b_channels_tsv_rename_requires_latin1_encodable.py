"""Guard (Stage-1B13): BrainVision headers are latin-1, so a channels.tsv containing a NON-latin-1 channel name must NOT produce a
rename plan (fail-closed → marker-only fallback); and, defense-in-depth, apply_repair on a forged non-latin-1 rename plan raises a
BrainvisionReadRepairError (the caught type) rather than letting a raw UnicodeEncodeError escape and crash the preflight/reader."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, expect_raises, make_brainvision_triplet, modern_channel_names

_REAL_BAD = modern_channel_names() + ["EEGα", "ECG"]      # 19 canonical (ascii) + a non-latin-1 extra name (Greek α)


def test_non_latin1_name_blocks_rename_plan():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL_BAD, with_marker=False,
                                    generic_header=True, write_channels_tsv=True)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan is not None and plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV   # no rename → marker-only fallback (fail-closed)
    ok("a channels.tsv with a non-latin-1 channel name → no rename plan (fail-closed, marker-only fallback)")


def test_apply_repair_on_non_latin1_plan_raises_caught_type():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-y_task-Rest_eeg", _REAL_BAD, with_marker=False,
                                    generic_header=True, write_channels_tsv=True)
    # forge a mode-C plan directly (bypassing the planner's guard) to prove apply_repair itself fails closed
    forged = BR.RepairPlan(mode=BR.MODE_CHANNEL_NAMES_FROM_TSV, disease="SCZ", cohort="ds003944", subject="sub-y",
                           recording="sub-y_task-Rest_eeg.vhdr", original_vhdr_path=vhdr,
                           data_target=vhdr[: -len(".vhdr")] + ".eeg", marker_target="",
                           reason="forged", channel_name_map=tuple(_REAL_BAD),
                           channels_tsv_path=BR._channels_tsv_path(vhdr))
    expect_raises(BR.BrainvisionReadRepairError, lambda: BR.apply_repair(forged, tempfile.mkdtemp()),
                  "apply_repair must convert a non-latin-1 encode error into BrainvisionReadRepairError")
    ok("apply_repair on a non-latin-1 rename plan raises BrainvisionReadRepairError (no raw UnicodeEncodeError escapes)")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename requires latin-1-encodable names")
    test_non_latin1_name_blocks_rename_plan()
    test_apply_repair_on_non_latin1_plan_raises_caught_type()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-LATIN1 GUARDS PASS")


if __name__ == "__main__":
    main()
