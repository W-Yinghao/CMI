"""Guard (Stage-1B13): the rename is a strict ROW-ORDER mapping (header Ch_i ← channels.tsv row i) — no fuzzy / similarity /
reordering. The repaired header's names equal the channels.tsv rows in their exact file order, whatever that order is."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names


def test_rename_is_row_order_not_similarity():
    import mne
    perm = list(reversed(modern_channel_names())) + ["VEOG", "ECG"]   # a NON-canonical channels.tsv row order
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", perm, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, channels_tsv_names=perm)
    plan = BR.plan_repair("SCZ", "ds003944", "sub-x", vhdr)
    assert plan.mode == BR.MODE_CHANNEL_NAMES_FROM_TSV and list(plan.channel_name_map) == perm
    repaired, _ = BR.apply_repair(plan, tempfile.mkdtemp())
    r = mne.io.read_raw_brainvision(repaired, preload=False, verbose="ERROR")
    assert list(r.ch_names) == perm            # EXACT channels.tsv row order — NOT reordered to canonical / by similarity
    ok("the rename maps header Ch_i ← channels.tsv row i exactly (row order, no fuzzy/similarity/reordering)")


def main():
    print("ACAR v5 Stage-1B13 guard: channels.tsv rename is strict row-order (no fuzzy matching)")
    test_rename_is_row_order_not_similarity()
    print("ALL V5 STAGE1B13-CHANNELS-TSV-RENAME-NO-FUZZY GUARDS PASS")


if __name__ == "__main__":
    main()
