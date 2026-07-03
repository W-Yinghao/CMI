"""Guard (Stage-1B12): being on the completion whitelist is NOT, by itself, authorization. ds004367 whitelists F7, yet a ds004367
recording missing F7 is completed ONLY with the variant pattern (conditional gate); and F7 remains NON-whitelisted for every other
cohort, so F7-missing there fails for a DIFFERENT reason (not on the whitelist)."""
from __future__ import annotations
from acar.v5.substrate import montage_completion as MC
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import ok, expect_raises, make_mne_raw, modern_channel_names


def test_f7_is_whitelisted_for_ds004367_but_not_unconditional():
    assert MC.allowed_missing_for("ds004367") == {"F7"}            # F7 IS whitelisted for ds004367
    present = [c for c in modern_channel_names() if c != "F7"] + ["AF3", "FC1"]   # missing F7, NO variant pattern
    err = None
    try:
        MC.complete_missing_channels(make_mne_raw(present, 2048, 256.0), "SCZ", "ds004367")
    except MC.MontageCompletionError as e:
        err = str(e)
    assert err is not None and "variant" in err.lower()           # failed on the CONDITIONAL gate, not the whitelist
    ok("F7 is whitelisted for ds004367 yet still fails without the variant pattern → whitelist ≠ unconditional authorization")


def test_f7_not_whitelisted_for_other_cohorts():
    assert "F7" not in MC.allowed_missing_for("ds004584") and "F7" not in MC.allowed_missing_for("ds004000")
    present = [c for c in modern_channel_names() if c != "F7"] + ["F7-0", "F7-1", "AF3"]   # even WITH variants present
    err = None
    try:
        MC.complete_missing_channels(make_mne_raw(present, 2048, 256.0), "PD", "ds004584")
    except MC.MontageCompletionError as e:
        err = str(e)
    assert err is not None and "whitelist" in err.lower()          # different failure mode: not whitelisted at all
    ok("F7 is NOT whitelisted for ds004584/ds004000 → F7-missing fails as not-whitelisted even with the variant pattern present")


def test_conditional_config_pins_ds004367_f7():
    cond = PC.PREPROCESSING_CONFIG["conditional_montage_completion"]
    assert cond == {"ds004367": {"channel": "F7", "require_variant_names": ["F7-0", "F7-1"]}}
    ok("the conditional-completion policy pins exactly ds004367 → F7 requires F7-0/F7-1")


def main():
    print("ACAR v5 Stage-1B12 guard: ds004367 F7 whitelist is conditional, not unconditional")
    test_f7_is_whitelisted_for_ds004367_but_not_unconditional()
    test_f7_not_whitelisted_for_other_cohorts()
    test_conditional_config_pins_ds004367_f7()
    print("ALL V5 STAGE1B12-DS004367-F7-CONDITIONAL GUARDS PASS")


if __name__ == "__main__":
    main()
