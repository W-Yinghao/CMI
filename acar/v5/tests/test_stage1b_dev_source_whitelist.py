"""Guard (Stage-1B0): DEV-source paths are WHITELIST-validated (disease-matched frozen DEV cohort only; no other-disease cohort /
external site / artifact / cache). String checks only — opens nothing. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_manifest as MAN
from acar.v5.tests._util import expect_raises, ok


def test_disease_matched_dev_path_ok():
    assert MAN.validate_dev_source_path("PD", "/projects/datalake/raw/bids/ds002778/sub-001")
    assert MAN.validate_dev_source_path("SCZ", "/projects/datalake/raw/bids/ds003944/sub-010")
    ok("a disease-matched frozen DEV cohort path passes the whitelist (string-only; nothing opened)")


def test_other_disease_cohort_rejected():
    expect_raises(MAN.Stage1bWhitelistError, lambda: MAN.validate_dev_source_path("PD", "/data/ds003944/sub-1"), "SCZ cohort for PD")
    expect_raises(MAN.Stage1bWhitelistError, lambda: MAN.validate_dev_source_path("SCZ", "/data/ds002778/sub-1"), "PD cohort for SCZ")
    ok("an other-disease DEV cohort path → rejected (disease-matched whitelist)")


def test_no_dev_cohort_rejected():
    expect_raises(MAN.Stage1bWhitelistError, lambda: MAN.validate_dev_source_path("PD", "/data/random/sub-1"), "no DEV cohort")
    expect_raises(MAN.Stage1bWhitelistError, lambda: MAN.validate_dev_source_path("PD", ""), "empty")
    ok("a path with no disease DEV cohort token → rejected")


def test_external_sites_and_artifacts_rejected():
    bad = ["/data/zenodo14808296/sub-1", "/data/ds007526/sub-1", "/data/zenodo14178398/sub-1", "/data/ds007020/sub-1",
           "/projects/EEG-foundation-model/datalake/raw/scps/cache/PD.npz",
           "/home/x/acar_v4_regen_outputs/enc.pt", "/x/feat_dump_v4/audit_PD_ds002778_erm_0.npz"]
    for b in bad:
        expect_raises(MAN.Stage1bWhitelistError, lambda b=b: MAN.validate_dev_source_path("PD", b), b)
    ok("external sites (primary/provisional/excluded) + scps cache + v4 artifacts → rejected even if 'PD'")


def test_dev_path_with_site_token_rejected():
    # even a path that also mentions a DEV cohort is rejected if it references an external site or artifact marker
    expect_raises(MAN.Stage1bWhitelistError,
                  lambda: MAN.validate_dev_source_path("PD", "/x/ds002778/but_also/ds007526/leak"))
    expect_raises(MAN.Stage1bWhitelistError,
                  lambda: MAN.validate_dev_source_path("PD", "/x/ds002778/feat_dump_v4/leak"))
    ok("a DEV path that ALSO references a site/artifact marker → still rejected")


def main():
    print("ACAR v5 Stage-1B0 guard: DEV-source whitelist")
    test_disease_matched_dev_path_ok()
    test_other_disease_cohort_rejected()
    test_no_dev_cohort_rejected()
    test_external_sites_and_artifacts_rejected()
    test_dev_path_with_site_token_rejected()
    print("ALL V5 STAGE1B-DEV-SOURCE-WHITELIST GUARDS PASS")


if __name__ == "__main__":
    main()
