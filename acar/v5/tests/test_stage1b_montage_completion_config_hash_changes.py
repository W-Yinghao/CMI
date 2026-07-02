"""Guard (Stage-1B11): the montage-completion policy is pinned in preprocessing_config (so it is part of preprocessing_config_sha256);
the channel-alias and montage-completion policy sub-hashes are deterministic, 64-hex, and distinct. No mne needed."""
from __future__ import annotations
import copy
import hashlib
import json
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import ok

_HEX = "0123456789abcdef"


def _hex64(h):
    return len(h) == 64 and all(c in _HEX for c in h)


def test_policy_pinned_and_hashes():
    cfg = PC.PREPROCESSING_CONFIG
    assert cfg["montage_completion_policy_version"] == "ACAR_V5_STAGE1B11_MONTAGE_COMPLETION_V1"
    assert cfg["allowed_missing_by_cohort"] == {"ds004584": ["Pz"], "ds004000": ["F3", "F4", "P3", "P4"]}
    ca, mc, full = PC.channel_alias_policy_sha256(), PC.montage_completion_policy_sha256(), PC.config_sha256()
    assert _hex64(ca) and _hex64(mc) and _hex64(full)
    assert ca != mc and ca != full and mc != full
    assert (ca, mc) == (PC.channel_alias_policy_sha256(), PC.montage_completion_policy_sha256())   # deterministic
    ok("montage-completion policy pinned; channel_alias/montage_completion policy sub-hashes 64-hex, deterministic, distinct")


def test_montage_policy_is_part_of_config_sha256():
    d = copy.deepcopy(PC.PREPROCESSING_CONFIG)
    del d["allowed_missing_by_cohort"]                        # dropping a montage field changes the config hash
    h2 = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert h2 != PC.config_sha256()
    ok("the montage-completion policy is part of preprocessing_config_sha256 (removing a field changes it)")


def main():
    print("ACAR v5 Stage-1B11 guard: montage completion config hash changes")
    test_policy_pinned_and_hashes()
    test_montage_policy_is_part_of_config_sha256()
    print("ALL V5 STAGE1B-MONTAGE-COMPLETION-CONFIG-HASH GUARDS PASS")


if __name__ == "__main__":
    main()
