"""Guard (Stage-1B10): channel-name case is normalized so upper-case Fp names (FP1/FP2, common in modern caps) collapse to the
canonical Fp1/Fp2. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.tests._util import ok, modern_channel_names


def test_fp_case_normalized():
    for name in ("FP1", "fp1", "Fp1", " Fp1 "):
        assert CA.normalize_channel(name) == "Fp1", name
    for name in ("FP2", "fp2", "Fp2"):
        assert CA.normalize_channel(name) == "Fp2", name
    # a recording using FP1/FP2 + modern temporals still resolves to all 19 canonical
    ch = ["FP1", "FP2"] + [c for c in modern_channel_names() if c not in ("Fp1", "Fp2")]
    assert set(CA.resolve_canonical_sources(ch)) == set(PC.CHANNELS_19)
    ok("FP1/FP2/fp1 → Fp1/Fp2 (case-normalized); a FP*-cap recording resolves to all 19 canonical")


def main():
    print("ACAR v5 Stage-1B10 guard: channel aliases FP case normalized")
    test_fp_case_normalized()
    print("ALL V5 STAGE1B-CHANNEL-ALIASES-FP-CASE GUARDS PASS")


if __name__ == "__main__":
    main()
