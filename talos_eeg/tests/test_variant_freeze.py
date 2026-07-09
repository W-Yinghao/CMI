from __future__ import annotations

from talos_eeg.red_team.variant_freeze import (
    ALLOWED_VARIANTS,
    VariantFreezeFailure,
    build_variant_freeze_config,
    validate_variant_freeze,
)


def test_variant_freeze_accepts_exact_talos00a_universe():
    payload = build_variant_freeze_config(seed=0)
    result = validate_variant_freeze(payload)
    assert result.passed
    assert result.allowed_variants == ALLOWED_VARIANTS


def test_variant_freeze_rejects_runtime_variant_addition():
    payload = build_variant_freeze_config(seed=0)
    payload["allowed_variants"] = list(ALLOWED_VARIANTS) + ["TALOS_LR"]
    try:
        validate_variant_freeze(payload)
    except VariantFreezeFailure:
        return
    raise AssertionError("variant freeze must reject TALOS_LR in TALOS_00A")
