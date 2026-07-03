"""ACAR V5 Stage-1B montage COMPLETION (mne + numpy LAZY inside the interpolation branch; nothing heavy at module load). The
canonical OUTPUT montage is unchanged (old-10-20 CHANNELS_19). For a small, per-cohort WHITELISTED set of missing canonical
electrodes, this layer interpolates them (mne spherical-spline over standard positions) so the substrate's 19-channel logical layout
is preserved for cohorts whose recordings lack a few of the canonical electrodes. Everything else fails closed.

Design (validated): the recording keeps its MODERN channel names during interpolation (they have standard_1020 positions — old
T3/T4/T5/T6 do NOT). The whitelisted missing channels (Pz, F3/F4/P3/P4) are montage-name-agnostic (identical old/modern, present in
standard_1020), so they are added as flat channels, positioned via standard_1020, marked bad, and interpolated from good-position EEG
donors. The subsequent canonical pick (channel_aliases) drops non-canonical donors and emits the old-canonical 19 in canonical order.

Fail-closed: a missing set NOT in the cohort whitelist, more than the max interpolated, a duplicate logical channel, insufficient
donor geometry, or non-finite interpolated output all raise MontageCompletionError.
"""
from __future__ import annotations
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import preprocessing_config as PC

_STD_MONTAGE = "standard_1020"


class MontageCompletionError(RuntimeError):
    pass


def allowed_missing_for(cohort):
    """The per-cohort whitelist of canonical channels that may be interpolated (empty set if the cohort is not whitelisted)."""
    return set(PC.PREPROCESSING_CONFIG["allowed_missing_by_cohort"].get(cohort, ()))


def _require_conditional(cohort, missing, ch_names):
    """Stage-1B12: a whitelisted channel that is CONDITIONAL (ds004367 F7) may be completed only when the raw header carries the
    required duplicate-variant pattern (F7-0 AND F7-1). The variant names are NEVER aliased/kept-first/averaged into the canonical
    channel — F7 is interpolated from good-position donors like any other whitelisted missing channel. Fail-closed otherwise: being
    on the whitelist is not, by itself, authorization."""
    cond = PC.PREPROCESSING_CONFIG["conditional_montage_completion"].get(cohort)
    if not cond:
        return
    ch = cond["channel"]
    if ch not in missing:
        return                                                   # channel present natively → no completion, no condition to check
    present = {str(n).strip().casefold() for n in ch_names}
    needed = {str(v).strip().casefold() for v in cond["require_variant_names"]}
    if not needed <= present:
        raise MontageCompletionError(
            f"{cohort}: {ch} completion requires the variant channels {list(cond['require_variant_names'])} in the raw header "
            f"(present={sorted(needed & present)}) — whitelist alone is not sufficient; fail-closed")


def _missing_and_dups(ch_names):
    canon_to_src, dups = {}, []
    for n in ch_names:
        c = CA.normalize_channel(n)
        if c is None:
            continue
        if c in canon_to_src:
            dups.append(c)
        else:
            canon_to_src[c] = n
    missing = [c for c in PC.CHANNELS_19 if c not in canon_to_src]
    return missing, sorted(set(dups))


def complete_missing_channels(raw, disease, cohort, mne=None):
    """Return (raw, provenance). No-op (no mne needed) if the recording already has all 19 canonical channels after aliasing.
    Otherwise interpolate the WHITELISTED missing canonical channels; fail-closed on anything outside the policy.
    provenance = {'interpolated': [...canonical...], 'n_interpolated': int, 'donor_count': int}."""
    cfg = PC.PREPROCESSING_CONFIG
    missing, dups = _missing_and_dups(list(raw.ch_names))
    if dups:
        raise MontageCompletionError(f"duplicate logical channel(s) {dups} — never interpolated (fail-closed)")
    if not missing:
        return raw, {"interpolated": [], "n_interpolated": 0, "donor_count": 0}
    allowed = allowed_missing_for(cohort)
    if not set(missing) <= allowed:
        raise MontageCompletionError(f"{disease}/{cohort}: missing canonical {missing} not in the cohort whitelist "
                                     f"{sorted(allowed)} — no montage completion authorized")
    _require_conditional(cohort, missing, list(raw.ch_names))    # ds004367 F7 requires the F7-0/F7-1 variant pattern (fail-closed)
    if len(missing) > int(cfg["max_interpolated_canonical_channels_per_recording"]):
        raise MontageCompletionError(f"{disease}/{cohort}: {len(missing)} missing > max "
                                     f"{cfg['max_interpolated_canonical_channels_per_recording']}")
    if mne is None:
        import mne as _mne  # lazy — only when interpolation is actually needed
        mne = _mne
    import numpy as np
    try:
        flat = mne.io.RawArray(np.zeros((len(missing), raw.n_times)),
                               mne.create_info(list(missing), raw.info["sfreq"], ["eeg"] * len(missing)), verbose="ERROR")
        raw = raw.copy().add_channels([flat], force_update_info=True)
        raw.set_montage(_STD_MONTAGE, on_missing="ignore", match_case=False, verbose="ERROR")
        pos = raw.get_montage().get_positions()["ch_pos"]

        def _has_pos(ch):
            p = pos.get(ch)
            return p is not None and bool(np.isfinite(list(p)).all())
        # donor_policy: unknown_position_channels_ignored — DROP non-canonical channels with no standard position (e.g. ds004367's
        # F7-0/F7-1, GSR/ECG) so they cannot poison the interpolation matrix with NaN coordinates; they are non-canonical and would
        # be dropped by the later canonical pick anyway. The missing channels we are interpolating keep their (added) positions.
        drop = [ch for ch in raw.ch_names if ch not in missing and not _has_pos(ch)]
        if drop:
            raw = raw.copy().drop_channels(drop)
        donors = [ch for ch in raw.ch_names if ch not in missing and _has_pos(ch)]
        if len(donors) < int(cfg["min_donor_channels"]):
            raise MontageCompletionError(f"{disease}/{cohort}: insufficient donor geometry ({len(donors)} < "
                                         f"{cfg['min_donor_channels']})")
        raw.info["bads"] = list(missing)
        raw.interpolate_bads(reset_bads=True, mode=cfg["interpolation_mode"], verbose="ERROR")
        interp_data = raw.get_data(picks=list(missing))
    except MontageCompletionError:
        raise
    except Exception as e:  # noqa: BLE001 — any mne/interp failure is fail-closed
        raise MontageCompletionError(f"{disease}/{cohort}: interpolation failed: {e}")
    if not bool(np.isfinite(interp_data).all()):
        raise MontageCompletionError(f"{disease}/{cohort}: interpolation produced non-finite data for {missing}")
    return raw, {"interpolated": sorted(missing), "n_interpolated": len(missing), "donor_count": len(donors)}
