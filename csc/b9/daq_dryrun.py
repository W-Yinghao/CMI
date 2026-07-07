"""B9.3 acquisition-readiness / DAQ dry-run (development-only; NON-HUMAN, SIMULATED; NO real EEG, NO scientific claim).

Verifies the acquisition PLUMBING before any human recording, on MOCK event logs (no real signal, no Z, no statistical
null): (1) the assignment player reads the hash-pinned table READ-ONLY and does NOT regenerate the assignment; (2) event
markers are frozen + unambiguous and trial_id is recoverable from the companion log; (3) the executed log trial_id-joins to
the pinned table with full-tuple adherence; (4) the operations checkpoint stays blinded to scientific endpoints; (5) forced
corruptions of the executed log REFUSE (CONTRACT_INVALID) before any p-value. This is DAQ readiness, NOT Analysis A.
"""
import numpy as np
from csc.b9 import randomization_table as RT
from csc.b9 import acquisition_adherence as AA
from csc.b9 import operations_checkpoint as OC

# frozen cue codes (mirror of b9_stage3_event_marker_map.json): tens=C+1, ones=Y_design+1
CUE_CODE = {(0, 0): 11, (0, 1): 12, (1, 0): 21, (1, 1): 22}


def assignment_player(pinned_table, jitter_seed=0):
    """SIMULATED read-only assignment player. Reads the pinned table in row order and emits a mock recording: a trigger
    stream (cue codes) + a companion trial log. It does NOT modify or regenerate the assignment (pure reader). Timing
    jitter is simulated + logged; it never changes C/Y_design/trial_id. Returns (companion_log, trigger_stream, played_hash)."""
    rng = np.random.default_rng(jitter_seed)
    S, MB, TID, C, Yd = (np.asarray(pinned_table["subject"]), np.asarray(pinned_table["microblock"]),
                         np.asarray(pinned_table["trial_id"]), np.asarray(pinned_table["C"]), np.asarray(pinned_table["Y_design"]))
    companion, triggers = [], []
    t = 0.0
    for i in range(len(TID)):
        t += 1500.0 + float(rng.normal(0, 8.0))              # scheduled ISI + simulated timing jitter (LOGGED, not structural)
        code = CUE_CODE[(int(C[i]), int(Yd[i]))]
        companion.append(dict(subject_slot=int(S[i]), microblock_id=int(MB[i]), trial_id=int(TID[i]),
                              C=int(C[i]), Y_design=int(Yd[i]), t_cue_ms=round(t, 1), cue_code=code, artifact_flag=0))
        triggers.append(code)
    # the player is a pure READER: what it 'played' must equal the pinned table (no regeneration)
    played_hash = RT.table_hash([r["C"] for r in companion], [r["Y_design"] for r in companion],
                                [r["subject_slot"] for r in companion], [r["microblock_id"] for r in companion])
    return companion, triggers, played_hash


def _executed_from_companion(companion):
    return dict(subject=np.array([r["subject_slot"] for r in companion]), microblock=np.array([r["microblock_id"] for r in companion]),
                trial_id=np.array([r["trial_id"] for r in companion]), C=np.array([r["C"] for r in companion]),
                Y_design=np.array([r["Y_design"] for r in companion]))


def _marker_integrity(companion, triggers):
    """Every emitted trigger cue_code must equal CUE_CODE[(C,Y_design)] of its companion row (redundant integrity)."""
    return all(triggers[i] == CUE_CODE[(int(companion[i]["C"]), int(companion[i]["Y_design"]))] for i in range(len(companion)))


def daq_readiness_check(pinned_table):
    """Full DAQ readiness dry-run on the pinned table. Returns a dict of checks + forced-corruption results. NO Z, NO
    p-value, NO alert -- acquisition plumbing only."""
    companion, triggers, played_hash = assignment_player(pinned_table)
    exec_rows = _executed_from_companion(companion)
    # (1) read-only player: the played schedule equals the pinned table -- BOTH the (order-invariant) composition hash AND a
    # STRICT row-for-row / trial_id-sequence attestation (a reordering or trial_id-reassigning player must FAIL this).
    comp_hash_ok = (played_hash == pinned_table["manifest"]["table_hash"])
    row_exact = all(np.array_equal(np.asarray(exec_rows[k]), np.asarray(pinned_table[tk]))
                    for k, tk in (("trial_id", "trial_id"), ("C", "C"), ("Y_design", "Y_design"),
                                  ("subject", "subject"), ("microblock", "microblock")))
    player_readonly = bool(comp_hash_ok and row_exact)
    # (2) marker integrity + trial_id recoverable
    marker_ok = _marker_integrity(companion, triggers)
    tid_recoverable = (len(set(r["trial_id"] for r in companion)) == len(companion))
    # (3) executed log trial_id-join adherence (Z/T-blind validator; VALID on a faithful full recording)
    st, dg = AA.check_contract_acquisition(exec_rows, pinned_table)
    adherence_ok = (st is None and float(dg["adherence"]) >= 1.0)
    # (4) operations checkpoint blinded (only logistics; never p/alert)
    cp = OC.checkpoint_report(exec_rows, pinned_table)
    checkpoint_blind = not (set(cp) & set(OC.FORBIDDEN_FIELDS))
    # (5) forced corruptions of the executed log MUST refuse before any p-value
    rng = np.random.default_rng(3)
    n = len(exec_rows["C"]); C = exec_rows["C"]; Yd = exec_rows["Y_design"]; S = exec_rows["subject"]; MB = exec_rows["microblock"]
    def _sub(mask):
        k = np.where(~mask)[0]
        return {kk: np.asarray(v)[k] for kk, v in exec_rows.items()}
    dropped_imb = (C == RT.HI) & (Yd == 1) & (rng.random(n) < 0.5)
    corruptions = {
        "missing_table":        (None, exec_rows),
        "flipped_C":            (pinned_table, dict(exec_rows, C=1 - C)),
        "shuffled_Ydesign":     (pinned_table, dict(exec_rows, Y_design=Yd[rng.permutation(n)])),
        "trial_id_mismatch":    (pinned_table, dict(exec_rows, trial_id=np.asarray(exec_rows["trial_id"]) + 10_000_000)),
        "microblock_imbalance": (pinned_table, _sub(dropped_imb)),
        "condition_lock":       (pinned_table, dict(exec_rows, C=np.array([RT.HI if (int(s) + int(m)) % 2 == 0 else RT.LO for s, m in zip(S, MB)], int))),
        "marker_code_mismatch": "MARKER",   # cue code disagrees with companion (C,Y_design)
    }
    forced = {}
    for name, spec in corruptions.items():
        if spec == "MARKER":
            bad_triggers = list(triggers); bad_triggers[0] = 99   # a wrong/undeclared cue code
            forced[name] = dict(refused=(not _marker_integrity(companion, bad_triggers)), via="marker_integrity")
        else:
            pin, er = spec
            st2, dg2 = AA.check_contract_acquisition(er, pin)
            forced[name] = dict(refused=(st2 == "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND"), state=str(st2),
                                reasons=list(dg2.get("invalid_reasons", [])))
    all_forced_refuse = all(v["refused"] for v in forced.values())
    return dict(player_readonly=bool(player_readonly), player_composition_hash_ok=bool(comp_hash_ok), player_row_exact=bool(row_exact),
                marker_integrity=bool(marker_ok), trial_id_recoverable=bool(tid_recoverable),
                executed_adherence_valid=bool(adherence_ok), checkpoint_blinded=bool(checkpoint_blind),
                forced_corruptions_all_refuse=bool(all_forced_refuse), forced=forced,
                checkpoint_fields=sorted(cp.keys()), n_trials=len(companion), median_timing_jitter_logged=True,
                note="SIMULATED non-human DAQ dry-run; NO real EEG, NO Z, NO statistical null, NO alert -- acquisition plumbing only")
