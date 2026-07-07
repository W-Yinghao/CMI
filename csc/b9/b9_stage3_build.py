"""B9.3 acquisition-readiness build (development-only; SIMULATED non-human DAQ dry-run; NO real EEG, NO scientific claim).
Runs the DAQ readiness dry-run against the pinned B9.2 30-slot assignment table and writes the readiness checks. Real
human recording is GATED on institutional governance and NOT performed here."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")
from csc.b9 import randomization_table as RT
from csc.b9 import daq_dryrun as DAQ

PKG = "/home/infres/yinwang/CMI_AAAI_csc/results/b9_stage3_governance_readiness"
SLOTS, MB, R, TABLE_SEED = 30, 10, 3, 920_000_001   # SAME pinned table as B9.2


def _with_trial_id(tab):
    tab = dict(tab); tab["trial_id"] = np.arange(len(tab["C"])); return tab


def main():
    os.makedirs(PKG, exist_ok=True)
    tab = _with_trial_id(RT.make_assignment_table(np.arange(SLOTS), MB, R, TABLE_SEED))
    chk = DAQ.daq_readiness_check(tab)

    rows = [dict(check=k, value=chk[k], diagnostic_only=True) for k in
            ("player_readonly", "marker_integrity", "trial_id_recoverable", "executed_adherence_valid",
             "checkpoint_blinded", "forced_corruptions_all_refuse")]
    for name, res in chk["forced"].items():
        rows.append(dict(check="forced_corruption", case=name, refused=res["refused"],
                         detail={k: v for k, v in res.items() if k != "refused"}, diagnostic_only=True))
    with open(f"{PKG}/b9_stage3_daq_dryrun_rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, default=str) + "\n")

    ready = bool(chk["player_readonly"] and chk["marker_integrity"] and chk["trial_id_recoverable"]
                 and chk["executed_adherence_valid"] and chk["checkpoint_blinded"] and chk["forced_corruptions_all_refuse"])
    json.dump(dict(scope="B9.3 acquisition-app / DAQ readiness checks (SIMULATED non-human; NO real EEG, NO Z, NO statistical null, NO alert)",
                   pinned_table="the B9.2 30-slot hash-pinned assignment table (regenerated deterministically from seed 920000001)",
                   table_hash=tab["manifest"]["table_hash"],
                   checks=dict(assignment_player_reads_table_read_only_no_regen=bool(chk["player_readonly"]),
                               event_markers_frozen_and_unambiguous=bool(chk["marker_integrity"]),
                               trial_id_recoverable_from_log=bool(chk["trial_id_recoverable"]),
                               executed_log_trial_id_join_adherence=bool(chk["executed_adherence_valid"]),
                               operations_checkpoint_blinded_no_scientific_endpoint=bool(chk["checkpoint_blinded"]),
                               forced_corruptions_refuse_before_pvalue=bool(chk["forced_corruptions_all_refuse"]),
                               forced_cases={k: v["refused"] for k, v in chk["forced"].items()}),
                   checkpoint_exposed_fields=chk["checkpoint_fields"],
                   daq_ready=ready,
                   not_science="acquisition PLUMBING only; NO real EEG collected; NOT Analysis A; NOT validation; NOT a power/biological claim",
                   governance="real human recording gated on institutional approval (status not_started); this dry-run does not authorize collection"),
              open(f"{PKG}/b9_stage3_acquisition_app_checks.json", "w"), indent=1, default=str)
    print(f"B9.3 DAQ dry-run: player_readonly={chk['player_readonly']} marker={chk['marker_integrity']} tid={chk['trial_id_recoverable']} "
          f"adherence={chk['executed_adherence_valid']} checkpoint_blind={chk['checkpoint_blinded']} forced_all_refuse={chk['forced_corruptions_all_refuse']} | DAQ_READY={ready}")
    print("B9_STAGE3_DAQ_DRYRUN_OK" if ready else "B9_STAGE3_DAQ_DRYRUN_FAIL")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
