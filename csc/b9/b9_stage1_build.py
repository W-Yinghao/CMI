"""B9.1A prospective-pilot PRE-ACQUISITION build (development-only; NO real data collected; NO scientific/biological claim).

Produces the pre-registration artifacts for a real randomized-audit pilot AND verifies the B9.0 hardened pipeline against
the pilot's ACTUAL hash-pinned assignment table -- on SYNTHETIC Z only (machinery, NOT biology). Actual human-data
collection is GATED on the institution's governance process and is NOT performed here.

Outputs (to results/b9_stage1_prospective_pilot/):
  b9_stage1_assignment_table.csv (+ .sha256)   the PILOT table, generated BEFORE any recording (the crux provenance artifact)
  b9_stage1_manifest.json                        pilot table manifest (generated_before_recording / Y_design_pre_assignment)
  b9_stage1_rows.jsonl                           machinery-verification rows: (B) forced-violation + (C) semi-synthetic diagnostic
  b9_stage1_tables.json                          summary + the SIZE-GATE finding + analysis A = PENDING_ACQUISITION
  b9_stage1_contract_checks.json                 pipeline-verification invariants
"""
import os, sys, json, hashlib, csv
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")
from csc.b9 import randomization_table as RT
from csc.b9 import state_machine as SM

HI, LO = RT.HI, RT.LO
PKG = "/home/infres/yinwang/CMI_AAAI_csc/results/b9_stage1_prospective_pilot"
D = 10
# LOCKED pilot dimensions (reviewer B9.1A: subjects 6-10, microblocks 8-12, trials/2x2-cell/microblock 2-4)
PILOT_SUBJECTS, PILOT_MB, PILOT_R = 8, 10, 3          # -> 12 trials/microblock, 120 trials/subject, 960 total
POWER_SIM_SUBJECTS = 24                               # power-SIZING SIMULATION only (n_eligible>=20); NOT the real pilot
TABLE_SEED = 910_000_001                              # pre-recording assignment-table seed (pinned)
SYNTH_SEED = 915_000_000                              # synthetic-Z seed (machinery only)


def _make_Z(Y_design, C, rng, concept=False):
    """SYNTHETIC Z (machinery only, NOT biological): a boundary for Y_design; if concept=True the boundary is rotated in
    C=HI trials (a genuine C x Z interaction the exact null can detect). Identical structure to the B9.0 dry-run."""
    n = len(Y_design); w = np.zeros(D); w[0] = 1.0; wr = np.zeros(D); wr[1] = 1.0
    sgn = np.where(np.asarray(Y_design) == 1, 1.0, -1.0)
    Z = rng.standard_normal((n, D)) + 1.4 * sgn[:, None] * w[None, :]
    if concept:
        hi = np.asarray(C) == HI
        Z[hi] -= 1.4 * sgn[hi, None] * w[None, :]; Z[hi] += 1.8 * sgn[hi, None] * wr[None, :]
    return Z


def _with_trial_id(tab):
    """Add a per-row trial_id (0..N-1 in table order) -- the canonical join key a real B9.1 execution adheres to."""
    n = len(tab["C"]); tab = dict(tab); tab["trial_id"] = np.arange(n); return tab


def _write_table_csv(tab, path):
    with open(path, "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["subject_id", "microblock_id", "trial_id", "C", "Y_design"])
        for i in range(len(tab["C"])):
            wtr.writerow([int(tab["subject"][i]), int(tab["microblock"][i]), int(tab["trial_id"][i]),
                          int(tab["C"][i]), int(tab["Y_design"][i])])
    open(path + ".sha256", "w").write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "  " + os.path.basename(path) + "\n")


_UNSET = object()   # distinguishes "use the default pilot table" from an explicit table=None ("missing table" violation)


def certify_synth(tab, seed, concept, table_override=_UNSET, exec_override=None):
    """Run the B9.0 pipeline on SYNTHETIC Z following `tab` (machinery only). exec_override(dict) can replace the executed
    (C/Y_design/subject/microblock) to model a forced violation; table_override (may be None = MISSING table) replaces the
    pinned table/manifest; _UNSET keeps the default pilot table."""
    rng = np.random.default_rng(seed)
    S, MB, C, Yd = (np.asarray(tab["subject"]), np.asarray(tab["microblock"]), np.asarray(tab["C"]), np.asarray(tab["Y_design"]))
    ZC, ZYd = (exec_override or {}).get("C", C), (exec_override or {}).get("Y_design", Yd)
    ZS, ZMB = (exec_override or {}).get("subject", S), (exec_override or {}).get("microblock", MB)
    natural = (exec_override or {}).get("natural_prevalence_requested", False)
    Z = _make_Z(Yd, C, rng, concept=concept)            # Z built from the ORIGINAL pre-assigned cue Yd
    if exec_override and "Z_len" in exec_override:       # attrition: drop rows
        keep = exec_override["Z_len"]; Z = Z[keep]
    table = tab if table_override is _UNSET else table_override   # None => MISSING table (a real violation)
    r = SM.b9_certify(Z, ZYd, ZC, ZS, ZMB, table, seed=seed, natural_prevalence_requested=natural)
    return r


def main():
    os.makedirs(PKG, exist_ok=True)
    # 1. PILOT assignment table -- generated BEFORE any recording, hash-pinned (the crux provenance artifact)
    pilot = _with_trial_id(RT.make_assignment_table(np.arange(PILOT_SUBJECTS), PILOT_MB, PILOT_R, TABLE_SEED))
    _write_table_csv(pilot, f"{PKG}/b9_stage1_assignment_table.csv")
    json.dump(dict(pilot.get("manifest", {}), n_trials=len(pilot["C"]),
                   pilot_dims=dict(subjects=PILOT_SUBJECTS, microblocks=PILOT_MB, R=PILOT_R, trials_per_subject=PILOT_MB * PILOT_R * 4),
                   note="generated BEFORE recording; executed acquisition MUST adhere row-for-row (join on trial_id); NO data collected yet (governance-gated)"),
              open(f"{PKG}/b9_stage1_manifest.json", "w"), indent=1, default=str)

    rows = []
    # 2. (B) FORCED-VIOLATION validator tests on manifest/table COPIES (NOT the real raw data) -> must REFUSE before p
    S, MB, C, Yd = pilot["subject"], pilot["microblock"], pilot["C"], pilot["Y_design"]
    violations = {
        "V_flipped_C":            dict(C=1 - np.asarray(C)),
        "V_shuffled_Ydesign":     dict(Y_design=np.asarray(Yd)[np.random.default_rng(1).permutation(len(Yd))]),
        "V_missing_table":        dict(_table=None),
        "V_condition_lock":       dict(C=np.array([HI if (int(s) + int(mb)) % 2 == 0 else LO for s, mb in zip(S, MB)], int)),
        "V_microblock_imbalance": dict(_drop=(np.asarray(C) == HI) & (np.asarray(Yd) == 1) & (np.random.default_rng(2).random(len(C)) < 0.5)),
        "V_corrupt_pinned_hash":  dict(_hash="deadbeefdeadbeef"),
        "V_posthoc_manifest":     dict(_manifest=dict(generated_before_recording=False, Y_design_pre_assignment=False)),
    }
    for name, ov in violations.items():
        tab_ov, exec_ov = _UNSET, {}     # _UNSET = keep the default pilot table
        if "_table" in ov and ov["_table"] is None:
            tab_ov = None                # explicit None = MISSING table violation
        elif "_hash" in ov:
            tab_ov = dict(pilot); tab_ov["manifest"] = dict(pilot["manifest"], table_hash=ov["_hash"])
        elif "_manifest" in ov:
            tab_ov = dict(pilot); tab_ov["manifest"] = dict(pilot["manifest"], **ov["_manifest"])
        elif "_drop" in ov:
            keep = ~ov["_drop"]
            exec_ov = dict(C=np.asarray(C)[keep], Y_design=np.asarray(Yd)[keep], subject=np.asarray(S)[keep],
                           microblock=np.asarray(MB)[keep], Z_len=keep)
        else:
            exec_ov = {k: v for k, v in ov.items() if not k.startswith("_")}
        r = certify_synth(pilot, SYNTH_SEED + 1, concept=False, table_override=tab_ov, exec_override=exec_ov)
        refused = (str(r["b9_state"]) == "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND") and not r["ran_test"]
        rows.append(dict(analysis="B_forced_violation", case=name, b9_state=str(r["b9_state"]), ran_test=bool(r["ran_test"]),
                         refused_before_pvalue=bool(refused), invalid_reasons=list(r["invalid_reasons"]), diagnostic_only=True))

    # 3. (C) SEMI-SYNTHETIC diagnostic on synthetic Z following the LOCKED table (machinery/power-sizing; NOT biological)
    #    (C1) at PILOT n=8: crossfit RUNS but n_eligible=8 < 20 -> size gate BLOCKS any alert (the disclosed pilot-size finding)
    r1 = certify_synth(pilot, SYNTH_SEED + 10, concept=True)
    rows.append(dict(analysis="C_semisynth_pilot_n", case="pilot_concept", b9_state=str(r1["b9_state"]), ran_test=bool(r1["ran_test"]),
                     n_eligible=int(r1["n_eligible"]), p_meanT=float(r1["p_meanT"]), p_stud=float(r1["p_stud"]),
                     note="pilot n<20 -> size gate blocks alert regardless of signal", diagnostic_only=True))
    #    (C2/C3) POWER-SIZING SIMULATION n=24 (NOT the real 6-10 pilot): does the pipeline RUN + DETECT + CONTROL at alert-capable n?
    powtab = _with_trial_id(RT.make_assignment_table(np.arange(POWER_SIM_SUBJECTS), PILOT_MB, PILOT_R, TABLE_SEED + 5))
    rc = certify_synth(powtab, SYNTH_SEED + 20, concept=True)
    rn = certify_synth(powtab, SYNTH_SEED + 21, concept=False)
    rows.append(dict(analysis="C_powersizing_sim_n24", case="concept", b9_state=str(rc["b9_state"]), ran_test=bool(rc["ran_test"]),
                     n_eligible=int(rc["n_eligible"]), p_meanT=float(rc["p_meanT"]), p_stud=float(rc["p_stud"]),
                     note="SIMULATION on synthetic Z, NOT the real pilot, NOT biological -- power-sizing only", diagnostic_only=True))
    rows.append(dict(analysis="C_powersizing_sim_n24", case="null_no_concept", b9_state=str(rn["b9_state"]), ran_test=bool(rn["ran_test"]),
                     n_eligible=int(rn["n_eligible"]), p_meanT=float(rn["p_meanT"]), p_stud=float(rn["p_stud"]),
                     note="SIMULATION: contract-valid null at alert-capable n -> should NOT alert", diagnostic_only=True))
    #    (A) real contract-valid null audit on ACQUIRED data -> PENDING (governance-gated)
    rows.append(dict(analysis="A_real_null_audit", case="PENDING_ACQUISITION", b9_state="PENDING_ACQUISITION",
                     note="requires real acquisition of the pilot cohort under the pinned table + institutional governance approval; NOT run", diagnostic_only=True))

    with open(f"{PKG}/b9_stage1_rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, default=str) + "\n")

    b_ok = all(r["refused_before_pvalue"] for r in rows if r["analysis"] == "B_forced_violation")
    tables = dict(
        scope="B9.1A prospective-pilot PRE-ACQUISITION: protocol freeze + hash-pinned pilot table + pipeline verification on SYNTHETIC Z. NO real data, NO biological claim, NOT validation, NO tag.",
        pilot_table=dict(sha256=open(f"{PKG}/b9_stage1_assignment_table.csv.sha256").read().split()[0],
                         n_trials=len(pilot["C"]), dims=dict(subjects=PILOT_SUBJECTS, microblocks=PILOT_MB, R=PILOT_R)),
        analysis_B_forced_violation=dict(all_refused_before_pvalue=bool(b_ok),
                                         cases={r["case"]: r["b9_state"] for r in rows if r["analysis"] == "B_forced_violation"}),
        analysis_C_semisynth=dict(pilot_n8=dict(state=r1["b9_state"], n_eligible=int(r1["n_eligible"]), ran_test=bool(r1["ran_test"])),
                                  powersizing_sim_n24_concept=dict(state=rc["b9_state"], n_eligible=int(rc["n_eligible"])),
                                  powersizing_sim_n24_null=dict(state=rn["b9_state"], n_eligible=int(rn["n_eligible"]))),
        analysis_A_real_null_audit="PENDING_ACQUISITION (governance-gated; not run)",
        SIZE_GATE_FINDING=("The B9.0 alert conjunction requires n_eligible >= min_confirm_pairs=20 SUBJECTS. The authorized "
                           "pilot size (6-10 subjects) is BELOW that, so the pilot CANNOT emit B9_CONCEPT_ALERT regardless of "
                           "signal -- it is a contract-feasibility + exact-null-runs + violations-refuse pilot, and its "
                           "'no false alert' is size-gate-trivial, NOT a strong null test. Alert-level power AND alert-level "
                           "null-control require a cohort of >=20 eligible subjects (shown here only by a SYNTHETIC power-sizing "
                           "simulation). This is a DESIGN FINDING for the reviewer, NOT a gate change (min_confirm_pairs is UNCHANGED)."),
        governance="Actual human-EEG collection is gated on the institution's governance process; NO data collected here. This is not legal/IRB advice.",
        next="B9.1A real pilot: after governance approval, acquire the pilot cohort under the pinned table, then run analysis A. For alert-level claims a >=20-subject cohort is required (reviewer decision).")
    json.dump(tables, open(f"{PKG}/b9_stage1_tables.json", "w"), indent=1, default=str)

    json.dump(dict(scope="B9.1A pipeline-verification invariants (synthetic Z; machinery only, NO biological claim)",
                   B_forced_violations_all_refuse_before_pvalue=bool(b_ok),
                   C_pilot_n_size_gate_blocks_alert=bool(str(r1["b9_state"]) != "B9_CONCEPT_ALERT" and r1["ran_test"] and int(r1["n_eligible"]) < 20),
                   C_powersizing_sim_detects_concept=bool(str(rc["b9_state"]) == "B9_CONCEPT_ALERT"),
                   C_powersizing_sim_null_no_alert=bool(str(rn["b9_state"]) != "B9_CONCEPT_ALERT"),
                   A_real_null_audit="PENDING_ACQUISITION",
                   validator_hardened_from_b9_0="full-tuple adherence + provenance attestation carried unchanged into the pilot pipeline"),
              open(f"{PKG}/b9_stage1_contract_checks.json", "w"), indent=1, default=str)
    print(f"B9.1A build: B_all_refuse={b_ok} | C_pilot(n8)={r1['b9_state']}(nelig{r1['n_eligible']}) | "
          f"C_powsim(n24)concept={rc['b9_state']}(nelig{rc['n_eligible']}) null={rn['b9_state']} | A=PENDING")
    print("B9_STAGE1_BUILD_OK")


if __name__ == "__main__":
    main()
