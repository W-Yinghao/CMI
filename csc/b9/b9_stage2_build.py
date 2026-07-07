"""B9.2 prospective-acquisition PACKAGE build (development-only; NO real data collected; NO scientific/biological claim).

Generates the pre-registration for a >=24-analyzable-subject prospective randomized-audit cohort (30 enrollment slots) AND
verifies the B9.0/B9.2 pipeline against the pinned 30-slot table on SYNTHETIC Z only (machinery, NOT biology). Unlike the
6-10 B9.1A pilot (size-gate-blocked), at n>=20 the contract-valid null audit is a genuine (non-size-gate-trivial) null test and
the injected-positive diagnostic can alert (power). Real human-EEG collection is GATED on institutional governance and NOT
performed here.
"""
import os, sys, json, hashlib, csv
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")
from csc.b9 import randomization_table as RT
from csc.b9 import acquisition_adherence as AA
from csc.b9 import operations_checkpoint as OC

HI, LO = RT.HI, RT.LO
PKG = "/home/infres/yinwang/CMI_AAAI_csc/results/b9_stage2_prospective_acquisition"
D = 10
SLOTS, MB, R = 30, 10, 3                 # 30 enrollment slots x 10 microblocks x R=3 -> 12 trials/mb, 120/subject, 3600 total
TARGET_ANALYZABLE, MIN_ANALYZABLE = 24, 20
TABLE_SEED = 920_000_001                 # pre-recording assignment-table seed (pinned); disjoint from B9.1A 910e6
SYNTH_SEED = 925_000_000


def _make_Z(Y_design, C, rng, concept=False):
    """SYNTHETIC Z (machinery only, NOT biological): boundary for Y_design; concept=True rotates it in C=HI (C x Z)."""
    n = len(Y_design); w = np.zeros(D); w[0] = 1.0; wr = np.zeros(D); wr[1] = 1.0
    sgn = np.where(np.asarray(Y_design) == 1, 1.0, -1.0)
    Z = rng.standard_normal((n, D)) + 1.4 * sgn[:, None] * w[None, :]
    if concept:
        hi = np.asarray(C) == HI
        Z[hi] -= 1.4 * sgn[hi, None] * w[None, :]; Z[hi] += 1.8 * sgn[hi, None] * wr[None, :]
    return Z


def _exec_rows(tab, keep=None):
    """Executed rows (subset if keep mask) in table order, carrying trial_id."""
    idx = np.arange(len(tab["C"])) if keep is None else np.where(keep)[0]
    return dict(subject=np.asarray(tab["subject"])[idx], microblock=np.asarray(tab["microblock"])[idx],
                trial_id=np.asarray(tab["trial_id"])[idx], C=np.asarray(tab["C"])[idx], Y_design=np.asarray(tab["Y_design"])[idx]), idx


def _with_trial_id(tab):
    tab = dict(tab); tab["trial_id"] = np.arange(len(tab["C"])); return tab


def _write_csv(tab, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id", "microblock_id", "trial_id", "C", "Y_design"])
        for i in range(len(tab["C"])):
            w.writerow([int(tab["subject"][i]), int(tab["microblock"][i]), int(tab["trial_id"][i]), int(tab["C"][i]), int(tab["Y_design"][i])])
    open(path + ".sha256", "w").write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "  " + os.path.basename(path) + "\n")


def _certify(tab, exec_rows, seed, concept):
    """Run B9.2 acquisition certify on SYNTHETIC Z built from exec_rows' Y_design/C."""
    rng = np.random.default_rng(seed)
    Z = _make_Z(exec_rows["Y_design"], exec_rows["C"], rng, concept=concept)
    return AA.b9_2_certify(Z, exec_rows, tab, seed=seed)


def main():
    os.makedirs(PKG, exist_ok=True)
    tab = _with_trial_id(RT.make_assignment_table(np.arange(SLOTS), MB, R, TABLE_SEED))
    _write_csv(tab, f"{PKG}/b9_stage2_assignment_table.csv")
    S, MBb, C, Yd, TID = (np.asarray(tab["subject"]), np.asarray(tab["microblock"]), np.asarray(tab["C"]),
                          np.asarray(tab["Y_design"]), np.asarray(tab["trial_id"]))
    json.dump(dict(tab["manifest"], n_trials=len(C), dims=dict(slots=SLOTS, microblocks=MB, R=R, trials_per_subject=MB * R * 4),
                   note="generated BEFORE recording; executed acquisition adheres via trial_id join; NO data collected (governance-gated)"),
              open(f"{PKG}/b9_stage2_manifest.json", "w"), indent=1, default=str)
    json.dump(dict(scope="B9.2 subject-slot enrollment plan", n_enrollment_slots=SLOTS, target_analyzable=TARGET_ANALYZABLE,
                   hard_min_analyzable=MIN_ANALYZABLE, reason="alert-level analysis needs n_eligible>=min_confirm_pairs=20 (UNCHANGED); target 24 reserves for attrition",
                   stop_rule="stop acquisition after >=24 analyzable subjects OR all 30 slots exhausted; 'analyzable' fixed by predeclared quality/support criteria (b9_stage2_quality_criteria.json), NEVER by p-values/effect sizes",
                   slots=[dict(slot=i, status="not_enrolled") for i in range(SLOTS)]),
              open(f"{PKG}/b9_stage2_subject_slot_manifest.json", "w"), indent=1, default=str)

    rows = []
    # ---- (B) forced-violation validator tests (trial_id-join; on manifest/table COPIES; NOT raw data) ----
    rng = np.random.default_rng(7)
    def relabel_within(arr, by):
        out = np.asarray(arr).copy()
        keys = {}
        for i in range(len(out)): keys.setdefault(tuple(int(b[i]) for b in by), []).append(i)
        for idx in keys.values():
            idx = np.asarray(idx); out[idx] = out[idx][rng.permutation(len(idx))]
        return out
    er_full, _ = _exec_rows(tab)
    drop_imb = (C == HI) & (Yd == 1) & (rng.random(len(C)) < 0.5)
    drop_prior = (((C == HI) & (Yd == 0)) | ((C == LO) & (Yd == 1))) & (rng.random(len(C)) < 0.85)
    violations = {
        "V_missing_table":            ("NONE", er_full),
        "V_flipped_C":                (tab, dict(er_full, C=1 - C)),
        "V_shuffled_Ydesign":         (tab, dict(er_full, Y_design=relabel_within(Yd, (S, MBb, C)))),
        "V_tuple_mismatch":           (tab, dict(er_full, subject=np.roll(S, 1), microblock=np.roll(MBb, 1))),
        "V_microblock_imbalance":     (tab, _exec_rows(tab, ~drop_imb)[0]),
        "V_condition_lock":           (tab, dict(er_full, C=np.array([HI if (int(s) + int(m)) % 2 == 0 else LO for s, m in zip(S, MBb)], int))),
        "V_attrition_prior_shift":    (tab, _exec_rows(tab, ~drop_prior)[0]),
    }
    for name, (pin, er) in violations.items():
        pinned = None if pin == "NONE" else pin
        st, dg = AA.check_contract_acquisition(er, pinned)
        refused = (st == "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND")
        rows.append(dict(analysis="B_forced_violation", case=name, b9_state=str(st), refused_before_pvalue=bool(refused),
                         invalid_reasons=list(dg.get("invalid_reasons", [])), diagnostic_only=True))

    # ---- (A-synth) contract-valid NULL audit at ALERT-CAPABLE n (genuine non-size-gate-trivial null test) ----
    r_full = _certify(tab, er_full, SYNTH_SEED + 1, concept=False)
    # predeclared attrition: 6 whole slots fail quality -> 24 analyzable (balance preserved -> reduced support, NOT invalid)
    keep24 = np.isin(S, np.arange(24))
    er24, _ = _exec_rows(tab, keep24)
    r_attr = _certify(tab, er24, SYNTH_SEED + 2, concept=False)
    rows.append(dict(analysis="A_synth_null_audit", case="full_n30_no_concept", b9_state=str(r_full["b9_state"]),
                     ran_test=bool(r_full["ran_test"]), n_eligible=int(r_full["n_eligible"]), adherence=float(r_full["adherence"]),
                     attrition_fraction=float(r_full["attrition_fraction"]), p_meanT=float(r_full["p_meanT"]), p_stud=float(r_full["p_stud"]),
                     note="genuine (non-size-gate-trivial) null test at alert-capable n; synthetic Z, NOT biological", diagnostic_only=True))
    rows.append(dict(analysis="A_synth_null_audit", case="attrition_n24_no_concept", b9_state=str(r_attr["b9_state"]),
                     ran_test=bool(r_attr["ran_test"]), n_eligible=int(r_attr["n_eligible"]), adherence=float(r_attr["adherence"]),
                     attrition_fraction=float(r_attr["attrition_fraction"]), p_meanT=float(r_attr["p_meanT"]), p_stud=float(r_attr["p_stud"]),
                     note="6-slot whole-subject attrition -> 24 analyzable, balance preserved -> reduced support NOT violation; still >=20", diagnostic_only=True))
    # ---- (C) synthetic positive diagnostic (power) at n=24 and n=30 ----
    rc30 = _certify(tab, er_full, SYNTH_SEED + 11, concept=True)
    rc24 = _certify(tab, er24, SYNTH_SEED + 12, concept=True)
    rows.append(dict(analysis="C_synth_positive", case="n30_concept", b9_state=str(rc30["b9_state"]), ran_test=bool(rc30["ran_test"]),
                     n_eligible=int(rc30["n_eligible"]), p_meanT=float(rc30["p_meanT"]), p_stud=float(rc30["p_stud"]),
                     note="synthetic injected boundary; diagnostic power only, NOT biological", diagnostic_only=True))
    rows.append(dict(analysis="C_synth_positive", case="n24_concept", b9_state=str(rc24["b9_state"]), ran_test=bool(rc24["ran_test"]),
                     n_eligible=int(rc24["n_eligible"]), p_meanT=float(rc24["p_meanT"]), p_stud=float(rc24["p_stud"]),
                     note="power at the hard-minimum analyzable cohort", diagnostic_only=True))
    # ---- (A) real biological null audit -> PENDING ----
    rows.append(dict(analysis="A_real_null_audit", case="PENDING_ACQUISITION", b9_state="PENDING_ACQUISITION",
                     note="requires governance-approved acquisition of a >=24-analyzable cohort under the pinned table; NOT run", diagnostic_only=True))

    # ---- OPERATIONS-ONLY CHECKPOINT demo (6-10 subjects): logistics ONLY, scientific endpoints BLINDED ----
    keep8 = np.isin(S, np.arange(8)); er8, _ = _exec_rows(tab, keep8)
    cp_ok = OC.checkpoint_report(er8, tab)                                 # adhering partial cohort -> contract VALID at checkpoint
    cp_bad = OC.checkpoint_report(dict(er8, C=1 - er8["C"]), tab)          # flipped C -> contract INVALID at checkpoint
    checkpoint_blind_ok = not (set(cp_ok) & set(OC.FORBIDDEN_FIELDS)) and all(k in OC.ALLOWED_FIELDS or k == "BLIND_NOTE" for k in cp_ok)
    rows.append(dict(analysis="D_operations_checkpoint", case="valid_8subj", checkpoint_report=cp_ok, blind_ok=bool(checkpoint_blind_ok), diagnostic_only=True))
    rows.append(dict(analysis="D_operations_checkpoint", case="flippedC_8subj_contract_invalid", checkpoint_report=cp_bad, diagnostic_only=True))

    with open(f"{PKG}/b9_stage2_rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, default=str) + "\n")

    b_ok = all(r["refused_before_pvalue"] for r in rows if r["analysis"] == "B_forced_violation")
    tables = dict(
        scope="B9.2 prospective-acquisition PACKAGE (pre-acquisition): protocol freeze + hash-pinned 30-slot table + pipeline verification on SYNTHETIC Z at ALERT-CAPABLE n. NO real data, NO biological claim, NOT validation, NO tag.",
        cohort=dict(enrollment_slots=SLOTS, target_analyzable=TARGET_ANALYZABLE, hard_min_analyzable=MIN_ANALYZABLE,
                    min_confirm_pairs=20, note="min_confirm_pairs UNCHANGED; >=24 target reserves for attrition"),
        pilot_table=dict(sha256=open(f"{PKG}/b9_stage2_assignment_table.csv.sha256").read().split()[0], n_trials=len(C)),
        analysis_B_forced_violation=dict(all_refused=bool(b_ok), cases={r["case"]: r["b9_state"] for r in rows if r["analysis"] == "B_forced_violation"}),
        analysis_A_synth_null_audit=dict(full_n30=dict(state=r_full["b9_state"], n_eligible=int(r_full["n_eligible"]), ran_test=bool(r_full["ran_test"])),
                                         attrition_n24=dict(state=r_attr["b9_state"], n_eligible=int(r_attr["n_eligible"]), attrition=round(float(r_attr["attrition_fraction"]), 3))),
        analysis_C_synth_positive=dict(n30=dict(state=rc30["b9_state"], n_eligible=int(rc30["n_eligible"])),
                                       n24=dict(state=rc24["b9_state"], n_eligible=int(rc24["n_eligible"]))),
        analysis_A_real_null_audit="PENDING_ACQUISITION (governance-gated; not run)",
        operations_checkpoint=dict(blind_ok=bool(checkpoint_blind_ok), exposed_fields=sorted(cp_ok.keys()),
                                   note="6-10 subject checkpoint sees ONLY logistics (adherence/attrition/support/sampler-feasibility/contract-invalid-reasons); alert/p-values/observed_T/T_z NEVER computed"),
        RESOLVES_B9_1A_SIZE_GATE=("At the >=24-analyzable cohort the contract-valid null audit is a GENUINE non-size-gate-trivial null test (n_eligible>=20) and the injected positive CAN alert -- unlike the size-gate-trivial 6-10 pilot. min_confirm_pairs UNCHANGED (no gate tuning)."),
        governance="Human-EEG collection gated on institutional governance; NO data collected here. Not legal/IRB advice.",
        next="After governance approval, acquire >=24 analyzable subjects under the pinned table, then run analysis A. Early 6-10 checkpoint is operations-only. If only 12-18 analyzable: INSUFFICIENT_LABELS_OR_SUPPORT -> acquisition-budget redesign, do NOT relax the gate.")
    json.dump(tables, open(f"{PKG}/b9_stage2_tables.json", "w"), indent=1, default=str)
    json.dump({r["case"]: dict(state=r["b9_state"], reasons=r["invalid_reasons"], refused=r["refused_before_pvalue"])
               for r in rows if r["analysis"] == "B_forced_violation"} | dict(all_refused_before_pvalue=bool(b_ok)),
              open(f"{PKG}/b9_stage2_forced_violation_checks.json", "w"), indent=1, default=str)
    json.dump(dict(scope="B9.2 pipeline-verification invariants (synthetic Z; machinery only, NO biological claim)",
                   B_forced_violations_all_refuse=bool(b_ok),
                   A_synth_null_full_n30_no_alert=bool(str(r_full["b9_state"]) != "B9_CONCEPT_ALERT" and int(r_full["n_eligible"]) >= 20),
                   A_synth_null_attrition_n24_no_alert=bool(str(r_attr["b9_state"]) != "B9_CONCEPT_ALERT" and int(r_attr["n_eligible"]) >= 20),
                   C_synth_positive_n24_alert=bool(str(rc24["b9_state"]) == "B9_CONCEPT_ALERT"),
                   C_synth_positive_n30_alert=bool(str(rc30["b9_state"]) == "B9_CONCEPT_ALERT"),
                   checkpoint_blinded_no_scientific_endpoint=bool(checkpoint_blind_ok),
                   attrition_reduced_support_not_violation=bool(str(r_attr["b9_state"]) != "CONTRACT_INVALID_OR_OUT_OF_ESTIMAND"),
                   min_confirm_pairs_unchanged=20, A_real_null_audit="PENDING_ACQUISITION"),
              open(f"{PKG}/b9_stage2_contract_checks.json", "w"), indent=1, default=str)
    print(f"B9.2 build: B_all_refuse={b_ok} | A_null full_n30={r_full['b9_state']}(nelig{r_full['n_eligible']}) "
          f"attr_n24={r_attr['b9_state']}(nelig{r_attr['n_eligible']},attr{r_attr['attrition_fraction']:.2f}) | "
          f"C_pos n24={rc24['b9_state']} n30={rc30['b9_state']} | checkpoint_blind={checkpoint_blind_ok} | A_real=PENDING")
    print("B9_STAGE2_BUILD_OK")


if __name__ == "__main__":
    main()
