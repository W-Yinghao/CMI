"""B9.0 DRY-RUN (development-only; implementation/contract-feasibility ONLY; NO scientific evidence, NO power claim, NO
method-success claim). Builds a SMALL SYNTHETIC toy substrate (NOT Lee2019, NOT validation) with a pre-recording
assignment table, then runs b9_certify over 3 VALID + 6 INVALID (+1 out-of-estimand) worlds. Verifies: VALID_* run the
exact null and emit well-formed states; every INVALID/out-of-estimand world is REFUSED BEFORE any p-value (ran_test False);
states are disjoint and drawn from the declared set. Fail-loud."""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")
from csc.b9 import randomization_table as RT
from csc.b9 import state_machine as SM

HI, LO = RT.HI, RT.LO
D = 10               # synthetic feature dim
BASE_SEED = 990_000_000   # dry-run only; NEVER a real acquisition seed


def _make_Z(Y_design, C, rng, concept=False):
    """Synthetic Z: a boundary for Y_design (so Y_design is decodable); C has NO marginal effect. If concept=True, the
    Y_design boundary is ROTATED in C=HI trials (a genuine C x Z interaction) so the exact null CAN detect it."""
    n = len(Y_design)
    w = np.zeros(D); w[0] = 1.0                       # base Y_design boundary along axis 0
    wr = np.zeros(D); wr[1] = 1.0                     # rotated boundary along axis 1
    sgn = np.where(np.asarray(Y_design) == 1, 1.0, -1.0)
    Z = rng.standard_normal((n, D))
    Z += 1.4 * sgn[:, None] * w[None, :]              # Y_design signal (C=LO and no-concept)
    if concept:
        hi = np.asarray(C) == HI
        Z[hi] -= 1.4 * sgn[hi, None] * w[None, :]     # remove base boundary in C=HI ...
        Z[hi] += 1.8 * sgn[hi, None] * wr[None, :]    # ... and put a rotated one -> C x Z interaction
    return Z


def build_world(world, seed):
    """Return (Z, Y_design, C, subject, microblock, table, natural_prevalence_requested). Synthetic; the assignment table
    is generated BEFORE Z. VALID worlds execute C == table; INVALID worlds deviate (or omit the table)."""
    rng = np.random.default_rng(seed)
    if world == "INVALID_insufficient_support":
        subjects = np.arange(3); n_mb, R = 6, 4       # < n_folds*2 eligible subjects -> INSUFFICIENT
    elif world == "VALID_underpowered_size_gate":
        subjects = np.arange(12); n_mb, R = 6, 4      # crossfit RUNS but n_eligible<20 -> size_ok blocks any ALERT
    else:
        subjects = np.arange(24); n_mb, R = 6, 4
    tab = RT.make_assignment_table(subjects, n_mb, R, seed + 1)
    S, MB, Ct, Yd = tab["subject"], tab["microblock"], tab["C"].copy(), tab["Y_design"]
    C = Ct.copy(); natural = False; table = tab

    if world in ("VALID_balanced_assignment", "VALID_no_concept", "INVALID_insufficient_support"):
        # insufficient_support executes a VALID contract but with too few subjects (<n_folds*2 eligible) -> the exact null
        # cannot run -> INSUFFICIENT_LABELS_OR_SUPPORT (a refuse-before-p state, structurally valid contract)
        Z = _make_Z(Yd, C, rng, concept=False)
    elif world == "INVALID_executed_deviates_from_table":
        # executed C is RE-RANDOMIZED within (subject,microblock,Y_design) -> still balanced + supported + no prior shift,
        # but does NOT follow the pinned table (adherence<1) -> the pre-registration is not binding -> REFUSE (anti-p-hack)
        C = C.copy()
        rng2 = np.random.default_rng(seed + 7)
        keys = {}
        for i in range(len(C)):
            keys.setdefault((int(S[i]), int(MB[i]), int(Yd[i])), []).append(i)
        for idx in keys.values():
            idx = np.asarray(idx); C[idx] = C[idx][rng2.permutation(len(idx))]
        Z = _make_Z(Yd, C, rng, concept=False)
    elif world == "INVALID_post_hoc_ydesign_manifest":
        # C == table (adherent), balanced, supported -- but the manifest ATTESTS the table was NOT generated before
        # recording / Y_design is a post-hoc label -> the B9-vs-B8 provenance floor fails -> REFUSE (the crux differentiator)
        Z = _make_Z(Yd, C, rng, concept=False)
        table = dict(table); table["manifest"] = dict(table["manifest"],
                                                      generated_before_recording=False, Y_design_pre_assignment=False)
    elif world == "VALID_boundary_signal":
        Z = _make_Z(Yd, C, rng, concept=True)
    elif world == "VALID_underpowered_size_gate":
        # a genuine concept but only 12 eligible subjects (< min_confirm_pairs=20): the crossfit RUNS (ran_test True) yet
        # size_ok=False must BLOCK any B9_CONCEPT_ALERT -> exercises the alert-conjunction size gate
        Z = _make_Z(Yd, C, rng, concept=True)
    elif world == "INVALID_executed_ydesign_relabel":
        # executed C == table (C adherent) BUT executed Y_design is RELABELED within (subject,microblock,C) blocks -->
        # per-(subject,microblock) C x Y_design counts (=> balance/prior/support) preserved, only elementwise Y_design
        # deviates from the pinned table -> full-tuple adherence < 1 -> REFUSE (anti-p-hacking on the OTHER factor).
        # Z is generated from the ORIGINAL cue Yd; the relabeled Yd is what an analyst post-hoc claims.
        Z = _make_Z(Yd, C, rng, concept=False)
        Yd = Yd.copy(); rng2 = np.random.default_rng(seed + 11)
        for s in np.unique(S):
            for mb in np.unique(MB[S == s]):
                for c in (LO, HI):
                    idx = np.where((S == s) & (MB == mb) & (C == c))[0]
                    if len(idx) > 1:
                        Yd[idx] = Yd[idx][rng2.permutation(len(idx))]
    elif world == "INVALID_pinned_hash_corrupt":
        # PRESENT table + fully-adherent executed (C,Yd) but the pinned table_hash is CORRUPTED -> hash integrity fails
        # (exercises the present-table hash branch, distinct from the table=None missing branch)
        Z = _make_Z(Yd, C, rng, concept=False)
        table = dict(table); table["manifest"] = dict(table["manifest"], table_hash="deadbeefdeadbeef")
    elif world == "OUT_OF_ESTIMAND_natural_prevalence":
        Z = _make_Z(Yd, C, rng, concept=False); natural = True     # a natural-prevalence request -> refuse
    elif world == "INVALID_missing_assignment_table":
        Z = _make_Z(Yd, C, rng, concept=False); table = None       # no contract
    elif world == "INVALID_cxy_design_imbalance":
        # executed drops ~half of the (C=HI, Y=1) trials -> within-stratum C x Y_design imbalance
        drop = (C == HI) & (Yd == 1) & (rng.random(len(C)) < 0.5)
        keep = ~drop
        S, MB, C, Yd = S[keep], MB[keep], C[keep], Yd[keep]
        Z = _make_Z(Yd, C, rng, concept=False)
    elif world == "INVALID_condition_lock":
        # C constant within each (subject, microblock) -> per-microblock label, no within-block randomization support
        C = np.array([HI if (int(s) + int(mb)) % 2 == 0 else LO for s, mb in zip(S, MB)], int)
        Z = _make_Z(Yd, C, rng, concept=False)
    elif world == "INVALID_session_confounding":
        # C constant within each SUBJECT -> a session/subject label
        C = np.array([HI if int(s) % 2 == 0 else LO for s in S], int)
        Z = _make_Z(Yd, C, rng, concept=False)
    elif world == "INVALID_attrition_prior_shift":
        # non-random attrition: drop most (C=HI, Y=0) and (C=LO, Y=1) -> P(Y_design|C) shifts + imbalance
        drop = (((C == HI) & (Yd == 0)) | ((C == LO) & (Yd == 1))) & (rng.random(len(C)) < 0.8)
        keep = ~drop
        S, MB, C, Yd = S[keep], MB[keep], C[keep], Yd[keep]
        Z = _make_Z(Yd, C, rng, concept=False)
    else:
        raise ValueError(world)
    return Z, Yd, C, S, MB, table, natural


# world -> (kind, has_concept, ok_states, primary_reason). primary_reason (for refuse worlds) MUST appear in
# invalid_reasons -- reasons are NON-EXCLUSIVE (a world may trip several) but the ROOT cause must be present.
WORLDS = {
    "VALID_balanced_assignment":            ("valid", False, ("B9_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE"), None),
    "VALID_no_concept":                     ("valid", False, ("B9_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE"), None),
    "VALID_boundary_signal":                ("valid", True,  ("B9_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE"), None),
    "VALID_underpowered_size_gate":         ("valid", True,  ("NO_ACTIONABLE_CONCEPT_EVIDENCE",), None),  # size_ok must BLOCK the alert
    "OUT_OF_ESTIMAND_natural_prevalence":   ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "natural_prevalence_out_of_estimand"),
    "INVALID_missing_assignment_table":     ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "missing_or_invalid_assignment_table"),
    "INVALID_pinned_hash_corrupt":          ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "missing_or_invalid_assignment_table"),
    "INVALID_post_hoc_ydesign_manifest":    ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "table_not_pre_recording_or_ydesign_post_hoc"),
    "INVALID_executed_deviates_from_table": ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "executed_deviates_from_registered_table"),
    "INVALID_executed_ydesign_relabel":     ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "executed_deviates_from_registered_table"),
    "INVALID_cxy_design_imbalance":         ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "cxy_design_imbalance"),
    "INVALID_condition_lock":               ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "condition_not_randomized_or_locked"),
    "INVALID_session_confounding":          ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "session_confounding"),
    "INVALID_attrition_prior_shift":        ("refuse", False, ("CONTRACT_INVALID_OR_OUT_OF_ESTIMAND",), "attrition_or_noncompliance_prior_shift"),
    "INVALID_insufficient_support":         ("refuse", False, ("INSUFFICIENT_LABELS_OR_SUPPORT",), None),
}


def main(out_path=None, n_boot=200):
    rows = []; fails = []
    print(f"{'world':38s} {'expect':>7} {'state':38s} {'ran_test':>8} {'reasons'}")
    for w, (kind, concept, ok_states, primary_reason) in WORLDS.items():
        seed = BASE_SEED + list(WORLDS).index(w)
        Z, Yd, C, S, MB, table, natural = build_world(w, seed)
        r = SM.b9_certify(Z, Yd, C, S, MB, table, seed=seed, n_boot=n_boot, natural_prevalence_requested=natural)
        st = str(r["b9_state"])
        rows.append(dict(world=w, world_kind=kind, has_concept=concept, seed=int(seed), b9_state=st,
                         contract_valid=bool(r["contract_valid"]), ran_test=bool(r["ran_test"]),
                         invalid_reasons=list(r["invalid_reasons"]), p_meanT=float(r["p_meanT"]),
                         p_stud=float(r["p_stud"]), n_eligible=int(r["n_eligible"]),
                         contract_max_cxy_imbalance=int(r.get("contract_max_cxy_imbalance", -1)),
                         contract_n_support_strata=int(r.get("contract_n_support_strata", 0)),
                         contract_prior_shift=float(r.get("contract_prior_shift", float("nan"))),
                         diagnostic_only=True))
        print(f"{w:38s} {kind:>7} {st:38s} {str(r['ran_test']):>8} {r['invalid_reasons'] if r['invalid_reasons'] else '-'}")
        # invariants (plumbing, NOT science)
        if st not in SM.STATES:
            fails.append(f"{w}: state {st} not in declared set")
        if st not in ok_states:
            fails.append(f"{w}: state {st} not in expected {ok_states}")
        if kind == "refuse" and r["ran_test"]:
            fails.append(f"{w}: REFUSE world reached the p-value (ran_test True) -- contract-first violated")
        if kind == "valid" and not r["ran_test"]:
            fails.append(f"{w}: VALID world did NOT run the exact null (ran_test False)")
        if kind == "valid" and not r["contract_valid"]:
            fails.append(f"{w}: VALID world contract_valid False")
        if primary_reason is not None and primary_reason not in r["invalid_reasons"]:
            fails.append(f"{w}: PRIMARY reason '{primary_reason}' absent from {r['invalid_reasons']}")
        if w == "VALID_underpowered_size_gate":
            # the size gate (n_eligible>=20) must be the binding guard: crossfit RAN, n_eligible<20, and NO alert
            if not (r["ran_test"] and r["n_eligible"] < 20 and st != "B9_CONCEPT_ALERT"):
                fails.append(f"{w}: size_ok gate not binding (ran_test={r['ran_test']} n_elig={r['n_eligible']} state={st})")

    if out_path:
        with open(out_path, "w") as f:
            for r in rows: f.write(json.dumps(r, default=str) + "\n")
    # DISCLOSED coverage gap (design red-team w95gn68da): SAMPLER_INVALID and the contract-level nsup<min_support branch
    # are reachable DEFENSIVE fail-closed states that a well-formed synthetic VALID contract cannot naturally trigger --
    # a balanced contract guarantees non-degenerate cross-fit (the null holds Y_design fixed, so ninv~0) and support scales
    # with the subjects needed for eligibility. They are verified reachable by code inspection and guard real-data
    # pathologies in B9.1; the synthetic panel does not manufacture a fake degeneracy to hit them.
    print("\n  [disclosed] SAMPLER_INVALID + contract-nsup<min_support: reachable defensive states NOT synthetically "
          "triggerable by a valid balanced contract (guard real-data pathologies in B9.1); not manufactured here.")
    print("\n=== B9.0 DRY-RUN verdict (plumbing only; NO scientific claim) ===")
    if fails:
        print("DRYRUN_FAILURES:")
        for f in fails: print("  " + f)
        print("B9_0_DRYRUN_FAIL"); sys.exit(1)
    print("B9_0_DRYRUN_OK: VALID worlds run the exact null + well-formed states; every INVALID/out-of-estimand world "
          "REFUSED before any p-value; states disjoint + in the declared set. Implementation-only, NO scientific evidence.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--n_boot", type=int, default=200)
    a = ap.parse_args()
    main(a.out, a.n_boot)
