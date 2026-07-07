# Theorem Ledger

Naming is locked (never `A0…A6` — those collide with the legacy/retracted chain in `notes/theory`).
Every non-identifiability entry ships an exact counterexample in
`counterexamples/run_counterexamples.py`.

| ID | Statement (short) | Regime | Key contracts | Identifiable object | Non-identifiable object | Certificate / proof file | Implementation hook |
|---|---|---|---|---|---|---|---|
| **OA-0** | A target functional `T` is identifiable under `(R,C)` iff it is constant on every compatibility set `K_{R,C}(o_R)`. | any | — (definitional) | — | — | `06_oaci_identifiability.md §5` | audit rule engine `check_claim_allowed` (registry.py) |
| **MONO-1** | If `O_a` refines `O_b` then `K_{a,C} ⊆ K_{b,C}`: identifiability only increases, identified sets only shrink, along `R0 ⊑ R1 ⊑ R2`. Source breadth ≠ target observation. | R0⊑R1⊑R2 | — | (monotone refinement) | target coord not shrinkable by source breadth | `06 §8`, `01 §6`; **CE-MONO-1** | registry R0/R1/R2 checkability is monotone (MONO-1 self-test) |
| **TOS-1** | Under a target-free-coordinate contract set, every non-trivial target functional is non-identifiable under R0. | R0 | target-free coordinate | source law functionals only | target **risk**, **prior**, **concept**, adaptation **gain + sign**, **harm** | `03_tos_source_only_ceiling.md §2`; **CE-R0-1/2/3** | eval_bridge marks R0 target metrics `identifiable_estimand=null` |
| **TU-1** | Under C1∧C2∧C3 the target prior `π_T` is identifiable from the observed mixture `p_T(z)=Σ_y π_T(y)p_ref(z\|y)` (full-column-rank `B`). | R1 | C1, C2, C3 | target prior `π_T` | `π_T` when `B` rank-deficient (→ CE-R1-2) | `04_prior_decoupled_theory.md §5`; **CE-R1-2** (failure) | eval_bridge admits `target_prior` only under C1∧C2∧C3 |
| **TU-2** | The target concept `p_T(Y\|X)` is non-identifiable in R1 even with source + observed `p_T(X)` held equal. | R1 | (holds for any) | — | target concept `p_T(Y\|X)` | `06 §6` table; **CE-R1-1** | audit rejects `target_concept` in R0/R1 |
| **MP-1** | In R2 the transport transform is identifiable under enough valid anchors (C8) and true pairing (C11); too few anchors or fake pairing ⇒ non-identifiable. | R2 | C8, C11 | transport (with C8∧C11) | transport on un-anchored directions (¬C8); true transport under fake pairing (¬C11) | `06 §6`; **CE-MP-1**, **CE-C11-1** | audit requires C8∧C11 for `transport` |
| **PD-1** | After prior decoupling (`Ĩ(Y;D)=0`): `Ĩ(Z;D\|Y) = Ĩ(Y;D\|Z) + Ĩ(Z;D)`, all terms ≥ 0 — no forced encoder-vs-decoder trade-off; driving `Ĩ(Z;D\|Y)→0` forces both others to 0. | source-side (R0) | C7 (source-side reweighting) | additive source-side leakage relation | (NOT target accuracy, NOT concept shift) | `04 §3` | `cmi/hierarchical.py` signed `Î_j`, `align/reference_marginal.py` |
| **ID-1** | Chain-rule identity `I(Z;D\|Y) − I(Y;D\|Z) = I(Z;D) − I(Y;D)` (the safe residue of the legacy "A1"). | source-side | — | exact identity | — | `04 §3`, `00_repo_audit.md` | `eval/leakage.py` cross-fit + within-(Y,Pa) permutation null |

## Notes

- **Target-law axiom caveat.** A contract that fully fixes `P_T` as a function of `P_S` can make target
  functionals computable, but that is a **declared target-law axiom**, not source-only identification
  (`06 §10`, `03 §2` Remark). The audit records such assumptions explicitly; they are never reported as
  R0 evidence.
- **Retraction boundary.** PD-1/ID-1 replace the retracted legacy claims (posterior-KL upper bound,
  zero-Bayes escape, concept-shift-from-`I(Y;D\|Z)`, source-only target prior) — see `h2cmi/THEORY.md`
  P0-2..P0-5 and `04 §1`.
