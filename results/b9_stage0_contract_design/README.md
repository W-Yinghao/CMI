# B9.0 prospective randomized-audit contract design (diagnostic-only, NO scientific claim)

Reviewer-authorized 2026-07-07 after **B8.3 (INSUFFICIENT)** closed the B8 emulator line: post-hoc label balancing on an
observed/generated label `Y` cannot remove the collider. **B9 moves label/class balance + condition randomization INTO the
acquisition contract** — a hash-pinned assignment table generated *before* recording, where `Y_design` is a **pre-assignment
cue** (not the observed/generated `Y` that created B8.3's collider), and natural prior shift → **refuse** (out of estimand),
not repair.

**B9.0 is design + machinery + dry-run ONLY.** It makes **NO scientific claim, NO power claim**, and is **NOT validation**.
The dry-run substrate is a small synthetic toy (not Lee2019). B9.1 (real prospective acquisition, or an existing
genuinely-pre-randomized dataset) is a **separate future authorization**.

## What's here (`csc/b9/` + this package)

- `csc/b9/CONTRACT.md` — the contract spec (estimand in/out, valid/invalid criteria, exact null, states, hard stops).
- `csc/b9/contract_schema.json` — the assignment-contract manifest schema + enforcement + alert conjunction.
- `csc/b9/randomization_table.py` — pre-recording hash-pinned assignment table + **Z/T-blind contract validator**.
- `csc/b9/exact_randomization_null.py` — resample `C*` **only within the predeclared `(subject, microblock, Y_design)`
  randomization set** + recompute the byte-reused B3 contrast (no fitted models).
- `csc/b9/state_machine.py` — **contract-first** certifier (validator refuses before any p-value).
- `csc/b9/dry_run_checks.py` — the synthetic dry-run (4 VALID + 11 refuse worlds).
- `b9_stage0_protocol.json`, `b9_stage0_schema` (in code), `b9_stage0_dryrun_rows.jsonl`, `b9_stage0_contract_checks.json`,
  `b9_stage0_redteam_checks.json`, `SHA256SUMS`, this README.

## The key property: pre-registration binding (the single B9-vs-B8 differentiator)

The whole value of B9 over B8 is that it **requires + verifies a pre-registered acquisition contract** instead of repairing
a null after the fact. Two design red-team rounds (`wcmut149b`, `w95gn68da`) found — and hardened — this from *decorative* to
*enforced*. The validator now enforces (all **Z/T-blind**, **contract-first**):

1. a valid **hash-pinned** table exists + hash integrity;
2. **provenance attestation** — `manifest.generated_before_recording` AND `manifest.Y_design_pre_assignment` must be `True`;
3. **adherence** — the executed **full tuple `(C, Y_design, subject, microblock)`** must match the registered table
   row-for-row (anti-p-hacking on **both** randomized factors — an analyst may relabel neither `C` nor `Y_design`);
4. executed `C×Y_design` balance + no prior shift + randomization support.

**Honest limit:** data-level provenance (was the table *truly* generated before recording; is `Y_design` *truly* the
pre-assigned cue) is **inherently unverifiable from `(C, Y_design, Z)`** — the boolean manifest attestation is the enforceable
*floor*; the real guarantee comes from the **B9.1 acquisition protocol**. And the exact null is collider-free only
*conditional on* genuine within-stratum randomization of the executed trials (a Z-dependent, balance-preserving attrition is
out of reach of a Z-blind check).

## Dry-run result (`B9_0_DRYRUN_OK`, plumbing only)

4 VALID worlds run the exact null with well-formed states (`boundary_signal` → ALERT; `underpowered_size_gate` →
NO_ACTIONABLE via the `n_eligible≥20` gate); all 11 refuse worlds are **refused before any p-value** (`ran_test=False`) with
their primary reason — including the two that isolate the crux fixes (`INVALID_executed_ydesign_relabel`,
`INVALID_post_hoc_ydesign_manifest`) and the coverage worlds (`INVALID_pinned_hash_corrupt`, `VALID_underpowered_size_gate`).
States are disjoint and drawn from the 5 declared. `SAMPLER_INVALID` + the contract-`nsup<min_support` branch are disclosed
as reachable **defensive** states a valid balanced contract cannot synthetically trigger (they guard real-data pathologies
in B9.1). See `b9_stage0_contract_checks.json` for the full world→state→reason map and `b9_stage0_redteam_checks.json` for
the two red-team rounds.

## Next (reviewer decision, NOT authorized)

**B9.1**, in exactly one of two legitimate forms: (A) prospective randomized-audit data acquisition, or (B) an existing
dataset with genuine **pre-recording** `C×Y_design` randomized/counterbalanced assignment. **Lee2019 may serve as a code
dry-run substrate but never as B9 validation.** NOT authorized: B8.4, mean-T/p recalibration, selector/statistic/feature
changes, power frontier, Lee2019-as-validation, confirmatory tag, paper writing. Related: `../b8_stage3_label_balanced_contract/`
(B8.3), `notes/b8_1_class_balanced_contract.md`.
