# Claim-language audit

The safety rail for the manuscript. Every headline must stay inside the "allowed" column; the "forbidden"
column lists over-claims the committed evidence does not support; the risky→safer table gives drop-in
replacements. The final section maps each adversarial caveat (from the C15 reviewer-objection matrix) to where
it must appear in the draft.

## Allowed claims (supported by committed numbers)

- "On BNCI2014-001 LOSO with ShallowConvNet, OACI's selection-time leakage reductions do not become
  multiplicity-stable held-out audit evidence." (C8 K1: 11 nominal, 0 BH / 54; C10a Δsel −0.326 vs Δaudit +0.008.)
- "The pre-registered K2 endpoint test returns no reproducible worst-domain gain." (C8 K2 stop.)
- "A non-deployable source-audit oracle cannot identify a gain-reproducing checkpoint in OACI's trajectory
  *from held-out source signal*." (C10b case C; identity 216/216, 0 flips.)
- "Under the tested SRC configuration (seed 0, n=6), improving the source worst-domain endpoint anti-transfers
  to the target." (C12 ATI 1.0, pearson −0.947.)
- "The reusable contribution is a falsification battery; OACI and SRC are case studies falsified under this
  protocol; support-aware leakage is retained as measurement." (C14 verdict; method closure.)

## Forbidden claims (evidence does not support)

- ✗ "Domain generalization fails." / "EEG transfer is impossible."
- ✗ "Support-aware invariance is useless." / "OACI is mathematically wrong."
- ✗ "No good checkpoint exists in the trajectory." (Only: none rescuable by the source-audit oracle.)
- ✗ "OACI cannot reduce leakage." (It reduces selection leakage 54/54; the point is non-transfer.)
- ✗ "Every DG penalty / every source-robust objective anti-transfers." (Only the tested SRC configuration.)
- ✗ "BNCI2014-001 exhibits / demonstrates support mismatch." (Not quantified; balanced 4-class.)
- ✗ "The battery is validated / reusable across settings." (Instantiated once; discriminative validity unshown.)

## Risky phrase → safer replacement

| risky phrasing | safer replacement |
|---|---|
| "OACI failed" / "we show OACI is bad" | "the battery localizes where OACI's control hypothesis breaks under this protocol" |
| "the oracle cannot rescue OACI" | "no checkpoint is rescuable by the tested held-out source-audit oracle" |
| "source robustness hurts the target" | "under the tested SRC configuration, the source endpoint objective anti-transfers" |
| "leakage reduction is meaningless" | "selection-time leakage reduction alone is insufficient evidence of a transferable control mechanism" |
| "OACI improves worst-domain NLL" (as a win) | "worst-domain NLL improves on average but not reproducibly; the pre-registered both-endpoint K2 verdict is stop" |
| "under support mismatch" (in a claim of empirical fact) | "the support-aware construction is motivated by support mismatch; we do not quantify mismatch on this dataset" |
| "a reusable battery" (as demonstrated) | "a battery whose reuse and discriminative validity are future work" |
| "ATI = 1.0 / anti-transfer is a law" | "ATI = 1.0 over 6 seed-0 cells; a stress replication, not a law" |

## Where each adversarial caveat appears in the draft

| C15 caveat | must appear in |
|---|---|
| Support mismatch not quantified on BNCI2014-001 (balanced MI) | Methods (Problem setting, scope note); Limitation 2; Introduction (final paragraph) |
| "Ill-posed under mismatch" is a premise, no naive-vs-support-aware contrast | Methods (scope note); Limitation 3 |
| `L_Q^ov` is probe-relative (fixed probe family / prior) | Methods; Limitation 4 |
| Oracle is source-audit, not target; "no rescuable checkpoint" not "no checkpoint exists" | Results R3 (interpretation); Methods (G4); Limitation 5; audit "safer replacement" |
| Do not cherry-pick bAcc; worst-domain NLL improves-but-not-reproducibly | Results R2 (interpretation); Discussion (what is closed); audit table |
| SRC anti-transfer seed-0, n=6, no λ/lr sweep; guard NLL ≈ 0.09 = memorization | Results R4 (interpretation); Limitation 6; Abstract ("under the tested configuration") |
| No positive control / discriminative validity (battery only ever "falsified") | Methods (battery verdict paragraph); Discussion; Limitation 7 |
| One dataset / one backbone / minimum-seed / paused protocol | Methods (protocol); Limitations 1, 8; Abstract/Intro scoping |

## Standing rule
If a sentence would still read as true after deleting "under this protocol / on BNCI2014-001 / under the
tested configuration," it is probably an over-claim — rescope it before it ships.
