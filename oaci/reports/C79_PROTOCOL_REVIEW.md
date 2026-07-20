# C79 — Seed-4 Locked Confirmation Protocol Review

## Final Mode-R gate

```text
C79_PROTOCOL_OR_TIMING_REPAIR_REQUIRED
```

Primary taxonomy: `C79-F_protocol_timing_provenance_or_isolation_blocker`.

## Decision

The committed C79 JSON is byte-stable and its SHA-256 is correct, but it is not a
prospectively locked strict-confirmation protocol under the C79 timing rule.
The only pre-C78S-outcome C79 artifact is explicitly a non-final skeleton.  The
final JSON was generated at `2026-07-11T10:43:47Z`, after H1 completed at
`10:41:28Z`, H3/H4/H5 at `10:43:26Z`, and H2 at `10:43:27Z`; it was first committed
with C78S result commit `43a046c5ba6632de415bfdbaacfa524d82c5395e`.

More importantly, the generator selected `['H3', 'H4', 'H5']` from
`active_after_Holm`.  That is an outcome-adaptive confirmation scope.  A protocol
hash can prove immutability after creation; it cannot move creation before the
outcomes that informed it.

The generator rule was itself committed before outcomes in
`e561a15865934036bdccbc1e3b2ff126ad84821f`.  This is transparent
and excludes hidden post-outcome code editing.  It still does not satisfy the
handoff's stricter requirement for a final, exact H1-H6 protocol committed before
outcome access: the rule materializes an outcome-filtered H3/H4/H5 scope and the
resulting registry is incomplete.

## Registry completeness

Only `2` of
`16` required exact registry
components are present.  Missing or insufficiently bound components include:

- primary targets and target-4 exclusion
- construction/evaluation trial-ID hashes
- H1 reliability, top-k, regret formulas and gates
- H2 exact model, sign convention, and held-target split
- H3 exact feature block, kernel, bandwidth, and scaling
- H4 exact F2 base model, cross-fit, null, and qualification
- H5 exact F4 base model, cross-fit, null, and qualification
- H6 exact positive-control effect and Holm family ordering
- outer and inner grouping units
- permutation/bootstrap construction and counts
- null RNG streams
- retry and additive-repair policy
- same-label oracle reachability
- success, failure, and claim taxonomy

## Replayed evidence

- C78F and C78S commit chains replay through `ef2a01a4e948143be5eb58c3370142b5eecf7178`.
- C78S protocol SHA-256 replays as `df85699090a65d1e1766d754bcebd9eb5648cc13e4441d8074a3f4884487c7f8`.
- All `22` registered C78S reference values replay within tolerance.
- The C78S regression-provenance correction `dcd4c283573b4cdebe72c8ed3e181403232b28b7` passes 8/8
  independent checks and changed no code, protocol, result, estimand, statistic,
  null, or taxonomy.

## Execution boundary

Mode R stopped at Phase 1.  It did not create a C79 implementation/execution lock
or expected seed-4 manifest because those artifacts would falsely imply that the
protocol review passed.  It performed zero seed-4 EEG loads, jobs, training,
forward/re-inference, GPU work, checkpoints, caches, label-view access, or outcome
reads.  The same-label oracle remained closed.

## Required PM repair

The scientifically clean option is to relabel the next study as a prospectively
locked **post-seed3 seed-4 replication/robustness study**, fully specifying H1-H6
before seed-4 access.  It cannot be described as a C79 protocol that predates C78S
outcomes.  Alternatively, stop the seed-4 campaign.  The current artifact cannot
authorize strict C79 confirmation as written.

No C79 Mode-E authorization should be requested until PM resolves that claim and
protocol category.  No C80, additional seed, BNCI2014_004, feature/kernel search,
oracle analysis, or manuscript work is authorized.

## Regression and final red team

Authoritative CPU regressions on commit `70c31bb` passed:

```text
focused C79:   21 passed
C65-C79:      277 passed, 1 intentional finalized-C78F guard skip
C23-C79:      684 passed, 1 intentional finalized-C78F guard skip
full OACI:  1,612 passed, 1 intentional finalized-C78F guard skip
```

All four stderr logs are empty.  Final-report red team passed `21/21` checks and
confirmed that the report presents a protocol blocker, not execution readiness.
