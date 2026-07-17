# OACI EEG-DG Project Memory Through C85T

## Current State

```text
milestone:
  C85T V3

gate:
  C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED

result/report commit:
  53e47126679a38ce70536bd03a4366871031b194

V3 lock commit:
  b1a5ba3aca002de7e302fc375298cc69c1ed82a8

V3 lock SHA-256:
  3ee51a994969ebaaad9c1228d52df76e5222284c38eadbc77a50ce6178cdc8a9

C85T authorization consumed:
  true

C85V authorized:
  false

C85E authorized:
  false
```

C85T V3 is complete. One atomic external bundle contains all exact and Monte
Carlo synthetic outputs and seven proof candidates. Formal theorem statuses
remain `OPEN` pending a separately approved C85V review.

## Empirical Science Remains Unchanged

C84S remains the immutable confirmatory multi-dataset result:

```text
primary:
  C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous

label frontier:
  C84-L4
```

C84A remains a read-only post-scientific audit. C85P, C85R, C85TL, C85TR1,
C85TR2, and C85T are theory/synthetic milestones. They do not alter C84-D,
C84-L4, any empirical selector result, or any field artifact.

## C85 Foundation

C85P prospectively separated:

```text
statistical experiments;
unrestricted information value;
registered-policy value;
policy approximation gap;
realized policy dependence and collapse;
partial identification and minimax regret;
mean, worst-group, and CVaR target risk;
near-optimal action geometry;
costly full-information label testing.
```

C85R repaired the S0-S10 semantic contract before execution. Its operative V2
generator SHA-256 is:

```text
e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a
```

C85TL fixed execution modes, RNG, estimands, S8 rational certificates, and
proof artifact schemas. C85TR1 fixed S9 int64 bytes, intervals, replicate
persistence, single-use authorization, proof-stage separation, and lifecycle
evidence. C85TR2 fixed receipt-validated context creation, one-rename atomic
publication, post-rename recovery, exception precedence, and semantic replay.

Historical V1/V2 execution locks are preserved but non-operative. Never use
`c85t_execute` or `c85t_execute_v2` for an official result.

## C85T Authorization

The fresh direct statement was bound to one V3 authorization:

```text
authorization commit:
  b26b21f6b8378188dd59890c5701944c41fad823

authorization ID:
  9ec012be-dbf2-4f1f-ab99-a5406596c31c

authorization binding SHA-256:
  38b8c3f2111df926c388ba7ab60292aa43714b9f0dace1a2beaa978f30a918fc

authorization file SHA-256:
  b0c283967c7741ebe7eecd0c0207c7dbec7e3f8ccd435db4ae594de41e19e501

consumption receipt SHA-256:
  81651f69513fe5986a47975a5616ede4a2bcab2c82696ab620efc61a9c855d67
```

The external `O_EXCL` receipt is consumed permanently. No retry is authorized.

## Execution Identity

```text
Slurm job:
  899524

state / exit:
  COMPLETED / 0:0

partition / node:
  cpu-high / nodecpu01

requested CPU / allocated CPU / GPU:
  1 / 2 / 0

memory / wall limit / runtime:
  8 GiB / 30 minutes / 7 seconds

attempt ID:
  3ced6e5b117a4a95b9982a5fd22b5e4a
```

No `sacct` evidence was used. Job stderr is empty.

## External Bundle Identity

```text
root:
  /projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3/
    c85t-v3-3ee51a994969ebaa-9ec012bedbf24f1f

files / bytes:
  21 / 1,325,040

result SHA-256:
  ecaff65e942dbb81d93a3bdb61589fa9f1f6590f7188947688e6b30617140cec

manifest SHA-256:
  a727beebcb45598ea0f92f37bed8ef32369b1c793ecad9efc2f5d9941bd5bb0e

semantic receipt SHA-256:
  735edf13a24c074cb3c18e56d168ebd905b3a7bcb29e3c273b3652bb1b7dcc6e

completion receipt SHA-256:
  418f74e4c3cf60847b11bf18a890ffebf870ed8adee1a75d304b01075646e65d
```

The final bundle exists and the staging bundle is absent. All 18 manifest
payload rows replay exact path, size, and SHA. The 12-event lifecycle ends at
`ATOMIC_PUBLISH_COMMIT_READY`.

## Exact Synthetic Results

```text
S0 optimal constant risk:
  1/2

S1 coarse / rich registered / rich unrestricted:
  1/2 / 1 / 0

S2 action divergence; reference / registered risk:
  0; 7/25 / 7/25

S3 selected action / regret / Spearman:
  1 / 0.0050000000000000044 / -0.2571428571428571

S4 top-4 localization / selected regret:
  1 / 4/5

S5 candidate CVaR region:
  (13/20,1), endpoints excluded, proof target only

S6 near-optimal count / Hill-2 / entropy size:
  5 / 4.902454545613457 / 4.950568827806065

S7 near-optimal count / Hill-2 / entropy size:
  1 / 1.000000000027776 / 1.0000000003610865

S8 randomized action / minimax / pure / gain:
  (8/21,1/3,2/7) / 44/105 / 4/5 / 8/21

S9 passive / Neyman allocation:
  51/13 / 18/46

S10 coarse / rich unrestricted / rich registered / reversal:
  11/40 / 0 / 3/5 / 13/40
```

These are synthetic contract results. They are not new empirical EEG results.

## Monte Carlo Freeze

```text
S6 rows:
  4,096

S7 rows:
  4,096

S9 replicate-design rows:
  8,192

S9 raw int64 digest rows:
  4,096

S9 registered digest replays:
  4,096
```

Frozen summaries:

```text
S6 top1 / outside / mean regret:
  0.25048828125 / 0 / 0.0017648925781250015

S7 top1 / outside / mean regret:
  1 / 0 / 0

S9 passive correct-best / regret:
  1 / 0

S9 Neyman correct-best / regret:
  1 / 0

S9 universal active superiority claim:
  false
```

All summaries were reconstructed from reloaded persisted arrays. The benchmark
was not rerun during report collection.

## Proof-Candidate Freeze

```text
T1: PROPOSED_PROOF          / OPEN
T2: PROPOSED_COUNTEREXAMPLE / OPEN
T3: PROPOSED_PROOF          / OPEN
T4: PROPOSED_PROOF          / OPEN
T5: INCOMPLETE_OPEN         / OPEN
T6: PROPOSED_COUNTEREXAMPLE / OPEN
T7: PROPOSED_PROOF          / OPEN
```

The check class is only
`PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY`. C85T did not perform
independent proof review and did not transition any theorem status.

## Verification

```text
semantic replay:
  PASS

scenario results:
  11

formal OPEN statuses:
  7

protected counters zero:
  true

final red team:
  80 / 80 PASS

focused:
  409 passed, 1 deselected

C65:
  1,020 passed, 1 skipped, 4 deselected

C23:
  1,431 passed, 1 skipped, 4 deselected

full OACI:
  2,355 passed, 1 skipped, 4 deselected

accepted stderr:
  empty
```

The extra post-execution deselection is the readiness-only assertion that no
V3 authorization record exists. No lock-bound test was modified.

## Protected Boundary

```text
real project data / EEG / labels / logits:
  0 / 0 / 0 / 0

training / forward / GPU:
  0 / 0 / 0

active acquisition:
  0

theorem-status transitions:
  0

C85V / C85E authorized:
  false / false

new data/model zoo:
  false

manuscript modified:
  false
```

## Reports

```text
overall Markdown SHA-256:
  85c950b7bf63c2691e00aee9342c25067a0844299700b5673d910190d8531cd5

overall JSON SHA-256:
  740552432b838acb8927d83e42431c4897ef238007fee0336ccc1ea3eeb2fd59

result/report commit:
  53e47126679a38ce70536bd03a4366871031b194
```

## Next Boundary

The next possible stage is:

```text
C85V - independent read-only proof-candidate validation
```

C85V requires a new protocol/readiness decision and separate PM approval. It
must not rerun the Monte Carlo benchmark. C85E, active acquisition, real data,
new data/model zoos, and manuscript work remain unauthorized.
