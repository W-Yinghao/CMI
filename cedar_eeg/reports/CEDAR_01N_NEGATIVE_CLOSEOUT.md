CEDAR_01N - Negative Closeout

Status: CEDAR_01 real frozen-latent shadow audit is accepted as a real EEG
negative / diagnostic-only readout. This closeout freezes the CEDAR source-only
latent-surgery-to-P1 route. It does not run a new experiment.

Frozen reference

```text
negative readout commit: 78487e8 Add CEDAR_01 real shadow audit readout
feature supply commit:   724c620 Add CEDAR_01F Route C completion readout
dataset:                 BNCI2014_001
backbones:               EEGNetMini, EEGConformerMini
seed:                    0
fold universe:           9 LOSO target folds per backbone
```

Gate summary

```text
feature supply: PASS
real shadow audit: COMPLETE_NEGATIVE
candidate universe: drop_top_1, drop_top_2, drop_top_4
candidate count: 54
accepted: 0
report_only: 54
red-team failures: 0
red-team warnings: 18
deployable artifacts: none
target role: diagnostic-only
```

Required artifacts already exist under:

```text
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/
```

Canonical handoff hash:

```text
03d97199352892bf39afeed3ce826aa185735a766351b82aa2083587621d289c
```

Candidate table hash:

```text
d87e0277747cd07567e8f5310cff2c3bd2eaa6a32cf76f674eda2ede9aa65137
```

Conclusion

CEDAR_01 completed the first real-EEG source-only frozen-latent shadow audit on
BNCI2014_001 using EEGNetMini and EEGConformerMini source-ERM feature dumps.
All 18 feature artifacts passed handoff, schema, hash, and source-view
validation. The fixed candidate universe was evaluated under grouped
conditional leakage probes, permutation nulls, matched random-subspace controls,
and target-quarantined red-team checks. No candidate met the acceptance
criteria: 54/54 candidates were REPORT_ONLY, 0/54 were ACCEPT. Therefore,
CEDAR does not proceed to P1 channel pruning. The result supports a
diagnostic-only conclusion: source-side conditional leakage evidence did not
yield an actionable latent deletion under the frozen CEDAR contract.

Scientific interpretation

This is not an engineering failure. It is a clean negative result:

```text
measurement -> source-only latent mask selection -> actionability
```

The chain did not pass on the approved real EEG frozen latents. CEDAR_00 showed
that the machinery can identify a domain-heavy/task-light latent unit in a
controlled synthetic witness. CEDAR_01 shows that the same red-team contract
rejects all candidates on BNCI2014_001 real frozen features.

Per-backbone conclusion

```text
EEGNetMini: no accepted candidate
EEGConformerMini: no accepted candidate
```

Some folds show measurable leakage reductions, but they do not become
actionable masks under the frozen source-only contract. The correct wording is:

```text
leakage evidence exists but actionability fails
```

Do not describe any candidate as "almost works".

Target diagnostic quarantine

Target diagnostics were computed after source-only selection and did not affect
candidate ranking, acceptance, tie-break, or failure taxonomy. They are retained
only as diagnostic context in:

```text
results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/target_diagnostics_DIAGNOSTIC_ONLY.json
```

The target diagnostics do not justify P1, P2, target comparison narratives, or
generalization claims.

Frozen prohibitions

```text
No P1 channel pruning from CEDAR_01.
No P2 TTA preconditioner.
No candidate universe expansion on this run.
No k sweep.
No utility rewrite.
No target-informed rescue.
No deployable mask artifact.
No generalization or safety-gate claim.
```

Project state

```text
CEDAR as method pipeline: STOP
CEDAR as red-team diagnostic framework: RETAIN
CEDAR as paper positive method: NO
CEDAR as negative evidence for measurement-to-control gap: YES
```

Any future work must be a new protocol with a new hypothesis. It must not be
framed as a CEDAR_01 continuation or a rule change to rescue this run.
