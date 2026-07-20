# C79P Post-Seed-3 Protocol Timing Audit

## Prospective object

The replacement protocol is explicitly designed after C78S outcomes.  Its protected
prospective object is every seed-4 learned parameter, checkpoint, optimizer state,
logit, representation, projection, cache, and model-specific target outcome.
The raw BNCI2014-001 trials and labels are not newly held out.

Before the replacement protocol commit:

```text
seed4 EEG loads through training/instrumentation:  0
seed4 training/forward/re-inference jobs:           0
seed4 checkpoints or optimizer states:              0
seed4 logits/z/Wz caches:                            0
seed4 model-specific target outcomes:                0
seed4 label-view provisioning:                       0
same-label oracle openings:                          0
```

Evidence is recorded in `c79p_tables/seed4_untouched_audit.csv` and descends from
the accepted C79 Mode-R boundary at `2ea4ec3`.

Required order:

```text
2ea4ec3 Mode-R acceptance
< replacement protocol commit and push
< mechanical implementation commit
< field-generation execution-lock commit
< scientific-analysis execution-lock commit
< future direct PI authorization
< first seed4 EEG load or Slurm submission
< first seed4 model-specific outcome access
```

The commit containing this file is the operative protocol commit.  It is bound by
full hash in the later execution locks; a Git commit cannot self-contain its own
hash.  No seed-4 execution is authorized by this timing audit.
