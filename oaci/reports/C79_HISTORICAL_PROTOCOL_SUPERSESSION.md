# C79 Historical Protocol Supersession

The historical protocol SHA-256
`7732986513793725d58933d487f5bc8f4fc68bfad0857bb4734a450b41ca5dd4`
remains its valid content identity.  It is not an execution identity.

The pre-outcome artifact at `23f549d` is explicitly
`SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED`.  The adaptive generator was transparently
committed at `e561a15`, but the final protocol was materialized after C78S outcomes,
selected H3/H4/H5 through `active_after_Holm`, bound only 2/16 registry categories,
and first appeared with result commit `43a046c`.

No historical file or commit is deleted, renamed, amended, or overwritten.  The
historical protocol is additively marked:

```text
pre_C78S_confirmation_valid = false
seed4_execution_authority = false
superseded_for_future_execution = true
history_rewritten = false
```

The replacement is a distinct post-seed-3 protocol prospective only to the seed-4
checkpoint/outcome field.  Its hash must be used in any future C79E authorization;
the historical hash must not be reused.
