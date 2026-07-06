# C20-B — External-dataset validation PROTOCOL (document only; NO execution in C20)

This is a protocol skeleton. C20 executes NO second dataset. BNCI2014_004 stays BARRED pending explicit future approval + a dedicated protocol; it is NOT unbarred here.

## Questions the protocol must answer before any C21 execution
- which dataset, and WHY it counts as external (different site/subjects/acquisition, not a re-split);
- what splits (leakage-safe, recording-grouped), and which target labels are used POST HOC only;
- exact frozen artifacts carried over (config hash `664007686afb520f`, robust-core features);
- required artifacts (identity gate, feature availability, abstention, permutation baseline);
- failure modes (feature non-availability vs relationship absence) and a pre-registered STOP RULE;
- forbidden: selector emission, target-metric selection, deployment language.

## Sequencing
C20-A (this) -> C20-B protocol (this) -> C21 external execution ONLY after approval.

## Candidate matrix
See c20_tables/external_dataset_candidate_matrix.csv. BNCI2014_004 = BARRED_pending_explicit_approval.
