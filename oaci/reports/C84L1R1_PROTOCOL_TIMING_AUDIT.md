# C84L1R1 Protocol Timing Audit

## Chronology

1. C84L1P protocol was committed before level-1 real-data access.
2. The C84L1C execution lock was committed at `3eafd70795344c43e0c6326e5c190ecaea4c2934`.
3. Direct authorization was committed at `05bfca1` and consumed by job `895928`.
4. Job `895928` accessed engineering data and stopped before a complete Lee field because the locked float32 linear replay tolerance failed.
5. Target-y access, target scientific metrics, construction/evaluation views, and oracle access remained zero.
6. This C84L1R1 repair protocol is committed before any replacement implementation, lock, authorization, or retry.

## Interpretation

C84L1R1 is an additive post-engineering-failure repair. It does not alter training, models, data views, candidate IDs, the fixed support deletion, or any scientific threshold. It changes only the cross-kernel float32 `zW+b` engineering replay tolerance from `1e-5` to `2e-5`; saved softmax and repeated tensor identity checks remain at `1e-6`.

The failed root is preserved and non-reusable. A replacement canary must retrain all 243 units in a fresh content-addressed root under a new execution lock and fresh direct authorization.
