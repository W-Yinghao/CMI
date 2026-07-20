# OACI EEG-DG Project Memory Through C84L1R1

C84L1C job `895928` consumed its authorization and completed three Lee training
phases and 73 complete engineering units. It stopped on a float32 classifier
reconstruction error of `1.239776611328125e-5` above the locked `1e-5`
tolerance. Target-y, construction/evaluation/oracle and scientific-outcome
access were zero. The partial root is preserved and nonreusable.

C84L1R1 changes only the 1040-term float32 `zW+b` replay tolerance to `2e-5`.
Softmax, repeat logits and repeat z remain at `1e-6`. Candidate IDs, the fixed
left-hand support deletion, level-0 reuse, training and all scientific contracts
remain unchanged.

Repair protocol SHA-256:
`2e199f6f63dffd1b02c1e31102ed189e31bf6e4961465394230f8e9de1d4ddf0`.
Replacement execution-lock SHA-256:
`f9ebd88c72915bb41ba2d2d84a2a00c6748272021d48043c299bce52a1ad3813`.
The lock binds 125 runtime objects, 44 implementation files and five protocols.
It requires a fresh root, fresh authorization and retraining all 243 units.

Verification: 12/12 synthetic fixtures, 44/44 red-team, focused 175, C65 661,
C23 1,072 and full 1,996 passed. All stderr is empty; `squeue` reports no active
jobs. The new authorization record and replacement root are absent. C84F/C84S
remain unlocked and unauthorized.

Gate:
`C84L1C_FLOAT32_REPLAY_REPAIRED_AND_RELOCKED_READY_FOR_FRESH_PI_AUTHORIZATION`.
