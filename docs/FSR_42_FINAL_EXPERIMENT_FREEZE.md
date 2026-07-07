# FSR_42 — Final Experiment Freeze (Phase 7 close)

**Project FSR.** PM decision (2026-07-07): after Phase 7C, **all experiments stop.** 7A/7B/7C are accepted;
the cheap-mechanism line is closed. This document is the authoritative experiment-freeze boundary for Paper 2.

## Accepted phases (final)
- **7A** — Recoverability theory (FSR_36/37); merged into paper §08.
- **7B** — Head-only prevalence-reweighting probe (FSR_38/39): **fail-closed**, `gate_pass=false`. Prevalence skew
  induces the subject↔class correlation exactly but is **not even learned** (task signal is a sufficient
  statistic). Ledger **C18** (READY_WITH_CAVEAT).
- **7C** — Head-only task-conflict probe (FSR_40/41): **fail-closed at Q7C-b**. The subject-keyed, task-conflicting
  relabeling (P(y) held exact) is **learnable in-sample** (conflict-subset fit 0.70 vs 0.20 floor) but its
  held-out/target harm **does not exceed a subject-scramble control** (`beats_shuffle=false` on both datasets) and
  is not localized to the subject subspace; it beats only matched random noise. **`weaponization_confirmed=false`**;
  repair gated `none`. Ledger **C19** (READY_WITH_CAVEAT).

## Mechanism bridge (the science that closes the line)
```
7B: prevalence reweighting on true labels        -> not even learned (task sufficient statistic)
7C: subject-keyed task-conflict corruption       -> learnable in-sample, but harm is generic
                                                    subject-blocked corruption, NOT subject-structure-
                                                    specific / transferable reliance
=> neither cheap head-level manipulation turns naturally present subject leakage into a transferable,
   subject-structure-specific harmful reliance. This SPLITS three things that a single leakage score
   conflates: prevalence correlation / task-conflict corruption / transferable subject-structured weaponization.
```

## Frozen boundary — NO MORE EXPERIMENTS
```text
no PC2 GPU refit           (prevalence-stress full-backbone learned-reliance study)
no Lee2019 preset          (the 3rd-dataset add for leave-one-dataset-out)
no new head stress         (no further head-only weaponization variants)
no new repair primitive     (E4/E4b/ERASE/Hreg are the closed set; 4G second-moment = none)
no new backbone training
```

## PC2 — remains PAUSED (future work only)
The pause is now over-determined: (1) 7B — the PC2 prevalence mechanism is **inert** on the cheapest, most
detectable head-only probe; (2) 7C — even a learnable subject-keyed task-conflict corruption does **not** form a
transferable subject-structure-specific reliance; (3) 4G — second-moment / stochastic repair primitives do not
hold; (4) PC2 needs GPU **and** ≥3 preset-ready datasets, and would most likely yield an expensive negative that
is no clearer than the current head-only evidence. Manuscript wording (do **not** write "PC2 planned/next"):
> A full-backbone learned-reliance study (PC2) would require a new mechanism, a stronger information contract, and
> at least three preset-ready datasets; current evidence does not authorize it.

## Firewall / discipline (unchanged, final)
Target-label firewall held throughout (target y only for final scoring; verified clean in 7B/7C). Every phase was
pre-registered + design-red-teamed **before** the run and adversarially verified **after** — this caught real
BLOCKERs in every phase (7C: the Q7C-a-measured-on-held-out gate-collapse; and 3 result-stage over-claims).

## Status
Experiments: **FROZEN.** Next = Paper 2 manuscript freeze (FSR_43) → proofread / submission hardening. Paper 1
(Prior-Decoupled TTA) is independent and unaffected.
