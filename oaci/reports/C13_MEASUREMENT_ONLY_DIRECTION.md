# C13 — Direction memo: measurement-only / source-target instability

**This is a project-direction document, not a manuscript.** It records the pivot forced by C8→C12 and scopes
the next version of the work. It supersedes the "OACI-as-control" framing.

## What was tested, and what failed

Every attempt to use a **source-side signal as a control objective** for BNCI2014-001 worst-domain DG failed
to transfer to the held-out target. Three independent interventions, one estimand (target worst-domain
bAcc/NLL under LOSO):

| line | intervention | result |
|---|---|---|
| **C8** | OACI conditional-domain leakage minimization, native K1/K2 (seeds [0,1,2]) | K1 no multiplicity-surviving reduction; **K2 stop_no_reproducible_gain** |
| **C10** | counterfactual selector replay over OACI's own trajectory (S0–S5), incl. source_audit **oracle** | **case C** — no source-only selector AND not even the oracle recovers K2 gain; a better OACI checkpoint does not exist in the trajectory even by held-out source signal |
| **C11** | SRC endpoint objective (smooth worst-domain balanced CE), one fold | no signal; target NLL blowup at level 0, ERM fallback at level 1 |
| **C12** | SRC across 3 targets × 2 temperatures (12 cells) | **stop** — target NLL blowup 6/12; **source improves ~1 nat while target worsens (anti-transfer)**; τ=0.3 no rescue; level-1 always ERM-fallback |

The C12 anti-transfer table is the crux: SRC **genuinely optimizes** the source worst-domain endpoint
(source_guard NLL down 0.92–1.31 nats) and that improvement makes the **target strictly worse** (target NLL
up 0.13–1.92 nats, past the confidently-wrong threshold). The source-side objective is not merely
uninformative about the target — under this protocol it is **anti-correlated** when optimized.

## Core conclusion

- **Support-aware leakage (`L_Q^ov`) + the K1/K2 machinery remain valuable — as MEASUREMENT and FALSIFICATION
  instruments.** They rigorously *detect* and *quantify* the failure.
- **They must NOT be used as a control objective under the current BNCI2014-001 protocol.** Both
  leakage-control (C8/C10) and endpoint-control (C11/C12) source-side interventions fail to transfer.
- **The scientific object shifts from "a better DG penalty" to "the measurement→control gap and the
  source→target instability itself."** The contribution is characterizing *why* source-side signal does not
  (and here anti-) transfers, and giving a protocol that would have *caught* any such penalty before it was
  believed.

This is consistent with the project-wide accumulation ([[cmi-survivor-audit]], [[cmi-gate-falsification]]):
the measurement is rock-solid; no source-free control mechanism converts it into deployed benefit.

## Next-version directions (choose 1–2; all REUSE the existing stack)

1. **Support-aware leakage AUDIT framework.** Package the support graph + estimable-cell logic +
   grouped-cross-fit `L_Q^ov` probe + K1 grouped-permutation null as a standalone *audit* tool: "how much
   conditional-domain information is extractable from a frozen representation, on estimable cells, with an
   honest permutation null." Deliverable = a measurement API + a falsification report, no training objective.

2. **Source→target instability diagnostics.** Formalize and quantify the transfer failure: per-fold
   correlation between source-side endpoint/leakage deltas and target-side deltas (C10a already shows
   |pearson|<0.14; C12 shows anti-transfer). Deliverable = an instability metric + a diagnostic that predicts,
   from source-only quantities, whether ANY source-side intervention can help (here: no).

3. **Falsification protocol for DG penalties in EEG.** Generalize C7/C8's pre-registered K1/K2 + C10's
   counterfactual selector replay (incl. the oracle) into a reusable *gate*: any proposed DG penalty must pass
   (a) held-out leakage reduction survives multiplicity, (b) reproducible worst-domain K2 gain, (c) even the
   held-out-source oracle finds a target-improving checkpoint. C8/C10/C12 are the first application; it kills
   selector/split "escape hatches" up front. This is arguably the strongest publishable artifact.

4. **Endpoint correlation / transfer-failure analysis.** The empirical study behind (2): across BNCI subjects
   (and later other MOABB datasets), characterize when source worst-domain optimization transfers vs
   anti-transfers, and what data property (support mismatch, subject≈label, sampler-distorted p(D|Y)) predicts
   it. This turns the negative into a positive scientific finding about EEG DG.

**Recommended entry point:** (3) + (2) — the falsification protocol is a concrete, reusable artifact that the
whole C8→C12 arc already instantiates, and the instability diagnostic explains *why* it keeps firing. (1) and
(4) are natural follow-ons.

## Do NOT

- build an OACI-v2 selector, tune the OACI adversary, or invent a THIRD DG control penalty
- run OACI/SRC seeds [3,4] or add BNCI2014_004 under a control hypothesis
- treat C8/C10/C12's weak nominal signals as salvage evidence
- use `source_audit` or `target` in any deployable selector
- jump to a formal manuscript before the direction (audit / instability / falsification-protocol) is scoped

## Immediate next step

Pick the entry point (default (3)+(2)) and scope it as C14: a support-aware **measurement/falsification**
package — no new control objective. The kill-decisions (C11a, this memo) stand; SRC and OACI-control are
closed.
