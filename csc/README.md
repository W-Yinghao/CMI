# csc — Falsifiable Concept-Shift Certificates with Abstention

**When is EEG concept shift detectable? Conditional-MI certificates with abstention.**

A self-contained research package (it does **not** import or mutate the AAAI `cmi/` or the
`h2cmi/` packages — the three coexist). It runs entirely on a controllable shift simulator,
so every component is exercised by a fast unit test.

This direction is the **identifiability / counterexample phase** the project's Evidence
Ledger named as the way forward after the gate line closed
([`notes/EVIDENCE_LEDGER.md`](../notes/EVIDENCE_LEDGER.md)). The A0 work proved that
density / CMI scores are *anti-aligned* with adaptation harm
([`notes/A0_FALSIFICATION_FROZEN.md`](../notes/A0_FALSIFICATION_FROZEN.md)); csc turns that
negative result into a positive, falsifiable object — a certificate that **abstains where
detection is provably impossible**.

## The one idea

You **cannot** build a universal concept-shift detector from unlabeled target data: a
pure conditional shift (`P(Y|Z)` changes, `P(Z)` fixed) is invisible to any `Z`-only
detector (proof in [`THEORY.md`](THEORY.md) §1.1). So instead of a binary detector, csc
emits a **three-state certificate**:

| state | meaning |
|---|---|
| `COVARIATE_ADAPTABLE` | visible marginal shift inside the covariate atlas; the source proved the boundary is stable there → adaptation is in scope |
| `CONCEPT_SUSPECT` | visible shift aligned with where the source boundary *did* move, and the source residual test is significant |
| `UNIDENTIFIABLE` | **abstain** — invisible / out-of-atlas / ambiguous / no valid concept atlas |

It never returns a positive safety verdict for a shift it cannot see. Being wrong by
over-abstaining (low power) is allowed; being wrong by **false-certifying safety** is not.

## Layout

```
csc/
  THEORY.md              formal framework: taxonomy, impossibility result, T statistic, certifier
  PREREGISTRATION.md     data design + falsification / termination (kill) criteria
  sim/shift_simulator.py controllable generator: clean / covariate / boundary-coupled / pure-conditional
  certificate/
    residual_test.py     cross-fitted, permutation-calibrated residual-decoder test T + support-graph gate
    atlas.py             source shift atlas (covariate vs concept directions, between-domain spread)
    certifier.py         the three-state certificate + scoring maps (ACCEPTABLE / FORBIDDEN)
  run_synthetic.py       multi-seed eval: the two pre-registered numbers (false-cert rate, power)
  tests/                 test_validity_gate / test_null_calibration / test_power
```

## Quickstart

```bash
# component smoke tests (each module is runnable):
conda run -n icml python -m csc.sim.shift_simulator
conda run -n icml python -m csc.certificate.residual_test
conda run -n icml python -m csc.certificate.atlas
conda run -n icml python -m csc.certificate.certifier

# correctness unit tests:
conda run -n icml python -m csc.tests.test_validity_gate
conda run -n icml python -m csc.tests.test_null_calibration
conda run -n icml python -m csc.tests.test_power

# the pre-registered evaluation:
conda run -n icml python -m csc.run_synthetic --seeds 30 --n_perm 100
```

## Synthetic result so far (12 seeds; see PREREGISTRATION §3)

```
                                COVARIATE  CONCEPT  UNIDENTIFIABLE
clean            (NONE)                 0        0        12
covariate        (COVARIATE)          11        1         0
boundary_coupled (CONCEPT_VISIBLE)     0       11         1
pure_conditional (CONCEPT_INVISIBLE)   0        0        12

false-certification on INVISIBLE concept shift : 0.000   (strict 0.000)
power on VISIBLE concept positive control       : 0.917
false concept alarm on CLEAN                    : 0.000
```

The invisible-shift guard is the headline: a relabel-only target (`Z` byte-identical to
clean) is **always** `UNIDENTIFIABLE` — a naive low-marginal-shift = "safe" rule would
false-certify it. That is the abstention the impossibility result demands, working.

## Scaling to real EEG

`certify(atlas, source_test, Z_target)` and `residual_decoder_test(Z, Y, D)` take ordinary
`(Z[n,d], y, D)` arrays. Point them at the AAAI loaders (`cmi/data/*.py`) or the audited
deployment-encoder dumps (the A0 `erm:0 = CITA-no-LPC` embeddings) to run the real-data
conditions in PREREGISTRATION §2 — **PD medication ON/OFF** (the positive control) and
**SCZ/PD cross-site** (the real null). Those, not the simulator, are the actual test.

## Status / honesty

This is a **research implementation on a simulator**: it demonstrates the framework is
correct, calibrated, and composes, and that the impossibility-driven abstention works. It
is **not** a real-EEG result. The thresholds are hand-set on synthetic and need
leave-one-source-domain-out calibration (PREREGISTRATION §4); the atlas is mean-shift only
(higher-moment concept shifts currently return `UNIDENTIFIABLE`, which is safe but
low-power). The direction is **killed** if it cannot control false-certification on real
invisible/null shift, or has no power on PD ON/OFF — see PREREGISTRATION §5.

Naming discipline (binding, per the Evidence Ledger): the source statistic is "extractable
conditional domain information" / a residual-decoder concept-evidence test — **not** "precise
CMI". The certificate is a diagnostic with abstention, **not** a "safety gate".
