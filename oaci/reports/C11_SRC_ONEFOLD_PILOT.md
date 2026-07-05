# C11c — SRC one-fold pilot (BNCI2014_001 target-001 seed-0)

> method-polishing pilot; NOT a confirmatory conclusion. K1 leakage is MEASUREMENT ONLY.

- target_fit_ids empty (all levels): **True** · SRC selector read target: **False** (must be False) · pilot deep-verify: **True**

| level | method | tgt worst bAcc | tgt worst NLL | ΔK2 bAcc(vsERM) | ΔK2 NLL | ΔK1 leak(meas) | risk-feasible |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | ERM | +0.5104 | +1.2786 | — | — | — | — |
| 0 | OACI | +0.4410 | +1.2444 | -0.0694 | -0.0342 | -0.0144 | True |
| 0 | SRC | +0.4653 | +2.4487 | -0.0451 | +1.1701 | +0.0421 | True |
| 1 | ERM | +0.4983 | +1.2208 | — | — | — | — |
| 1 | OACI | +0.4358 | +1.3647 | -0.0625 | +0.1439 | +0.0036 | True |
| 1 | SRC | +0.4983 | +1.2208 | +0.0000 | +0.0000 | +0.0000 | True |

## SRC selector behaviour

- level 0: reason=source_endpoint_best, fallback_ERM=False, feasible=33, guard_pass=13, selector roles read=['meta', 'source_guard', 'source_risk'], target_read=False
- level 1: reason=erm_fallback, fallback_ERM=True, feasible=35, guard_pass=0, selector roles read=['meta', 'source_guard', 'source_risk'], target_read=False

## C11c signal

- risk-feasible all levels: **True**
- target worst bAcc improves (all levels): **False**
- target worst NLL not worse (all levels): **False**
- SRC fell back to ERM in **1/2** levels
- **SRC shows signal: `False`**

> SRC shows NO one-fold signal here -> source-only endpoint optimization does not transfer under this protocol; consider measurement-only / negative-result direction rather than another DG penalty