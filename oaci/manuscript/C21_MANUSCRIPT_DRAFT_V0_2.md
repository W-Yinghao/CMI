# When Source-Side Signals Do Not Transfer: Falsification, Observability, and Estimand Boundaries in EEG Domain Generalization

## Abstract (seed = canonical conclusion)

Under strict target isolation, the tested source-side DG control objectives do not transfer to reproducible target worst-domain improvements. However, the failure is not a simple absence of target-good states: OACI trajectories contain target-accuracy-good checkpoints that are invisible to simple source-audit selectors. Source-only competence information is weakly present in multivariate, deletion-robust observables and can be recovered by a pre-registered low-freedom diagnostic probe in-regime. Yet this signal does not establish stable cross-regime external validation: held-out support-stress regimes reveal that within-target ranking information does not transport as a pooled cross-target competence estimand. The contribution is therefore a falsification and observability framework, not a deployable target-free selector.

## Sections

1. [01_introduction](sections/01_introduction.md) — strict-DG EEG; the measurement->control gap; contribution = falsification + observability + estimand boundary
2. [02_problem_setting](sections/02_problem_setting.md) — target isolation; support graph; overlap-aware leakage; K1/K2; estimands
3. [03_falsification_battery](sections/03_falsification_battery.md) — C14: OACI/SRC control objectives falsified under protocol (C8/C12/C14)
4. [04_mechanism_results](sections/04_mechanism_results.md) — C16: target-good checkpoints exist but source-unobservable; calibration barrier; SRC memorization
5. [05_identifiability_and_probe](sections/05_identifiability_and_probe.md) — C17 weak multivariate identifiability; C18 support-stress + endpoint-estimability; C19 pre-registered in-regime weak positive
6. [06_external_boundary](sections/06_external_boundary.md) — C20 frozen cross-regime validation not established; estimand boundary (within-target vs pooled)
7. [07_discussion_limitations](sections/07_discussion_limitations.md) — single dataset/backbone; diagnostic-only; future work = calibration estimand + external protocol

> STORY LOCKED (C21). No new experiments; every result maps to a committed C14->C20 artifact.