# C16-C — Battery discriminative validity (synthetic controls)

- **discriminative_validity: `True`** — positive certified 1/1, negative falsified 2/2

| regime | rho | expected | battery | correct |
|---|---:|---|---|:--:|
| positive_transfer | +0.9 | control_hypothesis_supported | control_hypothesis_supported | True |
| decoupled | +0.0 | falsified | falsified | True |
| anti_transfer | -0.9 | falsified | falsified | True |

> The battery certifies a SIMULATED transferring method (positive control) and falsifies decoupled / anti-transfer regimes -> it is not merely a negative-result wrapper. This is a SYNTHETIC feature-level validity check of the battery's decision logic; validating the measurement machinery's sensitivity on real transferring EEG methods remains future work.