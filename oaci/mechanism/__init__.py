"""C16 — Mechanism & Discriminative-Validity Deep Dive. Explains the measurement->control decoupling and
source->target anti-transfer WITHOUT proposing a new control objective, and tests whether the falsification
battery has discriminative validity. Real-data analyses read ONLY committed C8/C10/C12 artifacts (no
retraining); the target-oracle ceiling is a POST-HOC, NON-DEPLOYABLE diagnostic (it reads target_audit only
to decide whether a target-good checkpoint EXISTS in the trajectory, never for a deployable selector).
Discriminative-validity controls use deterministic synthetic feature-level simulations. Imports only `oaci`."""
