# C58 - Real-EEG Training Gate

Decision: `TRAINING_NEEDED_BUT_NOT_AUTHORIZED`.

C58 is complete from existing artifacts for finite-population partition bounds and empirical lower-bound candidates. It does not run new real-EEG training, GPU jobs, or re-inference.

Future training would be scientifically justified only if the next approved milestone explicitly asks for split-label cache, atom trace, per-trial logits/probabilities, or independent checkpoint-field replication. BNCI2014_004 and seeds [3,4] remain reserved unless the user explicitly releases them.

Any future campaign must be pre-registered, submitted through Slurm, and quarantined from method tuning or selector construction.
