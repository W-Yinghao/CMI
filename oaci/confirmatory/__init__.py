"""Confirmatory protocol layer.

confirmatory_v2.yaml is a DRAFT *protocol* schema (per-dataset blocks, LOSO over all subjects, 5 seeds,
k1/k2 decision rules) -- it is NOT a runnable manifest_v2. This package turns ONE (dataset, held-out
target) into a runnable full-budget manifest_v2 (a `pilot` manifest) and runs that single fold.

Scope of the current step (deliberately narrow): BNCI2014_001 only, target = subject-001, full budget.
A one-fold run is PIPELINE VALIDATION, never confirmatory efficacy evidence.
"""
