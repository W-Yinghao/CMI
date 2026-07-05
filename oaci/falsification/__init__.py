"""C14 — EEG-DG Falsification Battery. Support-aware leakage, selector-oracle replay, and source->target
instability diagnostics, packaged as a reusable MEASUREMENT / FALSIFICATION instrument (NOT a control
method). Aggregates the already-committed C8 (K1/K2), C10 (selector replay + oracle) and C12 (SRC
anti-transfer) evidence into six gates + a battery verdict, deciding whether a DG-penalty control hypothesis
is worth continuing. No GPU, no retraining, no target used for selection. Imports only within `oaci`."""
