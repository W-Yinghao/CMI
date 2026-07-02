"""OACI pre-registered decision layer (C7).

K1 = grouped-permutation held-out leakage null (``k1_permutation`` + ``k1_decision``); K2 = reproducible
multi-seed worst-domain gain (``k2_decision``). ``plans`` reads the strict manifest K1/K2 blocks;
``payloads`` (de)serialises the artifact decision records; ``report`` renders the acceptance report. The
layer is additive and read-only w.r.t. training / selection / audit / prediction / metrics — it consumes
their frozen outputs and never redefines them.
"""
