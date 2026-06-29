"""ACAR v4 (CURB: Control-First Utility/Risk-Budgeted Adaptation Routing).

NON-BINDING / POST-V3 DEV_STOP. This package is post-v3 hypothesis generation: it reuses the v2/v3 estimand
(ΔR_a(B) = R_B(f_a) − R_B(f_0)), action set, cohorts (DEV-only), and provenance machinery, but never edits acar/v3,
acar (v2), the v3 protocol commit/tag (817b04f / acar-v3-dev-design-v1), or any v3 result. The first modules
(frontiers, policies) are pure-numpy and synthetic-capable: they read no real cohort, fit nothing, select nothing,
and freeze nothing. See notes/ACAR_V4_DESIGN_DRAFT.md and notes/ACAR_V4_DEV_EXPLORATION_PLAN.md.
"""
