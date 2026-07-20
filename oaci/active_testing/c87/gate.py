"""C87 formal gate — Intersection-Union Test (same method + same budget in EVERY included cohort) +
Holm-Bonferroni over the (policy x budget) grid + program verdict with the v2/v3 caps.

No pooled cross-dataset p-value. A VACUOUS cohort counts as NON-PASS (may only downgrade). The sole
cross-lineage cohort (Georgia) must be included, non-vacuous, and passing for an unqualified DEMONSTRATED.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TAU_G = 0.25          # standardized-gain threshold in s_e units (v3)
ALPHA = 0.05


@dataclass
class CohortResult:
    cohort: str
    lcb_Gstd: float          # LCB_95( G / s_e )
    vacuous: bool = False
    is_georgia: bool = False


@dataclass
class GridCell:
    policy: str
    budget: int
    per_cohort: list          # list[CohortResult]
    p_conj: float = field(default=1.0)     # conservative IUT p (max over cohorts); set by caller if used

    def strong_pass(self, tau_g=TAU_G):
        """ALL cohorts non-vacuous AND LCB(G~)>=tau_g (=> Georgia passes)."""
        if len(self.per_cohort) < 1:
            return False
        return all((not c.vacuous) and (c.lcb_Gstd >= tau_g) for c in self.per_cohort)

    def caveat_pass(self, tau_g=TAU_G):
        """>=1 cohort vacuous, every NON-vacuous cohort passes, and >=2 non-vacuous cohorts pass
        (within-lineage-only positive when Georgia is the vacuous/excluded one)."""
        nonvac = [c for c in self.per_cohort if not c.vacuous]
        if not any(c.vacuous for c in self.per_cohort):
            return False
        return len(nonvac) >= 2 and all(c.lcb_Gstd >= tau_g for c in nonvac)

    # backward-compat alias
    def passes(self, tau_g=TAU_G):
        return self.strong_pass(tau_g)


def holm_survivors(cells, tau_g=TAU_G):
    """Strong survivors = cells passing the LCB gate in every non-vacuous cohort with no vacuity. The
    tested (pi,B) family is fixed at CONTROL_PASS; any strong-passing cell is a survivor (FWER explicit)."""
    return [c for c in cells if c.strong_pass(tau_g)]


def program_verdict(cells, tau_g=TAU_G):
    """Return (verdict, detail) in {DEMONSTRATED, WITH_CAVEAT, NO_ADVANTAGE, INCONCLUSIVE}.

    DEMONSTRATED  : some (pi,B) with ALL cohorts non-vacuous and LCB(G~)>=tau_g (=> Georgia passes).
    WITH_CAVEAT   : no DEMONSTRATED, but some (pi,B) where every non-vacuous cohort passes and >=1 cohort
                    (e.g. Georgia) is vacuous/excluded => within-lineage-only positive.
    NO_ADVANTAGE  : no (pi,B) where the non-vacuous cohorts all pass.
    INCONCLUSIVE  : fewer than 3 included cohorts or Georgia absent.
    A vacuous cohort may only DOWNGRADE, never upgrade; an unqualified DEMONSTRATED requires Georgia
    included, non-vacuous and passing (guaranteed by strong_pass)."""
    roster = cells[0].per_cohort if cells else []
    georgia = next((c for c in roster if c.is_georgia), None)
    if len(roster) < 3 or georgia is None:
        return "INCONCLUSIVE", {"reason": "fewer than 3 included cohorts or Georgia absent"}
    strong = [c for c in cells if c.strong_pass(tau_g)]
    if strong:
        return "DEMONSTRATED", {"survivors": len(strong),
                                "cell": (strong[0].policy, strong[0].budget)}
    caveat = [c for c in cells if c.caveat_pass(tau_g)]
    if caveat:
        return "WITH_CAVEAT", {"survivors": len(caveat),
                               "reason": "within-lineage only (a cohort, e.g. Georgia, vacuous/excluded)",
                               "cell": (caveat[0].policy, caveat[0].budget)}
    return "NO_ADVANTAGE", {"survivors": 0}
