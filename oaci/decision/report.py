"""Render a per-level K1/K2 decision summary from the artifact decision payloads (reused by the C8 report)."""
from __future__ import annotations


def _f(x, nd=4) -> str:
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def render_decision_summary(per_level, *, title="OACI pre-registered decision (K1/K2)") -> str:
    """``per_level`` = ``[{level, k1: <k1.json body>, k2: <k2.json body>}, ...]``."""
    L = [f"# {title}", ""]
    for e in sorted(per_level, key=lambda x: int(x["level"])):
        k1, k2 = e["k1"], e["k2"]
        L += [f"## level {int(e['level'])}",
              f"- **K1** {k1['k1_status']} — Δ audit_ucl(OACI−ERM) = {_f(k1.get('observed_delta'))}, "
              f"p_lower = {k1.get('p_lower')}, alpha = {k1.get('alpha')}, n_perm = {k1.get('n_permutations')}, "
              f"plan = `{str(k1.get('permutation_plan_hash'))[:12]}`",
              f"- **K2** {k2['k2_status']}"
              + (f" — {k2['reason']}" if k2.get("reason") else "")
              + (f" (reproduced: {k2['reproduced_endpoints']})" if k2.get("reproduced_endpoints") else ""),
              ""]
    return "\n".join(L)
