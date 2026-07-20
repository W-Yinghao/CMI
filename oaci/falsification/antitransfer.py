"""C14 gate G5 — source->target transfer. From C12's SRC stress cells: when a source-side endpoint objective
IMPROVES the source worst-domain endpoint, does the target endpoint improve (transfer), stay flat, or WORSEN
(anti-transfer)? Anti-transfer is the strongest falsification: the control signal is not merely uninformative
about the target — optimizing it harms the target."""
from __future__ import annotations

from .schema import ANTITRANSFER_DETECTED, G5, NO_ANTITRANSFER, gate
from .transfer import instability_metrics, transfer_correlations


def g5_source_target_transfer(c12, c10_part1) -> dict:
    cells = c12.get("cells", [])
    im = instability_metrics(cells)
    corr = transfer_correlations(cells, c10_part1)
    n_blowup = c12["verdict"].get("n_target_nll_blowup", 0)
    n_nontransfer = c12["verdict"].get("n_source_improved_not_transferred", 0)
    detected = (im["n_anti_transfer"] or 0) >= 1 or n_nontransfer >= 1 or n_blowup >= 1
    return gate(G5, ANTITRANSFER_DETECTED if detected else NO_ANTITRANSFER,
                anti_transfer_index_NLL=im["ATI_NLL"], anti_transfer_severity=im["ATI_severity_mean_target_nll_harm"],
                source_target_instability_score=im["source_target_instability_score"],
                n_anti_transfer=im["n_anti_transfer"], n_source_improved=im["n_source_improved"],
                n_active=im["n_active"], n_target_nll_blowup=n_blowup, n_source_improved_not_transferred=n_nontransfer,
                source_nll_to_target_nll_pearson=corr["source_nll_to_target_nll"]["pearson"].get("r"),
                c12_verdict=c12["verdict"].get("verdict"))
