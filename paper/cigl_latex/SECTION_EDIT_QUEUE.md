# CIGL — Section Edit Queue (Phase 4J handoff)

> Per-section triage for the PI prose edit. "Claim-boundary risk" = how easily a careless edit here could
> over-claim. No edits are made in this packet; this is a worklist.

| section | current purpose | what to check | likely prose weakness | expansion needed? | claim-boundary risk |
|---|---|---|---|---|---|
| **Abstract** | one-paragraph bounded claim | every adjective; proxy/partial/retention/fold9 language intact | dense; many caveats in one block | maybe split into 2 sentences | **HIGH** (easiest place to over-claim) |
| **§1 Introduction** | motivate measurement→control gap; C1–C4 | does the gap read as compelling, not incremental? contributions crisp? | opening could hook harder; gap stated, not *felt* | slight (sharpen motivation) | **HIGH** |
| **§2 Related Work** | position vs DG / graph-EEG / decoders / CMI | coverage; does it justify "we differ by auditing first"? | **thinnest section (~120 words)** | **YES — expand before TMLR** | medium |
| **§3 Method** | proxy defn, two-step regularizer, audit/null | reads as a method (not a log); proxy-not-CMI explicit; equations clear | a touch terse; could add one intuition sentence | small | **HIGH** (CMI wording) |
| **§4 Protocol** | source-only firewall, fixed-candidate selection | firewall list complete; "evaluation-only" prominent | fine; concise | no | medium (firewall) |
| **§5 Results** | two-dataset confirmation + F2 | persuasive *without* SOTA; gate-based retention; fold9 visible; points to T3/T4 | could foreground the "partial, honest" framing more | no | **HIGH** (retention/fold9) |
| **§6 Analysis & Negative Results** | negatives as evidence; why this backbone | negatives framed as *intentional scoping*; dynamic-edge non-causal | risk of sounding like excuses if reordered | no | **HIGH** (dynamic-edge causal) |
| **§7 Limitations & Conclusion** | bound the claim; future work | honest but not self-defeating; future work ≠ admitting the paper is incomplete | could end on the contribution, not the caveats | no | medium |

## Cross-section watch items

- **Negative results (§6) must stay "evidence, not excuses."** Frame as: each failed check *rules in* the
  chosen scope. F4 + Table 5 carry the trace.
- **Limitations (§7) must not undercut the contribution.** Order: state the bounded result first, then the
  honest scope; do not open §7 with "this is only…".
- **Related Work (§2) is the main expansion target** for TMLR (review culture rewards thorough positioning).
- **Do not introduce numbers or new claims** during the prose edit; all numbers trace to Tables 2–5 /
  CIGL_25/29/31.

## Suggested edit order (highest leverage first)

1. §2 Related Work (expand). 2. §1 Introduction (sharpen the gap). 3. Abstract (split/clarify).
4. §5 Results (foreground honest framing). 5. §6/§7 (tone polish). 6. §3/§4 (minor).
