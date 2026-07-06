# Paper integration plan V3 (method-deepen -> manuscript) --- OUTLINE ONLY, no prose edits yet

Prereq: method-deepen line FROZEN (tag tos-cmi-method-deepen-v2-final @ 0ef7be3). This is a PLAN; the actual
manuscript (sections/*.tex) is NOT edited until the PM approves "start manuscript rewrite". Venue = TMLR
(two-build main.tex + tmlr_main.tex). Claim contract = claim_evidence_table.md C14 (allowed/forbidden) + CLAIMS_LEDGER.

## Core narrative upgrade
Old (uploaded 2a-only snapshot): "measurement works; control is not guaranteed; certification and abstention matter."
New: **"Measurement works; erasure is not control; source-only certified control is REFUSAL-FIRST; target-
beneficial deployment-shift interventions may require target information or stronger shift assumptions."**
Positioning: from an EEG diagnostic-negative into a decision-theoretic study of when source-only selective
invariance can and cannot act.

## 1. MAIN TEXT (in)
* Multi-dataset C12 validation (real EEG, 9 valid cells) --- compact table (tab:bigN_compact).
* Concept-erasure controls (LEACE/RLACE/INLP vs TOS).
* Capacity x architecture factorial (Track C; the dim<->type confound answer).
* **Track B source-OOD gate --- compact table** (0 false-accept, 8/8 harms prevented; naive controllers false-accept).
* **V2 source-only acceptance ceiling --- one short results subsection** (EEGNet clean ceiling robust across
  n_source; naive-vs-gate-vs-oracle contrast; the non-identifiability Proposition). Headline scatter figure.

## 2. APPENDIX (in)
* Full multi-dataset C12 table (tab:bigN_full).
* Full Track B table + naive-controller baselines.
* Phase 2 dry-run (task-preserving erasure preserves task, transfer-flat).
* V2 Stage-2 full table (World A/EEGNet by n_source; B/C by backbone x n_source).
* TSMNet World A NOT-clean appendix note (honest asymmetry + mechanism).
* V2 naive-controller table + non-identifiability Proposition (W+/W-).

## 3. INTERNAL EVIDENCE ONLY (not in paper)
* worldA_search 144-cell grid; Stage-1B nuisance_fraction sweep (support the "TSMNet not clean" caveat, cited not tabled).
* Per-shard logs / manifests; smoke/full-lite intermediate runs.

## 4. OLD CLAIMS TO DELETE / SOFTEN in the manuscript
* "Single dataset" limitation -> REPLACE with the multi-dataset scope (C12) + method-deepen breadth.
* Any wording implying the gate could ACCEPT / has acceptance power -> the ceiling wording (forbidden list in C14).
* "certified deletion mostly abstains" as a bare weakness -> reframe: abstention/refusal is the CERTIFIED correct
  action; the gate has safe refusal power + an acceptance ceiling.
* "no end-to-end TOS training" limitation -> keep, but reposition: e2e (Track E) is future work, not required for
  the decision-theoretic claim; source-only acceptance has a ceiling regardless.

## 5. LIMITATIONS TO REWRITE
* External validity: was single-dataset; now multi-dataset (C12) + semi-synthetic capacity sweep (V2).
* Controller behavior: add "refusal-first; acceptance ceiling; TSMNet World-A not cleanly demonstrable under the
  appended-nuisance construction (high-dim-latent limitation)".
* Scope of V2: semi-synthetic (real latents + injected nuisance); a limit result, not a real-EEG positive.
* Future work: target-informed acceptance ("how much target info crosses the ceiling"); Track E only if a
  source-visible-benefit setting or target-informed branch is opened.

## Not in paper / future work (parked)
target-informed branch; Track E; full big-N LPC / capacity Wave 2; World-A redesign.

## Execution order once approved
1. Draft the two NEW subsections (Track B compact + V2 ceiling) against the claim contract.
2. Rewrite the limitations + scope-of-validation paragraphs.
3. Add appendix tables/figures from the frozen results (aggregates only; no NPZ; scrub sbatch local paths).
4. Run check_claims.sh / anonymity / forbidden-wording gates; two-build 0/0/0.
