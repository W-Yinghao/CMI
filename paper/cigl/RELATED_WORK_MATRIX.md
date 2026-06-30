# CIGL Related-Work Matrix (created Phase 4B; verification updated Phase 4C/4E)

> **Verification status now lives in `REFERENCES_VERIFIED.md`** (Phase 4C). Where a key there is
> "verified" (MOABB, EEGNet, ShallowConvNet/DeepConvNet, RGNN, LGGNet, conditional-invariant Li 2018, CCMI),
> treat the `TODO: verify citation` cells below as resolved (exact vol/DOI still `TODO: verify`). Rows with
> no confident reference (DGCNN exact venue, domain-adversarial DANN, dynamic-graph, dataset primaries)
> remain `TODO`. No bibliographic details are fabricated.


> Positioning grid. **Citations are not fabricated** — author-named where confidently known and marked
> `TODO: verify citation` for exact venue/year/DOI; rows with no confident reference say
> `TODO: verify source`. Each row: what they do · how CIGL differs · novelty risk · safe contribution wording.

| Topic | Representative ref (verify) | What they do | How CIGL differs | Novelty risk | Safe contribution wording |
|---|---|---|---|---|---|
| EEG domain generalization / cross-subject MI | MOABB benchmark (Jayaram & Barachant) `TODO: verify citation` | Standardized cross-subject MI evaluation, many pipelines | We use MOABB datasets/LOSO but target **leakage in graph reps**, not accuracy ranking; strict source-only | Medium — DG is crowded | "We adopt MOABB MI datasets/LOSO to study leakage, not to top an accuracy benchmark." |
| DGCNN / graph EEG networks | DGCNN (Song et al.) `TODO: verify citation` | Dynamical graph CNN with learned adjacency for EEG | We **reuse DGCNN as the task-capable backbone** and audit/regularize its graph/node objects; we do not propose a new graph architecture | Medium — backbone is prior art | "We use a DGCNN backbone as a substrate for a leakage audit + graph/node regularizer." |
| RGNN / node-level graph EEG | RGNN (Zhong et al.) `TODO: verify citation` | Regularized graph net with node/adjacency structure for EEG emotion | We focus on **node-level leakage** (per-electrode) explicitly and conditionally on label | Low–Medium | "We audit per-electrode (node) label-conditional leakage, complementary to node-feature modeling." |
| LGGNet / local-global graph EEG | LGGNet (Ding et al.) `TODO: verify citation` | Local-global temporal-graph features for EEG | We borrow temporal-stem intuition only; our contribution is leakage audit/control, not the stem | Low | "Temporal-graph stems are prior art; our novelty is conditional leakage measurement/reduction." |
| Known-good conv decoders | EEGNet (Lawhern et al.); ShallowConvNet/DeepConvNet (Schirrmeister et al.) `TODO: verify citation` | Compact/temporal CNNs for EEG decoding | Used **only as sanity references** for task learnability (Phase 3A-S), not as our method | Low | "We use known-good decoders solely to verify the protocol is learnable." |
| Conditional invariant representations | conditional-DG / CDANN-style `TODO: verify citation` | Learn representations invariant to domain conditioned on label | CIGL **measures** label-conditional leakage with a permutation-null proxy and penalizes graph/node objects specifically | Medium–High — conceptual overlap | "We instantiate conditional invariance as an *audited* graph/node posterior-KL penalty, source-only." |
| CMI / classifier-based CMI / posterior-KL leakage proxies | classifier-based MI/CMI estimators `TODO: verify citation` | Estimate (C)MI via classifiers/variational bounds | We use a **posterior-KL plug-in proxy** and explicitly disclaim unbiased CMI; add a retrained within-label null | High — must not overclaim estimator | "We use a posterior-KL plug-in proxy (not an unbiased CMI estimator) with a permutation null." |
| Domain-adversarial / marginal invariance | DANN / marginal `I(Z;D)` penalties `TODO: verify citation` | Adversarial or marginal domain alignment | We penalize **label-conditional** `I(Z;D\|Y)`-style leakage, not marginal; and audit before/after | Medium | "We target conditional (not marginal) domain leakage, avoiding label erasure." |
| EEG dynamic graph / graph disentanglement | dynamic-adjacency / disentangled graph EEG `TODO: verify citation` | Per-sample/dynamic adjacency, disentangled graph factors | We **tried** dynamic-edge backbones; they **overfit** under source-only (reported as negative result); we stay static graph/node | Medium — reviewers may ask for dynamic edge | "We report that dynamic-edge designs overfit here; edge-CMI is future work, not a claim." |

## Cross-cutting positioning

- **Contribution stance:** not a new architecture or a SOTA decoder; a **source-only leakage audit +
  bounded graph/node regularizer**, with negative results that justify the backbone/scope choice.
- **Biggest novelty risks:** (1) conditional-invariance overlap → emphasize the *audit + permutation null +
  source-only firewall*; (2) estimator overclaim → always say *posterior-KL proxy, not unbiased CMI*; (3)
  "why not edge/dynamic graph" → answered by the reported overfitting negative result.
- **Citation policy:** before submission, replace every `TODO: verify citation`/`TODO: verify source` with
  a checked reference; do not ship fabricated bibliographic details.
