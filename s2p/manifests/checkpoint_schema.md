# S2P checkpoint schema (Phase 9A)

Per run: `checkpoint_epoch{E}.pt` (+ `best_by_pretrainval.pt`, `last.pt`), `pretrain_log.csv`
(epoch, train_loss, pretrain_val_loss, lr, time), `pretrain_val_subjects.csv` (held-out pretrain subjects, DISJOINT
from pretrain-train), `hashes.json` (corpus config 4704743c, channel-pipeline hash, subject_subset hash, code SHA).
**Checkpoint selection = min pretrain-val loss ONLY** (last-epoch reported as sensitivity). NEVER select by
downstream/target performance. No target labels anywhere in pretraining.
