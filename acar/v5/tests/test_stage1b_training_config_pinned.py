"""Guard (Stage-1B5): the EEGNet/source-state training spec is PINNED in code + deterministically hashed, and encodes the FIT-only
discipline. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import training_config as TC
from acar.v5.tests._util import ok


def test_pinned_values():
    c = TC.TRAINING_CONFIG
    assert c["architecture"] == "EEGNet" and c["n_chans"] == 19 and c["n_times"] == 512 and c["n_classes"] == 2
    assert c["deterministic"] is True and c["torch_threads"] == 1
    assert c["trains_on"] == "FIT_train_val_only" and c["reads_labels"] == "FIT_training_only"
    assert c["early_stopping_metric"] == "val_loss" and c["checkpoint_selection"] == "best_val_loss"
    ok("training config pinned: EEGNet 19x512 / 2 classes / deterministic / FIT-only train + labels")


def test_hash_deterministic_64hex():
    h1, h2 = TC.config_sha256(), TC.config_sha256()
    assert h1 == h2 and len(h1) == 64 and all(ch in "0123456789abcdef" for ch in h1)
    ok(f"training config sha256 deterministic 64-hex ({h1[:12]}…)")


def main():
    print("ACAR v5 Stage-1B5 guard: training config pinned")
    test_pinned_values()
    test_hash_deterministic_64hex()
    print("ALL V5 STAGE1B-TRAINING-CONFIG GUARDS PASS")


if __name__ == "__main__":
    main()
