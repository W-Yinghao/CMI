from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.data.tueg_ssl_batch_stream import _batches, _epoch_stream


def test_common_and_replacement_streams_are_deterministic_and_separate():
    universe = [f"window-{index:03d}" for index in range(64)]
    common_a = _epoch_stream(universe, 0, "common", 2)
    common_b = _epoch_stream(universe, 0, "common", 2)
    replacement = _epoch_stream(universe, 0, "replacement", 2)
    assert common_a == common_b
    assert common_a != replacement
    assert len(_batches(common_a)) == 2
    assert canonical_hash(_batches(common_a)) != canonical_hash(_batches(replacement))
