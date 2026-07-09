from __future__ import annotations

import json

import numpy as np

from tta_mech_eeg.data.artifact_inventory import build_artifact_inventory, sha256_file


def test_artifact_inventory_reports_npz_schema_without_replay(tmp_path):
    feature = tmp_path / "features.npz"
    np.savez(
        feature,
        z=np.ones((4, 2), dtype=np.float32),
        y=np.array([0, 1, 0, 1]),
        domain=np.array(["s0", "s0", "s1", "s1"]),
        groups=np.array(["g0", "g1", "g2", "g3"]),
    )
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "per_artifact_hashes": [
                    {
                        "path": str(feature),
                        "file_sha256": sha256_file(feature),
                        "dataset": "toy",
                        "backbone": "toybackbone",
                        "fold_id": "1",
                        "seed": "0",
                    }
                ]
            }
        )
    )
    payload = build_artifact_inventory(handoff)
    rec = payload["records"][0]
    assert rec["contains_z"] is True
    assert rec["contains_y"] is True
    assert rec["contains_domain"] is True
    assert rec["contains_groups"] is True
    assert rec["status"] == "AVAILABLE"
    assert payload["real_eeg_replay_run"] is False
    assert payload["target_metrics_computed"] is False
