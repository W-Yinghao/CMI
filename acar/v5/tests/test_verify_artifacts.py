"""Guard: file-byte artifact verification is fail-closed (missing file / sha mismatch / bad expected-hash). Synthetic temp files;
NO model load, NO real artifact read."""
from __future__ import annotations
import hashlib
import os
import tempfile
from acar.v5.substrate import verify_artifacts as VA
from acar.v5.tests._util import expect_raises, ok


def test_match_mismatch_missing():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "artifact.bin")
        payload = b"acar-v5 synthetic artifact"
        with open(p, "wb") as f:
            f.write(payload)
        good = hashlib.sha256(payload).hexdigest()
        assert VA.verify_artifact_file(p, good) == good
        expect_raises(VA.ArtifactHashMismatch, lambda: VA.verify_artifact_file(p, "0" * 64), "wrong sha")
        expect_raises(FileNotFoundError, lambda: VA.verify_artifact_file(os.path.join(d, "nope.bin"), good), "missing file")
        expect_raises(ValueError, lambda: VA.verify_artifact_file(p, "not-hex"), "bad expected hash")
    ok("verify_artifact_file: match ok; sha mismatch / missing file / bad-hex → fail-closed")


def main():
    print("ACAR v5 guard: verify artifacts (file-byte sha)")
    test_match_mismatch_missing()
    print("ALL V5 VERIFY-ARTIFACTS GUARDS PASS")


if __name__ == "__main__":
    main()
