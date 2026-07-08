"""P1 channel/filter surgery entrypoint.

P1 is intentionally blocked until P0 has an accepted source-side result.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-json", required=True)
    args = ap.parse_args()
    with open(args.p0_json) as f:
        p0 = json.load(f)
    if not p0.get("selected"):
        raise SystemExit("P1 blocked: P0 has no ACCEPT candidate")
    raise SystemExit("P1 implementation is blocked until P0 is reviewed and approved")


if __name__ == "__main__":
    main()
