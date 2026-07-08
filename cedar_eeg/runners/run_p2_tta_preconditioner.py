"""P2 TTA preconditioner entrypoint.

P2 is secondary and cannot run until P0 and P1 pass without target-label
selection.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-json", required=True)
    args = ap.parse_args()
    with open(args.p1_json) as f:
        p1 = json.load(f)
    if not p1.get("selected"):
        raise SystemExit("P2 blocked: P1 has no accepted structured surgery")
    raise SystemExit("P2 implementation is blocked until P1 is reviewed and approved")


if __name__ == "__main__":
    main()
