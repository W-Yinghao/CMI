"""Run CEDAR red-team checks on a result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cedar_eeg.red_team import validate_p0_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-json", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-perm-null-adv", type=float, default=0.05)
    args = ap.parse_args()
    result = validate_p0_result(args.p0_json, max_perm_null_adv=args.max_perm_null_adv)
    payload = result.to_dict()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
