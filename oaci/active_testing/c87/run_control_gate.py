"""Run the full C87 synthetic control gate and persist the signed result + a markdown report.

Usage:  python -m oaci.active_testing.c87.run_control_gate [out_dir]
"""
import json
import os
import sys

from .controls import run_control_gate


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    payload = run_control_gate(verbose=True)
    with open(os.path.join(out_dir, "c87_control_gate_result.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    # markdown report
    lines = [f"# C87 synthetic control gate — {payload['verdict']}",
             "", f"signature `{payload['signature']}`  ·  elapsed {payload['elapsed_s']}s", "",
             "```text"] + payload["logs"] + ["```", "", "## Per-control verdicts", ""]
    for k, v in payload["results"].items():
        lines.append(f"- **{k}**: {'PASS' if v['passed'] else 'FAIL'}")
    with open(os.path.join(out_dir, "C87_CONTROL_GATE_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print("VERDICT:", payload["verdict"], "signature:", payload["signature"])
    return 0 if payload["verdict"] == "CONTROL_PASS" else 2


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "c87_control_out"
    sys.exit(main(out))
