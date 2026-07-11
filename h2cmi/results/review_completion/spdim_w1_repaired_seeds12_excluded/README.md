# Excluded P100 Launch 892385

Array `892385` was canceled because the frozen PyTorch build does not support Tesla P100 compute capability sm_60. This is a real compatibility failure, not a harmless warning.

- accepted result rows: `0`
- result-like files in this evidence directory: `0`
- tasks 0 and 1 produced launch/error logs; tasks 2-7 never started.

| task | stdout sha256 | stderr sha256 | verdict |
|---:|---|---|---|
| 0 | `1a6fad83bfc8cece8f379c5000d5554043b96c4ef8ab29b12ca7944009b2dffc` | `2c22cd820e0e9cd6824224f0299cb9c94b4c6904ef1460b350683a284c1fbc4e` | unsupported_gpu_architecture_real_failure |
| 1 | `75e2219a1f6fead5ee98fafc3d45d730784958a3d0be27975326de116c2b53b2` | `29ba97aaccd0caba9c8340407d377411bbb0ef5150b28c3f6952592ed390f027` | unsupported_gpu_architecture_real_failure |

## Internal Validation Review

- No P100 row enters any accepted or final CSV.
- The incompatibility warning remains classified as a real failure.
