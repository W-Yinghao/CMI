# OACI EEG-DG Project Memory Through C84R3

C84C job `895366` consumed the V3/V2 authorization and stopped on a float32 linear replay error of `2.86102294921875e-6`. It accessed three Lee views and two source-label arrays, with zero target-y access, zero target scientific metrics, and zero complete units.

C84R3 preserves that failure and supersedes the old lock additively. The V4 canary protocol is `cc54b5e6f92e4b0d338bf297c92823b4d60a8628a55dcff547ef9d808ee43afb` and the V3 execution lock is `c198607fb9e46ea2353ffa57d6b71bfa966c36e8ece53fdc40292681bba8bd1a`. Only the 1040-term float32 linear replay tolerance is `1e-5`; all strict identity tolerances remain `1e-6`. The failed root is not reusable and all 243 units must be retrained.

Final gate: `C84C_FLOAT32_REPLAY_REPAIRED_AND_RELOCKED_READY_FOR_FRESH_PI_AUTHORIZATION`. A fresh direct C84C authorization is required. C84F and C84S remain unauthorized and have no execution locks.
