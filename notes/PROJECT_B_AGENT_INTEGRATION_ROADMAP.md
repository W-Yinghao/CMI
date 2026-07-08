# Project B x EEGAgent Integration Roadmap

*Roadmap / discussion, not an evaluated result. EEGAgent references summarize the uploaded EEGAgent
document as described by the PM.*

## 1. Role separation
- Foundation EEG model: backend / representation / decoder (parametric tool).
- Project B router: safety governor / action selector / OACI audit.
- EEGAgent: workflow orchestrator / tool scheduler / report generator.

## 2. Why Project B is a router tool, not an agent
EEGAgent (an LLM-enhanced framework for scheduling EEG tools and generating reports; it supports
perception, exploration, event detection, user interaction, and reporting; its toolbox mixes parametric
and non-parametric tools organized by temporal/spatial granularity) is an orchestration layer. Project B
answers a narrow, safety-critical decision — refuse / output identity / adapt — that must be governed by
calibrated risk, not by an LLM plan. So Project B is a TOOL the agent calls, not the agent itself.

## 3. Tool API
    {
      "tool_name": "refusal_first_router",
      "input": {"target_eeg_summary": "...", "model_outputs": "...",
                "support_diagnostics": "...", "candidate_actions": ["IDENTITY","PRIOR_ONLY","OFFLINE_TTA"]},
      "output": {"action": "REFUSE|IDENTITY|OFFLINE_TTA", "accepted": true,
                 "oaci_reason_codes": ["..."], "diagnostics": {}, "recommended_next_step": "..."}
    }

## 4. OACI-to-report translation
EEGAgent translates OACI reason codes into clinician/user-readable language (e.g. "adaptation not run
because support mismatch and degenerate harm calibration; suggest acquiring calibration data").

## 5. Refusal workflows
On REFUSE, EEGAgent may schedule complementary tools from its toolbox (artifact check, PSD, symmetry,
amplitude, baseInfo) and request calibration acquisition, then re-query the router.

## 6. What EEGAgent may do
Explain, schedule other tools, generate structured reports, handle user follow-up, and surface the
router's recommended next step.

## 7. What EEGAgent must not do
- It must NOT override Project B's refusal / no-TTA decision.
- It must NOT present an adaptation the router blocked as safe.
- It must NOT tune router thresholds on target outcomes.

## 8. Minimal demo plan (future work)
Wrap the frozen RefusalFirstRouter as an EEGAgent tool; on a held-out target, show the agent (a) calling
the router, (b) translating OACI codes, (c) scheduling an artifact/PSD check on REFUSE, (d) producing a
report — with the router decision authoritative throughout. This is future work; it is not evaluated here
and does not itself improve decoding safety.
