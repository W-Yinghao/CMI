"""C85U held-evaluation identities — SENSITIVE. Imported ONLY by the D2 held
evaluator (and lazily by verify_c85u_identity). Stage D1 (selection) must never
import this module, so a D1 process holds no held-evaluation path.
"""
from __future__ import annotations

C85U_ACCEPTANCE_MANIFEST = ("/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
                            "c85u-v2-77382c16a593f7c2-91a428488a634268/final_acceptance_bundle/"
                            "C85U_RESULT_ARTIFACT_MANIFEST.json")
C85U_ACCEPTANCE_SHA = "dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620"
C85U_UTILITY_INDEX = ("/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
                      "c85u-v2-77382c16a593f7c2-91a428488a634268/stage_u1_candidate_utility_v2/"
                      "candidate_utility_index.csv")
C85U_UTILITY_INDEX_SHA = "83bddf56290c4e06a306d64dadfc9611115a177f479d433fe0e4485b0c181509"
C85U_FIELD_IDENTITY = {"contexts": 944, "candidates_per_context": 81,
                       "candidate_rows": 76_464, "evaluation_label_table_rows": 4_848}
