"""
Person 2: Prompt Templates & Dynamic Prompt Engineering
Contains structured prompts for near-miss parsing, precursor hazard extraction,
and risk classification (Low, Medium, High) with dynamic few-shot exemplar injection.
"""
from typing import List, Optional
from person1_data_pipeline.schema import RiskOverride


class PromptFactory:
    """Constructs calibrated prompts for the Incident Precursor Reasoning Agent."""

    SYSTEM_SAFETY_INSPECTOR = """You are an elite Senior Industrial Safety Officer and OSHA Precursor Incident Analyst.
Your mandate is to inspect free-text safety observation and near-miss reports, identify critical incident precursors,
and assign an objective Risk Level (Low, Medium, High).

Risk Level Classification Rubric:
- HIGH RISK:
  * Involves high-energy hazards or direct fatal precursors (falls > 4ft, heavy mobile equipment, high voltage/arc flash,
    toxic/corrosive chemicals under pressure, confined spaces, bypassed mechanical interlocks, suspended loads).
  * Multiple barriers failed or missing.
  * Near-miss where luck, timing, or reflex was the sole factor preventing severe injury or fatality.

- MEDIUM RISK:
  * Significant hazards that can cause lacerations, fractures, moderate chemical contact, or heat burns, but lack immediate fatal/amputation potential.
  * Primary physical barrier failed, but secondary administrative safeguard partially held.

- LOW RISK:
  * Minor slips, trips without fall, minor ergonomic strain, clean water leaks, minor housekeeping or PPE labeling defects.
  * No potential for lost time or permanent impairment.

You MUST follow established OSHA standards and prioritize PRECURSOR SIGNALS (early warnings of catastrophe) over superficial outcome severity."""

    @classmethod
    def build_extraction_and_classification_prompt(
        cls,
        report_text: str,
        department: str,
        location: str,
        retrieved_osha_context: str = "",
        historical_overrides: Optional[List[RiskOverride]] = None,
    ) -> str:
        """Constructs a prompt enriched with RAG regulatory context and human safety officer overrides."""
        prompt = [
            f"Department: {department}",
            f"Location: {location}",
            f"Raw Near-Miss Observation Text:\n\"\"\"\n{report_text}\n\"\"\"\n",
        ]

        if retrieved_osha_context:
            prompt.append("=== GROUNDED OSHA REGULATORY BENCHMARKS & MITIGATION GUIDANCE ===")
            prompt.append(retrieved_osha_context)
            prompt.append("=================================================================\n")

        # Dynamic Few-Shot Injection: injects real human safety corrections
        if historical_overrides and len(historical_overrides) > 0:
            prompt.append("=== DYNAMIC FEW-SHOT CORRECTIONS FROM SENIOR SAFETY OFFICERS ===")
            prompt.append("The following are recent human overrides where the safety officer corrected an initial assessment.")
            prompt.append("Learn from their rationale to calibrate your classification:")
            for idx, ov in enumerate(historical_overrides, 1):
                prompt.append(
                    f"Example {idx}: [Original Model Risk: {ov.original_risk.value}] -> [Officer Override: {ov.overridden_risk.value}]\n"
                    f"Officer Justification: \"{ov.override_reason}\"\n"
                )
            prompt.append("=================================================================\n")

        prompt.append(
            "TASK:\n"
            "Analyze the report text and return a JSON object with the following exact keys:\n"
            "{\n"
            '  "primary_hazard": "Concise title of the hazard",\n'
            '  "hazard_category": "One of: Chemical / Toxic Hazard | Mechanical / Pinch / Struck-by | Electrical / Arc Flash | Slip / Trip / Surface Hazard | Working at Height / Falling Objects | Thermal / Fire / Hot Work | Atmospheric / Confined Space | Ergonomic / Heavy Lifting | Process Safety / Pressure Excursion",\n'
            '  "precursor_events": ["list", "of", "immediate", "unsafe", "conditions", "or", "early", "warning", "signs"],\n'
            '  "affected_assets": ["list", "of", "personnel", "or", "assets", "at", "risk"],\n'
            '  "failed_safeguards": ["list", "of", "bypassed", "or", "failed", "controls"],\n'
            '  "recommended_mitigation": "Immediate actionable control",\n'
            '  "risk_level": "Low" or "Medium" or "High",\n'
            '  "risk_score": float between 0.0 and 10.0,\n'
            '  "rationale": "Comprehensive explanation justifying the risk level based on precursor severity and safety margins",\n'
            '  "osha_citations": ["relevant OSHA standards or benchmarks"],\n'
            '  "precursor_severity_signals": ["Fatal Precursor", "Energy Isolation Defect", etc.],\n'
            '  "escalation_potential": true or false\n'
            "}\n"
            "Return valid JSON only. Do not include markdown fences or preamble."
        )

        return "\n".join(prompt)
