def build_extraction_prompt(report: str) -> str:
    return f"""
You are a workplace safety analysis assistant.

Analyze the following safety report.

Extract ONLY information that is explicitly stated
or strongly supported by the report.

Do not invent information.

Return valid JSON with exactly these fields:

{{
    "risk_factors": [],
    "location": null,
    "department": null,
    "hazards": [],
    "potential_consequences": [],
    "missing_information": []
}}

Definitions:

risk_factors:
- Specific conditions or behaviors that increase risk.
- Examples: "no PPE", "oil spill", "exposed wiring".

location:
- Where the incident happened, if mentioned.

department:
- Department or work area, if mentioned.

hazards:
- One or more hazards identified in the report.

potential_consequences:
- What could happen because of the hazards.

missing_information:
- Important information that would help assess the risk
  but is not present in the report.

Safety report:
---
{report}
---

Return JSON only.
"""


def build_classification_prompt(
    report: str,
    extracted_data: dict,
    few_shot_examples: list
) -> str:

    examples_text = ""

    if few_shot_examples:
        examples_text = "\n\nHere are examples of previous safety-officer corrections:\n"

        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"""
Example {i}

Report:
{example["report"]}

Original AI classification:
{example["original_label"]}

Safety officer correction:
{example["corrected_label"]}

Officer reason:
{example["reason"]}
"""

    return f"""
You are a workplace safety risk classification assistant.

Classify the safety report as exactly one of:

LOW
MEDIUM
HIGH

Use the following general guidance.

LOW:
- Minor hazard
- Little immediate danger
- No significant potential consequence
- Easily corrected

MEDIUM:
- Meaningful safety concern
- Could cause injury or damage if not corrected
- Requires attention

HIGH:
- Serious or immediate hazard
- Could cause serious injury, fatality, major equipment damage,
  fire, electrical incident, chemical exposure, etc.
- Multiple interacting hazards may increase the risk

Important:
- Consider the actual hazards described.
- Consider potential consequences.
- Consider multiple hazards together.
- Do not assume facts that are not present.
- The examples below are previous human corrections and should
  influence your reasoning when they are relevant.
{examples_text}

Current report:
---
{report}
---

Extracted information:
{extracted_data}

Return valid JSON only:

{{
    "risk_level": "LOW",
    "confidence": 0.0,
    "reason": "...",
    "human_review_required": false
}}

Confidence must be a number between 0 and 1.

Set human_review_required to true when the classification
is uncertain or confidence is below 0.60.
"""


class PromptFactory:
    """Adapter class providing backward compatibility for agent orchestrator."""
    @classmethod
    def build_extraction_prompt(cls, report: str) -> str:
        return build_extraction_prompt(report)

    @classmethod
    def build_classification_prompt(cls, report: str, extracted_data: dict, few_shot_examples: list) -> str:
        return build_classification_prompt(report, extracted_data, few_shot_examples)

    @classmethod
    def build_extraction_and_classification_prompt(
        cls,
        report_text: str,
        department: str = "",
        location: str = "",
        retrieved_osha_context: str = "",
        historical_overrides = None
    ) -> str:
        few_shots = []
        if historical_overrides:
            for ov in historical_overrides:
                orig = ov.original_risk.value if hasattr(ov, 'original_risk') and hasattr(ov.original_risk, 'value') else str(getattr(ov, 'original_risk', 'Medium'))
                corr = ov.overridden_risk.value if hasattr(ov, 'overridden_risk') and hasattr(ov.overridden_risk, 'value') else str(getattr(ov, 'overridden_risk', 'High'))
                reason = getattr(ov, 'override_reason', '')
                few_shots.append({
                    "report": getattr(ov, 'report_id', 'Historical Observation'),
                    "original_label": orig,
                    "corrected_label": corr,
                    "reason": reason
                })
        return build_classification_prompt(report_text, {"department": department, "location": location, "osha_context": retrieved_osha_context}, few_shots)