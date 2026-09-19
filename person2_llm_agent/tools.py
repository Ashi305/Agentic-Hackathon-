import json
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    """
    Convert an LLM response into a Python dictionary.
    Handles cases where the model accidentally adds
    markdown code fences.
    """

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    try:
        return json.loads(text)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON.\n"
            f"Response was:\n{text}"
        ) from e


def validate_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes sure extraction output contains the fields
    our application expects.
    """

    required_fields = [
        "risk_factors",
        "location",
        "department",
        "hazards",
        "potential_consequences",
        "missing_information"
    ]

    for field in required_fields:
        if field not in data:
            data[field] = []

    return data


def validate_classification(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize classification output.
    """

    valid_levels = {"LOW", "MEDIUM", "HIGH"}

    risk_level = str(
        data.get("risk_level", "MEDIUM")
    ).upper()

    if risk_level not in valid_levels:
        risk_level = "MEDIUM"

    confidence = data.get("confidence", 0.5)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5

    confidence = max(0.0, min(1.0, confidence))

    reason = str(
        data.get("reason", "No reason provided.")
    )

    human_review = confidence < 0.60

    return {
        "risk_level": risk_level,
        "confidence": round(confidence, 2),
        "reason": reason,
        "human_review_required": human_review
    }