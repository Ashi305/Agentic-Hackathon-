"""
Person 2: Dynamic Few-Shot Injection Manager
Pulls logged safety officer risk overrides from Person 1's SQLite/JSON storage,
formats them as structured pedagogical exemplars, and dynamically injects them into LLM inference.
"""
from typing import List, Dict, Any, Optional
from person1_data_pipeline.schema import RiskOverride
from person1_data_pipeline.storage import SafetyStorage


class DynamicFewShotManager:
    """Manages the active learning feedback loop between human overrides and model prompts."""

    def __init__(self, storage: Optional[SafetyStorage] = None):
        self.storage = storage or SafetyStorage()

    def get_latest_exemplars(self, max_examples: int = 4) -> List[RiskOverride]:
        """Queries the storage layer for the most recent human corrections."""
        try:
            return self.storage.get_recent_overrides(limit=max_examples)
        except Exception:
            return []

    def format_exemplars_for_prompt(self, overrides: Optional[List[RiskOverride]] = None) -> str:
        """Transforms override objects into high-impact few-shot learning text."""
        if overrides is None:
            overrides = self.get_latest_exemplars()

        if not overrides:
            return "No previous human overrides logged. Follow standard baseline classification."

        lines = [
            "ACTIVE HUMAN-IN-THE-LOOP FEEDBACK EXAMPLARS:",
            "The safety committee has established the following precedence rules based on recent overrides:\n"
        ]

        for i, ov in enumerate(overrides, start=1):
            direction = f"{ov.original_risk.value} -> {ov.overridden_risk.value}"
            lines.append(f"[{i}] Correction ({direction}) by {ov.safety_officer_id}:")
            lines.append(f"    Report Ref: {ov.report_id}")
            lines.append(f"    Guiding Rationale: \"{ov.override_reason}\"")
            lines.append(f"    Action: When similar precursor conditions appear, classify as '{ov.overridden_risk.value}'.\n")

        return "\n".join(lines)

    def get_override_statistics(self) -> Dict[str, Any]:
        """Provides audit metrics on human override patterns."""
        all_ovs = self.storage.get_all_overrides()
        total = len(all_ovs)
        escalations = sum(1 for o in all_ovs if o.original_risk.value < o.overridden_risk.value)
        de_escalations = sum(1 for o in all_ovs if o.original_risk.value > o.overridden_risk.value)
        
        return {
            "total_overrides_logged": total,
            "escalations_to_higher_risk": escalations,
            "de_escalations_to_lower_risk": de_escalations,
            "active_few_shot_count": min(total, 5)
        }
