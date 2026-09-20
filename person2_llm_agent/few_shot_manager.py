from typing import List, Dict


class FewShotManager:
    """
    Stores safety-officer corrections and provides
    examples that can be inserted into future prompts.
    """

    def __init__(self):
        self.corrections: List[Dict] = []

    def add_correction(
        self,
        report: str,
        original_label: str,
        corrected_label: str,
        reason: str
    ):
        correction = {
            "report": report,
            "original_label": original_label,
            "corrected_label": corrected_label,
            "reason": reason
        }

        self.corrections.append(correction)

    def get_all_corrections(self) -> List[Dict]:
        return self.corrections

    def get_recent_corrections(self, limit: int = 5) -> List[Dict]:
        return self.corrections[-limit:]

    def clear(self):
        self.corrections.clear()


class DynamicFewShotManager:
    """Connects SQLite storage overrides to agent prompt contexts."""
    def __init__(self, storage=None):
        from person1_data_pipeline.storage import SafetyStorage
        self.storage = storage or SafetyStorage()

    def get_latest_exemplars(self, max_examples: int = 5):
        try:
            return self.storage.get_recent_overrides(limit=max_examples)
        except Exception:
            return []

    def format_exemplars_for_prompt(self, overrides=None):
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