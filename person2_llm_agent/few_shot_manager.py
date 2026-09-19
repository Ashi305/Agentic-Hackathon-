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