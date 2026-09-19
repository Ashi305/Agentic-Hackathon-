import re
from typing import List, Dict


class RAGEngine:
    """
    Simple retrieval engine for finding relevant previous
    safety-officer corrections.

    This intentionally uses lightweight keyword overlap
    instead of a vector database.
    """

    def __init__(self, few_shot_manager):
        self.few_shot_manager = few_shot_manager

    def _tokenize(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        stop_words = {
            "the", "and", "was", "were", "with",
            "this", "that", "from", "into", "near",
            "after", "before", "there", "worker"
        }

        return set(words) - stop_words

    def _similarity(self, text1: str, text2: str) -> float:
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)

        return len(intersection) / len(words1.union(words2))

    def retrieve(
        self,
        current_report: str,
        top_k: int = 3
    ) -> List[Dict]:

        corrections = self.few_shot_manager.get_all_corrections()

        scored_examples = []

        for correction in corrections:
            similarity = self._similarity(
                current_report,
                correction["report"]
            )

            scored_examples.append(
                (similarity, correction)
            )

        scored_examples.sort(
            key=lambda x: x[0],
            reverse=True
        )

        results = []

        for similarity, correction in scored_examples[:top_k]:
            if similarity > 0:
                example = correction.copy()
                example["similarity"] = round(similarity, 3)
                results.append(example)

        return results