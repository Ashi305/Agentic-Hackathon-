try:
    from .prompts import build_extraction_prompt, build_classification_prompt
    from .tools import extract_json, validate_extraction, validate_classification
    from .rag_engine import RAGEngine
    from .llm_client import LLMClient
except (ImportError, ValueError):
    from prompts import build_extraction_prompt, build_classification_prompt
    from tools import extract_json, validate_extraction, validate_classification
    from rag_engine import RAGEngine
    from llm_client import LLMClient


class SafetyReportAgent:

    def __init__(self, few_shot_manager):

        self.llm = LLMClient()

        self.few_shot_manager = few_shot_manager

        self.rag_engine = RAGEngine(
            few_shot_manager
        )

    def extract_information(self, report: str):

        prompt = build_extraction_prompt(report)

        response = self.llm.generate(prompt)

        data = extract_json(response)

        return validate_extraction(data)

    def classify_risk(
        self,
        report: str,
        extracted_data: dict
    ):

        # Find relevant previous human corrections
        examples = self.rag_engine.retrieve(
            current_report=report,
            top_k=3
        )

        prompt = build_classification_prompt(
            report=report,
            extracted_data=extracted_data,
            few_shot_examples=examples
        )

        response = self.llm.generate(prompt)

        classification = extract_json(response)

        return validate_classification(
            classification
        )

    def analyze(self, report: str):

        if not report or not report.strip():
            raise ValueError(
                "Safety report cannot be empty."
            )

        # STEP 1
        extracted_data = self.extract_information(
            report
        )

        # STEP 2
        classification = self.classify_risk(
            report,
            extracted_data
        )

        # STEP 3
        result = {
            "report": report,
            "extraction": extracted_data,
            "classification": classification
        }

        return result