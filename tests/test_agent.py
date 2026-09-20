"""
Unit tests for Person 2: SafetyReportAgent, FewShotManager, RAGEngine, and Extraction/Classification Validators
"""
import unittest
from person2_llm_agent.few_shot_manager import FewShotManager, DynamicFewShotManager
from person2_llm_agent.rag_engine import RAGEngine
from person2_llm_agent.tools import extract_json, validate_extraction, validate_classification
from person4_frontend.agent_service import FrontendAgentService


class TestPerson2Agent(unittest.TestCase):
    def setUp(self):
        self.few_shot_manager = FewShotManager()
        self.rag_engine = RAGEngine(self.few_shot_manager)
        self.agent_service = FrontendAgentService()

    def test_json_extraction_tool(self):
        raw_llm_output = '```json\n{"risk_level": "HIGH", "confidence": 0.95, "reason": "Severe chemical burn hazard", "human_review_required": false}\n```'
        parsed = extract_json(raw_llm_output)
        self.assertEqual(parsed["risk_level"], "HIGH")
        self.assertEqual(parsed["confidence"], 0.95)

    def test_validate_extraction_tool(self):
        data = {
            "risk_factors": ["unshielded flange", "leaking acid"],
            "hazards": ["acid splash"]
        }
        validated = validate_extraction(data)
        self.assertIn("location", validated)
        self.assertIn("department", validated)
        self.assertIn("potential_consequences", validated)
        self.assertIn("missing_information", validated)

    def test_validate_classification_tool(self):
        data = {
            "risk_level": "high",
            "confidence": "0.92",
            "reason": "High pressure leak"
        }
        validated = validate_classification(data)
        self.assertEqual(validated["risk_level"], "HIGH")
        self.assertEqual(validated["confidence"], 0.92)
        self.assertFalse(validated["human_review_required"])

    def test_human_review_required_trigger(self):
        data = {
            "risk_level": "medium",
            "confidence": 0.45,
            "reason": "Uncertain details"
        }
        validated = validate_classification(data)
        self.assertTrue(validated["human_review_required"])

    def test_few_shot_manager_and_rag_retrieval(self):
        self.few_shot_manager.add_correction(
            report="Worker slipped on loose oil patch near press.",
            original_label="LOW",
            corrected_label="HIGH",
            reason="Active oil leak next to operating press ram creates imminent crush precursor."
        )
        self.assertEqual(len(self.few_shot_manager.get_all_corrections()), 1)

        # RAG retrieval based on lexical similarity
        retrieved = self.rag_engine.retrieve("oil leak near machine", top_k=2)
        self.assertGreaterEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["corrected_label"], "HIGH")

    def test_frontend_agent_service_integration(self):
        trace = self.agent_service.analyze_report(
            raw_text="Operator slipped on minor puddle of clean water near drinking fountain. Held railing, no injury.",
            department="Office & Admin",
            location="Breakroom Corridor",
        )
        self.assertIsNotNone(trace.final_enriched_report)
        self.assertEqual(trace.final_enriched_report.effective_risk_level.value, "Low")
        self.assertIn("risk_factors", trace.raw_extraction_data)


if __name__ == "__main__":
    unittest.main()
