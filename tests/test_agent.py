"""
Unit tests for Person 2: Agent, Tools, RAG, and Dynamic Few-Shot Engine
"""
import unittest
from person1_data_pipeline.storage import SafetyStorage
from person1_data_pipeline.schema import RiskLevel, RiskOverride
from person2_llm_agent.tools import SafetyToolbox
from person2_llm_agent.rag_engine import SafetyRAGEngine
from person2_llm_agent.few_shot_manager import DynamicFewShotManager
from person2_llm_agent.agent_orchestrator import IncidentPrecursorAgent


class TestPerson2Agent(unittest.TestCase):
    def setUp(self):
        self.storage = SafetyStorage()
        self.rag = SafetyRAGEngine()
        self.toolbox = SafetyToolbox(storage=self.storage, rag_engine=self.rag)
        self.few_shot_manager = DynamicFewShotManager(storage=self.storage)
        self.agent = IncidentPrecursorAgent(storage=self.storage)

    def test_severity_calculation_tool(self):
        res = self.toolbox.calculate_hazard_severity_index(
            energy_level="High",
            barrier_redundancy="Defeated",
            exposure_frequency="Continuous",
            uncontrolled_precursors=2
        )
        self.assertEqual(res["recommended_risk_tier"], "High")
        self.assertTrue(res["escalation_potential"])
        self.assertGreaterEqual(res["hazard_severity_score"], 7.5)

    def test_rag_retrieval(self):
        retrieved = self.rag.retrieve("acid spray shield flange leak", top_k=2)
        self.assertGreater(len(retrieved), 0)
        chunk, score = retrieved[0]
        self.assertIn("1910", chunk.standard_code)
        self.assertGreater(score, 0.0)

    def test_precursor_signal_detection(self):
        narrative = "Forklift approached blind intersection without sounding horn, missing pedestrian by 12 inches."
        res = self.toolbox.detect_fatal_precursor_signals(narrative)
        self.assertTrue(res["sif_precursor_detected"])
        self.assertIn("Mobile Equipment", res["precursor_tag"])

    def test_dynamic_few_shot_injection(self):
        # Log a test override
        self.storage.log_override(RiskOverride(
            report_id="NM-DYNAMIC-TEST",
            original_risk=RiskLevel.LOW,
            overridden_risk=RiskLevel.HIGH,
            safety_officer_id="SO-Test",
            override_reason="Unshielded live blade presents severe cut hazard",
        ))
        prompt_text = self.few_shot_manager.format_exemplars_for_prompt()
        self.assertIn("Unshielded live blade", prompt_text)
        self.assertIn("NM-DYNAMIC-TEST", prompt_text)

    def test_agent_orchestrator_end_to_end(self):
        trace = self.agent.analyze_report(
            raw_text="Operator slipped on minor puddle of clean water near drinking fountain. Held railing, no injury.",
            department="Office & Admin",
            location="Breakroom Corridor",
        )
        self.assertIsNotNone(trace.final_enriched_report)
        self.assertGreater(len(trace.tool_calls), 0)
        self.assertIn("detect_fatal_precursor_signals", [t.tool_name for t in trace.tool_calls])
        self.assertIn("calculate_hazard_severity_index", [t.tool_name for t in trace.tool_calls])
        self.assertEqual(trace.final_enriched_report.effective_risk_level, RiskLevel.LOW)


if __name__ == "__main__":
    unittest.main()
