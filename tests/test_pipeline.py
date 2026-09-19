"""
Unit tests for Person 1: Data Pipeline & Storage Layer
"""
import unittest
import os
import tempfile
from person1_data_pipeline.schema import (
    NearMissReport,
    StructuredExtraction,
    RiskAssessment,
    RiskOverride,
    EnrichedReport,
    RiskLevel,
    HazardCategory,
)
from person1_data_pipeline.storage import SafetyStorage


class TestPerson1Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_warehouse.db")
        self.json_path = os.path.join(self.temp_dir, "test_reports.json")
        self.storage = SafetyStorage(db_path=self.db_path, json_backup_path=self.json_path)

    def test_schema_instantiation(self):
        rep = NearMissReport(
            id="NM-TEST-001",
            facility="Test Plant",
            department="Chemical Processing",
            location_specific="Bay 1",
            raw_text="Test incident narrative",
        )
        self.assertEqual(rep.id, "NM-TEST-001")
        self.assertEqual(rep.facility, "Test Plant")

    def test_save_and_retrieve_enriched_report(self):
        rep = NearMissReport(
            id="NM-TEST-002",
            facility="Test Facility",
            department="Logistics",
            location_specific="Aisle 1",
            raw_text="Forklift near pedestrian",
        )
        ext = StructuredExtraction(
            report_id="NM-TEST-002",
            primary_hazard="Forklift Blind Intersection",
            hazard_category=HazardCategory.MECHANICAL_CRUSH,
            precursor_events=["Obstructed convex mirror"],
            recommended_mitigation="Realight mirror",
        )
        ass = RiskAssessment(
            report_id="NM-TEST-002",
            risk_level=RiskLevel.HIGH,
            risk_score=9.0,
            rationale="High crush risk",
        )
        enriched = EnrichedReport(report=rep, extraction=ext, assessment=ass, overrides=[], effective_risk_level=RiskLevel.HIGH)

        self.storage.save_enriched_report(enriched)
        loaded = self.storage.get_enriched_reports()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].report.id, "NM-TEST-002")
        self.assertEqual(loaded[0].effective_risk_level, RiskLevel.HIGH)

    def test_risk_override_logging(self):
        ov = RiskOverride(
            report_id="NM-TEST-002",
            original_risk=RiskLevel.LOW,
            overridden_risk=RiskLevel.HIGH,
            safety_officer_id="SO-Test",
            override_reason="Overhead dropped load hazard makes this high risk",
        )
        self.storage.log_override(ov)
        overrides = self.storage.get_recent_overrides(limit=5)
        self.assertEqual(len(overrides), 1)
        self.assertEqual(overrides[0].overridden_risk, RiskLevel.HIGH)
        self.assertIn("Overhead dropped load", overrides[0].override_reason)


if __name__ == "__main__":
    unittest.main()
