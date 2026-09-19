"""
Unit tests for Person 3: Analytics, Aggregations, and Thematic Clustering
"""
import unittest
from person1_data_pipeline.storage import SafetyStorage
from person3_analytics.aggregation import SafetyAggregator
from person3_analytics.clustering_themes import PrecursorPatternClusterer
from person3_analytics.metrics import SafetyMetricsCalculator


class TestPerson3Analytics(unittest.TestCase):
    def setUp(self):
        self.storage = SafetyStorage()
        self.storage.seed_initial_data(target_count=42)
        self.df = self.storage.to_dataframe()

    def test_dataframe_columns(self):
        self.assertFalse(self.df.empty)
        self.assertIn("report_id", self.df.columns)
        self.assertIn("effective_risk_level", self.df.columns)
        self.assertIn("hotspot_score" not in self.df.columns, [True])

    def test_risk_distribution(self):
        dist = SafetyAggregator.get_risk_distribution(self.df)
        self.assertFalse(dist.empty)
        levels = dist["effective_risk_level"].tolist()
        self.assertIn("High", levels)
        self.assertAlmostEqual(dist["percentage"].sum(), 100.0, places=0)

    def test_location_hotspots(self):
        hotspots = SafetyAggregator.get_location_hotspots(self.df, top_n=5)
        self.assertFalse(hotspots.empty)
        self.assertIn("hotspot_score", hotspots.columns)
        # Check sorted descending
        scores = hotspots["hotspot_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_recurring_precursors(self):
        top_factors = SafetyAggregator.get_top_recurring_risk_factors(self.df, top_n=5)
        self.assertFalse(top_factors.empty)
        self.assertIn("frequency", top_factors.columns)

    def test_clustering_themes(self):
        clusters = PrecursorPatternClusterer.cluster_reports_by_theme(self.df)
        self.assertGreater(len(clusters), 0)
        self.assertIn("theme_title", clusters[0])
        self.assertGreater(clusters[0]["report_count"], 0)

    def test_kpi_metrics(self):
        kpis = SafetyMetricsCalculator.compute_summary_kpis(self.df)
        self.assertGreaterEqual(kpis["total_reports"], 40)
        self.assertGreaterEqual(kpis["high_risk_count"], 1)


if __name__ == "__main__":
    unittest.main()
