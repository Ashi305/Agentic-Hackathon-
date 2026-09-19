"""
Person 3: Pattern Analytics & Aggregation Engine Module
Provides cross-report Pandas aggregations, location hotspot analytics,
precursor frequency metrics, and semantic clustering for systemic risk themes.
"""
from .aggregation import SafetyAggregator
from .clustering_themes import PrecursorPatternClusterer
from .metrics import SafetyMetricsCalculator

__all__ = [
    "SafetyAggregator",
    "PrecursorPatternClusterer",
    "SafetyMetricsCalculator",
]
