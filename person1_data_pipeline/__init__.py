"""
Person 1: Data Pipeline & Synthetic Data Module
Defines the central schema, synthetic near-miss generator, and persistence storage layer.
"""
from .schema import (
    RiskLevel,
    HazardCategory,
    NearMissReport,
    StructuredExtraction,
    RiskAssessment,
    RiskOverride,
    EnrichedReport,
)
from .storage import SafetyStorage
from .synthetic_generator import generate_synthetic_reports

__all__ = [
    "RiskLevel",
    "HazardCategory",
    "NearMissReport",
    "StructuredExtraction",
    "RiskAssessment",
    "RiskOverride",
    "EnrichedReport",
    "SafetyStorage",
    "generate_synthetic_reports",
]
