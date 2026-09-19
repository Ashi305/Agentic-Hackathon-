"""
Person 3: Safety KPIs & Metric Velocity Engine
Calculates real-time performance indicators and safety velocity metrics.
"""
from typing import Dict, Any
import pandas as pd


class SafetyMetricsCalculator:
    """Computes high-level KPIs for safety executive dashboards."""

    @classmethod
    def compute_summary_kpis(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates core KPI figures for dashboard metric cards."""
        if df.empty:
            return {
                "total_reports": 0,
                "high_risk_count": 0,
                "high_risk_pct": 0.0,
                "sif_escalations": 0,
                "avg_risk_score": 0.0,
                "override_count": 0,
                "top_hotspot_location": "None",
            }

        total = len(df)
        high_risk = int((df["effective_risk_level"] == "High").sum())
        high_risk_pct = round((high_risk / total) * 100, 1) if total > 0 else 0.0
        escalations = int(df["escalation_potential"].sum()) if "escalation_potential" in df.columns else 0
        avg_score = round(float(df["risk_score"].mean()), 2) if "risk_score" in df.columns else 0.0
        overrides = int(df["has_override"].sum()) if "has_override" in df.columns else 0

        top_location = "N/A"
        if "location_specific" in df.columns and not df["location_specific"].empty:
            top_location = df["location_specific"].value_counts().index[0]

        return {
            "total_reports": total,
            "high_risk_count": high_risk,
            "high_risk_pct": high_risk_pct,
            "sif_escalations": escalations,
            "avg_risk_score": avg_score,
            "override_count": overrides,
            "top_hotspot_location": top_location,
        }
