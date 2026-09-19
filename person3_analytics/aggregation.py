"""
Person 3: Pandas Cross-Report Aggregation Routines
Calculates location hotspots, department distributions, recurring risk factors,
and human override alignment analytics across the near-miss dataset.
"""
from typing import Dict, Any, List
import pandas as pd
from person1_data_pipeline.storage import SafetyStorage


class SafetyAggregator:
    """Performs statistical and trend aggregations on enriched safety observations."""

    def __init__(self, storage: SafetyStorage):
        self.storage = storage

    def load_data(self) -> pd.DataFrame:
        """Returns the current enriched dataset as a Pandas DataFrame."""
        return self.storage.to_dataframe()

    @classmethod
    def get_risk_distribution(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Returns distribution counts and proportions across Low, Medium, and High risk levels."""
        if df.empty:
            return pd.DataFrame(columns=["effective_risk_level", "count", "percentage"])
        
        dist = df["effective_risk_level"].value_counts().reset_index()
        dist.columns = ["effective_risk_level", "count"]
        
        # Ensure standard ordering: High, Medium, Low
        order_map = {"High": 1, "Medium": 2, "Low": 3}
        dist["order"] = dist["effective_risk_level"].map(order_map).fillna(4)
        dist = dist.sort_values("order").drop(columns=["order"])
        
        total = dist["count"].sum()
        dist["percentage"] = (dist["count"] / total * 100).round(1)
        return dist

    @classmethod
    def get_location_hotspots(cls, df: pd.DataFrame, top_n: int = 7) -> pd.DataFrame:
        """
        Calculates location-level risk hotspots by combining total incident frequency,
        average severity score, and count of high-risk precursors.
        """
        if df.empty:
            return pd.DataFrame(columns=["location_specific", "incident_count", "avg_risk_score", "high_risk_count", "hotspot_score"])

        grouped = df.groupby("location_specific").agg(
            incident_count=("report_id", "count"),
            avg_risk_score=("risk_score", "mean"),
            high_risk_count=("effective_risk_level", lambda s: (s == "High").sum()),
            escalation_count=("escalation_potential", lambda s: s.sum()),
        ).reset_index()

        grouped["avg_risk_score"] = grouped["avg_risk_score"].round(2)
        # Hotspot score: Weighted composite of frequency and severity
        grouped["hotspot_score"] = (
            (grouped["incident_count"] * 1.5) +
            (grouped["avg_risk_score"] * 2.0) +
            (grouped["high_risk_count"] * 3.0)
        ).round(1)

        return grouped.sort_values("hotspot_score", ascending=False).head(top_n)

    @classmethod
    def get_department_risk_matrix(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Pivots department against risk level for cross-departmental benchmarking."""
        if df.empty:
            return pd.DataFrame()

        pivot = pd.crosstab(
            df["department"],
            df["effective_risk_level"],
            margins=True,
            margins_name="Total"
        ).reset_index()
        
        # Calculate high-risk percentage per department
        if "High" in pivot.columns and "Total" in pivot.columns:
            pivot["High_Risk_Pct"] = ((pivot["High"] / pivot["Total"]) * 100).round(1)
        
        return pivot.sort_values("Total", ascending=False)

    @classmethod
    def get_top_recurring_risk_factors(cls, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Flattens nested precursor lists across reports to rank top recurring incident precursors."""
        if df.empty or "precursor_events" not in df.columns:
            return pd.DataFrame(columns=["precursor", "frequency", "associated_hazard_category"])

        precursor_rows = []
        for _, row in df.iterrows():
            events = row.get("precursor_events", [])
            cat = row.get("hazard_category", "General")
            if isinstance(events, list):
                for ev in events:
                    if ev and isinstance(ev, str):
                        precursor_rows.append({"precursor": ev.strip(), "hazard_category": cat})

        if not precursor_rows:
            return pd.DataFrame(columns=["precursor", "frequency", "hazard_category"])

        precursor_df = pd.DataFrame(precursor_rows)
        top = precursor_df.groupby("precursor").agg(
            frequency=("precursor", "count"),
            primary_hazard_category=("hazard_category", lambda s: s.mode()[0] if not s.empty else "General")
        ).reset_index().sort_values("frequency", ascending=False).head(top_n)

        return top

    @classmethod
    def get_failed_safeguards_frequency(cls, df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
        """Aggregates defeated or missing safety barriers (Safeguard Deficit Analysis)."""
        if df.empty or "failed_safeguards" not in df.columns:
            return pd.DataFrame(columns=["safeguard", "failure_count"])

        safeguards = []
        for _, row in df.iterrows():
            items = row.get("failed_safeguards", [])
            if isinstance(items, list):
                for item in items:
                    if item and isinstance(item, str):
                        safeguards.append(item.strip())

        if not safeguards:
            return pd.DataFrame(columns=["safeguard", "failure_count"])

        s_series = pd.Series(safeguards)
        res = s_series.value_counts().reset_index()
        res.columns = ["safeguard", "failure_count"]
        return res.head(top_n)

    @classmethod
    def get_override_audit_summary(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Summarizes human safety officer override activity and calibration impact."""
        if df.empty or "has_override" not in df.columns:
            return {
                "total_reports": 0,
                "overridden_count": 0,
                "override_percentage": 0.0,
                "escalations": 0,
                "de_escalations": 0,
            }

        overridden = df[df["has_override"] == True]
        total = len(df)
        count_ov = len(overridden)

        # Risk order for comparison
        tier_weight = {"Low": 1, "Medium": 2, "High": 3}
        escalations = 0
        de_escalations = 0

        for _, row in overridden.iterrows():
            orig_w = tier_weight.get(row.get("original_risk_level"), 2)
            eff_w = tier_weight.get(row.get("effective_risk_level"), 2)
            if eff_w > orig_w:
                escalations += 1
            elif eff_w < orig_w:
                de_escalations += 1

        return {
            "total_reports": total,
            "overridden_count": count_ov,
            "override_percentage": round((count_ov / total * 100), 1) if total > 0 else 0.0,
            "escalations_by_officer": escalations,
            "de_escalations_by_officer": de_escalations,
        }
