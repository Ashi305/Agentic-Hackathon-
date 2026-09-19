"""
Person 3: Recurring Theme Extraction & Precursor Clustering Engine
Identifies high-impact systemic hazard clusters across safety reports and generates
concise executive risk summaries for senior leadership.
"""
from typing import List, Dict, Any
import pandas as pd


class PrecursorPatternClusterer:
    """Discovers recurring precursor themes and synthesizes systemic hazard insights."""

    THEME_TAXONOMY = {
        "Hazardous Energy Isolation & Interlock Bypass": {
            "keywords": ["loto", "interlock", "bypass", "guard", "switchgear", "circuit", "lockout", "ram", "curtain", "shunt"],
            "precursor_significance": "Direct precursor to mechanical crush, amputation, and electrical arc blast."
        },
        "Mobile Equipment & Pedestrian Corridor Congestion": {
            "keywords": ["forklift", "truck", "aisle", "blind", "pedestrian", "corner", "mirror", "horn", "trailer", "dock"],
            "precursor_significance": "Leading cause of struck-by fatalities in warehousing and logistics hubs."
        },
        "Pressurized Toxic / Corrosive Fluid Containment": {
            "keywords": ["acid", "flange", "leak", "spray", "shield", "pump", "gasket", "fume", "solvent", "chemical", "h2so4"],
            "precursor_significance": "Imminent precursor to severe chemical burns and toxic inhalation injuries."
        },
        "Working at Height & Unsecured Falling Objects": {
            "keywords": ["catwalk", "mezzanine", "dropped", "tool", "fall", "anchor", "roof", "toe-board", "ladder", "height"],
            "precursor_significance": "High-velocity blunt force trauma hazard for ground-level personnel."
        },
        "Atmospheric Testing & Confined Space Verification": {
            "keywords": ["confined", "h2s", "gas", "detector", "calibration", "bump", "meter", "atmosphere", "basin"],
            "precursor_significance": "Zero-warning asphyxiation and toxic poisoning precursor."
        }
    }

    @classmethod
    def cluster_reports_by_theme(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Groups reports into strategic precursor themes based on narrative and extraction keywords."""
        if df.empty:
            return []

        clusters: Dict[str, List[Dict[str, Any]]] = {theme: [] for theme in cls.THEME_TAXONOMY}
        uncategorized: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            text = f"{row.get('raw_text', '')} {row.get('primary_hazard', '')} {' '.join(row.get('precursor_events', []))}".lower()
            matched = False
            for theme, config in cls.THEME_TAXONOMY.items():
                if any(kw in text for kw in config["keywords"]):
                    clusters[theme].append({
                        "report_id": row.get("report_id"),
                        "department": row.get("department"),
                        "location": row.get("location_specific"),
                        "risk_level": row.get("effective_risk_level"),
                        "risk_score": row.get("risk_score", 5.0),
                        "primary_hazard": row.get("primary_hazard"),
                    })
                    matched = True
                    break
            if not matched:
                uncategorized.append({
                    "report_id": row.get("report_id"),
                    "department": row.get("department"),
                    "location": row.get("location_specific"),
                    "risk_level": row.get("effective_risk_level"),
                    "risk_score": row.get("risk_score", 5.0),
                    "primary_hazard": row.get("primary_hazard"),
                })

        results = []
        for theme, items in clusters.items():
            if not items:
                continue
            item_df = pd.DataFrame(items)
            high_count = (item_df["risk_level"] == "High").sum()
            avg_score = round(item_df["risk_score"].mean(), 1)
            results.append({
                "theme_title": theme,
                "report_count": len(items),
                "high_risk_count": int(high_count),
                "avg_risk_score": avg_score,
                "precursor_significance": cls.THEME_TAXONOMY[theme]["precursor_significance"],
                "representative_reports": [i["report_id"] for i in items[:4]],
                "affected_departments": list(set(i["department"] for i in items if i["department"]))[:3],
            })

        results.sort(key=lambda x: (x["high_risk_count"], x["avg_risk_score"]), reverse=True)
        return results

    @classmethod
    def generate_executive_theme_summary(cls, clusters: List[Dict[str, Any]]) -> str:
        """Generates an automated executive briefing summarizing systemic industrial vulnerabilities."""
        if not clusters:
            return "No systemic precursor themes detected. Operations are within baseline variance."

        top_cluster = clusters[0]
        summary_lines = [
            f"**EXECUTIVE PRECURSOR ALERT:** Systemic hazard pattern detected under **'{top_cluster['theme_title']}'**.",
            f"Across {top_cluster['report_count']} logged observations, **{top_cluster['high_risk_count']} cases were classified as High Risk** (Average Severity: {top_cluster['avg_risk_score']}/10.0).",
            f"*Critical Precursor Implication:* {top_cluster['precursor_significance']}",
            f"*Primary Operating Sectors Affected:* {', '.join(top_cluster['affected_departments'])}.",
            f"**Recommended Immediate Priority:** Deploy targeted engineering audits to inspect interlocks, physical barriers, and verify secondary containment in affected zones before precursor recurrence escalates into a recordable incident."
        ]
        return "\n\n".join(summary_lines)
