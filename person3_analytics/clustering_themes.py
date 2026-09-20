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
            "precursor_significance": "Direct precursor to mechanical crush, amputation, and electrical arc blast.",
            "recommended_mitigation": (
                "Conduct a full LOTO (Lockout/Tagout) compliance audit per OSHA 1910.147. "
                "Verify all interlock circuits are functional and tamper-evident. "
                "Replace any bypassed safety guards with hardwired, non-defeatable interlocks "
                "and retrain affected operators on zero-energy verification procedures."
            )
        },
        "Mobile Equipment & Pedestrian Corridor Congestion": {
            "keywords": ["forklift", "truck", "aisle", "blind", "pedestrian", "corner", "mirror", "horn", "trailer", "dock"],
            "precursor_significance": "Leading cause of struck-by fatalities in warehousing and logistics hubs.",
            "recommended_mitigation": (
                "Install physical pedestrian barriers, convex mirrors, and LED warning beacons "
                "at all blind corners and dock intersections. Enforce mandatory horn-at-corner "
                "protocols and deploy proximity-sensor speed limiters on all powered industrial trucks "
                "per OSHA 1910.178."
            )
        },
        "Pressurized Toxic / Corrosive Fluid Containment": {
            "keywords": ["acid", "flange", "leak", "spray", "shield", "pump", "gasket", "fume", "solvent", "chemical", "h2so4"],
            "precursor_significance": "Imminent precursor to severe chemical burns and toxic inhalation injuries.",
            "recommended_mitigation": (
                "Inspect and replace all degraded flange gaskets and install bolt-on spray shields "
                "on high-pressure chemical transfer lines. Deploy secondary containment drip trays "
                "under all acid pumps and verify emergency eyewash/shower stations are within "
                "10-second travel distance per ANSI Z358.1."
            )
        },
        "Working at Height & Unsecured Falling Objects": {
            "keywords": ["catwalk", "mezzanine", "dropped", "tool", "fall", "anchor", "roof", "toe-board", "ladder", "height"],
            "precursor_significance": "High-velocity blunt force trauma hazard for ground-level personnel.",
            "recommended_mitigation": (
                "Install toe-boards, tool-tethering lanyards, and mesh screens on all elevated "
                "platforms and catwalks. Verify fall-arrest anchor points are rated and inspected "
                "per OSHA 1910.28. Establish mandatory tool-inventory checklists before and after "
                "every elevated work task and barricade ground-level exclusion zones beneath active work areas."
            )
        },
        "Atmospheric Testing & Confined Space Verification": {
            "keywords": ["confined", "h2s", "gas", "detector", "calibration", "bump", "meter", "atmosphere", "basin"],
            "precursor_significance": "Zero-warning asphyxiation and toxic poisoning precursor.",
            "recommended_mitigation": (
                "Enforce mandatory pre-entry atmospheric testing with bump-tested, calibrated "
                "4-gas monitors (O2, LEL, CO, H2S). Require a dedicated attendant and rescue "
                "tripod at every confined space entry per OSHA 1910.146. Implement a digital "
                "permit-to-work system that blocks entry if atmospheric readings are out of range."
            )
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
                        "recommended_mitigation": row.get("recommended_mitigation", ""),
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
                    "recommended_mitigation": row.get("recommended_mitigation", ""),
                })

        results = []
        for theme, items in clusters.items():
            if not items:
                continue
            item_df = pd.DataFrame(items)
            high_count = (item_df["risk_level"] == "High").sum()
            avg_score = round(item_df["risk_score"].mean(), 1)

            # Collect unique, non-empty mitigations from DB (prefer high-risk reports)
            sorted_items = sorted(items, key=lambda x: x.get("risk_score", 0), reverse=True)
            db_mitigations = []
            seen = set()
            for item in sorted_items:
                m = (item.get("recommended_mitigation") or "").strip()
                if m and m.lower() not in seen and m.lower() != "inspect area.":
                    seen.add(m.lower())
                    db_mitigations.append(m)
                if len(db_mitigations) >= 3:
                    break

            results.append({
                "theme_title": theme,
                "report_count": len(items),
                "high_risk_count": int(high_count),
                "avg_risk_score": avg_score,
                "precursor_significance": cls.THEME_TAXONOMY[theme]["precursor_significance"],
                "representative_reports": [i["report_id"] for i in items[:4]],
                "affected_departments": list(set(i["department"] for i in items if i["department"]))[:3],
                "db_mitigations": db_mitigations,
            })

        results.sort(key=lambda x: (x["high_risk_count"], x["avg_risk_score"]), reverse=True)
        return results

    @classmethod
    def generate_executive_theme_summary(cls, clusters: List[Dict[str, Any]], top_n: int = 3) -> str:
        """Generates an automated executive briefing summarizing systemic industrial vulnerabilities."""
        if not clusters:
            return "No systemic precursor themes detected. Operations are within baseline variance."

        summary_lines = ["# 🚨 Executive Precursor Briefing\n"]

        for i, cluster in enumerate(clusters[:top_n], start=1):
            theme_title = cluster["theme_title"]

            # Prefer real mitigations pulled from the database;
            # fall back to taxonomy-level defaults only when DB has none
            db_mitigations = cluster.get("db_mitigations", [])
            if db_mitigations:
                mitigation_block = "\n".join(
                    f"  {j}. {m}" for j, m in enumerate(db_mitigations, start=1)
                )
            else:
                fallback = cls.THEME_TAXONOMY.get(theme_title, {}).get(
                    "recommended_mitigation",
                    "Conduct a targeted engineering audit of affected areas and review existing safeguard controls."
                )
                mitigation_block = f"  1. {fallback}"

            summary_lines.append(
                f"## Priority {i}: {theme_title}\n"
                f"Across **{cluster['report_count']}** logged observations, "
                f"**{cluster['high_risk_count']} cases were classified as High Risk** "
                f"(Average Severity: {cluster['avg_risk_score']}/10.0).\n\n"
                f"*Critical Precursor Implication:* {cluster['precursor_significance']}\n\n"
                f"*Primary Operating Sectors Affected:* {', '.join(cluster['affected_departments'])}.\n\n"
                f"**Recommended Mitigations (from report analysis):**\n{mitigation_block}"
            )

        return "\n\n---\n\n".join(summary_lines)
