"""
Person 2: Decorated Agentic Tools
Contains fully documented Python tools (@tool) with deterministic mathematical
and analytical logic for the Incident Precursor Agent.
"""
import functools
import re
from typing import Dict, Any, List, Optional
from person1_data_pipeline.schema import RiskLevel, RiskOverride
from person1_data_pipeline.storage import SafetyStorage
from .rag_engine import SafetyRAGEngine


def tool(func):
    """Decorator marking callable functions as agent tools with metadata."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.is_agent_tool = True
    wrapper.tool_name = func.__name__
    wrapper.tool_doc = func.__doc__ or "No documentation provided."
    return wrapper


class SafetyToolbox:
    """Production toolbox providing verifiable algorithmic routines for the reasoning agent."""

    def __init__(self, storage: Optional[SafetyStorage] = None, rag_engine: Optional[SafetyRAGEngine] = None):
        self.storage = storage or SafetyStorage()
        self.rag_engine = rag_engine or SafetyRAGEngine()

    @tool
    def calculate_hazard_severity_index(
        self,
        energy_level: str,
        barrier_redundancy: str,
        exposure_frequency: str,
        uncontrolled_precursors: int = 1
    ) -> Dict[str, Any]:
        """
        Calculates a deterministic quantitative hazard severity score (0.0 - 10.0) based on
        kinetic/chemical/electrical potential, safeguard defense integrity, and precursor density.

        Args:
            energy_level: 'High' (pressurized gas, chemical, high-voltage, mobile plant, height > 4ft),
                          'Medium' (hand tools, light conveyors, low-voltage),
                          'Low' (static desk, ambient water, manual packing).
            barrier_redundancy: 'Defeated' (zero barriers active), 'Degraded' (partial/single barrier), 'Intact' (multiple safeguards).
            exposure_frequency: 'Continuous' (routine path/shift), 'Intermittent' (daily), 'Rare' (monthly).
            uncontrolled_precursors: Count of active uncorrected precursors detected.

        Returns:
            Dictionary with calculated score, calculated risk tier (Low/Medium/High), and escalation flag.
        """
        energy_weights = {"High": 5.0, "Medium": 3.0, "Low": 1.0}
        barrier_weights = {"Defeated": 4.0, "Degraded": 2.0, "Intact": 0.5}
        exposure_weights = {"Continuous": 1.0, "Intermittent": 0.5, "Rare": 0.2}

        e_score = energy_weights.get(energy_level, 3.0)
        b_score = barrier_weights.get(barrier_redundancy, 2.0)
        exp_score = exposure_weights.get(exposure_frequency, 0.5)
        precursor_bump = min(uncontrolled_precursors * 0.4, 1.2)

        raw_score = e_score + b_score + exp_score + precursor_bump
        normalized_score = min(max(round(raw_score, 1), 1.0), 10.0)

        if normalized_score >= 7.0 or (energy_level == "High" and barrier_redundancy == "Defeated"):
            tier = "High"
            escalate = True
        elif normalized_score >= 4.0:
            tier = "Medium"
            escalate = False
        else:
            tier = "Low"
            escalate = False

        return {
            "hazard_severity_score": normalized_score,
            "recommended_risk_tier": tier,
            "escalation_potential": escalate,
            "energy_assessment": energy_level,
            "barrier_status": barrier_redundancy,
        }

    @tool
    def lookup_osha_standards(self, query_terms: str) -> Dict[str, Any]:
        """
        Performs grounded vector/lexical retrieval across OSHA General Industry regulations
        and NFPA guidelines to retrieve relevant compliance benchmarks and mandatory engineering controls.

        Args:
            query_terms: Key hazard concepts e.g., 'acid spray shield', 'forklift blind corner', 'LOTO bypass'.

        Returns:
            Dictionary containing matched OSHA codes, titles, mandatory controls, and relevance scores.
        """
        results = self.rag_engine.retrieve(query_terms, top_k=2)
        citations = []
        for chunk, score in results:
            citations.append({
                "standard": chunk.standard_code,
                "title": chunk.title,
                "benchmark_severity": chunk.severity_level,
                "relevance_score": round(score, 3),
                "mandatory_controls": chunk.mandatory_controls,
            })
        return {
            "query": query_terms,
            "matched_standards": citations,
            "primary_citation": citations[0]["standard"] if citations else "OSHA General Duty Clause 5(a)(1)"
        }

    @tool
    def detect_fatal_precursor_signals(self, narrative_text: str) -> Dict[str, Any]:
        """
        Analyzes narrative text using industrial precursor pattern heuristics
        to identify high-energy early warnings of potential Serious Injuries or Fatalities (SIF).

        Args:
            narrative_text: Raw near-miss observation text.

        Returns:
            Dictionary listing identified precursor triggers, severity category, and SIF potential.
        """
        text = narrative_text.lower()
        signals = []

        patterns = {
            "Mobile Equipment / Pedestrian Interaction": [r"\bforklift\b", r"\breach truck\b", r"\byard truck\b", r"\bpedestrian\b", r"\bblind corner\b", r"\bcrossing\b"],
            "LOTO & Energy Isolation Defeat": [r"\bloto\b", r"\binterlock\b", r"\bbypass\b", r"\bkeyed open\b", r"\bdefeat\b", r"\bteach pendant\b"],
            "Working at Height / Falling Object": [r"\bcatwalk\b", r"\bmezzanine\b", r"\bdrop\b", r"\bplummet\b", r"\bracking\b", r"\b4 feet\b", r"\btoe-board\b"],
            "Pressurized / Corrosive Chemical Contact": [r"\bacid\b", r"\bflange\b", r"\bspray shield\b", r"\bleak\b", r"\bh2so4\b", r"\bface shield\b"],
            "Electrical Arc Flash / High Voltage": [r"\barc flash\b", r"\bswitchgear\b", r"\b480v\b", r"\bbreaker\b", r"\bracking\b", r"\benergized\b"],
            "Confined Space & Toxic Atmosphere": [r"\bh2s\b", r"\bconfined space\b", r"\btoxic\b", r"\bfume hood\b", r"\bcalibrated\b", r"\bbump test\b"],
            "Explosion & Flammable Vapors": [r"\bgrounding\b", r"\bstatic\b", r"\bflammable\b", r"\btoluene\b", r"\bnitrogen purge\b", r"\bspark\b"],
        }

        for category, regex_list in patterns.items():
            matches = [term for term in regex_list if re.search(term, text)]
            if len(matches) >= 2 or (category == "LOTO & Energy Isolation Defeat" and len(matches) >= 1):
                signals.append({"precursor_domain": category, "triggers": matches})

        is_sif_precursor = len(signals) > 0
        return {
            "sif_precursor_detected": is_sif_precursor,
            "active_signals": signals,
            "signal_count": len(signals),
            "precursor_tag": signals[0]["precursor_domain"] if signals else "Standard Observation"
        }

    @tool
    def query_historical_precedents(self, keywords: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches historical near-miss database for similar past incidents to compare
        how previous hazards were scored and whether human overrides occurred.

        Args:
            keywords: Hazard terms to match against historical database.
            limit: Maximum past reports to return.

        Returns:
            List of matching past incident summaries with their effective risk scores and overrides.
        """
        all_reports = self.storage.get_enriched_reports()
        tokens = re.findall(r"\w+", keywords.lower())
        
        matches = []
        for item in all_reports:
            content = f"{item.report.raw_text} {item.extraction.primary_hazard}".lower()
            overlap = sum(1 for t in tokens if t in content)
            if overlap > 0:
                matches.append((overlap, {
                    "id": item.report.id,
                    "hazard": item.extraction.primary_hazard,
                    "department": item.report.department,
                    "effective_risk": item.effective_risk_level.value,
                    "has_override": len(item.overrides) > 0,
                    "match_strength": overlap,
                }))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    @tool
    def record_safety_officer_override(
        self,
        report_id: str,
        original_risk: str,
        overridden_risk: str,
        safety_officer_id: str,
        override_reason: str
    ) -> Dict[str, Any]:
        """
        Compulsory Add-on Tool:
        Saves a safety officer's human correction into SQLite for future dynamic few-shot prompt injection.

        Args:
            report_id: ID of the report being modified.
            original_risk: 'Low', 'Medium', or 'High'.
            overridden_risk: New calibrated risk 'Low', 'Medium', or 'High'.
            safety_officer_id: Badge or name of the officer.
            override_reason: Justification explaining the operational/precursor reason.

        Returns:
            Status confirmation and updated active few-shot count.
        """
        ov = RiskOverride(
            report_id=report_id,
            original_risk=RiskLevel(original_risk),
            overridden_risk=RiskLevel(overridden_risk),
            safety_officer_id=safety_officer_id,
            override_reason=override_reason,
        )
        self.storage.log_override(ov)
        recent = self.storage.get_recent_overrides(limit=5)
        return {
            "status": "success",
            "message": f"Risk level for {report_id} updated from {original_risk} to {overridden_risk}.",
            "active_few_shot_pool_size": len(recent),
        }
