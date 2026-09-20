"""
Person 4: Self-Contained Agent Service & Backend Bridge
Seamlessly integrates with Person 1's SQLite Storage and Person 2's SafetyReportAgent,
FewShotManager, and RAGEngine, while providing a robust offline deterministic fallback.
"""
import os
import re
import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from person1_data_pipeline.schema import (
    NearMissReport,
    StructuredExtraction,
    RiskAssessment,
    RiskLevel,
    HazardCategory,
    EnrichedReport,
    RiskOverride,
)
from person1_data_pipeline.storage import SafetyStorage


class ToolExecutionRecord(BaseModel):
    tool_name: str
    input_args: Dict[str, Any]
    output_result: Dict[str, Any]
    execution_time_ms: float


class AgentExecutionTrace(BaseModel):
    report_id: str
    steps_executed: List[str] = Field(default_factory=list)
    tool_calls: List[ToolExecutionRecord] = Field(default_factory=list)
    retrieved_citations: List[str] = Field(default_factory=list)
    dynamic_few_shot_applied: List[str] = Field(default_factory=list)
    llm_provider_used: str = "Deterministic Expert Engine (Offline Guaranteed)"
    raw_extraction_data: Optional[Dict[str, Any]] = None
    raw_classification_data: Optional[Dict[str, Any]] = None
    final_enriched_report: Optional[EnrichedReport] = None


class FrontendFewShotManager:
    """Bridges Person 1's SQLite storage with Person 2's FewShotManager."""
    def __init__(self, storage: Optional[SafetyStorage] = None):
        self.storage = storage or SafetyStorage()
        try:
            from person2_llm_agent.few_shot_manager import FewShotManager
            self.backend_fsm = FewShotManager()
            self._sync_backend_fsm()
        except Exception:
            self.backend_fsm = None

    def _sync_backend_fsm(self):
        """Pre-loads recent SQLite overrides into Person 2's in-memory FewShotManager."""
        if not self.backend_fsm:
            return
        self.backend_fsm.clear()
        for ov in self.get_latest_exemplars(max_examples=10):
            self.backend_fsm.add_correction(
                report=ov.report_id,
                original_label=ov.original_risk.value,
                corrected_label=ov.overridden_risk.value,
                reason=ov.override_reason,
            )

    def log_override(self, override: RiskOverride):
        """Commits override to SQLite and synchronizes Person 2's FewShotManager."""
        self.storage.log_override(override)
        if self.backend_fsm:
            self.backend_fsm.add_correction(
                report=override.report_id,
                original_label=override.original_risk.value,
                corrected_label=override.overridden_risk.value,
                reason=override.override_reason,
            )

    def get_latest_exemplars(self, max_examples: int = 5) -> List[RiskOverride]:
        try:
            return self.storage.get_recent_overrides(limit=max_examples)
        except Exception:
            return []

    def format_exemplars_for_prompt(self, overrides: Optional[List[RiskOverride]] = None) -> str:
        if overrides is None:
            overrides = self.get_latest_exemplars()
        if not overrides:
            return "No previous human overrides logged. Follow standard baseline classification."
        lines = [
            "ACTIVE HUMAN-IN-THE-LOOP FEEDBACK EXAMPLARS:",
            "The safety committee has established the following precedence rules based on recent overrides:\n"
        ]
        for i, ov in enumerate(overrides, start=1):
            direction = f"{ov.original_risk.value} -> {ov.overridden_risk.value}"
            lines.append(f"[{i}] Correction ({direction}) by {ov.safety_officer_id}:")
            lines.append(f"    Report Ref: {ov.report_id}")
            lines.append(f"    Guiding Rationale: \"{ov.override_reason}\"")
            lines.append(f"    Action: When similar precursor conditions appear, classify as '{ov.overridden_risk.value}'.\n")
        return "\n".join(lines)


class FrontendAgentService:
    """Unified reasoning service connecting Person 2's LLM engine to the frontend."""

    def __init__(self, storage: Optional[SafetyStorage] = None):
        self.storage = storage or SafetyStorage()
        self.few_shot_manager = FrontendFewShotManager(storage=self.storage)

    def parse_pdf_document(self, pdf_source: Any, filename: str = "document.pdf") -> Dict[str, Any]:
        """
        Connects Person 1's parser component to extract text and structure
        safety report entities from uploaded PDFs.
        """
        from person1_data_pipeline.parser import parse_pdf_report
        return parse_pdf_report(pdf_source, filename=filename)

    def batch_process_pdf_directory(self, directory_path: Optional[str] = None) -> int:
        """
        Executes Person 1's batch PDF directory processing pipeline.
        """
        from person1_data_pipeline.parser import process_pdf_directory
        if not directory_path:
            directory_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "person1_data_pipeline",
                "data",
                "pdf_reports"
            )
        return process_pdf_directory(directory_path)

    def analyze_report(
        self,
        raw_text: str,
        department: str,
        location: str,
        facility: str = "Industrial Complex Alpha",
        reporter_role: str = "Field Observer",
        equipment: str = "N/A",
        report_id: Optional[str] = None,
    ) -> AgentExecutionTrace:
        trace = AgentExecutionTrace(report_id=report_id or f"NM-LIVE-{int(time.time())}")
        report_id = trace.report_id

        # Query recent overrides for dynamic few-shot injection
        recent_overrides = self.few_shot_manager.get_latest_exemplars(max_examples=3)
        for ov in recent_overrides:
            trace.dynamic_few_shot_applied.append(
                f"[{ov.report_id}] {ov.original_risk.value} -> {ov.overridden_risk.value} ('{ov.override_reason}')"
            )

        # Attempt Person 2's live LLM reasoning if valid API key is present
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if api_key and len(api_key) > 10:
            try:
                enriched = self._run_person2_llm_agent(
                    raw_text=raw_text,
                    department=department,
                    location=location,
                    facility=facility,
                    reporter_role=reporter_role,
                    equipment=equipment,
                    report_id=report_id,
                    trace=trace,
                )
                if enriched:
                    trace.final_enriched_report = enriched
                    self.storage.save_enriched_report(enriched)
                    return trace
            except Exception as e:
                trace.steps_executed.append(f"Notice: Person 2 live LLM encountered error ({e}). Using deterministic expert engine fallback.")

        # Robust Deterministic Expert Heuristic Synthesis (Always runs successfully)
        enriched = self._run_deterministic_agent(
            raw_text=raw_text,
            department=department,
            location=location,
            facility=facility,
            reporter_role=reporter_role,
            equipment=equipment,
            report_id=report_id,
            recent_overrides=recent_overrides,
            trace=trace,
        )
        trace.final_enriched_report = enriched
        self.storage.save_enriched_report(enriched)
        return trace

    def _run_person2_llm_agent(self, raw_text, department, location, facility, reporter_role, equipment, report_id, trace) -> Optional[EnrichedReport]:
        """Runs Person 2's SafetyReportAgent directly."""
        from person2_llm_agent.agent_orchestrator import SafetyReportAgent
        from person2_llm_agent.few_shot_manager import FewShotManager

        fsm = FewShotManager()
        for ov in self.few_shot_manager.get_latest_exemplars(max_examples=5):
            fsm.add_correction(
                report=ov.report_id,
                original_label=ov.original_risk.value,
                corrected_label=ov.overridden_risk.value,
                reason=ov.override_reason,
            )

        p2_agent = SafetyReportAgent(few_shot_manager=fsm)

        # Step 1: Extraction
        trace.steps_executed.append("1. Invoking Person 2 'extract_information' LLM prompt.")
        t0 = time.time()
        extracted = p2_agent.extract_information(raw_text)
        trace.raw_extraction_data = extracted
        trace.tool_calls.append(ToolExecutionRecord(
            tool_name="extract_information",
            input_args={"report_length": len(raw_text)},
            output_result=extracted,
            execution_time_ms=round((time.time() - t0) * 1000, 2),
        ))

        # Step 2: RAG Retrieval of human corrections
        trace.steps_executed.append("2. Invoking Person 2 'RAGEngine.retrieve' for relevant historical corrections.")
        t0 = time.time()
        retrieved_corrections = p2_agent.rag_engine.retrieve(raw_text, top_k=3)
        trace.tool_calls.append(ToolExecutionRecord(
            tool_name="rag_engine.retrieve",
            input_args={"query_preview": raw_text[:60]},
            output_result={"matched_corrections": retrieved_corrections},
            execution_time_ms=round((time.time() - t0) * 1000, 2),
        ))

        # Step 3: Classification
        trace.steps_executed.append("3. Invoking Person 2 'classify_risk' LLM prompt with few-shot injection.")
        t0 = time.time()
        classification = p2_agent.classify_risk(raw_text, extracted)
        trace.raw_classification_data = classification
        trace.tool_calls.append(ToolExecutionRecord(
            tool_name="classify_risk",
            input_args={"extracted_keys": list(extracted.keys())},
            output_result=classification,
            execution_time_ms=round((time.time() - t0) * 1000, 2),
        ))

        trace.llm_provider_used = "Person 2 Gemini Agent (Live Cloud Inference)"

        # Map to central schema
        risk_map = {"LOW": RiskLevel.LOW, "MEDIUM": RiskLevel.MEDIUM, "HIGH": RiskLevel.HIGH}
        risk_lvl = risk_map.get(str(classification.get("risk_level", "MEDIUM")).upper(), RiskLevel.MEDIUM)
        confidence = float(classification.get("confidence", 0.90))
        reason = str(classification.get("reason", "Evaluated by Person 2 LLM Agent."))
        human_review = bool(classification.get("human_review_required", False))

        hazards = extracted.get("hazards", [])
        primary_hazard = hazards[0] if hazards else "Unspecified Workplace Hazard"
        risk_factors = extracted.get("risk_factors", [])
        missing_info = extracted.get("missing_information", [])
        consequences = extracted.get("potential_consequences", [])

        # Score mapping
        score = 8.8 if risk_lvl == RiskLevel.HIGH else (5.8 if risk_lvl == RiskLevel.MEDIUM else 2.8)

        # Citations
        citations = ["OSHA General Industry 1910.147 (Energy Control)" if risk_lvl == RiskLevel.HIGH else "OSHA 1910.22 (General Safety)"]
        trace.retrieved_citations = citations

        raw_rep = NearMissReport(
            id=report_id,
            facility=facility,
            department=department,
            location_specific=location,
            reporter_role=reporter_role,
            raw_text=raw_text,
            equipment_involved=equipment,
            environmental_factors="Assessed by Person 2 Agent",
            immediate_action_taken="Processed via LLM pipeline",
        )

        extraction_obj = StructuredExtraction(
            report_id=report_id,
            primary_hazard=primary_hazard,
            hazard_category=HazardCategory.MECHANICAL_CRUSH if "press" in raw_text.lower() or "forklift" in raw_text.lower() else HazardCategory.SLIP_TRIP_FALL,
            precursor_events=risk_factors or ["Unsafe condition flagged by model"],
            affected_assets=["Operating Personnel", equipment],
            failed_safeguards=consequences or ["Missing/bypassed controls"],
            recommended_mitigation=f"Safety Review: {missing_info[0] if missing_info else 'Inspect zone controls.'}",
        )

        assessment_obj = RiskAssessment(
            report_id=report_id,
            risk_level=risk_lvl,
            risk_score=score,
            confidence=confidence,
            rationale=reason + (f" [HUMAN REVIEW REQUIRED: Missing {len(missing_info)} item(s)]" if human_review else ""),
            osha_citations=citations,
            precursor_severity_signals=risk_factors,
            escalation_potential=(risk_lvl == RiskLevel.HIGH or human_review),
        )

        return EnrichedReport(
            report=raw_rep,
            extraction=extraction_obj,
            assessment=assessment_obj,
            overrides=[],
            effective_risk_level=risk_lvl,
        )

    def _run_deterministic_agent(self, raw_text, department, location, facility, reporter_role, equipment, report_id, recent_overrides, trace) -> EnrichedReport:
        """Deterministic expert heuristic reasoning ensuring 100% functionality without API keys."""
        trace.steps_executed.append("1. Invoked 'detect_fatal_precursor_signals' tool to scan for SIF triggers.")
        t0 = time.time()
        precursor_result = self._tool_detect_precursors(raw_text)
        trace.tool_calls.append(ToolExecutionRecord(
            tool_name="detect_fatal_precursor_signals",
            input_args={"narrative_text": raw_text[:80] + "..."},
            output_result=precursor_result,
            execution_time_ms=round((time.time() - t0) * 1000, 2),
        ))

        trace.steps_executed.append("2. Invoked 'lookup_osha_standards' RAG tool for regulatory benchmarks.")
        t0 = time.time()
        rag_result = self._tool_lookup_osha(department, raw_text, precursor_result.get("precursor_tag", ""))
        trace.tool_calls.append(ToolExecutionRecord(
            tool_name="lookup_osha_standards",
            input_args={"query_terms": f"{department} {precursor_result.get('precursor_tag', '')}"},
            output_result=rag_result,
            execution_time_ms=round((time.time() - t0) * 1000, 2),
        ))
        citations = [item["standard"] for item in rag_result.get("matched_standards", [])]
        trace.retrieved_citations = citations

        trace.steps_executed.append("3. Invoked 'calculate_hazard_severity_index' tool for mathematical scoring.")
        is_sif = precursor_result["sif_precursor_detected"]
        energy = "High" if is_sif else ("Low" if "slip" in raw_text.lower() and "stair" not in raw_text.lower() and "chemical" not in raw_text.lower() else "Medium")
        barrier = "Intact" if energy == "Low" else ("Defeated" if any(w in raw_text.lower() for w in ["bypass", "missing", "expired", "failed", "broken", "unsecured", "taped"]) else "Degraded")
        exposure = "Intermittent" if energy == "Low" else "Continuous"

        t0 = time.time()
        severity_result = self._tool_calc_severity(energy, barrier, exposure, max(precursor_result["signal_count"], 1))
        trace.tool_calls.append(ToolExecutionRecord(
            tool_name="calculate_hazard_severity_index",
            input_args={"energy_level": energy, "barrier_redundancy": barrier, "exposure": exposure},
            output_result=severity_result,
            execution_time_ms=round((time.time() - t0) * 1000, 2),
        ))

        trace.steps_executed.append("4. Evaluated dynamic few-shot human overrides.")
        tier = severity_result["recommended_risk_tier"]
        score = severity_result["hazard_severity_score"]

        matched_override_reason = None
        for ov in recent_overrides:
            reason_words = [w.lower() for w in ov.override_reason.split() if len(w) > 4]
            if any(w in raw_text.lower() for w in reason_words):
                tier = ov.overridden_risk.value
                score = 8.8 if tier == "High" else (5.8 if tier == "Medium" else 3.2)
                matched_override_reason = ov.override_reason
                break

        risk_level = RiskLevel(tier)

        cat = HazardCategory.SLIP_TRIP_FALL
        text_l = raw_text.lower()
        if any(w in text_l for w in ["acid", "chemical", "fume", "toxic", "solvent"]): cat = HazardCategory.CHEMICAL_TOXIC
        elif any(w in text_l for w in ["forklift", "press", "ram", "pinch", "crush", "truck"]): cat = HazardCategory.MECHANICAL_CRUSH
        elif any(w in text_l for w in ["arc flash", "480v", "electric", "grounding", "voltage"]): cat = HazardCategory.ELECTRICAL
        elif any(w in text_l for w in ["catwalk", "mezzanine", "fall", "height", "dropped"]): cat = HazardCategory.FALLING_OBJECTS_HEIGHT
        elif any(w in text_l for w in ["fire", "ignite", "spark", "explosion", "toluene"]): cat = HazardCategory.THERMAL_FIRE
        elif any(w in text_l for w in ["confined space", "h2s", "oxygen", "tank"]): cat = HazardCategory.ATMOSPHERIC_CONFINED

        precursors = [s["precursor_domain"] for s in precursor_result.get("active_signals", [])]
        if not precursors: precursors = ["Unaddressed physical anomaly in operational work zone"]

        # Missing information heuristic
        missing_info = []
        if "ppe" not in text_l and "equipment" not in text_l: missing_info.append("PPE compliance status at time of observation")
        if "witness" not in text_l: missing_info.append("Secondary witness confirmation")
        if not missing_info: missing_info.append("Root cause preventive inspection date")

        consequences = ["Possible laceration or strain" if tier == "Low" else ("Serious injury or equipment damage" if tier == "Medium" else "Catastrophic Serious Injury or Fatality (SIF)")]

        trace.raw_extraction_data = {
            "risk_factors": precursors,
            "location": location,
            "department": department,
            "hazards": [f"{cat.value} - {precursor_result.get('precursor_tag', 'Operational Risk')}"],
            "potential_consequences": consequences,
            "missing_information": missing_info,
        }

        trace.raw_classification_data = {
            "risk_level": tier.upper(),
            "confidence": 0.94 if not matched_override_reason else 0.98,
            "reason": f"Analyzed under '{cat.value}'. Regulatory benchmark: {citations[0] if citations else 'OSHA 1910.22'}.",
            "human_review_required": len(missing_info) >= 2 or tier == "High",
        }

        rationale = (
            f"Precursor analysis flagged {len(precursors)} critical risk signal(s) under '{cat.value}'. "
            f"Evaluated against benchmark {citations[0] if citations else 'OSHA General Duty Clause'}. "
            f"Severity score: {score:.1f}/10 with barrier integrity assessed as {severity_result['barrier_status']}."
        )
        if matched_override_reason:
            rationale += f" [Dynamically Calibrated via Senior Officer Override: '{matched_override_reason}']"

        raw_rep = NearMissReport(
            id=report_id,
            facility=facility,
            department=department,
            location_specific=location,
            reporter_role=reporter_role,
            raw_text=raw_text,
            equipment_involved=equipment,
            environmental_factors="Assessed by Expert Agent",
            immediate_action_taken="Ingested by Incident Precursor Agent",
        )

        extraction_obj = StructuredExtraction(
            report_id=report_id,
            primary_hazard=f"{cat.value} Precursor - {precursor_result.get('precursor_tag', 'Operational Risk')}",
            hazard_category=cat,
            precursor_events=precursors,
            affected_assets=["Operational Personnel", equipment if equipment != "N/A" else "Workstation Area"],
            failed_safeguards=consequences,
            recommended_mitigation=f"Audit barrier reliability according to {citations[0] if citations else 'OSHA guidelines'}; verify: {missing_info[0]}.",
        )

        assessment_obj = RiskAssessment(
            report_id=report_id,
            risk_level=risk_level,
            risk_score=score,
            confidence=0.94,
            rationale=rationale,
            osha_citations=citations,
            precursor_severity_signals=precursors,
            escalation_potential=severity_result.get("escalation_potential", False),
        )

        return EnrichedReport(
            report=raw_rep,
            extraction=extraction_obj,
            assessment=assessment_obj,
            overrides=[],
            effective_risk_level=risk_level,
        )

    def _tool_detect_precursors(self, text: str) -> Dict[str, Any]:
        text_l = text.lower()
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
            matches = [term for term in regex_list if re.search(term, text_l)]
            if len(matches) >= 2 or (category == "LOTO & Energy Isolation Defeat" and len(matches) >= 1):
                signals.append({"precursor_domain": category, "triggers": matches})
        is_sif = len(signals) > 0
        return {
            "sif_precursor_detected": is_sif,
            "active_signals": signals,
            "signal_count": len(signals),
            "precursor_tag": signals[0]["precursor_domain"] if signals else "Standard Observation"
        }

    def _tool_lookup_osha(self, dept: str, text: str, precursor: str) -> Dict[str, Any]:
        standards = []
        combined = f"{dept} {text} {precursor}".lower()
        if "chemical" in combined or "acid" in combined:
            standards.append({"standard": "OSHA 1910.1200 / 1910.133", "title": "Hazard Communication & Eye/Face Protection", "benchmark_severity": "High"})
        if "forklift" in combined or "truck" in combined:
            standards.append({"standard": "OSHA 1910.178", "title": "Powered Industrial Trucks & Pedestrian Safety", "benchmark_severity": "High"})
        if "catwalk" in combined or "height" in combined or "drop" in combined:
            standards.append({"standard": "OSHA 1910.28", "title": "Fall Protection and Falling Object Protection", "benchmark_severity": "High / Medium"})
        if "interlock" in combined or "loto" in combined or "robot" in combined:
            standards.append({"standard": "OSHA 1910.147", "title": "Control of Hazardous Energy (Lockout/Tagout)", "benchmark_severity": "High"})
        if "confined" in combined or "h2s" in combined:
            standards.append({"standard": "OSHA 1910.146", "title": "Permit-Required Confined Spaces", "benchmark_severity": "High"})
        if not standards:
            standards.append({"standard": "OSHA 1910.22", "title": "General Walking-Working Surfaces", "benchmark_severity": "Low / Medium"})
        return {"matched_standards": standards}

    def _tool_calc_severity(self, energy: str, barrier: str, exposure: str, precursors: int) -> Dict[str, Any]:
        e_weights = {"High": 5.0, "Medium": 3.0, "Low": 1.0}
        b_weights = {"Defeated": 4.0, "Degraded": 2.0, "Intact": 0.5}
        exp_weights = {"Continuous": 1.0, "Intermittent": 0.5, "Rare": 0.2}

        raw = e_weights.get(energy, 3.0) + b_weights.get(barrier, 2.0) + exp_weights.get(exposure, 0.5) + min(precursors * 0.4, 1.2)
        score = min(max(round(raw, 1), 1.0), 10.0)

        if score >= 7.0 or (energy == "High" and barrier == "Defeated"):
            tier = "High"
            escalate = True
        elif score >= 4.0:
            tier = "Medium"
            escalate = False
        else:
            tier = "Low"
            escalate = False

        return {
            "hazard_severity_score": score,
            "recommended_risk_tier": tier,
            "escalation_potential": escalate,
            "energy_assessment": energy,
            "barrier_status": barrier,
        }
