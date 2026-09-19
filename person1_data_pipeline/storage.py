"""
Person 1: Simple JSON & SQLite Storage Layer
Provides persistent dual-mode storage for raw near-miss reports, structured extractions,
risk assessments, and logged safety officer overrides (used for dynamic few-shot learning).
"""
import os
import json
import sqlite3
from typing import List, Optional, Dict, Any
import pandas as pd

from .schema import (
    NearMissReport,
    StructuredExtraction,
    RiskAssessment,
    RiskOverride,
    EnrichedReport,
    RiskLevel,
    HazardCategory,
)


class SafetyStorage:
    def __init__(self, db_path: Optional[str] = None, json_backup_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self.db_path = db_path or os.path.join(data_dir, "safety_warehouse.db")
        self.json_backup_path = json_backup_path or os.path.join(data_dir, "near_miss_reports.json")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes SQLite schema for reports, extractions, assessments, and overrides."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    facility TEXT,
                    department TEXT,
                    location_specific TEXT,
                    reporter_role TEXT,
                    raw_text TEXT,
                    equipment_involved TEXT,
                    environmental_factors TEXT,
                    immediate_action_taken TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    report_id TEXT PRIMARY KEY,
                    primary_hazard TEXT,
                    hazard_category TEXT,
                    precursor_events TEXT,
                    affected_assets TEXT,
                    failed_safeguards TEXT,
                    recommended_mitigation TEXT,
                    FOREIGN KEY (report_id) REFERENCES reports(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    report_id TEXT PRIMARY KEY,
                    risk_level TEXT,
                    risk_score REAL,
                    confidence REAL,
                    rationale TEXT,
                    osha_citations TEXT,
                    precursor_severity_signals TEXT,
                    escalation_potential INTEGER,
                    FOREIGN KEY (report_id) REFERENCES reports(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT,
                    original_risk TEXT,
                    overridden_risk TEXT,
                    safety_officer_id TEXT,
                    override_reason TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (report_id) REFERENCES reports(id)
                )
            """)
            conn.commit()

    def save_enriched_report(self, enriched: EnrichedReport) -> None:
        """Saves or updates a full enriched report in SQLite."""
        rep = enriched.report
        ext = enriched.extraction
        ass = enriched.assessment

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO reports (
                    id, timestamp, facility, department, location_specific,
                    reporter_role, raw_text, equipment_involved, environmental_factors, immediate_action_taken
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rep.id, rep.timestamp, rep.facility, rep.department, rep.location_specific,
                rep.reporter_role, rep.raw_text, rep.equipment_involved,
                rep.environmental_factors, rep.immediate_action_taken
            ))

            cursor.execute("""
                INSERT OR REPLACE INTO extractions (
                    report_id, primary_hazard, hazard_category, precursor_events,
                    affected_assets, failed_safeguards, recommended_mitigation
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ext.report_id, ext.primary_hazard, ext.hazard_category.value if hasattr(ext.hazard_category, 'value') else str(ext.hazard_category),
                json.dumps(ext.precursor_events), json.dumps(ext.affected_assets),
                json.dumps(ext.failed_safeguards), ext.recommended_mitigation
            ))

            cursor.execute("""
                INSERT OR REPLACE INTO assessments (
                    report_id, risk_level, risk_score, confidence, rationale,
                    osha_citations, precursor_severity_signals, escalation_potential
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ass.report_id, ass.risk_level.value if hasattr(ass.risk_level, 'value') else str(ass.risk_level),
                ass.risk_score, ass.confidence, ass.rationale,
                json.dumps(ass.osha_citations), json.dumps(ass.precursor_severity_signals),
                1 if ass.escalation_potential else 0
            ))
            conn.commit()

    def log_override(self, override: RiskOverride) -> None:
        """
        Compulsory Add-On:
        Logs an officer's risk override with mandatory reason.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO overrides (
                    report_id, original_risk, overridden_risk, safety_officer_id,
                    override_reason, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                override.report_id,
                override.original_risk.value if hasattr(override.original_risk, 'value') else str(override.original_risk),
                override.overridden_risk.value if hasattr(override.overridden_risk, 'value') else str(override.overridden_risk),
                override.safety_officer_id,
                override.override_reason,
                override.timestamp
            ))
            conn.commit()

    def get_recent_overrides(self, limit: int = 5) -> List[RiskOverride]:
        """Fetches the latest human corrections for dynamic few-shot prompt injection."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, report_id, original_risk, overridden_risk, safety_officer_id, override_reason, timestamp
                FROM overrides
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [
                RiskOverride(
                    id=str(row["id"]),
                    report_id=row["report_id"],
                    original_risk=RiskLevel(row["original_risk"]),
                    overridden_risk=RiskLevel(row["overridden_risk"]),
                    safety_officer_id=row["safety_officer_id"],
                    override_reason=row["override_reason"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    def get_all_overrides(self) -> List[RiskOverride]:
        """Fetches all logged overrides for audit and metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, report_id, original_risk, overridden_risk, safety_officer_id, override_reason, timestamp
                FROM overrides
                ORDER BY id DESC
            """)
            rows = cursor.fetchall()
            return [
                RiskOverride(
                    id=str(row["id"]),
                    report_id=row["report_id"],
                    original_risk=RiskLevel(row["original_risk"]),
                    overridden_risk=RiskLevel(row["overridden_risk"]),
                    safety_officer_id=row["safety_officer_id"],
                    override_reason=row["override_reason"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    def get_enriched_reports(self) -> List[EnrichedReport]:
        """Retrieves all enriched reports combining SQLite records with latest overrides."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    r.id, r.timestamp, r.facility, r.department, r.location_specific,
                    r.reporter_role, r.raw_text, r.equipment_involved, r.environmental_factors, r.immediate_action_taken,
                    e.primary_hazard, e.hazard_category, e.precursor_events, e.affected_assets, e.failed_safeguards, e.recommended_mitigation,
                    a.risk_level, a.risk_score, a.confidence, a.rationale, a.osha_citations, a.precursor_severity_signals, a.escalation_potential
                FROM reports r
                LEFT JOIN extractions e ON r.id = e.report_id
                LEFT JOIN assessments a ON r.id = a.report_id
                ORDER BY r.timestamp DESC
            """)
            rows = cursor.fetchall()

        overrides_by_report: Dict[str, List[RiskOverride]] = {}
        for ov in self.get_all_overrides():
            overrides_by_report.setdefault(ov.report_id, []).append(ov)

        results: List[EnrichedReport] = []
        for row in rows:
            rep = NearMissReport(
                id=row["id"],
                timestamp=row["timestamp"],
                facility=row["facility"],
                department=row["department"],
                location_specific=row["location_specific"],
                reporter_role=row["reporter_role"] or "Staff",
                raw_text=row["raw_text"],
                equipment_involved=row["equipment_involved"] or "N/A",
                environmental_factors=row["environmental_factors"] or "Normal",
                immediate_action_taken=row["immediate_action_taken"] or "None logged",
            )
            
            hazard_cat = HazardCategory.SLIP_TRIP_FALL
            if row["hazard_category"]:
                try:
                    hazard_cat = HazardCategory(row["hazard_category"])
                except Exception:
                    hazard_cat = HazardCategory.SLIP_TRIP_FALL

            ext = StructuredExtraction(
                report_id=row["id"],
                primary_hazard=row["primary_hazard"] or "Unspecified",
                hazard_category=hazard_cat,
                precursor_events=json.loads(row["precursor_events"] or "[]"),
                affected_assets=json.loads(row["affected_assets"] or "[]"),
                failed_safeguards=json.loads(row["failed_safeguards"] or "[]"),
                recommended_mitigation=row["recommended_mitigation"] or "Inspect area.",
            )

            risk_lvl = RiskLevel.MEDIUM
            if row["risk_level"]:
                try:
                    risk_lvl = RiskLevel(row["risk_level"])
                except Exception:
                    risk_lvl = RiskLevel.MEDIUM

            ass = RiskAssessment(
                report_id=row["id"],
                risk_level=risk_lvl,
                risk_score=row["risk_score"] if row["risk_score"] is not None else 5.0,
                confidence=row["confidence"] if row["confidence"] is not None else 0.90,
                rationale=row["rationale"] or "Heuristic evaluation.",
                osha_citations=json.loads(row["osha_citations"] or "[]"),
                precursor_severity_signals=json.loads(row["precursor_severity_signals"] or "[]"),
                escalation_potential=bool(row["escalation_potential"]),
            )

            rep_overrides = overrides_by_report.get(row["id"], [])
            effective_risk = rep_overrides[0].overridden_risk if rep_overrides else ass.risk_level

            results.append(EnrichedReport(
                report=rep,
                extraction=ext,
                assessment=ass,
                overrides=rep_overrides,
                effective_risk_level=effective_risk,
            ))
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Flattens enriched reports into a Pandas DataFrame for Person 3 analytics."""
        records = []
        for item in self.get_enriched_reports():
            r = item.report
            e = item.extraction
            a = item.assessment
            has_override = len(item.overrides) > 0
            latest_reason = item.overrides[0].override_reason if has_override else ""

            records.append({
                "report_id": r.id,
                "timestamp": r.timestamp,
                "facility": r.facility,
                "department": r.department,
                "location_specific": r.location_specific,
                "reporter_role": r.reporter_role,
                "raw_text": r.raw_text,
                "equipment_involved": r.equipment_involved,
                "primary_hazard": e.primary_hazard,
                "hazard_category": e.hazard_category.value,
                "precursor_events": e.precursor_events,
                "precursor_count": len(e.precursor_events),
                "failed_safeguards": e.failed_safeguards,
                "original_risk_level": a.risk_level.value,
                "effective_risk_level": item.effective_risk_level.value,
                "risk_score": a.risk_score,
                "confidence": a.confidence,
                "escalation_potential": a.escalation_potential,
                "osha_citations": a.osha_citations,
                "precursor_severity_signals": a.precursor_severity_signals,
                "has_override": has_override,
                "override_reason": latest_reason,
            })
        return pd.DataFrame(records)

    def seed_initial_data(self, target_count: int = 42, force_reload: bool = False) -> int:
        """Seeds the SQLite warehouse and JSON file if database is empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reports")
            count = cursor.fetchone()[0]

        if count > 0 and not force_reload:
            return count

        from .synthetic_generator import generate_synthetic_reports
        reports = generate_synthetic_reports(target_count=target_count)

        # Pre-seed 3 realistic historical overrides to immediately demonstrate dynamic few-shot learning
        if len(reports) >= 4:
            reports[3].overrides.append(RiskOverride(
                report_id=reports[3].report.id,
                original_risk=RiskLevel.MEDIUM,
                overridden_risk=RiskLevel.HIGH,
                safety_officer_id="SO-Lead-EHS",
                override_reason="Unbarricaded active pedestrian walkway directly below overhead catwalk makes dropped spanner an imminent fatal crush precursor.",
                timestamp="2024-03-12 14:30:00"
            ))
            reports[3].effective_risk_level = RiskLevel.HIGH

        for rep in reports:
            self.save_enriched_report(rep)
            for ov in rep.overrides:
                self.log_override(ov)

        # Also write JSON export
        self.export_to_json(self.json_backup_path)
        return len(reports)

    def export_to_json(self, output_path: str) -> None:
        """Exports all enriched reports to formatted JSON."""
        enriched_list = self.get_enriched_reports()
        data = [json.loads(item.model_dump_json()) for item in enriched_list]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
