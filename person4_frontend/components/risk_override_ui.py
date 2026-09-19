"""
Person 4: Risk Override & Dynamic Few-Shot Feedback UI (Compulsory Add-On)
Allows senior safety officers to adjust risk levels with mandatory rationale,
stores the correction, and displays the active dynamic few-shot exemplars injected into the LLM.
"""
import streamlit as st
import datetime
from person1_data_pipeline.schema import RiskLevel, RiskOverride
from person1_data_pipeline.storage import SafetyStorage
from person2_llm_agent.few_shot_manager import DynamicFewShotManager


def render_risk_override_ui(storage: SafetyStorage, few_shot_manager: DynamicFewShotManager):
    """Renders the human-in-the-loop override console."""
    st.markdown("<div class='section-header'>⚖️ Safety Officer Override Hub & Active Few-Shot Memory</div>", unsafe_allow_html=True)
    st.markdown(
        "**Compulsory Add-On Requirement:** Calibrate model risk classifications by logging official justifications. "
        "Every logged override is immediately stored in SQLite and injected as a dynamic few-shot exemplar into subsequent agent runs."
    )

    enriched_reports = storage.get_enriched_reports()
    if not enriched_reports:
        st.warning("No reports available in storage to override.")
        return

    # Report selection
    report_options = {
        f"{r.report.id} - {r.extraction.primary_hazard[:45]}... [{r.effective_risk_level.value} Risk]": r.report.id
        for r in enriched_reports
    }

    selected_label = st.selectbox("Select Report to Inspect & Calibrate:", list(report_options.keys()))
    selected_id = report_options[selected_label]
    selected_item = next((r for r in enriched_reports if r.report.id == selected_id), None)

    if not selected_item:
        return

    r = selected_item.report
    e = selected_item.extraction
    a = selected_item.assessment
    effective_risk = selected_item.effective_risk_level

    col_details, col_override = st.columns([6, 5])

    with col_details:
        st.markdown("<div class='section-header'>📋 Original Assessment Details</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size: 0.85rem; color: #94a3b8;">Facility: {r.facility} | Department: {r.department}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">Zone: <code>{r.location_specific}</code></div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">{e.primary_hazard}</div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 6px; font-size: 0.86rem; color: #cbd5e1; margin-bottom: 10px;">
                    <b>Narrative:</b> {r.raw_text}
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="font-size: 0.85rem; color: #94a3b8;">Original Model Risk:</span>
                    <span class="badge badge-{a.risk_level.value.lower()}">{a.risk_level.value}</span>
                    <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 12px;">Effective Risk:</span>
                    <span class="badge badge-{effective_risk.value.lower()}">{effective_risk.value}</span>
                </div>
                <div style="font-size: 0.83rem; color: #94a3b8;">
                    <b>Model Rationale:</b> {a.rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_override:
        st.markdown("<div class='section-header'>✏️ Log Officer Override</div>", unsafe_allow_html=True)
        with st.form("override_submission_form"):
            officer_id = st.text_input("Safety Officer ID / Badge", value="SO-Lead-EHS")
            new_risk = st.selectbox(
                "Calibrated Risk Level:",
                ["High", "Medium", "Low"],
                index=["High", "Medium", "Low"].index(effective_risk.value) if effective_risk.value in ["High", "Medium", "Low"] else 1
            )
            override_reason = st.text_area(
                "Mandatory Operational & Precursor Rationale:",
                placeholder="Explain the specific risk factor, energy state, or operational context justifying this change...",
                height=110,
            )
            submit_override = st.form_submit_button("💾 Commit Override & Update Few-Shot Memory", use_container_width=True)

        if submit_override:
            if len(override_reason.strip()) < 8:
                st.error("Please provide a detailed justification (minimum 8 characters).")
            else:
                override = RiskOverride(
                    report_id=selected_id,
                    original_risk=effective_risk,
                    overridden_risk=RiskLevel(new_risk),
                    safety_officer_id=officer_id,
                    override_reason=override_reason.strip(),
                    timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                )
                storage.log_override(override)
                st.success(
                    f"✅ Override recorded! Report {selected_id} updated to '{new_risk}'. "
                    f"Correction is now actively injected into future LLM prompt contexts."
                )
                st.rerun()

    # Dynamic Few-Shot Memory Live Inspector
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🧠 Active Dynamic Few-Shot Prompt Memory (Live View)</div>", unsafe_allow_html=True)
    st.markdown(
        "Below are the active historical overrides retrieved by `person2_llm_agent/few_shot_manager.py`. "
        "These exemplars are injected into the LLM system prompt to align automated scoring with safety leadership."
    )

    recent_overrides = few_shot_manager.get_latest_exemplars(max_examples=5)
    if not recent_overrides:
        st.info("No human overrides have been recorded yet.")
    else:
        for idx, ov in enumerate(recent_overrides, start=1):
            direction_color = "#ef4444" if ov.overridden_risk.value == "High" else ("#f59e0b" if ov.overridden_risk.value == "Medium" else "#10b981")
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid {direction_color}; padding: 14px 18px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f8fafc;">Exemplar #{idx} | Report {ov.report_id}</span>
                        <span>
                            <span class="badge badge-{ov.original_risk.value.lower()}">{ov.original_risk.value}</span>
                            <span style="color: #94a3b8; margin: 0 4px;">➔</span>
                            <span class="badge badge-{ov.overridden_risk.value.lower()}">{ov.overridden_risk.value}</span>
                        </span>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.88rem; color: #cbd5e1;">
                        <b>Officer Justification:</b> <i>"{ov.override_reason}"</i>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">
                        Logged by: <code>{ov.safety_officer_id}</code> | Time: {ov.timestamp}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
