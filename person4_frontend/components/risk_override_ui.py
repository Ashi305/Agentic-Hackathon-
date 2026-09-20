"""
Person 4: Risk Override & Dynamic Few-Shot Feedback Hub (Compulsory Add-On)
Provides a high-precision review cockpit for Safety Officers to calibrate risk scores,
log operational justifications, and inspect live dynamic few-shot prompt memory in real time.
"""
import streamlit as st
import datetime
from person1_data_pipeline.schema import RiskLevel, RiskOverride
from person1_data_pipeline.storage import SafetyStorage
from person2_llm_agent.few_shot_manager import DynamicFewShotManager


def render_risk_override_ui(storage: SafetyStorage, few_shot_manager: DynamicFewShotManager):
    """Renders the human-in-the-loop override console and dynamic prompt inspector."""
    st.markdown(
        """
        <div class="view-title">
            Safety Officer Override Hub & Active Few-Shot Memory
        </div>
        <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 16px;">
            <b>Compulsory Add-On Requirement:</b> When an EHS safety officer overrides an automated risk score,
            the correction and its operational rationale are stored in SQLite and dynamically injected as few-shot
            exemplars into subsequent agent reasoning cycles.
        </p>
        """,
        unsafe_allow_html=True,
    )

    enriched_reports = storage.get_enriched_reports()
    if not enriched_reports:
        st.warning("No reports found in warehouse to calibrate.")
        return

    # Filter reports by risk tier
    col_f1, col_f2 = st.columns([4, 8])
    with col_f1:
        filter_tier = st.selectbox(
            "Filter Reports to Review:",
            ["All Reports", "Low Risk Only", "Medium Risk Only", "High Risk Only", "Already Overridden"]
        )

    display_reports = enriched_reports
    if filter_tier == "Low Risk Only":
        display_reports = [r for r in enriched_reports if r.effective_risk_level.value == "Low"]
    elif filter_tier == "Medium Risk Only":
        display_reports = [r for r in enriched_reports if r.effective_risk_level.value == "Medium"]
    elif filter_tier == "High Risk Only":
        display_reports = [r for r in enriched_reports if r.effective_risk_level.value == "High"]
    elif filter_tier == "Already Overridden":
        display_reports = [r for r in enriched_reports if len(r.overrides) > 0]

    if not display_reports:
        st.info("No reports match the current filter criteria.")
        return

    with col_f2:
        report_options = {
            f"[{r.report.id}] {r.extraction.primary_hazard[:55]}... ({r.effective_risk_level.value} Risk)": r.report.id
            for r in display_reports
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
    has_prior_override = len(selected_item.overrides) > 0

    # Side-by-Side Review Console
    col_left, col_right = st.columns([6, 6])

    with col_left:
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
                Current Automated Assessment
            </div>
            """,
            unsafe_allow_html=True,
        )
        orig_class = a.risk_level.value.lower()
        eff_class = effective_risk.value.lower()

        st.markdown(
            f"""
            <div class="glass-panel" style="min-height: 380px;">
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px;">
                    Facility: <b>{r.facility}</b> | Department: <b>{r.department}</b>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;">
                    Zone: <code>{r.location_specific}</code> | Asset: <code>{r.equipment_involved}</code>
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin-bottom: 8px;">
                    {e.primary_hazard}
                </div>
                
                <div style="background: #0b1120; padding: 12px; border-radius: 8px; font-size: 0.86rem; color: #f1f5f9; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08); max-height: 110px; overflow-y: auto;">
                    <b>Narrative:</b> <i>"{r.raw_text}"</i>
                </div>
                
                <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="font-size: 0.80rem; color: #94a3b8;">Original Model:</span>
                        <span class="badge-pill {orig_class}">{a.risk_level.value}</span>
                    </div>
                    <div>
                        <span style="font-size: 0.80rem; color: #94a3b8;">Effective:</span>
                        <span class="badge-pill {eff_class}">{effective_risk.value}</span>
                    </div>
                    {f'<span class="badge-pill override">Previously Overridden</span>' if has_prior_override else ''}
                </div>
                
                <div style="font-size: 0.83rem; color: #cbd5e1; line-height: 1.5;">
                    <b>Model Rationale:</b> {a.rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
                Safety Officer Calibration Console
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("officer_override_form"):
            c_off1, c_off2 = st.columns([5, 5])
            with c_off1:
                officer_id = st.text_input("Officer Badge ID", value="SO-Lead-EHS")
            with c_off2:
                risk_tiers = ["High", "Medium", "Low"]
                curr_idx = risk_tiers.index(effective_risk.value) if effective_risk.value in risk_tiers else 1
                new_risk_tier = st.selectbox("Calibrated Risk Tier:", risk_tiers, index=curr_idx)

            st.markdown(
                f"""
                <div style="background: #0b1120; border: 1px solid rgba(255,255,255,0.1); padding: 8px 12px; border-radius: 8px; margin: 8px 0; display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.82rem; color: #94a3b8;">Calibration Delta:</span>
                    <span>
                        <span class="badge-pill {effective_risk.value.lower()}">{effective_risk.value}</span>
                        <span style="margin: 0 8px; color: #94a3b8;">➔</span>
                        <span class="badge-pill {new_risk_tier.lower()}">{new_risk_tier}</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rationale_text = st.text_area(
                "Mandatory Operational & Precursor Rationale:",
                placeholder="Detail the specific energy state, unbarricaded path, missing guard, or operational factor that necessitates this correction...",
                height=100,
            )

            char_count = len(rationale_text.strip())
            st.caption(f"Justification Length: {char_count} characters (Minimum 8 required)")

            commit_btn = st.form_submit_button("Commit Override & Update Few-Shot Memory", use_container_width=True)

        if commit_btn:
            if char_count < 8:
                st.error("Please provide an operational justification (at least 8 characters).")
            else:
                override_rec = RiskOverride(
                    report_id=selected_id,
                    original_risk=effective_risk,
                    overridden_risk=RiskLevel(new_risk_tier),
                    safety_officer_id=officer_id.strip() or "SO-Field-EHS",
                    override_reason=rationale_text.strip(),
                    timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                )
                storage.log_override(override_rec)
                st.success(f"Override committed for {selected_id}. Dynamic few-shot prompt memory updated.")
                st.rerun()

    # -------------------------------------------------------------------------
    # Active Dynamic Few-Shot Prompt Memory Inspector
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="view-title">
            Active Dynamic Few-Shot Prompt Memory (Live View)
        </div>
        <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 12px;">
            The agent continuously queries the latest human corrections from SQLite and constructs the following
            exemplars into its prompt context to dynamically self-align with senior safety officer decisions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    recent_overrides = few_shot_manager.get_latest_exemplars(max_examples=5)

    if not recent_overrides:
        st.info("No human overrides have been recorded yet. The system is operating on baseline heuristics.")
    else:
        for idx, ov in enumerate(recent_overrides, start=1):
            direction_color = "#f43f5e" if ov.overridden_risk.value == "High" else ("#fbbf24" if ov.overridden_risk.value == "Medium" else "#10b981")
            st.markdown(
                f"""
                <div class="glass-panel" style="border-left: 4px solid {direction_color}; padding: 16px 20px; margin-bottom: 12px; background: #0d1527;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 800; color: #ffffff; font-size: 0.95rem;">
                            Pedagogical Exemplar #{idx} — Report Reference: <code>{ov.report_id}</code>
                        </span>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span class="badge-pill {ov.original_risk.value.lower()}">{ov.original_risk.value}</span>
                            <span style="color: #94a3b8;">➔</span>
                            <span class="badge-pill {ov.overridden_risk.value.lower()}">{ov.overridden_risk.value}</span>
                        </div>
                    </div>
                    
                    <div style="font-size: 0.90rem; color: #f1f5f9; margin: 8px 0;">
                        <b>Officer Guiding Rationale:</b> <i>"{ov.override_reason}"</i>
                    </div>
                    
                    <div style="font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between;">
                        <span>Verified by: <code>{ov.safety_officer_id}</code></span>
                        <span>Logged: {ov.timestamp} UTC</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View Raw Formatted Dynamic Few-Shot Prompt Segment"):
            formatted_prompt_text = few_shot_manager.format_exemplars_for_prompt(recent_overrides)
            st.code(formatted_prompt_text, language="markdown")
