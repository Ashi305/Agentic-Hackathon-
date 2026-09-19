"""
Person 4: Report Ingestion & Real-Time Agent Analysis Component
Supports free-text submission, walkie-talkie audio transcript simulation,
and preset industrial scenario loading.
"""
import streamlit as st
import time
from person2_llm_agent.agent_orchestrator import IncidentPrecursorAgent

PRESET_SCENARIOS = {
    "Acid Line Gasket Failure (High Risk Precursor)": {
        "dept": "Chemical Processing & Synthesis",
        "loc": "Reactor Bay 3 - Acid Dosing Skid",
        "equip": "P-104 Sulfuric Acid Metering Pump",
        "text": "During morning line flushing on Reactor 3, observed intermittent acid spitting from the PTFE flange gasket on metering pump P-104. Flange spray shield was improperly seated and secured with only a single zip-tie instead of required locking collar. Operator was wearing safety glasses instead of a full face shield. Acid droplets splashed onto sleeve of chemical apron but did not penetrate skin. If pressure had surged to normal 45 PSI dosing pressure, direct facial contact with 98% H2SO4 would have occurred."
    },
    "Forklift Blind Corner Near Miss (Struck-By Precursor)": {
        "dept": "High-Bay Warehousing & Logistics",
        "loc": "Aisle 14 - Cross-dock Intersection",
        "equip": "Toyota 3-Wheel Electric Forklift #7",
        "text": "Order picker walking out of Aisle 14 with hand truck was almost struck by Forklift #7 rounding the blind corner at speed. The ceiling-mounted parabolic convex mirror was twisted upwards and covered in dust, obscuring sightlines. The forklift driver's blue safety halo light was functioning, but the automatic horn interlock failed to sound upon entering the pedestrian intersection. Braking distance missed the pedestrian by less than 18 inches."
    },
    "Catwalk Dropped Heavy Wrench (Height Precursor)": {
        "dept": "Plant Facilities & Maintenance",
        "loc": "Boiler Room Mezzanine Level 2",
        "equip": "Chilled Water Expansion Tank TK-08",
        "text": "Mechanic working on 12-foot catwalk replacing 2-inch flange nuts dropped a 15-inch adjustable spanner wrench. The wrench slipped through the 1.5-inch toe-board gap on the open-grate mezzanine and plummeted to the ground-level walkway adjacent to the water treatment laboratory. No workers were directly below at the moment, but the walkway is a designated primary egress route without active barricades or drop netting."
    },
    "Robot Cell Interlock Magnetic Bypass (Fatal Precursor)": {
        "dept": "Heavy Fabrication & Stamping",
        "loc": "Robotic Welding Cell 4",
        "equip": "Fanuc 6-Axis Arc Mate Robot",
        "text": "Maintenance technician entered robot enclosure to clear weld wire bird-nest while interlock access gate was keyed open with an override magnetic bypass wedge. Robot was paused in hold mode rather than zero-energy state (LOTO). While mechanic was untangling wire nozzle, teach pendant was nudged on console table, causing robot arm to jog 15cm toward mechanic's shoulder before halting on torque limit."
    },
    "Cleanroom Minor Solvent Drip (Low Risk)": {
        "dept": "Semiconductor Cleanroom Assembly",
        "loc": "Photolithography Bay 6",
        "equip": "Stepper Track Coat Unit 2",
        "text": "Small puddle of isopropyl alcohol (IPA) approximately 100ml found pooled on conductive vinyl floor near solvent drain line. Operator's ESD shoe slipped slightly during wafer cassette transfer, but balance was maintained with handrail. Drain tubing connection had loosened due to thermal cycling from exhaust duct."
    }
}


def render_report_ingestion(agent: IncidentPrecursorAgent):
    """Renders the interactive report ingestion workbench."""
    st.markdown("<div class='section-header'>📝 Ingest Unstructured Safety Observation</div>", unsafe_allow_html=True)
    st.markdown("Submit free-text field reports or simulate real-time radio/voice transcripts for automated precursor analysis.")

    # Preset Quick-Loader
    col_preset, col_voice = st.columns([7, 3])
    with col_preset:
        preset_choice = st.selectbox(
            "⚡ Quick-Fill Realistic Operational Scenario:",
            ["Custom Report (Type Below)"] + list(PRESET_SCENARIOS.keys()),
        )
    with col_voice:
        simulate_voice = st.checkbox("🎙️ Simulate Voice/Walkie-Talkie Mode", help="Formats raw text as live speech-to-text transcript")

    # Determine default values based on preset
    default_text = ""
    default_dept = "Chemical Processing & Synthesis"
    default_loc = "Main Production Deck"
    default_equip = "N/A"

    if preset_choice != "Custom Report (Type Below)":
        p_data = PRESET_SCENARIOS[preset_choice]
        default_text = p_data["text"]
        default_dept = p_data["dept"]
        default_loc = p_data["loc"]
        default_equip = p_data["equip"]

    with st.form("safety_ingestion_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            department = st.text_input("Department / Operating Sector", value=default_dept)
        with c2:
            location = st.text_input("Specific Zone / Deck / Bay", value=default_loc)
        with c3:
            equipment = st.text_input("Machinery / Asset Involved", value=default_equip)

        report_text = st.text_area(
            "Free-Text Near-Miss / Observation Narrative:",
            value=default_text,
            height=160,
            placeholder="Describe what occurred, unsafe conditions observed, immediate actions taken, and potential hazards...",
        )

        submit_btn = st.form_submit_button("🚀 Run Precursor Analysis & Reason", use_container_width=True)

    if submit_btn:
        if not report_text.strip():
            st.error("Please enter observation text before triggering analysis.")
            return

        final_text = report_text
        if simulate_voice:
            final_text = f"[VOICE RADIO TRANSCRIPT - CHANNEL 4 LOG]: {report_text} [OVER]"

        with st.spinner("Agent invoking perception tools, retrieving OSHA guidelines, and calculating severity..."):
            trace = agent.analyze_report(
                raw_text=final_text,
                department=department,
                location=location,
                equipment=equipment,
            )
            st.session_state["latest_trace"] = trace
            st.success(f"Analysis completed for report {trace.report_id}!")

    # Display results if present
    if "latest_trace" in st.session_state and st.session_state["latest_trace"].final_enriched_report:
        trace = st.session_state["latest_trace"]
        rep = trace.final_enriched_report
        ext = rep.extraction
        ass = rep.assessment

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎯 Agent Classification & Risk Extraction</div>", unsafe_allow_html=True)

        badge_class = f"badge-{ass.risk_level.value.lower()}"
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">{ext.primary_hazard}</span>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Report ID: <code>{rep.report.id}</code> | Category: <b>{ext.hazard_category.value}</b></div>
                    </div>
                    <div>
                        <span class="badge {badge_class}">{ass.risk_level.value} Risk</span>
                        <span class="badge" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); margin-left: 6px;">Score: {ass.risk_score}/10</span>
                    </div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 8px; margin-bottom: 14px; border-left: 3px solid #38bdf8;">
                    <b>Agent Technical Rationale:</b> {ass.rationale}
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-top: 10px;">
                    <div>
                        <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">⚠️ Precursor Signals Detected</div>
                        <ul style="margin-top: 6px; padding-left: 18px; font-size: 0.88rem; color: #f1f5f9;">
                            {''.join([f'<li>{p}</li>' for p in ext.precursor_events])}
                        </ul>
                    </div>
                    <div>
                        <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">🛡️ Failed Safeguards</div>
                        <ul style="margin-top: 6px; padding-left: 18px; font-size: 0.88rem; color: #f1f5f9;">
                            {''.join([f'<li>{s}</li>' for s in ext.failed_safeguards])}
                        </ul>
                    </div>
                    <div>
                        <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">📜 OSHA Grounded Citations</div>
                        <ul style="margin-top: 6px; padding-left: 18px; font-size: 0.88rem; color: #38bdf8;">
                            {''.join([f'<li><code>{c}</code></li>' for c in ass.osha_citations])}
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.88rem; color: #10b981; background: rgba(16, 185, 129, 0.1); padding: 10px; border-radius: 6px;">
                    <b>Recommended Action:</b> {ext.recommended_mitigation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
