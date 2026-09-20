"""
Person 4: Multi-Modal Ingestion Workbench
Supports Free-Text Observation Forms, Field Audio / Walkie-Talkie Simulation,
Batch JSON Ingestion, and Direct Ingestion from Person 2's report.csv Dataset.
"""
import streamlit as st
import json
import time
import os
import pandas as pd
from typing import List, Dict, Any
from person1_data_pipeline.schema import NearMissReport
from person4_frontend.agent_service import FrontendAgentService
from person4_frontend.components.ui_utils import render_html

PRESET_SCENARIOS = {
    "Acid Flange Spray Shield Defect (High Risk Precursor)": {
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
    "Robot Welding Cell Interlock Bypass (Fatal Precursor)": {
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

RADIO_VOICE_MEMOS = {
    "Logistics Channel 4 (Loading Dock Separation)": {
        "speaker": "Hostler Unit 12 (Marcus Vance)",
        "channel": "Channel 4 - Yard & Logistics",
        "raw_audio_transcript": "Dispatch, this is Marcus on Yard Jockey 12 at Dock Door 8. We got a severe near miss here. Backed up into trailer 402, green light was showing on the panel, but the automated lock jaw was jammed open with mud. The trailer crept out almost six inches while the forklift had its drive tires on the plate. Forklift operator slammed the emergency brake just in time. Dock lock sensor is throwing false positive greens. Requesting immediate maintenance lockout before somebody tips off the ledge. Over."
    },
    "Chemical Ops Channel 2 (Scrubber Exhaust Alarm)": {
        "speaker": "Operator Senior Lead (Elena Rostova)",
        "channel": "Channel 2 - Reaction & Synthesis",
        "raw_audio_transcript": "Control room, Elena calling from Scrubber Tower B. We have a low airflow warning on the caustic circulation line. Someone bypassed the alarm siren with a jumper on the terminal block, so nobody heard the alert. Caustic flow dropped to near zero for at least 15 minutes during the chlorine purge cycle. Faint chlorine smell near the vent pipe. We are initiating manual shutdown now. Over."
    },
    "Electrical Channel 6 (Substation Breaker Hot Spot)": {
        "speaker": "Senior Electrician (David Zhao)",
        "channel": "Channel 6 - High Voltage Infrastructure",
        "raw_audio_transcript": "EHS desk, David Zhao in Substation Alpha. During routine thermal imaging, phase B connection on the 480-volt feeder breaker registered 185 degrees Celsius under 60 percent load. Bolt torque was loose and the fiber insulating barrier has heat discoloration. If we had surged to peak load during shift change, it would have caused an arc flash blowout in the main MCC room. Requesting immediate line clearance. Over."
    }
}


def _get_csv_samples() -> Dict[str, Dict[str, str]]:
    """Loads sample reports from Person 2's report.csv if present."""
    samples = {}
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "person2_llm_agent", "report.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, nrows=10)
            df = df.dropna(subset=["Report_Text"])
            for idx, row in df.head(3).iterrows():
                rep_id = row.get("Report_ID", f"CSV-{idx+1}")
                loc = str(row.get("Location", "Plant Floor"))
                dept = str(row.get("Department", "Operations"))
                severity = str(row.get("Ground_Truth_Severity", "Unknown"))
                label = f"Dataset Report {rep_id} [{loc}] - Ground Truth: {severity}"
                samples[label] = {
                    "dept": dept if dept != "nan" else "Operations",
                    "loc": loc if loc != "nan" else "Facility Zone",
                    "equip": "Equipment in zone",
                    "text": str(row["Report_Text"]),
                }
        except Exception:
            pass
    return samples


def render_report_ingestion(agent: FrontendAgentService):
    """Renders the multi-modal report ingestion hub."""
    render_html("""
    <div class="view-title">
        Multi-Modal Safety Observation Ingestion
    </div>
    """)

    ingestion_mode = st.radio(
        "Select Ingestion Modality:",
        [
            "Structured Free-Text Narrative",
            "Radio / Voice Transcript Simulator",
            "PDF Incident Report Ingestion (Person 1 Parser)",
            "Batch JSON File Ingestion",
        ],
        horizontal=True,
    )

    # -------------------------------------------------------------------------
    # MODALITY 1: Structured Free-Text Narrative
    # -------------------------------------------------------------------------
    if ingestion_mode == "Structured Free-Text Narrative":
        render_html("""
        <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 12px;">
            Input free-text incident observations directly or select pre-calibrated operational test scenarios from Person 2's dataset.
        </div>
        """)

        csv_samples = _get_csv_samples()
        all_options = ["Custom Narrative (Type Manually)"] + list(PRESET_SCENARIOS.keys()) + list(csv_samples.keys())

        preset_choice = st.selectbox(
            "Quick-Fill Operational Test Scenarios (Synthetic & Real Benchmark):",
            all_options,
        )

        default_text = ""
        default_dept = "Chemical Processing & Synthesis"
        default_loc = "Reactor Bay 3 - Acid Dosing Skid"
        default_equip = "P-104 Sulfuric Acid Metering Pump"

        if preset_choice in PRESET_SCENARIOS:
            p = PRESET_SCENARIOS[preset_choice]
            default_text = p["text"]
            default_dept = p["dept"]
            default_loc = p["loc"]
            default_equip = p["equip"]
        elif preset_choice in csv_samples:
            p = csv_samples[preset_choice]
            default_text = p["text"]
            default_dept = p["dept"]
            default_loc = p["loc"]
            default_equip = p["equip"]

        with st.form("free_text_ingestion_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                dept_val = st.text_input("Operating Department", value=default_dept)
            with c2:
                loc_val = st.text_input("Specific Facility Zone / Deck", value=default_loc)
            with c3:
                equip_val = st.text_input("Machinery / Equipment Involved", value=default_equip)

            text_val = st.text_area(
                "Unstructured Observation / Near-Miss Narrative:",
                value=default_text,
                height=150,
                placeholder="Describe unsafe conditions, precursor events, actions taken, and potential escalation risks...",
            )

            submit_btn = st.form_submit_button("Run Agent Reasoner & Ingest", use_container_width=True)

        if submit_btn:
            if not text_val.strip():
                st.error("Narrative text cannot be empty.")
            else:
                _execute_agent_analysis(agent, text_val, dept_val, loc_val, equip_val)

    # -------------------------------------------------------------------------
    # MODALITY 2: Field Audio / Walkie-Talkie Simulation
    # -------------------------------------------------------------------------
    elif ingestion_mode == "Radio / Voice Transcript Simulator":
        render_html("""
        <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 12px;">
            Simulates real-world two-way radio communications and speech-to-text audio perception feeds.
        </div>
        """)

        radio_choice = st.selectbox(
            "Select Field Radio Channel & Feed:",
            list(RADIO_VOICE_MEMOS.keys())
        )
        memo = RADIO_VOICE_MEMOS[radio_choice]

        render_html(f"""
        <div class="audio-deck">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.82rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; font-family: 'JetBrains Mono';">
                    [LIVE RECEPTION: {memo['channel']}]
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px; height: 26px;">
                <span class="waveform-bar" style="animation-delay: 0.1s; height: 18px;"></span>
                <span class="waveform-bar" style="animation-delay: 0.4s; height: 26px;"></span>
                <span class="waveform-bar" style="animation-delay: 0.2s; height: 12px;"></span>
                <span class="waveform-bar" style="animation-delay: 0.5s; height: 22px;"></span>
                <span class="waveform-bar" style="animation-delay: 0.3s; height: 15px;"></span>
                <span class="waveform-bar" style="animation-delay: 0.6s; height: 26px;"></span>
            </div>
            <div style="margin-left: auto; font-size: 0.82rem; color: #cbd5e1;">
                Speaker: <b>{memo['speaker']}</b> | Codec: <code>Opus/16kHz</code>
            </div>
        </div>
        """)

        st.markdown("**Perceived Audio Speech-to-Text Transcript:**")
        st.info(f"\"{memo['raw_audio_transcript']}\"")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            voice_dept = st.text_input("Transcribed Department", value=radio_choice.split(" (")[0].replace(" Channel", ""))
        with col_v2:
            voice_loc = st.text_input("Operational Area", value="Radio Sector Broadcast Zone")

        if st.button("Process Radio Transmission Through Agent", use_container_width=True):
            _execute_agent_analysis(
                agent,
                f"[RADIO DISPATCH TRANSCRIPT]: {memo['raw_audio_transcript']}",
                voice_dept,
                voice_loc,
                equipment="Two-Way Radio Network",
            )

    # -------------------------------------------------------------------------
    # MODALITY 3: PDF Incident Report Ingestion (Person 1 Parser Component)
    # -------------------------------------------------------------------------
    elif ingestion_mode == "PDF Incident Report Ingestion (Person 1 Parser)":
        render_html("""
        <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 14px;">
            Direct integration with Person 1's automated PDF Document Ingestion Engine (<code style="color: #38bdf8;">person1_data_pipeline/parser.py</code>).
            Extracts unstructured narrative text via PyPDF2 / pypdf, parses operational entities, and feeds them directly into the Precursor Reasoning Agent.
        </div>
        """)

        source_tab = st.radio(
            "Select PDF Input Source:",
            ["Upload Incident PDF Document", "Select Pre-Loaded Industrial PDF (Person 1 Archive)"],
            horizontal=True,
        )

        sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "person1_data_pipeline", "data", "pdf_reports"
        )
        sample_pdfs = []
        if os.path.exists(sample_dir):
            sample_pdfs = [f for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(".pdf")]

        if source_tab == "Select Pre-Loaded Industrial PDF (Person 1 Archive)":
            if not sample_pdfs:
                st.warning("No pre-loaded PDFs found in person1_data_pipeline/data/pdf_reports/.")
            else:
                selected_sample = st.selectbox(
                    "Select Pre-Loaded PDF Incident Report from Warehouse:",
                    sample_pdfs,
                    format_func=lambda x: f"[PDF REPORT] {x.replace('_', ' ').replace('.pdf', '')}"
                )
                sample_path = os.path.join(sample_dir, selected_sample)

                col_btn, col_info = st.columns([1, 2])
                with col_btn:
                    parse_now = st.button("Extract & Parse PDF Document", use_container_width=True)
                with col_info:
                    render_html(f"<div style='font-size: 0.82rem; color: #94a3b8; padding-top: 8px;'>Source File: <code>{selected_sample}</code></div>")

                if parse_now or ("active_pdf_data" in st.session_state and st.session_state.get("active_pdf_name") == selected_sample):
                    with st.spinner("Invoking Person 1 extract_text_from_pdf & parse_incident_text..."):
                        parsed_meta = agent.parse_pdf_document(sample_path, filename=selected_sample)
                        st.session_state["active_pdf_data"] = parsed_meta
                        st.session_state["active_pdf_name"] = selected_sample

        else:
            uploaded_pdf = st.file_uploader("Upload an OSHA or Industrial Near-Miss PDF Document", type=["pdf"])
            if uploaded_pdf is not None:
                if st.session_state.get("active_pdf_name") != uploaded_pdf.name:
                    with st.spinner(f"Person 1 Parser reading bytes from '{uploaded_pdf.name}'..."):
                        parsed_meta = agent.parse_pdf_document(uploaded_pdf, filename=uploaded_pdf.name)
                        st.session_state["active_pdf_data"] = parsed_meta
                        st.session_state["active_pdf_name"] = uploaded_pdf.name

        # Render parsed document inspection card & confirmation form
        if "active_pdf_data" in st.session_state and st.session_state["active_pdf_data"]:
            meta = st.session_state["active_pdf_data"]
            pdf_raw_text = meta.get("raw_description", "")

            render_html(f"""
            <div class="glass-panel" style="margin-top: 14px; border-left: 4px solid #38bdf8;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">
                            Document: {meta.get('source_file', 'Report.pdf')}
                        </span>
                        <div style="font-size: 0.80rem; color: #94a3b8; margin-top: 2px;">
                            Engine: <b>Person 1 PyPDF2 Parser</b> | Characters Extracted: <b>{len(pdf_raw_text)}</b> | Status: <b style="color: #34d399;">Parsed Successfully</b>
                        </div>
                    </div>
                    <span class="badge-pill" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4);">
                        Parser Ready
                    </span>
                </div>
            </div>
            """)

            with st.form("pdf_inspection_form"):
                render_html("<div style='font-weight: 700; color: #38bdf8; margin-bottom: 6px;'>Extracted Safety Precursor Metadata:</div>")
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    pdf_dept = st.text_input("Extracted Department", value=meta.get("department", "General Operations"))
                with col_p2:
                    pdf_loc = st.text_input("Specific Location / Zone", value=meta.get("location_specific", "Facility Floor"))
                with col_p3:
                    pdf_equip = st.text_input("Equipment Involved", value=meta.get("equipment_involved", "Industrial Machinery"))

                pdf_narrative = st.text_area(
                    "Extracted Document Text / Observation Narrative:",
                    value=pdf_raw_text,
                    height=180,
                )

                submit_pdf = st.form_submit_button("Run Precursor Agent Analysis on PDF Content", use_container_width=True)

            if submit_pdf:
                if not pdf_narrative.strip():
                    st.error("Cannot process an empty document narrative.")
                else:
                    _execute_agent_analysis(agent, pdf_narrative, pdf_dept, pdf_loc, equipment=pdf_equip)

        # Batch Directory Processing Expander (Person 1 Feature)
        with st.expander("Batch Directory Ingestion: Parse All PDFs in Archive"):
            render_html("""
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;">
                Executes <code>process_pdf_directory()</code> from Person 1's backend module across the <code>person1_data_pipeline/data/pdf_reports/</code> folder.
                All scanned PDF records are automatically structured and saved into the SQLite warehouse.
            </div>
            """)
            if st.button("Run Person 1 Batch PDF Pipeline", use_container_width=True):
                with st.spinner("Iterating through PDF directory, extracting documents, and storing in warehouse..."):
                    count = agent.batch_process_pdf_directory()
                    st.success(f"Batch execution completed! Processed and cataloged {count} PDF report(s) in warehouse.")
                    time.sleep(1)
                    st.rerun()

    # -------------------------------------------------------------------------
    # MODALITY 4: Batch JSON File Ingestion
    # -------------------------------------------------------------------------
    elif ingestion_mode == "Batch JSON File Ingestion":
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 12px;">
                Upload multiple historical safety reports formatted as JSON. The agent validates the schema, executes batch precursor analysis, and stores records in the warehouse.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Choose a JSON report dataset", type=["json"])
        sample_batch_btn = st.button("Load Sample Batch JSON (3 Reports)")
        batch_data = None

        if sample_batch_btn:
            batch_data = [
                {
                    "facility": "Plant Alpha - Midwest Complex",
                    "department": "High-Bay Warehousing & Logistics",
                    "location_specific": "Battery Charging Dock Bay 4",
                    "equipment_involved": "Hyster Reach Truck",
                    "raw_text": "Battery hoist chain had a cracked master link. Operator noticed link opening under 2,000 lb load before battery was lifted clear of chassis."
                },
                {
                    "facility": "Plant Beta - Gulf Coast Refinery",
                    "department": "Chemical Processing & Synthesis",
                    "location_specific": "Benzene Stripper Column C-201",
                    "equipment_involved": "Pressure Relief Valve PRV-201",
                    "raw_text": "Car-seal on manual block valve upstream of PRV-201 was cut and valve was throttled 50 percent shut during routine shift turnover."
                },
                {
                    "facility": "Facility Delta - Apex Logistics Hub",
                    "department": "Plant Facilities & Maintenance",
                    "location_specific": "Main Air Compressor Room",
                    "equipment_involved": "Sullair 150HP Screw Compressor",
                    "raw_text": "High temperature cutoff sensor was unplugged and taped with electrical tape to keep compressor running despite overheating oil alarm."
                }
            ]
            st.session_state["staged_batch_data"] = batch_data

        if uploaded_file is not None:
            try:
                batch_data = json.load(uploaded_file)
                st.session_state["staged_batch_data"] = batch_data
            except Exception as e:
                st.error(f"Invalid JSON file format: {e}")

        if "staged_batch_data" in st.session_state and st.session_state["staged_batch_data"]:
            staged = st.session_state["staged_batch_data"]
            st.success(f"Successfully staged {len(staged)} report(s) for batch processing.")
            st.json(staged[:2])

            if st.button("Ingest & Analyze Batch Reports", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                for i, item in enumerate(staged):
                    status_text.text(f"Processing report {i+1} of {len(staged)}: {item.get('department', 'General')}...")
                    agent.analyze_report(
                        raw_text=item.get("raw_text", "No text"),
                        department=item.get("department", "General Operations"),
                        location=item.get("location_specific", "Facility Floor"),
                        facility=item.get("facility", "Industrial Plant"),
                        equipment=item.get("equipment_involved", "N/A"),
                    )
                    progress_bar.progress((i + 1) / len(staged))
                    time.sleep(0.1)

                status_text.text("Batch processing complete! All records added to warehouse.")
                st.success(f"Ingested and scored {len(staged)} reports successfully.")
                st.session_state.pop("staged_batch_data", None)
                st.rerun()

    # -------------------------------------------------------------------------
    # Render Latest Agent Analysis Output Card (Matching Person 2 Schema Exactly)
    # -------------------------------------------------------------------------
    if "latest_trace" in st.session_state and st.session_state["latest_trace"].final_enriched_report:
        trace = st.session_state["latest_trace"]
        rep = trace.final_enriched_report
        ext = rep.extraction
        ass = rep.assessment

        # Extract Person 2 specific fields if present in trace
        raw_ext = trace.raw_extraction_data or {}
        raw_cls = trace.raw_classification_data or {}

        risk_factors = raw_ext.get("risk_factors", ext.precursor_events)
        hazards = raw_ext.get("hazards", [ext.primary_hazard])
        missing_info = raw_ext.get("missing_information", [])
        consequences = raw_ext.get("potential_consequences", ext.failed_safeguards)
        confidence = float(raw_cls.get("confidence", ass.confidence))
        human_review_req = bool(raw_cls.get("human_review_required", ass.escalation_potential))

        render_html("<div style='height: 20px;'></div>")
        render_html("""
        <div class="view-title">
            Agent Precursor Extraction & Risk Classification
        </div>
        """)

        # Human Review Alert if required by Person 2's model
        if human_review_req:
            render_html("""
            <div style="background: rgba(251, 191, 36, 0.15); border: 1px solid rgba(251, 191, 36, 0.5); padding: 12px 16px; border-radius: 8px; margin-bottom: 14px;">
                <b style="color: #fde68a;">[HUMAN REVIEW REQUIRED]</b>
                <span style="color: #f1f5f9; font-size: 0.88rem; margin-left: 8px;">
                    Model flagged uncertainty (Confidence below 0.60 or critical precursor combination). A safety officer should verify on the Overrides tab.
                </span>
            </div>
            """)

        risk_class = ass.risk_level.value.lower()
        render_html(f"""
        <div class="glass-panel" style="border-left: 4px solid var(--tier-{ 'high' if risk_class == 'high' else ('medium' if risk_class == 'medium' else 'low') });">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <div>
                    <span style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{hazards[0] if hazards else ext.primary_hazard}</span>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                        Report ID: <code>{rep.report.id}</code> | Category: <b>{ext.hazard_category.value}</b>
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="badge-pill {risk_class}">{ass.risk_level.value} Risk</span>
                    <span class="badge-pill neutral">
                        Confidence: {int(confidence * 100)}%
                    </span>
                    <span class="badge-pill" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4);">
                        Severity: {ass.risk_score}/10
                    </span>
                </div>
            </div>
            
            <div style="background: #0b1120; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px; border-left: 3px solid #38bdf8; font-size: 0.90rem; line-height: 1.5; color: #f1f5f9;">
                <b style="color: #38bdf8;">Agent Reasoning Rationale:</b> {raw_cls.get("reason", ass.rationale)}
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px;">
                <div style="background: #0d1527; padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.76rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
                        Extracted Risk Factors
                    </div>
                    <ul style="margin: 8px 0 0 0; padding-left: 18px; font-size: 0.88rem; color: #f8fafc;">
                        {''.join([f'<li>{p}</li>' for p in risk_factors])}
                    </ul>
                </div>
                <div style="background: #0d1527; padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.76rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
                        Potential Consequences
                    </div>
                    <ul style="margin: 8px 0 0 0; padding-left: 18px; font-size: 0.88rem; color: #f8fafc;">
                        {''.join([f'<li>{s}</li>' for s in consequences])}
                    </ul>
                </div>
                <div style="background: #0d1527; padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.76rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
                        Missing Information in Report
                    </div>
                    <ul style="margin: 8px 0 0 0; padding-left: 18px; font-size: 0.88rem; color: #fcd34d;">
                        {''.join([f'<li>{m}</li>' for m in missing_info]) if missing_info else '<li>All essential fields identified.</li>'}
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 14px; font-size: 0.88rem; color: #34d399; background: rgba(16, 185, 129, 0.12); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">
                <b>Actionable Mitigation:</b> {ext.recommended_mitigation}
            </div>
        </div>
        """)


def _execute_agent_analysis(agent, text, dept, loc, equipment):
    """Executes agent pipeline with live visual spinner feedback."""
    with st.spinner("Agent running perception scans, OSHA vector grounding, and severity calculation..."):
        trace = agent.analyze_report(
            raw_text=text,
            department=dept,
            location=loc,
            equipment=equipment,
        )
        st.session_state["latest_trace"] = trace
        st.success(f"Analysis completed for Report {trace.report_id}!")
