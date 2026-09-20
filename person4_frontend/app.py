"""
Person 4: Streamlit Master Application
Incident Precursor Reasoning Agent (IPRA) - EHS Command Center
Track: Perception, Voice & Document Reasoning Agents
"""
import os
import sys

# Ensure repository root is on Python module search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st
import pandas as pd

from person1_data_pipeline.storage import SafetyStorage
from person4_frontend.agent_service import FrontendAgentService, FrontendFewShotManager
from person4_frontend.components import (
    render_dashboard_view,
    render_report_ingestion,
    render_risk_override_ui,
    render_agent_investigation_ui,
)


def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_app_state():
    """Initializes shared singleton services in session state."""
    if "storage" not in st.session_state:
        storage = SafetyStorage()
        storage.seed_initial_data(target_count=42)
        st.session_state["storage"] = storage

    if "agent" not in st.session_state:
        agent = FrontendAgentService(storage=st.session_state["storage"])
        st.session_state["agent"] = agent

    if "few_shot_manager" not in st.session_state:
        st.session_state["few_shot_manager"] = FrontendFewShotManager(storage=st.session_state["storage"])


def main():
    st.set_page_config(
        page_title="Incident Precursor Reasoning Agent | EHS Command Center",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_custom_css()
    init_app_state()

    storage: SafetyStorage = st.session_state["storage"]
    agent: FrontendAgentService = st.session_state["agent"]
    few_shot_manager: FrontendFewShotManager = st.session_state["few_shot_manager"]

    # Sidebar Navigation & System Monitor
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 8px;">
                <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">
                    EHS COMMAND
                </h2>
                <p style="margin: 2px 0 0 0; font-size: 0.74rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                    Incident Precursor Reasoner
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Architecture & Storage Status
        df = storage.to_dataframe()
        report_count = len(df)
        override_count = len(storage.get_all_overrides())
        active_exemplars = len(few_shot_manager.get_latest_exemplars(max_examples=5))

        st.markdown(
            """
            <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px;">
                System Telemetry
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); font-size: 0.85rem; line-height: 1.6;">
                <div style="margin-bottom: 6px;">
                    <span style="color: #94a3b8;">Active Warehouse:</span><br>
                    <b style="color: #ffffff;">{report_count} Reports Ingested</b>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color: #94a3b8;">Human Corrections:</span><br>
                    <b style="color: #c084fc;">{override_count} Logged Overrides</b>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color: #94a3b8;">Dynamic Few-Shot Pool:</span><br>
                    <b style="color: #38bdf8;">{active_exemplars} Active Exemplars</b>
                </div>
                <div>
                    <span style="color: #94a3b8;">Storage Engine:</span><br>
                    <code>SQLite (safety_warehouse.db)</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        llm_engine_label = "Gemini Cloud API" if has_gemini else "Deterministic Expert Rule Engine"
        st.markdown(
            f"""
            <div style="margin-top: 12px; padding: 10px; background: rgba(56, 189, 248, 0.08); border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.25); font-size: 0.80rem;">
                <span style="font-weight: 700; color: #38bdf8;">AGENT INFERENCE:</span><br>
                <span style="color: #ffffff;">{llm_engine_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            """
            <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px;">
                Utilities
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Reload Warehouse Benchmark (42 Reports)", use_container_width=True):
            storage.seed_initial_data(target_count=42, force_reload=True)
            st.success("Warehouse refreshed!")
            st.rerun()

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size: 0.72rem; color: #64748b; line-height: 1.4;'>"
            "National Level Agentic AI Hackathon<br>"
            "Track: Perception, Voice & Document Reasoning Agents<br>"
            "Person 4: Streamlit Frontend Lead"
            "</div>",
            unsafe_allow_html=True,
        )

    # Master Application Header Banner
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.08);">
            <div>
                <h1 style="margin: 0; font-size: 2.15rem; font-weight: 800; letter-spacing: -0.03em; color: #ffffff;">
                    Safety Report Analysis Agent
                </h1>
                <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.95rem;">
                    Autonomous incident precursor extraction, grounded OSHA RAG scoring, and dynamic few-shot feedback.
                </p>
            </div>
            <div style="text-align: right;">
                <span class="badge-pill high" style="font-size: 0.80rem;">LIVE PRECURSOR MONITORING</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Tab Navigation without emojis
    tab1, tab2, tab3, tab4 = st.tabs([
        "Executive Risk Dashboard",
        "Multi-Modal Ingestion Workbench",
        "Safety Officer Overrides (Compulsory Add-On)",
        "Agent Reasoning & Tool Inspector",
    ])

    with tab1:
        render_dashboard_view(df)

    with tab2:
        render_report_ingestion(agent)

    with tab3:
        render_risk_override_ui(storage, few_shot_manager)

    with tab4:
        render_agent_investigation_ui()


if __name__ == "__main__":
    main()
