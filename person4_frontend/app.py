"""
Person 4: Streamlit Master Application
Safety Report Analysis Agent (Incident Precursor Detector)
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
from person2_llm_agent.agent_orchestrator import IncidentPrecursorAgent
from person2_llm_agent.few_shot_manager import DynamicFewShotManager
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
        agent = IncidentPrecursorAgent(storage=st.session_state["storage"])
        st.session_state["agent"] = agent

    if "few_shot_manager" not in st.session_state:
        st.session_state["few_shot_manager"] = DynamicFewShotManager(storage=st.session_state["storage"])


def main():
    st.set_page_config(
        page_title="Incident Precursor Agent | EHS Command Center",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_custom_css()
    init_app_state()

    storage: SafetyStorage = st.session_state["storage"]
    agent: IncidentPrecursorAgent = st.session_state["agent"]
    few_shot_manager: DynamicFewShotManager = st.session_state["few_shot_manager"]

    # Sidebar
    with st.sidebar:
        st.markdown("## 🛡️ EHS Agentic Command")
        st.markdown("<p style='font-size: 0.85rem; color: #94a3b8;'>Incident Precursor Detector & Dynamic Few-Shot Reasoner</p>", unsafe_allow_html=True)
        st.divider()

        # System Health & Architecture Status
        df = storage.to_dataframe()
        report_count = len(df)
        override_count = len(storage.get_all_overrides())
        active_exemplars = len(few_shot_manager.get_latest_exemplars(max_examples=5))

        st.markdown("### ⚙️ System Status")
        st.markdown(f"📦 **Reports Ingested:** `{report_count}`")
        st.markdown(f"⚖️ **Logged Overrides:** `{override_count}`")
        st.markdown(f"🧠 **Active Few-Shot Exemplars:** `{active_exemplars}`")
        st.markdown(f"🗄️ **Storage Layer:** `SQLite (safety_warehouse.db)`")
        
        has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        llm_badge = "🟢 Gemini Connected" if has_gemini else "⚡ Deterministic Heuristic Engine (Offline Validated)"
        st.markdown(f"🤖 **Inference Mode:** <br><span style='font-size: 0.82rem; color: #38bdf8;'>{llm_badge}</span>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### 🚀 Quick Utilities")
        if st.button("🔄 Reload & Re-seed Database", use_container_width=True):
            storage.seed_initial_data(target_count=42, force_reload=True)
            st.success("Warehouse re-seeded!")
            st.rerun()

        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size: 0.75rem; color: #64748b; line-height: 1.4;'>"
            "National Level Agentic AI Hackathon<br>"
            "Track: Perception, Voice & Document Reasoning Agents<br>"
            "Person 1: Data Pipeline | Person 2: Agent Engine<br>"
            "Person 3: Analytics | Person 4: Streamlit UI"
            "</div>",
            unsafe_allow_html=True,
        )

    # Main Application Header
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc;">
                    Incident Precursor Detector
                </h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                    Autonomous extraction of incident precursors, grounded OSHA RAG scoring, and dynamic few-shot feedback.
                </p>
            </div>
            <div style="text-align: right;">
                <span class="badge badge-high">Active Monitoring</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Risk Dashboard",
        "📥 Report Ingestion & Real-Time Agent",
        "⚖️ Safety Officer Overrides (Compulsory Add-On)",
        "🔬 Agent Reasoning & Tool Inspector",
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
