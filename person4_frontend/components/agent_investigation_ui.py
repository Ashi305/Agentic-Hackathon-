"""
Person 4: Agent Reasoning & Tool Trace Inspector Component
Provides transparent observability into intermediate agent reasoning steps,
decorated tool call arguments, execution latencies, and grounded OSHA citations.
"""
import streamlit as st
import json
from person4_frontend.agent_service import AgentExecutionTrace


def render_agent_investigation_ui():
    """Renders step-by-step agent tool execution traces and regulatory audit logs."""
    st.markdown(
        """
        <div class="view-title">
            Agent Reasoning & Tool Orchestration Inspector
        </div>
        <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 16px;">
            Full observability into the multi-step ReAct reasoning process executed by <code>IncidentPrecursorAgent</code>,
            including decorated tool calls (<code>@tool</code>), execution latency, retrieved regulatory citations, and dynamic few-shot injection.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if "latest_trace" not in st.session_state or not st.session_state["latest_trace"]:
        st.info("No active agent analysis trace in session. Ingest or analyze an observation in the Ingestion tab first.")
        return

    trace: AgentExecutionTrace = st.session_state["latest_trace"]

    # Top Metric Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-wrapper">
                <div class="kpi-label">Report ID Analyzed</div>
                <div class="kpi-number" style="font-size: 1.5rem; font-family: 'JetBrains Mono';">{trace.report_id}</div>
                <div class="kpi-footer">Active Ingestion Session</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-wrapper">
                <div class="kpi-label">Inference Engine</div>
                <div class="kpi-number" style="font-size: 1.25rem; color: #38bdf8;">{trace.llm_provider_used.split(' ')[0]}</div>
                <div class="kpi-footer">{trace.llm_provider_used}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-wrapper success">
                <div class="kpi-label">Decorated Tools Called</div>
                <div class="kpi-number" style="color: #34d399;">{len(trace.tool_calls)}</div>
                <div class="kpi-footer">Deterministic Python Logic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        total_tool_ms = sum(tc.execution_time_ms for tc in trace.tool_calls)
        st.markdown(
            f"""
            <div class="kpi-wrapper purple">
                <div class="kpi-label">Tool Latency</div>
                <div class="kpi-number" style="color: #c084fc;">{total_tool_ms:.1f}<span style="font-size: 1rem; color: #94a3b8;">ms</span></div>
                <div class="kpi-footer">Combined Execution Time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Sequential Reasoning Workflow
    st.markdown(
        """
        <div class="view-title" style="font-size: 1.15rem;">
            Sequential Reasoning Execution Trace
        </div>
        """,
        unsafe_allow_html=True,
    )
    for step in trace.steps_executed:
        st.markdown(
            f"""
            <div style="background: #0d1527; border-left: 3px solid #38bdf8; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #f1f5f9; border-top: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05);">
                {step}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Decorated Tool Invocations
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="view-title" style="font-size: 1.15rem;">
            Decorated Tool Calls (@tool) & Verifiable Deterministic Logic
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, tc in enumerate(trace.tool_calls, start=1):
        with st.expander(f"Tool Call #{idx}: `{tc.tool_name}` — Latency: {tc.execution_time_ms} ms", expanded=True):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Tool Arguments (Typed Input):**")
                st.json(tc.input_args)
            with t_col2:
                st.markdown("**Deterministic Output (Calculated Result):**")
                st.json(tc.output_result)

    # Grounded RAG Citations
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="view-title" style="font-size: 1.15rem;">
            Grounded OSHA Regulatory Citations
        </div>
        """,
        unsafe_allow_html=True,
    )
    if trace.retrieved_citations:
        st.markdown(
            f"""
            <div class="glass-panel" style="padding: 14px 20px; background: #0f172a;">
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                    The following OSHA standards were retrieved via vector RAG and incorporated into the reasoning benchmark:
                </div>
                <div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px;">
                    {''.join([f'<span class="badge-pill" style="background: rgba(56, 189, 248, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4);">{c}</span>' for c in trace.retrieved_citations])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No specific regulatory standards cited.")

    # Export Full JSON Trace
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="Export Full Execution Trace (JSON)",
        data=trace.model_dump_json(indent=2),
        file_name=f"agent_trace_{trace.report_id}.json",
        mime="application/json",
        width="stretch",
    )
