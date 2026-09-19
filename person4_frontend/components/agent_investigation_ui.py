"""
Person 4: Agent Reasoning & Tool Trace Inspector Component
Provides transparent visibility into intermediate agent reasoning steps,
tool arguments, execution latencies, and grounded OSHA citations.
"""
import streamlit as st
import json
from person2_llm_agent.agent_orchestrator import AgentExecutionTrace


def render_agent_investigation_ui():
    """Renders step-by-step agent tool execution traces and regulatory audit logs."""
    st.markdown("<div class='section-header'>🔬 Agent Reasoning & Tool Orchestration Inspector</div>", unsafe_allow_html=True)
    st.markdown(
        "Inspect the multi-step ReAct reasoning process executed by `IncidentPrecursorAgent`, including "
        "decorated tool calls (`@tool`), latency, retrieved regulatory citations, and dynamic few-shot injection."
    )

    if "latest_trace" not in st.session_state or not st.session_state["latest_trace"]:
        st.info("No active agent analysis trace in session. Run an observation analysis in the Ingestion tab first.")
        return

    trace: AgentExecutionTrace = st.session_state["latest_trace"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Report Analyzed", trace.report_id)
    with c2:
        st.metric("Inference Engine", trace.llm_provider_used)
    with c3:
        st.metric("Tools Invoked", len(trace.tool_calls))

    # Sequential step list
    st.markdown("<div class='section-header'>🪜 Reasoning Workflow Sequence</div>", unsafe_allow_html=True)
    for step in trace.steps_executed:
        st.markdown(f"<div class='trace-step'>{step}</div>", unsafe_allow_html=True)

    # Tool Invocations
    st.markdown("<div class='section-header'>🛠️ Decorated Tool Calls & Verifiable Logic</div>", unsafe_allow_html=True)
    for idx, tc in enumerate(trace.tool_calls, start=1):
        with st.expander(f"Tool Call #{idx}: `{tc.tool_name}` ({tc.execution_time_ms} ms)", expanded=True):
            col_in, col_out = st.columns(2)
            with col_in:
                st.markdown("**Tool Arguments (Input):**")
                st.json(tc.input_args)
            with col_out:
                st.markdown("**Deterministic Output (Result):**")
                st.json(tc.output_result)

    # Grounded RAG Citations
    st.markdown("<div class='section-header'>📚 Grounded OSHA Citations Retrieved</div>", unsafe_allow_html=True)
    if trace.retrieved_citations:
        st.write(", ".join([f"`{c}`" for c in trace.retrieved_citations]))
    else:
        st.write("No specific regulatory standards cited.")

    # Dynamic Few-Shot Injected
    st.markdown("<div class='section-header'>💡 Dynamic Few-Shot Corrections Injected Into Prompt</div>", unsafe_allow_html=True)
    if trace.dynamic_few_shot_applied:
        for exemplar in trace.dynamic_few_shot_applied:
            st.markdown(f"- <code>{exemplar}</code>", unsafe_allow_html=True)
    else:
        st.info("Baseline classification prompt utilized (no prior matching overrides).")
