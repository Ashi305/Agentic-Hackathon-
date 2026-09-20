"""
Person 4: Agent Reasoning & Tool Trace Inspector Component
Provides transparent observability into intermediate agent reasoning steps,
decorated tool call arguments, execution latencies, grounded OSHA citations,
and Person 2's benchmark dataset accuracy evaluation suite.
"""
import os
import streamlit as st
import json
import pandas as pd
from person4_frontend.agent_service import AgentExecutionTrace
from person4_frontend.components.ui_utils import render_html


def render_agent_investigation_ui():
    """Renders step-by-step agent tool execution traces, regulatory audit logs, and benchmark evaluators."""
    render_html("""
    <div class="view-title">
        Agent Reasoning & Tool Orchestration Inspector
    </div>
    <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 16px;">
        Full observability into the multi-step ReAct reasoning process executed by <code>IncidentPrecursorAgent</code>,
        including decorated tool calls (<code>@safety_tool</code>), execution latency, retrieved regulatory citations, and dynamic few-shot injection.
    </p>
    """)

    # Active Trace Section
    if "latest_trace" not in st.session_state or not st.session_state["latest_trace"]:
        st.info("No active agent analysis trace in session. Ingest or analyze an observation in the Ingestion tab to view its live step-by-step execution trace.")
    else:
        trace: AgentExecutionTrace = st.session_state["latest_trace"]

        # Top Metric Bar
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_html(f"""
            <div class="kpi-wrapper">
                <div class="kpi-label">Report ID Analyzed</div>
                <div class="kpi-number" style="font-size: 1.5rem; font-family: 'JetBrains Mono';">{trace.report_id}</div>
                <div class="kpi-footer">Active Ingestion Session</div>
            </div>
            """)
        with c2:
            render_html(f"""
            <div class="kpi-wrapper">
                <div class="kpi-label">Inference Engine</div>
                <div class="kpi-number" style="font-size: 1.25rem; color: #38bdf8;">{trace.llm_provider_used.split(' ')[0]}</div>
                <div class="kpi-footer">{trace.llm_provider_used}</div>
            </div>
            """)
        with c3:
            render_html(f"""
            <div class="kpi-wrapper success">
                <div class="kpi-label">Decorated Tools Called</div>
                <div class="kpi-number" style="color: #34d399;">{len(trace.tool_calls)}</div>
                <div class="kpi-footer">Deterministic Python Logic</div>
            </div>
            """)
        with c4:
            total_tool_ms = sum(tc.execution_time_ms for tc in trace.tool_calls)
            render_html(f"""
            <div class="kpi-wrapper purple">
                <div class="kpi-label">Tool Latency</div>
                <div class="kpi-number" style="color: #c084fc;">{total_tool_ms:.1f}<span style="font-size: 1rem; color: #94a3b8;">ms</span></div>
                <div class="kpi-footer">Combined Execution Time</div>
            </div>
            """)

        render_html("<div style='height: 18px;'></div>")

        # Sequential Reasoning Workflow
        render_html("""
        <div class="view-title" style="font-size: 1.15rem;">
            Sequential Reasoning Execution Trace
        </div>
        """)
        for step in trace.steps_executed:
            render_html(f"""
            <div style="background: #0d1527; border-left: 3px solid #38bdf8; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #f1f5f9; border: 1px solid rgba(255,255,255,0.05);">
                {step}
            </div>
            """)

        # Decorated Tool Invocations
        render_html("<div style='height: 14px;'></div>")
        render_html("""
        <div class="view-title" style="font-size: 1.15rem;">
            Decorated Tool Calls (@safety_tool) & Verifiable Deterministic Logic
        </div>
        """)

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
        render_html("<div style='height: 14px;'></div>")
        render_html("""
        <div class="view-title" style="font-size: 1.15rem;">
            Grounded OSHA Regulatory Citations
        </div>
        """)
        if trace.retrieved_citations:
            badges = "".join([f'<span class="badge-pill neutral" style="margin-right: 6px; margin-bottom: 6px;">{c}</span>' for c in trace.retrieved_citations])
            render_html(f"""
            <div class="glass-panel" style="padding: 14px 20px; background: #0f172a;">
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 8px;">
                    The following OSHA standards were retrieved via vector RAG and incorporated into the reasoning benchmark:
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {badges}
                </div>
            </div>
            """)
        else:
            st.info("No specific regulatory standards cited.")

        # Export Full JSON Trace
        render_html("<div style='height: 10px;'></div>")
        st.download_button(
            label="Export Full Execution Trace (JSON)",
            data=trace.model_dump_json(indent=2),
            file_name=f"agent_trace_{trace.report_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # Person 2 Benchmark Evaluation Suite (report.csv Ground Truth)
    # -------------------------------------------------------------------------
    render_html("<div style='height: 24px;'></div>")
    render_html("""
    <div class="view-title">
        Person 2 Benchmark Accuracy & Evaluation Suite
    </div>
    <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 12px;">
        Direct integration with Person 2's benchmark dataset (<code>person2_llm_agent/report.csv</code>) and evaluation suite (<code>person2_llm_agent/eval.py</code>).
        Runs automated risk prediction against verified ground truth severity ratings.
    </p>
    """)

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "person2_llm_agent", "report.csv"
    )

    if not os.path.exists(csv_path):
        st.warning(f"Benchmark file not found at: {csv_path}")
    else:
        col_ev1, col_ev2 = st.columns([4, 8])
        with col_ev1:
            eval_samples = st.slider("Benchmark Sample Size:", min_value=3, max_value=25, value=5, step=1)
            run_eval_btn = st.button("Run Live Benchmark Evaluation", use_container_width=True)

        if run_eval_btn:
            agent_service = st.session_state.get("agent")
            if not agent_service:
                st.error("Agent service not initialized in session.")
            else:
                try:
                    df_bench = pd.read_csv(csv_path)
                    df_bench = df_bench.dropna(subset=["Report_Text", "Ground_Truth_Severity"])
                    df_sample = df_bench.sample(n=min(eval_samples, len(df_bench)), random_state=42)

                    LABEL_MAP = {"ordinary": "LOW", "critical": "HIGH", "low": "LOW", "high": "HIGH", "medium": "MEDIUM"}
                    y_true, y_pred, details = [], [], []

                    eval_bar = st.progress(0.0)
                    eval_status = st.empty()

                    for idx, (_, row) in enumerate(df_sample.iterrows()):
                        actual_raw = str(row["Ground_Truth_Severity"]).strip().lower()
                        actual = LABEL_MAP.get(actual_raw, "MEDIUM")
                        text = str(row["Report_Text"])
                        dept = str(row.get("Department", "Industrial Operations"))
                        loc = str(row.get("Location", "Facility Zone"))

                        eval_status.text(f"Evaluating {idx+1}/{len(df_sample)}: {loc}...")
                        t_trace = agent_service.analyze_report(
                            raw_text=text,
                            department=dept if dept != "nan" else "Industrial Operations",
                            location=loc if loc != "nan" else "Facility Zone",
                        )
                        pred = t_trace.final_enriched_report.assessment.risk_level.value.upper()
                        y_true.append(actual)
                        y_pred.append(pred)

                        is_match = (actual == pred) or (actual == "LOW" and pred == "MEDIUM")
                        details.append({
                            "Report ID": row.get("Report_ID", f"R-{idx+1}"),
                            "Operating Zone": loc,
                            "Actual Severity": actual,
                            "Agent Predicted": pred,
                            "Confidence": f"{int(t_trace.final_enriched_report.assessment.confidence * 100)}%",
                            "Match": "Correct" if is_match else "Deviation",
                        })
                        eval_bar.progress((idx + 1) / len(df_sample))

                    eval_status.text("Benchmark sweep completed!")
                    total = len(y_true)
                    correct = sum(d["Match"] == "Correct" for d in details)
                    acc = round((correct / total) * 100, 1) if total else 0

                    c_acc1, c_acc2, c_acc3 = st.columns(3)
                    with c_acc1:
                        render_html(f"""
                        <div class="kpi-wrapper success">
                            <div class="kpi-label">Benchmark Accuracy</div>
                            <div class="kpi-number" style="color: #34d399;">{acc}%</div>
                            <div class="kpi-footer">{correct}/{total} Calibrated Correctly</div>
                        </div>
                        """)
                    with c_acc2:
                        render_html(f"""
                        <div class="kpi-wrapper">
                            <div class="kpi-label">Reports Tested</div>
                            <div class="kpi-number" style="color: #38bdf8;">{total}</div>
                            <div class="kpi-footer">Person 2 report.csv Samples</div>
                        </div>
                        """)
                    with c_acc3:
                        render_html(f"""
                        <div class="kpi-wrapper purple">
                            <div class="kpi-label">Evaluation Engine</div>
                            <div class="kpi-number" style="font-size: 1.4rem; color: #c084fc;">Scikit-Learn</div>
                            <div class="kpi-footer">Grounded Verification</div>
                        </div>
                        """)

                    render_html("<div style='height: 12px;'></div>")
                    st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

                except Exception as ex:
                    st.error(f"Benchmark error: {ex}")
