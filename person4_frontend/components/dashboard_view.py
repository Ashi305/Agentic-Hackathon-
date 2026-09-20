"""
Person 4: Executive Dashboard Component
Renders interactive KPI metric cards, high-contrast Plotly analytical charts
(Sunburst / Treemaps, Hotspot bars, Risk Donuts), systemic precursor themes,
and Person 3's Defeated Safeguards & Cross-Department Risk matrices.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from person3_analytics.aggregation import SafetyAggregator
from person3_analytics.clustering_themes import PrecursorPatternClusterer
from person3_analytics.metrics import SafetyMetricsCalculator
from person4_frontend.components.ui_utils import render_html


def render_dashboard_view(df: pd.DataFrame):
    """Renders the comprehensive Safety Precursor Command Dashboard."""
    if df.empty:
        st.info("No safety reports loaded in warehouse. Ingest new reports or seed initial data.")
        return

    # Filter Bar
    with st.container():
        f_col1, f_col2, f_col3 = st.columns([4, 4, 4])
        with f_col1:
            departments = ["All Departments"] + sorted(df["department"].dropna().unique().tolist())
            selected_dept = st.selectbox("Department Filter:", departments)
        with f_col2:
            facilities = ["All Facilities"] + sorted(df["facility"].dropna().unique().tolist()) if "facility" in df.columns else ["All Facilities"]
            selected_fac = st.selectbox("Facility Complex Filter:", facilities)
        with f_col3:
            risk_filter = st.selectbox("Risk Level Filter:", ["All Risk Levels", "High", "Medium", "Low"])

    # Apply filters
    filtered_df = df.copy()
    if selected_dept != "All Departments":
        filtered_df = filtered_df[filtered_df["department"] == selected_dept]
    if selected_fac != "All Facilities" and "facility" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["facility"] == selected_fac]
    if risk_filter != "All Risk Levels":
        filtered_df = filtered_df[filtered_df["effective_risk_level"] == risk_filter]

    if filtered_df.empty:
        st.warning("No incident observations match the selected filters.")
        return

    # 1. High-Level KPI Metric Cards
    kpis = SafetyMetricsCalculator.compute_summary_kpis(filtered_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_html(f"""
        <div class="kpi-wrapper">
            <div class="kpi-label">Observations Ingested</div>
            <div class="kpi-number">{kpis['total_reports']}</div>
            <div class="kpi-footer">Active Operating Zones</div>
        </div>
        """)
    with c2:
        render_html(f"""
        <div class="kpi-wrapper danger">
            <div class="kpi-label">High-Risk Precursors</div>
            <div class="kpi-number" style="color: #fb7185;">{kpis['high_risk_count']} <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">({kpis['high_risk_pct']}%)</span></div>
            <div class="kpi-footer">Immediate Escalation Potential</div>
        </div>
        """)
    with c3:
        render_html(f"""
        <div class="kpi-wrapper warning">
            <div class="kpi-label">SIF Precursors (Fatal)</div>
            <div class="kpi-number" style="color: #fcd34d;">{kpis['sif_escalations']}</div>
            <div class="kpi-footer">Critical Barrier Deficits</div>
        </div>
        """)
    with c4:
        render_html(f"""
        <div class="kpi-wrapper">
            <div class="kpi-label">Average Risk Severity</div>
            <div class="kpi-number" style="color: #38bdf8;">{kpis['avg_risk_score']}<span style="font-size: 1rem; color: #94a3b8;">/10</span></div>
            <div class="kpi-footer">Deterministic Score Index</div>
        </div>
        """)
    with c5:
        render_html(f"""
        <div class="kpi-wrapper purple">
            <div class="kpi-label">Human Overrides</div>
            <div class="kpi-number" style="color: #c084fc;">{kpis['override_count']}</div>
            <div class="kpi-footer">Active Few-Shot Exemplars</div>
        </div>
        """)

    render_html("<div style='height: 16px;'></div>")

    # 2. Executive Precursor Alert Banner (Person 3 Themes)
    clusters = PrecursorPatternClusterer.cluster_reports_by_theme(filtered_df)
    if clusters:
        top_cluster = clusters[0]
        sectors_str = ", ".join(top_cluster.get("affected_departments", [])) or "Industrial Complex"
        render_html(f"""
        <div class="glass-panel" style="border-left: 4px solid #f43f5e; padding: 18px 22px; background: #0f172a; margin-bottom: 16px;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #fb7185; margin-bottom: 6px;">
                EXECUTIVE PRECURSOR ALERT: Systemic Hazard Pattern in '{top_cluster['theme_title']}'
            </div>
            <div style="font-size: 0.90rem; color: #f1f5f9; line-height: 1.6;">
                Across <b>{top_cluster['report_count']}</b> logged observations, <b style="color: #fb7185;">{top_cluster['high_risk_count']} cases</b> were classified as High Risk (Average Severity: <b>{top_cluster['avg_risk_score']}/10.0</b>).<br>
                <span style="color: #38bdf8; font-weight: 600;">Critical Precursor Implication:</span> {top_cluster['precursor_significance']}<br>
                <span style="color: #94a3b8; font-weight: 600;">Primary Operating Sectors Affected:</span> {sectors_str}.<br>
                <span style="color: #34d399; font-weight: 600;">Recommended Immediate Priority:</span> Deploy targeted engineering audits to inspect interlocks, physical barriers, and verify secondary containment in affected zones.
            </div>
        </div>
        """)

    # 3. Interactive Plotly Charts Row 1
    col_chart1, col_chart2 = st.columns([6, 4])

    with col_chart1:
        render_html("""
        <div class="view-title">
            Operational Hazard Hierarchy
        </div>
        """)
        chart_mode = st.radio(
            "Hierarchy View Type:",
            ["Sunburst Multi-Level", "Treemap Area"],
            horizontal=True,
            label_visibility="collapsed"
        )

        hierarchy_df = filtered_df.copy()
        if "hazard_category" not in hierarchy_df.columns:
            hierarchy_df["hazard_category"] = "General"

        color_map = {"High": "#f43f5e", "Medium": "#fbbf24", "Low": "#10b981"}

        if chart_mode == "Sunburst Multi-Level":
            fig_hierarchy = px.sunburst(
                hierarchy_df,
                path=["department", "hazard_category", "effective_risk_level"],
                color="effective_risk_level",
                color_discrete_map=color_map,
                maxdepth=3,
            )
        else:
            fig_hierarchy = px.treemap(
                hierarchy_df,
                path=["department", "hazard_category", "effective_risk_level"],
                color="effective_risk_level",
                color_discrete_map=color_map,
            )

        fig_hierarchy.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans", size=12),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hierarchy, use_container_width=True)

    with col_chart2:
        render_html("""
        <div class="view-title">
            Calibrated Risk Distribution
        </div>
        """)
        dist_df = SafetyAggregator.get_risk_distribution(filtered_df)
        fig_donut = px.pie(
            dist_df,
            values="count",
            names="effective_risk_level",
            hole=0.66,
            color="effective_risk_level",
            color_discrete_map=color_map,
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#070c18', width=2)),
            textfont=dict(color='#ffffff', size=13)
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
            annotations=[dict(text=f"<b>{kpis['total_reports']}</b><br><span style='font-size:12px;color:#cbd5e1;'>REPORTS</span>", x=0.5, y=0.5, font_size=20, font_color="#ffffff", showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # 4. Interactive Plotly Charts Row 2
    col_chart3, col_chart4 = st.columns([5, 5])

    with col_chart3:
        render_html("""
        <div class="view-title">
            Facility Zone Hotspot Index
        </div>
        """)
        hotspots = SafetyAggregator.get_location_hotspots(filtered_df, top_n=7)
        if not hotspots.empty:
            fig_hotspot = px.bar(
                hotspots,
                x="hotspot_score",
                y="location_specific",
                orientation="h",
                color="high_risk_count",
                color_continuous_scale=["#fde047", "#fb923c", "#f43f5e"],
                labels={"hotspot_score": "Composite Hotspot Score", "location_specific": "Operating Zone", "high_risk_count": "High Risk Events"},
            )
            fig_hotspot.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
                xaxis=dict(color="#cbd5e1", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(autorange="reversed", color="#cbd5e1"),
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_colorbar=dict(title=dict(text="High Risk", font=dict(color="#f8fafc")), tickfont=dict(color="#cbd5e1"), thickness=12, len=0.7),
            )
            st.plotly_chart(fig_hotspot, use_container_width=True)

    with col_chart4:
        render_html("""
        <div class="view-title">
            Top Recurring Precursor Signals
        </div>
        """)
        precursors = SafetyAggregator.get_top_recurring_risk_factors(filtered_df, top_n=7)
        if not precursors.empty:
            fig_precursors = px.bar(
                precursors,
                x="frequency",
                y="precursor",
                orientation="h",
                color="frequency",
                color_continuous_scale=["#38bdf8", "#818cf8", "#c084fc"],
                labels={"frequency": "Detection Frequency", "precursor": "Precursor Hazard"},
            )
            fig_precursors.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
                xaxis=dict(color="#cbd5e1", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(autorange="reversed", color="#cbd5e1"),
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_precursors, use_container_width=True)

    # 5. Defeated Safeguards & Department Risk Matrix (Person 3 Integration)
    col_safe1, col_safe2 = st.columns([5, 5])
    with col_safe1:
        render_html("""
        <div class="view-title" style="margin-top: 10px;">
            Defeated Safeguards & Barrier Deficits
        </div>
        """)
        safeguards_df = SafetyAggregator.get_failed_safeguards_frequency(filtered_df, top_n=6)
        if not safeguards_df.empty:
            fig_safeguards = px.bar(
                safeguards_df,
                x="failure_count",
                y="safeguard",
                orientation="h",
                color="failure_count",
                color_continuous_scale=["#38bdf8", "#fb923c", "#f43f5e"],
                labels={"failure_count": "Failure Count", "safeguard": "Compromised Barrier"},
            )
            fig_safeguards.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
                xaxis=dict(color="#cbd5e1", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(autorange="reversed", color="#cbd5e1"),
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_safeguards, use_container_width=True)
        else:
            st.info("No safeguard failure data recorded.")

    with col_safe2:
        render_html("""
        <div class="view-title" style="margin-top: 10px;">
            Cross-Department Risk Distribution Matrix
        </div>
        """)
        dept_matrix = SafetyAggregator.get_department_risk_matrix(filtered_df)
        if not dept_matrix.empty:
            st.dataframe(dept_matrix, use_container_width=True, hide_index=True)
        else:
            st.info("No department matrix available.")

    # 6. Cross-Report Systemic Precursor Themes
    render_html("""
    <div class="view-title" style="margin-top: 14px;">
        Cross-Report Precursor Theme Clusters (Person 3 Engine)
    </div>
    """)
    cluster_records = []
    for c in clusters:
        cluster_records.append({
            "Systemic Hazard Theme": c["theme_title"],
            "Observations": c["report_count"],
            "High Risk Incidents": c["high_risk_count"],
            "Avg Severity (0-10)": c["avg_risk_score"],
            "Affected Operating Sectors": ", ".join(c["affected_departments"]),
            "Precursor Significance": c["precursor_significance"]
        })
    st.dataframe(pd.DataFrame(cluster_records), use_container_width=True, hide_index=True)
