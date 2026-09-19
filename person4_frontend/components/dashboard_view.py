"""
Person 4: Executive Dashboard Component
Renders interactive KPI metric cards, high-contrast Plotly analytical charts
(Sunburst / Treemaps, Hotspot bars, Risk Donuts), and systemic precursor themes.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from person3_analytics.aggregation import SafetyAggregator
from person3_analytics.clustering_themes import PrecursorPatternClusterer
from person3_analytics.metrics import SafetyMetricsCalculator


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
            selected_dept = st.selectbox("🏢 Filter by Operating Department:", departments)
        with f_col2:
            facilities = ["All Facilities"] + sorted(df["facility"].dropna().unique().tolist()) if "facility" in df.columns else ["All Facilities"]
            selected_fac = st.selectbox("📍 Filter by Facility Site:", facilities)
        with f_col3:
            risk_filter = st.selectbox("⚡ Filter by Risk Tier:", ["All Risk Levels", "High", "Medium", "Low"])

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
        st.markdown(
            f"""
            <div class="kpi-wrapper">
                <div class="kpi-label">Observations Ingested</div>
                <div class="kpi-number">{kpis['total_reports']}</div>
                <div class="kpi-footer">Across Active Operating Zones</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-wrapper danger">
                <div class="kpi-label">High-Risk Precursors</div>
                <div class="kpi-number" style="color: #fb7185;">{kpis['high_risk_count']} <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">({kpis['high_risk_pct']}%)</span></div>
                <div class="kpi-footer">Immediate Escalation Potential</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-wrapper warning">
                <div class="kpi-label">SIF Precursors (Fatal)</div>
                <div class="kpi-number" style="color: #fcd34d;">{kpis['sif_escalations']}</div>
                <div class="kpi-footer">Barrier Failure Warning Signs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-wrapper">
                <div class="kpi-label">Average Risk Severity</div>
                <div class="kpi-number" style="color: #38bdf8;">{kpis['avg_risk_score']}<span style="font-size: 1rem; color: #64748b;">/10</span></div>
                <div class="kpi-footer">Deterministic Score Index</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="kpi-wrapper purple">
                <div class="kpi-label">Human Overrides</div>
                <div class="kpi-number" style="color: #c084fc;">{kpis['override_count']}</div>
                <div class="kpi-footer">Active Dynamic Few-Shot Pool</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Executive Precursor Alert Banner
    clusters = PrecursorPatternClusterer.cluster_reports_by_theme(filtered_df)
    alert_summary = PrecursorPatternClusterer.generate_executive_theme_summary(clusters)
    st.markdown(
        f"""
        <div class="glass-panel" style="border-left: 4px solid #f43f5e; padding: 18px 22px;">
            <div style="font-size: 0.95rem; line-height: 1.6; color: #e2e8f0;">
                {alert_summary.replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Interactive Plotly Charts Row 1 (Hierarchical Sunburst/Treemap & Calibrated Donut)
    col_chart1, col_chart2 = st.columns([6, 4])

    with col_chart1:
        st.markdown(
            """
            <div class="view-title">
                <span>🌐 Operational Hazard Hierarchy</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chart_mode = st.radio(
            "Hierarchy View Type:",
            ["Sunburst Multi-Level", "Treemap Area"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Build hierarchy DataFrame
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
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hierarchy, use_container_width=True)

    with col_chart2:
        st.markdown(
            """
            <div class="view-title">
                <span>⚡ Calibrated Risk Distribution</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
            marker=dict(line=dict(color='#0b0f19', width=2))
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
            annotations=[dict(text=f"<b>{kpis['total_reports']}</b><br><span style='font-size:12px;color:#94a3b8;'>REPORTS</span>", x=0.5, y=0.5, font_size=20, font_color="#ffffff", showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # 4. Interactive Plotly Charts Row 2 (Hotspot Bar & Precursor Frequency)
    col_chart3, col_chart4 = st.columns([5, 5])

    with col_chart3:
        st.markdown(
            """
            <div class="view-title">
                <span>🔥 Facility Zone Hotspot Index</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
                coloraxis_colorbar=dict(title="High Risk", thickness=12, len=0.7),
            )
            st.plotly_chart(fig_hotspot, use_container_width=True)

    with col_chart4:
        st.markdown(
            """
            <div class="view-title">
                <span>🚨 Top Recurring Precursor Signals</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_precursors, use_container_width=True)

    # 5. Cross-Report Systemic Precursor Themes
    st.markdown(
        """
        <div class="view-title" style="margin-top: 14px;">
            <span>🔍 Cross-Report Precursor Theme Clusters</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
