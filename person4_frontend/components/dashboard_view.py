"""
Person 4: Executive Dashboard Component
Renders interactive KPI metric cards, high-contrast Plotly analytical charts,
and cross-report systemic precursor themes.
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
        st.info("No safety reports loaded in warehouse. Use the Ingestion tab or seed initial data.")
        return

    # 1. High-Level KPI Metric Cards
    kpis = SafetyMetricsCalculator.compute_summary_kpis(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Reports Ingested</div>
                <div class="kpi-value">{kpis['total_reports']}</div>
                <div class="kpi-subtext">Active Near-Miss Observations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card high">
                <div class="kpi-title">High Risk Incidents</div>
                <div class="kpi-value" style="color: #f87171;">{kpis['high_risk_count']} <span style="font-size: 1rem; color: #94a3b8;">({kpis['high_risk_pct']}%)</span></div>
                <div class="kpi-subtext">Immediate Precursors to SIF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card high">
                <div class="kpi-title">Escalation Potential</div>
                <div class="kpi-value" style="color: #fb923c;">{kpis['sif_escalations']}</div>
                <div class="kpi-subtext">Catastrophe Precursor Signals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Severity Score</div>
                <div class="kpi-value" style="color: #38bdf8;">{kpis['avg_risk_score']}<span style="font-size: 1rem; color: #64748b;">/10</span></div>
                <div class="kpi-subtext">Normalized Precursor Index</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="kpi-card medium">
                <div class="kpi-title">Human Overrides</div>
                <div class="kpi-value" style="color: #c084fc;">{kpis['override_count']}</div>
                <div class="kpi-subtext">Active Few-Shot Exemplars</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2. Strategic Executive Alert Card
    clusters = PrecursorPatternClusterer.cluster_reports_by_theme(df)
    alert_summary = PrecursorPatternClusterer.generate_executive_theme_summary(clusters)
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid #ef4444;">
            <div style="font-size: 0.95rem; line-height: 1.6; color: #e2e8f0;">
                {alert_summary.replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Interactive Plotly Charts Row 1
    col_chart1, col_chart2 = st.columns([4, 6])

    with col_chart1:
        st.markdown("<div class='section-header'>⚡ Calibrated Risk Distribution</div>", unsafe_allow_html=True)
        dist_df = SafetyAggregator.get_risk_distribution(df)
        
        color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        fig_donut = px.pie(
            dist_df,
            values="count",
            names="effective_risk_level",
            hole=0.62,
            color="effective_risk_level",
            color_discrete_map=color_map,
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Inter"),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.markdown("<div class='section-header'>🔥 Location Hotspot Index</div>", unsafe_allow_html=True)
        hotspots = SafetyAggregator.get_location_hotspots(df, top_n=6)
        
        fig_hotspot = px.bar(
            hotspots,
            x="hotspot_score",
            y="location_specific",
            orientation="h",
            color="high_risk_count",
            color_continuous_scale="Reds",
            labels={"hotspot_score": "Composite Hotspot Score", "location_specific": "Facility Zone", "high_risk_count": "High-Risk Events"},
        )
        fig_hotspot.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Inter"),
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_hotspot, use_container_width=True)

    # 4. Interactive Plotly Charts Row 2
    col_chart3, col_chart4 = st.columns([6, 4])

    with col_chart3:
        st.markdown("<div class='section-header'>🚨 Top Recurring Incident Precursors</div>", unsafe_allow_html=True)
        precursors = SafetyAggregator.get_top_recurring_risk_factors(df, top_n=7)
        if not precursors.empty:
            fig_precursors = px.bar(
                precursors,
                x="frequency",
                y="precursor",
                orientation="h",
                color="frequency",
                color_continuous_scale="Viridis",
                labels={"frequency": "Observation Frequency", "precursor": "Precursor Hazard"},
            )
            fig_precursors.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", family="Inter"),
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_precursors, use_container_width=True)

    with col_chart4:
        st.markdown("<div class='section-header'>🛡️ Failed / Bypassed Safeguards</div>", unsafe_allow_html=True)
        safeguards = SafetyAggregator.get_failed_safeguards_frequency(df, top_n=6)
        if not safeguards.empty:
            fig_barriers = px.bar(
                safeguards,
                x="failure_count",
                y="safeguard",
                orientation="h",
                color_discrete_sequence=["#fb923c"],
                labels={"failure_count": "Failure Count", "safeguard": "Safeguard Control"},
            )
            fig_barriers.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc", family="Inter"),
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_barriers, use_container_width=True)

    # 5. Thematic Cluster Table
    st.markdown("<div class='section-header'>🔍 Systemic Precursor Clusters Across Reports</div>", unsafe_allow_html=True)
    cluster_records = []
    for c in clusters:
        cluster_records.append({
            "Systemic Theme": c["theme_title"],
            "Observations": c["report_count"],
            "High Risk Count": c["high_risk_count"],
            "Avg Severity (0-10)": c["avg_risk_score"],
            "Affected Sectors": ", ".join(c["affected_departments"]),
            "Precursor Hazard Significance": c["precursor_significance"]
        })
    st.dataframe(pd.DataFrame(cluster_records), use_container_width=True, hide_index=True)
