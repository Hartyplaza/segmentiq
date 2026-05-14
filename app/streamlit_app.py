import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="SegmentIQ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens — Light theme with VS Code style nav ─────────────────────────
BG_PAGE    = "#f8f9fc"
BG_CARD    = "#ffffff"
BG_CARD2   = "#f1f4f9"
BG_NAV     = "#1e2330"
BG_NAV2    = "#252b3b"
BORDER     = "#e2e8f0"
BORDER2    = "#cbd5e1"
TEXT_PRI   = "#0f172a"
TEXT_MUT   = "#64748b"
TEXT_NAV   = "#94a3b8"
TEXT_HEAD  = "#0f172a"
PRIMARY    = "#6366f1"
SECONDARY  = "#818cf8"
SUCCESS    = "#059669"
WARNING    = "#d97706"
DANGER     = "#dc2626"
ACCENT     = "#0ea5e9"

SEG_COLORS = {
    "Champions":        SUCCESS,
    "Lapsed Customers": DANGER,
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG_PAGE} !important;
    color: {TEXT_PRI} !important;
}}
.stApp {{ background-color: {BG_PAGE} !important; }}

/* Sidebar — dark VS Code style */
section[data-testid="stSidebar"] {{
    background-color: {BG_NAV} !important;
    border-right: 1px solid #2d3548 !important;
    width: 220px !important;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_NAV} !important; }}

/* Nav radio as vertical tabs */
div[data-testid="stRadio"] > div {{
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}}
div[data-testid="stRadio"] label {{
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 0.8rem !important;
    color: {TEXT_NAV} !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.15s !important;
    border-left: 3px solid transparent !important;
}}
div[data-testid="stRadio"] label:hover {{
    background: rgba(99,102,241,0.1) !important;
    color: #e2e8f0 !important;
    border-left-color: rgba(99,102,241,0.5) !important;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: rgba(99,102,241,0.15) !important;
    color: #ffffff !important;
    border-left: 3px solid {PRIMARY} !important;
    font-weight: 600 !important;
}}
div[data-testid="stRadio"] input {{ display: none !important; }}
div[data-testid="stRadio"] > label {{ display: none !important; }}

/* Cards */
.card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.card-title {{
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: {TEXT_MUT}; margin-bottom: 0.8rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid {BORDER};
}}

/* Metric tiles */
.metric-tile {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1.2rem 1.5rem;
    border-top: 3px solid {PRIMARY};
}}
.metric-label {{ color: {TEXT_MUT}; font-size: 0.72rem;
    font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.4rem; }}
.metric-value {{ font-size: 1.8rem; font-weight: 700; color: {TEXT_HEAD}; }}
.metric-sub {{ font-size: 0.75rem; color: {TEXT_MUT}; margin-top: 0.2rem; }}

/* Segment badges */
.badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
}}
.badge-champion {{
    background: rgba(5,150,105,0.1);
    border: 1px solid rgba(5,150,105,0.3);
    color: {SUCCESS};
}}
.badge-lapsed {{
    background: rgba(220,38,38,0.1);
    border: 1px solid rgba(220,38,38,0.3);
    color: {DANGER};
}}

/* Section header */
.section-header {{
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid {BORDER};
}}
.section-title {{
    font-size: 1.3rem; font-weight: 700;
    color: {TEXT_HEAD}; margin: 0;
}}
.section-sub {{
    color: {TEXT_MUT}; font-size: 0.85rem; margin-top: 0.3rem;
}}

/* Recommendation cards */
.rec-card {{
    border-radius: 10px; padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem; border: 1px solid;
}}
.rec-champion {{
    background: rgba(5,150,105,0.05);
    border-color: rgba(5,150,105,0.2);
}}
.rec-lapsed {{
    background: rgba(220,38,38,0.05);
    border-color: rgba(220,38,38,0.2);
}}

.stButton > button {{
    background: {PRIMARY} !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.85rem !important;
}}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {{
    color: {TEXT_MUT} !important; font-size: 0.78rem !important;
}}

/* Force Plotly axis labels dark on light theme */
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .g-xtitle text,
.js-plotly-plot .plotly .g-ytitle text {{
    fill: #0f172a !important;
    font-size: 12px !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        rfm   = pd.read_csv(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'clustered_customers.csv')
        )
        with open(os.path.join(os.path.dirname(__file__), '..', 'models', 'cluster_stats.json')) as f:
            stats = json.load(f)
        return rfm, stats, True
    except Exception as e:
        return None, None, False


rfm, stats, data_loaded = load_data()


def pcfg(h=300, bgcolor=BG_CARD):
    return dict(
        paper_bgcolor=bgcolor, plot_bgcolor=BG_CARD2,
        font=dict(color=TEXT_PRI, family="Inter"), height=h,
        margin=dict(t=30, b=30, l=10, r=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER2,
                   color=TEXT_PRI, gridwidth=0.5,
                   tickfont=dict(color=TEXT_PRI, size=11),
                   title_font=dict(color=TEXT_HEAD, size=12)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER2,
                   color=TEXT_PRI, gridwidth=0.5,
                   tickfont=dict(color=TEXT_PRI, size=11),
                   title_font=dict(color=TEXT_HEAD, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_PRI, size=11)),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:1.2rem 0.8rem 0.8rem'>
        <p style='font-size:1rem;font-weight:700;color:#ffffff;margin:0;
           letter-spacing:-0.01em'>SegmentIQ</p>
        <p style='color:{TEXT_NAV};font-size:0.72rem;margin:0.2rem 0 0'>
            Customer Segmentation
        </p>
    </div>
    <hr style='border-color:#2d3548;margin:0 0 0.5rem'>
    <p style='font-size:0.65rem;font-weight:600;text-transform:uppercase;
       letter-spacing:0.12em;color:#4a5568;padding:0.5rem 0.8rem 0.3rem'>
       Explorer
    </p>
    """, unsafe_allow_html=True)

    page = st.radio("", options=[
        "Overview",
        "Segment Analysis",
        "Snake Plot",
        "Revenue Intelligence",
        "Customer Explorer",
        "Recommendations",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <hr style='border-color:#2d3548;margin:0.8rem 0'>
    <p style='font-size:0.65rem;font-weight:600;text-transform:uppercase;
       letter-spacing:0.12em;color:#4a5568;padding:0.3rem 0.8rem'>
       Model Info
    </p>
    """, unsafe_allow_html=True)

    if data_loaded:
        for k, v in [
            ("Algorithm",  "K-Means"),
            ("Clusters",   str(stats['n_clusters'])),
            ("Silhouette", str(stats['silhouette_score'])),
            ("Customers",  f"{len(rfm):,}"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                 padding:0.25rem 0.8rem'>
                <span style='color:{TEXT_NAV};font-size:0.74rem'>{k}</span>
                <span style='color:#e2e8f0;font-size:0.74rem;font-weight:500'>{v}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:{DANGER};font-size:0.75rem;padding:0 0.8rem'>"
                    f"No data loaded</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    n_customers = len(rfm) if data_loaded else 0
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">Customer Segmentation Dashboard</p>
        <p class="section-sub">RFM-based segmentation of {n_customers:,} customers
        using K-Means clustering</p>
    </div>""", unsafe_allow_html=True)

    if not data_loaded:
        st.warning("Run notebooks 01-04 first to generate the processed data.")
        st.stop()

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    champ  = rfm[rfm['Segment'] == 'Champions']
    lapsed = rfm[rfm['Segment'] == 'Lapsed Customers']

    for col, label, val, sub, color in [
        (c1, "Total Customers",    f"{len(rfm):,}",
         "across all segments", PRIMARY),
        (c2, "Champions",          f"{len(champ):,}",
         f"{len(champ)/len(rfm)*100:.1f}% of customers", SUCCESS),
        (c3, "Lapsed Customers",   f"{len(lapsed):,}",
         f"{len(lapsed)/len(rfm)*100:.1f}% of customers", DANGER),
        (c4, "Silhouette Score",   str(stats['silhouette_score']),
         "cluster separation quality", PRIMARY),
    ]:
        col.markdown(f"""
        <div class="metric-tile" style="border-top-color:{color}">
            <p class="metric-label">{label}</p>
            <p class="metric-value" style="color:{color}">{val}</p>
            <p class="metric-sub">{sub}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        seg_counts = rfm['Segment'].value_counts()
        fig = go.Figure(go.Pie(
            labels=seg_counts.index,
            values=seg_counts.values, hole=0.55,
            marker=dict(
                colors=[SEG_COLORS.get(s, PRIMARY) for s in seg_counts.index],
                line=dict(color='white', width=2)
            ),
            textfont=dict(color=TEXT_PRI)
        ))
        fig.add_annotation(
            text=f"{len(rfm):,}<br>customers",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=TEXT_HEAD)
        )
        cfg = pcfg(h=320, bgcolor=BG_CARD)
        cfg['title'] = dict(text="Customer Distribution",
                            font=dict(color=TEXT_HEAD, size=13))
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_rev = rfm.groupby('Segment')['Monetary'].sum()
        fig = go.Figure(go.Bar(
            x=seg_rev.index,
            y=seg_rev.values,
            marker=dict(
                color=[SEG_COLORS.get(s, PRIMARY) for s in seg_rev.index],
                opacity=0.85
            ),
            text=[f'£{v:,.0f}' for v in seg_rev.values],
            textposition='outside',
            textfont=dict(color=TEXT_PRI)
        ))
        cfg = pcfg(h=320, bgcolor=BG_CARD)
        cfg['title'] = dict(text="Total Revenue by Segment",
                            font=dict(color=TEXT_HEAD, size=13))
        cfg['yaxis']['title'] = 'Revenue (£)'
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    # RFM summary table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Segment Profiles</p>', unsafe_allow_html=True)
    profile = rfm.groupby('Segment').agg(
        Customers  = ('CustomerID', 'count'),
        Recency    = ('Recency',    'mean'),
        Frequency  = ('Frequency',  'mean'),
        Monetary   = ('Monetary',   'mean'),
    ).round(2)
    profile['Avg Revenue'] = profile['Monetary'].apply(lambda x: f'£{x:,.2f}')
    profile['Recency']     = profile['Recency'].apply(lambda x: f'{x:.0f} days')
    profile['Frequency']   = profile['Frequency'].apply(lambda x: f'{x:.1f} orders')
    st.dataframe(
        profile[['Customers','Recency','Frequency','Avg Revenue']],
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Segment Analysis":
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">Segment Analysis</p>
        <p class="section-sub">Deep dive into RFM distributions per segment</p>
    </div>""", unsafe_allow_html=True)

    seg_choice = st.selectbox("Select Segment", rfm['Segment'].unique().tolist())
    subset     = rfm[rfm['Segment'] == seg_choice]
    color      = SEG_COLORS.get(seg_choice, PRIMARY)
    badge_cls  = "badge-champion" if seg_choice == "Champions" else "badge-lapsed"

    c1,c2,c3,c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Customers",      f"{len(subset):,}",
         f"{len(subset)/len(rfm)*100:.1f}% of total"),
        (c2, "Avg Recency",    f"{subset['Recency'].mean():.0f} days",
         "since last purchase"),
        (c3, "Avg Frequency",  f"{subset['Frequency'].mean():.1f}",
         "orders per customer"),
        (c4, "Avg Monetary",   f"£{subset['Monetary'].mean():,.2f}",
         "revenue per customer"),
    ]:
        col.markdown(f"""
        <div class="metric-tile" style="border-top-color:{color}">
            <p class="metric-label">{label}</p>
            <p class="metric-value" style="color:{color}">{val}</p>
            <p class="metric-sub">{sub}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    for ax, metric, label in [
        (col1, 'Recency',   'Recency (days)'),
        (col2, 'Frequency', 'Frequency (orders)'),
        (col3, 'Monetary',  'Monetary (£)'),
    ]:
        fig = go.Figure(go.Histogram(
            x=subset[metric].clip(0, subset[metric].quantile(0.99)),
            nbinsx=30,
            marker=dict(color=color, opacity=0.8,
                        line=dict(color='white', width=0.5))
        ))
        cfg = pcfg(h=260, bgcolor=BG_CARD)
        cfg['title'] = dict(text=label, font=dict(color=TEXT_HEAD, size=12))
        cfg['xaxis']['title'] = label
        cfg['yaxis']['title'] = 'Count'
        fig.update_layout(**cfg)
        ax.plotly_chart(fig, use_container_width=True)

    # Scatter
    fig = go.Figure(go.Scatter(
        x=subset['Recency'].clip(0, subset['Recency'].quantile(0.99)),
        y=subset['Monetary'].clip(0, subset['Monetary'].quantile(0.99)),
        mode='markers',
        marker=dict(color=color, opacity=0.4, size=5)
    ))
    cfg = pcfg(h=300, bgcolor=BG_CARD)
    cfg['title'] = dict(text=f"{seg_choice} — Recency vs Monetary",
                        font=dict(color=TEXT_HEAD, size=13))
    cfg['xaxis']['title'] = 'Recency (days)'
    cfg['yaxis']['title'] = 'Monetary (£)'
    fig.update_layout(**cfg)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SNAKE PLOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Snake Plot":
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">Snake Plot</p>
        <p class="section-sub">Normalised RFM comparison across segments.
        Higher is better on all axes (Recency is inverted).</p>
    </div>""", unsafe_allow_html=True)

    from sklearn.preprocessing import MinMaxScaler
    scaler_mm = MinMaxScaler()
    rfm_norm  = rfm.copy()
    rfm_norm[['Recency','Frequency','Monetary']] = scaler_mm.fit_transform(
        rfm[['Recency','Frequency','Monetary']]
    )
    rfm_norm['Recency'] = 1 - rfm_norm['Recency']

    snake = rfm_norm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
    axes_labels = ['Recency (inverted)', 'Frequency', 'Monetary']

    fig = go.Figure()
    for segment in snake.index:
        color = SEG_COLORS.get(segment, PRIMARY)
        fig.add_trace(go.Scatter(
            x=axes_labels,
            y=snake.loc[segment].values,
            mode='lines+markers+text',
            name=segment,
            line=dict(color=color, width=3),
            marker=dict(color=color, size=12,
                        line=dict(color='white', width=2)),
            text=[f'{v:.2f}' for v in snake.loc[segment].values],
            textposition='top center',
            textfont=dict(color=color, size=11)
        ))

    cfg = pcfg(h=420, bgcolor=BG_CARD)
    cfg['title'] = dict(
        text="Snake Plot — Normalised RFM by Segment",
        font=dict(color=TEXT_HEAD, size=14)
    )
    cfg['yaxis']['title'] = 'Normalised Score (0-1)'
    cfg['yaxis']['range'] = [-0.05, 1.2]
    cfg['xaxis']['title'] = 'RFM Dimension'
    fig.update_layout(**cfg)
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">How to Read the Snake Plot</p>',
                unsafe_allow_html=True)
    for point, desc in [
        ("Recency (inverted)",
         "Higher = bought more recently. Champions score high here — they purchased within the last 51 days on average."),
        ("Frequency",
         "Higher = buys more often. Champions average 12.7 orders vs 2.1 for Lapsed Customers."),
        ("Monetary",
         "Higher = spends more. Champions generate £6,547 vs £614 for Lapsed — a 10x difference."),
    ]:
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin-bottom:10px;padding:0.7rem;
             background:{BG_CARD2};border-radius:8px;border:1px solid {BORDER}'>
            <div style='width:3px;background:{PRIMARY};border-radius:3px;
                 flex-shrink:0'></div>
            <div>
                <p style='font-size:0.82rem;font-weight:600;
                   color:{TEXT_HEAD};margin:0'>{point}</p>
                <p style='font-size:0.78rem;color:{TEXT_MUT};
                   margin:0.2rem 0 0'>{desc}</p>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REVENUE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Revenue Intelligence":
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">Revenue Intelligence</p>
        <p class="section-sub">Revenue concentration and customer value analysis</p>
    </div>""", unsafe_allow_html=True)

    summary = rfm.groupby('Segment').agg(
        Customers = ('CustomerID', 'count'),
        TotalRev  = ('Monetary',   'sum'),
        AvgRev    = ('Monetary',   'mean'),
    ).round(2)
    summary['Pct_Customers'] = summary['Customers'] / summary['Customers'].sum() * 100
    summary['Pct_Revenue']   = summary['TotalRev']  / summary['TotalRev'].sum()  * 100

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        x = list(summary.index)
        fig.add_trace(go.Bar(
            name='% Customers', x=x,
            y=summary['Pct_Customers'],
            marker_color=PRIMARY, opacity=0.7
        ))
        fig.add_trace(go.Bar(
            name='% Revenue', x=x,
            y=summary['Pct_Revenue'],
            marker_color=[SEG_COLORS.get(s, PRIMARY) for s in x],
            opacity=0.95
        ))
        cfg = pcfg(h=320, bgcolor=BG_CARD)
        cfg['barmode'] = 'group'
        cfg['title'] = dict(text="Customers vs Revenue Share (%)",
                            font=dict(color=TEXT_HEAD, size=13))
        cfg['yaxis']['title'] = 'Percentage (%)'
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=list(summary.index),
            y=summary['AvgRev'],
            marker=dict(
                color=[SEG_COLORS.get(s, PRIMARY) for s in summary.index],
                opacity=0.85
            ),
            text=[f'£{v:,.0f}' for v in summary['AvgRev']],
            textposition='outside',
            textfont=dict(color=TEXT_PRI)
        ))
        cfg = pcfg(h=320, bgcolor=BG_CARD)
        cfg['title'] = dict(text="Average Revenue per Customer",
                            font=dict(color=TEXT_HEAD, size=13))
        cfg['yaxis']['title'] = 'Avg Revenue (£)'
        fig.update_layout(**cfg)
        st.plotly_chart(fig, use_container_width=True)

    # Reactivation calculator
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Reactivation Revenue Calculator</p>',
                unsafe_allow_html=True)

    champ_avg  = summary.loc['Champions',        'AvgRev']
    lapsed_avg = summary.loc['Lapsed Customers', 'AvgRev']
    lapsed_n   = summary.loc['Lapsed Customers', 'Customers']

    pct = st.slider(
        "If this % of Lapsed Customers become Champions:",
        min_value=1, max_value=30, value=10, step=1
    )
    converted    = int(lapsed_n * pct / 100)
    rev_uplift   = converted * (champ_avg - lapsed_avg)

    c1,c2,c3 = st.columns(3)
    c1.markdown(f"""
    <div class="metric-tile" style="border-top-color:{WARNING}">
        <p class="metric-label">Customers Converted</p>
        <p class="metric-value" style="color:{WARNING}">{converted:,}</p>
        <p class="metric-sub">{pct}% of {int(lapsed_n):,} lapsed</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="metric-tile" style="border-top-color:{SUCCESS}">
        <p class="metric-label">Revenue Uplift</p>
        <p class="metric-value" style="color:{SUCCESS}">£{rev_uplift:,.0f}</p>
        <p class="metric-sub">incremental annual revenue</p>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="metric-tile" style="border-top-color:{PRIMARY}">
        <p class="metric-label">Revenue per Convert</p>
        <p class="metric-value" style="color:{PRIMARY}">£{champ_avg - lapsed_avg:,.0f}</p>
        <p class="metric-sub">avg uplift per customer</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Explorer":
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">Customer Explorer</p>
        <p class="section-sub">Look up individual customer segments and RFM profiles</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        customer_id = st.number_input(
            "Enter Customer ID", min_value=1,
            value=int(rfm['CustomerID'].iloc[0])
        )
        lookup = st.button("Look Up Customer")

    if lookup:
        row = rfm[rfm['CustomerID'] == customer_id]
        if len(row) == 0:
            st.warning(f"Customer {customer_id} not found in dataset.")
        else:
            row = row.iloc[0]
            segment = row['Segment']
            color   = SEG_COLORS.get(segment, PRIMARY)
            badge   = "badge-champion" if segment == "Champions" else "badge-lapsed"

            with col2:
                st.markdown(f"""
                <div class="card">
                    <div style='display:flex;justify-content:space-between;
                         align-items:center;margin-bottom:1rem'>
                        <p style='font-size:1rem;font-weight:700;
                           color:{TEXT_HEAD};margin:0'>
                            Customer #{int(customer_id)}
                        </p>
                        <span class="badge {badge}">{segment}</span>
                    </div>
                    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem'>
                        <div style='text-align:center;padding:0.8rem;
                             background:{BG_CARD2};border-radius:8px;
                             border:1px solid {BORDER}'>
                            <p style='color:{TEXT_MUT};font-size:0.7rem;
                               text-transform:uppercase;letter-spacing:0.1em;margin:0'>
                               Recency</p>
                            <p style='font-size:1.4rem;font-weight:700;
                               color:{color};margin:0.3rem 0'>{int(row['Recency'])}</p>
                            <p style='color:{TEXT_MUT};font-size:0.72rem;margin:0'>days ago</p>
                        </div>
                        <div style='text-align:center;padding:0.8rem;
                             background:{BG_CARD2};border-radius:8px;
                             border:1px solid {BORDER}'>
                            <p style='color:{TEXT_MUT};font-size:0.7rem;
                               text-transform:uppercase;letter-spacing:0.1em;margin:0'>
                               Frequency</p>
                            <p style='font-size:1.4rem;font-weight:700;
                               color:{color};margin:0.3rem 0'>{int(row['Frequency'])}</p>
                            <p style='color:{TEXT_MUT};font-size:0.72rem;margin:0'>orders</p>
                        </div>
                        <div style='text-align:center;padding:0.8rem;
                             background:{BG_CARD2};border-radius:8px;
                             border:1px solid {BORDER}'>
                            <p style='color:{TEXT_MUT};font-size:0.7rem;
                               text-transform:uppercase;letter-spacing:0.1em;margin:0'>
                               Monetary</p>
                            <p style='font-size:1.4rem;font-weight:700;
                               color:{color};margin:0.3rem 0'>
                               £{row['Monetary']:,.0f}</p>
                            <p style='color:{TEXT_MUT};font-size:0.72rem;margin:0'>total spend</p>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # Browse all customers
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Browse Customers</p>', unsafe_allow_html=True)
    seg_filter = st.selectbox("Filter by Segment",
                               ["All"] + rfm['Segment'].unique().tolist())
    display_df = rfm if seg_filter == "All" else rfm[rfm['Segment'] == seg_filter]
    st.dataframe(
        display_df[['CustomerID','Segment','Recency','Frequency','Monetary',
                    'RFM_Score']].sort_values('Monetary', ascending=False).head(50),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">Business Recommendations</p>
        <p class="section-sub">Actionable strategies for each customer segment</p>
    </div>""", unsafe_allow_html=True)

    champ  = rfm[rfm['Segment'] == 'Champions']
    lapsed = rfm[rfm['Segment'] == 'Lapsed Customers']
    champ_avg_rev  = champ['Monetary'].mean()
    lapsed_avg_rev = lapsed['Monetary'].mean()

    # Champions
    st.markdown(f"""
    <div class="rec-card rec-champion">
        <div style='display:flex;justify-content:space-between;
             align-items:center;margin-bottom:1rem'>
            <p style='font-size:1.1rem;font-weight:700;
               color:{SUCCESS};margin:0'>Champions</p>
            <span class="badge badge-champion">
                {len(champ):,} customers &nbsp;·&nbsp;
                {len(champ)/len(rfm)*100:.1f}% of base
            </span>
        </div>
        <p style='font-size:0.85rem;font-weight:600;
           color:{TEXT_HEAD};margin:0 0 0.8rem'>
           Strategy: RETAIN AND REWARD
        </p>
    </div>""", unsafe_allow_html=True)

    for action, detail in [
        ("VIP Loyalty Programme",
         "Enrol Champions in an exclusive tier with early product access and priority support."),
        ("Personalised Communications",
         "Send thank-you messages and personalised product recommendations based on purchase history."),
        ("Upsell Premium Lines",
         f"Champions spend £{champ_avg_rev:,.0f} on average — they have capacity for premium products."),
        ("Referral Programme",
         "Champions are your brand advocates — incentivise referrals with rewards."),
        ("Subscription / Auto-replenishment",
         "Set up repeat purchase subscriptions to lock in their loyalty and reduce churn risk."),
    ]:
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin-bottom:8px;padding:0.7rem 1rem;
             background:{BG_CARD};border-radius:8px;border:1px solid {BORDER}'>
            <div style='width:3px;background:{SUCCESS};border-radius:3px;
                 flex-shrink:0;margin-top:2px'></div>
            <div>
                <p style='font-size:0.83rem;font-weight:600;
                   color:{TEXT_HEAD};margin:0'>{action}</p>
                <p style='font-size:0.78rem;color:{TEXT_MUT};
                   margin:0.2rem 0 0'>{detail}</p>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Lapsed
    rev_opportunity = len(lapsed) * 0.10 * (champ_avg_rev - lapsed_avg_rev)
    st.markdown(f"""
    <div class="rec-card rec-lapsed">
        <div style='display:flex;justify-content:space-between;
             align-items:center;margin-bottom:1rem'>
            <p style='font-size:1.1rem;font-weight:700;
               color:{DANGER};margin:0'>Lapsed Customers</p>
            <span class="badge badge-lapsed">
                {len(lapsed):,} customers &nbsp;·&nbsp;
                {len(lapsed)/len(rfm)*100:.1f}% of base
            </span>
        </div>
        <p style='font-size:0.85rem;font-weight:600;
           color:{TEXT_HEAD};margin:0 0 0.3rem'>
           Strategy: REACTIVATE
        </p>
        <p style='font-size:0.78rem;color:{TEXT_MUT};margin:0 0 0.8rem'>
           Converting 10% to Champions adds £{rev_opportunity:,.0f} in revenue
        </p>
    </div>""", unsafe_allow_html=True)

    for action, detail in [
        ("Win-back Email Campaign",
         "Send a time-limited discount (e.g. 15% off) to re-engage customers inactive for 6+ months."),
        ("Exit Survey",
         "Ask why they stopped purchasing — product gaps, pricing, or service issues."),
        ("New Product Showcase",
         "Show them products added since their last purchase — they may not know what they are missing."),
        ("Free Shipping Offer",
         "Remove friction from the next purchase with a free shipping promotion."),
        ("Recency Sub-segmentation",
         "Customers inactive for 100-200 days need softer messaging than those inactive for 300+ days."),
    ]:
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin-bottom:8px;padding:0.7rem 1rem;
             background:{BG_CARD};border-radius:8px;border:1px solid {BORDER}'>
            <div style='width:3px;background:{DANGER};border-radius:3px;
                 flex-shrink:0;margin-top:2px'></div>
            <div>
                <p style='font-size:0.83rem;font-weight:600;
                   color:{TEXT_HEAD};margin:0'>{action}</p>
                <p style='font-size:0.78rem;color:{TEXT_MUT};
                   margin:0.2rem 0 0'>{detail}</p>
            </div>
        </div>""", unsafe_allow_html=True)

    # Budget allocation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Recommended Marketing Budget Allocation</p>',
                unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=['Champion Retention', 'Lapsed Reactivation'],
        y=[60, 40],
        marker=dict(color=[SUCCESS, DANGER], opacity=0.85),
        text=['60%', '40%'],
        textposition='outside',
        textfont=dict(color=TEXT_PRI, size=14)
    ))
    cfg = pcfg(h=260, bgcolor=BG_CARD)
    cfg['yaxis']['title'] = 'Budget Allocation (%)'
    cfg['yaxis']['range'] = [0, 80]
    fig.update_layout(**cfg)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
