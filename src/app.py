from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(page_title="AURA", page_icon="●", layout="wide")

DATA_DIR = Path("data")

# Helpers


def safe_read_csv(path: Path, dtype=None):
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path, dtype=dtype)


def period_to_dt(p):
    p = str(p)
    if len(p) >= 6:
        return pd.to_datetime(p[:4] + "-" + p[4:6] + "-01")
    return pd.NaT


def period_label(p: str) -> str:
    p = str(p)
    return f"{p[:4]}-{p[4:]}" if len(p) >= 6 else p


def fmt_sar(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.0f} SAR"


def clamp(v, a=0, b=100):
    try:
        return max(a, min(b, float(v)))
    except:
        return np.nan


def kpi_delta(curr, prev):
    if pd.isna(curr) or pd.isna(prev) or prev == 0:
        return None
    return (curr - prev) / abs(prev)


THEMES = {
    "Dark": {
        "bg": "#071019",
        "panel": "rgba(255,255,255,0.06)",
        "panel2": "rgba(255,255,255,0.04)",
        "border": "rgba(148,163,184,0.18)",
        "text": "#E8EEF7",
        "muted": "rgba(232,238,247,0.72)",
        "grid": "rgba(148,163,184,0.18)",
        "brand": "#00B140",
        "brand2": "#00A7B5",
        "danger": "#FF4D4F",
        "warn": "#FFB020",
        "ok": "#22C55E",
        "shadow": "rgba(0,0,0,0.35)",
    },
}


def apply_theme(theme_name: str):
    t = THEMES[theme_name]
    st.markdown(
        f"""
<style>
:root {{
  --bg: {t["bg"]};
  --panel: {t["panel"]};
  --panel2: {t["panel2"]};
  --border: {t["border"]};
  --text: {t["text"]};
  --muted: {t["muted"]};
  --grid: {t["grid"]};
  --brand: {t["brand"]};
  --brand2: {t["brand2"]};
  --danger: {t["danger"]};
  --warn: {t["warn"]};
  --ok: {t["ok"]};
  --shadow: {t["shadow"]};
}}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 700px at 15% 8%, rgba(0,177,64,0.10), transparent 55%),
              radial-gradient(1000px 600px at 85% 10%, rgba(0,167,181,0.10), transparent 55%),
              var(--bg);
}}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(0,177,64,0.08), transparent 22%),
              linear-gradient(180deg, rgba(0,167,181,0.06), transparent 30%),
              var(--bg);
  border-right: 1px solid var(--border);
}}
h1,h2,h3,h4,h5,h6,p,div,span,label {{
  color: var(--text) !important;
}}
.small-muted {{
  color: var(--muted) !important;
  font-size: 0.92rem;
}}
.hr {{
  border-bottom: 1px solid var(--border);
  margin: 0.9rem 0 1.0rem 0;
}}
/* Premium cards */
.card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 14px 34px var(--shadow);
}}
.card-glow {{
  position: relative;
  overflow: hidden;
}}
.card-glow:before {{
  content:"";
  position:absolute; inset:-2px;
  background: radial-gradient(420px 160px at 20% 0%, rgba(0,177,64,0.18), transparent 65%),
              radial-gradient(420px 160px at 80% 0%, rgba(0,167,181,0.18), transparent 65%);
  pointer-events:none;
}}
.card-title {{
  font-size: 0.86rem;
  color: var(--muted) !important;
  margin-bottom: 6px;
}}
.card-value {{
  font-size: 1.55rem;
  font-weight: 900;
  line-height: 1.1;
  letter-spacing: -0.02em;
}}
.card-sub {{
  font-size: 0.92rem;
  color: var(--muted) !important;
  margin-top: 6px;
}}
.pill {{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 6px 10px;
  border-radius: 999px;

  font-weight: 800;
  font-size: 0.84rem;
}}
.dot {{
  width:10px; height:10px;
  border-radius: 999px;
  background: var(--brand);
  box-shadow: 0 0 0 4px rgba(0,177,64,0.16);
}}
.pill-high .dot {{ background: var(--danger); box-shadow: 0 0 0 4px rgba(255,77,79,0.16); }}
.pill-med .dot  {{ background: var(--warn); box-shadow: 0 0 0 4px rgba(255,176,32,0.16); }}
.pill-low .dot  {{ background: var(--ok); box-shadow: 0 0 0 4px rgba(34,197,94,0.16); }}

/* Buttons */
.stButton>button {{
  border-radius: 14px;
  border: 1px solid var(--border);
  padding: 0.6rem 0.9rem;
}}
/* Dataframe minimal */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def risk_pill(band: str):
    if band == "High Risk":
        return "<span class='pill pill-high'><span class='dot'></span>High Risk</span>"
    if band == "Medium Risk":
        return "<span class='pill pill-med'><span class='dot'></span>Medium Risk</span>"
    return "<span class='pill pill-low'><span class='dot'></span>Low Risk</span>"


@st.cache_data
def load_all():
    scores = safe_read_csv(
        DATA_DIR / "company_monthly_scores.csv",
        dtype={"company_id": str, "period": str},
    )
    preds = safe_read_csv(
        DATA_DIR / "company_risk_predictions.csv",
        dtype={"company_id": str, "period": str},
    )
    kpis = safe_read_csv(
        DATA_DIR / "company_monthly_kpis.csv", dtype={"company_id": str, "period": str}
    )
    seg = safe_read_csv(
        DATA_DIR / "avoidable_er_segments.csv", dtype={"company_id": str, "period": str}
    )

    for df in [scores, preds, kpis, seg]:
        df["period"] = df["period"].astype(str)
        df["period_dt"] = df["period"].apply(period_to_dt)
    return scores, preds, kpis, seg


scores, preds, kpis, seg = load_all()


st.sidebar.markdown("## Settings")
theme = st.sidebar.radio("Theme", ["Dark"], horizontal=True)
apply_theme(theme)
T = THEMES[theme]

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)

view = st.sidebar.selectbox(
    "Story View",
    [
        "Executive Overview",
        "Company Deep Dive",
        "Drivers (Where & Who)",
        "Portfolio Monitor",
        "Prediction Center",
    ],
)

companies = sorted(scores["company_id"].unique())
periods = sorted(scores["period"].unique())

company = st.sidebar.selectbox("Company", companies)
period = st.sidebar.selectbox("Month", periods, index=len(periods) - 1)

st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='small-muted'></div>",
    unsafe_allow_html=True,
)

# Snapshot for selected company/month
row_s = scores[(scores.company_id == company) & (scores.period == period)]
row_k = kpis[(kpis.company_id == company) & (kpis.period == period)]
row_p = preds[(preds.company_id == company) & (preds.period == period)]


def get_prev(df, col):
    d = df[df.company_id == company].sort_values("period")
    idx = d.index[d["period"] == period]
    if len(idx) == 0:
        return None, None
    i = list(d.index).index(idx[0])
    if i == 0:
        return float(d.loc[idx[0], col]), None
    curr = float(d.loc[idx[0], col])
    prev = float(d.iloc[i - 1][col])
    return curr, prev


IVI = float(row_s.IVI.iloc[0]) if not row_s.empty else np.nan
U = float(row_s.U_score.iloc[0]) if not row_s.empty else np.nan
band = row_s.risk_band.iloc[0] if not row_s.empty else "—"

avoid_rate = float(row_k.avoidable_er_rate.iloc[0]) if not row_k.empty else np.nan
avoid_cost = float(row_k.avoidable_er_cost.iloc[0]) if not row_k.empty else np.nan
er_per100 = (
    float(row_k.er_visits_per_100_members.iloc[0]) if not row_k.empty else np.nan
)
members_ct = (
    int(row_k.member_count.iloc[0])
    if not row_k.empty
    else (int(row_s.member_count.iloc[0]) if not row_s.empty else 0)
)

risk_prob = None if row_p.empty else float(row_p.risk_probability_next_month.iloc[0])

# deltas
ivi_curr, ivi_prev = get_prev(scores, "IVI")
u_curr, u_prev = get_prev(scores, "U_score")
er_curr, er_prev = get_prev(kpis, "er_visits_per_100_members")
avc_curr, avc_prev = get_prev(kpis, "avoidable_er_cost")

# Header

st.markdown("# AURA Early Warning")
st.markdown(
    f"<div class='small-muted'>A visual, decision ready view of <b>avoidable ER utilization</b> impact and next month risk. "
    f"Company <b>{company}</b> • Month <b>{period_label(period)}</b>.</div>",
    unsafe_allow_html=True,
)

# ---------------------------
# KPI Row (fancy)
# ---------------------------
c1, c2, c3, c4, c5 = st.columns([1.35, 1, 1, 1.2, 1.25], gap="large")


def delta_text(d):
    if d is None or pd.isna(d):
        return "—"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f}% vs prev"


with c1:
    st.markdown(
        f"<div class='card card-glow'>"
        f"<div class='card-title'>Company</div>"
        f"<div class='card-value'>{company}</div>"
        f"<div class='card-sub'>Members: {members_ct:,} • Month: {period_label(period)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"<div class='card'>"
        f"<div class='card-title'>IVI</div>"
        f"<div class='card-value'>{clamp(IVI):.1f}</div>"
        f"{risk_pill(band)}"
        f"<div class='card-sub'>{delta_text(kpi_delta(ivi_curr, ivi_prev))}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"<div class='card'>"
        f"<div class='card-title'>U-Score</div>"
        f"<div class='card-value'>{clamp(U):.1f}</div>"
        f"<div class='card-sub'>{delta_text(kpi_delta(u_curr, u_prev))}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c4:
    rp = "—" if risk_prob is None else f"{risk_prob * 100:.1f}%"
    st.markdown(
        f"<div class='card'>"
        f"<div class='card-title'>Next-month High Risk</div>"
        f"<div class='card-value'>{rp}</div>"
        f"<div class='card-sub'>Probability (model)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"<div class='card'>"
        f"<div class='card-title'>Avoidable ER Cost</div>"
        f"<div class='card-value'>{fmt_sar(avoid_cost)}</div>"
        f"<div class='card-sub'>ER/100: {er_per100:.2f} • {delta_text(kpi_delta(avc_curr, avc_prev))}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ---------------------------
# Plotly style helper
# ---------------------------
def style_plot(fig, height=380, y_range=None):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=T["text"]),
        title_font=dict(size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=None, gridcolor=T["grid"], zerolinecolor=T["grid"])
    fig.update_yaxes(title=None, gridcolor=T["grid"], zerolinecolor=T["grid"])
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def donut(value, title):
    # value expected 0..1
    v = 0 if value is None or pd.isna(value) else float(value)
    v = max(0, min(1, v))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Avoidable ER", "Other ER"],
                values=[v, 1 - v],
                hole=0.72,
                sort=False,
                textinfo="none",
                marker=dict(colors=[T["brand"], "rgba(148,163,184,0.25)"]),
            )
        ]
    )
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(
                text=f"{v * 100:.0f}%",
                x=0.5,
                y=0.52,
                font=dict(size=28, color=T["text"]),
                showarrow=False,
            ),
            dict(
                text=title,
                x=0.5,
                y=0.22,
                font=dict(size=12, color=T["muted"]),
                showarrow=False,
            ),
        ],
    )
    return fig


# ===========================
# VIEW 1 — Executive Overview
# ===========================
if view == "Executive Overview":
    left, right = st.columns([1.7, 1], gap="large")

    trend = scores[scores.company_id == company].sort_values("period").copy()
    trend["month"] = trend["period"].apply(period_label)

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.area(trend, x="month", y="IVI", title="IVI Trend (12 months)")
        fig.update_traces(line=dict(width=3), opacity=0.35)
        fig.update_traces(marker=dict(size=7))
        fig = style_plot(fig, height=420, y_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Risk & Utilization Snapshot")
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            st.plotly_chart(
                donut(avoid_rate if not pd.isna(avoid_rate) else 0, "Avoidable share"),
                use_container_width=True,
            )
        with d2:
            # Risk gauge-like bar
            rp = 0 if risk_prob is None else float(risk_prob)
            rp = max(0, min(1, rp))
            g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=rp * 100,
                    number={"suffix": "%", "font": {"size": 30}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": T["brand2"]},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0, 55], "color": "rgba(34,197,94,0.15)"},
                            {"range": [55, 75], "color": "rgba(255,176,32,0.15)"},
                            {"range": [75, 100], "color": "rgba(255,77,79,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": T["danger"], "width": 3},
                            "thickness": 0.75,
                            "value": 75,
                        },
                    },
                    title={"text": "Next-month High Risk probability"},
                )
            )
            g.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=T["text"]),
            )
            st.plotly_chart(g, use_container_width=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### Top driver (this month)")
        seg_f = seg[(seg.company_id == company) & (seg.period == period)].copy()
        if seg_f.empty:
            st.markdown(
                "<div class='small-muted'>No avoidable ER records for this month.</div>",
                unsafe_allow_html=True,
            )
        else:
            dim = st.selectbox(
                "Explain via",
                [
                    "PROVIDER_TOWN",
                    "PROVIDER_REGION",
                    "PROVIDER_NETWORK",
                    "PROVIDER_PRACTICE",
                    "PROV_NAME",
                ],
            )
            top = (
                seg_f.groupby(dim)["avoidable_er_cost"]
                .sum()
                .reset_index()
                .sort_values("avoidable_er_cost", ascending=False)
                .head(8)
            )
            fig2 = px.bar(
                top,
                x=dim,
                y="avoidable_er_cost",
                title="Where avoidable ER cost concentrates",
            )
            fig2.update_traces(marker_color=T["brand"])
            fig2 = style_plot(fig2, height=340)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### Recommended actions")
        actions = []
        if pd.notna(avoid_rate) and avoid_rate >= 0.75:
            actions += [
                "Deploy **ER vs Clinic** decision guidance in member channels (1-page visual + app banner).",
                "Enable **digital steering**: show nearest clinics + instant booking before ER.",
            ]
        if pd.notna(er_per100) and er_per100 >= 6:
            actions.append(
                "Add **tele-triage nudges** during peak hours (symptom checker / telemedicine)."
            )
        if risk_prob is not None and risk_prob >= 0.70:
            actions.append(
                "Trigger **Account Manager alert**: utilization review + targeted interventions."
            )
        if not actions:
            actions = ["Continue monitoring — no urgent intervention needed."]
        for i, a in enumerate(actions, 1):
            st.markdown(f"**{i}.** {a}")

        st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# VIEW 2 — Company Deep Dive
# ===========================
elif view == "Company Deep Dive":
    st.markdown("### Company Deep Dive")
    tabs = st.tabs(["Utilization", "Costs", "ER Mix"])

    t = kpis[kpis.company_id == company].sort_values("period").copy()
    t["month"] = t["period"].apply(period_label)

    with tabs[0]:
        a, b = st.columns([1.2, 1], gap="large")
        with a:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig = px.line(
                t,
                x="month",
                y=["er_visits_per_100_members"],
                title="ER visits per 100 members",
            )
            fig.update_traces(line=dict(width=3))
            fig = style_plot(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig = px.line(
                t, x="month", y="avoidable_er_rate", title="Avoidable ER rate"
            )
            fig.update_traces(line=dict(width=3))
            fig = style_plot(fig, height=420, y_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        a, b = st.columns([1.2, 1], gap="large")
        with a:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig = px.area(t, x="month", y="total_cost", title="Total cost trend")
            fig.update_traces(opacity=0.35, line=dict(width=3))
            fig = style_plot(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig = px.area(
                t, x="month", y="avoidable_er_cost", title="Avoidable ER cost trend"
            )
            fig.update_traces(opacity=0.35, line=dict(width=3))
            fig.update_traces(marker=dict(size=6))
            fig = style_plot(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### ER composition (avoidable vs admitted)")
        # Create a stacked bar for the month breakdown if possible using kpis totals
        # We only have total_er_visits and avoidable_er_visits; admitted ER = total - avoidable (approx)
        tmp = t.copy()
        tmp["admitted_er_visits"] = (
            tmp["total_er_visits"] - tmp["avoidable_er_visits"]
        ).clip(lower=0)
        melt = tmp.melt(
            id_vars=["month"],
            value_vars=["avoidable_er_visits", "admitted_er_visits"],
            var_name="type",
            value_name="visits",
        )
        fig = px.bar(
            melt,
            x="month",
            y="visits",
            color="type",
            title="ER visits mix (per month)",
            barmode="stack",
        )
        fig.update_layout(coloraxis_showscale=False)
        fig = style_plot(fig, height=440)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# VIEW 3 — Drivers (Where & Who)
# ===========================
elif view == "Drivers (Where & Who)":
    st.markdown("### Drivers — Where & Who (selected month)")
    seg_f = seg[(seg.company_id == company) & (seg.period == period)].copy()

    if seg_f.empty:
        st.info("No avoidable ER records for this month.")
    else:
        col1, col2 = st.columns([1.05, 1], gap="large")

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            dim = st.selectbox(
                "Dimension",
                [
                    "PROVIDER_REGION",
                    "PROVIDER_TOWN",
                    "PROVIDER_NETWORK",
                    "PROVIDER_PRACTICE",
                ],
                index=0,
            )
            agg = (
                seg_f.groupby(dim)
                .agg(
                    cost=("avoidable_er_cost", "sum"),
                    visits=("avoidable_er_visits", "sum"),
                )
                .reset_index()
                .sort_values("cost", ascending=False)
                .head(12)
            )

            fig = px.bar(agg, x=dim, y="cost", title=f"Avoidable ER cost by {dim}")
            fig.update_traces(marker_color=T["brand"])
            fig = style_plot(fig, height=430)
            st.plotly_chart(fig, use_container_width=True)

            # small bubble chart for cost vs visits
            figb = px.scatter(
                agg,
                x="visits",
                y="cost",
                size="cost",
                color=dim,
                title="Cost vs Visits",
            )
            figb = style_plot(figb, height=320)
            st.plotly_chart(figb, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            topn = st.slider("Top providers", 5, 25, 10)
            prov = (
                seg_f.groupby("PROV_NAME")
                .agg(
                    cost=("avoidable_er_cost", "sum"),
                    visits=("avoidable_er_visits", "sum"),
                )
                .reset_index()
                .sort_values("cost", ascending=False)
                .head(topn)
            )

            fig2 = px.bar(
                prov,
                x="PROV_NAME",
                y="cost",
                title="Top facilities by avoidable ER cost",
            )
            fig2.update_traces(marker_color=T["brand2"])
            fig2 = style_plot(fig2, height=500)
            st.plotly_chart(fig2, use_container_width=True)

            # only a tiny table for copy/paste (optional)
            with st.expander("Show table"):
                show = prov.copy()
                show["cost"] = show["cost"].round(0).astype(int)
                st.dataframe(show, use_container_width=True, hide_index=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# VIEW 4 — Portfolio Monitor
# ===========================
elif view == "Portfolio Monitor":
    st.markdown("### Portfolio Monitor (selected month)")
    col1, col2 = st.columns([1.2, 1], gap="large")

    # Risk distribution donut
    snap = scores[scores.period == period].copy()
    counts = (
        snap["risk_band"]
        .value_counts()
        .reindex(["High Risk", "Medium Risk", "Low Risk"])
        .fillna(0)
    )

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts.index.tolist(),
                    values=counts.values.tolist(),
                    hole=0.62,
                    sort=False,
                    marker=dict(colors=[T["danger"], T["warn"], T["ok"]]),
                    textinfo="label+percent",
                )
            ]
        )
        fig.update_layout(
            title="Risk bands distribution (companies)",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=T["text"]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Top 15 by lowest IVI (most risky)
        top = snap.sort_values("IVI").head(15)
        fig2 = px.bar(
            top, x="company_id", y="IVI", title="Lowest IVI companies (priority list)"
        )
        fig2.update_traces(marker_color=T["danger"])
        fig2 = style_plot(fig2, height=420, y_range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Region + Network portfolio drivers using segmentation (all companies)
    seg_m = seg[seg.period == period].copy()
    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        rg = (
            seg_m.groupby("PROVIDER_REGION")["avoidable_er_cost"]
            .sum()
            .reset_index()
            .sort_values("avoidable_er_cost", ascending=False)
            .head(12)
        )
        fig = px.bar(
            rg,
            x="PROVIDER_REGION",
            y="avoidable_er_cost",
            title="Avoidable ER cost by region",
        )
        fig.update_traces(marker_color=T["brand"])
        fig = style_plot(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        nw = (
            seg_m.groupby("PROVIDER_NETWORK")["avoidable_er_cost"]
            .sum()
            .reset_index()
            .sort_values("avoidable_er_cost", ascending=False)
            .head(12)
        )
        fig = px.bar(
            nw,
            x="PROVIDER_NETWORK",
            y="avoidable_er_cost",
            title="Avoidable ER cost by network",
        )
        fig.update_traces(marker_color=T["brand2"])
        fig = style_plot(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# VIEW 5 — Prediction Center
# ===========================
else:
    st.markdown("### Prediction Center (Next-month High Risk)")

    col1, col2 = st.columns([1.25, 1], gap="large")

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        pm = preds[preds.period == period].copy()
        if pm.empty:
            st.info("No prediction rows for this month.")
        else:
            pm["risk_probability_next_month"] = pm[
                "risk_probability_next_month"
            ].astype(float)
            top = pm.sort_values("risk_probability_next_month", ascending=False).head(
                20
            )

            fig = px.bar(
                top,
                x="company_id",
                y="risk_probability_next_month",
                title="Top 20 companies by predicted risk",
            )
            fig.update_traces(marker_color=T["danger"])
            fig = style_plot(fig, height=460, y_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Selected company probability trend")
        pc = preds[preds.company_id == company].sort_values("period").copy()
        pc["month"] = pc["period"].apply(period_label)
        if pc.empty:
            st.info("No prediction history for this company.")
        else:
            fig = px.line(
                pc,
                x="month",
                y="risk_probability_next_month",
                title="Risk probability over time",
                markers=True,
            )
            fig.update_traces(
                line=dict(width=3), marker=dict(size=7), marker_color=T["brand2"]
            )
            fig = style_plot(fig, height=360, y_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown("#### Explanation (drivers)")
            # show a simple explanation text based on current month values
            expl = []
            if pd.notna(avoid_rate) and avoid_rate >= 0.75:
                expl.append(
                    "Avoidable ER rate is high → indicates inappropriate site-of-care usage."
                )
            if pd.notna(er_per100) and er_per100 >= 6:
                expl.append(
                    "High ER intensity per 100 members → persistent utilization pressure."
                )
            if pd.notna(avoid_cost) and avoid_cost >= np.nanpercentile(
                kpis["avoidable_er_cost"], 75
            ):
                expl.append(
                    "Avoidable ER cost in the top quartile → material financial impact."
                )
            if not expl:
                expl.append(
                    "Risk seems driven by broader utilization patterns rather than a single spike."
                )
            for i, e in enumerate(expl, 1):
                st.markdown(f"**{i}.** {e}")

        st.markdown("</div>", unsafe_allow_html=True)

# Footer

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-muted'></div>",
    unsafe_allow_html=True,
)
