import streamlit as st
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="NYC Subway AI",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
# CUSTOM CSS — dark transit board aesthetic
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0d0d;
    color: #e8e8e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

/* Metric cards */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem !important;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888 !important;
}

/* Section headers */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.02em;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.08em;
}
.on-time   { background:#003d1f; color:#00e676; border:1px solid #00e676; }
.minor     { background:#3d2600; color:#ffb300; border:1px solid #ffb300; }
.major     { background:#3d0000; color:#ff1744; border:1px solid #ff1744; }

/* Line dot */
.line-dot {
    display: inline-block;
    width: 28px; height: 28px;
    border-radius: 50%;
    text-align: center;
    line-height: 28px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Dividers */
hr { border-color: #2a2a2a !important; }

/* Info / success boxes */
.stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# CONSTANTS — NYC subway line colours
# ──────────────────────────────────────────
LINE_COLORS = {
    "1": "#EE352E", "2": "#EE352E", "3": "#EE352E",
    "4": "#00933C", "5": "#00933C", "6": "#00933C",
    "7": "#B933AD",
    "A": "#2850AD", "C": "#2850AD", "E": "#2850AD",
    "B": "#FF6319", "D": "#FF6319", "F": "#FF6319", "M": "#FF6319",
    "G": "#6CBE45",
    "J": "#996633", "Z": "#996633",
    "L": "#A7A9AC",
    "N": "#FCCC0A", "Q": "#FCCC0A", "R": "#FCCC0A", "W": "#FCCC0A",
}
ALL_LINES = sorted(LINE_COLORS.keys())

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a #RRGGBB hex string to an rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ──────────────────────────────────────────
# DATA GENERATION — realistic rush-hour pattern
# ──────────────────────────────────────────
@st.cache_data
def generate_data(n=15000, seed=42):
    rng = np.random.default_rng(seed)
    trip_ids  = np.repeat(np.arange(n // 30), 30)
    stops     = np.tile(np.arange(1, 31), n // 30)
    hours     = rng.integers(0, 24, size=len(trip_ids))

    # Rush-hour base delay (AM 7-9, PM 17-19)
    rush = ((hours >= 7) & (hours <= 9)) | ((hours >= 17) & (hours <= 19))
    base_delay = np.where(rush,
                          rng.normal(150, 60, size=len(hours)),
                          rng.normal(45,  30, size=len(hours)))
    base_delay = np.clip(base_delay, 0, 600)

    # Delay accumulates along stops
    stop_factor = stops * rng.uniform(0.5, 2.5, size=len(stops))
    delay = base_delay + stop_factor + rng.normal(0, 15, size=len(hours))
    delay = np.clip(delay, 0, 600)

    lines = rng.choice(ALL_LINES, size=len(trip_ids))
    directions = rng.choice(["Uptown", "Downtown", "Crosstown"], size=len(trip_ids))
    passengers = np.where(rush,
                          rng.integers(300, 1000, size=len(trip_ids)),
                          rng.integers(50, 400, size=len(trip_ids)))

    df = pd.DataFrame({
        "trip_id":    trip_ids,
        "line":       lines,
        "direction":  directions,
        "hour":       hours,
        "stop_sequence": stops,
        "passengers": passengers,
        "delay":      delay,
    })

    df = df.sort_values(["trip_id", "stop_sequence"])
    df["prev_delay"]  = df.groupby("trip_id")["delay"].shift(1)
    df["rush_hour"]   = rush.astype(int)
    df["is_weekend"]  = rng.choice([0, 1], size=len(df), p=[5/7, 2/7])
    df = df.dropna().reset_index(drop=True)
    return df

df = generate_data()

# ──────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────
@st.cache_resource
def train_model(df):
    features = ["hour", "stop_sequence", "prev_delay", "rush_hour", "is_weekend", "passengers"]
    X = df[features]
    y = df["delay"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    return model, mae, features

model, mae, FEATURES = train_model(df)

# ──────────────────────────────────────────
# SIDEBAR — inputs
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚇 Trip Parameters")
    st.markdown("---")

    selected_line = st.selectbox("Subway Line", ALL_LINES, index=ALL_LINES.index("A"))
    lc = LINE_COLORS[selected_line]
    st.markdown(
        f'<div style="margin:-8px 0 12px;"><span class="line-dot" style="background:{lc};color:{"#000" if selected_line in ["N","Q","R","W","L"] else "#fff"}">'
        f'{selected_line}</span> <span style="color:#aaa;font-size:0.85rem;">line selected</span></div>',
        unsafe_allow_html=True
    )

    direction  = st.selectbox("Direction", ["Uptown", "Downtown", "Crosstown"])
    hour       = st.slider("Hour of Day", 0, 23, 8,
                           help="0 = midnight, 8 = morning rush, 17 = evening rush")
    stop_seq   = st.slider("Stop Sequence", 1, 30, 5,
                           help="How many stops into the trip")
    prev_delay = st.slider("Previous Stop Delay (sec)", 0, 600, 60)
    passengers = st.slider("Passenger Load", 50, 1000, 350)
    is_weekend = st.checkbox("Weekend service", value=False)

    st.markdown("---")
    st.caption(f"Model MAE: **{mae:.1f} sec**  \nFeatures: {len(FEATURES)}  \nTraining rows: {len(df):,}")

# ──────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────
rush_hour  = int((7 <= hour <= 9) or (17 <= hour <= 19))
input_data = pd.DataFrame([{
    "hour": hour, "stop_sequence": stop_seq, "prev_delay": prev_delay,
    "rush_hour": rush_hour, "is_weekend": int(is_weekend), "passengers": passengers
}])
prediction = float(model.predict(input_data)[0])
prediction = max(0, prediction)

now = datetime.datetime.now()
scheduled = now.strftime("%H:%M:%S")
predicted_arrival = now + datetime.timedelta(seconds=prediction)

if prediction < 60:
    status_label, status_cls = "ON TIME", "on-time"
elif prediction < 180:
    status_label, status_cls = "MINOR DELAY", "minor"
else:
    status_label, status_cls = "MAJOR DELAY", "major"

# ──────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────
st.markdown(
    f'<h1 style="color:#e8e8e8;margin-bottom:0">NYC Subway AI'
    f'<span style="font-size:1rem;color:#555;font-weight:400;margin-left:16px">delay predictor</span></h1>',
    unsafe_allow_html=True
)
col_line, col_dir, col_status = st.columns([1, 1, 2])
with col_line:
    st.markdown(
        f'<span class="line-dot" style="background:{lc};color:{"#000" if selected_line in ["N","Q","R","W","L"] else "#fff"};width:40px;height:40px;line-height:40px;font-size:1.1rem">'
        f'{selected_line}</span> <span style="font-size:1.1rem;color:#ccc">Line</span>',
        unsafe_allow_html=True)
with col_dir:
    st.markdown(f'<span style="color:#888;font-size:0.85rem">Direction</span><br>'
                f'<span style="font-size:1.1rem;color:#ccc">{direction}</span>', unsafe_allow_html=True)
with col_status:
    st.markdown(f'<span class="status-badge {status_cls}">{status_label}</span>', unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────
# METRICS ROW
# ──────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Predicted Delay", f"{round(prediction)} sec",
              delta=f"{round(prediction - prev_delay):+d} vs prev stop",
              delta_color="inverse")
with m2:
    st.metric("Predicted Arrival", predicted_arrival.strftime("%H:%M:%S"))
with m3:
    st.metric("Stop", f"{stop_seq} / 30")
with m4:
    st.metric("Confidence ±", f"{mae:.0f} sec", help="Model mean absolute error on hold-out set")

st.markdown("---")

# ──────────────────────────────────────────
# CHARTS — two columns
# ──────────────────────────────────────────
left, right = st.columns(2)

# 1. Delay forecast along remaining stops
with left:
    st.markdown("#### Delay Forecast — Remaining Stops")
    future_stops = range(stop_seq, min(stop_seq + 15, 31))
    forecasts = []
    curr_delay = prediction
    for s in future_stops:
        inp = pd.DataFrame([{
            "hour": hour, "stop_sequence": s, "prev_delay": curr_delay,
            "rush_hour": rush_hour, "is_weekend": int(is_weekend), "passengers": passengers
        }])
        curr_delay = max(0, float(model.predict(inp)[0]))
        forecasts.append({"Stop": s, "Delay (sec)": curr_delay})

    forecast_df = pd.DataFrame(forecasts)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=forecast_df["Stop"], y=forecast_df["Delay (sec)"],
        mode="lines+markers",
        line=dict(color=lc, width=2.5),
        marker=dict(size=6, color=lc),
        fill="tozeroy",
        fillcolor=hex_to_rgba(lc, 0.15),
        name="Predicted delay"
    ))
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", family="IBM Plex Mono"),
        xaxis=dict(showgrid=False, color="#555", title="Stop #"),
        yaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555", title="Delay (sec)"),
        margin=dict(l=0, r=0, t=10, b=0), height=280,
        showlegend=False,
    )
    st.plotly_chart(fig1, use_container_width=True)

# 2. Avg delay by hour heatmap (for selected line subset)
with right:
    st.markdown("#### Average Delay by Hour (All Lines)")
    heat = df.groupby("hour")["delay"].mean().reset_index()
    heat.columns = ["Hour", "Avg Delay (sec)"]
    fig2 = px.bar(heat, x="Hour", y="Avg Delay (sec)",
                  color="Avg Delay (sec)",
                  color_continuous_scale=["#1a3a1a", "#00c853", "#ffab00", "#ff1744"])
    fig2.add_vline(x=hour, line_dash="dash", line_color="#fff", line_width=1.5,
                   annotation_text="Now", annotation_font_color="#fff")
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", family="IBM Plex Mono"),
        xaxis=dict(showgrid=False, color="#555"),
        yaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555"),
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────
# SECOND ROW of charts
# ──────────────────────────────────────────
left2, right2 = st.columns(2)

# 3. Feature importance
with left2:
    st.markdown("#### Model — Feature Importance")
    fi = pd.DataFrame({
        "Feature": ["Hour", "Stop Seq", "Prev Delay", "Rush Hour", "Weekend", "Passengers"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale=["#1a2a3a", "#0077ff"])
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", family="IBM Plex Mono"),
        xaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555"),
        yaxis=dict(showgrid=False, color="#555"),
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig3, use_container_width=True)

# 4. Delay distribution (all vs selected line)
with right2:
    st.markdown(f"#### Delay Distribution — All vs Line {selected_line}")
    line_df = df[df["line"] == selected_line]["delay"].sample(min(500, len(df[df["line"] == selected_line])))
    all_df  = df["delay"].sample(2000)
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=all_df, nbinsx=40, name="All Lines",
                                marker_color="#2a2a2a", opacity=0.8))
    fig4.add_trace(go.Histogram(x=line_df, nbinsx=40, name=f"Line {selected_line}",
                                marker_color=lc, opacity=0.85))
    fig4.add_vline(x=prediction, line_dash="dash", line_color="#fff",
                   annotation_text=f"Prediction: {prediction:.0f}s",
                   annotation_font_color="#fff", annotation_font_size=10)
    fig4.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", family="IBM Plex Mono"),
        xaxis=dict(showgrid=False, color="#555", title="Delay (sec)"),
        yaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#aaa")),
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────
# LIVE TRIPS TABLE
# ──────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🕐 Simulated Live Board")

@st.cache_data(ttl=30)
def get_live_board():
    rng = np.random.default_rng(int(datetime.datetime.now().timestamp()) // 30)
    lines   = rng.choice(ALL_LINES, 8)
    delays  = rng.integers(0, 400, 8)
    stops   = rng.integers(1, 30, 8)
    dirs    = rng.choice(["Uptown ↑", "Downtown ↓", "Crosstown →"], 8)
    statuses = []
    for d in delays:
        if d < 60:   statuses.append("🟢 On Time")
        elif d < 180: statuses.append("🟡 Minor Delay")
        else:         statuses.append("🔴 Major Delay")
    now = datetime.datetime.now()
    arrivals = [(now + datetime.timedelta(seconds=int(d))).strftime("%H:%M") for d in delays]
    return pd.DataFrame({
        "Line": lines, "Direction": dirs,
        "Stop": stops, "Delay (sec)": delays,
        "Arrival": arrivals, "Status": statuses
    })

board = get_live_board()
st.dataframe(
    board.sort_values("Delay (sec)", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Last refreshed: {now.strftime('%H:%M:%S')} · Model MAE ≈ {mae:.1f}s · {len(df):,} training samples")