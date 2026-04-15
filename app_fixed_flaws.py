import streamlit as st
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(page_title="NYC Subway AI", page_icon="🚇",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d0d0d; color: #e8e8e8; }
section[data-testid="stSidebar"] { background-color: #141414; border-right: 1px solid #2a2a2a; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 2rem !important; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: #888 !important; }
h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; }
.status-badge { display:inline-block; padding:6px 16px; border-radius:4px; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:600; letter-spacing:0.08em; }
.on-time { background:#003d1f; color:#00e676; border:1px solid #00e676; }
.early   { background:#001f3d; color:#00b0ff; border:1px solid #00b0ff; }
.minor   { background:#3d2600; color:#ffb300; border:1px solid #ffb300; }
.major   { background:#3d0000; color:#ff1744; border:1px solid #ff1744; }
.line-dot { display:inline-block; width:28px; height:28px; border-radius:50%; text-align:center; line-height:28px; font-weight:700; font-size:0.85rem; margin-right:6px; font-family:'IBM Plex Mono',monospace; }
hr { border-color: #2a2a2a !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════
LINE_COLORS = {
    "1": "#EE352E", "2": "#EE352E", "3": "#EE352E",
    "4": "#00933C", "5": "#00933C", "6": "#00933C",
    "7": "#B933AD",
    "A": "#2850AD", "C": "#2850AD", "E": "#2850AD",
    "B": "#FF6319", "D": "#FF6319", "F": "#FF6319", "M": "#FF6319",
    "G": "#6CBE45", "J": "#996633", "Z": "#996633",
    "L": "#A7A9AC",
    "N": "#FCCC0A", "Q": "#FCCC0A", "R": "#FCCC0A", "W": "#FCCC0A",
}
ALL_LINES = sorted(LINE_COLORS.keys())
LIGHT_TEXT_LINES = {"N", "Q", "R", "W", "L"}
N_TRIPS = 500
N_STOPS = 30
SEED    = 42

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ══════════════════════════════════════════
# FIX 1 — SIMULATION
# Original flaw: scheduled = actual - random(60,300)  → delay IS the random number
#                stop_sequence was random per row, not sequential per trip
# Fixed:
#   • Schedule is defined first; actual = scheduled + delay (correct direction)
#   • Delays can be negative (early arrivals)
#   • stop_sequence tiles 1→N_STOPS correctly within each trip
#   • Delay propagates: 70% of previous stop's delay carries to next stop
#     + crowding dwell penalty + random shock (realistic autocorrelation)
# ══════════════════════════════════════════
def _hour_weights():
    w = np.ones(24)
    w[6:22] = 3
    w[7:10] = 6
    w[16:20] = 6
    return w / w.sum()

@st.cache_data
def generate_data(n_trips=N_TRIPS, n_stops=N_STOPS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for trip_id in range(n_trips):
        line      = rng.choice(ALL_LINES)
        direction = rng.choice(["Uptown", "Downtown", "Crosstown"])
        is_weekend = int(rng.random() < 2/7)
        hour      = int(rng.choice(range(24), p=_hour_weights()))
        rush_hour = int((7 <= hour <= 9) or (17 <= hour <= 19))
        sched_base = hour * 3600 + int(rng.integers(0, 3600))

        # Origin delay: rush = avg 2 min late, off-peak = sometimes early
        prev_delay = rng.normal(120, 60) if rush_hour else rng.normal(10, 40)

        for stop in range(1, n_stops + 1):
            sched_arrival = sched_base + stop * 90   # fixed schedule
            passengers = int(np.clip(
                rng.normal(700, 150) if rush_hour else rng.normal(200, 100),
                10, 1000))

            # Delay propagation: carry-over + crowding + random shock
            carry_over    = 0.7 * prev_delay
            dwell_penalty = max(0, (passengers - 400) * 0.05)
            shock         = rng.normal(0, 20)
            delay         = float(np.clip(carry_over + dwell_penalty + shock, -120, 600))

            rows.append({
                "trip_id": trip_id, "line": line, "direction": direction,
                "hour": hour, "stop_sequence": stop, "passengers": passengers,
                "rush_hour": rush_hour, "is_weekend": is_weekend,
                "scheduled_arrival_sec": sched_arrival,
                "actual_arrival_sec":    sched_arrival + delay,
                "delay": delay, "prev_delay": prev_delay,
            })
            prev_delay = delay
    return pd.DataFrame(rows)

df = generate_data()

# ══════════════════════════════════════════
# FIX 2 — MODEL WITH REAL SIGNAL + EVAL
# Original flaw: target was noise; no train/test split; no MAE reported
# Fixed: features have genuine causal link to target; proper 80/20 split;
#        MAE and R² reported on held-out test set
# ══════════════════════════════════════════
@st.cache_resource
def train_model(_df):
    from sklearn.metrics import r2_score
    features = ["hour", "stop_sequence", "prev_delay",
                "rush_hour", "is_weekend", "passengers"]
    X = _df[features]
    y = _df["delay"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    mdl = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.08,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    mdl.fit(X_tr, y_tr)
    preds  = mdl.predict(X_te)
    mae_v  = mean_absolute_error(y_te, preds)
    r2_v   = r2_score(y_te, preds)
    return mdl, mae_v, r2_v, features

model, mae, r2, FEATURES = train_model(df)

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚇 Trip Parameters")
    st.markdown("---")
    selected_line = st.selectbox("Subway Line", ALL_LINES, index=ALL_LINES.index("A"))
    lc      = LINE_COLORS[selected_line]
    txt_col = "#000" if selected_line in LIGHT_TEXT_LINES else "#fff"
    st.markdown(
        f'<div style="margin:-8px 0 12px">'
        f'<span class="line-dot" style="background:{lc};color:{txt_col}">{selected_line}</span>'
        f'<span style="color:#aaa;font-size:0.85rem">line selected</span></div>',
        unsafe_allow_html=True)

    direction  = st.selectbox("Direction", ["Uptown", "Downtown", "Crosstown"])
    hour       = st.slider("Hour of Day", 0, 23, 8)
    stop_seq   = st.slider("Current Stop", 1, N_STOPS, 5)
    prev_delay = st.slider("Previous Stop Delay (sec)", -120, 600, 60,
                           help="Negative = train arrived early at previous stop")
    passengers = st.slider("Passenger Load", 10, 1000, 350)
    is_weekend = st.checkbox("Weekend service", value=False)

    st.markdown("---")
    st.markdown("**Scheduled Arrival at This Stop**")
    sched_time = st.time_input(
        "From the timetable",
        value=datetime.time(hour, 0),
        help="Predicted arrival = scheduled arrival + predicted delay")

    st.markdown("---")
    st.caption(f"MAE: **{mae:.1f} sec** | R²: **{r2:.2f}**\n\n"
               f"{len(df):,} rows · {len(FEATURES)} features")

# ══════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════
rush_hour = int((7 <= hour <= 9) or (17 <= hour <= 19))
inp_df    = pd.DataFrame([{
    "hour": hour, "stop_sequence": stop_seq, "prev_delay": prev_delay,
    "rush_hour": rush_hour, "is_weekend": int(is_weekend), "passengers": passengers
}])
prediction = float(np.clip(model.predict(inp_df)[0], -120, 600))

# FIX 3 — PREDICTED ARRIVAL
# Original flaw: predicted_arrival = now + delay  (wall clock, meaningless)
# Fixed: predicted_arrival = scheduled_arrival + predicted_delay
today             = datetime.date.today()
scheduled_dt      = datetime.datetime.combine(today, sched_time)
predicted_arr_dt  = scheduled_dt + datetime.timedelta(seconds=prediction)

if prediction < -30:
    status_label, status_cls = "RUNNING EARLY", "early"
elif prediction < 60:
    status_label, status_cls = "ON TIME", "on-time"
elif prediction < 180:
    status_label, status_cls = "MINOR DELAY", "minor"
else:
    status_label, status_cls = "MAJOR DELAY", "major"

# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown(
    '<h1 style="color:#e8e8e8;margin-bottom:0">NYC Subway AI'
    '<span style="font-size:1rem;color:#555;font-weight:400;margin-left:16px">delay predictor</span></h1>',
    unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.markdown(
        f'<span class="line-dot" style="background:{lc};color:{txt_col};'
        f'width:40px;height:40px;line-height:40px;font-size:1.1rem">'
        f'{selected_line}</span> <span style="font-size:1.1rem;color:#ccc">Line</span>',
        unsafe_allow_html=True)
with c2:
    st.markdown(f'<span style="color:#888;font-size:0.85rem">Direction</span><br>'
                f'<span style="font-size:1.1rem;color:#ccc">{direction}</span>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<span class="status-badge {status_cls}">{status_label}</span>',
                unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════
m1, m2, m3, m4, m5 = st.columns(5)
delay_sign = "+" if prediction >= 0 else ""
with m1:
    st.metric("Predicted Delay", f"{delay_sign}{round(prediction)} sec",
              delta=f"{round(prediction - prev_delay):+d} vs prev stop",
              delta_color="inverse")
with m2:
    st.metric("Scheduled Arrival", sched_time.strftime("%H:%M"))
with m3:
    st.metric("Predicted Arrival", predicted_arr_dt.strftime("%H:%M:%S"))
with m4:
    st.metric("Model MAE ±", f"{mae:.1f} sec")
with m5:
    st.metric("Model R²", f"{r2:.2f}",
              help="Fraction of delay variance explained. 1.0 = perfect.")
st.markdown("---")

# ══════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════
left, right = st.columns(2)

# Chart 1 — Stop-ahead delay forecast
with left:
    st.markdown("#### Delay Forecast — Remaining Stops")
    forecasts, curr = [], prediction
    for s in range(stop_seq, min(stop_seq + 15, N_STOPS + 1)):
        inp = pd.DataFrame([{"hour": hour, "stop_sequence": s, "prev_delay": curr,
                              "rush_hour": rush_hour, "is_weekend": int(is_weekend),
                              "passengers": passengers}])
        curr = float(np.clip(model.predict(inp)[0], -120, 600))
        sched_s  = scheduled_dt + datetime.timedelta(seconds=(s - stop_seq) * 90)
        actual_s = sched_s + datetime.timedelta(seconds=curr)
        forecasts.append({"Stop": s, "Delay (sec)": curr,
                          "Predicted": actual_s.strftime("%H:%M:%S")})
    fdf = pd.DataFrame(forecasts)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=fdf["Stop"], y=fdf["Delay (sec)"],
        mode="lines+markers",
        line=dict(color=lc, width=2.5), marker=dict(size=6, color=lc),
        fill="tozeroy", fillcolor=hex_to_rgba(lc, 0.15),
        hovertemplate="Stop %{x}<br>Delay: %{y:.0f}s<extra></extra>"))
    fig1.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#aaa", family="IBM Plex Mono"),
                       xaxis=dict(showgrid=False, color="#555", title="Stop #"),
                       yaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555",
                                  title="Delay (sec)", zeroline=True, zerolinecolor="#555"),
                       margin=dict(l=0, r=0, t=10, b=0), height=280, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2 — Average delay by hour
with right:
    st.markdown("#### Average Delay by Hour")
    heat = df.groupby("hour")["delay"].mean().reset_index()
    heat.columns = ["Hour", "Avg Delay (sec)"]
    fig2 = px.bar(heat, x="Hour", y="Avg Delay (sec)", color="Avg Delay (sec)",
                  color_continuous_scale=["#1a3a1a", "#00c853", "#ffab00", "#ff1744"])
    fig2.add_vline(x=hour, line_dash="dash", line_color="#fff", line_width=1.5,
                   annotation_text="Now", annotation_font_color="#fff")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#aaa", family="IBM Plex Mono"),
                       xaxis=dict(showgrid=False, color="#555"),
                       yaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555"),
                       coloraxis_showscale=False,
                       margin=dict(l=0, r=0, t=10, b=0), height=280)
    st.plotly_chart(fig2, use_container_width=True)

left2, right2 = st.columns(2)

# Chart 3 — Feature importance
with left2:
    st.markdown("#### Model — Feature Importance")
    fi = pd.DataFrame({
        "Feature":    ["Hour", "Stop Seq", "Prev Delay", "Rush Hour", "Weekend", "Passengers"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale=["#1a2a3a", "#0077ff"])
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#aaa", family="IBM Plex Mono"),
                       xaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555"),
                       yaxis=dict(showgrid=False, color="#555"),
                       coloraxis_showscale=False,
                       margin=dict(l=0, r=0, t=10, b=0), height=280)
    st.plotly_chart(fig3, use_container_width=True)

# Chart 4 — FIX 4: Delay distribution with proper binning
# Original flaw: value_counts() on floats → ~all unique → flat useless chart
# Fixed: pd.cut() into 30-second buckets, then count
with right2:
    st.markdown(f"#### Delay Distribution — All vs Line {selected_line}")
    bins   = list(range(-120, 630, 30))
    labels = [str(b) for b in bins[:-1]]

    def bin_delays(series):
        return pd.cut(series, bins=bins, labels=labels).value_counts().sort_index()

    all_binned  = bin_delays(df["delay"])
    line_binned = bin_delays(df[df["line"] == selected_line]["delay"])

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=all_binned.index.astype(str), y=all_binned.values,
                          name="All Lines", marker_color="#2a2a2a", opacity=0.85))
    fig4.add_trace(go.Bar(x=line_binned.index.astype(str), y=line_binned.values,
                          name=f"Line {selected_line}", marker_color=lc, opacity=0.85))
    pred_bucket = str(int((prediction // 30) * 30))
    fig4.add_vline(x=pred_bucket, line_dash="dash", line_color="#fff",
                   annotation_text=f"Pred: {prediction:.0f}s",
                   annotation_font_color="#fff", annotation_font_size=10)
    fig4.update_layout(barmode="overlay",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#aaa", family="IBM Plex Mono"),
                       xaxis=dict(showgrid=False, color="#555", title="Delay bucket (sec)",
                                  tickangle=45, nticks=15),
                       yaxis=dict(showgrid=True, gridcolor="#1f1f1f", color="#555"),
                       legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#aaa")),
                       margin=dict(l=0, r=0, t=10, b=0), height=280)
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════
# LIVE BOARD — FIX 5: model-driven, not random
# Original flaw: delays were rng.integers(0, 400) with no model involved
# Fixed: each board row is predicted by the actual trained model
# ══════════════════════════════════════════
st.markdown("---")
st.markdown("#### 🕐 Simulated Live Board")

@st.cache_data(ttl=30)
def get_live_board(_model, n=10):
    rng = np.random.default_rng(int(datetime.datetime.now().timestamp()) // 30)
    now = datetime.datetime.now()
    records = []
    for _ in range(n):
        ln   = rng.choice(ALL_LINES)
        hr   = int(rng.integers(6, 23))
        stop = int(rng.integers(1, N_STOPS))
        pax  = int(rng.integers(50, 1000))
        rh   = int((7 <= hr <= 9) or (17 <= hr <= 19))
        wknd = int(rng.random() < 2/7)
        pd_  = float(rng.normal(60, 80))
        dr   = rng.choice(["Uptown ↑", "Downtown ↓", "Crosstown →"])

        row   = pd.DataFrame([{"hour": hr, "stop_sequence": stop, "prev_delay": pd_,
                                "rush_hour": rh, "is_weekend": wknd, "passengers": pax}])
        delay = float(np.clip(_model.predict(row)[0], -120, 600))
        sched  = now + datetime.timedelta(minutes=int(rng.integers(1, 20)))
        actual = sched + datetime.timedelta(seconds=delay)

        if delay < -30:   status = "🔵 Early"
        elif delay < 60:  status = "🟢 On Time"
        elif delay < 180: status = "🟡 Minor Delay"
        else:             status = "🔴 Major Delay"

        records.append({
            "Line": ln, "Direction": dr, "Stop": f"{stop}/{N_STOPS}",
            "Scheduled": sched.strftime("%H:%M"),
            "Predicted Arrival": actual.strftime("%H:%M"),
            "Delay (sec)": round(delay), "Status": status
        })
    return pd.DataFrame(records)

board = get_live_board(model)
st.dataframe(board.sort_values("Delay (sec)", ascending=False),
             use_container_width=True, hide_index=True)

st.caption(
    f"Refreshes every 30s · MAE ≈ {mae:.1f}s · R² = {r2:.2f} · "
    f"{len(df):,} training rows · {datetime.datetime.now().strftime('%H:%M:%S')}"
)