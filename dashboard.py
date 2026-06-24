import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── 1. CONFIG & STYLING ────────────────────────────────────────────────────────
st.set_page_config(page_title="Pairs Engine", layout="wide")

# Standardized path
DATA_DIR = "pairs_trading_data"

# ─── 2. DATA LOADERS ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f: return json.load(f)
    return None

@st.cache_data
def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path): return pd.read_csv(path)
    return pd.DataFrame()

# ─── 3. MAIN DASHBOARD ──────────────────────────────────────────────────────────
st.title("Market Intelligence")
tab1, tab2, tab3 = st.tabs(["Live Radar", "Historical Backtests", "Pair Explorer"])

# Load data inside the tabs to prevent errors if files are missing
with tab1:
    open_positions = load_json("open_positions.json")
    live_signals = load_json("live_signals.json")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Open Positions")
        if open_positions and "positions" in open_positions:
            st.dataframe(pd.DataFrame(open_positions["positions"]), use_container_width=True)
        else:
            st.info("No active positions.")
    with col2:
        st.subheader("Signals")
        if live_signals and "signals" in live_signals:
            st.dataframe(pd.DataFrame(live_signals["signals"]), use_container_width=True)

with tab2:
    backtest_summary = load_csv("backtest_summary.csv")
    if not backtest_summary.empty:
        st.dataframe(backtest_summary, use_container_width=True)
    else:
        st.info("Run 04_backtest.py to generate summary.")

with tab3:
    signals_df = load_csv("signals.csv")
    if not signals_df.empty:
        pair = st.selectbox("Select Pair", signals_df["pair"].unique())
        data = signals_df[signals_df["pair"] == pair]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["spread"], name="Spread"))
        fig.update_layout(template="plotly_dark")
        # Fixed the warning here by removing use_container_width
        st.plotly_chart(fig)
