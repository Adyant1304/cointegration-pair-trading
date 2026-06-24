import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Are we a pair?", layout="wide")

st.title("Are we a pair?")

# 1. Define the tabs
tab1, tab2, tab3 = st.tabs(["Live Radar", "Historical Backtests", "Pair Explorer"])

# ─── TAB 1: LIVE RADAR ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Open Positions")
    try:
        positions_df = pd.read_json("pairs_trading_data/open_positions.json")
        st.dataframe(positions_df, width='stretch')
    except:
        st.warning("Positions data not found.")

    st.subheader("Signals")
    try:
        signals_df = pd.read_csv("pairs_trading_data/signals.csv")
        st.dataframe(signals_df, width='stretch')
    except:
        st.warning("Signals data not found.")

# ─── TAB 2: HISTORICAL BACKTESTS ──────────────────────────────────────────────
with tab2:
    st.subheader("Strategy Performance Metrics")
    try:
        backtest_df = pd.read_csv("pairs_trading_data/backtest_summary.csv")
        st.dataframe(backtest_df, width='stretch')
    except:
        st.warning("Backtest data not found.")

# ─── TAB 3: PAIR EXPLORER ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Pair Explorer")
    try:
        signals_df = pd.read_csv("pairs_trading_data/signals.csv")
        pair_list = signals_df['pair'].unique().tolist()
        
        selected_pair = st.selectbox("Select Pair", pair_list)
        
        # --- PLOTTING LOGIC ---
        st.write(f"Displaying data for: {selected_pair}")
        
        # This assumes your data has columns like 'date', 'spread', or 'z_score'
        # Adjust the filenames/columns based on how you save your pair data
        # For this example, let's assume you have a file per pair or a master file
        
        # Example plot:
        fig = go.Figure()
        # Replace these with your actual data source logic
        # fig.add_trace(go.Scatter(x=df['date'], y=df['z_score'], name="Z-Score"))
        
        st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.warning(f"Could not load data for Pair Explorer: {e}")