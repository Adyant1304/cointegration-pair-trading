import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import json

st.set_page_config(page_title="Are we a pair?", layout="wide")

st.title("Are we a pair?")

# 1. Define Tabs
tab1, tab2, tab3 = st.tabs(["Live Radar", "Historical Backtests", "Pair Explorer"])

# ─── TAB 1: LIVE RADAR (SIDE-BY-SIDE) ─────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Open Positions")
        try:
            positions_df = pd.read_json("pairs_trading_data/open_positions.json")
            
            if 'positions' in positions_df.columns:
                parsed_records = []
                for val in positions_df['positions']:
                    try:
                        if isinstance(val, str):
                            parsed_records.append(json.loads(val))
                        elif isinstance(val, dict):
                            parsed_records.append(val)
                    except Exception:
                        pass
                
                expanded_df = pd.DataFrame(parsed_records)
                
                # --- LIVE PNL CALCULATION ENGINE ---
                if not expanded_df.empty and 'pair' in expanded_df.columns:
                    unique_tickers = set()
                    for p in expanded_df['pair']:
                        t1, t2 = p.split(' / ')
                        unique_tickers.add(f"{t1}.NS")
                        unique_tickers.add(f"{t2}.NS")
                    
                    if unique_tickers:
                        # Fetch latest prices for all open positions at once
                        live_prices = yf.download(list(unique_tickers), period="1d")["Close"].iloc[-1]
                        
                        for idx, row in expanded_df.iterrows():
                            t1, t2 = row['pair'].split(' / ')
                            try:
                                current_s1 = live_prices[f"{t1}.NS"]
                                current_s2 = live_prices[f"{t2}.NS"]
                                
                                # Update live prices
                                expanded_df.at[idx, 'live_s1'] = round(current_s1, 2)
                                expanded_df.at[idx, 'live_s2'] = round(current_s2, 2)
                                
                                # Calculate % PnL based on position type
                                if "SHORT" in str(row['position']).upper():
                                    pnl = ((row['entry_s1'] - current_s1) / row['entry_s1']) + ((current_s2 - row['entry_s2']) / row['entry_s2'])
                                else:
                                    pnl = ((current_s1 - row['entry_s1']) / row['entry_s1']) + ((row['entry_s2'] - current_s2) / row['entry_s2'])
                                
                                expanded_df.at[idx, 'open_pnl'] = round(pnl * 100, 2) # As percentage
                            except Exception:
                                expanded_df.at[idx, 'open_pnl'] = 0.0

                # Define columns to display
                target_pos_cols = ['pair', 'position', 'entry_s1', 'entry_s2', 'live_s1', 'live_s2', 'open_pnl']
                display_pos_cols = [c for c in target_pos_cols if c in expanded_df.columns]
                
                # --- APPLY COLOR STYLING ---
                def color_pnl(val):
                    try:
                        val = float(val)
                        if val > 0: return 'color: #00ff00' # Bright Green
                        elif val < 0: return 'color: #ff4b4b' # Red
                        return 'color: gray'
                    except:
                        return ''

                if display_pos_cols:
                    # Apply styling and percentage formatting to the dataframe
                    styled_df = expanded_df[display_pos_cols].style\
                        .map(color_pnl, subset=['open_pnl'])\
                        .format({'open_pnl': '{:+.2f}%'})
                    st.dataframe(styled_df, width='stretch')
                else:
                    st.dataframe(expanded_df, width='stretch')
                    
            else:
                st.dataframe(positions_df, width='stretch')
                
        except Exception as e:
            st.warning(f"Positions data not found. Error: {e}")

    with col2:
        st.subheader("Signals")
        try:
            signals_df = pd.read_csv("pairs_trading_data/signals.csv")
            
            def format_signal(val):
                val_str = str(val).upper()
                if "SHORT" in val_str: return "🔴 " + val_str
                elif "LONG" in val_str: return "🟢 " + val_str
                elif "EXIT" in val_str or "FLAT" in val_str: return "🟡 " + val_str
                elif "NO SIGNAL" in val_str or val_str == "NONE" or val_str == "0": return "⚪ NO SIGNAL"
                return val_str

            if 'signal' in signals_df.columns:
                signals_df['signal'] = signals_df['signal'].apply(format_signal)
                
            if 'z_score' in signals_df.columns:
                signals_df = signals_df.rename(columns={'z_score': 'live_z'})

            target_cols = ['pair', 'live_z', 'signal', 's1_price', 's2_price', 's1_chg', 's2_chg']
            display_cols = [col for col in target_cols if col in signals_df.columns]
            
            if len(display_cols) >= 3:
                st.dataframe(signals_df[display_cols], width='stretch')
            else:
                st.dataframe(signals_df, width='stretch')
                
        except Exception as e:
            st.warning(f"Signals data not found. Error: {e}")

# ─── TAB 2: HISTORICAL BACKTESTS ──────────────────────────────────────────────
with tab2:
    st.subheader("Historical Backtests")
    try:
        backtest_df = pd.read_csv("pairs_trading_data/backtest_summary.csv")
        st.dataframe(backtest_df, width='stretch')
    except Exception as e:
        st.warning("Backtest data not found.")

# ─── TAB 3: PAIR EXPLORER ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Pair Explorer")
    try:
        signals_df = pd.read_csv("pairs_trading_data/signals.csv")
        pair_list = signals_df['pair'].unique().tolist()
        
        selected_pair = st.selectbox("Select Pair", pair_list)
        st.write(f"Displaying live data for: {selected_pair}")
        
        ticker1, ticker2 = selected_pair.split(" / ")
        t1 = f"{ticker1}.NS"
        t2 = f"{ticker2}.NS"
        
        df1 = yf.Ticker(t1).history(period="1y")['Close']
        df2 = yf.Ticker(t2).history(period="1y")['Close']
        
        combined_df = pd.concat([df1, df2], axis=1, keys=[t1, t2]).dropna()
        
        if combined_df.empty:
            st.error("Could not align historical dates for these two tickers.")
        else:
            spread = combined_df[t1] - combined_df[t2]
            z_score = (spread - spread.mean()) / spread.std()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=z_score.index, 
                y=z_score, 
                name="Z-Score", 
                line=dict(color='royalblue')
            ))
            
            fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Short Spread")
            fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Long Spread")
            fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig, width='stretch')
            
    except Exception as e:
        st.error(f"Could not load Pair Explorer data. Error: {e}")
