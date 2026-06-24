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
                
                # --- ROBUST LIVE PNL ENGINE ---
                if not expanded_df.empty and 'pair' in expanded_df.columns:
                    # 1. Gather unique tickers
                    unique_tickers = set()
                    for p in expanded_df['pair']:
                        t1, t2 = p.split(' / ')
                        unique_tickers.add(f"{t1}.NS")
                        unique_tickers.add(f"{t2}.NS")
                    
                    # 2. Fetch prices individually (safer than bulk download)
                    live_prices = {}
                    for ticker in unique_tickers:
                        try:
                            # Grabs the absolute latest closing price
                            live_prices[ticker] = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                        except Exception:
                            pass
                    
                    # 3. Calculate Math
                    for idx, row in expanded_df.iterrows():
                        t1, t2 = row['pair'].split(' / ')
                        tk1, tk2 = f"{t1}.NS", f"{t2}.NS"
                        
                        if tk1 in live_prices and tk2 in live_prices:
                            current_s1 = live_prices[tk1]
                            current_s2 = live_prices[tk2]
                            
                            expanded_df.at[idx, 'live_s1'] = round(current_s1, 2)
                            expanded_df.at[idx, 'live_s2'] = round(current_s2, 2)
                            
                            # Calculate % PnL
                            if "SHORT" in str(row['position']).upper():
                                pnl = ((row['entry_s1'] - current_s1) / row['entry_s1']) + ((current_s2 - row['entry_s2']) / row['entry_s2'])
                            else:
                                pnl = ((current_s1 - row['entry_s1']) / row['entry_s1']) + ((row['entry_s2'] - current_s2) / row['entry_s2'])
                            
                            expanded_df.at[idx, 'open_pnl'] = round(pnl * 100, 2)
                        else:
                            expanded_df.at[idx, 'open_pnl'] = 0.00

                target_pos_cols = ['pair', 'position', 'entry_s1', 'entry_s2', 'live_s1', 'live_s2', 'open_pnl']
                display_pos_cols = [c for c in target_pos_cols if c in expanded_df.columns]
                
                # Apply Color
                def color_pnl(val):
                    try:
                        val = float(val)
                        if val > 0: return 'color: #00ff00'
                        elif val < 0: return 'color: #ff4b4b'
                        return 'color: gray'
                    except:
                        return ''

                if display_pos_cols:
                    styled_df = expanded_df[display_pos_cols].style\
                        .map(color_pnl, subset=['open_pnl'])\
                        .format({'open_pnl': '{:+.2f}%', 'entry_s1': '{:.2f}', 'entry_s2': '{:.2f}', 'live_s1': '{:.2f}', 'live_s2': '{:.2f}'})
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
            
            # --- FIXED EMOJI MAPPING ---
            def format_signal(val):
                val_str = str(val).upper().strip()
                if "SHORT" in val_str: return "🔴 " + val_str
                elif "LONG" in val_str: return "🟢 " + val_str
                elif "EXIT" in val_str or "FLAT" in val_str: return "🟡 " + val_str
                # Added "0.0" and "0" to catch those empty signal rows
                elif val_str in ["0", "0.0", "NONE", "NAN", "NO SIGNAL"]: return "⚪ NO SIGNAL"
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
