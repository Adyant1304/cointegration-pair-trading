"""
Pairs Trading — Phase 1: Data Collection (Real-Time Edition)
=============================================================
Downloads historical + latest real-time prices for Nifty 100 stocks
via yfinance. Saves clean price data for cointegration testing.

get_realtime_prices() is imported by the dashboard server (server.py)
to serve live quotes to the dashboard on demand.

Requirements:
    pip install yfinance pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

OUTPUT_DIR = "pairs_trading_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

NIFTY100_TICKERS = [
    "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS",
    "SBIN.NS","INDUSINDBK.NS","BANDHANBNK.NS","FEDERALBNK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS","CHOLAFIN.NS",
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS",
    "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS",
    "POWERGRID.NS","NTPC.NS","ADANIGREEN.NS","ADANIPORTS.NS",
    "MARUTI.NS","TATAMOTORS.NS","M&M.NS","BAJAJ-AUTO.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","TVSMOTOR.NS",
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS",
    "DABUR.NS","MARICO.NS","GODREJCP.NS","COLPAL.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS","MANKIND.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS","VEDL.NS","NMDC.NS",
    "ULTRACEMCO.NS","GRASIM.NS","SHREECEM.NS","ACC.NS","AMBUJACEM.NS","DLF.NS",
    "BHARTIARTL.NS","IDEA.NS",
    "LT.NS","ADANIENT.NS","SIEMENS.NS","ABB.NS","HAVELLS.NS",
    "TITAN.NS","TATACONSUM.NS","ZOMATO.NS",
]


# ── 1. Historical download ────────────────────────────────────────────────────

def download_prices(tickers, start, end):
    print(f"\nDownloading {len(tickers)} tickers: {start} → {end}")
    raw = yf.download(tickers=tickers, start=start, end=end,
                      auto_adjust=True, progress=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    print(f"Raw shape: {prices.shape}")
    return prices


def clean_prices(prices, min_coverage=0.80):
    coverage     = prices.notna().mean()
    good         = coverage[coverage >= min_coverage].index.tolist()
    dropped      = set(prices.columns) - set(good)
    if dropped:
        print(f"Dropped {len(dropped)} low-coverage tickers: {', '.join(sorted(dropped))}")
    prices = prices[good].ffill(limit=5).dropna(how="all").dropna(axis=1)
    print(f"Clean: {prices.shape}  |  {prices.index[0].date()} → {prices.index[-1].date()}")
    return prices


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


# ── 2. Real-time prices ───────────────────────────────────────────────────────

def get_realtime_prices(tickers=None):
    """
    Fetch latest real-time price for each ticker using yfinance fast_info.
    Returns dict: { "AXISBANK.NS": { price, prev_close, change, change_pct, volume, timestamp } }

    Called by:
      - server.py  → /api/realtime  (dashboard live price feed)
      - phase3     → to append today's price to spread calculation
      - phase4     → to mark current open P&L on active positions
    """
    if tickers is None:
        tickers = NIFTY100_TICKERS

    print(f"Fetching real-time prices for {len(tickers)} tickers...")
    results = {}

    for ticker in tickers:
        try:
            info       = yf.Ticker(ticker).fast_info
            price      = round(float(info.last_price), 2)
            prev_close = round(float(info.previous_close), 2)
            change     = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

            results[ticker] = {
                "price"      : price,
                "prev_close" : prev_close,
                "change"     : change,
                "change_pct" : change_pct,
                "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            print(f"  Warning: {ticker}: {e}")
            results[ticker] = None

    ok = sum(1 for v in results.values() if v is not None)
    print(f"Fetched {ok}/{len(tickers)} successfully.")
    return results


def save_realtime_snapshot(results):
    path = os.path.join(OUTPUT_DIR, "realtime_snapshot.json")
    with open(path, "w") as f:
        json.dump({"fetched_at": datetime.now().isoformat(), "prices": results}, f, indent=2)
    print(f"Saved real-time snapshot → {path}")
    return path


# ── 3. Main ───────────────────────────────────────────────────────────────────

def main():
    raw     = download_prices(NIFTY100_TICKERS, START_DATE, END_DATE)
    prices  = clean_prices(raw)
    returns = compute_log_returns(prices)

    prices.to_csv(os.path.join(OUTPUT_DIR, "prices.csv"))
    returns.to_csv(os.path.join(OUTPUT_DIR, "log_returns.csv"))
    print(f"\nSaved → {OUTPUT_DIR}/prices.csv")
    print(f"Saved → {OUTPUT_DIR}/log_returns.csv")

    print(f"\nPairs to test: {prices.shape[1]*(prices.shape[1]-1)//2:,}")

    print("\nFetching real-time snapshot...")
    rt = get_realtime_prices(prices.columns.tolist())
    save_realtime_snapshot(rt)

    print("\nPhase 1 complete.\n")
    return prices, returns


if __name__ == "__main__":
    prices, returns = main()