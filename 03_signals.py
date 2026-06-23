"""
Pairs Trading — Phase 3: Signal Generation (Real-Time Edition)
===============================================================
Generates historical signals on the trading window AND checks
live z-scores using today's real-time prices from yfinance.

The live signal check tells you RIGHT NOW whether any pair is
at an entry, exit, or stop-loss level based on current market prices.

Requirements:
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = "pairs_trading_data"
OUTPUT_DIR = "pairs_trading_data"

ENTRY_ZSCORE    = 2.0
EXIT_ZSCORE     = 0.0
STOPLOSS_ZSCORE = 3.5
ROLLING_WINDOW  = 30


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data():
    pairs_df  = pd.read_csv(os.path.join(INPUT_DIR, "cointegrated_pairs.csv"))
    formation = pd.read_csv(os.path.join(INPUT_DIR, "formation_prices.csv"),
                            index_col=0, parse_dates=True)
    trading   = pd.read_csv(os.path.join(INPUT_DIR, "trading_prices.csv"),
                            index_col=0, parse_dates=True)
    print(f"Loaded {len(pairs_df)} pairs | Trading: {trading.index[0].date()} → {trading.index[-1].date()}")
    return pairs_df, formation, trading


def load_realtime_snapshot():
    path = os.path.join(INPUT_DIR, "realtime_snapshot.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        snap = json.load(f)
    age_mins = (datetime.now() - datetime.fromisoformat(snap["fetched_at"])).seconds // 60
    print(f"Real-time snapshot age: {age_mins} min (fetched at {snap['fetched_at'][:19]})")
    return snap


# ── 2. Spread and z-score ─────────────────────────────────────────────────────

def compute_spread(s1, s2, hedge_ratio, alpha):
    return s1 - hedge_ratio * s2 - alpha


def compute_zscore(spread, window=ROLLING_WINDOW):
    mean = spread.rolling(window=window).mean()
    std  = spread.rolling(window=window).std()
    return (spread - mean) / std


# ── 3. Signal state machine ───────────────────────────────────────────────────

def generate_signals(z):
    signals  = pd.Series(0, index=z.index, dtype=float)
    position = 0
    for i in range(1, len(z)):
        curr_z = z.iloc[i]
        if np.isnan(curr_z):
            signals.iloc[i] = 0
            continue
        if abs(curr_z) >= STOPLOSS_ZSCORE:
            position = 0
        elif position ==  1 and curr_z >= EXIT_ZSCORE:
            position = 0
        elif position == -1 and curr_z <= EXIT_ZSCORE:
            position = 0
        elif position == 0 and curr_z <= -ENTRY_ZSCORE:
            position = 1
        elif position == 0 and curr_z >=  ENTRY_ZSCORE:
            position = -1
        signals.iloc[i] = position
    return signals


# ── 4. Live z-score from real-time prices ────────────────────────────────────

def compute_live_zscore(pair_row, trading, rt_snapshot):
    """
    Compute the current live z-score for a pair using:
    - Historical spread from the trading window (to get rolling stats)
    - Today's real-time price appended as the latest data point
    """
    t1, t2       = pair_row["stock_1"], pair_row["stock_2"]
    hedge_ratio  = pair_row["hedge_ratio"]
    alpha        = pair_row["alpha"]
    half_life    = pair_row["half_life"]
    window       = max(10, min(int(half_life), 60))

    if t1 not in trading.columns or t2 not in trading.columns:
        return None

    s1 = trading[t1].dropna()
    s2 = trading[t2].dropna()
    common = s1.index.intersection(s2.index)
    s1, s2 = s1[common], s2[common]

    # Append real-time price if available
    if rt_snapshot:
        rt = rt_snapshot.get("prices", {})
        p1 = rt.get(t1, {})
        p2 = rt.get(t2, {})
        if p1 and p2:
            today = pd.Timestamp(rt_snapshot["fetched_at"][:10])
            if today not in s1.index:
                s1 = pd.concat([s1, pd.Series([p1["price"]], index=[today])])
                s2 = pd.concat([s2, pd.Series([p2["price"]], index=[today])])

    spread  = compute_spread(s1, s2, hedge_ratio, alpha)
    z       = compute_zscore(spread, window=window)
    live_z  = z.iloc[-1]

    # Determine signal
    if abs(live_z) >= STOPLOSS_ZSCORE:
        signal_label = "🔴 STOP-LOSS ZONE"
    elif abs(live_z) >= ENTRY_ZSCORE:
        signal_label = "🟢 ENTRY SIGNAL" if live_z < 0 else "🔴 SHORT SIGNAL"
    elif abs(live_z) <= EXIT_ZSCORE + 0.2:
        signal_label = "🟡 EXIT / FLAT"
    else:
        signal_label = "⚪ NO SIGNAL"

    return {
        "pair"        : f"{t1.replace('.NS','')} / {t2.replace('.NS','')}",
        "live_z"      : round(live_z, 3),
        "signal"      : signal_label,
        "s1_price"    : p1.get("price") if rt_snapshot and p1 else None,
        "s2_price"    : p2.get("price") if rt_snapshot and p2 else None,
        "s1_chg"      : p1.get("change_pct") if rt_snapshot and p1 else None,
        "s2_chg"      : p2.get("change_pct") if rt_snapshot and p2 else None,
        "half_life"   : half_life,
        "window"      : window,
        "as_of"       : rt_snapshot["fetched_at"][:19] if rt_snapshot else "historical only",
    }


# ── 5. Process all pairs ──────────────────────────────────────────────────────

def process_all_pairs(pairs_df, trading, formation, rt_snapshot, top_n=10):
    print(f"\nGenerating signals for top {top_n} pairs...")
    all_signals  = {}
    live_signals = []

    for _, row in pairs_df.head(top_n).iterrows():
        t1, t2       = row["stock_1"], row["stock_2"]
        hedge_ratio  = row["hedge_ratio"]
        alpha        = row["alpha"]
        half_life    = row["half_life"]
        pair_label   = f"{t1.replace('.NS','')} / {t2.replace('.NS','')}"
        window       = max(10, min(int(half_life), 60))

        if t1 not in trading.columns or t2 not in trading.columns:
            continue

        s1     = trading[t1].dropna()
        s2     = trading[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 60:
            continue
        s1, s2 = s1[common], s2[common]

        spread  = compute_spread(s1, s2, hedge_ratio, alpha)
        z       = compute_zscore(spread, window=window)
        signals = generate_signals(z)

        n_trades   = (signals.diff().abs() > 0).sum() // 2
        in_market  = (signals != 0).sum() / len(signals) * 100

        all_signals[pair_label] = {
            "t1": t1, "t2": t2, "hedge_ratio": hedge_ratio, "alpha": alpha,
            "spread": spread, "z_score": z, "signals": signals,
            "s1": s1, "s2": s2, "window": window,
        }

        # Live z-score
        live = compute_live_zscore(row, trading, rt_snapshot)
        if live:
            live_signals.append(live)
            live_z_str = f"  live z={live['live_z']:+.3f}  {live['signal']}"
        else:
            live_z_str = ""

        print(f"  {pair_label:<35}  trades={n_trades:>3}  in-market={in_market:>5.1f}%{live_z_str}")

    return all_signals, live_signals


# ── 6. Print live signal dashboard ───────────────────────────────────────────

def print_live_signals(live_signals):
    if not live_signals:
        return
    print(f"\n{'═'*70}")
    print("  LIVE SIGNAL DASHBOARD")
    print(f"{'═'*70}")
    print(f"  {'Pair':<35} {'Z-Score':>8}  {'S1 Price':>10}  {'S2 Price':>10}  Signal")
    print(f"  {'─'*35} {'─'*8}  {'─'*10}  {'─'*10}  {'─'*20}")
    for ls in live_signals:
        s1p = f"₹{ls['s1_price']:>8.2f}" if ls["s1_price"] else "    N/A   "
        s2p = f"₹{ls['s2_price']:>8.2f}" if ls["s2_price"] else "    N/A   "
        print(f"  {ls['pair']:<35} {ls['live_z']:>+8.3f}  {s1p}  {s2p}  {ls['signal']}")
    print(f"{'═'*70}")
    print(f"  As of: {live_signals[0]['as_of']}")
    print(f"{'═'*70}\n")

    # Save live signals JSON for dashboard
    path = os.path.join(OUTPUT_DIR, "live_signals.json")
    with open(path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "signals": live_signals}, f, indent=2)
    print(f"Saved live signals → {path}")


# ── 7. Plot and save ──────────────────────────────────────────────────────────

def plot_pair_signals(pair_label, data):
    s1, s2   = data["s1"], data["s2"]
    spread   = data["spread"]
    z        = data["z_score"]
    signals  = data["signals"]
    window   = data["window"]

    rolling_mean = spread.rolling(window=window).mean()
    rolling_std  = spread.rolling(window=window).std()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Pairs Trading: {pair_label}", fontsize=13, fontweight="bold")

    t1l = data["t1"].replace(".NS", "")
    t2l = data["t2"].replace(".NS", "")

    axes[0].plot(s1 / s1.iloc[0], label=t1l, lw=1.3, color="#2196F3")
    axes[0].plot(s2 / s2.iloc[0], label=t2l, lw=1.3, color="#FF9800", alpha=0.85)
    axes[0].set_ylabel("Normalised Price"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.25)

    axes[1].plot(spread.index, spread, lw=1, color="steelblue", label="Spread")
    axes[1].plot(rolling_mean.index, rolling_mean, lw=1.2, color="black", linestyle="--")
    axes[1].fill_between(spread.index, rolling_mean-rolling_std, rolling_mean+rolling_std,
                         alpha=0.15, color="grey")
    axes[1].set_ylabel("Spread (₹)"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.25)

    axes[2].plot(z.index, z, lw=1, color="purple")
    axes[2].axhline(0, color="black", lw=0.8, linestyle="--")
    for level, col, ls in [(ENTRY_ZSCORE,"red","--"),(-ENTRY_ZSCORE,"red","--"),
                            (STOPLOSS_ZSCORE,"orange",":"),(- STOPLOSS_ZSCORE,"orange",":")]:
        axes[2].axhline(level, color=col, lw=0.9, linestyle=ls)
    axes[2].fill_between(signals.index, -5, 5, where=(signals==1),  alpha=0.12, color="green")
    axes[2].fill_between(signals.index, -5, 5, where=(signals==-1), alpha=0.12, color="red")
    axes[2].set_ylim(-5, 5); axes[2].set_ylabel("Z-score"); axes[2].grid(True, alpha=0.25)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)

    plt.tight_layout()
    safe = pair_label.replace("/","_").replace(" ","")
    path = os.path.join(OUTPUT_DIR, f"signals_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_signals(all_signals):
    rows = []
    for pair_label, data in all_signals.items():
        df = pd.DataFrame({
            "pair": pair_label, "t1": data["t1"], "t2": data["t2"],
            "hedge_ratio": data["hedge_ratio"], "alpha": data["alpha"],
            "signal": data["signals"], "z_score": data["z_score"],
            "spread": data["spread"], "s1_price": data["s1"], "s2_price": data["s2"],
        })
        rows.append(df)
    combined = pd.concat(rows).reset_index().rename(columns={"index": "Date"})
    path = os.path.join(OUTPUT_DIR, "signals.csv")
    combined.to_csv(path, index=False)
    print(f"Saved signals → {path}  ({len(combined):,} rows)")


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    pairs_df, formation, trading = load_data()
    rt_snapshot = load_realtime_snapshot()

    all_signals, live_signals = process_all_pairs(
        pairs_df, trading, formation, rt_snapshot, top_n=10)

    print_live_signals(live_signals)

    print("Plotting signal charts...")
    for pair_label, data in all_signals.items():
        plot_pair_signals(pair_label, data)
        print(f"  Saved → signals_{pair_label.replace('/','_').replace(' ','')}.png")

    save_signals(all_signals)
    print("\nPhase 3 complete.\n")
    return all_signals, live_signals


if __name__ == "__main__":
    all_signals, live_signals = main()