"""
Pairs Trading — Phase 4: Backtesting & Live P&L (Real-Time Edition)
=====================================================================
Backtests all signals from Phase 3 with realistic transaction costs.
Also computes OPEN P&L on any currently active positions using
today's real-time prices from yfinance.

Requirements:
    pip install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR    = "pairs_trading_data"
OUTPUT_DIR   = "pairs_trading_data"
CAPITAL      = 100_000
COST_PER_LEG = 0.0005
ANNUAL_RF    = 0.065


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_signals():
    path = os.path.join(INPUT_DIR, "signals.csv")
    df   = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date"})
    print(f"Loaded signals: {len(df):,} rows, {df['pair'].nunique()} pairs")
    print(f"Date range    : {df['date'].min().date()} → {df['date'].max().date()}\n")
    return df


def load_realtime_snapshot():
    path = os.path.join(INPUT_DIR, "realtime_snapshot.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_live_signals():
    path = os.path.join(INPUT_DIR, "live_signals.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("signals", [])


# ── 2. Backtest single pair ───────────────────────────────────────────────────

def backtest_pair(pair_data, capital=CAPITAL):
    df          = pair_data.sort_values("date").copy().reset_index(drop=True)
    hedge_ratio = df["hedge_ratio"].iloc[0]
    signals     = df["signal"].values
    s1          = df["s1_price"].values
    s2          = df["s2_price"].values
    n           = len(df)
    daily_pnl   = np.zeros(n)
    position    = np.zeros(n)
    trade_cost  = np.zeros(n)

    for i in range(1, n):
        prev_sig  = signals[i-1]
        curr_sig  = signals[i]
        s1_shares = capital / s1[i]
        s2_shares = (capital * hedge_ratio) / s2[i]

        if prev_sig == 1:
            pnl = s1_shares*(s1[i]-s1[i-1]) - s2_shares*(s2[i]-s2[i-1])
        elif prev_sig == -1:
            pnl = -s1_shares*(s1[i]-s1[i-1]) + s2_shares*(s2[i]-s2[i-1])
        else:
            pnl = 0.0

        daily_pnl[i] = pnl
        position[i]  = prev_sig

        if curr_sig != prev_sig:
            cost = capital * COST_PER_LEG * 2
            if curr_sig != 0 and prev_sig != 0:
                cost *= 2
            trade_cost[i] = cost

    net_pnl = daily_pnl - trade_cost
    result  = df[["date","pair","signal","z_score","spread"]].copy()
    result["gross_pnl"]  = daily_pnl
    result["trade_cost"] = trade_cost
    result["net_pnl"]    = net_pnl
    result["position"]   = position
    result["equity"]     = capital + net_pnl.cumsum()
    return result


# ── 3. Real-time open P&L ─────────────────────────────────────────────────────

def compute_open_pnl(signals_df, rt_snapshot, capital=CAPITAL):
    """
    For any pair currently in an active position (last signal != 0),
    compute the unrealised P&L using today's real-time prices.
    """
    if not rt_snapshot:
        return []

    rt      = rt_snapshot.get("prices", {})
    open_positions = []

    for pair_label in signals_df["pair"].unique():
        pair_data = signals_df[signals_df["pair"] == pair_label].sort_values("date")
        last_row  = pair_data.iloc[-1]

        if last_row["signal"] == 0:
            continue  # no open position

        t1          = last_row["t1"]
        t2          = last_row["t2"]
        hedge_ratio = last_row["hedge_ratio"]
        position    = int(last_row["signal"])  # +1 or -1

        entry_s1 = last_row["s1_price"]
        entry_s2 = last_row["s2_price"]

        rt1 = rt.get(t1)
        rt2 = rt.get(t2)
        if not rt1 or not rt2:
            continue

        live_s1 = rt1["price"]
        live_s2 = rt2["price"]

        s1_shares = capital / entry_s1
        s2_shares = (capital * hedge_ratio) / entry_s2

        if position == 1:   # long spread: long s1, short s2
            pnl = s1_shares*(live_s1-entry_s1) - s2_shares*(live_s2-entry_s2)
        else:               # short spread: short s1, long s2
            pnl = -s1_shares*(live_s1-entry_s1) + s2_shares*(live_s2-entry_s2)

        open_positions.append({
            "pair"       : pair_label,
            "position"   : "LONG spread" if position == 1 else "SHORT spread",
            "entry_s1"   : round(entry_s1, 2),
            "entry_s2"   : round(entry_s2, 2),
            "live_s1"    : round(live_s1, 2),
            "live_s2"    : round(live_s2, 2),
            "open_pnl"   : round(pnl, 2),
            "open_pnl_pct": round(pnl / capital * 100, 3),
            "as_of"      : rt_snapshot["fetched_at"][:19],
        })

    return open_positions


def print_open_positions(open_positions):
    if not open_positions:
        print("\nNo open positions currently.")
        return

    print(f"\n{'═'*75}")
    print("  OPEN POSITIONS (Real-Time P&L)")
    print(f"{'═'*75}")
    print(f"  {'Pair':<35} {'Side':<14} {'Open P&L':>10}  {'Ret%':>7}")
    print(f"  {'─'*35} {'─'*14} {'─'*10}  {'─'*7}")
    for p in open_positions:
        sign  = "+" if p["open_pnl"] >= 0 else ""
        color = "▲" if p["open_pnl"] >= 0 else "▼"
        print(f"  {p['pair']:<35} {p['position']:<14} "
              f"{sign}₹{p['open_pnl']:>8,.0f}  {sign}{p['open_pnl_pct']:>6.3f}%  {color}")
    print(f"{'═'*75}")
    total = sum(p["open_pnl"] for p in open_positions)
    sign  = "+" if total >= 0 else ""
    print(f"  Total open P&L: {sign}₹{total:,.0f}")
    print(f"  As of: {open_positions[0]['as_of']}")
    print(f"{'═'*75}\n")

    path = os.path.join(OUTPUT_DIR, "open_positions.json")
    with open(path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "positions": open_positions}, f, indent=2)
    print(f"Saved open positions → {path}")


# ── 4. Performance metrics ────────────────────────────────────────────────────

def compute_metrics(result, capital=CAPITAL):
    daily_ret = result["net_pnl"] / capital
    equity    = result["equity"]
    total_ret = (equity.iloc[-1] - capital) / capital
    n_days    = len(result)
    ann_ret   = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol   = daily_ret.std() * np.sqrt(252)
    daily_rf  = ANNUAL_RF / 252
    excess    = daily_ret - daily_rf
    sharpe    = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
    downside  = excess[excess < 0].std() * np.sqrt(252)
    sortino   = (ann_ret - ANNUAL_RF) / downside if downside > 0 else 0
    roll_max  = equity.cummax()
    drawdown  = (equity - roll_max) / roll_max
    max_dd    = drawdown.min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0

    if "position" in result.columns:
        in_pos   = result[result["position"] != 0]
        win_rate = (in_pos["net_pnl"] > 0).mean() if len(in_pos) > 0 else 0
    else:
        win_rate = (result[result["net_pnl"] != 0]["net_pnl"] > 0).mean()

    trades      = (result["signal"].diff().abs() > 0).sum() // 2 if "signal" in result.columns else 0
    gross_wins  = result[result["net_pnl"] > 0]["net_pnl"].sum()
    gross_loss  = result[result["net_pnl"] < 0]["net_pnl"].abs().sum()
    profit_fac  = gross_wins / gross_loss if gross_loss > 0 else np.inf

    return {
        "Total Return (%)"   : round(total_ret * 100, 2),
        "Ann. Return (%)"    : round(ann_ret * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio"       : round(sharpe, 3),
        "Sortino Ratio"      : round(sortino, 3),
        "Calmar Ratio"       : round(calmar, 3),
        "Max Drawdown (%)"   : round(max_dd * 100, 2),
        "Win Rate (%)"       : round(win_rate * 100, 2),
        "Profit Factor"      : round(profit_fac, 3),
        "Total Trades"       : int(trades),
        "Final Equity (₹)"   : round(equity.iloc[-1], 2),
    }


# ── 5. Tearsheet ──────────────────────────────────────────────────────────────

def plot_tearsheet(result, metrics, pair_label):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f"Backtest Tearsheet — {pair_label}", fontsize=13, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    dates  = result["date"]
    equity = result["equity"]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, lw=1.5, color="#2196F3", label="Strategy equity")
    ax1.axhline(CAPITAL, color="grey", lw=0.8, linestyle="--", label="Starting capital")
    ax1.fill_between(dates, CAPITAL, equity, where=(equity>=CAPITAL), alpha=0.15, color="green")
    ax1.fill_between(dates, CAPITAL, equity, where=(equity< CAPITAL), alpha=0.15, color="red")
    ax1.set_ylabel("Portfolio Value (₹)"); ax1.set_title("Equity Curve", fontsize=10, loc="left")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20)

    ax2 = fig.add_subplot(gs[1, 0])
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100
    ax2.fill_between(dates, drawdown, 0, alpha=0.5, color="red")
    ax2.plot(dates, drawdown, lw=0.8, color="darkred")
    ax2.set_ylabel("Drawdown (%)"); ax2.set_title("Drawdown", fontsize=10, loc="left")
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20)

    ax3 = fig.add_subplot(gs[1, 1])
    daily_ret = result["net_pnl"]
    ax3.hist(daily_ret[daily_ret!=0], bins=40, color="#7C4DFF", alpha=0.7, edgecolor="white")
    ax3.axvline(0, color="black", lw=1)
    ax3.axvline(daily_ret.mean(), color="orange", lw=1.2, linestyle="--",
                label=f"Mean: ₹{daily_ret.mean():.0f}")
    ax3.set_xlabel("Daily P&L (₹)"); ax3.set_ylabel("Frequency")
    ax3.set_title("Daily P&L Distribution", fontsize=10, loc="left")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[2, 0])
    roll_ret    = result["net_pnl"] / CAPITAL
    roll_sharpe = roll_ret.rolling(63).apply(
        lambda x: (x.mean()-ANNUAL_RF/252)/x.std()*np.sqrt(252) if x.std()>0 else 0)
    ax4.plot(dates, roll_sharpe, lw=1.2, color="teal")
    ax4.axhline(0, color="black", lw=0.8, linestyle="--")
    ax4.axhline(1, color="green", lw=0.8, linestyle=":", label="Sharpe = 1")
    ax4.set_ylabel("Rolling Sharpe"); ax4.set_title("63-Day Rolling Sharpe", fontsize=10, loc="left")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    table = ax5.table(
        cellText  = [[k, str(v)] for k, v in metrics.items()],
        colLabels = ["Metric", "Value"],
        cellLoc="left", loc="center", colWidths=[0.65, 0.35],
    )
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0: cell.set_facecolor("#E3F2FD"); cell.set_text_props(fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#F9F9F9")
    ax5.set_title("Performance Metrics", fontsize=10, loc="left", pad=12)

    safe = pair_label.replace("/","_").replace(" ","")
    path = os.path.join(OUTPUT_DIR, f"tearsheet_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved tearsheet → {path}")


# ── 6. Portfolio ──────────────────────────────────────────────────────────────

def compute_portfolio(all_results):
    pnl_list = []
    for r in all_results:
        s = r.set_index("date")["net_pnl"].rename(r["pair"].iloc[0])
        pnl_list.append(s)
    combined  = pd.concat(pnl_list, axis=1).fillna(0)
    portfolio = combined.sum(axis=1).reset_index()
    portfolio.columns = ["date", "net_pnl"]
    total_cap = CAPITAL * len(all_results)
    portfolio["equity"] = total_cap + portfolio["net_pnl"].cumsum()
    portfolio["pair"]   = "PORTFOLIO"
    return portfolio


def plot_portfolio(portfolio):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Portfolio Equity — All Pairs Combined", fontsize=13, fontweight="bold")
    total_cap = portfolio["equity"].iloc[0] - portfolio["net_pnl"].cumsum().iloc[0]
    eq  = portfolio["equity"]

    axes[0].plot(portfolio["date"], eq, lw=1.5, color="#1565C0")
    axes[0].fill_between(portfolio["date"], total_cap, eq,
                         where=(eq>=total_cap), alpha=0.15, color="green")
    axes[0].fill_between(portfolio["date"], total_cap, eq,
                         where=(eq< total_cap), alpha=0.15, color="red")
    axes[0].axhline(total_cap, color="grey", lw=0.8, linestyle="--")
    axes[0].set_ylabel("Portfolio Value (₹)"); axes[0].grid(True, alpha=0.25)

    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    axes[1].fill_between(portfolio["date"], dd, 0, alpha=0.5, color="red")
    axes[1].set_ylabel("Drawdown (%)"); axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "portfolio_equity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved portfolio chart → {path}")


# ── 7. Summary ────────────────────────────────────────────────────────────────

def print_and_save_summary(summary):
    df = pd.DataFrame(summary).sort_values("Sharpe Ratio", ascending=False)
    print(f"\n{'═'*90}")
    print("  BACKTEST SUMMARY")
    print(f"{'═'*90}")
    cols = ["pair","Total Return (%)","Ann. Return (%)","Sharpe Ratio","Max Drawdown (%)","Win Rate (%)","Total Trades"]
    print(df[cols].to_string(index=False))
    print(f"{'═'*90}")
    path = os.path.join(OUTPUT_DIR, "backtest_summary.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved summary → {path}")

    # Save JSON for dashboard
    records = df.to_dict(orient="records")
    with open(os.path.join(OUTPUT_DIR, "backtest_summary.json"), "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "results": records}, f, indent=2)


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    signals_df     = load_signals()
    rt_snapshot    = load_realtime_snapshot()
    live_signals   = load_live_signals()

    all_results = []
    summary     = []
    pairs       = signals_df["pair"].unique()

    print(f"Backtesting {len(pairs)} pairs...\n")
    for pair_label in pairs:
        pair_data = signals_df[signals_df["pair"] == pair_label].copy()
        if len(pair_data) < 60:
            continue
        result  = backtest_pair(pair_data)
        metrics = compute_metrics(result)
        print(f"  {pair_label:<38}  "
              f"Return={metrics['Ann. Return (%)']:>6.1f}%  "
              f"Sharpe={metrics['Sharpe Ratio']:>6.3f}  "
              f"MaxDD={metrics['Max Drawdown (%)']:>6.1f}%  "
              f"Trades={metrics['Total Trades']:>3}")
        plot_tearsheet(result, metrics, pair_label)
        all_results.append(result)
        summary.append({"pair": pair_label, **metrics})

    # Portfolio
    if all_results:
        print("\nComputing portfolio-level performance...")
        portfolio    = compute_portfolio(all_results)
        port_metrics = compute_metrics(portfolio, capital=CAPITAL * len(all_results))
        plot_portfolio(portfolio)
        print("\nPORTFOLIO METRICS:")
        for k, v in port_metrics.items():
            print(f"  {k:<25}: {v}")

    print_and_save_summary(summary)

    # Real-time open P&L
    print("\nChecking open positions against real-time prices...")
    open_positions = compute_open_pnl(signals_df, rt_snapshot)
    print_open_positions(open_positions)

    print("\nPhase 4 complete.\n")


if __name__ == "__main__":
    main()